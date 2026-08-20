#!/usr/bin/env python3
"""Run a bounded checkpoint-backed DeepSeek-V4 Flash MTP body replay.

The default input is a deterministic structural control, not natural corpus
calibration.  It exercises the real three-block checkpoint graph, the exact
target-tap projection, the joint anchor-plus-four-noise proposal, and natural
router-selected expert materialization without decoding every expert.  A
separate target-prefix capture is still required before this can qualify the
activation distribution used for production quantization.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
from typing import Any


SCHEMA = "ds4rt-gptqmodel-flash-mtp-replay-v1"
INPUT_KIND = "deterministic-structural-control-not-natural-calibration"


class ValidationError(RuntimeError):
    """The checkpoint-backed MTP replay violated its bounded contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    import torch

    raw = tensor.detach().cpu().contiguous()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as output:
        temporary = Path(output.name)
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def _build_harness(snapshot: Path, config: Any, model_definition: Any) -> Any:
    harness = object.__new__(model_definition)
    harness.model = SimpleNamespace(config=config)
    harness.model_local_path = os.fspath(snapshot)
    harness.quantize_config = SimpleNamespace(preprocessors=[])
    return harness


def _load_checkpoint_tensor(snapshot: Path, tensor_name: str, device: Any) -> Any:
    from safetensors import safe_open

    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    try:
        shard = snapshot / index[tensor_name]
    except KeyError as exc:
        raise ValidationError(f"checkpoint is missing {tensor_name}") from exc
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name).to(device=device)


def _materialize_envelope(harness: Any, auxiliary: Any, device: Any) -> list[str]:
    materialized: list[str] = []

    def leaf(name: str, module: Any) -> None:
        harness.materialize_mtp_replay_submodule(
            auxiliary, module, device=device
        )
        materialized.append(name)

    leaf("mtp.0.main_proj", auxiliary.block(0).main_proj)
    leaf("mtp.0.main_norm", auxiliary.block(0).main_norm)
    for block_index in range(3):
        block = auxiliary.block(block_index)
        prefix = f"mtp.{block_index}"
        leaf(f"{prefix}.input_layernorm", block.input_layernorm)
        leaf(f"{prefix}.attn_hc", block.attn_hc)
        leaf(f"{prefix}.self_attn.q_a_proj", block.self_attn.q_a_proj)
        leaf(f"{prefix}.self_attn.q_a_norm", block.self_attn.q_a_norm)
        leaf(f"{prefix}.self_attn.q_b_proj", block.self_attn.q_b_proj)
        leaf(f"{prefix}.self_attn.kv_proj", block.self_attn.kv_proj)
        leaf(f"{prefix}.self_attn.kv_norm", block.self_attn.kv_norm)
        leaf(f"{prefix}.self_attn.o_a_proj", block.self_attn.o_a_proj)
        leaf(f"{prefix}.self_attn.o_b_proj", block.self_attn.o_b_proj)
        auxiliary.materialize_nonquant_submodule(
            block.self_attn, device=device, recurse=False
        )
        materialized.append(f"{prefix}.self_attn.sinks")
        leaf(f"{prefix}.post_attention_layernorm", block.post_attention_layernorm)
        leaf(f"{prefix}.ffn_hc", block.ffn_hc)
        leaf(f"{prefix}.mlp.gate", block.mlp.gate)
        leaf(
            f"{prefix}.mlp.shared_experts.gate_proj",
            block.mlp.shared_experts.gate_proj,
        )
        leaf(
            f"{prefix}.mlp.shared_experts.up_proj",
            block.mlp.shared_experts.up_proj,
        )
        leaf(
            f"{prefix}.mlp.shared_experts.down_proj",
            block.mlp.shared_experts.down_proj,
        )
    return materialized


def _device_report(device: Any) -> dict[str, Any]:
    import torch

    result = {"requested": str(device), "type": device.type}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "index": device.index,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
            }
        )
    return result


def run(
    snapshot: Path,
    *,
    device_name: str,
    batch_size: int,
    main_rows: int,
    gptqmodel_lock: Path | None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    from gptqmodel.models.definitions.deepseek_v4 import (
        DeepSeekV4MTPReplay,
        DeepSeekV4MTPReplayBatch,
        DeepSeekV4QModel,
    )

    snapshot = snapshot.expanduser().resolve(strict=True)
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    if config_json.get("model_type") != "deepseek_v4":
        raise ValidationError("MTP replay requires a DeepSeek V4 checkpoint")
    if config_json.get("expert_dtype") != "fp4":
        raise ValidationError("Flash MTP replay expects the native FP4 expert source")
    if batch_size <= 0:
        raise ValidationError("batch_size must be positive")
    if not 1 <= main_rows <= int(config_json["sliding_window"]):
        raise ValidationError("main_rows must fit the checkpoint sliding window")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValidationError("CUDA replay requested but CUDA is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    lock = None
    if gptqmodel_lock is not None:
        lock = json.loads(
            gptqmodel_lock.expanduser().resolve(strict=True).read_text(
                encoding="utf-8"
            )
        )
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    harness = _build_harness(snapshot, config, DeepSeekV4QModel)
    auxiliary = harness.build_mtp_auxiliary()
    envelope = _materialize_envelope(harness, auxiliary, device)
    embedding = _load_checkpoint_tensor(snapshot, "embed.weight", device)
    if embedding.dtype is not torch.bfloat16:
        raise ValidationError(f"target embedding dtype is {embedding.dtype}, expected BF16")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0xD54A_0731)
    target_taps = tuple(
        torch.randn(
            batch_size,
            main_rows,
            int(config.hidden_size),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype=torch.bfloat16, device=device)
        for _ in range(3)
    )
    anchor_token_ids = torch.arange(
        17, 17 + batch_size, dtype=torch.long, device=device
    )
    main_position_ids = torch.arange(
        main_rows, dtype=torch.long, device=device
    ).expand(batch_size, -1)
    main_attention_mask = torch.ones(
        batch_size, main_rows, dtype=torch.bool, device=device
    )
    batch = DeepSeekV4MTPReplayBatch(
        target_taps=target_taps,
        anchor_token_ids=anchor_token_ids,
        main_position_ids=main_position_ids,
        main_attention_mask=main_attention_mask,
    )

    selected: dict[int, set[int]] = {index: set() for index in range(3)}
    expert_modules: dict[tuple[int, int], list[str]] = {}

    def prepare_ffn(block_index: int, block: Any, hidden: Any, token_ids: Any) -> None:
        del token_ids
        with torch.inference_mode():
            _, _, indices = block.mlp.gate(hidden)
        for expert_index in sorted(set(int(value) for value in indices.flatten().tolist())):
            key = (block_index, expert_index)
            if key in expert_modules:
                continue
            expert = block.mlp.experts[expert_index]
            names = []
            for projection_name in ("gate_proj", "up_proj", "down_proj"):
                harness.materialize_mtp_replay_submodule(
                    auxiliary,
                    getattr(expert, projection_name),
                    device=device,
                )
                names.append(
                    f"mtp.{block_index}.mlp.experts.{expert_index}.{projection_name}"
                )
            expert_modules[key] = names
            selected[block_index].add(expert_index)

    replay = DeepSeekV4MTPReplay(auxiliary, embedding_weight=embedding)
    with torch.inference_mode():
        result = replay.replay(batch, prepare_ffn=prepare_ffn)
        repeated = replay.replay(batch, prepare_ffn=prepare_ffn)
    if not torch.equal(result.projected_main, repeated.projected_main):
        raise ValidationError("projected target-main replay is not bit-repeatable")
    if not torch.equal(result.terminal_residual, repeated.terminal_residual):
        raise ValidationError("MTP terminal residual replay is not bit-repeatable")
    for actual, again in zip(result.routes, repeated.routes):
        if not torch.equal(actual.indices, again.indices) or not torch.equal(
            actual.weights, again.weights
        ):
            raise ValidationError(
                f"MTP block {actual.block_index} route replay is not bit-repeatable"
            )

    route_reports = []
    for route in result.routes:
        unique = sorted(set(int(value) for value in route.indices.flatten().tolist()))
        if unique != sorted(selected[route.block_index]):
            raise ValidationError(
                f"MTP block {route.block_index} executed route set differs from materialized experts"
            )
        route_reports.append(
            {
                "block": route.block_index,
                "rows": batch_size * 5,
                "top_k": int(route.indices.shape[-1]),
                "selected_experts": unique,
                "selected_expert_count": len(unique),
                "indices_sha256": tensor_sha256(route.indices),
                "weights_sha256": tensor_sha256(route.weights),
                "logits_dtype": str(route.logits.dtype),
                "weights_dtype": str(route.weights.dtype),
                "bit_repeatable": True,
            }
        )

    return {
        "schema": SCHEMA,
        "status": "checkpoint-body-replay-passed-natural-input-pending",
        "production_ready": False,
        "input_kind": INPUT_KIND,
        "snapshot": os.fspath(snapshot),
        "checkpoint_revision": snapshot.name,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "gptqmodel_version": importlib.metadata.version("gptqmodel"),
        "gptqmodel_revision": None if lock is None else lock.get("revision"),
        "gptqmodel_source_tree_sha256": None
        if lock is None
        else lock.get("source_tree_sha256"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "python_gil_enabled": bool(getattr(__import__("sys"), "_is_gil_enabled")()),
        "device": _device_report(device),
        "geometry": {
            "batch_size": batch_size,
            "main_rows": main_rows,
            "target_taps": list(config.dspark_target_layer_ids),
            "hidden_size": int(config.hidden_size),
            "hc_mult": int(config.hc_mult),
            "proposal_rows_per_batch": 5,
            "joint_proposal_rows": batch_size * 5,
            "noise_token_id": int(config.dspark_noise_token_id),
        },
        "materialized_envelope_modules": envelope,
        "materialized_envelope_module_count": len(envelope),
        "materialized_routed_projection_count": sum(
            len(names) for names in expert_modules.values()
        ),
        "projected_main_sha256": tensor_sha256(result.projected_main),
        "terminal_residual_sha256": tensor_sha256(result.terminal_residual),
        "proposal_token_ids": result.proposal_token_ids.cpu().tolist(),
        "proposal_position_ids": result.proposal_position_ids.cpu().tolist(),
        "routes": route_reports,
        "remaining_gate": (
            "Replace the deterministic target-tap control with taps and anchors "
            "captured from quantized-prefix target replay over the calibration corpus; "
            "then feed these same block inputs into EXL3 Hessian collection."
        ),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--main-rows", type=int, default=8)
    parser.add_argument(
        "--gptqmodel-lock", type=Path, default=root / "third_party/gptqmodel.lock.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports/quantization/flash-mtp-replay.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(
        args.snapshot,
        device_name=args.device,
        batch_size=args.batch_size,
        main_rows=args.main_rows,
        gptqmodel_lock=args.gptqmodel_lock,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validate-flash-mtp-replay: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
