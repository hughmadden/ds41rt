#!/usr/bin/env python3
"""Validate the real checkpoint-backed Flash target-to-MTP prefix runtime.

This bounded check loads a meta target shell, materializes only the official
MTP main projector/norm plus the target HC head, norm, vocabulary head, and
embedding, and exercises deterministic projection and greedy anchor
resolution.  It also installs one replay-ready position into GPTQModel's
disjoint three-block auxiliary input adapter without materializing an MTP body
block.  It does not quantize weights or substitute for natural corpus capture
through the already-quantized target prefix.
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
import time
from types import SimpleNamespace
from typing import Any


SCHEMA = "ds41rt-gptqmodel-flash-mtp-prefix-runtime-v1"
INPUT_KIND = "deterministic-runtime-control-not-natural-calibration"


class ValidationError(RuntimeError):
    """The source checkpoint or materialized target/MTP prefix is inconsistent."""


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


def _load_lock(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"GPTQModel lock is not an object: {resolved}")
    return value


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


def _tensor_record(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def _resolved_source(turtle: Any, module_path: str, leaf: str) -> str:
    source = turtle._resolve_checkpoint_tensor_source(module_path, leaf)
    if source[0] is None or any(value is not None for value in source[1:]):
        raise ValidationError(
            f"runtime tensor {module_path}.{leaf} resolved ambiguously: {source}"
        )
    return str(source[0])


def _source_mapping(model: Any, runtime: Any) -> dict[str, str]:
    mapping = {
        "mtp.0.main_proj.weight": _resolved_source(
            runtime.auxiliary.turtle_model, "mtp.0.main_proj", "weight"
        ),
        "mtp.0.main_norm.weight": _resolved_source(
            runtime.auxiliary.turtle_model, "mtp.0.main_norm", "weight"
        ),
        "model.hc_head.hc_fn": _resolved_source(
            model.turtle_model, "model.hc_head", "hc_fn"
        ),
        "model.hc_head.hc_base": _resolved_source(
            model.turtle_model, "model.hc_head", "hc_base"
        ),
        "model.hc_head.hc_scale": _resolved_source(
            model.turtle_model, "model.hc_head", "hc_scale"
        ),
        "model.norm.weight": _resolved_source(
            model.turtle_model, "model.norm", "weight"
        ),
        "lm_head.weight": _resolved_source(model.turtle_model, "lm_head", "weight"),
        "model.embed_tokens.weight": _resolved_source(
            model.turtle_model, "model.embed_tokens", "weight"
        ),
    }
    expected = {
        "mtp.0.main_proj.weight": "mtp.0.main_proj.weight",
        "mtp.0.main_norm.weight": "mtp.0.main_norm.weight",
        "model.hc_head.hc_fn": "hc_head_fn",
        "model.hc_head.hc_base": "hc_head_base",
        "model.hc_head.hc_scale": "hc_head_scale",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "head.weight",
        "model.embed_tokens.weight": "embed.weight",
    }
    if mapping != expected:
        raise ValidationError(
            f"target/MTP prefix checkpoint mapping changed: actual={mapping} expected={expected}"
        )
    return mapping


def run(
    snapshot: Path,
    *,
    device_name: str,
    gptqmodel_lock: Path | None,
) -> dict[str, Any]:
    import torch

    snapshot = snapshot.expanduser().resolve(strict=True)
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    if config_json.get("model_type") != "deepseek_v4":
        raise ValidationError("MTP prefix validation requires a DeepSeek V4 checkpoint")
    if config_json.get("expert_dtype") != "fp4":
        raise ValidationError("Flash MTP prefix validation expects native FP4 experts")
    if config_json.get("torch_dtype") != "bfloat16":
        raise ValidationError("Flash MTP prefix validation expects a BF16 target checkpoint")
    if not index_path.is_file():
        raise ValidationError("Flash checkpoint has no safetensors index")

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValidationError("CUDA prefix validation requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValidationError(f"unsupported prefix validation device {device}")

    from gptqmodel import GPTQModel
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.models.definitions.deepseek_v4 import (
        DeepSeekV4MTPReplayBatch,
        DeepSeekV4QModel,
    )
    from gptqmodel.quantization import AutoModuleDecoderConfig, EXL3Config

    lock = _load_lock(gptqmodel_lock)
    routed_expert = (
        r"^model\.layers\.0\.mlp\.experts\.0\."
        r"(?:gate_proj|up_proj|down_proj)$"
    )
    qcfg = EXL3Config(
        bits=2.0,
        module_include=[routed_expert],
        preprocessors=[AutoModuleDecoderConfig(target_dtype=torch.bfloat16)],
        offload_to_disk=True,
        device=str(device),
    )
    started = time.perf_counter()
    model = GPTQModel.load(
        os.fspath(snapshot),
        quantize_config=qcfg,
        trust_remote_code=False,
    )
    if not isinstance(model, DeepSeekV4QModel):
        raise ValidationError(f"unexpected model definition {type(model).__name__}")
    shell_seconds = time.perf_counter() - started

    started = time.perf_counter()
    runtime = model.build_mtp_prefix_runtime(
        device=device,
        position_chunk_size=1,
        vocab_chunk_size=8192,
    )
    materialize_seconds = time.perf_counter() - started
    mapping = _source_mapping(model, runtime)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0xD54A_0731)
    hidden_size = int(model.model.config.hidden_size)
    hc_mult = int(model.model.config.hc_mult)
    target_taps = tuple(
        torch.randn(1, 2, hidden_size, generator=generator, dtype=torch.float32)
        .to(dtype=torch.bfloat16, device=device)
        for _ in range(3)
    )
    raw_hidden = torch.randn(
        1,
        2,
        hc_mult,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16, device=device)
    input_ids = torch.tensor([[17, 18]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, 2), dtype=torch.bool, device=device)
    decode_mask = torch.tensor([[False, True]], dtype=torch.bool, device=device)
    position_ids = torch.tensor([[31, 32]], dtype=torch.long, device=device)

    started = time.perf_counter()
    with torch.inference_mode():
        projected = runtime.project_target_taps(target_taps)
        anchors = runtime.anchor_resolver(
            raw_hidden,
            input_ids,
            attention_mask,
            decode_mask,
            position_ids,
        )
        repeated_projected = runtime.project_target_taps(target_taps)
        repeated_anchors = runtime.anchor_resolver(
            raw_hidden,
            input_ids,
            attention_mask,
            decode_mask,
            position_ids,
        )
    exercise_seconds = time.perf_counter() - started
    if not torch.equal(projected, repeated_projected):
        raise ValidationError("MTP main projection is not bit-repeatable")
    if not torch.equal(anchors, repeated_anchors):
        raise ValidationError("target greedy anchor resolution is not bit-repeatable")
    if anchors.tolist()[0][0] != -1:
        raise ValidationError("ineligible target position received an anchor")
    anchor = int(anchors[0, 1])
    if not 0 <= anchor < int(model.model.config.vocab_size):
        raise ValidationError(f"eligible target anchor {anchor} is outside the vocabulary")
    replay = runtime.build_replay()
    if replay.embedding_weight is not runtime.target_embedding.weight:
        raise ValidationError("MTP replay does not own the materialized target embedding")

    started = time.perf_counter()
    adapter = model.build_mtp_quantization_model(
        runtime,
        calibration_embedding_device="cpu",
    )
    replay_batch = DeepSeekV4MTPReplayBatch(
        target_taps=None,
        projected_main=projected.detach().cpu(),
        anchor_token_ids=torch.tensor([anchor], dtype=torch.long),
        main_position_ids=position_ids.detach().cpu(),
        main_attention_mask=attention_mask.detach().cpu(),
    )
    prepared = adapter.prepare_dataset(
        [replay_batch],
        calibration_dataset_sort=None,
        batch_size=1,
    )
    looper = SimpleNamespace(
        gptq_model=adapter,
        _batch_row_count=lambda value: int(value[0].shape[0]),
    )
    cache = StageInputsCapture(looper).cache_inputs(
        layers=list(adapter.model.mtp),
        layer_names=[f"mtp.{index}" for index in range(3)],
        calibration_data=prepared,
        use_cache=False,
    )
    adapter_seconds = time.perf_counter() - started
    if len(cache.layer_inputs) != 1 or len(cache.layer_inputs[0]) != 1:
        raise ValidationError("MTP auxiliary adapter did not capture one joint input")
    captured_residual = cache.layer_inputs[0][0]
    if tuple(captured_residual.shape) != (1, 5, hc_mult, hidden_size):
        raise ValidationError(
            "MTP auxiliary adapter changed the five-row residual geometry: "
            f"{tuple(captured_residual.shape)}"
        )
    if len(cache.position_ids) != 1 or tuple(cache.position_ids[0].shape) != (1, 5):
        raise ValidationError("MTP auxiliary adapter lost proposal positions")
    captured_kwargs = cache.layer_input_kwargs[0]
    required_replay_kwargs = {
        "_gptqmodel_mtp_projected_main",
        "_gptqmodel_mtp_main_position_ids",
        "_gptqmodel_mtp_proposal_token_ids",
        "_gptqmodel_mtp_joint_attention_mask",
        "_gptqmodel_mtp_proposal_position_embeddings",
        "_gptqmodel_mtp_main_position_embeddings",
    }
    missing_replay_kwargs = sorted(required_replay_kwargs - set(captured_kwargs))
    if missing_replay_kwargs:
        raise ValidationError(
            "MTP auxiliary adapter lost replay kwargs: "
            + ", ".join(missing_replay_kwargs)
        )
    layer_modules = adapter.simple_layer_modules(
        model_config=adapter.model.config,
        quantize_config=adapter.quantize_config,
    )
    routed_suffixes = [name for group in layer_modules for name in group]
    expected_per_block = int(model.model.config.n_routed_experts) * 3
    if len(routed_suffixes) != expected_per_block:
        raise ValidationError(
            "MTP auxiliary quantization scope is incomplete: "
            f"actual={len(routed_suffixes)} expected={expected_per_block}"
        )
    if any(
        name.startswith("self_attn.") or name.startswith("mlp.shared_experts.")
        for name in routed_suffixes
    ):
        raise ValidationError("MTP auxiliary adapter exposed coordinator modules")

    main_proj_raw = runtime.auxiliary.checkpoint_tensors_for_submodule(
        runtime.auxiliary.block(0).main_proj
    )
    if not isinstance(main_proj_raw.get("weight"), torch.Tensor) or not isinstance(
        main_proj_raw.get("weight_scale"), torch.Tensor
    ):
        raise ValidationError("MTP main projection lacks its packed weight/scale pair")

    materialized = {
        "mtp.0.main_proj.weight": _tensor_record(
            runtime.auxiliary.block(0).main_proj.weight
        ),
        "mtp.0.main_norm.weight": _tensor_record(
            runtime.auxiliary.block(0).main_norm.weight
        ),
        "model.hc_head.hc_fn": _tensor_record(runtime.target_hc_head.hc_fn),
        "model.hc_head.hc_base": _tensor_record(runtime.target_hc_head.hc_base),
        "model.hc_head.hc_scale": _tensor_record(runtime.target_hc_head.hc_scale),
        "model.norm.weight": _tensor_record(runtime.target_norm.weight),
        "lm_head.weight": _tensor_record(runtime.target_lm_head.weight),
        "model.embed_tokens.weight": _tensor_record(runtime.target_embedding.weight),
    }
    return {
        "schema": SCHEMA,
        "status": "checkpoint-prefix-and-mtp-adapter-passed-natural-capture-pending",
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
        "durations_seconds": {
            "shell_load": shell_seconds,
            "prefix_materialization": materialize_seconds,
            "projection_and_anchor_control_twice": exercise_seconds,
            "auxiliary_input_adapter": adapter_seconds,
        },
        "runtime_checkpoint_mapping": mapping,
        "materialized_tensors": materialized,
        "main_projection_source": {
            "weight": _tensor_record(main_proj_raw["weight"]),
            "scale": _tensor_record(main_proj_raw["weight_scale"]),
            "decoded_dtype": str(runtime.auxiliary.block(0).main_proj.weight.dtype),
        },
        "control": {
            "rows": 2,
            "eligible_anchor_rows": 1,
            "projected_main_sha256": tensor_sha256(projected),
            "anchors_sha256": tensor_sha256(anchors),
            "anchors": anchors.cpu().tolist(),
            "bit_repeatable": True,
        },
        "auxiliary_quantization_adapter": {
            "layer_names": ["mtp.0", "mtp.1", "mtp.2"],
            "quantization_scope": "routed_experts_only",
            "routed_projections_per_block": len(routed_suffixes),
            "routed_projections_total": len(routed_suffixes) * 3,
            "captured_residual": _tensor_record(captured_residual),
            "captured_residual_sha256": tensor_sha256(captured_residual),
            "proposal_position_ids": cache.position_ids[0].cpu().tolist(),
            "replay_kwarg_keys": sorted(required_replay_kwargs),
            "first_block_materialized_for_capture": False,
        },
        "remaining_gate": (
            "Run the target quantization loop over the immutable calibration corpus "
            "with this runtime installed as the synchronous prefix-store projector "
            "and anchor resolver, then feed the durable replay batches into MTP EXL3 "
            "Hessian collection."
        ),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gptqmodel-lock", type=Path, default=root / "third_party/gptqmodel.lock.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports/quantization/flash-mtp-prefix-runtime.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(
        args.snapshot,
        device_name=args.device,
        gptqmodel_lock=args.gptqmodel_lock,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validate-flash-mtp-prefix-runtime: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
