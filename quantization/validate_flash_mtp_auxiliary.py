#!/usr/bin/env python3
"""Validate GPTQModel's separate DeepSeek-V4 Flash MTP materialization path.

This is a bounded checkpoint/materialization qualification, not activation
calibration. It proves that the auxiliary shell consumes the complete MTP
namespace without entering ordinary target-layer traversal, materializes every
router with FP32 routing math, and decodes representative routed projections
through GPTQModel's auto-module decoder. Natural MTP activation replay remains
a separate gate because it requires the three target taps, projected main KV,
and joint anchor/noise proposal rows.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any


SCHEMA = "ds4rt-gptqmodel-flash-mtp-auxiliary-v1"
DEFAULT_CASES = ((0, 0, "w1"), (1, 127, "w2"), (2, 255, "w3"))
PROJECTION_RUNTIME_NAMES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}


class ValidationError(RuntimeError):
    """The Flash checkpoint or auxiliary shell violated the MTP contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    raw = tensor.detach().cpu().contiguous()
    return hashlib.sha256(raw.view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def mapping_sha256(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for runtime_name, checkpoint_name in sorted(entries):
        digest.update(runtime_name.encode())
        digest.update(b"\0")
        digest.update(checkpoint_name.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def parse_case(value: str) -> tuple[int, int, str]:
    path, separator, projection = value.rpartition(".")
    if not separator or projection not in PROJECTION_RUNTIME_NAMES:
        raise ValidationError(
            f"invalid case {value!r}; expected mtp.<block>:<expert>.<w1|w2|w3>"
        )
    block_path, separator, expert_text = path.partition(":")
    if not separator or not block_path.startswith("mtp."):
        raise ValidationError(
            f"invalid case {value!r}; expected mtp.<block>:<expert>.<w1|w2|w3>"
        )
    block_text = block_path.removeprefix("mtp.")
    if not block_text.isdigit() or not expert_text.isdigit():
        raise ValidationError(
            f"invalid case {value!r}; expected non-negative block/expert indices"
        )
    return int(block_text), int(expert_text), projection


def parse_cases(values: list[str] | None) -> list[tuple[int, int, str]]:
    cases = list(DEFAULT_CASES) if not values else [parse_case(value.strip()) for value in values]
    if len(set(cases)) != len(cases):
        raise ValidationError("MTP auxiliary cases must be unique")
    return cases


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


def _build_harness(snapshot: Path, config: Any, model_definition: Any) -> Any:
    harness = object.__new__(model_definition)
    harness.model = SimpleNamespace(config=config)
    harness.model_local_path = os.fspath(snapshot)
    harness.quantize_config = SimpleNamespace(preprocessors=[])
    return harness


def _resolved_shell_mapping(auxiliary: Any, expected_primary: set[str]) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    tensors = list(auxiliary.model.named_parameters()) + list(auxiliary.model.named_buffers())
    for runtime_name, _ in tensors:
        module_path, separator, leaf = runtime_name.rpartition(".")
        if not separator:
            raise ValidationError(f"MTP shell tensor has no module path: {runtime_name}")
        source = auxiliary.turtle_model._resolve_checkpoint_tensor_source(module_path, leaf)
        if source[0] is None or any(value is not None for value in source[1:]):
            raise ValidationError(f"MTP shell tensor {runtime_name} resolved ambiguously: {source}")
        resolved.append((runtime_name, source[0]))

    checkpoint_names = [checkpoint_name for _, checkpoint_name in resolved]
    duplicates = sorted(
        name for name in set(checkpoint_names) if checkpoint_names.count(name) != 1
    )
    actual = set(checkpoint_names)
    if duplicates or actual != expected_primary:
        raise ValidationError(
            "MTP shell/checkpoint mapping is not one-to-one: "
            f"duplicates={duplicates[:8]} missing={sorted(expected_primary - actual)[:8]} "
            f"unexpected={sorted(actual - expected_primary)[:8]}"
        )
    return resolved


def run(
    snapshot: Path,
    cases: list[tuple[int, int, str]],
    *,
    gptqmodel_lock: Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    from gptqmodel.models.definitions.deepseek_v4 import (
        MTP_BLOCK_COUNT,
        DeepSeekV4QModel,
        expected_deepseek_v4_mtp_checkpoint_keys,
    )
    from gptqmodel.utils.model_dequant import dequantize_deepseek_v4_fp4_expert

    snapshot = snapshot.expanduser().resolve(strict=True)
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    if config_json.get("model_type") != "deepseek_v4":
        raise ValidationError("MTP auxiliary validation requires a DeepSeek V4 checkpoint")
    if config_json.get("expert_dtype") != "fp4":
        raise ValidationError(
            f"expected expert_dtype=fp4, found {config_json.get('expert_dtype')!r}"
        )
    lock = _load_lock(gptqmodel_lock)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    harness = _build_harness(snapshot, config, DeepSeekV4QModel)
    auxiliary = harness.build_mtp_auxiliary()

    expected = expected_deepseek_v4_mtp_checkpoint_keys(config)
    expected_scales = {name for name in expected if name.endswith(".scale")}
    expected_primary = expected - expected_scales
    resolved = _resolved_shell_mapping(auxiliary, expected_primary)

    target_roots = DeepSeekV4QModel.extract_layers_node()
    auxiliary_roots = DeepSeekV4QModel._expand_module_tree_prefixes(
        DeepSeekV4QModel.mtp_auxiliary_module_tree
    )
    if target_roots != ["model.layers"] or auxiliary_roots != ["mtp"]:
        raise ValidationError(
            f"unexpected traversal roots: target={target_roots} auxiliary={auxiliary_roots}"
        )
    try:
        auxiliary.model()
    except RuntimeError as exc:
        if "must not be appended to target layers" not in str(exc):
            raise
    else:
        raise ValidationError("generic MTP forward did not fail closed")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0xD54A_0731)
    router_input = torch.randn(
        (4, int(config.hidden_size)), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    router_reports: list[dict[str, Any]] = []
    for block_index in range(MTP_BLOCK_COUNT):
        gate = auxiliary.block(block_index).mlp.gate
        auxiliary.materialize_nonquant_submodule(gate, device="cpu")
        with torch.inference_mode():
            logits, weights, indices = gate(router_input)
            repeated_logits, repeated_weights, repeated_indices = gate(router_input)
        if not (
            torch.equal(logits, repeated_logits)
            and torch.equal(weights, repeated_weights)
            and torch.equal(indices, repeated_indices)
        ):
            raise ValidationError(f"MTP block {block_index} router is not bit-repeatable")
        if logits.dtype is not torch.float32 or weights.dtype is not torch.float32:
            raise ValidationError(f"MTP block {block_index} router did not compute in FP32")
        router_reports.append(
            {
                "block": block_index,
                "weight_dtype": str(gate.weight.dtype),
                "bias_dtype": str(gate.e_score_correction_bias.dtype),
                "logits_dtype": str(logits.dtype),
                "weights_dtype": str(weights.dtype),
                "rows": router_input.shape[0],
                "top_k": indices.shape[1],
                "indices_sha256": tensor_sha256(indices),
                "weights_sha256": tensor_sha256(weights),
                "bit_repeatable": True,
            }
        )

    num_experts = int(config.n_routed_experts)
    projection_reports: list[dict[str, Any]] = []
    for block_index, expert_index, checkpoint_projection in cases:
        if not 0 <= block_index < MTP_BLOCK_COUNT:
            raise ValidationError(f"MTP block {block_index} outside [0, {MTP_BLOCK_COUNT})")
        if not 0 <= expert_index < num_experts:
            raise ValidationError(f"MTP expert {expert_index} outside [0, {num_experts})")
        runtime_projection = PROJECTION_RUNTIME_NAMES[checkpoint_projection]
        target = getattr(
            auxiliary.block(block_index).mlp.experts[expert_index], runtime_projection
        )
        raw = auxiliary.checkpoint_tensors_for_submodule(target)
        packed = raw.get("weight")
        scales = raw.get("weight_scale")
        if not isinstance(packed, torch.Tensor) or not isinstance(scales, torch.Tensor):
            raise ValidationError(
                f"mtp.{block_index} expert {expert_index} {checkpoint_projection} lacks weight/scale"
            )
        decoded = harness.build_mtp_quant_source_module(
            auxiliary, target, target_dtype=torch.bfloat16
        )
        reference = dequantize_deepseek_v4_fp4_expert(
            packed, scales, target_dtype=torch.bfloat16
        )
        if not torch.equal(decoded.weight, reference):
            raise ValidationError(
                f"mtp.{block_index} expert {expert_index} {checkpoint_projection} decode differs"
            )
        checkpoint_base = (
            f"mtp.{block_index}.ffn.experts.{expert_index}.{checkpoint_projection}"
        )
        projection_reports.append(
            {
                "checkpoint_base": checkpoint_base,
                "runtime_module": (
                    f"mtp.{block_index}.mlp.experts.{expert_index}.{runtime_projection}"
                ),
                "packed_shape": list(packed.shape),
                "packed_dtype": str(packed.dtype),
                "scale_shape": list(scales.shape),
                "scale_dtype": str(scales.dtype),
                "decoded_shape": list(decoded.weight.shape),
                "decoded_dtype": str(decoded.weight.dtype),
                "decoded_bf16_sha256": tensor_sha256(decoded.weight),
                "reference_max_abs": 0.0,
                "bit_exact": True,
            }
        )
        del raw, packed, scales, decoded, reference
        gc.collect()

    return {
        "schema": SCHEMA,
        "status": "exact-materialization",
        "production_ready": False,
        "remaining_gate": (
            "natural MTP activation replay from three target taps, projected main KV, "
            "and jointly issued anchor/noise proposal rows"
        ),
        "snapshot": os.fspath(snapshot),
        "checkpoint_revision": snapshot.name,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "gptqmodel_version": importlib.metadata.version("gptqmodel"),
        "gptqmodel_revision": None if lock is None else lock.get("revision"),
        "gptqmodel_source_tree_sha256": (
            None if lock is None else lock.get("source_tree_sha256")
        ),
        "checkpoint_contract": auxiliary.checkpoint_contract,
        "namespace": {
            "tensor_count": len(expected),
            "primary_tensor_count": len(expected_primary),
            "scale_tensor_count": len(expected_scales),
        },
        "traversal": {
            "target_roots": target_roots,
            "auxiliary_roots": auxiliary_roots,
            "disjoint": set(target_roots).isdisjoint(auxiliary_roots),
            "generic_forward_fails_closed": True,
        },
        "runtime_checkpoint_mapping": {
            "entry_count": len(resolved),
            "one_to_one": True,
            "complete": True,
            "sha256": mapping_sha256(resolved),
        },
        "routers": router_reports,
        "representative_projection_decodes": projection_reports,
        "activation_replay": "not-exercised-by-this-validator",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        help="repeatable representative decode as mtp.<block>:<expert>.<w1|w2|w3>",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gptqmodel-lock",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "third_party"
        / "gptqmodel.lock.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(
        args.snapshot,
        parse_cases(args.case),
        gptqmodel_lock=args.gptqmodel_lock,
    )
    if args.output is not None:
        _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"flash-mtp-auxiliary: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
