#!/usr/bin/env python3
"""Compare GPTQModel and DS4RT decoding on real DeepSeek-V4-Flash experts.

This is a bounded source qualification, not a quantization run. It reads one
expert from representative early/middle/late target layers and all three MTP
blocks, decodes every w1/w2/w3 projection through the GPTQModel fast and
Torch-only paths and the independent DS4RT reference, and requires bit-exact
BF16 equality. The report also makes model-traversal coverage explicit: the
current Transformers DeepSeek-V4 shell ignores ``mtp.*`` even though those
weights must ultimately be calibrated and quantized for integrated dSpark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any


SCHEMA = "ds4rt-gptqmodel-flash-source-decode-v1"
DEFAULT_CASES = (
    "layers.0:0",
    "layers.21:127",
    "layers.42:255",
    "mtp.0:0",
    "mtp.1:127",
    "mtp.2:255",
)
CASE_RE = re.compile(r"^(layers|mtp)\.([0-9]+):([0-9]+)$")
PROJECTION_ALIASES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}


class ValidationError(RuntimeError):
    """The real source checkpoint failed its bounded decode contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    raw = tensor.detach().cpu().contiguous()
    # ``view(torch.uint8)`` is intentionally kept local so this module remains
    # importable by lightweight unit tests that do not import PyTorch.
    import torch

    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def parse_cases(values: list[str] | tuple[str, ...]) -> list[tuple[str, int, int]]:
    parsed: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for value in values:
        match = CASE_RE.fullmatch(value.strip())
        if match is None:
            raise ValidationError(
                f"invalid case {value!r}; expected layers.<id>:<expert> or mtp.<id>:<expert>"
            )
        case = (match.group(1), int(match.group(2)), int(match.group(3)))
        if case not in seen:
            parsed.append(case)
            seen.add(case)
    if not parsed:
        raise ValidationError("at least one source-decode case is required")
    return parsed


def checkpoint_base(namespace: str, block_id: int, expert_id: int, stem: str) -> str:
    if stem not in PROJECTION_ALIASES:
        raise ValidationError(f"unsupported projection stem: {stem!r}")
    return f"{namespace}.{block_id}.ffn.experts.{expert_id}.{stem}"


def runtime_module_path(
    namespace: str, block_id: int, expert_id: int, stem: str
) -> str | None:
    if namespace != "layers":
        return None
    return (
        f"model.layers.{block_id}.mlp.experts.{expert_id}."
        f"{PROJECTION_ALIASES[stem]}"
    )


def validate_case_geometry(
    cases: list[tuple[str, int, int]], config: dict[str, Any]
) -> None:
    layer_count = int(config["num_hidden_layers"])
    mtp_count = len(config.get("dspark_target_layer_ids") or [])
    experts = int(config["n_routed_experts"])
    for namespace, block_id, expert_id in cases:
        limit = layer_count if namespace == "layers" else mtp_count
        if not 0 <= block_id < limit:
            raise ValidationError(
                f"{namespace}.{block_id} is outside the checkpoint range 0..{limit - 1}"
            )
        if not 0 <= expert_id < experts:
            raise ValidationError(
                f"expert {expert_id} is outside the checkpoint range 0..{experts - 1}"
            )


def _build_turtle(snapshot: Path, model_definition: Any, lazy_turtle: Any) -> Any:
    import torch

    class DeepSeekV4Identity(torch.nn.Module):
        base_model_prefix = "model"

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="deepseek_v4")

    turtle = lazy_turtle.maybe_create(
        model_local_path=os.fspath(snapshot),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
        module_tree=model_definition.module_tree,
        target_model=DeepSeekV4Identity(),
    )
    if turtle is None:
        raise ValidationError("GPTQModel LazyTurtle could not open the Flash checkpoint")
    return turtle


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


def run(
    snapshot: Path,
    cases: list[tuple[str, int, int]],
    *,
    gptqmodel_lock: Path | None = None,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    from ds4rt_runtime.exl3_quantizer import dequantize_native_fp4_projection
    from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
    from gptqmodel.quantization import dtype as gptq_dtype
    from gptqmodel.utils.structure import LazyTurtle

    snapshot = snapshot.expanduser().resolve(strict=True)
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "deepseek_v4":
        raise ValidationError(
            f"expected model_type=deepseek_v4, found {config.get('model_type')!r}"
        )
    if config.get("expert_dtype") != "fp4":
        raise ValidationError(
            f"expected expert_dtype=fp4, found {config.get('expert_dtype')!r}"
        )
    validate_case_geometry(cases, config)
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    lock = None
    if gptqmodel_lock is not None:
        lock_path = gptqmodel_lock.expanduser().resolve(strict=True)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(lock, dict):
            raise ValidationError(f"GPTQModel lock is not an object: {lock_path}")
    turtle = _build_turtle(snapshot, DeepSeekV4QModel, LazyTurtle)

    reports: list[dict[str, Any]] = []
    logical_weights = 0
    source_bytes = 0
    for namespace, block_id, expert_id in cases:
        for stem in PROJECTION_ALIASES:
            base = checkpoint_base(namespace, block_id, expert_id, stem)
            weight_name = f"{base}.weight"
            scale_name = f"{base}.scale"
            try:
                weight_shard = snapshot / weight_map[weight_name]
                scale_shard = snapshot / weight_map[scale_name]
            except KeyError as exc:
                raise ValidationError(f"checkpoint index is missing {exc.args[0]}") from exc

            with safe_open(weight_shard, framework="pt", device="cpu") as handle:
                packed = handle.get_tensor(weight_name)
            with safe_open(scale_shard, framework="pt", device="cpu") as handle:
                scales = handle.get_tensor(scale_name)

            if packed.dtype is not torch.int8 or scales.dtype is not torch.float8_e8m0fnu:
                raise ValidationError(
                    f"{base} has unexpected source dtypes: {packed.dtype}, {scales.dtype}"
                )
            expected_dense_shape = (
                (int(config["moe_intermediate_size"]), int(config["hidden_size"]))
                if stem in {"w1", "w3"}
                else (int(config["hidden_size"]), int(config["moe_intermediate_size"]))
            )
            if tuple(packed.shape[:-1]) + (packed.shape[-1] * 2,) != expected_dense_shape:
                raise ValidationError(
                    f"{base} packed geometry {tuple(packed.shape)} does not decode to "
                    f"{expected_dense_shape}"
                )
            if tuple(scales.shape) != (
                expected_dense_shape[0],
                expected_dense_shape[1] // 32,
            ):
                raise ValidationError(
                    f"{base} scale geometry {tuple(scales.shape)} is not E8M0 K32"
                )

            fast = gptq_dtype.dequantize_f4_e2m1(
                packed,
                scale=scales,
                axis=None,
                target_dtype=torch.bfloat16,
            ).contiguous()
            reference = gptq_dtype._dequantize_f4_reference(
                packed.view(torch.uint8),
                scale=scales,
                axis=None,
                target_dtype=torch.bfloat16,
            ).contiguous()
            ds4rt = (
                dequantize_native_fp4_projection(packed, scales)
                .T.to(torch.bfloat16)
                .contiguous()
            )
            if not torch.equal(fast, reference) or not torch.equal(fast, ds4rt):
                raise ValidationError(
                    f"{base} source decode differs: "
                    f"gptq_reference_max_abs={float((fast.float() - reference.float()).abs().max())} "
                    f"ds4rt_max_abs={float((fast.float() - ds4rt.float()).abs().max())}"
                )
            if not bool(torch.isfinite(fast).all().item()):
                raise ValidationError(f"{base} decoded non-finite values")

            runtime_path = runtime_module_path(
                namespace, block_id, expert_id, stem
            )
            alias_status = "not-modeled"
            if runtime_path is not None:
                resolved = turtle._resolve_checkpoint_tensor_source(
                    runtime_path, "weight"
                )
                if resolved != (weight_name, None, None, None):
                    raise ValidationError(
                        f"GPTQModel alias {runtime_path} resolved to {resolved}, "
                        f"expected {weight_name}"
                    )
                alias_status = "exact"

            logical_weights += fast.numel()
            source_bytes += packed.numel() * packed.element_size()
            source_bytes += scales.numel() * scales.element_size()
            reports.append(
                {
                    "checkpoint_base": base,
                    "runtime_module": runtime_path,
                    "alias_status": alias_status,
                    "packed_shape": list(packed.shape),
                    "scale_shape": list(scales.shape),
                    "decoded_shape": list(fast.shape),
                    "logical_weights": fast.numel(),
                    "source_weight_sha256": tensor_sha256(packed),
                    "source_scale_sha256": tensor_sha256(scales),
                    "decoded_bf16_sha256": tensor_sha256(fast),
                    "gptq_reference_max_abs": 0.0,
                    "ds4rt_max_abs": 0.0,
                }
            )
            del packed, scales, fast, reference, ds4rt
            gc.collect()

    layer_roots = DeepSeekV4QModel.extract_layers_node()
    missing_roots = [root for root in ("mtp",) if root not in layer_roots]
    return {
        "schema": SCHEMA,
        "status": "exact",
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
        "gptq_fast_floatx_extension": gptq_dtype._load_floatx_cpu_ops() is not None,
        "cases": [f"{namespace}.{block_id}:{expert_id}" for namespace, block_id, expert_id in cases],
        "projection_count": len(reports),
        "logical_weights": logical_weights,
        "source_bytes": source_bytes,
        "all_bf16_bit_exact": True,
        "projections": reports,
        "model_traversal": {
            "layer_roots": layer_roots,
            "missing_required_roots": missing_roots,
            "status": "complete" if not missing_roots else "mtp-auxiliary-path-required",
            "note": (
                "Transformers ignores mtp.*; exact source decoding is qualified, "
                "but MTP activation capture/quantization requires an explicit auxiliary path."
                if missing_roots
                else None
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help=(
            "representative block/expert as layers.<id>:<expert> or mtp.<id>:<expert>; "
            "repeat as needed"
        ),
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
        parse_cases(args.cases or list(DEFAULT_CASES)),
        gptqmodel_lock=args.gptqmodel_lock,
    )
    if args.output:
        _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"flash-source-decode: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
