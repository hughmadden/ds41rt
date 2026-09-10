#!/usr/bin/env python3
"""Replay natural Flash router inputs through GPTQModel's V4 router path.

This is a bounded equivalence test, not a calibration corpus.  It verifies the
captured input and route records, resolves the real checkpoint router tensors
through GPTQModel's LazyTurtle aliases, and compares both stock Transformers
BF16 routing and DS41RT's patched FP32 calibration routing with the native
serving records.  The patched route must select exactly the same expert set on
every row, reproduce route weights within a small FP32 tolerance, and repeat
bit-for-bit on the same GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import tempfile
from types import SimpleNamespace
from typing import Any


SCHEMA = "ds41rt-gptqmodel-flash-router-replay-v1"
INPUT_SCHEMA = "ds41rt-flash-router-replay-subset-v1"
REQUIRED_PURPOSE = "bounded route equivalence only; never GPTQ calibration input"
ROUTE_VALUE = struct.Struct("<Hf")
DEFAULT_WEIGHT_ATOL = 5e-7


class ValidationError(RuntimeError):
    """The captured records or replay result violated the route contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_record_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValidationError(f"record filename must be a plain basename: {value!r}")
    path = (root / value).resolve()
    if path.parent != root.resolve():
        raise ValidationError(f"record path escapes replay directory: {value!r}")
    return path


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.expanduser().resolve(strict=True)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read replay manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValidationError("replay manifest must be a JSON object")
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValidationError(
            f"expected input schema {INPUT_SCHEMA!r}, found {manifest.get('schema')!r}"
        )
    if manifest.get("purpose") != REQUIRED_PURPOSE:
        raise ValidationError("replay subset is not marked as non-calibration evidence")
    revision = manifest.get("checkpoint_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValidationError("replay manifest checkpoint_revision is not a 40-hex revision")
    try:
        int(revision, 16)
    except ValueError as exc:
        raise ValidationError("replay manifest checkpoint_revision is not hexadecimal") from exc
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValidationError("replay manifest must contain at least one record")
    normalized: list[dict[str, Any]] = []
    seen_layers: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValidationError("every replay record must be a JSON object")
        layer = record.get("layer")
        rows = record.get("rows")
        if not isinstance(layer, int) or layer < 0 or layer in seen_layers:
            raise ValidationError(f"invalid or duplicate replay layer: {layer!r}")
        if not isinstance(rows, int) or rows <= 0:
            raise ValidationError(f"invalid replay row count for layer {layer}: {rows!r}")
        normalized_record = {"layer": layer, "rows": rows}
        for key in ("activation", "routes"):
            normalized_record[key] = os.fspath(_safe_record_path(path.parent, record.get(key)))
            digest = record.get(f"{key}_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValidationError(f"invalid {key} SHA-256 for layer {layer}")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ValidationError(f"invalid {key} SHA-256 for layer {layer}") from exc
            normalized_record[f"{key}_sha256"] = digest
        seen_layers.add(layer)
        normalized.append(normalized_record)
    return manifest, normalized


def verify_record_files(
    record: dict[str, Any], *, hidden_size: int, top_k: int
) -> dict[str, Any]:
    activation = Path(record["activation"])
    routes = Path(record["routes"])
    rows = int(record["rows"])
    expected_activation_bytes = rows * hidden_size * 2
    expected_route_bytes = rows * top_k * ROUTE_VALUE.size
    for key, path, expected_bytes in (
        ("activation", activation, expected_activation_bytes),
        ("routes", routes, expected_route_bytes),
    ):
        try:
            actual_bytes = path.stat().st_size
        except OSError as exc:
            raise ValidationError(f"cannot stat layer {record['layer']} {key}: {exc}") from exc
        if actual_bytes != expected_bytes:
            raise ValidationError(
                f"layer {record['layer']} {key} is {actual_bytes} bytes, "
                f"expected {expected_bytes}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != record[f"{key}_sha256"]:
            raise ValidationError(
                f"layer {record['layer']} {key} SHA-256 mismatch: "
                f"expected {record[f'{key}_sha256']}, found {actual_digest}"
            )
    return {
        "activation_bytes": expected_activation_bytes,
        "route_bytes": expected_route_bytes,
    }


def decode_route_file(path: Path, *, rows: int, top_k: int) -> tuple[list[list[int]], list[list[float]]]:
    raw = path.read_bytes()
    expected = rows * top_k * ROUTE_VALUE.size
    if len(raw) != expected:
        raise ValidationError(f"route record is {len(raw)} bytes, expected {expected}")
    values = list(ROUTE_VALUE.iter_unpack(raw))
    indices: list[list[int]] = []
    weights: list[list[float]] = []
    for row in range(rows):
        selected = values[row * top_k : (row + 1) * top_k]
        ids = [expert_id for expert_id, _ in selected]
        if len(set(ids)) != top_k:
            raise ValidationError(f"route row {row} contains duplicate expert IDs")
        indices.append(ids)
        weights.append([weight for _, weight in selected])
    return indices, weights


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


def _router(config: Any, weight: Any, bias: Any, device: Any) -> Any:
    import torch
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter

    router = DeepseekV4TopKRouter(config).to(device)
    router.weight = torch.nn.Parameter(weight.to(device), requires_grad=False)
    router.e_score_correction_bias = bias.to(device)
    router.eval()
    return router


def _patched_router(config: Any, weight: Any, bias: Any, device: Any) -> Any:
    import torch

    from gptqmodel.models.definitions.deepseek_v4 import patch_deepseek_v4_router_precision

    class Mlp(torch.nn.Module):
        def __init__(self, gate: Any) -> None:
            super().__init__()
            self.gate = gate

    class Layer(torch.nn.Module):
        def __init__(self, gate: Any) -> None:
            super().__init__()
            self.mlp = Mlp(gate)

    wrapper = torch.nn.Module()
    wrapper.layer = Layer(_router(config, weight, bias, device))
    if patch_deepseek_v4_router_precision(wrapper) != 1:
        raise ValidationError("GPTQModel did not patch the replay router")
    return wrapper.layer.mlp.gate


def _route_metrics(reference_indices: Any, reference_weights: Any, indices: Any, weights: Any) -> dict[str, Any]:
    import torch

    ranked_rows = torch.any(reference_indices != indices, dim=1)
    reference_order = torch.argsort(reference_indices, dim=1)
    candidate_order = torch.argsort(indices, dim=1)
    reference_ids = torch.gather(reference_indices, 1, reference_order)
    candidate_ids = torch.gather(indices, 1, candidate_order)
    set_rows = torch.any(reference_ids != candidate_ids, dim=1)
    exact_rows = ~set_rows
    max_abs: float | None = None
    if bool(exact_rows.any().item()):
        reference_aligned = torch.gather(reference_weights, 1, reference_order)
        candidate_aligned = torch.gather(weights, 1, candidate_order)
        max_abs = float(
            (reference_aligned[exact_rows] - candidate_aligned[exact_rows])
            .abs()
            .max()
            .item()
        )
    return {
        "expert_set_mismatch_rows": int(set_rows.sum().item()),
        "ranked_route_mismatch_rows": int(ranked_rows.sum().item()),
        "exact_expert_set_rows": int(exact_rows.sum().item()),
        "aligned_weight_max_abs_on_exact_set_rows": max_abs,
    }


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
    manifest_path: Path,
    *,
    device_name: str,
    weight_atol: float = DEFAULT_WEIGHT_ATOL,
    gptqmodel_lock: Path | None = None,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig

    from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
    from gptqmodel.utils.structure import LazyTurtle

    snapshot = snapshot.expanduser().resolve(strict=True)
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest, records = load_manifest(manifest_path)
    if snapshot.name != manifest["checkpoint_revision"]:
        raise ValidationError(
            f"snapshot revision {snapshot.name} differs from replay revision "
            f"{manifest['checkpoint_revision']}"
        )
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    if config_json.get("model_type") != "deepseek_v4":
        raise ValidationError("router replay requires a DeepSeek V4 checkpoint")
    hidden_size = int(config_json["hidden_size"])
    top_k = int(config_json["num_experts_per_tok"])
    expert_count = int(config_json["n_routed_experts"])
    layer_count = int(config_json["num_hidden_layers"])
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    lock = None
    if gptqmodel_lock is not None:
        lock = json.loads(
            gptqmodel_lock.expanduser().resolve(strict=True).read_text(encoding="utf-8")
        )
    if not torch.cuda.is_available() and device_name.startswith("cuda"):
        raise ValidationError("CUDA replay requested but CUDA is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    turtle = _build_turtle(snapshot, DeepSeekV4QModel, LazyTurtle)

    layer_reports: list[dict[str, Any]] = []
    total_rows = 0
    control_set_mismatches = 0
    candidate_set_mismatches = 0
    candidate_max_abs = 0.0
    for record in records:
        layer = int(record["layer"])
        rows = int(record["rows"])
        if layer >= layer_count:
            raise ValidationError(f"replay layer {layer} is outside 0..{layer_count - 1}")
        geometry = verify_record_files(record, hidden_size=hidden_size, top_k=top_k)
        route_ids, route_weights = decode_route_file(
            Path(record["routes"]), rows=rows, top_k=top_k
        )
        reference_indices = torch.tensor(route_ids, dtype=torch.long, device=device)
        reference_weights = torch.tensor(route_weights, dtype=torch.float32, device=device)
        if bool((reference_indices < 0).any().item()) or bool(
            (reference_indices >= expert_count).any().item()
        ):
            raise ValidationError(f"layer {layer} route record contains invalid expert IDs")
        if not bool(torch.isfinite(reference_weights).all().item()):
            raise ValidationError(f"layer {layer} route record contains non-finite weights")

        activation = torch.from_file(
            record["activation"], shared=False, size=rows * hidden_size, dtype=torch.bfloat16
        ).reshape(rows, hidden_size)
        activation = activation.to(device)
        runtime = f"model.layers.{layer}.mlp.gate"
        weight_name = f"layers.{layer}.ffn.gate.weight"
        bias_name = f"layers.{layer}.ffn.gate.bias"
        for parameter, expected in (
            ("weight", weight_name),
            ("e_score_correction_bias", bias_name),
        ):
            resolved = turtle._resolve_checkpoint_tensor_source(runtime, parameter)
            if resolved != (expected, None, None, None):
                raise ValidationError(
                    f"GPTQModel alias {runtime}.{parameter} resolved to {resolved}, "
                    f"expected {expected}"
                )
        try:
            weight_shard = snapshot / weight_map[weight_name]
            bias_shard = snapshot / weight_map[bias_name]
        except KeyError as exc:
            raise ValidationError(f"checkpoint index is missing {exc.args[0]}") from exc
        with safe_open(weight_shard, framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(weight_name)
        with safe_open(bias_shard, framework="pt", device="cpu") as handle:
            bias = handle.get_tensor(bias_name)
        if weight.dtype is not torch.bfloat16 or bias.dtype is not torch.float32:
            raise ValidationError(
                f"layer {layer} router dtypes are {weight.dtype}/{bias.dtype}, "
                "expected BF16/FP32"
            )

        with torch.inference_mode():
            stock = _router(config, weight, bias, device)
            _, stock_weights, stock_indices = stock(activation)
            patched = _patched_router(config, weight, bias, device)
            _, candidate_weights, candidate_indices = patched(activation)
            _, repeated_weights, repeated_indices = patched(activation)
        if not torch.equal(candidate_indices, repeated_indices) or not torch.equal(
            candidate_weights, repeated_weights
        ):
            raise ValidationError(f"layer {layer} FP32 router replay is not bit-repeatable")
        control = _route_metrics(
            reference_indices, reference_weights, stock_indices, stock_weights.float()
        )
        candidate = _route_metrics(
            reference_indices, reference_weights, candidate_indices, candidate_weights
        )
        candidate_abs = candidate["aligned_weight_max_abs_on_exact_set_rows"]
        if candidate["expert_set_mismatch_rows"] != 0:
            raise ValidationError(
                f"layer {layer} FP32 replay changed expert sets on "
                f"{candidate['expert_set_mismatch_rows']} rows"
            )
        if candidate_abs is None or candidate_abs > weight_atol:
            raise ValidationError(
                f"layer {layer} FP32 replay route weights differ by {candidate_abs}, "
                f"limit {weight_atol}"
            )
        total_rows += rows
        control_set_mismatches += control["expert_set_mismatch_rows"]
        candidate_set_mismatches += candidate["expert_set_mismatch_rows"]
        candidate_max_abs = max(candidate_max_abs, candidate_abs)
        layer_reports.append(
            {
                "layer": layer,
                "rows": rows,
                "activation_sha256": record["activation_sha256"],
                "routes_sha256": record["routes_sha256"],
                "geometry": geometry,
                "runtime_module": runtime,
                "checkpoint_weight": weight_name,
                "checkpoint_bias": bias_name,
                "stored_weight_dtype": str(weight.dtype),
                "stored_bias_dtype": str(bias.dtype),
                "control_stock_transformers_bf16": control,
                "candidate_gptqmodel_fp32": {
                    **candidate,
                    "bit_repeatable": True,
                    "logits_dtype": "torch.float32",
                    "route_weight_dtype": str(candidate_weights.dtype),
                },
            }
        )
        del activation, weight, bias, stock, patched
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "schema": SCHEMA,
        "status": "exact-expert-sets",
        "snapshot": os.fspath(snapshot),
        "checkpoint_revision": snapshot.name,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "input_manifest": os.fspath(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "gptqmodel_version": importlib.metadata.version("gptqmodel"),
        "gptqmodel_revision": None if lock is None else lock.get("revision"),
        "gptqmodel_source_tree_sha256": None if lock is None else lock.get("source_tree_sha256"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "python_gil_enabled": bool(getattr(__import__("sys"), "_is_gil_enabled")()),
        "device": _device_report(device),
        "router_contract": {
            "stored_weight_dtype": "torch.bfloat16",
            "stored_bias_dtype": "torch.float32",
            "compute_dtype": "torch.float32",
            "top_k": top_k,
            "expert_count": expert_count,
            "weight_atol": weight_atol,
        },
        "total_rows": total_rows,
        "total_routes": total_rows * top_k,
        "control_stock_transformers_bf16_expert_set_mismatch_rows": control_set_mismatches,
        "candidate_gptqmodel_fp32_expert_set_mismatch_rows": candidate_set_mismatches,
        "candidate_gptqmodel_fp32_weight_max_abs": candidate_max_abs,
        "candidate_bit_repeatable": True,
        "layers": layer_reports,
        "qualification": (
            "The bounded FP32 replay matches native serving expert sets and route "
            "weights. The subset is route-equivalence evidence only and must not be "
            "used as GPTQ calibration data."
        ),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "reports/quantization/router-replay-inputs/manifest.json",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--weight-atol", type=float, default=DEFAULT_WEIGHT_ATOL)
    parser.add_argument(
        "--gptqmodel-lock", type=Path, default=root / "third_party/gptqmodel.lock.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports/quantization/flash-router-replay.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.weight_atol < 0:
        raise ValidationError("--weight-atol must be non-negative")
    report = run(
        args.snapshot,
        args.manifest,
        device_name=args.device,
        weight_atol=args.weight_atol,
        gptqmodel_lock=args.gptqmodel_lock,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validate-flash-router-replay: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
