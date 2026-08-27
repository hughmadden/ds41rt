#!/usr/bin/env python3
"""Compare an external full-model or rank-sliced Flash EXL3 control with native FP4."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from ds4rt_runtime.exl3_quantizer import (  # noqa: E402
    deterministic_expert_row_indices,
    load_activation_corpus,
    load_activation_layer_samples,
    read_native_model_config,
)
from ds4rt_runtime.native_experts import (  # noqa: E402
    NativeExpertConfig,
    load_native_expert_reference_layer,
    read_native_expert_config,
)


REPORT_SCHEMA = "ds4rt-external-exl3-control-quality-v1"
PROJECTIONS = ("w1", "w3", "w2")


@dataclass(frozen=True)
class ControlLayout:
    name: str
    ranks: int
    expert_parallel: bool | None
    local_intermediate_size: int
    marker: str
    weight_map: dict[str, str]
    metadata_paths: tuple[Path, ...]
    declared_shard_sha256: dict[str, str]

    def tensor_prefix(
        self,
        layer_id: int,
        expert_id: int,
        projection: str,
        rank: int,
    ) -> str:
        prefix = control_tensor_prefix(layer_id, expert_id, projection)
        return f"{prefix}.rank{rank}" if self.ranks > 1 else prefix


def stratified_expert_ids(expert_count: int) -> tuple[int, ...]:
    if expert_count < 6:
        raise ValueError("DeepSeek V4 control comparison requires at least six experts")
    return tuple(index * (expert_count - 1) // 5 for index in range(6))


def parse_expert_ids(raw: str, expert_count: int) -> tuple[int, ...]:
    if raw.strip().lower() == "auto":
        return stratified_expert_ids(expert_count)
    try:
        values = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("--expert-ids must be a comma-separated integer list") from error
    if (
        len(values) != 6
        or len(set(values)) != len(values)
        or tuple(sorted(values)) != values
        or values[0] < 0
        or values[-1] >= expert_count
    ):
        raise ValueError(
            "control comparison requires six unique sorted in-range expert IDs"
        )
    return values


def read_control_config(
    snapshot: Path, native: NativeExpertConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_type": "deepseek_v4",
        "hidden_size": native.hidden_size,
        "moe_intermediate_size": native.intermediate_size,
        "num_hidden_layers": native.num_hidden_layers,
        "n_routed_experts": native.global_experts,
        "num_experts_per_tok": native.top_k,
        "swiglu_limit": native.swiglu_limit,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(
                f"external EXL3 control requires config.{key}={value!r}, "
                f"got {raw.get(key)!r}"
            )
    quant = raw.get("quantization_config") or {}
    if quant.get("quant_method") != "exl3":
        raise ValueError("external control is not an EXL3 checkpoint")
    if quant.get("codebook") != "mcg":
        raise ValueError("external control must use the EXL3 mcg codebook")
    if not isinstance(quant.get("bits"), (int, float)) or float(quant["bits"]) <= 0:
        raise ValueError("external control has no positive EXL3 bits setting")
    return raw, quant


def control_tensor_prefix(layer_id: int, expert_id: int, projection: str) -> str:
    if layer_id < 0 or expert_id < 0 or projection not in {"w1", "w2", "w3"}:
        raise ValueError("invalid external EXL3 control tensor coordinates")
    return f"layers.{layer_id}.ffn.experts.{expert_id}.{projection}"


def resolve_control_layout(
    snapshot: Path,
    quant: dict[str, Any],
    *,
    layer_id: int,
    intermediate_size: int,
    expert_count: int,
    native_revision: str | None = None,
) -> ControlLayout:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get(
            "weight_map"
        )
        if not isinstance(weight_map, dict):
            raise ValueError("external control index has no weight_map")
        return ControlLayout(
            name="unsharded_full_model",
            ranks=1,
            expert_parallel=None,
            local_intermediate_size=intermediate_size,
            marker=str(quant["codebook"]),
            weight_map={str(key): str(value) for key, value in weight_map.items()},
            metadata_paths=(snapshot / "config.json", index_path),
            declared_shard_sha256={},
        )

    manifest_path = snapshot / "EXL3_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(
            "external control has neither a safetensors index nor an EXL3 manifest"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        quant.get("version") != "rank-sliced-deepseek-v4-v1"
        or quant.get("codebook") != "mcg"
        or manifest.get("tensor_parallel_size") != 4
        or manifest.get("expert_parallel") is not False
    ):
        raise ValueError("external rank-sliced control is not strict TP4 MCG")
    if (
        native_revision is not None
        and manifest.get("source_revision") != native_revision
    ):
        raise ValueError("external rank-sliced control source revision differs from native")
    if intermediate_size % 4:
        raise ValueError("external strict TP4 control has an indivisible intermediate size")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("external rank-sliced control manifest has no file inventory")
    inventory = {
        str(entry["name"]): str(entry["sha256"])
        for entry in files
        if isinstance(entry, dict)
        and isinstance(entry.get("name"), str)
        and isinstance(entry.get("sha256"), str)
    }
    weight_map: dict[str, str] = {}
    declared: dict[str, str] = {}
    for rank in range(4):
        shard_name = f"exl3-layer-{layer_id:03d}-tp4-rank{rank}.safetensors"
        try:
            declared[shard_name] = inventory[shard_name]
        except KeyError as error:
            raise ValueError(
                f"external rank-sliced control manifest omits {shard_name}"
            ) from error
        for expert_id in range(expert_count):
            for projection in PROJECTIONS:
                prefix = f"{control_tensor_prefix(layer_id, expert_id, projection)}.rank{rank}"
                for suffix in ("suh", "svh", "trellis", "mcg"):
                    weight_map[f"{prefix}.{suffix}"] = shard_name
    return ControlLayout(
        name="strict_tp4_rank_sliced",
        ranks=4,
        expert_parallel=False,
        local_intermediate_size=intermediate_size // 4,
        marker="mcg",
        weight_map=weight_map,
        metadata_paths=(snapshot / "config.json", manifest_path),
        declared_shard_sha256=declared,
    )


def output_metrics(actual: Any, expected: Any) -> dict[str, float]:
    import torch

    actual = actual.float()
    expected = expected.float()
    difference = actual - expected
    expected_norm = torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
    row_cosines = torch.nn.functional.cosine_similarity(actual, expected, dim=1)
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual.reshape(1, -1), expected.reshape(1, -1)
            ).item()
        ),
        "min_row_cosine": float(row_cosines.min().item()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / expected_norm),
        "rmse": float(difference.square().mean().sqrt().item()),
        "max_abs": float(difference.abs().max().item()),
        "reference_rms": float(expected.square().mean().sqrt().item()),
        "control_rms": float(actual.square().mean().sqrt().item()),
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _control_linear(
    store: Any,
    weight_map: dict[str, str],
    handles: dict[str, Any],
    snapshot: Path,
    prefix: str,
    in_features: int,
    out_features: int,
    marker: str,
    device: Any,
) -> Any:
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from safetensors import safe_open

    tensors = {}
    expected: dict[str, tuple[tuple[int, ...] | None, Any]] = {
        "suh": ((in_features,), torch.float16),
        "svh": ((out_features,), torch.float16),
        "trellis": (None, torch.int16),
        marker: ((), torch.int32),
    }
    for suffix, (shape, dtype) in expected.items():
        name = f"{prefix}.{suffix}"
        try:
            shard_name = weight_map[name]
        except KeyError as error:
            raise ValueError(f"external control is missing tensor {name}") from error
        if shard_name not in handles:
            shard = snapshot / shard_name
            if not shard.is_file():
                raise FileNotFoundError(
                    f"external control shard required for layer comparison is absent: {shard}"
                )
            handles[shard_name] = store.enter_context(
                safe_open(shard, framework="pt", device="cpu")
            )
        tensor = handles[shard_name].get_tensor(name)
        if tensor.dtype != dtype or (shape is not None and tuple(tensor.shape) != shape):
            raise ValueError(
                f"external control tensor {name} has shape={tuple(tensor.shape)} "
                f"dtype={tensor.dtype}, expected shape={shape} dtype={dtype}"
            )
        tensors[suffix] = tensor.to(device=device)
    trellis = tensors["trellis"]
    if trellis.ndim != 3 or trellis.shape[-1] % 16:
        raise ValueError(f"external control tensor {prefix}.trellis is malformed")
    return LinearEXL3(
        None,
        in_features,
        out_features,
        suh=tensors["suh"],
        svh=tensors["svh"],
        trellis=trellis,
        mcg=tensors.get("mcg"),
        out_dtype=torch.float16,
        key=prefix,
    )


def compare(args: argparse.Namespace) -> dict[str, Any]:
    exllama_source = ROOT / "third_party" / "exllamav3"
    tools_source = ROOT / "python" / "tools"
    sys.path[:0] = [str(exllama_source), str(tools_source)]

    import torch
    import _pinned_sparkinfer  # noqa: F401

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "control comparison requires exactly one CUDA_VISIBLE_DEVICES GPU"
        )
    if args.rows <= 0:
        raise ValueError("--rows must be positive")

    native_snapshot = args.native_snapshot.resolve(strict=True)
    control_snapshot = args.control_snapshot.resolve(strict=True)
    native_config = read_native_expert_config(native_snapshot)
    control_raw, control_quant = read_control_config(control_snapshot, native_config)
    native_raw = json.loads(
        (native_snapshot / "config.json").read_text(encoding="utf-8")
    )
    if control_raw.get("routed_scaling_factor") != native_raw.get(
        "routed_scaling_factor"
    ):
        raise ValueError("external control routed scaling differs from native Flash")
    if not 0 <= args.layer_id < native_config.num_hidden_layers:
        raise ValueError("external control comparison supports base model layers only")
    expert_ids = parse_expert_ids(args.expert_ids, native_config.global_experts)
    layout = resolve_control_layout(
        control_snapshot,
        control_quant,
        layer_id=args.layer_id,
        intermediate_size=native_config.intermediate_size,
        expert_count=native_config.global_experts,
        native_revision=native_snapshot.name,
    )
    weight_map = layout.weight_map
    required_shards = sorted(
        {
            weight_map[
                f"{layout.tensor_prefix(args.layer_id, expert_id, projection, rank)}.{suffix}"
            ]
            for rank in range(layout.ranks)
            for expert_id in expert_ids
            for projection in PROJECTIONS
            for suffix in ("suh", "svh", "trellis", layout.marker)
        }
    )
    for shard_name in required_shards:
        if not (control_snapshot / shard_name).is_file():
            raise FileNotFoundError(
                f"external control comparison requires missing shard {shard_name}"
            )

    _, model_shape = read_native_model_config(native_snapshot)
    corpus = load_activation_corpus(
        args.activation_corpus,
        snapshot=native_snapshot,
        shape=model_shape,
    )
    captured = load_activation_layer_samples(corpus, args.layer_id)
    if args.rows > int(captured.shape[0]):
        raise ValueError(
            f"--rows={args.rows} exceeds {int(captured.shape[0])} held-out rows"
        )
    indices = deterministic_expert_row_indices(
        total_rows=int(captured.shape[0]),
        rows=args.rows,
        layer_id=args.layer_id,
        expert_id=0,
        seed=args.seed,
    )
    device = torch.device("cuda:0")
    hidden = captured.index_select(0, indices).to(device=device)
    control_hidden = hidden.to(torch.float16)

    generator = torch.Generator(device=device).manual_seed(args.seed + args.layer_id)
    topk_ids = torch.tensor(expert_ids, dtype=torch.int32, device=device).repeat(
        args.rows, 1
    )
    topk_weights = torch.rand(
        (args.rows, native_config.top_k),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    routed_scale = float(control_raw["routed_scaling_factor"])
    topk_weights *= routed_scale / topk_weights.sum(dim=1, keepdim=True)

    native_layer = load_native_expert_reference_layer(
        native_snapshot, args.layer_id, expert_ids
    )
    native_output = native_layer.run_partial(hidden, topk_ids, topk_weights).float().clone()
    torch.cuda.synchronize(device)
    del native_layer
    gc.collect()
    torch.cuda.empty_cache()

    control_output = torch.zeros_like(native_output)
    handles: dict[str, Any] = {}
    with ExitStack() as store:
        for rank in range(layout.ranks):
            for route_slot, expert_id in enumerate(expert_ids):
                prefix = layout.tensor_prefix(
                    args.layer_id, expert_id, "w1", rank
                )
                gate_linear = _control_linear(
                    store,
                    weight_map,
                    handles,
                    control_snapshot,
                    prefix,
                    native_config.hidden_size,
                    layout.local_intermediate_size,
                    layout.marker,
                    device,
                )
                prefix = layout.tensor_prefix(
                    args.layer_id, expert_id, "w3", rank
                )
                up_linear = _control_linear(
                    store,
                    weight_map,
                    handles,
                    control_snapshot,
                    prefix,
                    native_config.hidden_size,
                    layout.local_intermediate_size,
                    layout.marker,
                    device,
                )
                gate = gate_linear.forward(
                    control_hidden, {"reconstruct": True}
                ).float()
                up = up_linear.forward(
                    control_hidden, {"reconstruct": True}
                ).float()
                gate.clamp_(max=native_config.swiglu_limit)
                up.clamp_(
                    min=-native_config.swiglu_limit,
                    max=native_config.swiglu_limit,
                )
                intermediate = (torch.nn.functional.silu(gate) * up).to(
                    torch.float16
                )
                del gate_linear, up_linear, gate, up

                prefix = layout.tensor_prefix(
                    args.layer_id, expert_id, "w2", rank
                )
                down_linear = _control_linear(
                    store,
                    weight_map,
                    handles,
                    control_snapshot,
                    prefix,
                    layout.local_intermediate_size,
                    native_config.hidden_size,
                    layout.marker,
                    device,
                )
                expert_output = down_linear.forward(
                    intermediate, {"reconstruct": True}
                ).float()
                control_output.add_(
                    expert_output * topk_weights[:, route_slot, None]
                )
                del down_linear, intermediate, expert_output
                gc.collect()
                torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    return {
        "schema": REPORT_SCHEMA,
        "native_snapshot": str(native_snapshot),
        "control_snapshot": str(control_snapshot),
        "control_metadata_sha256": {
            path.name: hash_file(path) for path in layout.metadata_paths
        },
        "control_layout": {
            "name": layout.name,
            "tensor_parallel_size": layout.ranks,
            "expert_parallel": layout.expert_parallel,
            "local_intermediate_size": layout.local_intermediate_size,
            "marker": layout.marker,
        },
        "control_shards": [
            {
                "name": shard_name,
                "size": (control_snapshot / shard_name).stat().st_size,
                "resolved_name": (control_snapshot / shard_name).resolve().name,
                **(
                    {"declared_sha256": layout.declared_shard_sha256[shard_name]}
                    if shard_name in layout.declared_shard_sha256
                    else {}
                ),
            }
            for shard_name in required_shards
        ],
        "control_quantization": control_quant,
        "layer_id": args.layer_id,
        "expert_ids": list(expert_ids),
        "rows": args.rows,
        "seed": args.seed,
        "routed_scaling_factor": routed_scale,
        "input": {
            "mode": "checkpoint_bound_heldout_activation",
            "manifest": str(corpus.manifest_path),
            "corpus_sha256": corpus.corpus_sha256,
            "selected_row_indices_sha256": hashlib.sha256(
                indices.numpy().tobytes()
            ).hexdigest(),
            "native_dtype": "bfloat16",
            "control_dtype": "float16",
        },
        "native_to_external_exl3": output_metrics(control_output, native_output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-snapshot", type=Path, required=True)
    parser.add_argument("--control-snapshot", type=Path, required=True)
    parser.add_argument("--activation-corpus", type=Path, required=True)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--expert-ids", default="auto")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
