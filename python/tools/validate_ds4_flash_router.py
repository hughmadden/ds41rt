#!/usr/bin/env python3
"""Validate the native DeepSeek-V4-Flash router against its PyTorch contract."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile


HIDDEN = 4096
EXPERTS = 256
TOP_K = 6
VOCAB = 129280
ROUTE_SCALE = 1.5
CAPTURE_RE = re.compile(
    r"^capture_(?P<capture>[0-9]+_[0-9]+)_layer_(?P<layer>[0-9]+)_"
    r"rows_(?P<rows>[0-9]+)_expert_input\.bf16$"
)
ROUTE_RECORD = struct.Struct("<Hf")
ROUTE_REPLAY_REPORT_SCHEMA = "ds4rt-flash-route-replay-v1"
ROUTE_REPLAY_REPORT_FILENAME = "route-replay.json"


def tensor_pointer(tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def configure_native(path: Path):
    library = ctypes.CDLL(str(path.resolve()))
    function = library.ds4rt_cuda_ds4_flash_router_topk_bf16_async
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    function.restype = ctypes.c_int
    library.ds4rt_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    library.ds4rt_last_error.restype = ctypes.c_int
    return library, function


def configure_hybrid_native(library):
    linear = library.ds4rt_cuda_linear_bf16_f32_cublas_async
    linear.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    linear.restype = ctypes.c_int
    refine = library.ds4rt_cuda_ds4_flash_router_refine_topk_bf16_async
    refine.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    refine.restype = ctypes.c_int
    return linear, refine


def allocate_hybrid_workspace(rows: int, device):
    import torch

    return torch.empty(rows, EXPERTS, device=device, dtype=torch.float32)


def invoke_hybrid(
    native,
    linear,
    refine,
    hidden,
    weight,
    bias,
    indices,
    scores,
    weights,
    workspace,
    rows: int,
    stream,
) -> None:
    linear_status = linear(
        tensor_pointer(hidden),
        tensor_pointer(weight),
        tensor_pointer(workspace),
        rows,
        HIDDEN,
        EXPERTS,
        stream,
    )
    if linear_status != 0:
        raise RuntimeError(
            f"hybrid router shortlist GEMM failed with status {linear_status}: "
            f"{native_error(native)}"
        )
    refine_status = refine(
        tensor_pointer(hidden),
        tensor_pointer(weight),
        tensor_pointer(bias),
        tensor_pointer(workspace),
        tensor_pointer(indices),
        tensor_pointer(scores),
        tensor_pointer(weights),
        rows,
        stream,
    )
    if refine_status != 0:
        raise RuntimeError(
            f"hybrid router refinement failed with status {refine_status}: "
            f"{native_error(native)}"
        )


def native_error(library) -> str:
    message = ctypes.create_string_buffer(1024)
    library.ds4rt_last_error(message, len(message))
    return message.value.decode("utf-8", errors="replace")


def load_layer(snapshot: Path, layer: int):
    import torch
    from safetensors import safe_open

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"layers.{layer}.ffn.gate"
    names = [f"{prefix}.weight"]
    hash_routing = layer < 3
    names.append(f"{prefix}.tid2eid" if hash_routing else f"{prefix}.bias")
    tensors = {}
    for name in names:
        with safe_open(snapshot / index[name], framework="pt", device="cpu") as file:
            tensors[name] = file.get_tensor(name)
    weight = tensors[f"{prefix}.weight"]
    bias = None if hash_routing else tensors[f"{prefix}.bias"]
    token_to_experts = tensors.get(f"{prefix}.tid2eid")
    assert weight.shape == (EXPERTS, HIDDEN) and weight.dtype == torch.bfloat16
    if hash_routing:
        assert token_to_experts is not None
        assert token_to_experts.shape == (VOCAB, TOP_K)
        assert token_to_experts.dtype == torch.int64
    else:
        assert bias is not None and bias.shape == (EXPERTS,)
        assert bias.dtype == torch.float32
    return weight, bias, token_to_experts, hash_routing


def synthetic_layer(seed: int, hash_routing: bool):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(EXPERTS, HIDDEN, generator=generator).to(torch.bfloat16)
    if hash_routing:
        token_offsets = torch.arange(VOCAB, dtype=torch.int64).unsqueeze(1) * 17
        route_offsets = torch.arange(TOP_K, dtype=torch.int64).unsqueeze(0)
        token_to_experts = (token_offsets + route_offsets).remainder(EXPERTS)
        return weight, None, token_to_experts, True
    bias = torch.randn(EXPERTS, generator=generator, dtype=torch.float32) * 0.05
    return weight, bias, None, False


def parse_layer_selection(value: str) -> list[int]:
    layers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("layer selection contains an empty item")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid descending layer range {part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    if not layers or min(layers) < 3 or max(layers) >= 43:
        raise ValueError("capture replay layers must be learned-router layers in 3..42")
    return sorted(layers)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resolve_capture_metadata(
    activation_root: Path, metadata_path: Path | None
) -> Path:
    activation_root = activation_root.expanduser().resolve(strict=True)
    if metadata_path is None:
        manifest_path = activation_root / "manifest.json"
        metadata_path = (
            manifest_path if manifest_path.is_file() else activation_root / "progress.json"
        )
    return metadata_path.expanduser().resolve(strict=True)


def load_capture_records(
    activation_root: Path,
    metadata_path: Path | None,
    layers: list[int],
) -> list[dict]:
    activation_root = activation_root.expanduser().resolve(strict=True)
    metadata_path = resolve_capture_metadata(activation_root, metadata_path)
    metadata = json.loads(metadata_path.read_text())
    subset_records = metadata.get("records")
    if isinstance(subset_records, list):
        selected = set(layers)
        records = []
        for record in subset_records:
            if not isinstance(record, dict) or not isinstance(record.get("layer"), int):
                raise ValueError("capture subset has an invalid record")
            if record["layer"] not in selected:
                continue
            activation = record.get("activation")
            routes = record.get("routes")
            rows = record.get("rows")
            if (
                not isinstance(activation, str)
                or not isinstance(routes, str)
                or not isinstance(rows, int)
                or rows < 1
            ):
                raise ValueError("capture subset record has invalid paths or rows")
            path = activation_root / activation
            route_path = activation_root / routes
            match = CAPTURE_RE.fullmatch(path.name)
            if (
                match is None
                or int(match.group("layer")) != record["layer"]
                or int(match.group("rows")) != rows
            ):
                raise ValueError(f"capture subset metadata disagrees with {path.name}")
            records.append(
                {
                    "layer": record["layer"],
                    "rows": rows,
                    "path": path,
                    "route_path": route_path,
                    "sha256": record.get("activation_sha256"),
                    "route_sha256": record.get("routes_sha256"),
                }
            )
        missing = sorted(selected - {record["layer"] for record in records})
        if missing:
            raise ValueError(f"capture subset has no retained rows for layers {missing}")
        return records
    prompts = metadata.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError(f"capture metadata {metadata_path} has no prompt list")
    selected = set(layers)
    records: list[dict] = []
    for prompt in prompts:
        prompt_records = prompt.get("capture_files") if isinstance(prompt, dict) else None
        if not isinstance(prompt_records, list):
            raise ValueError("capture metadata has an invalid prompt capture list")
        for record in prompt_records:
            if not isinstance(record, dict) or not isinstance(record.get("layer_id"), int):
                raise ValueError("capture metadata has an invalid capture record")
            if record["layer_id"] not in selected:
                continue
            path_value = record.get("path")
            route_value = record.get("route_path")
            if not isinstance(path_value, str) or not isinstance(route_value, str):
                raise ValueError("capture record has no activation/route paths")
            path = activation_root / path_value
            route_path = activation_root / route_value
            match = CAPTURE_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"capture path has an invalid name: {path}")
            rows = int(match.group("rows"))
            layer = int(match.group("layer"))
            if layer != record["layer_id"] or rows != record.get("rows"):
                raise ValueError(f"capture metadata disagrees with filename {path.name}")
            if path.parent != route_path.parent or route_path.name != (
                path.name.removesuffix(".bf16") + "_routes_u16_f32.bin"
            ):
                raise ValueError(f"capture route path does not pair with {path}")
            records.append(
                {
                    "layer": layer,
                    "rows": rows,
                    "path": path,
                    "route_path": route_path,
                    "sha256": record.get("sha256"),
                    "route_sha256": record.get("route_sha256"),
                }
            )
    missing = sorted(selected - {record["layer"] for record in records})
    if missing:
        raise ValueError(f"capture metadata has no retained rows for layers {missing}")
    return records


def run_capture_replay(args) -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "capture replay requires CUDA_VISIBLE_DEVICES=0 and exactly one visible GPU"
        )
    if args.snapshot is None:
        raise ValueError("capture replay requires --snapshot")
    activation_root = args.activation_root.expanduser().resolve(strict=True)
    metadata_path = resolve_capture_metadata(activation_root, args.metadata)
    metadata_payload = metadata_path.read_bytes()
    snapshot = args.snapshot.expanduser().resolve(strict=True)
    native_library = args.native_library.expanduser().resolve(strict=True)
    layers = parse_layer_selection(args.layers)
    records = load_capture_records(activation_root, metadata_path, layers)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    native, function = configure_native(args.native_library)
    hybrid_functions = configure_hybrid_native(native) if args.hybrid else None
    null = ctypes.c_void_p()
    layer_summaries = []
    mismatch_examples = []
    for layer in layers:
        weight, bias, token_to_experts, hash_routing = load_layer(snapshot, layer)
        assert bias is not None and token_to_experts is None and not hash_routing
        weight = weight.to(device)
        bias = bias.to(device)
        remaining = args.max_capture_rows_per_layer
        summary = {
            "layer": layer,
            "rows": 0,
            "routes": 0,
            "ranked_mismatch_rows": 0,
            "set_mismatch_rows": 0,
            "route_entry_mismatches": 0,
            "weight_bit_mismatches": 0,
            "weight_max_abs": 0.0,
        }
        for record in (record for record in records if record["layer"] == layer):
            if remaining == 0:
                break
            activation_bytes = record["path"].read_bytes()
            route_bytes = record["route_path"].read_bytes()
            if record["sha256"] is not None and sha256_bytes(activation_bytes) != record["sha256"]:
                raise ValueError(f"activation checksum differs for {record['path']}")
            if (
                record["route_sha256"] is not None
                and sha256_bytes(route_bytes) != record["route_sha256"]
            ):
                raise ValueError(f"route checksum differs for {record['route_path']}")
            rows = record["rows"]
            if len(activation_bytes) != rows * HIDDEN * 2:
                raise ValueError(f"activation geometry differs for {record['path']}")
            if len(route_bytes) != rows * TOP_K * ROUTE_RECORD.size:
                raise ValueError(f"route geometry differs for {record['route_path']}")
            selected_rows = rows if remaining < 0 else min(rows, remaining)
            activation_bytes = activation_bytes[: selected_rows * HIDDEN * 2]
            route_bytes = route_bytes[: selected_rows * TOP_K * ROUTE_RECORD.size]
            route_records = list(ROUTE_RECORD.iter_unpack(route_bytes))
            recorded_indices = torch.tensor(
                [record[0] for record in route_records], dtype=torch.int64
            ).view(selected_rows, TOP_K)
            recorded_weights = torch.tensor(
                [record[1] for record in route_records], dtype=torch.float32
            ).view(selected_rows, TOP_K)
            hidden = torch.frombuffer(bytearray(activation_bytes), dtype=torch.bfloat16)
            hidden = hidden.view(selected_rows, HIDDEN).to(device)
            token_ids = torch.zeros(selected_rows, device=device, dtype=torch.int64)
            indices = torch.empty(
                selected_rows, TOP_K, device=device, dtype=torch.uint32
            )
            score_workspace = torch.empty(
                selected_rows * (TOP_K + EXPERTS),
                device=device,
                dtype=torch.float32,
            )
            weights = torch.empty(
                selected_rows, TOP_K, device=device, dtype=torch.float32
            )
            hybrid_workspace = (
                allocate_hybrid_workspace(selected_rows, device)
                if args.hybrid
                else None
            )
            first = None
            for repeat in range(args.repeats):
                if hybrid_functions is None:
                    status = function(
                        tensor_pointer(hidden),
                        tensor_pointer(weight),
                        tensor_pointer(bias),
                        null,
                        tensor_pointer(token_ids),
                        tensor_pointer(indices),
                        tensor_pointer(score_workspace),
                        tensor_pointer(weights),
                        selected_rows,
                        0,
                        stream,
                    )
                    if status != 0:
                        raise RuntimeError(
                            f"native Flash router failed with status {status}: "
                            f"{native_error(native)}"
                        )
                else:
                    invoke_hybrid(
                        native,
                        *hybrid_functions,
                        hidden,
                        weight,
                        bias,
                        indices,
                        score_workspace[: selected_rows * TOP_K],
                        weights,
                        hybrid_workspace,
                        selected_rows,
                        stream,
                    )
                torch.cuda.synchronize(device)
                current = (indices.clone(), weights.clone())
                if first is None:
                    first = current
                elif not all(
                    torch.equal(current_value, first_value)
                    for current_value, first_value in zip(current, first, strict=True)
                ):
                    raise AssertionError(
                        f"capture {record['path'].name} replay {repeat} was not bitwise stable"
                    )
            actual_indices = indices.to(torch.int64).cpu()
            actual_weights = weights.cpu()
            ranked = actual_indices != recorded_indices
            set_mismatch = (
                actual_indices.sort(dim=1).values
                != recorded_indices.sort(dim=1).values
            )
            weight_bits = (
                actual_weights.view(torch.int32) != recorded_weights.view(torch.int32)
            )
            ranked_rows = ranked.any(dim=1)
            set_rows = set_mismatch.any(dim=1)
            summary["rows"] += selected_rows
            summary["routes"] += selected_rows * TOP_K
            summary["ranked_mismatch_rows"] += int(ranked_rows.sum())
            summary["set_mismatch_rows"] += int(set_rows.sum())
            summary["route_entry_mismatches"] += int(ranked.sum())
            summary["weight_bit_mismatches"] += int(weight_bits.sum())
            summary["weight_max_abs"] = max(
                summary["weight_max_abs"],
                float((actual_weights - recorded_weights).abs().max()),
            )
            if ranked_rows.any() and len(mismatch_examples) < 8:
                for row in ranked_rows.nonzero().flatten().tolist():
                    mismatch_examples.append(
                        {
                            "capture": record["path"].name,
                            "row": row,
                            "recorded": recorded_indices[row].tolist(),
                            "replayed": actual_indices[row].tolist(),
                        }
                    )
                    if len(mismatch_examples) == 8:
                        break
            if remaining > 0:
                remaining -= selected_rows
        layer_summaries.append(summary)
    return {
        "schema": ROUTE_REPLAY_REPORT_SCHEMA,
        "activation_root": str(activation_root),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_bytes(metadata_payload),
        "snapshot": str(snapshot),
        "native_library": str(native_library),
        "native_library_sha256": sha256_bytes(native_library.read_bytes()),
        "layers": layers,
        "repeats": args.repeats,
        "hybrid": args.hybrid,
        "max_capture_rows_per_layer": args.max_capture_rows_per_layer,
        "rows": sum(summary["rows"] for summary in layer_summaries),
        "routes": sum(summary["routes"] for summary in layer_summaries),
        "ranked_mismatch_rows": sum(
            summary["ranked_mismatch_rows"] for summary in layer_summaries
        ),
        "set_mismatch_rows": sum(
            summary["set_mismatch_rows"] for summary in layer_summaries
        ),
        "route_entry_mismatches": sum(
            summary["route_entry_mismatches"] for summary in layer_summaries
        ),
        "weight_bit_mismatches": sum(
            summary["weight_bit_mismatches"] for summary in layer_summaries
        ),
        "mismatch_examples": mismatch_examples,
        "layer_summaries": layer_summaries,
    }


def write_json_atomic(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run(args) -> dict[str, float | int | bool]:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "validation requires CUDA_VISIBLE_DEVICES=0 and exactly one visible GPU"
        )
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    native, function = configure_native(args.native_library)
    hybrid_functions = configure_hybrid_native(native) if args.hybrid else None

    if args.snapshot is None:
        weight, bias, token_to_experts, hash_routing = synthetic_layer(
            args.seed, args.hash_routing
        )
    else:
        weight, bias, token_to_experts, hash_routing = load_layer(
            args.snapshot, args.layer
        )

    hidden = (
        torch.randn(args.rows, HIDDEN, device=device, dtype=torch.float32)
        / math.sqrt(HIDDEN)
    ).to(torch.bfloat16)
    weight = weight.to(device)
    bias = None if bias is None else bias.to(device)
    token_to_experts = (
        None if token_to_experts is None else token_to_experts.to(device)
    )
    token_ids = torch.randint(
        0, VOCAB, (args.rows,), device=device, dtype=torch.int64
    )
    indices = torch.empty(args.rows, TOP_K, device=device, dtype=torch.uint32)
    score_workspace = torch.empty(
        args.rows * (TOP_K + (0 if hash_routing else EXPERTS)),
        device=device,
        dtype=torch.float32,
    )
    scores = score_workspace[: args.rows * TOP_K].view(args.rows, TOP_K)
    weights = torch.empty(args.rows, TOP_K, device=device, dtype=torch.float32)
    null = ctypes.c_void_p()
    if args.hybrid and hash_routing:
        raise ValueError("hybrid refinement only applies to learned routing")
    hybrid_workspace = (
        allocate_hybrid_workspace(args.rows, device) if args.hybrid else None
    )

    first_native = None
    for repeat in range(args.repeats):
        if hybrid_functions is None:
            status = function(
                tensor_pointer(hidden),
                tensor_pointer(weight),
                null if bias is None else tensor_pointer(bias),
                null if token_to_experts is None else tensor_pointer(token_to_experts),
                tensor_pointer(token_ids),
                tensor_pointer(indices),
                tensor_pointer(score_workspace),
                tensor_pointer(weights),
                args.rows,
                int(hash_routing),
                stream,
            )
            if status != 0:
                raise RuntimeError(
                    f"native Flash router failed with status {status}: "
                    f"{native_error(native)}"
                )
        else:
            invoke_hybrid(
                native,
                *hybrid_functions,
                hidden,
                weight,
                bias,
                indices,
                scores,
                weights,
                hybrid_workspace,
                args.rows,
                stream,
            )
        torch.cuda.synchronize(device)
        current_native = (indices.clone(), scores.clone(), weights.clone())
        if first_native is None:
            first_native = current_native
        elif not all(
            torch.equal(current, first)
            for current, first in zip(current_native, first_native, strict=True)
        ):
            raise AssertionError(f"native router replay {repeat} was not bitwise stable")

    logits = functional.linear(hidden.float(), weight.float())
    reference_scores = functional.softplus(logits).sqrt()
    if hash_routing:
        reference_indices = token_to_experts[token_ids]
    else:
        reference_indices = (reference_scores + bias).topk(TOP_K, dim=-1).indices
    reference_selected = reference_scores.gather(1, reference_indices)
    reference_weights = (
        reference_selected
        / reference_selected.sum(dim=-1, keepdim=True)
        * ROUTE_SCALE
    )
    indices_i64 = indices.to(torch.int64)
    ranked_mismatched_rows = (indices_i64 != reference_indices).any(dim=1).nonzero().flatten()
    native_sets = indices_i64.sort(dim=1).values
    reference_sets = reference_indices.sort(dim=1).values
    set_mismatched_rows = (native_sets != reference_sets).any(dim=1).nonzero().flatten()
    if set_mismatched_rows.numel():
        examples = []
        for row in set_mismatched_rows[:8].cpu().tolist():
            examples.append(
                {
                    "row": row,
                    "native": indices_i64[row].cpu().tolist(),
                    "reference": reference_indices[row].cpu().tolist(),
                }
            )
        raise AssertionError(
            f"router expert sets differ in {set_mismatched_rows.numel()}/{args.rows} rows: "
            f"examples={examples}"
        )
    if not hash_routing:
        reference_selected = reference_scores.gather(1, indices_i64)
        reference_weights = (
            reference_selected
            / reference_selected.sum(dim=-1, keepdim=True)
            * ROUTE_SCALE
        )
    score_error = (scores - reference_selected).abs().max().item()
    weight_error = (weights - reference_weights).abs().max().item()
    if score_error > 2.0e-4 or weight_error > 2.0e-4:
        raise AssertionError(
            f"router numeric mismatch: score_max_abs={score_error} "
            f"weight_max_abs={weight_error}"
        )
    return {
        "rows": args.rows,
        "repeats": args.repeats,
        "hash_routing": hash_routing,
        "hybrid": args.hybrid,
        "ranked_mismatched_rows": ranked_mismatched_rows.numel(),
        "score_max_abs": score_error,
        "weight_max_abs": weight_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--activation-root",
        type=Path,
        help="replay learned-router records from a collected activation corpus",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="capture manifest/progress path (defaults to activation-root metadata)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "durable replay report path; production uses "
            f"activation-root/{ROUTE_REPLAY_REPORT_FILENAME}"
        ),
    )
    parser.add_argument(
        "--layers",
        default="3-42",
        help="comma-separated learned-router layers or inclusive ranges",
    )
    parser.add_argument(
        "--max-capture-rows-per-layer",
        type=int,
        default=1024,
        help="bounded replay rows per layer; -1 replays every retained row",
    )
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--hash-routing", action="store_true")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="validate the FP32 shortlist plus exact 12-candidate refinement path",
    )
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.max_capture_rows_per_layer == 0 or args.max_capture_rows_per_layer < -1:
        parser.error("--max-capture-rows-per-layer must be -1 or positive")
    if args.snapshot is not None and not (0 <= args.layer < 43):
        parser.error("--layer must be in 0..42")
    if args.activation_root is not None:
        try:
            summary = run_capture_replay(args)
        except ValueError as error:
            parser.error(str(error))
        mismatch = any(
            summary[key]
            for key in (
                "ranked_mismatch_rows",
                "set_mismatch_rows",
                "route_entry_mismatches",
                "weight_bit_mismatches",
            )
        )
        summary["status"] = "mismatch" if mismatch else "exact"
        if args.report is not None:
            write_json_atomic(args.report, summary)
        print(json.dumps(summary, sort_keys=True))
        if mismatch:
            raise SystemExit(1)
    else:
        if args.metadata is not None:
            parser.error("--metadata requires --activation-root")
        if args.report is not None:
            parser.error("--report requires --activation-root")
        print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
