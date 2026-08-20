#!/usr/bin/env python3
"""Compose a standard EXL3 target artifact with calibrated K2 dSpark experts.

This is a serving diagnostic, not a quantization pass.  Base expert tensors and
all retained native tensors come from ``--target``.  Only the three routed MTP
expert blocks come from ``--k2``.  Existing target shards are hardlinked and a
single replacement final shard is streamed from K2, so composition writes only
the dSpark expert payload.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any, BinaryIO

from ds4rt_runtime.exl3_mix import MIX_PLAN_META_KEY, MIX_RECIPE
from ds4rt_runtime.exl3_quantizer import SourceTensor, read_artifact_index
from ds4rt_runtime.exl3_tiers import ExpertBitPlan


COPY_CHUNK_BYTES = 64 * 1024**2
SHARD_RE = re.compile(r"model-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.safetensors\Z")
MODULE_RE = re.compile(
    r"^(?:(?:model\.)?layers\.(?P<base>[0-9]+)|mtp\.(?P<mtp>[0-9]+))"
    r"\.mlp\.experts\.(?P<expert>[0-9]+)"
    r"\.(?:gate_proj|up_proj|down_proj)\Z"
)
PUBLIC_STATIC_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--k2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def is_mtp_expert(name: str) -> bool:
    return name.startswith("mtp.") and ".mlp.experts." in name


def _copy_tensor(source: SourceTensor, destination: BinaryIO, offset: int) -> None:
    source_fd = os.open(source.source_file, os.O_RDONLY)
    try:
        remaining = source.nbytes
        source_offset = source.source_offset
        destination_fd = destination.fileno()
        while remaining:
            count = min(remaining, COPY_CHUNK_BYTES)
            payload = os.pread(source_fd, count, source_offset)
            if len(payload) != count:
                raise IOError(f"short K2 dSpark read for {source.name}")
            if os.pwrite(destination_fd, payload, offset) != count:
                raise IOError(f"short K2 dSpark write for {source.name}")
            source_offset += count
            offset += count
            remaining -= count
    finally:
        os.close(source_fd)


def write_overlay(path: Path, tensors: tuple[SourceTensor, ...]) -> int:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    payload = canonical(header)
    while (8 + len(payload)) % 8:
        payload += b" "
    data_start = 8 + len(payload)
    with path.open("w+b", buffering=0) as destination:
        destination.write(struct.pack("<Q", len(payload)))
        destination.write(payload)
        destination.truncate(data_start + offset)
        tensor_offset = data_start
        for tensor in tensors:
            _copy_tensor(tensor, destination, tensor_offset)
            tensor_offset += tensor.nbytes
        destination.flush()
        os.fsync(destination.fileno())
    return data_start + offset


def final_draft_shard(
    weight_map: dict[str, Any],
) -> tuple[str, tuple[str, ...]]:
    parsed: list[tuple[int, int, str]] = []
    for raw in set(weight_map.values()):
        if not isinstance(raw, str) or (match := SHARD_RE.fullmatch(raw)) is None:
            raise ValueError(f"target has a nonstandard shard name: {raw!r}")
        parsed.append((int(match.group("index")), int(match.group("count")), raw))
    counts = {count for _, count, _ in parsed}
    if len(counts) != 1 or len(parsed) != next(iter(counts)):
        raise ValueError("target shard numbering is not a complete standard sequence")
    final_index = next(iter(counts))
    final_name = next(
        (name for index, _, name in parsed if index == final_index),
        None,
    )
    if final_name is None:
        raise ValueError("target has no final shard")
    final_names = tuple(
        name for name, shard in weight_map.items() if shard == final_name
    )
    if not final_names or any(not is_mtp_expert(name) for name in final_names):
        raise ValueError("target final shard is not exclusively routed dSpark experts")
    return final_name, final_names


def combine_quantization_config(
    *,
    target: dict[str, Any],
    k2: dict[str, Any],
    hidden_layers: int,
    mtp_layers: int,
    experts: int,
) -> tuple[dict[str, Any], ExpertBitPlan]:
    target_storage = target.get("tensor_storage")
    k2_storage = k2.get("tensor_storage")
    if not isinstance(target_storage, dict) or not isinstance(k2_storage, dict):
        raise ValueError("target and K2 configs require tensor_storage maps")
    if target_storage.keys() != k2_storage.keys():
        raise ValueError("target and K2 tensor_storage module sets differ")
    if float(k2.get("bits", -1)) != 2.0:
        raise ValueError("dSpark source is not uniform K2")

    storage: dict[str, Any] = {}
    tiers: dict[tuple[int, int], set[int]] = {}
    for module in sorted(target_storage):
        match = MODULE_RE.fullmatch(module)
        if match is None:
            raise ValueError(f"unexpected routed module name: {module}")
        logical = int(match.group("base") or match.group("mtp"))
        layer_id = logical if match.group("base") is not None else hidden_layers + logical
        expert_id = int(match.group("expert"))
        source = k2_storage if match.group("mtp") is not None else target_storage
        entry = source[module]
        bits = entry.get("bits_per_weight") if isinstance(entry, dict) else None
        if bits not in {2, 3}:
            raise ValueError(f"module {module} has no integer EXL3 storage tier")
        storage[module] = entry
        tiers.setdefault((layer_id, expert_id), set()).add(int(bits))

    total_layers = hidden_layers + mtp_layers
    k3_by_layer: list[tuple[int, ...]] = []
    for layer_id in range(total_layers):
        promoted: list[int] = []
        for expert_id in range(experts):
            observed = tiers.get((layer_id, expert_id))
            if observed not in ({2}, {3}):
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} has inconsistent projections: {observed}"
                )
            if observed == {3}:
                promoted.append(expert_id)
        k3_by_layer.append(tuple(promoted))
    if any(k3_by_layer[layer_id] for layer_id in range(hidden_layers, total_layers)):
        raise ValueError("composed dSpark blocks are not uniformly K2")

    promoted_count = sum(map(len, k3_by_layer))
    family_count = total_layers * experts
    target_bpw = Decimal(2) + Decimal(promoted_count) / Decimal(family_count)
    plan = ExpertBitPlan(
        layer_count=total_layers,
        experts_per_layer=experts,
        target_bpw=str(target_bpw),
        k3_experts_by_layer=tuple(k3_by_layer),
        selection_method="target-base-with-uniform-k2-dspark-diagnostic",
    )
    selection = {
        "schema": "ds4rt.exl3-k2-dspark-control-selection-v1",
        "target_source_bits": target.get("bits"),
        "draft_source_bits": 2.0,
        "plan": plan.to_dict(),
        "plan_sha256": plan.sha256,
    }
    mixed = {
        key: value
        for key, value in target.items()
        if key not in {"bits", "meta", "tensor_storage"}
    }
    mixed["bits"] = plan.realized_bpw
    mixed["tensor_storage"] = storage
    mixed["meta"] = {
        "fallback": None,
        MIX_PLAN_META_KEY: {
            "schema": "ds4rt.exl3-mixed-k2-k3-v1",
            "recipe": MIX_RECIPE,
            "selection_sha256": hashlib.sha256(canonical(selection)).hexdigest(),
            "selection": selection,
        },
    }
    return mixed, plan


def compose(target: Path, k2: Path, output: Path) -> dict[str, Any]:
    target = target.expanduser().resolve(strict=True)
    k2 = k2.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")

    target_index_json = read_object(target / "model.safetensors.index.json")
    k2_index_json = read_object(k2 / "model.safetensors.index.json")
    target_map = target_index_json.get("weight_map")
    k2_map = k2_index_json.get("weight_map")
    if not isinstance(target_map, dict) or target_map.keys() != k2_map.keys():
        raise ValueError("target and K2 tensor-name sets differ")
    final_name, final_names = final_draft_shard(target_map)

    target_tensors = read_artifact_index(target)
    k2_tensors = read_artifact_index(k2)
    overlay_names = tuple(sorted(name for name in target_tensors if is_mtp_expert(name)))
    if not overlay_names:
        raise ValueError("artifacts contain no routed dSpark expert tensors")
    if set(final_names) != set(overlay_names):
        raise ValueError(
            "target routed dSpark experts are not isolated in its final shard"
        )
    for name in overlay_names:
        left, right = target_tensors[name], k2_tensors[name]
        if left.dtype != right.dtype or (
            not name.endswith(".trellis") and left.shape != right.shape
        ):
            raise ValueError(f"target/K2 dSpark metadata differs for {name}")

    model_config = read_object(target / "config.json")
    hidden_layers = int(model_config["num_hidden_layers"])
    mtp_layers = len(model_config["dspark_target_layer_ids"])
    experts = int(model_config["n_routed_experts"])
    quant, plan = combine_quantization_config(
        target=read_object(target / "quantize_config.json"),
        k2=read_object(k2 / "quantize_config.json"),
        hidden_layers=hidden_layers,
        mtp_layers=mtp_layers,
        experts=experts,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name in PUBLIC_STATIC_FILES:
            shutil.copy2((target / name).resolve(strict=True), work / name)

        combined_map = dict(target_map)
        for name in overlay_names:
            combined_map[name] = final_name
        referenced = set(combined_map.values())
        for shard in sorted(referenced - {final_name}):
            os.link((target / shard).resolve(strict=True), work / shard)
        overlay_bytes = write_overlay(
            work / final_name,
            tuple(k2_tensors[name] for name in overlay_names),
        )

        tensor_bytes = sum(
            (k2_tensors[name] if name in overlay_names else target_tensors[name]).nbytes
            for name in target_tensors
        )
        index = {
            "metadata": {"total_size": tensor_bytes},
            "weight_map": combined_map,
        }
        (work / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (work / "quantize_config.json").write_text(
            json.dumps(quant, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        declaration = {key: value for key, value in quant.items() if key != "tensor_storage"}
        model_config["quantization_config"] = declaration
        (work / "config.json").write_text(
            json.dumps(model_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(work, output)
    finally:
        if work.exists():
            shutil.rmtree(work)

    return {
        "schema": "ds4rt.exl3-k2-dspark-control-artifact-v1",
        "output": str(output),
        "target": str(target),
        "k2": str(k2),
        "base_layers": hidden_layers,
        "dspark_layers": mtp_layers,
        "experts": experts,
        "realized_bpw": plan.realized_bpw,
        "k3_base_expert_families": plan.k3_expert_families,
        "k2_dspark_expert_families": mtp_layers * experts,
        "overlay_tensors": len(overlay_names),
        "overlay_bytes": overlay_bytes,
        "tensor_bytes": tensor_bytes,
        "shards": len(referenced),
        "plan_sha256": plan.sha256,
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            compose(args.target, args.k2, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
