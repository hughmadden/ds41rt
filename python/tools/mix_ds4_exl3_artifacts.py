#!/usr/bin/env python3
"""Stream two canonical uniform EXL3 snapshots into one mixed K2/K3 artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
from typing import Any, BinaryIO

from ds4rt_runtime.exl3_mix import (
    build_mixed_quantization_config,
    mixed_expert_tensor_identity,
)
from ds4rt_runtime.exl3_quantizer import SourceTensor, read_artifact_index
from ds4rt_runtime.exl3_tiers import ExpertBitPlan


MAX_SHARD_BYTES = 8 * 1024**3
COPY_CHUNK_BYTES = 64 * 1024**2
STATIC_ASSETS = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def bound(value: dict[str, Any], field: str) -> dict[str, Any]:
    body = dict(value)
    body[field] = hashlib.sha256(canonical(body)).hexdigest()
    return body


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class OutputLocation:
    file_name: str
    absolute_offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k2", type=Path, required=True)
    parser.add_argument("--k3", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-shard-bytes", type=int, default=MAX_SHARD_BYTES)
    return parser.parse_args()


def _retained_identity(snapshot: Path) -> tuple[Any, ...]:
    report = read_object(snapshot / "ds4rt-exl3-retained-native.json")
    return (
        report.get("retained_tensor_count"),
        report.get("retained_bytes"),
        report.get("aggregate_sha256"),
    )


def _source_artifact_identity(snapshot: Path) -> dict[str, Any]:
    artifact = snapshot / "ds4rt-gptqmodel-artifact.json"
    plan = snapshot / "ds4rt-gptqmodel-plan.json"
    run = snapshot / "ds4rt-gptqmodel-run.json"
    return {
        "artifact_sha256": sha256_file(artifact),
        "plan_sha256": sha256_file(plan),
        "run_sha256": sha256_file(run),
    }


def _choose_tensors(
    *,
    k2: Path,
    k3: Path,
    plan: ExpertBitPlan,
    hidden_layers: int,
) -> tuple[SourceTensor, ...]:
    tiers = {2: read_artifact_index(k2), 3: read_artifact_index(k3)}
    if tiers[2].keys() != tiers[3].keys():
        raise ValueError("K2/K3 artifact tensor-name sets differ")
    chosen: list[SourceTensor] = []
    for name in tiers[2]:
        identity = mixed_expert_tensor_identity(name, hidden_layers=hidden_layers)
        bits = 2 if identity is None else plan.bits_for(identity[0], identity[1])
        tensor = tiers[bits][name]
        other = tiers[3 if bits == 2 else 2][name]
        if tensor.dtype != other.dtype or (
            not name.endswith(".trellis") and tensor.shape != other.shape
        ):
            raise ValueError(f"K2/K3 tensor metadata is incompatible for {name}")
        chosen.append(tensor)
    # Preserve the canonical source ordering: retained tensors first in their
    # K2 artifact order, then generated tensors by module name.
    chosen.sort(
        key=lambda tensor: (
            mixed_expert_tensor_identity(tensor.name, hidden_layers=hidden_layers)
            is not None,
            tensor.name,
        )
    )
    return tuple(chosen)


def _partition(
    tensors: tuple[SourceTensor, ...], max_shard_bytes: int
) -> tuple[tuple[SourceTensor, ...], ...]:
    if max_shard_bytes <= 0:
        raise ValueError("max shard bytes must be positive")
    shards: list[list[SourceTensor]] = [[]]
    size = 0
    for tensor in tensors:
        if size and size + tensor.nbytes > max_shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(tensor)
        size += tensor.nbytes
    return tuple(tuple(shard) for shard in shards)


def _create_shards(
    root: Path, shards: tuple[tuple[SourceTensor, ...], ...]
) -> tuple[dict[str, OutputLocation], dict[str, int]]:
    locations: dict[str, OutputLocation] = {}
    sizes: dict[str, int] = {}
    count = len(shards)
    for shard_id, tensors in enumerate(shards, 1):
        name = f"model-{shard_id:05}-of-{count:05}.safetensors"
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
        path = root / name
        with path.open("wb") as stream:
            stream.write(struct.pack("<Q", len(payload)))
            stream.write(payload)
            stream.truncate(data_start + offset)
        position = data_start
        for tensor in tensors:
            locations[tensor.name] = OutputLocation(name, position)
            position += tensor.nbytes
        sizes[name] = data_start + offset
    return locations, sizes


def _validate_existing_shards(
    root: Path,
    shards: tuple[tuple[SourceTensor, ...], ...],
) -> tuple[dict[str, OutputLocation], dict[str, int]]:
    # Reconstruct headers in a temporary sibling, compare only their immutable
    # prefix and final size, then discard the template.
    template = root / ".layout"
    template.mkdir()
    try:
        locations, sizes = _create_shards(template, shards)
        for name, size in sizes.items():
            actual = root / name
            expected = template / name
            if not actual.is_file() or actual.stat().st_size != size:
                raise ValueError(f"resume shard layout differs: {actual}")
            with expected.open("rb") as left, actual.open("rb") as right:
                raw_length = left.read(8)
                header_length = struct.unpack("<Q", raw_length)[0]
                expected_header = raw_length + left.read(header_length)
                if right.read(len(expected_header)) != expected_header:
                    raise ValueError(f"resume shard header differs: {actual}")
        return locations, sizes
    finally:
        shutil.rmtree(template)


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
                raise IOError(f"short mixed-source read for {source.name}")
            if os.pwrite(destination_fd, payload, offset) != count:
                raise IOError(f"short mixed-artifact write for {source.name}")
            source_offset += count
            offset += count
            remaining -= count
    finally:
        os.close(source_fd)


def _copy_tensors(
    *,
    root: Path,
    tensors: tuple[SourceTensor, ...],
    locations: dict[str, OutputLocation],
    identity: dict[str, Any],
    resume: bool,
) -> None:
    state_path = root / ".ds4rt-exl3-mix-state.json"
    next_tensor = 0
    if resume:
        state = read_object(state_path)
        if state.get("identity") != identity:
            raise ValueError("mixed artifact resume identity changed")
        next_tensor = int(state.get("next_tensor", -1))
        if not 0 <= next_tensor <= len(tensors):
            raise ValueError("mixed artifact resume frontier is invalid")
    handles: dict[str, BinaryIO] = {}
    dirty: set[str] = set()
    try:
        for index in range(next_tensor, len(tensors)):
            tensor = tensors[index]
            location = locations[tensor.name]
            handle = handles.get(location.file_name)
            if handle is None:
                handle = (root / location.file_name).open("r+b", buffering=0)
                handles[location.file_name] = handle
            _copy_tensor(tensor, handle, location.absolute_offset)
            dirty.add(location.file_name)
            if (index + 1) % 256 == 0 or index + 1 == len(tensors):
                for name in sorted(dirty):
                    os.fsync(handles[name].fileno())
                dirty.clear()
                temporary = state_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps({"identity": identity, "next_tensor": index + 1}, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                temporary.replace(state_path)
    finally:
        for handle in handles.values():
            handle.close()


def _global_layer(record: dict[str, Any], hidden_layers: int) -> int:
    logical = int(record["logical_layer"])
    return logical if record["block_namespace"] == "base" else hidden_layers + logical


def _merge_ledgers(
    *, k2: Path, k3: Path, output: Path, plan: ExpertBitPlan, hidden_layers: int
) -> dict[str, Any]:
    sources = [
        (k2 / "ds4rt-exl3-error-ledger.jsonl").open(encoding="utf-8"),
        (k3 / "ds4rt-exl3-error-ledger.jsonl").open(encoding="utf-8"),
    ]
    ledger_path = output / "ds4rt-exl3-error-ledger.jsonl"
    digest = hashlib.sha256()
    projections = families = 0
    try:
        with ledger_path.open("wb") as destination:
            line_number = 0
            while True:
                lines = [source.readline() for source in sources]
                if not any(lines):
                    break
                line_number += 1
                if not all(lines):
                    raise ValueError("K2/K3 ledgers have different record counts")
                records = [json.loads(line) for line in lines]
                identity = tuple(
                    records[0].get(key)
                    for key in ("record_kind", "block_namespace", "logical_layer", "expert")
                )
                if identity != tuple(
                    records[1].get(key)
                    for key in ("record_kind", "block_namespace", "logical_layer", "expert")
                ) or records[0].get("projection") != records[1].get("projection"):
                    raise ValueError(f"K2/K3 ledger ordering differs at line {line_number}")
                layer_id = _global_layer(records[0], hidden_layers)
                expert_id = int(records[0]["expert"])
                bits = plan.bits_for(layer_id, expert_id)
                selected = records[bits - 2]
                if selected.get("bits") != bits:
                    raise ValueError(f"mixed ledger tier mismatch at line {line_number}")
                payload = canonical(selected) + b"\n"
                destination.write(payload)
                digest.update(payload)
                if selected["record_kind"] == "projection":
                    projections += 1
                elif selected["record_kind"] == "expert_family":
                    families += 1
                else:
                    raise ValueError("mixed ledger contains an unknown record kind")
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        for source in sources:
            source.close()
    if projections != plan.total_expert_families * 3 or families != plan.total_expert_families:
        raise ValueError("mixed ledger did not close every expert family")
    manifest = {
        "schema": "ds4rt.exl3-error-ledger",
        "schema_version": 1,
        "ledger": ledger_path.name,
        "ledger_sha256": digest.hexdigest(),
        "projection_records": projections,
        "complete_family_records": families,
        "total_records": projections + families,
    }
    (output / "ds4rt-exl3-error-ledger.manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


def _write_metadata(
    *,
    root: Path,
    k2: Path,
    k3: Path,
    selection: dict[str, Any],
    plan: ExpertBitPlan,
    tensors: tuple[SourceTensor, ...],
    locations: dict[str, OutputLocation],
    shard_sizes: dict[str, int],
) -> None:
    for name in STATIC_ASSETS:
        source = k2 / name
        if source.is_file():
            shutil.copy2(source, root / name)
    k2_config = read_object(k2 / "config.json")
    k2_quant = read_object(k2 / "quantize_config.json")
    k3_quant = read_object(k3 / "quantize_config.json")
    quant = build_mixed_quantization_config(
        k2_quant=k2_quant, k3_quant=k3_quant, plan=plan, selection=selection
    )
    model_config = dict(k2_config)
    model_config["quantization_config"] = quant
    (root / "config.json").write_text(
        json.dumps(model_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "quantize_config.json").write_text(
        json.dumps(quant, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    index = {
        "metadata": {"total_size": sum(tensor.nbytes for tensor in tensors)},
        "weight_map": {name: location.file_name for name, location in locations.items()},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "ds4rt-exl3-mix-selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = _merge_ledgers(
        k2=k2,
        k3=k3,
        output=root,
        plan=plan,
        hidden_layers=int(model_config["num_hidden_layers"]),
    )
    sources = bound(
        {
            "schema": "ds4rt.exl3-mixed-k2-k3-sources-v1",
            "selection_sha256": hashlib.sha256(canonical(selection)).hexdigest(),
            "k2": _source_artifact_identity(k2),
            "k3": _source_artifact_identity(k3),
            "retained_native_identity": list(_retained_identity(k2)),
            "tensor_count": len(tensors),
            "tensor_bytes": index["metadata"]["total_size"],
            "shards": shard_sizes,
            "ledger": ledger,
        },
        "sources_sha256",
    )
    (root / "ds4rt-exl3-mix-sources.json").write_text(
        json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan_record = bound(
        {
            "schema": "ds4rt.exl3-mixed-k2-k3-plan-v1",
            "recipe": "deepseek_v4_exl3_trellis_mixed_k2_k3_v1",
            "selection": selection,
            "sources_sha256": sources["sources_sha256"],
        },
        "plan_sha256",
    )
    (root / "ds4rt-gptqmodel-plan.json").write_text(
        json.dumps(plan_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_record = bound(
        {
            "schema": "ds4rt.exl3-mixed-k2-k3-run-v1",
            "status": "complete",
            "plan_sha256": plan_record["plan_sha256"],
        },
        "run_sha256",
    )
    (root / "ds4rt-gptqmodel-run.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_artifact_manifest(root: Path) -> None:
    excluded = {
        ".ds4rt-exl3-mix-state.json",
        "ds4rt-gptqmodel-artifact.json",
        "ds4rt-gptqmodel-run.json",
    }
    records: dict[str, Any] = {}
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name in excluded:
            continue
        records[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    artifact = bound(
        {
            "schema": "ds4rt.exl3-mixed-k2-k3-artifact-v1",
            "files": records,
            "file_count": len(records),
            "total_bytes": sum(record["bytes"] for record in records.values()),
        },
        "manifest_sha256",
    )
    (root / "ds4rt-gptqmodel-artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    k2 = args.k2.expanduser().resolve(strict=True)
    k3 = args.k3.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    work = output.with_name(f".{output.name}.mix-work")
    selection = read_object(args.selection.expanduser().resolve(strict=True))
    plan = ExpertBitPlan.from_dict(selection["plan"])
    if selection.get("plan_sha256") != plan.sha256:
        raise ValueError("selection plan digest is invalid")
    k2_config = read_object(k2 / "config.json")
    hidden_layers = int(k2_config["num_hidden_layers"])
    if _retained_identity(k2) != _retained_identity(k3):
        raise ValueError("K2/K3 retained-native identities differ")
    tensors = _choose_tensors(
        k2=k2, k3=k3, plan=plan, hidden_layers=hidden_layers
    )
    shards = _partition(tensors, args.max_shard_bytes)
    identity = {
        "schema": "ds4rt.exl3-mixed-k2-k3-copy-v1",
        "selection_sha256": hashlib.sha256(canonical(selection)).hexdigest(),
        "k2": str(k2),
        "k3": str(k3),
        "tensor_count": len(tensors),
        "tensor_bytes": sum(tensor.nbytes for tensor in tensors),
        "max_shard_bytes": args.max_shard_bytes,
    }
    if output.exists():
        raise ValueError(f"mixed output already exists: {output}")
    if args.resume:
        if not work.is_dir():
            raise ValueError(f"mixed resume work directory is missing: {work}")
        locations, shard_sizes = _validate_existing_shards(work, shards)
    else:
        if work.exists():
            raise ValueError(f"mixed work directory already exists: {work}")
        work.mkdir(parents=True)
        locations, shard_sizes = _create_shards(work, shards)
        state = work / ".ds4rt-exl3-mix-state.json"
        state.write_text(
            json.dumps({"identity": identity, "next_tensor": 0}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _copy_tensors(
        root=work,
        tensors=tensors,
        locations=locations,
        identity=identity,
        resume=args.resume,
    )
    _write_metadata(
        root=work,
        k2=k2,
        k3=k3,
        selection=selection,
        plan=plan,
        tensors=tensors,
        locations=locations,
        shard_sizes=shard_sizes,
    )
    _write_artifact_manifest(work)
    (work / ".ds4rt-exl3-mix-state.json").unlink()
    os.replace(work, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "tensors": len(tensors),
                "tensor_bytes": sum(tensor.nbytes for tensor in tensors),
                "shards": len(shards),
                "realized_bpw": plan.realized_bpw,
                "plan_sha256": plan.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
