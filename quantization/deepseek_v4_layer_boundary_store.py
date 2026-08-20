#!/usr/bin/env python3
"""Crash-consistent rolling decoder-layer boundaries for DeepSeek V4 EXL3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import xxhash
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_file

from gptqmodel.looper.input_cache import TensorLifetimeDiagnostic
from gptqmodel.utils.exl3_projection_checkpoint import (
    EXL3ProjectionCheckpointStore,
)
from gptqmodel.utils.exl3_remote import validate_exl3_hessian_metrics


BOUNDARY_SCHEMA = "ds4rt.deepseek-v4-layer-boundary"
BOUNDARY_SCHEMA_VERSION = 2
BOUNDARY_CONTRACT = f"{BOUNDARY_SCHEMA}-v{BOUNDARY_SCHEMA_VERSION}"
PAYLOAD_HASH_ALGORITHM = "xxh3-128"
MANIFEST_FILENAME = "manifest.json"
_COMMITTED_DIRECTORY = re.compile(
    r"layer-(?P<layer>[0-9]{6})-(?P<digest>[0-9a-f]{16})\Z"
)
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_PROJECTION_MODULE = re.compile(
    r"model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)\Z"
)
_MTP_PROJECTION_MODULE = re.compile(
    r"mtp\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)\Z"
)
_LEDGER_PROJECTION = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


class LayerBoundaryError(RuntimeError):
    """A rolling activation boundary is incomplete or inconsistent."""


class LayerBoundaryStop(RuntimeError):
    """Execution stopped intentionally after a durable layer commit."""

    def __init__(self, layer_index: int) -> None:
        super().__init__(f"stopped after durable decoder layer {layer_index}")
        self.layer_index = layer_index


@dataclass(frozen=True)
class DeepSeekV4LayerBoundary:
    """Validated inputs for the layer immediately after ``layer_index``."""

    layer_index: int
    layer_name: str
    layer_inputs: Sequence[list[torch.Tensor]]
    projection_entries: tuple[dict[str, str], ...]
    manifest_sha256: str


class _LazyBoundaryActivations(Sequence[list[torch.Tensor]]):
    """Replay a durable boundary one mapped activation batch at a time."""

    def __init__(
        self,
        directory: Path,
        records: list[dict[str, Any]],
        *,
        hidden_size: int,
        hc_mult: int,
    ) -> None:
        self._directory = directory
        self._records = records
        self._hidden_size = hidden_size
        self._hc_mult = hc_mult
        self._lifetime = TensorLifetimeDiagnostic([])

    def __len__(self) -> int:
        return len(self._records)

    @property
    def row_counts(self) -> list[int]:
        return [int(record["tensor"]["shape"][0]) for record in self._records]

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record = self._records[index]
        path = self._directory / record["file"]
        with safe_open(path, framework="pt", device="cpu") as source:
            if set(source.keys()) != {"hidden"}:
                raise LayerBoundaryError(
                    f"activation shard {index} tensor set changed after validation"
                )
            hidden = source.get_tensor("hidden")
        if (
            _tensor_spec(hidden) != record["tensor"]
            or hidden.dtype != torch.bfloat16
            or hidden.ndim != 4
            or tuple(hidden.shape[2:]) != (self._hc_mult, self._hidden_size)
        ):
            raise LayerBoundaryError(
                f"activation shard {index} geometry changed after validation"
            )
        self._lifetime.observe(hidden)
        return [hidden]

    def lifetime_diagnostic(self) -> dict[str, Any]:
        return self._lifetime.lifetime_diagnostic()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def xxh3_128_file(path: Path) -> str:
    digest = xxhash.xxh3_128()
    with path.open("rb") as source:
        while block := source.read(32 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(canonical_json_bytes(value) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise LayerBoundaryError("activation metadata contains a non-tensor")
    host = tensor.detach().to(device="cpu").contiguous()
    byte_view = host.reshape(-1).view(torch.uint8).numpy()
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "bytes": host.numel() * host.element_size(),
        "sha256": hashlib.sha256(memoryview(byte_view)).hexdigest(),
    }


def _tensor_spec(tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise LayerBoundaryError("activation boundary contains a non-tensor")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def _metadata_value_identity(value: Any, path: str) -> Any:
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", **_tensor_identity(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LayerBoundaryError(f"non-finite metadata at {path}")
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise LayerBoundaryError(f"non-string metadata key at {path}")
        return {
            key: _metadata_value_identity(item, f"{path}.{key}")
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [
            _metadata_value_identity(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise LayerBoundaryError(
        f"unsupported activation metadata at {path}: {type(value).__name__}"
    )


def replay_metadata_identity(
    *,
    layer_input_kwargs: list[dict[str, Any]],
    position_ids: list[torch.Tensor | None],
    attention_masks: list[torch.Tensor | None],
) -> dict[str, Any]:
    body = {
        "layer_input_kwargs": _metadata_value_identity(
            layer_input_kwargs, "layer_input_kwargs"
        ),
        "position_ids": _metadata_value_identity(position_ids, "position_ids"),
        "attention_masks": _metadata_value_identity(
            attention_masks, "attention_masks"
        ),
    }
    return {**body, "sha256": sha256_bytes(canonical_json_bytes(body))}


def _read_error_journal(path: Path) -> set[str]:
    if not path.is_file() or path.is_symlink():
        raise LayerBoundaryError(f"EXL3 error journal is unavailable: {path}")
    records: set[str] = set()
    with path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            if not line.endswith(b"\n"):
                raise LayerBoundaryError(
                    "EXL3 error journal ends with a partial record"
                )
            try:
                bound = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise LayerBoundaryError(
                    f"EXL3 error journal line {line_number} is invalid"
                ) from error
            if not isinstance(bound, dict):
                raise LayerBoundaryError(
                    f"EXL3 error journal line {line_number} is not an object"
                )
            digest = bound.get("record_sha256")
            record = {
                key: value for key, value in bound.items() if key != "record_sha256"
            }
            if (
                not isinstance(digest, str)
                or sha256_bytes(canonical_json_bytes(record)) != digest
                or record.get("record_kind") != "projection"
            ):
                raise LayerBoundaryError(
                    f"EXL3 error journal line {line_number} failed validation"
                )
            metrics = record.get("quantizer_metrics")
            sample_count = record.get("sample_count")
            provenance = record.get("provenance")
            family_join = (
                provenance.get("family_join")
                if isinstance(provenance, dict)
                else None
            )
            quantizer_numerics = (
                family_join.get("quantizer_numerics")
                if isinstance(family_join, dict)
                else None
            )
            sigma_reg = (
                quantizer_numerics.get("sigma_reg")
                if isinstance(quantizer_numerics, dict)
                else None
            )
            if (
                not isinstance(metrics, dict)
                or isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count <= 0
                or isinstance(sigma_reg, bool)
                or not isinstance(sigma_reg, (int, float))
                or not math.isfinite(float(sigma_reg))
                or float(sigma_reg) <= 0.0
            ):
                raise LayerBoundaryError(
                    f"EXL3 error journal line {line_number} has incomplete "
                    "numerical evidence"
                )
            try:
                validate_exl3_hessian_metrics(
                    metrics,
                    sample_count=sample_count,
                    sigma_reg=float(sigma_reg),
                )
            except RuntimeError as error:
                raise LayerBoundaryError(
                    f"EXL3 error journal line {line_number} has invalid "
                    "numerical evidence"
                ) from error
            records.add(digest)
    return records


class DeepSeekV4LayerBoundaryStore:
    """Retain only the latest complete quantized decoder output boundary."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        plan_sha256: str,
        family_join: dict[str, Any],
        projection_checkpoint_root: str | os.PathLike[str],
        error_journal_path: str | os.PathLike[str],
        hidden_size: int,
        hc_mult: int,
        routed_experts: int,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.plan_sha256 = str(plan_sha256)
        self.family_join = family_join
        self.projection_store = EXL3ProjectionCheckpointStore(
            projection_checkpoint_root
        )
        self.error_journal_path = Path(error_journal_path).expanduser().resolve()
        self.hidden_size = int(hidden_size)
        self.hc_mult = int(hc_mult)
        self.routed_experts = int(routed_experts)
        source = family_join.get("source") if isinstance(family_join, dict) else None
        geometry = source.get("geometry") if isinstance(source, dict) else None
        dspark_target_layer_ids = (
            geometry.get("dspark_target_layer_ids")
            if isinstance(geometry, dict)
            else None
        )
        self.dspark_block_count = (
            len(dspark_target_layer_ids)
            if isinstance(dspark_target_layer_ids, list)
            and dspark_target_layer_ids
            and all(
                not isinstance(layer_id, bool)
                and isinstance(layer_id, int)
                and layer_id >= 0
                for layer_id in dspark_target_layer_ids
            )
            else 0
        )
        if (
            re.fullmatch(r"[0-9a-f]{64}", self.plan_sha256) is None
            or self.hidden_size <= 0
            or self.hc_mult <= 0
            or self.routed_experts <= 0
        ):
            raise LayerBoundaryError("invalid layer-boundary construction contract")

    @property
    def expected_projection_count(self) -> int:
        return self.routed_experts * len(_PROJECTIONS)

    def _expected_modules(self, layer_index: int) -> set[str]:
        return {
            f"model.layers.{layer_index}.mlp.experts.{expert}.{projection}"
            for expert in range(self.routed_experts)
            for projection in _PROJECTIONS
        }

    def _validate_projection_entries(
        self,
        layer_index: int,
        projection_entries: Iterable[dict[str, str]],
    ) -> tuple[dict[str, str], ...]:
        entries = tuple(
            sorted(
                (dict(entry) for entry in projection_entries),
                key=lambda entry: entry.get("module", ""),
            )
        )
        modules = [entry.get("module") for entry in entries]
        if (
            len(entries) != self.expected_projection_count
            or len(set(modules)) != len(modules)
            or set(modules) != self._expected_modules(layer_index)
        ):
            raise LayerBoundaryError(
                f"layer {layer_index} has incomplete EXL3 projection coverage"
            )

        journal = _read_error_journal(self.error_journal_path)
        for entry in entries:
            module_name = entry.get("module")
            request_sha256 = entry.get("request_sha256")
            record_sha256 = entry.get("record_sha256")
            if any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (request_sha256, record_sha256)
            ):
                raise LayerBoundaryError(
                    f"layer {layer_index} has an invalid projection identity"
                )
            loaded = self.projection_store.load_committed(request_sha256)
            if loaded is None:
                raise LayerBoundaryError(
                    f"projection checkpoint is absent for {module_name}"
                )
            request, _tensors, result = loaded
            ledger_record = result.get("ledger_record")
            provenance = (
                ledger_record.get("provenance")
                if isinstance(ledger_record, dict)
                else None
            )
            if (
                request.get("module") != module_name
                or request.get("processor_layer_index") != layer_index
                or request.get("family_join") != self.family_join
                or not isinstance(ledger_record, dict)
                or ledger_record.get("module") != module_name
                or ledger_record.get("processor_layer_index") != layer_index
                or not isinstance(provenance, dict)
                or provenance.get("family_join") != self.family_join
                or record_sha256 not in journal
                or sha256_bytes(canonical_json_bytes(ledger_record))
                != record_sha256
            ):
                raise LayerBoundaryError(
                    f"projection evidence differs for {module_name}"
                )
        return entries

    def discover_completed_projection_layers(
        self,
    ) -> dict[int, tuple[dict[str, str], ...]]:
        """Discover the contiguous fully committed decoder-layer prefix.

        Manifest discovery avoids rebuilding Hessians.  It authenticates all
        JSON evidence and journal membership now; the packed tensor bytes are
        hashed and decoded later by ``restore_completed_layer_checkpoints``
        immediately before a layer is replayed.
        """

        journal = _read_error_journal(self.error_journal_path)
        grouped: dict[int, dict[str, dict[str, str]]] = {}
        mtp_entries: dict[str, dict[str, str]] = {}
        try:
            inspected = self.projection_store.inspect_committed_manifests()
        except ValueError as error:
            raise LayerBoundaryError(
                "cannot discover committed EXL3 projection checkpoints"
            ) from error
        for request, result in inspected:
            module_name = request.get("module")
            match = (
                _PROJECTION_MODULE.fullmatch(module_name)
                if isinstance(module_name, str)
                else None
            )
            mtp_match = (
                _MTP_PROJECTION_MODULE.fullmatch(module_name)
                if isinstance(module_name, str)
                else None
            )
            if match is None and mtp_match is not None:
                logical_layer = int(mtp_match.group("layer"))
                expert_index = int(mtp_match.group("expert"))
                projection = mtp_match.group("projection")
                ledger_record = result.get("ledger_record")
                provenance = (
                    ledger_record.get("provenance")
                    if isinstance(ledger_record, dict)
                    else None
                )
                route_evidence = request.get("route_evidence")
                ledger_route_evidence = (
                    ledger_record.get("route_evidence")
                    if isinstance(ledger_record, dict)
                    else None
                )
                record_sha256 = (
                    sha256_bytes(canonical_json_bytes(ledger_record))
                    if isinstance(ledger_record, dict)
                    else None
                )
                request_sha256 = request.get("request_sha256")
                if (
                    not 0 <= logical_layer < self.dspark_block_count
                    or not 0 <= expert_index < self.routed_experts
                    or request.get("processor_layer_index") != logical_layer
                    or request.get("family_join") != self.family_join
                    or not isinstance(request_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None
                    or not isinstance(ledger_record, dict)
                    or ledger_record.get("record_kind") != "projection"
                    or ledger_record.get("module") != module_name
                    or ledger_record.get("processor_layer_index") != logical_layer
                    or ledger_record.get("block_namespace") != "mtp"
                    or ledger_record.get("logical_layer") != logical_layer
                    or ledger_record.get("expert") != expert_index
                    or ledger_record.get("projection")
                    != _LEDGER_PROJECTION[projection]
                    or not isinstance(provenance, dict)
                    or provenance.get("family_join") != self.family_join
                    or not isinstance(route_evidence, dict)
                    or ledger_route_evidence != route_evidence
                    or route_evidence.get("schema")
                    != "ds4rt.exl3-natural-route"
                    or route_evidence.get("block_namespace") != "mtp"
                    or route_evidence.get("logical_layer") != logical_layer
                    or route_evidence.get("expert") != expert_index
                    or record_sha256 not in journal
                ):
                    raise LayerBoundaryError(
                        f"dSpark projection discovery evidence differs for {module_name}"
                    )
                entry = {
                    "module": module_name,
                    "request_sha256": request_sha256,
                    "record_sha256": record_sha256,
                }
                previous = mtp_entries.setdefault(module_name, entry)
                if previous != entry:
                    raise LayerBoundaryError(
                        f"dSpark projection discovery collision for {module_name}"
                    )
                # Decoder-layer recovery deliberately scopes its contiguous
                # prefix to ``model.layers``.  Valid later MTP checkpoints are
                # retained for the MTP processor, not interpreted as decoder
                # catch-up layers.
                continue
            if match is None:
                raise LayerBoundaryError(
                    "projection checkpoint contains an unexpected module"
                )
            layer_index = int(match.group("layer"))
            expert_index = int(match.group("expert"))
            ledger_record = result.get("ledger_record")
            provenance = (
                ledger_record.get("provenance")
                if isinstance(ledger_record, dict)
                else None
            )
            record_sha256 = (
                sha256_bytes(canonical_json_bytes(ledger_record))
                if isinstance(ledger_record, dict)
                else None
            )
            request_sha256 = request.get("request_sha256")
            if (
                not 0 <= expert_index < self.routed_experts
                or request.get("processor_layer_index") != layer_index
                or request.get("family_join") != self.family_join
                or not isinstance(request_sha256, str)
                or not isinstance(ledger_record, dict)
                or ledger_record.get("module") != module_name
                or ledger_record.get("processor_layer_index") != layer_index
                or not isinstance(provenance, dict)
                or provenance.get("family_join") != self.family_join
                or record_sha256 not in journal
            ):
                raise LayerBoundaryError(
                    f"projection discovery evidence differs for {module_name}"
                )
            entry = {
                "module": module_name,
                "request_sha256": request_sha256,
                "record_sha256": record_sha256,
            }
            previous = grouped.setdefault(layer_index, {}).setdefault(
                module_name, entry
            )
            if previous != entry:
                raise LayerBoundaryError(
                    f"projection discovery collision for {module_name}"
                )

        complete: dict[int, tuple[dict[str, str], ...]] = {}
        if not grouped:
            return complete
        maximum_layer = max(grouped)
        first_incomplete: int | None = None
        for layer_index in range(maximum_layer + 1):
            entries = grouped.get(layer_index, {})
            if set(entries) != self._expected_modules(layer_index):
                first_incomplete = layer_index
                break
            complete[layer_index] = tuple(
                entries[module_name] for module_name in sorted(entries)
            )
        if first_incomplete is not None and any(
            layer_index > first_incomplete for layer_index in grouped
        ):
            raise LayerBoundaryError(
                "projection checkpoints are not a contiguous decoder-layer prefix"
            )
        return complete

    def _validate_completed_projection_index(
        self,
        layer_index: int,
        projection_entries: Iterable[dict[str, str]],
    ) -> tuple[dict[str, str], ...]:
        entries = tuple(
            sorted(
                (dict(entry) for entry in projection_entries),
                key=lambda entry: entry.get("module", ""),
            )
        )
        expected_modules = set().union(
            *(self._expected_modules(index) for index in range(layer_index + 1))
        )
        modules = [entry.get("module") for entry in entries]
        if (
            len(entries) != self.expected_projection_count * (layer_index + 1)
            or len(set(modules)) != len(modules)
            or set(modules) != expected_modules
        ):
            raise LayerBoundaryError(
                f"layer {layer_index} has an incomplete cumulative projection index"
            )
        for entry in entries:
            for field in ("request_sha256", "record_sha256"):
                value = entry.get(field)
                if (
                    not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-f]{64}", value) is None
                ):
                    raise LayerBoundaryError(
                        "cumulative projection index has an invalid identity"
                    )
        return entries

    def _committed_directories(self) -> list[tuple[int, Path]]:
        if not self.root.exists():
            return []
        if not self.root.is_dir() or self.root.is_symlink():
            raise LayerBoundaryError("layer-boundary root is not a regular directory")
        committed: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            match = _COMMITTED_DIRECTORY.fullmatch(path.name)
            if match is None:
                if path.name.startswith(".layer-") and path.name.endswith(".tmp"):
                    if not path.is_dir() or path.is_symlink():
                        raise LayerBoundaryError(
                            "incomplete layer-boundary entry is unsafe"
                        )
                    continue
                raise LayerBoundaryError(
                    f"unexpected entry in layer-boundary root: {path.name}"
                )
            if not path.is_dir() or path.is_symlink():
                raise LayerBoundaryError("committed layer boundary is not a directory")
            committed.append((int(match.group("layer")), path))
        return sorted(committed)

    def _prune_incomplete_directories(self) -> None:
        if not self.root.exists():
            return
        changed = False
        for path in self.root.iterdir():
            if path.name.startswith(".layer-") and path.name.endswith(".tmp"):
                if not path.is_dir() or path.is_symlink():
                    raise LayerBoundaryError(
                        "incomplete layer-boundary entry is unsafe"
                    )
                shutil.rmtree(path)
                changed = True
        if changed:
            _fsync_directory(self.root)

    def _read_manifest(self, directory: Path) -> dict[str, Any]:
        path = directory / MANIFEST_FILENAME
        if not path.is_file() or path.is_symlink():
            raise LayerBoundaryError("layer-boundary manifest is unavailable")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LayerBoundaryError("cannot read layer-boundary manifest") from error
        if not isinstance(manifest, dict):
            raise LayerBoundaryError("layer-boundary manifest is not an object")
        digest = manifest.get("manifest_sha256")
        body = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        directory_match = _COMMITTED_DIRECTORY.fullmatch(directory.name)
        if (
            manifest.get("schema") != BOUNDARY_SCHEMA
            or manifest.get("schema_version") != BOUNDARY_SCHEMA_VERSION
            or manifest.get("payload_hash_algorithm") != PAYLOAD_HASH_ALGORITHM
            or manifest.get("plan_sha256") != self.plan_sha256
            or manifest.get("hidden_size") != self.hidden_size
            or manifest.get("hc_mult") != self.hc_mult
            or not isinstance(digest, str)
            or sha256_bytes(canonical_json_bytes(body)) != digest
            or directory_match is None
            or directory_match.group("digest") != digest[:16]
            or int(directory_match.group("layer")) != manifest.get("layer_index")
        ):
            raise LayerBoundaryError("layer-boundary manifest failed validation")
        return manifest

    def load_latest(
        self,
        *,
        layer_input_kwargs: list[dict[str, Any]],
        position_ids: list[torch.Tensor | None],
        attention_masks: list[torch.Tensor | None],
    ) -> DeepSeekV4LayerBoundary | None:
        self._prune_incomplete_directories()
        committed = self._committed_directories()
        if not committed:
            return None
        layer_index, directory = committed[-1]
        manifest = self._read_manifest(directory)
        current_metadata = replay_metadata_identity(
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
        )
        if manifest.get("replay_metadata") != current_metadata:
            raise LayerBoundaryError(
                "current calibration replay metadata differs from the boundary"
            )
        self._validate_projection_entries(
            layer_index, manifest.get("projection_entries", ())
        )
        completed_entries = self._validate_completed_projection_index(
            layer_index, manifest.get("completed_projection_entries", ())
        )
        shards = manifest.get("activation_shards")
        if not isinstance(shards, list) or not shards:
            raise LayerBoundaryError("layer boundary contains no activation shards")
        actual_files: set[str] = set()
        for path in directory.rglob("*"):
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                raise LayerBoundaryError(
                    f"layer boundary contains a symbolic link: {relative}"
                )
            if path.is_dir():
                if relative != "activations":
                    raise LayerBoundaryError(
                        f"layer boundary contains an unexpected directory: {relative}"
                    )
                continue
            if not path.is_file():
                raise LayerBoundaryError(
                    f"layer boundary contains an unsupported entry: {relative}"
                )
            actual_files.add(relative)
        expected_files = {MANIFEST_FILENAME} | {
            str(record.get("file")) for record in shards
        }
        if len(expected_files) != len(shards) + 1 or actual_files != expected_files:
            raise LayerBoundaryError("layer-boundary file set is inconsistent")

        activation_records: list[dict[str, Any]] = []
        total_bytes = 0
        for batch_index, record in enumerate(shards):
            relative = record.get("file") if isinstance(record, dict) else None
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise LayerBoundaryError("unsafe layer-boundary activation path")
            path = directory / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != record.get("bytes")
                or xxh3_128_file(path) != record.get("xxh3_128")
            ):
                raise LayerBoundaryError(
                    f"activation shard {batch_index} failed content validation"
                )
            with safe_open(path, framework="pt", device="cpu") as source:
                keys = set(source.keys())
                hidden = source.get_tensor("hidden") if keys == {"hidden"} else None
            if (
                keys != {"hidden"}
                or not isinstance(hidden, torch.Tensor)
                or _tensor_spec(hidden) != record.get("tensor")
                or hidden.dtype != torch.bfloat16
                or hidden.ndim != 4
                or tuple(hidden.shape[2:]) != (self.hc_mult, self.hidden_size)
            ):
                raise LayerBoundaryError(
                    f"activation shard {batch_index} has invalid tensor geometry"
                )
            total_bytes += hidden.numel() * hidden.element_size()
            activation_records.append(dict(record))
            del hidden
        if (
            manifest.get("activation_batches") != len(activation_records)
            or manifest.get("activation_bytes") != total_bytes
        ):
            raise LayerBoundaryError("layer-boundary activation totals differ")
        layer_inputs = _LazyBoundaryActivations(
            directory,
            activation_records,
            hidden_size=self.hidden_size,
            hc_mult=self.hc_mult,
        )
        return DeepSeekV4LayerBoundary(
            layer_index=layer_index,
            layer_name=str(manifest.get("layer_name", "")),
            layer_inputs=layer_inputs,
            projection_entries=completed_entries,
            manifest_sha256=manifest["manifest_sha256"],
        )

    def commit(
        self,
        *,
        layer_index: int,
        layer_name: str,
        layer_outputs: list[list[torch.Tensor]],
        layer_input_kwargs: list[dict[str, Any]],
        position_ids: list[torch.Tensor | None],
        attention_masks: list[torch.Tensor | None],
        projection_entries: Iterable[dict[str, str]],
    ) -> dict[str, Any]:
        if layer_index < 0 or not layer_outputs:
            raise LayerBoundaryError("cannot commit an empty layer boundary")
        entries = self._validate_projection_entries(layer_index, projection_entries)
        metadata = replay_metadata_identity(
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise LayerBoundaryError("layer-boundary root cannot be a symbolic link")
        self._prune_incomplete_directories()
        previous_entries: tuple[dict[str, str], ...] = ()
        committed = self._committed_directories()
        if committed:
            previous_layer, previous_directory = committed[-1]
            previous_manifest = self._read_manifest(previous_directory)
            if previous_layer != layer_index - 1:
                raise LayerBoundaryError(
                    "new layer boundary is not contiguous with the committed boundary"
                )
            previous_entries = self._validate_completed_projection_index(
                previous_layer,
                previous_manifest.get("completed_projection_entries", ()),
            )
        elif layer_index != 0:
            raise LayerBoundaryError(
                "the first committed layer boundary must be decoder layer zero"
            )
        completed_entries = self._validate_completed_projection_index(
            layer_index, (*previous_entries, *entries)
        )
        temporary = Path(
            tempfile.mkdtemp(prefix=f".layer-{layer_index:06d}-", suffix=".tmp", dir=self.root)
        )
        try:
            shard_dir = temporary / "activations"
            shard_dir.mkdir()
            shard_records: list[dict[str, Any]] = []
            activation_bytes = 0
            replay_root = getattr(layer_outputs, "root", None)
            replay_manifest = getattr(layer_outputs, "manifest", None)
            direct_replay = (
                isinstance(replay_root, Path)
                and isinstance(replay_manifest, dict)
            )
            if direct_replay:
                replay_root = replay_root.resolve(strict=True)
                replay_shards = replay_manifest.get("shards")
                if (
                    replay_manifest.get("status") != "complete"
                    or replay_manifest.get("hash_algorithm") != "xxh3-128"
                    or replay_manifest.get("layer_index") != layer_index
                    or replay_manifest.get("batch_count") != len(layer_outputs)
                    or replay_manifest.get("shard_batches") != 1
                    or not isinstance(replay_shards, list)
                    or len(replay_shards) != len(layer_outputs)
                    or replay_root.stat().st_dev != temporary.stat().st_dev
                ):
                    raise LayerBoundaryError(
                        "post-quant replay cannot be promoted without copying"
                    )
                for batch_index, replay_shard in enumerate(replay_shards):
                    shapes = (
                        replay_shard.get("shapes")
                        if isinstance(replay_shard, dict)
                        else None
                    )
                    replay_name = replay_shard.get("path", "")
                    shape = shapes[0] if isinstance(shapes, list) and len(shapes) == 1 else None
                    if (
                        replay_shard.get("start") != batch_index
                        or not isinstance(replay_name, str)
                        or Path(replay_name).name != replay_name
                        or not isinstance(shape, list)
                        or len(shape) != 4
                        or tuple(shape[2:]) != (self.hc_mult, self.hidden_size)
                        or replay_shard.get("dtype") != str(torch.bfloat16)
                        or not isinstance(replay_shard.get("bytes"), int)
                        or not isinstance(replay_shard.get("xxh3_128"), str)
                    ):
                        raise LayerBoundaryError(
                            f"post-quant replay batch {batch_index} has invalid geometry"
                        )
                    source_path = replay_root / replay_name
                    if not source_path.is_file() or source_path.is_symlink():
                        raise LayerBoundaryError(
                            f"post-quant replay batch {batch_index} is unavailable"
                        )
                    relative = (
                        Path("activations")
                        / f"batch-{batch_index:06d}.safetensors"
                    )
                    path = temporary / relative
                    os.link(source_path, path)
                    tensor_bytes = math.prod(shape) * 2
                    activation_bytes += tensor_bytes
                    shard_records.append(
                        {
                            "file": relative.as_posix(),
                            "bytes": replay_shard["bytes"],
                            "xxh3_128": replay_shard["xxh3_128"],
                            "tensor": {
                                "shape": shape,
                                "dtype": str(torch.bfloat16),
                                "bytes": tensor_bytes,
                            },
                        }
                    )
            else:
                for batch_index, outputs in enumerate(layer_outputs):
                    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
                        raise LayerBoundaryError(
                            f"layer output batch {batch_index} is not one primary tensor"
                        )
                    hidden = outputs[0]
                    if (
                        not isinstance(hidden, torch.Tensor)
                        or hidden.dtype != torch.bfloat16
                        or hidden.ndim != 4
                        or tuple(hidden.shape[2:]) != (self.hc_mult, self.hidden_size)
                    ):
                        raise LayerBoundaryError(
                            f"layer output batch {batch_index} has invalid BF16 geometry"
                        )
                    host = hidden.detach().to(device="cpu").contiguous()
                    relative = Path("activations") / f"batch-{batch_index:06d}.safetensors"
                    path = temporary / relative
                    save_safetensors_file({"hidden": host}, path)
                    descriptor = os.open(path, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    tensor_bytes = host.numel() * host.element_size()
                    activation_bytes += tensor_bytes
                    shard_records.append(
                        {
                            "file": relative.as_posix(),
                            "bytes": path.stat().st_size,
                            "xxh3_128": xxh3_128_file(path),
                            "tensor": _tensor_spec(host),
                        }
                    )
            _fsync_directory(shard_dir)
            body = {
                "schema": BOUNDARY_SCHEMA,
                "schema_version": BOUNDARY_SCHEMA_VERSION,
                "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
                "plan_sha256": self.plan_sha256,
                "layer_index": int(layer_index),
                "layer_name": str(layer_name),
                "hidden_size": self.hidden_size,
                "hc_mult": self.hc_mult,
                "activation_batches": len(shard_records),
                "activation_bytes": activation_bytes,
                "activation_shards": shard_records,
                "replay_metadata": metadata,
                "projection_entries": list(entries),
                "completed_projection_entries": list(completed_entries),
            }
            manifest = {
                **body,
                "manifest_sha256": sha256_bytes(canonical_json_bytes(body)),
            }
            _atomic_json(temporary / MANIFEST_FILENAME, manifest)
            _fsync_directory(temporary)
            destination = self.root / (
                f"layer-{layer_index:06d}-{manifest['manifest_sha256'][:16]}"
            )
            if destination.exists():
                existing = self._read_manifest(destination)
                if existing != manifest:
                    raise LayerBoundaryError("layer-boundary commit collision")
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, destination)
            _fsync_directory(self.root)

            # Promotion above is the durability barrier. Only after it succeeds
            # may an older complete generation be discarded.
            for _old_layer, old_directory in self._committed_directories():
                if old_directory != destination:
                    shutil.rmtree(old_directory)
            _fsync_directory(self.root)
            return manifest
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise


class DeepSeekV4LayerBoundaryController:
    """Bridge the durable store to GPTQModel's layer-loop checkpoint hooks."""

    def __init__(
        self,
        store: DeepSeekV4LayerBoundaryStore,
        *,
        defer_publication_materialization: bool = False,
        stop_after_layer: int | None = None,
    ) -> None:
        if not isinstance(store, DeepSeekV4LayerBoundaryStore):
            raise TypeError("layer-boundary controller requires its durable store")
        if stop_after_layer is not None and (
            isinstance(stop_after_layer, bool)
            or not isinstance(stop_after_layer, int)
            or stop_after_layer < 0
        ):
            raise ValueError("stop-after layer must be a non-negative integer")
        self.store = store
        self._catchup_layers: dict[int, tuple[dict[str, str], ...]] = {}
        self._prepared_catchup_layers: set[int] = set()
        self._deferred_prefix_layers: dict[
            int, tuple[dict[str, str], ...]
        ] = {}
        self.defer_publication_materialization = bool(
            defer_publication_materialization
        )
        self.stop_after_layer = stop_after_layer
        self.stopped_after_layer: int | None = None
        self._deferred_processor: Any | None = None

    @staticmethod
    def _processor(processors: list[Any]) -> Any:
        candidates = [
            processor
            for processor in processors
            if callable(
                getattr(processor, "completed_layer_checkpoint_entries", None)
            )
            and callable(
                getattr(processor, "restore_completed_layer_checkpoints", None)
            )
        ]
        if len(candidates) != 1:
            raise LayerBoundaryError(
                "DeepSeek V4 layer-boundary resume requires exactly one "
                "boundary-capable EXL3 processor"
            )
        return candidates[0]

    @staticmethod
    def _prune_active_source(model: Any, layer_index: int) -> None:
        turtle = getattr(model, "turtle_model", None)
        prune = getattr(turtle, "prune_active_source_through", None)
        if callable(prune):
            prune(layer_index)

    @staticmethod
    def _defer_completed_native_layer(
        model: Any,
        *,
        layer_index: int,
        layer_name: str,
    ) -> None:
        """Return unchanged decoder state to META after its durable boundary.

        Routed projections have already been replaced by metadata-only EXL3
        shells. The remaining attention, router, shared-expert, and norm
        tensors are unchanged source state and would otherwise accumulate on
        CPU by one decoder block per iteration. The checkpoint-backed turtle
        rematerializes those tensors for final publication.
        """

        expected_name = f"model.layers.{layer_index}"
        if layer_name != expected_name:
            raise LayerBoundaryError(
                "completed decoder layer has a noncanonical module name"
            )
        target_model = getattr(model, "model", None)
        turtle = getattr(model, "turtle_model", None)
        if target_model is None or not callable(
            getattr(target_model, "get_submodule", None)
        ):
            raise LayerBoundaryError(
                "completed decoder layer has no materializable model root"
            )
        if not callable(getattr(turtle, "materialize_submodule", None)) or not callable(
            getattr(turtle, "sync_all_meta", None)
        ):
            raise LayerBoundaryError(
                "completed decoder layer has no checkpoint-backed source"
            )
        try:
            layer = target_model.get_submodule(layer_name)
        except (AttributeError, KeyError) as error:
            raise LayerBoundaryError(
                f"completed decoder layer is absent: {layer_name}"
            ) from error
        if not isinstance(layer, torch.nn.Module):
            raise LayerBoundaryError(
                f"completed decoder layer is not a module: {layer_name}"
            )

        layer.to_empty(device=torch.device("meta"), recurse=True)
        materialized = [
            name
            for name, tensor in (
                *layer.named_parameters(recurse=True),
                *layer.named_buffers(recurse=True),
            )
            if tensor.device.type != "meta"
        ]
        if materialized:
            raise LayerBoundaryError(
                "completed decoder layer retained materialized tensors: "
                + ", ".join(sorted(materialized)[:8])
            )

    def restore(self, *, model: Any, processors: list[Any]) -> int:
        processor = self._processor(processors)
        self._deferred_processor = processor
        cache = processor.inputs_cache
        boundary = self.store.load_latest(
            layer_input_kwargs=cache.layer_input_kwargs,
            position_ids=cache.position_ids,
            attention_masks=cache.attention_masks,
        )
        start_layer_index = 0
        if boundary is not None:
            entries_by_layer: dict[int, list[dict[str, str]]] = {}
            for entry in boundary.projection_entries:
                match = re.match(r"model\.layers\.([0-9]+)\.", entry["module"])
                layer_index = int(match.group(1)) if match is not None else -1
                if not 0 <= layer_index <= boundary.layer_index:
                    raise LayerBoundaryError(
                        "completed projection index contains an invalid decoder layer"
                    )
                entries_by_layer.setdefault(layer_index, []).append(entry)
            if set(entries_by_layer) != set(range(boundary.layer_index + 1)):
                raise LayerBoundaryError(
                    "completed projection layers are not contiguous"
                )
            processor.receive_layer_inputs(boundary.layer_inputs)
            start_layer_index = boundary.layer_index + 1
            self._deferred_prefix_layers = {
                layer_index: tuple(entries_by_layer[layer_index])
                for layer_index in range(start_layer_index)
            }
            discard_frontiers = getattr(
                processor, "discard_capture_frontiers_through", None
            )
            if callable(discard_frontiers):
                discard_frontiers(boundary.layer_index)
            self._prune_active_source(model, boundary.layer_index)

        discovered = self.store.discover_completed_projection_layers()
        missing_boundary_layers = set(range(start_layer_index)) - set(discovered)
        if missing_boundary_layers:
            raise LayerBoundaryError(
                "a committed boundary references undiscoverable projection layers"
            )
        for layer_index, expected_entries in self._deferred_prefix_layers.items():
            if discovered.get(layer_index) != expected_entries:
                raise LayerBoundaryError(
                    f"decoder layer {layer_index} checkpoint index differs from boundary"
                )
        self._catchup_layers = {
            layer_index: entries
            for layer_index, entries in discovered.items()
            if layer_index >= start_layer_index
        }
        self._prepared_catchup_layers.clear()
        return start_layer_index

    def materialize_deferred_prefix(
        self,
        *,
        model: Any,
        processors: list[Any] | None = None,
        force: bool = False,
    ) -> None:
        """Install skipped packed layers once, immediately before finalization.

        Activation-boundary recovery executes directly from the next layer's
        saved inputs, so eager reconstruction of every completed packed module
        only makes restart latency and memory scale with prior progress.  The
        raw publication model still requires those modules; defer that cost to
        the one successful run that reaches processor finalization.
        """

        if not self._deferred_prefix_layers:
            return
        if self.defer_publication_materialization and not force:
            return
        processor = (
            self._processor(processors)
            if processors
            else self._deferred_processor
        )
        if processor is None:
            raise LayerBoundaryError(
                "deferred packed-prefix materialization lost its EXL3 processor"
            )
        for layer_index in sorted(self._deferred_prefix_layers):
            processor.restore_completed_layer_checkpoints(
                model=model,
                layer_index=layer_index,
                projection_entries=list(self._deferred_prefix_layers[layer_index]),
            )
            memory_summary = getattr(processor, "log_capture_memory_summary", None)
            if callable(memory_summary):
                memory_summary(
                    f"deferred-prefix-layer-{layer_index}-restored",
                    model=model,
                )
            release_host_memory = getattr(processor, "release_host_memory", None)
            if callable(release_host_memory):
                release_host_memory(
                    f"deferred-prefix-layer-{layer_index}-restored",
                    model=model,
                )
        self._deferred_prefix_layers.clear()

    def is_catchup_layer(self, layer_index: int) -> bool:
        return layer_index in self._catchup_layers

    def prepare_catchup_layer(
        self,
        *,
        model: Any,
        processors: list[Any],
        layer_index: int,
    ) -> Any:
        """Install one fully committed packed layer immediately before replay."""

        processor = self._processor(processors)
        entries = self._catchup_layers.get(layer_index)
        if entries is None:
            raise LayerBoundaryError(
                f"decoder layer {layer_index} is not a catch-up layer"
            )
        if layer_index in self._prepared_catchup_layers:
            raise LayerBoundaryError(
                f"decoder layer {layer_index} catch-up was prepared twice"
            )
        replay_device = getattr(processor.qcfg, "device", None)
        if replay_device is None or torch.device(replay_device).type == "cpu":
            raise LayerBoundaryError(
                "packed catch-up requires an accelerator replay device"
            )
        processor.restore_completed_layer_checkpoints(
            model=model,
            layer_index=layer_index,
            projection_entries=list(entries),
            materialize_device=replay_device,
        )
        self._prepared_catchup_layers.add(layer_index)
        return processor

    def finalize_catchup_layer(
        self,
        *,
        model: Any,
        processor: Any,
        layer_index: int,
    ) -> None:
        if layer_index not in self._prepared_catchup_layers:
            raise LayerBoundaryError(
                f"decoder layer {layer_index} catch-up was not prepared"
            )
        offload = getattr(
            processor, "offload_restored_layer_checkpoints", None
        )
        if not callable(offload):
            raise LayerBoundaryError(
                "boundary-capable EXL3 processor cannot offload restored layers"
            )
        offload(model=model, layer_index=layer_index)

    def commit_layer(
        self,
        *,
        model: Any,
        processor: Any,
        layer_index: int,
        layer_name: str,
    ) -> dict[str, Any]:
        entries = processor.completed_layer_checkpoint_entries(layer_index)
        cache = processor.inputs_cache
        memory_summary = getattr(processor, "log_capture_memory_summary", None)
        if callable(memory_summary):
            memory_summary(
                f"layer-{layer_index}-before-boundary",
                model=model,
            )
        replay_outputs = cache.layer_inputs
        manifest = self.store.commit(
            layer_index=layer_index,
            layer_name=layer_name,
            layer_outputs=cache.layer_inputs,
            layer_input_kwargs=cache.layer_input_kwargs,
            position_ids=cache.position_ids,
            attention_masks=cache.attention_masks,
            projection_entries=entries,
        )
        replay_root = getattr(replay_outputs, "root", None)
        replay_manifest = getattr(replay_outputs, "manifest", None)
        if isinstance(replay_root, Path) and isinstance(replay_manifest, dict):
            boundary = self.store.load_latest(
                layer_input_kwargs=cache.layer_input_kwargs,
                position_ids=cache.position_ids,
                attention_masks=cache.attention_masks,
            )
            if boundary is None or boundary.layer_index != layer_index:
                raise LayerBoundaryError(
                    "committed replay did not promote to the rolling boundary"
                )
            processor.receive_layer_inputs(boundary.layer_inputs)
            if replay_root.is_dir() and not replay_root.is_symlink():
                shutil.rmtree(replay_root)
                _fsync_directory(replay_root.parent)
        if self.defer_publication_materialization:
            existing = self._deferred_prefix_layers.get(layer_index)
            deferred_entries = tuple(entries)
            if existing is not None and existing != deferred_entries:
                raise LayerBoundaryError(
                    "completed layer differs from its deferred publication index"
                )
            defer_layer = getattr(
                processor, "defer_completed_layer_checkpoints", None
            )
            if not callable(defer_layer):
                raise LayerBoundaryError(
                    "boundary-capable EXL3 processor cannot defer a completed layer"
                )
            defer_layer(
                model=model,
                layer_index=layer_index,
                projection_entries=entries,
            )
            self._deferred_prefix_layers[layer_index] = deferred_entries
            self._defer_completed_native_layer(
                model,
                layer_index=layer_index,
                layer_name=layer_name,
            )
        if callable(memory_summary):
            memory_summary(
                f"layer-{layer_index}-after-boundary",
                model=model,
            )
        discard_frontiers = getattr(
            processor, "discard_capture_frontiers_through", None
        )
        if callable(discard_frontiers):
            discard_frontiers(layer_index)
        release_host_memory = getattr(processor, "release_host_memory", None)
        if callable(release_host_memory):
            release_host_memory(
                f"layer-{layer_index}-after-boundary",
                model=model,
            )
        self._prune_active_source(model, layer_index)
        if self.stop_after_layer == layer_index:
            self.stopped_after_layer = layer_index
            raise LayerBoundaryStop(layer_index)
        return manifest


__all__ = [
    "BOUNDARY_CONTRACT",
    "PAYLOAD_HASH_ALGORITHM",
    "DeepSeekV4LayerBoundary",
    "DeepSeekV4LayerBoundaryController",
    "DeepSeekV4LayerBoundaryStore",
    "LayerBoundaryError",
    "LayerBoundaryStop",
    "replay_metadata_identity",
]
