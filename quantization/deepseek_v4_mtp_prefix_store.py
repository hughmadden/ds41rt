"""Durable quantized-prefix boundary store for DeepSeek-V4 MTP calibration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence


SCHEMA = "ds41rt-deepseek-v4-mtp-prefix-store-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ANCHOR_SELECTION_CONTRACT = "ds41rt-mtp-anchor-stratified-v1"
FIXED_REPLAY_BATCH_CONTRACT = "ds41rt-mtp-flat-fixed-anchor-batches-v1"
SEQUENCE_REPLAY_BATCH_CONTRACT = "ds41rt-mtp-source-sequence-anchor-batches-v1"
_UINT64_MASK = (1 << 64) - 1


class PrefixStoreError(RuntimeError):
    """The target-tap stream or an existing durable record is inconsistent."""


def _splitmix64(value: int) -> int:
    """Return one stable 64-bit mixer result without global RNG state."""

    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _stratified_anchor_indices(total: int, count: int, seed: int) -> array:
    """Choose exactly ``count`` source-ordered anchors across the full stream.

    Every stratum contributes one anchor. This retains broad corpus/language
    coverage, avoids a mutable random-number-generator implementation, and can
    be reproduced from the prefix identity plus the declared integer seed.
    """

    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise PrefixStoreError("eligible anchor count must be positive")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 < count <= total
    ):
        raise PrefixStoreError("selected anchor count must be in [1, total]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PrefixStoreError("anchor selection seed must be an integer")
    if count == total:
        return array("I", range(total))

    result = array("I")
    normalized_seed = seed & _UINT64_MASK
    for stratum in range(count):
        lower = (stratum * total) // count
        upper = ((stratum + 1) * total) // count
        width = upper - lower
        result.append(
            lower + (_splitmix64(normalized_seed ^ stratum) % width)
        )
    return result


@dataclass(frozen=True)
class DeepSeekV4MTPStoredReplayBatch:
    replay_batch: Any
    sources: tuple[dict[str, int], ...]


class DeepSeekV4MTPReplayDataset(Sequence[DeepSeekV4MTPStoredReplayBatch]):
    """Lazy, re-iterable view of selected natural dSpark positions.

    A million-token target corpus produces roughly a million dSpark positions.
    Each position needs a different causal suffix of the same projected target
    sequence, so eagerly collecting tensor replay batches duplicates close to a
    terabyte for Flash. This view retains only four compact integer columns and
    keeps a small LRU of the original per-sequence safetensor files. Tensor
    batches are assembled only when GPTQModel asks for that index.

    The production dSpark path deterministically selects anchors *before* any
    auxiliary block executes, then groups every selected anchor from one source
    sequence into a single outer replay item. The five proposal rows belonging
    to an anchor remain jointly issued inside that item. An optional anchor cap
    only splits unusually large source sequences; it never splits proposal rows.
    """

    def __init__(
        self,
        store: "DeepSeekV4MTPPrefixStore",
        *,
        replay_batch_size: int,
        device: Any = "cpu",
        max_positions: int | None = None,
        file_cache_entries: int = 4,
        anchor_sample_count: int | None = None,
        anchor_sample_seed: int = 0,
        batch_by_source_sequence: bool = False,
        source_sequence_anchor_cap: int | None = None,
    ) -> None:
        import torch
        from safetensors.torch import load_file

        if store.manifest.get("status") != "complete":
            raise PrefixStoreError("prefix store is incomplete")
        if replay_batch_size <= 0:
            raise PrefixStoreError("replay_batch_size must be positive")
        if max_positions is not None and max_positions <= 0:
            raise PrefixStoreError("max_positions must be positive when set")
        if anchor_sample_count is not None and anchor_sample_count <= 0:
            raise PrefixStoreError("anchor_sample_count must be positive when set")
        if max_positions is not None and anchor_sample_count is not None:
            raise PrefixStoreError(
                "max_positions and anchor_sample_count are mutually exclusive"
            )
        if isinstance(anchor_sample_seed, bool) or not isinstance(
            anchor_sample_seed, int
        ):
            raise PrefixStoreError("anchor_sample_seed must be an integer")
        if (
            source_sequence_anchor_cap is not None
            and source_sequence_anchor_cap <= 0
        ):
            raise PrefixStoreError(
                "source_sequence_anchor_cap must be positive when set"
            )
        if source_sequence_anchor_cap is not None and not batch_by_source_sequence:
            raise PrefixStoreError(
                "source_sequence_anchor_cap requires source-sequence batching"
            )
        if file_cache_entries <= 0:
            raise PrefixStoreError("file_cache_entries must be positive")

        self.store = store
        self.replay_batch_size = int(replay_batch_size)
        self.device = torch.device(device)
        self.max_positions = max_positions
        self.anchor_sample_count = anchor_sample_count
        self.anchor_sample_seed = int(anchor_sample_seed)
        self.batch_by_source_sequence = bool(batch_by_source_sequence)
        self.source_sequence_anchor_cap = source_sequence_anchor_cap
        self.file_cache_entries = int(file_cache_entries)
        self._batch_indices = array("I")
        self._sequence_indices = array("I")
        self._token_indices = array("I")
        self._absolute_positions = array("I")
        self._paths: dict[int, Path] = {}
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        batch_count = int(store.manifest["batch_count"])
        remaining = max_positions
        for batch_index in range(batch_count):
            record = store.manifest["batches"].get(f"{batch_index:06d}", {})
            main_record = record.get("projected_main")
            if not isinstance(main_record, dict):
                raise PrefixStoreError(
                    f"batch {batch_index} has no projected-main manifest record"
                )
            path = store.root / main_record["path"]
            if not path.is_file() or sha256_file(path) != main_record.get("sha256"):
                raise PrefixStoreError(
                    f"batch {batch_index} projected-main file failed verification"
                )
            tensors = load_file(path, device="cpu")
            required = {
                "projected_main",
                "anchor_token_ids",
                "main_attention_mask",
                "main_position_ids",
                "dspark_decode_mask",
            }
            if not required.issubset(tensors):
                raise PrefixStoreError(
                    f"batch {batch_index} projected-main record lacks "
                    f"{sorted(required - set(tensors))}"
                )
            projected = tensors["projected_main"]
            anchors = tensors["anchor_token_ids"]
            attention = tensors["main_attention_mask"].to(dtype=torch.bool)
            positions = tensors["main_position_ids"].to(dtype=torch.long)
            decode = tensors["dspark_decode_mask"].to(dtype=torch.bool)
            if projected.ndim != 3 or projected.shape[-1] != store.hidden_size:
                raise PrefixStoreError("stored projected-main geometry is invalid")
            expected = tuple(projected.shape[:2])
            if any(
                tuple(value.shape) != expected
                for value in (anchors, attention, positions, decode)
            ):
                raise PrefixStoreError(
                    "stored replay metadata geometry is inconsistent"
                )
            selected = torch.nonzero(decode, as_tuple=False)
            for sequence_index, token_index in selected.tolist():
                if not bool(attention[sequence_index, token_index]):
                    raise PrefixStoreError(
                        "dSpark decode mask selected a padded token"
                    )
                absolute = int(positions[sequence_index, token_index])
                if min(batch_index, sequence_index, token_index, absolute) < 0:
                    raise PrefixStoreError("stored replay coordinates must be non-negative")
                self._batch_indices.append(batch_index)
                self._sequence_indices.append(sequence_index)
                self._token_indices.append(token_index)
                self._absolute_positions.append(absolute)
                if remaining is not None:
                    remaining -= 1
                    if remaining == 0:
                        break
            self._paths[batch_index] = path
            del tensors, projected, anchors, attention, positions, decode, selected
            if remaining == 0:
                break

        if not self._batch_indices:
            raise PrefixStoreError("prefix store contains no eligible dSpark positions")

        self.source_position_count = len(self._batch_indices)
        if anchor_sample_count is not None:
            if anchor_sample_count > self.source_position_count:
                raise PrefixStoreError(
                    "anchor_sample_count exceeds eligible dSpark positions: "
                    f"{anchor_sample_count} > {self.source_position_count}"
                )
            selected = _stratified_anchor_indices(
                self.source_position_count,
                anchor_sample_count,
                self.anchor_sample_seed,
            )
            self._batch_indices = array(
                "I", (self._batch_indices[index] for index in selected)
            )
            self._sequence_indices = array(
                "I", (self._sequence_indices[index] for index in selected)
            )
            self._token_indices = array(
                "I", (self._token_indices[index] for index in selected)
            )
            self._absolute_positions = array(
                "I", (self._absolute_positions[index] for index in selected)
            )

        coordinate_digest = hashlib.sha256()
        for coordinates in zip(
            self._batch_indices,
            self._sequence_indices,
            self._token_indices,
            self._absolute_positions,
            strict=True,
        ):
            coordinate_digest.update(
                struct.pack("<QQQQ", *(int(value) for value in coordinates))
            )
        self.anchor_selection_identity = {
            "contract": (
                ANCHOR_SELECTION_CONTRACT
                if anchor_sample_count is not None
                else "ds41rt-mtp-all-eligible-anchors-v1"
            ),
            "seed": (
                self.anchor_sample_seed if anchor_sample_count is not None else None
            ),
            "source_position_count": self.source_position_count,
            "selected_position_count": len(self._batch_indices),
            "selected_coordinates_sha256": coordinate_digest.hexdigest(),
        }
        self._batch_spans = self._build_batch_spans()

    def _build_batch_spans(self) -> list[tuple[int, int]]:
        if not self.batch_by_source_sequence:
            return [
                (start, min(start + self.replay_batch_size, self.position_count))
                for start in range(0, self.position_count, self.replay_batch_size)
            ]

        spans: list[tuple[int, int]] = []
        start = 0
        while start < self.position_count:
            source = (
                int(self._batch_indices[start]),
                int(self._sequence_indices[start]),
            )
            stop = start + 1
            while stop < self.position_count and (
                int(self._batch_indices[stop]),
                int(self._sequence_indices[stop]),
            ) == source:
                stop += 1
            cap = self.source_sequence_anchor_cap
            if cap is None:
                spans.append((start, stop))
            else:
                spans.extend(
                    (chunk_start, min(chunk_start + cap, stop))
                    for chunk_start in range(start, stop, cap)
                )
            start = stop
        return spans

    @property
    def position_count(self) -> int:
        return len(self._batch_indices)

    @property
    def row_counts(self) -> list[int]:
        return [stop - start for start, stop in self._batch_spans]

    @property
    def replay_batching_identity(self) -> dict[str, Any]:
        row_digest = hashlib.sha256()
        for count in self.row_counts:
            row_digest.update(struct.pack("<Q", count))
        return {
            "contract": (
                SEQUENCE_REPLAY_BATCH_CONTRACT
                if self.batch_by_source_sequence
                else FIXED_REPLAY_BATCH_CONTRACT
            ),
            "source_sequence_anchor_cap": self.source_sequence_anchor_cap,
            "batch_count": len(self),
            "minimum_anchors": min(self.row_counts),
            "maximum_anchors": max(self.row_counts),
            "row_counts_sha256": row_digest.hexdigest(),
        }

    @property
    def gptqmodel_calibration_summary(self) -> dict[str, int]:
        # Each eligible position jointly issues the anchor plus four proposal
        # rows.  Expose this without forcing LoopProcessor to iterate the lazy
        # tensor dataset merely to calculate display statistics.
        return {
            "batch_count": len(self),
            "input_ids_total_length": self.position_count * 5,
            "input_ids_max_length": 5,
            "total_calibration_tokens": self.position_count * 5,
        }

    def __len__(self) -> int:
        return len(self._batch_spans)

    def _load_batch(self, batch_index: int) -> dict[str, Any]:
        from safetensors.torch import load_file

        cached = self._cache.pop(batch_index, None)
        if cached is None:
            path = self._paths.get(batch_index)
            if path is None:
                raise PrefixStoreError(f"replay source batch {batch_index} is absent")
            cached = load_file(path, device="cpu")
        self._cache[batch_index] = cached
        while len(self._cache) > self.file_cache_entries:
            self._cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int | slice):
        import torch

        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        start, stop = self._batch_spans[index]
        pending: list[dict[str, Any]] = []
        for flat_index in range(start, stop):
            batch_index = int(self._batch_indices[flat_index])
            sequence_index = int(self._sequence_indices[flat_index])
            token_index = int(self._token_indices[flat_index])
            tensors = self._load_batch(batch_index)
            attention = tensors["main_attention_mask"].to(dtype=torch.bool)
            positions = tensors["main_position_ids"].to(dtype=torch.long)
            valid = torch.nonzero(
                attention[sequence_index, : token_index + 1], as_tuple=False
            ).flatten()[-128:]
            replay_positions = positions[sequence_index, valid]
            if replay_positions.numel() == 0 or (
                replay_positions.numel() > 1
                and not torch.equal(
                    replay_positions[1:], replay_positions[:-1] + 1
                )
            ):
                raise PrefixStoreError(
                    "stored replay main positions are empty or non-contiguous"
                )
            pending.append(
                {
                    "main": tensors["projected_main"][sequence_index, valid],
                    "positions": replay_positions,
                    "anchor": int(tensors["anchor_token_ids"][sequence_index, token_index]),
                    "source": {
                        "batch_index": batch_index,
                        "sequence_index": sequence_index,
                        "token_index": token_index,
                        "absolute_position": int(self._absolute_positions[flat_index]),
                    },
                }
            )

        width = max(int(item["main"].shape[0]) for item in pending)
        if width > 128:
            raise PrefixStoreError(f"stored replay window {width} exceeds 128")
        dtype = pending[0]["main"].dtype
        projected = torch.zeros(len(pending), width, self.store.hidden_size, dtype=dtype)
        positions = torch.zeros(len(pending), width, dtype=torch.long)
        mask = torch.zeros(len(pending), width, dtype=torch.bool)
        anchors = torch.empty(len(pending), dtype=torch.long)
        sources = []
        for row, item in enumerate(pending):
            rows = int(item["main"].shape[0])
            offset = width - rows
            projected[row, offset:].copy_(item["main"])
            positions[row, offset:].copy_(item["positions"])
            mask[row, offset:] = True
            anchors[row] = int(item["anchor"])
            sources.append(item["source"])

        from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4MTPReplayBatch

        return DeepSeekV4MTPStoredReplayBatch(
            replay_batch=DeepSeekV4MTPReplayBatch(
                target_taps=None,
                projected_main=projected.to(device=self.device),
                anchor_token_ids=anchors.to(device=self.device),
                main_position_ids=positions.to(device=self.device),
                main_attention_mask=mask.to(device=self.device),
            ),
            sources=tuple(sources),
        )

    def __iter__(self) -> Iterator[DeepSeekV4MTPStoredReplayBatch]:
        for index in range(len(self)):
            yield self[index]


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


def _atomic_safetensors(path: Path, tensors: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as output:
        temporary = Path(output.name)
    try:
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in tensors.items()
            },
            temporary,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tensor_records(tensors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": tensor_sha256(tensor),
        }
        for name, tensor in sorted(tensors.items())
    }


class DeepSeekV4MTPPrefixStore:
    """Persist three target taps and reduce them to replay-ready main rows.

    The callback is synchronous by design. Each incoming event is consumed or
    durably written before GPTQModel releases that layer's replay outputs.
    ``projector`` receives the three collapsed BF16 taps on its explicitly
    materialized projection device. ``anchor_resolver`` receives the raw final
    target residual and the original token metadata plus the shifted dSpark
    eligibility mask on that same device, and must return the target model's
    declared anchor IDs for eligible sequence rows. Only one replay batch is
    staged at a time so dynamic target-layer placement cannot change this
    device contract or retain a corpus-sized duplicate.
    """

    def __init__(
        self,
        root: Path,
        *,
        target_layer_ids: tuple[int, int, int],
        hidden_size: int,
        hc_mult: int,
        projector: Callable[[tuple[Any, Any, Any]], Any],
        anchor_resolver: Callable[[Any, Any, Any, Any, Any], Any],
        projection_device: Any,
        projection_dtype: Any,
        provenance: dict[str, Any],
    ) -> None:
        import torch

        if len(target_layer_ids) != 3 or tuple(sorted(target_layer_ids)) != tuple(
            target_layer_ids
        ):
            raise PrefixStoreError("target_layer_ids must be three ordered layer IDs")
        if hidden_size <= 0 or hc_mult <= 0:
            raise PrefixStoreError("hidden_size and hc_mult must be positive")
        if not callable(projector) or not callable(anchor_resolver):
            raise PrefixStoreError("projector and anchor_resolver must be callable")
        try:
            projection_device = torch.device(projection_device)
        except (TypeError, RuntimeError) as exc:
            raise PrefixStoreError("projection_device must be a torch device") from exc
        if (
            not isinstance(projection_dtype, torch.dtype)
            or not projection_dtype.is_floating_point
        ):
            raise PrefixStoreError("projection_dtype must be a floating torch dtype")
        if not isinstance(provenance, dict) or not provenance:
            raise PrefixStoreError("prefix-store provenance must be a non-empty object")
        json.dumps(provenance, sort_keys=True)
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.target_layer_ids = tuple(int(value) for value in target_layer_ids)
        self.hidden_size = int(hidden_size)
        self.hc_mult = int(hc_mult)
        self.projector = projector
        self.anchor_resolver = anchor_resolver
        self.projection_device = projection_device
        self.projection_dtype = projection_dtype
        self.provenance = provenance
        self.read_only = False
        self.manifest = self._load_or_create_manifest()

    @classmethod
    def open_complete(
        cls,
        root: Path,
        *,
        expected_manifest_sha256: str | None = None,
        expected_provenance: dict[str, Any] | None = None,
    ) -> "DeepSeekV4MTPPrefixStore":
        """Open an immutable replay-only fan-out without target callables."""

        raw_root = root.expanduser()
        if raw_root.is_symlink() or not raw_root.is_dir():
            raise PrefixStoreError("prefix-store root must be one regular directory")
        resolved_root = raw_root.resolve(strict=True)
        manifest_path = resolved_root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise PrefixStoreError("complete prefix store has no regular manifest")
        manifest_digest = sha256_file(manifest_path)
        if (
            expected_manifest_sha256 is not None
            and manifest_digest != expected_manifest_sha256
        ):
            raise PrefixStoreError("prefix-store manifest identity mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PrefixStoreError(f"cannot read prefix-store manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise PrefixStoreError("complete prefix-store manifest is not an object")
        target_layer_ids = manifest.get("target_layer_ids")
        hidden_size = manifest.get("hidden_size")
        hc_mult = manifest.get("hc_mult")
        batch_count = manifest.get("batch_count")
        provenance = manifest.get("provenance")
        batches = manifest.get("batches")
        if (
            manifest.get("schema") != SCHEMA
            or manifest.get("status") != "complete"
            or not isinstance(target_layer_ids, list)
            or len(target_layer_ids) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in target_layer_ids
            )
            or target_layer_ids != sorted(set(target_layer_ids))
            or isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
            or isinstance(hc_mult, bool)
            or not isinstance(hc_mult, int)
            or hc_mult <= 0
            or isinstance(batch_count, bool)
            or not isinstance(batch_count, int)
            or batch_count <= 0
            or not isinstance(provenance, dict)
            or not provenance
            or not isinstance(batches, dict)
            or set(batches) != {f"{index:06d}" for index in range(batch_count)}
            or manifest.get("completed_layers") != target_layer_ids
        ):
            raise PrefixStoreError("complete prefix-store manifest is inconsistent")
        if expected_provenance is not None and provenance != expected_provenance:
            raise PrefixStoreError("prefix-store provenance mismatch")
        for batch_index in range(batch_count):
            record = batches[f"{batch_index:06d}"]
            projected = record.get("projected_main") if isinstance(record, dict) else None
            target_taps = record.get("target_taps") if isinstance(record, dict) else None
            if (
                not isinstance(projected, dict)
                or not isinstance(target_taps, dict)
                or set(target_taps) != {str(value) for value in target_layer_ids}
            ):
                raise PrefixStoreError(
                    f"complete prefix-store batch {batch_index} is incomplete"
                )
            relative = projected.get("path")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(projected.get("sha256"), str)
                or SHA256_RE.fullmatch(projected["sha256"]) is None
            ):
                raise PrefixStoreError(
                    f"complete prefix-store batch {batch_index} has an invalid projection record"
                )
            raw_projected_path = resolved_root / relative
            projected_path = raw_projected_path.resolve()
            if (
                not projected_path.is_relative_to(resolved_root)
                or raw_projected_path.is_symlink()
                or not projected_path.is_file()
                or projected_path.stat().st_size != projected.get("bytes")
                or sha256_file(projected_path) != projected["sha256"]
            ):
                raise PrefixStoreError(
                    f"complete prefix-store batch {batch_index} projection failed validation"
                )

        instance = object.__new__(cls)
        instance.root = resolved_root
        instance.manifest_path = manifest_path
        instance.target_layer_ids = tuple(target_layer_ids)
        instance.hidden_size = hidden_size
        instance.hc_mult = hc_mult
        instance.projector = None
        instance.anchor_resolver = None
        instance.projection_device = None
        instance.projection_dtype = None
        instance.provenance = provenance
        instance.read_only = True
        instance.manifest = manifest
        instance.manifest_sha256 = manifest_digest
        return instance

    def _new_manifest(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "status": "incomplete",
            "target_layer_ids": list(self.target_layer_ids),
            "hidden_size": self.hidden_size,
            "hc_mult": self.hc_mult,
            "provenance": self.provenance,
            "batch_count": None,
            "completed_layers": [],
            "batches": {},
        }

    def _load_or_create_manifest(self) -> dict[str, Any]:
        expected = self._new_manifest()
        if not self.manifest_path.exists():
            _atomic_json(self.manifest_path, expected)
            return expected
        try:
            actual = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PrefixStoreError(f"cannot read prefix-store manifest: {exc}") from exc
        for key in (
            "schema",
            "target_layer_ids",
            "hidden_size",
            "hc_mult",
            "provenance",
        ):
            if actual.get(key) != expected[key]:
                raise PrefixStoreError(
                    f"prefix-store manifest {key} mismatch: "
                    f"actual={actual.get(key)!r} expected={expected[key]!r}"
                )
        if not isinstance(actual.get("batches"), dict):
            raise PrefixStoreError("prefix-store manifest batches must be an object")
        return actual

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise PrefixStoreError(f"prefix-store path escapes root: {resolved}")
        return resolved.relative_to(self.root).as_posix()

    def _write_or_verify(
        self,
        path: Path,
        tensors: dict[str, Any],
        *,
        committed: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from safetensors.torch import load_file

        expected_tensors = _tensor_records(tensors)
        if path.exists() and committed is not None:
            if not isinstance(committed, dict):
                raise PrefixStoreError(
                    f"committed prefix-store record is invalid: {path}"
                )
            existing = load_file(path, device="cpu")
            actual_tensors = _tensor_records(existing)
            if (
                committed.get("path") != self._relative(path)
                or committed.get("bytes") != path.stat().st_size
                or committed.get("sha256") != sha256_file(path)
                or committed.get("tensors") != actual_tensors
            ):
                raise PrefixStoreError(
                    f"committed prefix-store record failed verification: {path}"
                )
            if actual_tensors != expected_tensors:
                raise PrefixStoreError(
                    f"existing prefix-store record differs from replay: {path}"
                )
        else:
            # A callback can fail after atomically writing one or more batch
            # files but before atomically advancing the event-level manifest.
            # Such files are deliberately not resumable evidence: replacing
            # them prevents a later replay from mixing two executions of one
            # target layer. Manifest-committed files take the strict path
            # above and can never be silently replaced.
            _atomic_safetensors(path, tensors)
        return {
            "path": self._relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "tensors": expected_tensors,
        }

    def _validate_event(self, event: Any) -> tuple[int, int]:
        layer_index = int(getattr(event, "layer_index", -1))
        if layer_index not in self.target_layer_ids:
            raise PrefixStoreError(f"unexpected target-tap layer {layer_index}")
        collapsed = getattr(event, "collapsed_target_taps", None)
        raw = getattr(event, "raw_layer_outputs", None)
        if not isinstance(collapsed, tuple) or not collapsed:
            raise PrefixStoreError("target-tap event has no collapsed batches")
        if not isinstance(raw, tuple) or len(raw) != len(collapsed):
            raise PrefixStoreError("target-tap raw/collapsed batch count mismatch")
        expected_batches = self.manifest.get("batch_count")
        if expected_batches is not None and expected_batches != len(collapsed):
            raise PrefixStoreError(
                f"target-tap batch count changed: {len(collapsed)} != {expected_batches}"
            )
        return layer_index, len(collapsed)

    @staticmethod
    def _batch_metadata(event: Any, batch_index: int) -> tuple[Any, Any, Any, Any]:
        import torch

        from gptqmodel.models.definitions.deepseek_v4 import (
            MTP_CAPTURE_ATTENTION_MASK,
            MTP_CAPTURE_DECODE_MASK,
            MTP_CAPTURE_INPUT_IDS,
        )

        kwargs = event.layer_input_kwargs[batch_index]
        input_ids = kwargs.get(MTP_CAPTURE_INPUT_IDS)
        attention_mask = kwargs.get(MTP_CAPTURE_ATTENTION_MASK)
        decode_mask = kwargs.get(MTP_CAPTURE_DECODE_MASK)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise PrefixStoreError("target-tap batch lacks original rank-2 input_ids")
        if not isinstance(attention_mask, torch.Tensor) or tuple(
            attention_mask.shape
        ) != tuple(input_ids.shape):
            raise PrefixStoreError(
                "target-tap batch lacks its original rank-2 attention mask"
            )
        if not isinstance(decode_mask, torch.Tensor) or tuple(
            decode_mask.shape
        ) != tuple(input_ids.shape):
            raise PrefixStoreError(
                "target-tap batch lacks its shifted rank-2 dSpark decode mask"
            )
        if len(event.position_ids) != len(event.collapsed_target_taps):
            raise PrefixStoreError("target-tap event lacks per-batch position_ids")
        position_ids = event.position_ids[batch_index]
        if not isinstance(position_ids, torch.Tensor) or tuple(position_ids.shape) != tuple(
            input_ids.shape
        ):
            raise PrefixStoreError("target-tap position_ids geometry mismatch")
        return (
            input_ids.to(dtype=torch.long),
            attention_mask.to(dtype=torch.bool),
            decode_mask.to(dtype=torch.bool),
            position_ids.to(dtype=torch.long),
        )

    def _tap_path(self, layer_index: int, batch_index: int) -> Path:
        return (
            self.root
            / "target-taps"
            / f"layer_{layer_index:02d}"
            / f"batch_{batch_index:06d}.safetensors"
        )

    def _main_path(self, batch_index: int) -> Path:
        return self.root / "projected-main" / f"batch_{batch_index:06d}.safetensors"

    def _load_tap(self, layer_index: int, batch_index: int, device: Any) -> Any:
        from safetensors.torch import load_file

        path = self._tap_path(layer_index, batch_index)
        if not path.is_file():
            raise PrefixStoreError(
                f"target tap {layer_index} batch {batch_index} is missing"
            )
        expected = (
            self.manifest.get("batches", {})
            .get(f"{batch_index:06d}", {})
            .get("target_taps", {})
            .get(str(layer_index))
        )
        if not isinstance(expected, dict) or sha256_file(path) != expected.get("sha256"):
            raise PrefixStoreError(
                f"target tap {layer_index} batch {batch_index} failed manifest verification"
            )
        return load_file(path, device=str(device))["target_tap"]

    def __call__(self, event: Any) -> None:
        import torch

        if self.read_only:
            raise PrefixStoreError("replay-only prefix store cannot accept target taps")

        layer_index, batch_count = self._validate_event(event)
        self.manifest["batch_count"] = batch_count
        batches = self.manifest["batches"]
        for batch_index, current in enumerate(event.collapsed_target_taps):
            raw = event.raw_layer_outputs[batch_index]
            if not isinstance(current, torch.Tensor) or current.ndim != 3:
                raise PrefixStoreError("collapsed target tap must be rank-3")
            if current.shape[-1] != self.hidden_size:
                raise PrefixStoreError("collapsed target tap hidden width mismatch")
            if not isinstance(raw, torch.Tensor) or raw.ndim != 4:
                raise PrefixStoreError("raw target output must be rank-4")
            if tuple(raw.shape[:2]) != tuple(current.shape[:2]) or tuple(
                raw.shape[2:]
            ) != (self.hc_mult, self.hidden_size):
                raise PrefixStoreError("raw/collapsed target tap geometry mismatch")

            batch_key = f"{batch_index:06d}"
            batch_record = batches.setdefault(batch_key, {"target_taps": {}})
            committed_tap = batch_record.setdefault("target_taps", {}).get(
                str(layer_index)
            )
            tap_record = self._write_or_verify(
                self._tap_path(layer_index, batch_index),
                {"target_tap": current},
                committed=committed_tap,
            )
            batch_record["target_taps"][str(layer_index)] = tap_record

            if layer_index != self.target_layer_ids[-1]:
                continue
            input_ids, attention_mask, decode_mask, position_ids = self._batch_metadata(
                event, batch_index
            )
            if (
                current.dtype != self.projection_dtype
                or raw.dtype != self.projection_dtype
            ):
                raise PrefixStoreError(
                    "target residuals changed dtype before MTP prefix projection"
                )
            device = self.projection_device
            prior_taps = tuple(
                self._load_tap(target, batch_index, device)
                for target in self.target_layer_ids[:-1]
            )
            staged_current = current.to(device=device, non_blocking=False)
            staged_raw = raw.to(device=device, non_blocking=False)
            staged_input_ids = input_ids.to(device=device, non_blocking=False)
            staged_attention_mask = attention_mask.to(device=device, non_blocking=False)
            staged_decode_mask = decode_mask.to(device=device, non_blocking=False)
            staged_position_ids = position_ids.to(device=device, non_blocking=False)
            taps = (*prior_taps, staged_current)
            if any(tap.dtype != self.projection_dtype for tap in taps):
                raise PrefixStoreError("target taps changed dtype before projection")
            projected_main = self.projector(taps)
            if not isinstance(projected_main, torch.Tensor) or tuple(
                projected_main.shape
            ) != tuple(current.shape):
                raise PrefixStoreError("target-tap projector returned invalid geometry")
            if (
                projected_main.device != device
                or projected_main.dtype != self.projection_dtype
            ):
                raise PrefixStoreError(
                    "target-tap projector returned an invalid device or dtype"
                )
            anchors = self.anchor_resolver(
                staged_raw,
                staged_input_ids,
                staged_attention_mask,
                staged_decode_mask,
                staged_position_ids,
            )
            if (
                not isinstance(anchors, torch.Tensor)
                or tuple(anchors.shape) != tuple(input_ids.shape)
                or anchors.device != device
            ):
                raise PrefixStoreError("target anchor resolver returned invalid geometry")
            if anchors.dtype not in (torch.int32, torch.int64):
                raise PrefixStoreError("target anchor resolver must return integer IDs")
            eligible_anchors = anchors[staged_decode_mask & staged_attention_mask]
            if eligible_anchors.numel() and bool(torch.any(eligible_anchors < 0)):
                raise PrefixStoreError(
                    "target anchor resolver left an eligible position unresolved"
                )
            final_tensors = {
                "projected_main": projected_main,
                "anchor_token_ids": anchors.to(dtype=torch.long),
                "input_ids": staged_input_ids,
                "main_attention_mask": staged_attention_mask,
                "dspark_decode_mask": staged_decode_mask,
                "main_position_ids": staged_position_ids,
            }
            batch_record["projected_main"] = self._write_or_verify(
                self._main_path(batch_index),
                final_tensors,
                committed=batch_record.get("projected_main"),
            )
            del (
                anchors,
                eligible_anchors,
                final_tensors,
                prior_taps,
                projected_main,
                staged_attention_mask,
                staged_current,
                staged_decode_mask,
                staged_input_ids,
                staged_position_ids,
                staged_raw,
                taps,
            )

        completed = {int(value) for value in self.manifest["completed_layers"]}
        completed.add(layer_index)
        self.manifest["completed_layers"] = sorted(completed)
        if tuple(self.manifest["completed_layers"]) == self.target_layer_ids:
            for batch_index in range(batch_count):
                record = batches.get(f"{batch_index:06d}", {})
                if not isinstance(record.get("projected_main"), dict):
                    raise PrefixStoreError(
                        f"batch {batch_index} has no projected-main record"
                    )
            self.manifest["status"] = "complete"
        _atomic_json(self.manifest_path, self.manifest)

    def replay_dataset(
        self,
        *,
        replay_batch_size: int,
        device: Any = "cpu",
        max_positions: int | None = None,
        anchor_sample_count: int | None = None,
        anchor_sample_seed: int = 0,
        batch_by_source_sequence: bool = False,
        source_sequence_anchor_cap: int | None = None,
    ) -> DeepSeekV4MTPReplayDataset:
        """Return the bounded-memory random-access replay view."""

        return DeepSeekV4MTPReplayDataset(
            self,
            replay_batch_size=replay_batch_size,
            device=device,
            max_positions=max_positions,
            anchor_sample_count=anchor_sample_count,
            anchor_sample_seed=anchor_sample_seed,
            batch_by_source_sequence=batch_by_source_sequence,
            source_sequence_anchor_cap=source_sequence_anchor_cap,
        )

    def iter_replay_batches(
        self,
        *,
        replay_batch_size: int,
        device: Any = "cpu",
        max_positions: int | None = None,
    ):
        """Yield deterministic joint batches of independent dSpark positions."""

        import torch
        from safetensors.torch import load_file

        from gptqmodel.models.definitions.deepseek_v4 import (
            DeepSeekV4MTPReplayBatch,
        )

        if self.manifest.get("status") != "complete":
            raise PrefixStoreError("prefix store is incomplete")
        if replay_batch_size <= 0:
            raise PrefixStoreError("replay_batch_size must be positive")
        if max_positions is not None and max_positions <= 0:
            raise PrefixStoreError("max_positions must be positive when set")
        pending: list[dict[str, Any]] = []
        emitted = 0

        def emit():
            if not pending:
                return None
            width = max(int(item["main"].shape[0]) for item in pending)
            if width > 128:
                raise PrefixStoreError(f"stored replay window {width} exceeds 128")
            dtype = pending[0]["main"].dtype
            projected = torch.zeros(
                len(pending), width, self.hidden_size, dtype=dtype
            )
            positions = torch.zeros(len(pending), width, dtype=torch.long)
            mask = torch.zeros(len(pending), width, dtype=torch.bool)
            anchors = torch.empty(len(pending), dtype=torch.long)
            sources = []
            for row, item in enumerate(pending):
                rows = int(item["main"].shape[0])
                start = width - rows
                projected[row, start:].copy_(item["main"])
                positions[row, start:].copy_(item["positions"])
                mask[row, start:] = True
                anchors[row] = int(item["anchor"])
                sources.append(item["source"])
            result = DeepSeekV4MTPStoredReplayBatch(
                replay_batch=DeepSeekV4MTPReplayBatch(
                    target_taps=None,
                    projected_main=projected.to(device=device),
                    anchor_token_ids=anchors.to(device=device),
                    main_position_ids=positions.to(device=device),
                    main_attention_mask=mask.to(device=device),
                ),
                sources=tuple(sources),
            )
            pending.clear()
            return result

        batch_count = int(self.manifest["batch_count"])
        for batch_index in range(batch_count):
            record = self.manifest["batches"].get(f"{batch_index:06d}", {})
            main_record = record.get("projected_main")
            if not isinstance(main_record, dict):
                raise PrefixStoreError(
                    f"batch {batch_index} has no projected-main manifest record"
                )
            path = self.root / main_record["path"]
            if not path.is_file() or sha256_file(path) != main_record.get("sha256"):
                raise PrefixStoreError(
                    f"batch {batch_index} projected-main file failed verification"
                )
            tensors = load_file(path, device="cpu")
            required = {
                "projected_main",
                "anchor_token_ids",
                "main_attention_mask",
                "main_position_ids",
                "dspark_decode_mask",
            }
            if not required.issubset(tensors):
                raise PrefixStoreError(
                    f"batch {batch_index} projected-main record lacks {sorted(required - set(tensors))}"
                )
            projected = tensors["projected_main"]
            anchors = tensors["anchor_token_ids"]
            attention = tensors["main_attention_mask"].to(dtype=torch.bool)
            positions = tensors["main_position_ids"].to(dtype=torch.long)
            decode = tensors["dspark_decode_mask"].to(dtype=torch.bool)
            if projected.ndim != 3 or projected.shape[-1] != self.hidden_size:
                raise PrefixStoreError("stored projected-main geometry is invalid")
            expected = tuple(projected.shape[:2])
            if any(tuple(value.shape) != expected for value in (anchors, attention, positions, decode)):
                raise PrefixStoreError("stored replay metadata geometry is inconsistent")
            for sequence_index in range(expected[0]):
                for token_index in range(expected[1]):
                    if not bool(decode[sequence_index, token_index]):
                        continue
                    if not bool(attention[sequence_index, token_index]):
                        raise PrefixStoreError("dSpark decode mask selected a padded token")
                    valid = torch.nonzero(
                        attention[sequence_index, : token_index + 1], as_tuple=False
                    ).flatten()
                    valid = valid[-128:]
                    replay_positions = positions[sequence_index, valid]
                    if replay_positions.numel() == 0 or (
                        replay_positions.numel() > 1
                        and not torch.equal(
                            replay_positions[1:], replay_positions[:-1] + 1
                        )
                    ):
                        raise PrefixStoreError(
                            "stored replay main positions are empty or non-contiguous"
                        )
                    pending.append(
                        {
                            "main": projected[sequence_index, valid],
                            "positions": replay_positions,
                            "anchor": int(anchors[sequence_index, token_index]),
                            "source": {
                                "batch_index": batch_index,
                                "sequence_index": sequence_index,
                                "token_index": token_index,
                                "absolute_position": int(
                                    positions[sequence_index, token_index]
                                ),
                            },
                        }
                    )
                    emitted += 1
                    if len(pending) == replay_batch_size:
                        yield emit()
                    if max_positions is not None and emitted >= max_positions:
                        final = emit()
                        if final is not None:
                            yield final
                        return
        final = emit()
        if final is not None:
            yield final


__all__ = [
    "DeepSeekV4MTPReplayDataset",
    "DeepSeekV4MTPPrefixStore",
    "DeepSeekV4MTPStoredReplayBatch",
    "ANCHOR_SELECTION_CONTRACT",
    "FIXED_REPLAY_BATCH_CONTRACT",
    "PrefixStoreError",
    "SCHEMA",
    "SEQUENCE_REPLAY_BATCH_CONTRACT",
    "sha256_file",
    "tensor_sha256",
]
