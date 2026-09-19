"""Content-preserving descriptor remapping for regrouped replay batches.

Experiment C regroups captured rows (independent singletons, duplicates,
mixed reference/candidate pairs, swapped order).  A regrouped batch must not
carry stale predecessor indices: every remapped descriptor is re-derived from
the *original* geometry and checked to address exactly the same operand bytes
as before, or the remap is rejected.

Two addressing strategies are provided:

* ``wave`` — keep earlier-wave predecessors as earlier-wave rows of the
  regrouped batch (adjacency re-checked after the remap);
* ``compact_pending`` — rewrite each row's original predecessor into a fresh
  compact pending slot that holds byte-identical operand content, so a
  regrouped row never depends on which other rows happen to share the batch.

Both are pure CPU byte bookkeeping: no float arithmetic, no device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .schema import (
    DESCRIPTOR_SENTINEL,
    EARLIER_WAVE_ROW,
    INVALID_SENTINEL,
    PENDING_ROW_BYTES,
    PENDING_SLOT,
    UNRESOLVED,
    resolve_predecessor,
)


class RemapError(Exception):
    """A remap would change operand semantics or reference stale indices."""


def _row(blob: bytes, index: int, row_bytes: int) -> bytes:
    start = index * row_bytes
    return blob[start:start + row_bytes]


def compact_pending_slots(
    pending_kv: bytes,
    pending_scores: bytes,
    referenced_slots: Sequence[int],
    slot_count: int,
) -> Tuple[bytes, bytes, Dict[int, int]]:
    """Build [len(referenced_slots), 512] FP32 planes holding exactly the
    referenced rows, byte-identical to the source plane rows.

    Returns ``(kv_plane, scores_plane, {old_slot: new_slot})``.
    """
    unique = list(dict.fromkeys(referenced_slots))
    kv_rows, scores_rows = [], []
    for slot in unique:
        if not (0 <= slot < slot_count):
            raise RemapError(f"pending slot {slot} outside plane {slot_count}")
        kv_rows.append(_row(pending_kv, slot, PENDING_ROW_BYTES))
        scores_rows.append(_row(pending_scores, slot, PENDING_ROW_BYTES))
    return (b"".join(kv_rows), b"".join(scores_rows),
            {old: new for new, old in enumerate(unique)})


def remap_descriptors(
    descriptors_in_new_order: Sequence[int],
    original_rows_in_new_order: Sequence[int],
    *,
    original_slot_count: int,
    row_map: Dict[int, int],
    slot_map: Dict[int, int],
    new_slot_count: int,
    strategy: str,
) -> Tuple[int, ...]:
    """Remap captured device descriptors for a regrouped batch.

    ``descriptors_in_new_order[new_pos]`` is the descriptor recorded for the
    row now sitting at ``new_pos``; ``original_rows_in_new_order[new_pos]``
    is that row's index in its original capture.  ``row_map`` maps an
    original wave row to its position in the regrouped batch;
    ``slot_map`` maps an original pending slot to its (content-identical)
    compact slot.

    Semantic rules (enforced, not assumed):
      * the sentinel stays the sentinel (an even-position row still pools
        nothing regardless of regrouping);
      * a pending-slot descriptor is rewritten through ``slot_map`` and must
        not address an unmapped slot;
      * an earlier-wave descriptor is rewritten through ``row_map`` and the
        target must still strictly precede its owner — a remap that would
        make a row pool itself or a later row is rejected, never clamped;
      * unresolved descriptors are rejected.
    """
    if not (len(descriptors_in_new_order) == len(original_rows_in_new_order)):
        raise RemapError("descriptor and row-order lengths differ")
    if strategy not in ("wave", "compact_pending"):
        raise RemapError(f"unknown remap strategy {strategy!r}")
    remapped: List[int] = []
    for new_pos, (descriptor, original_row) in enumerate(
        zip(descriptors_in_new_order, original_rows_in_new_order)
    ):
        if original_row not in row_map.values():
            raise RemapError(
                f"original row {original_row} has no position in the regrouped batch"
            )
        predecessor = resolve_predecessor(descriptor, original_row,
                                          original_slot_count)
        if predecessor["kind"] == INVALID_SENTINEL:
            remapped.append(DESCRIPTOR_SENTINEL)
        elif predecessor["kind"] == PENDING_SLOT:
            slot = predecessor["slot"]
            if slot not in slot_map:
                raise RemapError(
                    f"row {original_row} pending slot {slot} has no content "
                    "mapping in the regrouped batch"
                )
            remapped.append(slot_map[slot])
        elif predecessor["kind"] == EARLIER_WAVE_ROW:
            earlier = predecessor["wave_row"]
            if earlier not in row_map:
                raise RemapError(
                    f"row {original_row} predecessor wave row {earlier} is "
                    "absent from the regrouped batch"
                )
            target = row_map[earlier]
            if strategy == "wave":
                if target >= new_pos:
                    raise RemapError(
                        f"remap would make new row {new_pos} pool row {target} "
                        "which does not strictly precede it"
                    )
                remapped.append(new_slot_count + target)
            else:  # compact_pending: handled by the caller's slot map
                raise RemapError(
                    "earlier-wave descriptors require the caller to install "
                    "compact pending slots; use remap_batch"
                )
        else:
            raise RemapError(
                f"row {original_row} descriptor {descriptor:#x} is unresolved"
            )
    return tuple(remapped)


@dataclass
class BatchRemap:
    """Result of a full batch remap: descriptors plus the compact-plane source
    map the caller must materialize with :func:`build_compact_planes`."""

    descriptors: Tuple[int, ...]
    slot_count: int
    strategy: str
    provenance: Tuple[str, ...]
    # ("pending", slot) | ("wave", row) -> new compact slot; materialize with
    # build_compact_planes.
    source_map: Dict[Tuple[str, int], int]


def remap_batch(
    descriptors_in_new_order: Sequence[int],
    original_rows_in_new_order: Sequence[int],
    *,
    original_slot_count: int,
    row_map: Dict[int, int],
    pending_kv: bytes,
    pending_scores: bytes,
    strategy: str,
    compact_slots: Dict[int, int] = frozenset(),
) -> BatchRemap:
    """Full batch remap with operand-content preservation.

    ``strategy == "wave"`` keeps earlier-wave references in-batch (the
    referenced rows must be present in ``row_map``); the returned
    ``source_map`` is empty and no pending plane is needed.  ``strategy ==
    "compact_pending"`` converts *every* non-sentinel predecessor into a
    compact pending slot.  ``compact_slots`` maps an original *wave row* to
    the original pending slot whose content should stand in for it (identical
    FP32 bytes installed by the caller); pending-slot predecessors keep their
    own slot content.
    """
    if strategy == "wave":
        slot_map: Dict[int, int] = {}
        descriptors = remap_descriptors(
            descriptors_in_new_order, original_rows_in_new_order,
            original_slot_count=original_slot_count, row_map=row_map,
            slot_map=slot_map, new_slot_count=0, strategy=strategy,
        )
        provenance = tuple(
            f"new row {p} keeps original descriptor {d:#x} semantics"
            for p, d in enumerate(descriptors_in_new_order)
        )
        return BatchRemap(descriptors, 0, strategy, provenance, {})

    if strategy != "compact_pending":
        raise RemapError(f"unknown remap strategy {strategy!r}")

    # First pass: decide the compact slot each row's predecessor needs.
    slot_sources: List[Tuple[str, int]] = []   # ("pending", slot) | ("wave", row)
    for descriptor, original_row in zip(descriptors_in_new_order,
                                        original_rows_in_new_order):
        predecessor = resolve_predecessor(descriptor, original_row,
                                          original_slot_count)
        if predecessor["kind"] == INVALID_SENTINEL:
            continue
        if predecessor["kind"] == PENDING_SLOT:
            slot_sources.append(("pending", predecessor["slot"]))
        elif predecessor["kind"] == EARLIER_WAVE_ROW:
            source_row = predecessor["wave_row"]
            if source_row not in compact_slots:
                raise RemapError(
                    f"row {original_row} earlier-wave predecessor {source_row} "
                    "needs a compact slot content source"
                )
            slot_sources.append(("wave", compact_slots[source_row]))
        else:
            raise RemapError(
                f"row {original_row} descriptor {descriptor:#x} is unresolved"
            )

    unique_sources = list(dict.fromkeys(slot_sources))
    source_map = {source: index for index, source in enumerate(unique_sources)}
    new_slot_count = len(unique_sources)

    descriptors = []
    provenance = []
    for new_pos, (descriptor, original_row) in enumerate(
        zip(descriptors_in_new_order, original_rows_in_new_order)
    ):
        predecessor = resolve_predecessor(descriptor, original_row,
                                          original_slot_count)
        if predecessor["kind"] == INVALID_SENTINEL:
            descriptors.append(DESCRIPTOR_SENTINEL)
            provenance.append(f"new row {new_pos}: sentinel (no pooling)")
        elif predecessor["kind"] == PENDING_SLOT:
            source = ("pending", predecessor["slot"])
            descriptors.append(source_map[source])
            provenance.append(
                f"new row {new_pos}: compact pending slot {source_map[source]} "
                f"= original pending slot {predecessor['slot']} (bytes preserved)"
            )
        else:
            source = ("wave", compact_slots[predecessor["wave_row"]])
            descriptors.append(source_map[source])
            provenance.append(
                f"new row {new_pos}: compact pending slot {source_map[source]} "
                f"= original wave row {predecessor['wave_row']} operands (bytes preserved)"
            )
    return BatchRemap(
        descriptors=tuple(descriptors),
        slot_count=new_slot_count,
        strategy=strategy,
        provenance=tuple(provenance),
        source_map=source_map,
    )


def build_compact_planes(
    source_map: Dict[Tuple[str, int], int],
    *,
    pending_kv: bytes,
    pending_scores: bytes,
    wave_kv_rows: Dict[int, bytes],
    wave_scores_rows: Dict[int, bytes],
    original_slot_count: int,
) -> Tuple[bytes, bytes]:
    """Assemble the [len(source_map), 512] FP32 compact planes for a
    ``compact_pending`` remap, byte-copying each source row."""
    kv_rows: List[bytes] = [b""] * len(source_map)
    scores_rows: List[bytes] = [b""] * len(source_map)
    for (kind, index), new_slot in source_map.items():
        if kind == "pending":
            if not (0 <= index < original_slot_count):
                raise RemapError(f"pending slot {index} outside source plane")
            kv_rows[new_slot] = _row(pending_kv, index, PENDING_ROW_BYTES)
            scores_rows[new_slot] = _row(pending_scores, index, PENDING_ROW_BYTES)
        else:
            if index not in wave_kv_rows:
                raise RemapError(f"wave row {index} has no captured operands")
            kv_rows[new_slot] = wave_kv_rows[index]
            scores_rows[new_slot] = wave_scores_rows[index]
    if any(not row for row in kv_rows):
        raise RemapError("compact plane has an unmapped slot")
    return b"".join(kv_rows), b"".join(scores_rows)
