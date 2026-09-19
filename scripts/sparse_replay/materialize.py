"""Materialize small private replay fixtures from captured rows (CPU only).

The captured shared compressed pool (hundreds of thousands of rows) is never
allocated or copied.  Only the captured *referenced* source rows are placed
into a compact private pool; referenced physical pages are remapped compactly
while preserving logical ids and offset mod 256.  Untouched page-table
entries point outside the compact capacity so no key is inadvertently
unmasked.

Every materialized fixture is verified with the reviewed integer oracle:
the ordered slot mask (masked flag / tag / logical id) must be identical on
the original and compact views, the exact quantized value/scale bytes at
each live slot must be unchanged, and the extractor's canonical digest must
be reproduced exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from .capture import (
    CapturedRow,
    RING_ROWS,
    RING_SCALE_BYTES,
    RING_VALUE_BYTES,
    SOURCE_SCALE_BYTES,
    SOURCE_VALUE_BYTES,
    WINDOW_SCALE_BYTES,
    WINDOW_VALUE_BYTES,
)

# Canonical digest format string (must match the pinned extractor).
CANONICAL_FORMAT = "ds41rt-sparse-attention-input-canonical-v1"

# Page-table sentinel for untouched pages: maps far outside any compact
# capacity both in 32-bit and 64-bit physical arithmetic (never wraps low).
SENTINEL_PAGE = 0xFFFFFFFF

# Root-designated counterfactual site: private_source column 294 low nibble.
COUNTERFACTUAL_COLUMN = 294

# Per-tag quantized codec labels (must match the extractor's slot dtypes).
_TAG_CODECS = {
    "ring": ("fp8e4m3", "uint8-e8m0"),
    "private_window": ("fp8e4m3", "uint8-e8m0"),
    "paged_source": ("fp4e2m1", "fp8e4m3"),
    "private_source": ("fp4e2m1", "fp8e4m3"),
}


class MaterializeError(Exception):
    """Hard materialization/verification failure (never a silent skip)."""


class MissingLivePagedRowError(MaterializeError):
    """A live (kernel-unmasked) paged source id has no captured row."""


def _row(arena: bytes, index: int, row_bytes: int, role: str) -> bytes:
    if not 0 <= index < len(arena) // row_bytes:
        raise MaterializeError(f"{role} row {index} outside the captured arena")
    return arena[index * row_bytes:(index + 1) * row_bytes]


def _resolve(row: CapturedRow, oracle, pages, source_capacity: int) -> dict:
    return oracle.resolve_attention_row(
        row.metadata10,
        row.selected512,
        pages,
        row.view_scalars(source_capacity=source_capacity),
        row.window_end,
        row.source_end,
        window_begin=row.replay_begin,
        width=row.width,
    )


def slot_bytes(row: CapturedRow, oracle, entry: dict,
               pool: Optional[tuple] = None) -> tuple:
    """Exact quantized value/scale bytes the kernel reads for one live slot.

    ``pool`` is the optional ``(pool_values, pool_scales)`` compact arena for
    the *compact* view; when given, paged rows are read at the oracle's new
    physical index.  When absent (original view), paged rows are read through
    the captured ``logical_to_physical`` mapping.
    """
    tag = entry["tag"]
    physical = entry["physical"]
    if tag == oracle.TAG_RING:
        return (
            _row(row.ring_values, physical, RING_VALUE_BYTES, "ring values"),
            _row(row.ring_scales, physical, RING_SCALE_BYTES, "ring scales"),
        )
    if tag == oracle.TAG_PRIVATE_WINDOW:
        return (
            _row(row.window_proposal_values, physical, WINDOW_VALUE_BYTES,
                 "window proposal values"),
            _row(row.window_proposal_scales, physical, WINDOW_SCALE_BYTES,
                 "window proposal scales"),
        )
    if tag == oracle.TAG_PRIVATE_SOURCE:
        return (
            _row(row.source_proposal_values, physical, SOURCE_VALUE_BYTES,
                 "source proposal values"),
            _row(row.source_proposal_scales, physical, SOURCE_SCALE_BYTES,
                 "source proposal scales"),
        )
    if tag == oracle.TAG_PAGED_SOURCE:
        logical = entry["logical"]
        if pool is not None:
            pool_values, pool_scales = pool
            return (
                _row(pool_values, physical, SOURCE_VALUE_BYTES,
                     "compact pool values"),
                _row(pool_scales, physical, SOURCE_SCALE_BYTES,
                     "compact pool scales"),
            )
        if logical not in row.logical_to_physical:
            raise MissingLivePagedRowError(
                f"live paged logical id {logical} has no captured "
                "logical_to_physical entry"
            )
        captured_physical = row.logical_to_physical[logical]
        if captured_physical != physical:
            raise MaterializeError(
                f"paged logical {logical}: oracle physical {physical} != "
                f"captured {captured_physical}"
            )
        try:
            index = row.referenced_physical_rows.index(physical)
        except ValueError as error:
            raise MissingLivePagedRowError(
                f"live paged physical row {physical} is absent from the "
                "captured referenced rows"
            ) from error
        return (
            _row(row.referenced_values, index, SOURCE_VALUE_BYTES,
                 "referenced values"),
            _row(row.referenced_scales, index, SOURCE_SCALE_BYTES,
                 "referenced scales"),
        )
    raise MaterializeError(f"unknown keyslot tag {tag!r}")


def canonical_digest(width: int, compressed: int, query_sha256: str,
                     sink_sha256: str, slots: list) -> str:
    """Extractor-identical canonical digest over ordered slot evidence."""
    body = {
        "format": CANONICAL_FORMAT,
        "effective_width": width,
        "source_format": compressed,
        "query_sha256": query_sha256,
        "sink_sha256": sink_sha256,
        "slots": [
            {
                "masked": slot["masked"],
                "value_dtype": slot["value_dtype"],
                "scale_dtype": slot["scale_dtype"],
                "value_sha256": slot["value_sha256"],
                "scale_sha256": slot["scale_sha256"],
            }
            for slot in slots
        ],
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _evidence_slots(row: CapturedRow, oracle, resolved: dict,
                    pool: Optional[tuple] = None) -> list:
    """Ordered slot evidence (hashes) for a resolved row view."""
    slots = []
    for entry in resolved["keyslots"]:
        record = {
            "slot": entry["slot"],
            "masked": bool(entry["masked"]),
            "tag": entry["tag"],
            "logical": entry["logical"],
            "physical": entry["physical"],
            "value_dtype": None,
            "scale_dtype": None,
            "value_sha256": None,
            "scale_sha256": None,
        }
        if not entry["masked"] and resolved["whole_row_valid"]:
            value_bytes, scale_bytes = slot_bytes(row, oracle, entry, pool)
            value_dtype, scale_dtype = _TAG_CODECS[entry["tag"]]
            record["value_dtype"] = value_dtype
            record["scale_dtype"] = scale_dtype
            record["value_sha256"] = hashlib.sha256(value_bytes).hexdigest()
            record["scale_sha256"] = hashlib.sha256(scale_bytes).hexdigest()
        slots.append(record)
    return slots


@dataclass
class MaterializedRow:
    """Compact private fixture for one captured row (exact raw bytes)."""

    source: CapturedRow
    source_capacity: int
    pool_values: bytes          # source_capacity * 256
    pool_scales: bytes          # source_capacity * 32
    pages_compact: tuple        # page_stride entries; sentinel outside capacity
    page_map: dict              # old page -> new page
    physical_map: dict          # old referenced physical -> new physical
    slot_evidence_compact: list
    canonical_digest_compact: str
    pool_bytes: int = field(init=False)

    def __post_init__(self):
        self.pool_bytes = len(self.pool_values) + len(self.pool_scales)

    def private_source_row_index(self, oracle) -> int:
        """Proposal-row index of the live private-source overlay slot."""
        index = None
        for entry in self.slot_evidence_compact:
            if entry["masked"] or entry["tag"] != oracle.TAG_PRIVATE_SOURCE:
                continue
            if index is not None:
                raise MaterializeError("multiple live private-source slots")
            index = entry["physical"]
        if index is None:
            raise MaterializeError("no live private-source slot in this row")
        return index


def materialize(row: CapturedRow, oracle) -> MaterializedRow:
    """Build and oracle-verify a compact private fixture for one row.

    Steps:
      1. Resolve the original captured view with the oracle.
      2. Require every kernel-live paged source id to have a captured row.
      3. Remap referenced physical pages compactly (offset mod 256 preserved;
         untouched pages point outside the compact capacity).
      4. Re-resolve the compact view and require an identical ordered slot
         mask with byte-identical quantized content at every live slot, and
         the same canonical digest.
    """
    resolved_orig = _resolve(row, oracle, row.pages, row.source_capacity_rows)
    if not resolved_orig["whole_row_valid"]:
        raise MaterializeError(
            f"captured row {row.key} is kernel-invalid "
            f"({resolved_orig['reason']})"
        )
    slots_orig = _evidence_slots(row, oracle, resolved_orig)

    # --- every live paged id must be captured (failure when a live paged ---
    # --- row is missing or the capture is truncated) -----------------------
    m = row.metadata10
    if row.selected512 is not None:
        for selected_id in set(row.selected512):
            if 0 <= selected_id < m[6]:
                if selected_id not in row.logical_to_physical:
                    raise MissingLivePagedRowError(
                        f"selected id {selected_id} is kernel-live "
                        f"(0 <= id < source committed end {m[6]}) but has no "
                        "captured logical_to_physical entry"
                    )

    # --- compact page remap -------------------------------------------------
    # The kernel looks a logical id up through pages[logical // 256], so the
    # remap is indexed by *logical* page.  Physical pages that carry
    # referenced content are renumbered compactly (ascending); every logical
    # page that carries no referenced content points outside the compact
    # capacity, so no key can be inadvertently unmasked.
    physicals = row.referenced_physical_rows
    if tuple(sorted(physicals)) != physicals:
        raise MaterializeError("referenced physical rows are not ascending")
    for logical, physical in row.logical_to_physical.items():
        if row.pages[logical // 256] * 256 + logical % 256 != physical:
            raise MaterializeError(
                f"logical {logical}: page table and logical_to_physical "
                "disagree on the physical row"
            )
    logical_pages = sorted({logical // 256 for logical in row.logical_to_physical})
    physical_pages = sorted({row.pages[logical_page]
                             for logical_page in logical_pages})
    page_map = {old: new for new, old in enumerate(physical_pages)}
    pages_compact = [SENTINEL_PAGE] * row.page_stride
    for logical_page in logical_pages:
        pages_compact[logical_page] = page_map[row.pages[logical_page]]
    pages_compact = tuple(pages_compact)
    physical_map = {
        old: page_map[old // 256] * 256 + old % 256 for old in physicals
    }
    capacity = max(physical_map.values()) + 1
    if capacity > 256 * len(page_map):
        raise MaterializeError("compact capacity exploded; page remap broken")

    pool_values = bytearray(capacity * SOURCE_VALUE_BYTES)
    pool_scales = bytearray(capacity * SOURCE_SCALE_BYTES)
    for old_physical, new_physical in physical_map.items():
        index = physicals.index(old_physical)
        pool_values[new_physical * SOURCE_VALUE_BYTES:(new_physical + 1) * SOURCE_VALUE_BYTES] = \
            row.referenced_values[index * SOURCE_VALUE_BYTES:(index + 1) * SOURCE_VALUE_BYTES]
        pool_scales[new_physical * SOURCE_SCALE_BYTES:(new_physical + 1) * SOURCE_SCALE_BYTES] = \
            row.referenced_scales[index * SOURCE_SCALE_BYTES:(index + 1) * SOURCE_SCALE_BYTES]

    # --- oracle-verify the compact view -------------------------------------
    resolved_compact = _resolve(row, oracle, pages_compact, capacity)
    if not resolved_compact["whole_row_valid"]:
        raise MaterializeError(
            f"compact fixture for {row.key} is kernel-invalid "
            f"({resolved_compact['reason']})"
        )
    if resolved_compact["keyslot_count"] != resolved_orig["keyslot_count"]:
        raise MaterializeError("compact view changed the keyslot count")
    slots_compact = _evidence_slots(
        row, oracle, resolved_compact,
        pool=(bytes(pool_values), bytes(pool_scales)),
    )

    for orig, compact in zip(slots_orig, slots_compact):
        if compact["masked"] != orig["masked"]:
            raise MaterializeError(
                f"slot {orig['slot']}: compact mask changed "
                f"({orig['masked']} -> {compact['masked']})"
            )
        if compact["masked"]:
            continue
        for field_name in ("tag", "logical", "value_dtype", "scale_dtype",
                           "value_sha256", "scale_sha256"):
            if compact[field_name] != orig[field_name]:
                raise MaterializeError(
                    f"slot {orig['slot']}: compact {field_name} changed: "
                    f"{orig[field_name]!r} -> {compact[field_name]!r}"
                )
        if orig["tag"] == oracle.TAG_PAGED_SOURCE:
            new_physical = compact["physical"]
            if new_physical != physical_map[orig["physical"]]:
                raise MaterializeError(
                    f"slot {orig['slot']}: compact physical {new_physical} "
                    f"!= remapped {physical_map[orig['physical']]}"
                )
            if new_physical % 256 != orig["physical"] % 256:
                raise MaterializeError(
                    f"slot {orig['slot']}: offset mod 256 not preserved"
                )
            if not new_physical < capacity:
                raise MaterializeError(
                    f"slot {orig['slot']}: compact physical {new_physical} "
                    f"outside capacity {capacity}"
                )
        else:
            if compact["physical"] != orig["physical"]:
                raise MaterializeError(
                    f"slot {orig['slot']}: non-paged physical changed"
                )

    digest_compact = canonical_digest(
        resolved_compact["width"], row.compressed,
        hashlib.sha256(row.query_row).hexdigest(),
        hashlib.sha256(row.sink).hexdigest(),
        slots_compact,
    )
    captured_digest = row.evidence.get("canonical_digest")
    if captured_digest is not None and digest_compact != captured_digest:
        raise MaterializeError(
            f"compact canonical digest {digest_compact} != captured "
            f"{captured_digest}"
        )
    digest_orig = canonical_digest(
        resolved_orig["width"], row.compressed,
        hashlib.sha256(row.query_row).hexdigest(),
        hashlib.sha256(row.sink).hexdigest(),
        slots_orig,
    )
    if digest_orig != digest_compact:
        raise MaterializeError(
            "original and compact canonical digests differ; the page remap "
            "changed the kernel-visible content"
        )

    return MaterializedRow(
        source=row,
        source_capacity=capacity,
        pool_values=bytes(pool_values),
        pool_scales=bytes(pool_scales),
        pages_compact=pages_compact,
        page_map=page_map,
        physical_map=physical_map,
        slot_evidence_compact=slots_compact,
        canonical_digest_compact=digest_compact,
    )


# ---------------------------------------------------------------------------
# Counterfactual single-nibble mutation (CPU-only, exact).
# ---------------------------------------------------------------------------

def set_fp4_low_nibble(value_bytes: bytes, column: int, nibble: int) -> bytes:
    """Return ``value_bytes`` with one FP4 column's low nibble replaced."""
    if not 0 <= nibble <= 0xF:
        raise ValueError(f"nibble {nibble} out of range")
    byte_index, low = divmod(column, 2)
    if byte_index >= len(value_bytes):
        raise ValueError(f"column {column} outside {len(value_bytes)} bytes")
    if low != 0:
        raise ValueError(
            f"column {column} is a high nibble; the counterfactual site is "
            "the low nibble"
        )
    out = bytearray(value_bytes)
    out[byte_index] = (out[byte_index] & 0xF0) | nibble
    return bytes(out)


def mutate_private_source_column(
    row: CapturedRow,
    oracle,
    nibble: int,
    column: int = COUNTERFACTUAL_COLUMN,
) -> dict:
    """Clone a row changing ONLY the private-source FP4 low nibble.

    The private-source overlay slot (source proposal wave) column's low
    nibble is replaced; the high nibble and every other byte (including all
    scales) stay intact.  Returns the mutated row plus byte-level evidence of
    the single-nibble change.
    """
    resolved = _resolve(row, oracle, row.pages, row.source_capacity_rows)
    slots = _evidence_slots(row, oracle, resolved)
    private = None
    for entry in slots:
        if not entry["masked"] and entry["tag"] == oracle.TAG_PRIVATE_SOURCE:
            if private is not None:
                raise MaterializeError("multiple live private-source slots")
            private = entry
    if private is None:
        raise MaterializeError("no live private-source slot to mutate")
    proposal_index = private["physical"]
    before = _row(row.source_proposal_values, proposal_index,
                  SOURCE_VALUE_BYTES, "source proposal values")
    after = set_fp4_low_nibble(before, column, nibble)

    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    if len(changed) != 1:
        raise MaterializeError(
            f"mutation changed {len(changed)} bytes, expected exactly 1"
        )
    at = changed[0]
    if (before[at] & 0xF0) != (after[at] & 0xF0):
        raise MaterializeError("mutation touched the high nibble")
    if (after[at] & 0x0F) != nibble:
        raise MaterializeError("mutation did not set the requested nibble")
    if hashlib.sha256(after).hexdigest() == private["value_sha256"]:
        raise MaterializeError("mutation request reproduced the input bytes")

    mutated_source_proposal_values = (
        row.source_proposal_values[:proposal_index * SOURCE_VALUE_BYTES]
        + after
        + row.source_proposal_values[(proposal_index + 1) * SOURCE_VALUE_BYTES:]
    )
    if len(mutated_source_proposal_values) != len(row.source_proposal_values):
        raise MaterializeError("mutated proposal arena changed size")

    clone = CapturedRow(**{**row.__dict__, "source_proposal_values":
                           mutated_source_proposal_values})
    evidence = {
        "tag": "private_source",
        "proposal_row": proposal_index,
        "column": column,
        "byte_index": at,
        "before": before[at],
        "after": after[at],
        "changed_bytes": 1,
        "nibble": nibble,
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
        "scales_sha256": hashlib.sha256(
            _row(row.source_proposal_scales, proposal_index,
                 SOURCE_SCALE_BYTES, "source proposal scales")
        ).hexdigest(),
        "proposal_values_sha256": hashlib.sha256(
            mutated_source_proposal_values
        ).hexdigest(),
    }
    return {"row": clone, "evidence": evidence}
