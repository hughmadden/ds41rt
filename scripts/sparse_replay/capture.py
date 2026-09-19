"""Load captured v41 sparse-attention input rows (CPU only, raw bytes).

``load_capture_dir`` reuses the pinned extractor for validated evidence
(ordered key slots, per-slot byte hashes, canonical digest) and then reads the
raw record bytes through the extractor's own reviewed record validator.  All
arrays are kept as exact raw bytes; nothing is dequantized or float-cast.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Fixed native row geometries (bytes per physical row).
RING_ROWS = 128
RING_VALUE_BYTES = 512
RING_SCALE_BYTES = 16
VALUE_ELEMENTS = 512
WINDOW_VALUE_BYTES = 512
WINDOW_SCALE_BYTES = 16
SOURCE_VALUE_BYTES = 256
SOURCE_SCALE_BYTES = 32

QUERY_ROW_BYTES = 64 * 512 * 2  # bfloat16 [64,512]
SINK_BYTES = 64 * 4
METADATA_ROW_BYTES = 10 * 8
SELECTED_ROW_BYTES = 512 * 4

OUTPUT_RECORD = "layer{layer}-attention-values.bin"
OUTPUT_ROW_BYTES = QUERY_ROW_BYTES


class CaptureError(Exception):
    """Hard failure while loading a capture (never a silent skip)."""


@dataclass
class CapturedRow:
    """One query row of one captured request, with exact raw bytes."""

    wave: str
    manifest_path: Path
    manifest_sha256: str
    request_id: int
    flat_row: int
    local_row: int
    position: int
    launch_kind: str
    width: int
    replay_begin: int
    metadata10: tuple
    selected512: Optional[tuple]
    query_row: bytes
    sink: bytes
    window_end: int
    ring_values: bytes
    ring_scales: bytes
    window_proposal_capacity: int
    window_proposal_values: bytes
    window_proposal_scales: bytes
    compressed: int
    source_capacity_rows: int
    source_proposal_capacity: int
    page_stride: int
    pages: tuple
    source_end: int
    source_proposal_values: bytes
    source_proposal_scales: bytes
    referenced_count: int
    referenced_physical_rows: tuple
    logical_to_physical: dict
    referenced_values: bytes
    referenced_scales: bytes
    output_row: bytes
    output_sha256: str
    evidence: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.wave, self.request_id)

    def view_scalars(self, source_capacity: Optional[int] = None) -> dict:
        """Oracle ``view_scalar_dict`` for this row (original or compact)."""
        return {
            "window_proposal_capacity": self.window_proposal_capacity,
            "source_capacity": (
                self.source_capacity_rows
                if source_capacity is None
                else source_capacity
            ),
            "source_proposal_capacity": self.source_proposal_capacity,
            "page_stride": self.page_stride,
            "compressed": self.compressed,
        }


def _record_bytes(extractor, base: Path, rec: Any, role: str, **kwargs) -> bytes:
    loaded = extractor._load_record(base, rec, role=role, **kwargs)
    return extractor._read_range(loaded.path, 0, loaded.bytes)


def _scalar_u64(extractor, base: Path, rec: Any, role: str) -> int:
    if not isinstance(rec, dict) or not isinstance(rec.get("file"), str):
        raise CaptureError(f"{role}: scalar record is malformed")
    path = extractor._resolve_member(base, rec["file"])
    data = extractor._read_range(path, 0, 8)
    if path.stat().st_size != 8:
        raise CaptureError(f"{role}: scalar file {path.name} is not 8 bytes")
    value = int.from_bytes(data, "little", signed=False)
    declared = rec.get("value")
    if isinstance(declared, int) and declared != value:
        raise CaptureError(
            f"{role}: device scalar {value} != manifest {declared}"
        )
    return value


def _load_output_row(manifest: dict, base: Path, flat_row: int, extractor) -> bytes:
    layer = manifest.get("layer")
    rows = manifest["rows"]
    path = base / OUTPUT_RECORD.format(layer=layer)
    if not path.is_file():
        raise CaptureError(f"captured attention output missing: {path.name}")
    size = path.stat().st_size
    if size != rows * OUTPUT_ROW_BYTES:
        raise CaptureError(
            f"{path.name} is {size} bytes; expected rows*65536 "
            f"({rows * OUTPUT_ROW_BYTES})"
        )
    return extractor._read_range(path, flat_row * OUTPUT_ROW_BYTES, OUTPUT_ROW_BYTES)


def load_capture_dir(
    manifest_path: Any,
    request_id: int,
    extractor,
) -> CapturedRow:
    """Load one request row from one capture wave directory.

    ``extractor`` is the pinned extractor module (see ``pins.load_extractor``).
    The extractor's ``extract_row`` performs the full record validation and
    returns the ordered-slot evidence; raw bytes are then read through the
    extractor's own ``_load_record``/``_read_range`` helpers so extents are
    checked twice, independently.
    """
    manifest_file = extractor._manifest_path(manifest_path)
    base = manifest_file.resolve().parent
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest = extractor._validate_manifest(manifest)
    extractor._validate_request_layout(manifest)

    slot, request = extractor._select_request(manifest, request_id)
    flat_range = request["flattened_row_range"]
    if request["rows"] != 1 or flat_range[1] - flat_range[0] != 1:
        raise CaptureError(
            f"request {request_id} spans {request['rows']} rows; "
            "the replay harness loads exactly one row per request"
        )
    flat_row = flat_range[0]
    rows = manifest["rows"]
    if not 0 <= flat_row < rows:
        raise CaptureError(f"flat row {flat_row} outside {rows} captured rows")

    # Validated evidence (records, scalars, oracle slots, canonical digest).
    evidence = extractor.extract_row(
        manifest_file, request_id, flat_row, row_mode="flat"
    )
    if not evidence["whole_row_valid"]:
        raise CaptureError(
            f"request {request_id} row is kernel-invalid "
            f"({evidence['invalid_reason']}); replay of invalid rows is "
            "out of scope for this harness"
        )

    wave = base.name
    launch_kind = manifest["launch_kind"]
    wave_rec = manifest["wave"]

    query_row = _record_bytes(
        extractor, base, wave_rec["query"], role="wave.query",
        dtype="bfloat16", shape=[rows, 64, 512], row_bytes=QUERY_ROW_BYTES,
        byte_count=rows * QUERY_ROW_BYTES,
    )
    query_row = query_row[flat_row * QUERY_ROW_BYTES:(flat_row + 1) * QUERY_ROW_BYTES]
    sink = _record_bytes(
        extractor, base, wave_rec["sink"], role="wave.sink",
        dtype="float32", shape=[64], row_bytes=4, byte_count=SINK_BYTES,
    )
    metadata = _record_bytes(
        extractor, base, wave_rec["metadata"], role="wave.metadata",
        dtype="uint64", shape=[rows, 10], row_bytes=METADATA_ROW_BYTES,
        byte_count=rows * METADATA_ROW_BYTES,
    )
    metadata10 = tuple(
        extractor._ints_from(
            metadata[flat_row * METADATA_ROW_BYTES:(flat_row + 1) * METADATA_ROW_BYTES],
            8, False, "metadata row",
        )
    )
    selected = None
    if wave_rec.get("selected") is not None:
        selected_bytes = _record_bytes(
            extractor, base, wave_rec["selected"], role="wave.selected",
            dtype="int32", shape=[rows, 512], row_bytes=SELECTED_ROW_BYTES,
            byte_count=rows * SELECTED_ROW_BYTES,
        )
        selected = tuple(
            extractor._ints_from(
                selected_bytes[flat_row * SELECTED_ROW_BYTES:(flat_row + 1) * SELECTED_ROW_BYTES],
                4, True, "selected row",
            )
        )

    replay_shape = [rows] if launch_kind == "descriptor_batch" else [request["rows"]]
    _, replay_values = extractor._load_replay(
        base, request.get("replay_begins"), role="request.replay_begins",
        expected_shape=replay_shape,
    )
    replay_index = flat_row if launch_kind == "descriptor_batch" else 0
    replay_begin = replay_values[replay_index]

    window = request["window"]
    window_end = _scalar_u64(extractor, base, window["window_end"], "window.window_end")
    ring_values = _record_bytes(
        extractor, base, window["ring_values"], role="window.ring_values",
        dtype="fp8e4m3", shape=[RING_ROWS, RING_VALUE_BYTES],
        row_bytes=RING_VALUE_BYTES,
    )
    ring_scales = _record_bytes(
        extractor, base, window["ring_scales"], role="window.ring_scales",
        dtype="uint8-e8m0", shape=[RING_ROWS, RING_SCALE_BYTES],
        row_bytes=RING_SCALE_BYTES,
    )
    win_cap = window["proposal_capacity"]
    window_proposal_values = _record_bytes(
        extractor, base, window["proposal_values"], role="window.proposal_values",
        dtype="fp8e4m3", shape=[win_cap, WINDOW_VALUE_BYTES],
        row_bytes=WINDOW_VALUE_BYTES,
    )
    window_proposal_scales = _record_bytes(
        extractor, base, window["proposal_scales"], role="window.proposal_scales",
        dtype="uint8-e8m0", shape=[win_cap, WINDOW_SCALE_BYTES],
        row_bytes=WINDOW_SCALE_BYTES,
    )

    source = request.get("source")
    if source is None:
        raise CaptureError("capture carries no compressed source view")
    compressed = source["format"]
    if compressed != 2:
        raise CaptureError(
            f"source format {compressed} != 2 (FP4 K16); this harness only "
            "replays the FP4 compressed captures"
        )
    source_end = _scalar_u64(extractor, base, source["source_end"], "source.source_end")
    page_stride = source["page_stride"]
    pages_bytes = _record_bytes(
        extractor, base, source["pages"], role="source.pages",
        dtype="uint32", shape=[page_stride], row_bytes=4,
        byte_count=page_stride * 4,
    )
    pages = tuple(extractor._ints_from(pages_bytes, 4, False, "pages"))
    src_cap = source["proposal_capacity"]
    source_proposal_values = _record_bytes(
        extractor, base, source["proposal_values"], role="source.proposal_values",
        dtype="fp4e2m1", shape=[src_cap, VALUE_ELEMENTS],
        row_bytes=SOURCE_VALUE_BYTES,
    )
    source_proposal_scales = _record_bytes(
        extractor, base, source["proposal_scales"], role="source.proposal_scales",
        dtype="fp8e4m3", shape=[src_cap, SOURCE_SCALE_BYTES],
        row_bytes=SOURCE_SCALE_BYTES,
    )
    referenced = source["referenced"]
    physical_rows = tuple(referenced["physical_rows"])
    count = referenced["count"]
    if count != len(physical_rows):
        raise CaptureError("referenced.count != len(physical_rows)")
    referenced_values = _record_bytes(
        extractor, base, referenced["values"], role="source.referenced.values",
        dtype="fp4e2m1", shape=[count, VALUE_ELEMENTS],
        row_bytes=SOURCE_VALUE_BYTES, byte_count=count * SOURCE_VALUE_BYTES,
    )
    referenced_scales = _record_bytes(
        extractor, base, referenced["scales"], role="source.referenced.scales",
        dtype="fp8e4m3", shape=[count, SOURCE_SCALE_BYTES],
        row_bytes=SOURCE_SCALE_BYTES, byte_count=count * SOURCE_SCALE_BYTES,
    )
    logical_to_physical = {}
    for pair in referenced["logical_to_physical"]:
        logical_to_physical[int(pair["logical"])] = int(pair["physical"])

    output_row = _load_output_row(manifest, base, flat_row, extractor)

    return CapturedRow(
        wave=wave,
        manifest_path=manifest_file,
        manifest_sha256=evidence["manifest_sha256"],
        request_id=request_id,
        flat_row=flat_row,
        local_row=0,
        position=evidence["position"],
        launch_kind=launch_kind,
        width=request.get("width", 0),
        replay_begin=replay_begin,
        metadata10=metadata10,
        selected512=selected,
        query_row=query_row,
        sink=sink,
        window_end=window_end,
        ring_values=ring_values,
        ring_scales=ring_scales,
        window_proposal_capacity=win_cap,
        window_proposal_values=window_proposal_values,
        window_proposal_scales=window_proposal_scales,
        compressed=compressed,
        source_capacity_rows=source["capacity_rows"],
        source_proposal_capacity=src_cap,
        page_stride=page_stride,
        pages=pages,
        source_end=source_end,
        source_proposal_values=source_proposal_values,
        source_proposal_scales=source_proposal_scales,
        referenced_count=count,
        referenced_physical_rows=physical_rows,
        logical_to_physical=logical_to_physical,
        referenced_values=referenced_values,
        referenced_scales=referenced_scales,
        output_row=output_row,
        output_sha256=hashlib.sha256(output_row).hexdigest(),
        evidence=evidence,
    )


def load_activations(activations_dir: Any, extractor) -> dict:
    """Load every capture wave under an activations directory.

    Returns ``{(wave, request_id): CapturedRow}``.  Every wave sink must be
    byte-identical (the batched kernels share one sink buffer).  Every row's
    captured metadata position must match its provenance position (the
    extractor already enforces this; the wave set is cross-checked here).
    """
    base = Path(activations_dir)
    if not base.is_dir():
        raise CaptureError(f"activations directory not found: {base}")
    index = {}
    sink_sha = None
    for manifest in sorted(base.glob("*/layer*-attention-inputs.json")):
        manifest_obj = json.loads(manifest.read_bytes())
        for request in manifest_obj.get("requests", []):
            row = load_capture_dir(manifest, request["request_id"], extractor)
            if row.key in index:
                raise CaptureError(f"duplicate capture key {row.key}")
            row_sink_sha = hashlib.sha256(row.sink).hexdigest()
            if sink_sha is None:
                sink_sha = row_sink_sha
            elif row_sink_sha != sink_sha:
                raise CaptureError(
                    f"wave {row.wave} sink differs; the replay batches share "
                    "one sink buffer and require identical sinks"
                )
            index[row.key] = row
    if not index:
        raise CaptureError(f"no captures found under {base}")
    return index
