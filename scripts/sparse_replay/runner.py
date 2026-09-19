"""GPU runner: replay planned cases through the pinned native library.

Imported lazily by the CLI only when ``--execute`` is given (torch is never
imported at package scope).  All inputs are uploaded as raw uint8 bytes and
only *viewed* as their kernel dtypes — no float casts anywhere.  Tensor
lifetimes are retained explicitly until every launch that references them
has synchronized.

No timing is recorded: repeats exist for stability, not performance claims.
Launch return codes (lifecycle) are recorded separately from byte-exact
output comparisons; a case "passes" only when every repeat produced rc 0 and
every output row is byte-identical to the captured attention-values row.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .capture import (
    QUERY_ROW_BYTES,
    METADATA_ROW_BYTES,
    SELECTED_ROW_BYTES,
    SINK_BYTES,
    RING_ROWS,
    RING_VALUE_BYTES,
    RING_SCALE_BYTES,
    WINDOW_VALUE_BYTES,
    WINDOW_SCALE_BYTES,
    SOURCE_VALUE_BYTES,
    SOURCE_SCALE_BYTES,
)
from .cases import Case, PlannedRow
from .materialize import (
    MaterializedRow,
    materialize,
    mutate_private_source_column,
    COUNTERFACTUAL_COLUMN,
)
from .pins import sha256_file
from .cases import COUNTERFACTUAL_NIBBLE

VIEW_BYTES = 120


class View(C.Structure):
    _fields_ = [
        ("values", C.c_void_p * 4),
        ("scales", C.c_void_p * 4),
        ("window_end", C.c_void_p),
        ("pages", C.c_void_p),
        ("source_end", C.c_void_p),
        ("window_capacity", C.c_uint64),
        ("source_capacity", C.c_uint64),
        ("source_proposal_capacity", C.c_uint64),
        ("page_stride", C.c_uint32),
        ("compressed", C.c_uint32),
    ]


assert C.sizeof(View) == VIEW_BYTES

# Exact native ABI, mirrored from native/include/ds41rt_v41_sparse_attention.h.
# Keep these in lockstep with the header: the batch validator takes *two*
# descriptor arguments (host then device), and only the launch entry point
# takes the stream.  A regression that passes the stream where the device
# descriptor buffer belongs makes the native side reject the batch before any
# kernel launch (the device span check fails).
BOUNDED_ARGTYPES = (
    [C.c_void_p] * 5 + [C.c_int32, C.c_int32, C.POINTER(View)]
    + [C.c_void_p] * 3 + [C.c_uint64, C.c_int32]
)
BATCH_VALIDATE_ARGTYPES = (
    [C.c_void_p] * 5 + [C.c_int32, C.POINTER(View)]
    + [C.c_void_p] * 3 + [C.c_uint64, C.c_int32, C.c_int32]
)
BATCH_ARGTYPES = (
    [C.c_void_p] * 5 + [C.c_int32] + [C.c_void_p] * 4 + [C.c_int32] * 2
)


class RunnerError(Exception):
    pass


def _bf16_diff(expected: bytes, actual: bytes) -> dict:
    """Byte/element diff stats between two BF16 rows (numpy, no torch)."""
    import numpy as np

    if len(expected) != len(actual):
        raise RunnerError("row length mismatch in comparison")
    e = np.frombuffer(expected, dtype=np.uint8)
    a = np.frombuffer(actual, dtype=np.uint8)
    byte_diffs = int(np.count_nonzero(e != a))
    e16 = np.frombuffer(expected, dtype=np.uint16)
    a16 = np.frombuffer(actual, dtype=np.uint16)
    element_diffs = int(np.count_nonzero(e16 != a16))
    e32 = e16.astype(np.uint32) << 16
    a32 = a16.astype(np.uint32) << 16
    ef = e32.view(np.float32)
    af = a32.view(np.float32)
    finite = np.isfinite(ef) & np.isfinite(af)
    max_abs = 0.0
    if np.any(finite):
        max_abs = float(np.max(np.abs(ef[finite] - af[finite])))
    return {
        "byte_diffs": byte_diffs,
        "element_diffs": element_diffs,
        "max_abs": max_abs,
        "actual_sha256": hashlib.sha256(actual).hexdigest(),
    }


@dataclass
class _Variant:
    """Uploaded device arrays for one (possibly counterfactual) row variant."""

    variant_id: str
    keepalive: list
    view: View
    source: object
    materialized: MaterializedRow
    mutation_evidence: dict or None


class _Session:
    def __init__(self, torch, native_path, expected_sha256, device=0):
        self.torch = torch
        native_path = Path(native_path)
        digest = sha256_file(native_path)
        if digest != expected_sha256:
            raise RunnerError(
                f"native library sha256 mismatch: expected {expected_sha256}, "
                f"got {digest}"
            )
        self.native_path = str(native_path)
        self.native_sha256 = digest
        torch.cuda.set_device(device)
        self.lib = C.CDLL(str(native_path))
        if self.lib.ds41rt_v41_sparse_attention_initialize() != 0:
            raise RunnerError("sparse attention initialize failed")
        self.bounded = self.lib.ds41rt_v41_sparse_attention_bounded
        self.bounded.argtypes = BOUNDED_ARGTYPES
        self.bounded.restype = C.c_int32
        self.batch_validate = self.lib.ds41rt_v41_sparse_attention_batch_validate
        self.batch_validate.argtypes = BATCH_VALIDATE_ARGTYPES
        self.batch_validate.restype = C.c_int32
        self.batch = self.lib.ds41rt_v41_sparse_attention_batch
        self.batch.argtypes = BATCH_ARGTYPES
        self.batch.restype = C.c_int32

    def _upload(self, blob: bytes, dtype=None, shape=None):
        """Upload raw bytes as uint8; optionally return a typed view."""
        host = self.torch.frombuffer(bytearray(blob), dtype=self.torch.uint8)
        dev = self.torch.empty(len(blob), dtype=self.torch.uint8, device="cuda")
        dev.copy_(host, non_blocking=False)
        if dtype is None:
            return dev
        return dev.view(dtype).view(shape) if shape is not None else dev.view(dtype)

    def build_variant(self, source_row, oracle, counterfactual) -> _Variant:
        """Upload one row variant (exact raw bytes; no float casts)."""
        t = self.torch
        mutation = None
        row = source_row
        if counterfactual is not None:
            nibble = COUNTERFACTUAL_NIBBLE[counterfactual]
            mutation = mutate_private_source_column(
                row, oracle, nibble, column=COUNTERFACTUAL_COLUMN
            )
            row = mutation["row"]
        mat = materialize(row, oracle)

        keepalive = []
        ring_values = self._upload(row.ring_values)
        ring_scales = self._upload(row.ring_scales)
        window_values = self._upload(row.window_proposal_values)
        window_scales = self._upload(row.window_proposal_scales)
        pool_values = self._upload(mat.pool_values)
        pool_scales = self._upload(mat.pool_scales)
        source_values = self._upload(row.source_proposal_values)
        source_scales = self._upload(row.source_proposal_scales)
        window_end = self._upload(
            int(row.window_end).to_bytes(8, "little")
        ).view(t.uint64)
        source_end = self._upload(
            int(row.source_end).to_bytes(8, "little")
        ).view(t.uint64)
        pages_blob = b"".join(
            int(p).to_bytes(4, "little") for p in mat.pages_compact
        )
        pages = self._upload(pages_blob).view(t.int32)
        keepalive += [
            ring_values, ring_scales, window_values, window_scales,
            pool_values, pool_scales, source_values, source_scales,
            window_end, source_end, pages,
        ]
        view = View(
            (C.c_void_p * 4)(
                ring_values.data_ptr(), window_values.data_ptr(),
                pool_values.data_ptr(), source_values.data_ptr(),
            ),
            (C.c_void_p * 4)(
                ring_scales.data_ptr(), window_scales.data_ptr(),
                pool_scales.data_ptr(), source_scales.data_ptr(),
            ),
            window_end.data_ptr(),
            pages.data_ptr(),
            source_end.data_ptr(),
            row.window_proposal_capacity,
            mat.source_capacity,
            row.source_proposal_capacity,
            row.page_stride,
            row.compressed,
        )
        return _Variant(
            variant_id=(
                f"{row.wave}:{row.request_id}"
                + (f":{counterfactual}" if counterfactual else "")
            ),
            keepalive=keepalive,
            view=view,
            source=row,
            materialized=mat,
            mutation_evidence=(mutation["evidence"] if mutation else None),
        )


def launch_batch(
    session,
    *,
    query,
    sink,
    metadata,
    selected,
    output,
    host_views,
    descriptors,
    stream,
    bounds,
    scratch,
    rows,
    parts,
    compressed,
):
    """Validate host descriptors, upload them, then launch the device batch.

    Extracted so the exact call path can be contract-tested on CPU with
    duck-typed buffers (only ``session.torch.frombuffer`` and the descriptor
    ``copy_`` are exercised).  The native validator receives the *device
    descriptor buffer* address -- never the stream handle -- and the host
    descriptors are uploaded only after validation succeeds, so a rejected
    batch performs no device upload and no launch.

    ``host_views`` is the ``(View * rows)`` array; its raw bytes are exactly
    ``rows * VIEW_BYTES`` (120 bytes per descriptor row).  The device buffer and
    the referenced allocations must stay alive through the launch.
    """
    status = session.batch_validate(
        query.data_ptr(), sink.data_ptr(), metadata.data_ptr(),
        selected.data_ptr(), output.data_ptr(), rows,
        host_views,
        descriptors.data_ptr(),
        bounds.data_ptr(), scratch.data_ptr(),
        scratch.numel() * 4, parts, compressed,
    )
    if status != 0:
        return status
    host_bytes = bytes(memoryview(host_views))
    upload_host = session.torch.frombuffer(
        bytearray(host_bytes), dtype=session.torch.uint8
    )
    descriptors.copy_(upload_host, non_blocking=False)
    return session.batch(
        query.data_ptr(), sink.data_ptr(), metadata.data_ptr(),
        selected.data_ptr(), output.data_ptr(), rows,
        descriptors.data_ptr(), stream, bounds.data_ptr(),
        scratch.data_ptr(), parts, compressed,
    )


def classify_case(warmup_rc, repeat_records):
    """Classify a case as pass / numerical_mismatch / unexecuted.

    A rejected warmup or launch means the kernels never ran, so the case is
    reported as *unexecuted* (unscored) and never as an arithmetic mismatch.
    """
    executed = warmup_rc == 0 and all(
        repeat["executed"] for repeat in repeat_records
    )
    numerical_pass = executed and all(
        row["byte_exact"]
        for repeat in repeat_records for row in repeat["rows"]
    )
    if not executed:
        status = "unexecuted"
    elif numerical_pass:
        status = "pass"
    else:
        status = "numerical_mismatch"
    return {
        "executed": executed,
        "numerical_pass": numerical_pass,
        "status": status,
        "unscored": not executed,
    }


def run_cases(cases, index, oracle, native_path, native_sha256, output_dir,
              repeats=3, device=0) -> dict:
    """Execute every case; write per-case raw outputs and a summary.

    Returns the machine summary dict (also written to output_dir/summary.json).
    """
    import torch

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise RunnerError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    session = _Session(torch, native_path, native_sha256, device=device)
    summary = {
        "native_library": session.native_path,
        "native_library_sha256": session.native_sha256,
        "repeats": repeats,
        "parts": 10,
        "counterfactual_column": COUNTERFACTUAL_COLUMN,
        "cases": [],
    }

    for case in cases:
        case_dir = output_dir / case.name
        case_dir.mkdir()
        # Build/upload one variant per distinct planned row.
        variants = []
        for planned in case.rows:
            variants.append(
                session.build_variant(index[planned.key], oracle,
                                      planned.counterfactual)
            )
        rows = case.row_count
        parts = case.parts
        t = torch

        sink = session._upload(index[case.rows[0].key].sink)
        query = session._upload(
            b"".join(v.source.query_row for v in variants)
        ).view(t.bfloat16).view(rows, 64, 512)
        metadata = session._upload(
            b"".join(
                b"".join(int(m).to_bytes(8, "little") for m in v.source.metadata10)
                for v in variants
            )
        ).view(t.uint64).view(rows, 10)
        if all(v.source.selected512 is not None for v in variants):
            selected = session._upload(
                b"".join(
                    b"".join(
                        int(s).to_bytes(4, "little", signed=True)
                        for s in v.source.selected512
                    )
                    for v in variants
                )
            ).view(t.int32).view(rows, 512)
        else:
            raise RunnerError("all replay rows must carry captured selected ids")
        bounds = session._upload(
            b"".join(int(v.source.replay_begin).to_bytes(8, "little") for v in variants)
        ).view(t.uint64)
        output = session._upload(b"\x00" * (rows * QUERY_ROW_BYTES)).view(
            t.bfloat16
        ).view(rows, 64, 512)
        scratch = t.empty((rows, parts, 64, 514), dtype=t.float32, device="cuda")
        keepalive = [sink, query, metadata, selected, bounds, output, scratch]

        stream = t.cuda.current_stream().cuda_stream
        descriptors = t.empty(rows * VIEW_BYTES, dtype=t.uint8, device="cuda")
        keepalive.append(descriptors)

        expected_rows = [
            index[planned.expected_key or planned.key].output_row
            for planned in case.rows
        ]

        def launch_single(row_pos):
            view = variants[row_pos].view
            return session.bounded(
                query[row_pos:].data_ptr(), sink.data_ptr(),
                metadata[row_pos:].data_ptr(), selected[row_pos:].data_ptr(),
                output[row_pos:].data_ptr(), 1, 0, C.byref(view),
                stream, bounds[row_pos:].data_ptr(), scratch.data_ptr(),
                scratch.numel() * 4, parts,
            )

        def launch_batch_case():
            return launch_batch(
                session,
                query=query, sink=sink, metadata=metadata,
                selected=selected, output=output,
                host_views=(View * rows)(*[v.view for v in variants]),
                descriptors=descriptors, stream=stream, bounds=bounds,
                scratch=scratch, rows=rows, parts=parts, compressed=2,
            )

        # Warmup launch + sync (probe pattern), discarded.  A non-zero rc is
        # recorded (lifecycle), not raised: comparison stays separate.
        output.zero_()
        warmup_rc = launch_single(0) if case.kind == "single" else launch_batch_case()
        t.cuda.synchronize()

        repeat_records = []
        for repeat in range(repeats):
            output.zero_()
            if case.kind == "single":
                rc = [launch_single(0)]
            else:
                rc = [launch_batch_case()]
            t.cuda.synchronize()
            # Raw output rows are always written, including the untouched-zero
            # buffer when a launch was rejected: the bytes are preserved for
            # inspection, but rejected repeats are reported as unexecuted
            # rather than as numerical mismatches.
            executed = all(code == 0 for code in rc)
            actual = output.view(t.uint8).cpu().numpy().tobytes()
            row_results = []
            for row_pos, expected in enumerate(expected_rows):
                got = actual[row_pos * QUERY_ROW_BYTES:(row_pos + 1) * QUERY_ROW_BYTES]
                stats = _bf16_diff(expected, got)
                row_results.append({
                    "position": row_pos,
                    "variant": variants[row_pos].variant_id,
                    "expected_output_sha256": hashlib.sha256(
                        expected
                    ).hexdigest(),
                    "executed": executed,
                    "unscored": not executed,
                    "status": (
                        "unexecuted" if not executed
                        else "byte_exact" if stats["byte_diffs"] == 0
                        else "numerical_mismatch"
                    ),
                    "numerical_mismatch": executed and stats["byte_diffs"] != 0,
                    # byte_exact is only a claim when the launch actually ran.
                    "byte_exact": executed and stats["byte_diffs"] == 0,
                    **stats,
                })
                (case_dir / f"{case.kind}_repeat{repeat}_row{row_pos}.bin").write_bytes(got)
            repeat_records.append({
                "repeat": repeat,
                "launch_rc": rc,
                "executed": executed,
                "status": "executed" if executed else "unexecuted",
                "rows": row_results,
            })

        outcome = classify_case(warmup_rc, repeat_records)
        case_record = {
            "name": case.name,
            "kind": case.kind,
            "rows": rows,
            "parts": parts,
            "variants": [
                {
                    "variant_id": v.variant_id,
                    "source_capacity": v.materialized.source_capacity,
                    "pool_bytes": v.materialized.pool_bytes,
                    "page_map": {str(k): val for k, val in sorted(v.materialized.page_map.items())},
                    "canonical_digest": v.materialized.canonical_digest_compact,
                    "mutation": v.mutation_evidence,
                }
                for v in variants
            ],
            "expected_row_outputs_identical": len({
                hashlib.sha256(r).hexdigest() for r in expected_rows
            }) == 1,
            "warmup_rc": warmup_rc,
            # A rejected warmup/launch is unexecuted, not a numerical mismatch;
            # the raw (often zero) output rows above are still preserved.
            "executed": outcome["executed"],
            "status": outcome["status"],
            "numerical_pass": outcome["numerical_pass"],
            "unscored": outcome["unscored"],
            "all_repeats_rc0_and_byte_exact": outcome["numerical_pass"],
            "repeats": repeat_records,
        }
        summary["cases"].append(case_record)
        (case_dir / "case.json").write_text(
            json.dumps(case_record, indent=2, sort_keys=True)
        )
        del keepalive, variants

    # process_rc 0 only means the runner itself completed; it is explicitly
    # not a numerical pass.  Unscored cases are rejected before/at launch.
    summary["process_rc"] = 0
    summary["process_rc_meaning"] = (
        "runner completed; process rc 0 alone is not a numerical pass"
    )
    summary["unscored_cases"] = [
        case["name"] for case in summary["cases"] if not case["executed"]
    ]
    summary["numerical_pass"] = all(
        case["numerical_pass"] for case in summary["cases"]
    )
    summary["status"] = (
        "pass" if summary["numerical_pass"]
        else "unexecuted" if summary["unscored_cases"]
        else "numerical_mismatch"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary
