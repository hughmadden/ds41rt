"""GPU runner: replay planned compressor experiments through the native library.

Imported lazily by the CLI only when ``--execute`` is given (torch is never
imported at package scope).  All operands are uploaded as raw uint8 bytes and
only *viewed* as their kernel dtypes — no float casts anywhere.  One
compressor handle/workspace (4 MiB, 256-byte aligned) is created per session
and serialized on one stream; every buffer stays alive until the launches
that reference it have synchronized.

Lifecycle return codes are recorded separately from byte-exact comparisons:
a rejected launch makes its stage (and every dependent stage) UNSCORED — no
download, no comparison, and no claim of any kind about the bytes.

No timing is recorded: repeats exist for stability, not performance claims.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ffi import (
    WORKSPACE_ALIGNMENT,
    WORKSPACE_BYTES,
    NativeBindings,
    check_workspace_pointer,
    verify_native_library,
)

DTYPE_NAMES = ("bfloat16", "float32", "uint64", "uint8")


class RunnerError(Exception):
    """Harness/runtime failure (distinct from a native stage rc)."""


# ---------------------------------------------------------------------------
# Session: one serialized handle/workspace per stream.
# ---------------------------------------------------------------------------

class CompressorSession:
    """Owns the native compressor handle, its workspace and the stream."""

    def __init__(self, torch, bindings: NativeBindings, device: int = 0):
        self.torch = torch
        self.bindings = bindings
        self.device = device
        self._closed = False
        if hasattr(torch, "cuda"):
            torch.cuda.set_device(device)
        self.workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8,
                                     device="cuda")
        address = self.workspace.data_ptr()
        check_workspace_pointer(address)
        handle = C.c_void_p()
        rc = bindings.create(address, WORKSPACE_BYTES, C.byref(handle))
        if rc != 0:
            raise RunnerError(
                f"compressor_create rc {rc}: workspace must be a live "
                f"{WORKSPACE_BYTES}-byte {WORKSPACE_ALIGNMENT}-aligned buffer"
            )
        if not handle.value:
            raise RunnerError("compressor_create returned a null handle")
        self.handle = handle.value
        if hasattr(torch, "cuda"):
            self.stream = torch.cuda.current_stream().cuda_stream
        else:  # pragma: no cover - CPU fakes provide their own stream
            self.stream = None

    def synchronize(self) -> None:
        if hasattr(self.torch, "cuda"):
            self.torch.cuda.synchronize()

    def close(self) -> None:
        """Synchronize before destroy (stream-ordered work must land first)."""
        if self._closed:
            return
        self._closed = True
        self.synchronize()
        rc = self.bindings.destroy(self.handle)
        self.handle = None
        self.workspace = None
        if rc != 0:
            raise RunnerError(f"compressor_destroy rc {rc}")

    def __enter__(self) -> "CompressorSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_session(torch, native_path, expected_sha256, device: int = 0
                 ) -> CompressorSession:
    digest = verify_native_library(native_path, expected_sha256)
    lib = C.CDLL(str(native_path))
    bindings = NativeBindings.from_library(lib, path=str(native_path),
                                           sha256=digest)
    return CompressorSession(torch, bindings, device=device)


# ---------------------------------------------------------------------------
# Raw-byte upload: bits are preserved, dtypes are views only.
# ---------------------------------------------------------------------------

def upload(session, blob: bytes, dtype: str = None, shape=None):
    """Upload exact raw bytes as uint8; optionally return a dtype *view*.

    This is the only path device operands take: the byte string that left the
    capture file is the byte string the kernel receives.
    """
    torch = session.torch
    host = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    dev = torch.empty(len(blob), dtype=torch.uint8, device="cuda")
    dev.copy_(host, non_blocking=False)
    if dtype is None:
        return dev
    if dtype not in DTYPE_NAMES:
        raise RunnerError(f"unknown dtype view {dtype!r}")
    view = dev.view(getattr(torch, dtype))
    return view.view(shape) if shape is not None else view


def download(session, tensor) -> bytes:
    """Device-to-host copy of the exact raw bits (caller synchronizes first)."""
    return tensor.view(session.torch.uint8).cpu().numpy().tobytes()


# ---------------------------------------------------------------------------
# Launch helpers — the exact native call paths, stub-testable on CPU.
# ---------------------------------------------------------------------------

def _ptr(tensor):
    return tensor.data_ptr() if tensor is not None else None


def launch_project(session, *, input_t, weight_t, output_t, rows: int,
                   ratio: int, stream) -> int:
    """ds41rt_v41_compressor_project: BF16 [rows,5120] x BF16 [512,5120].

    ratio=2 -> FP32 [rows,512] output (WKV and Wgate both use this path at
    ratio two; the weight pointer selects which projection runs).
    """
    return session.bindings.project(
        session.handle, _ptr(input_t), _ptr(weight_t), _ptr(output_t),
        rows, ratio, stream,
    )


def launch_pool(session, *, kv_t, scores_t, pending_kv_t, pending_scores_t,
                predecessors_t, norm_t, output_t, rows: int, slots: int,
                stream) -> int:
    """ds41rt_v41_compressor_pool.

    Pointer roles (see ffi.PoolRoles): the predecessor *device descriptor*
    buffer sits in slot 4 and the stream in slot 9 — the recorded sparse-replay
    pitfall was passing the stream where a descriptor/metadata buffer belongs,
    which the native span checks reject before any kernel launch.
    """
    return session.bindings.pool(
        _ptr(kv_t), _ptr(scores_t), _ptr(pending_kv_t), _ptr(pending_scores_t),
        _ptr(predecessors_t), _ptr(norm_t), _ptr(output_t), rows, slots, stream,
    )


def launch_pack(session, *, input_t, frequencies_t, values_t, scales_t,
                rows: int, stream) -> int:
    """ds41rt_v41_compressed_kv_pack: BF16 [rows,512] -> FP4 [rows,256] +
    E4M3 scales [rows,32].  ``frequencies_t=None`` passes a NULL pointer."""
    return session.bindings.pack(
        _ptr(input_t), _ptr(frequencies_t), _ptr(values_t), _ptr(scales_t),
        rows, stream,
    )


@dataclass
class StageRecord:
    stage: str
    rc: int
    dependent_skipped: bool = False


def run_pool_pack(session, *, kv_t, scores_t, pending_kv_t, pending_scores_t,
                  predecessors_t, norm_t, frequencies_t, output_t, values_t,
                  scales_t, rows: int, slots: int, stream) -> dict:
    """Pool then pack on one stream; rc failure stops the dependent chain.

    Ordering contract (stub-tested on CPU):
      * pack is launched only when pool returned rc 0;
      * downloads happen only when every launch returned rc 0;
      * the raw output bytes are never compared after a rejected launch.
    """
    stages = []
    pool_rc = launch_pool(
        session, kv_t=kv_t, scores_t=scores_t, pending_kv_t=pending_kv_t,
        pending_scores_t=pending_scores_t, predecessors_t=predecessors_t,
        norm_t=norm_t, output_t=output_t, rows=rows, slots=slots, stream=stream,
    )
    stages.append(StageRecord("pool", pool_rc).__dict__)
    if pool_rc != 0:
        stages.append(StageRecord("pack", 0, dependent_skipped=True).__dict__)
        return {"stages": stages, "executed": False, "outputs": None}
    pack_rc = launch_pack(
        session, input_t=output_t, frequencies_t=frequencies_t,
        values_t=values_t, scales_t=scales_t, rows=rows, stream=stream,
    )
    stages.append(StageRecord("pack", pack_rc).__dict__)
    if pack_rc != 0:
        return {"stages": stages, "executed": False, "outputs": None}
    session.synchronize()
    return {
        "stages": stages,
        "executed": True,
        "outputs": {
            "output": download(session, output_t),
            "values": download(session, values_t),
            "scales": download(session, scales_t),
        },
    }


def run_projection(session, *, input_t, wkv_t, wgate_t, projected_out,
                   scores_out, rows: int, ratio: int, stream) -> dict:
    """Project the captured input through WKV and Wgate at the original batch
    size.  The two GEMMs are independent (either may run without the other),
    so both rcs are recorded and the outputs download only when both are 0.
    """
    stages = []
    rc_wkv = launch_project(
        session, input_t=input_t, weight_t=wkv_t, output_t=projected_out,
        rows=rows, ratio=ratio, stream=stream,
    )
    stages.append(StageRecord("project_wkv", rc_wkv).__dict__)
    rc_wgate = launch_project(
        session, input_t=input_t, weight_t=wgate_t, output_t=scores_out,
        rows=rows, ratio=ratio, stream=stream,
    )
    stages.append(StageRecord("project_wgate", rc_wgate).__dict__)
    if rc_wkv != 0 or rc_wgate != 0:
        return {"stages": stages, "executed": False, "outputs": None}
    session.synchronize()
    return {
        "stages": stages,
        "executed": True,
        "outputs": {
            "projected": download(session, projected_out),
            "scores": download(session, scores_out),
        },
    }


# ---------------------------------------------------------------------------
# Experiment execution.
# ---------------------------------------------------------------------------

@dataclass
class PlannedOperand:
    role: str
    blob: bytes
    dtype: str
    shape: tuple
    source: str                      # provenance, e.g. "capture:x:projected row1"


@dataclass
class PlannedExperiment:
    name: str
    kind: str                        # A/B/C/D
    rows: int
    slots: int
    ratio: int = 2
    counterfactual: bool = False
    operands: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    outputs: tuple = ()              # roles whose raw bytes are recorded
    expected: dict = field(default_factory=dict)   # role -> oracle bytes
    notes: tuple = ()


def _check_role_aliasing(experiment: PlannedExperiment, tensors: dict) -> None:
    """Distinct produced roles must never share a buffer address.

    A produced role aliasing a read-only operand would let a launch overwrite
    the operand before a later stage reads it; the same address under two
    different role names is always a harness bug, never a kernel contract.
    """
    address_roles: dict = {}
    for role, tensor in tensors.items():
        address = tensor.data_ptr()
        previous = address_roles.get(address)
        if previous is not None and previous != role:
            raise RunnerError(
                f"experiment {experiment.name}: roles {previous!r} and "
                f"{role!r} alias the same buffer {address:#x}"
            )
        address_roles[address] = role


def _first_nonzero_stage(stages) -> Optional[dict]:
    """The first stage whose recorded rc is a nonzero native failure."""
    for stage in stages:
        rc = stage.get("rc")
        if rc is not None and rc != 0:
            return stage
    return None


def execute_experiment(session, experiment: PlannedExperiment, exp_dir: Path,
                       repeats: int = 3) -> dict:
    """Warm up once (probe, discarded) then run ``repeats`` recorded runs.

    A rejected launch stops the dependent chain, suppresses every download,
    byte comparison and output file write for that run, and is reported
    unexecuted/unscored.  Nothing zero-filled ever stands in for a kernel
    result: raw output bytes are recorded only from runs whose stages all
    returned rc 0.
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=False)

    tensors = {}
    keepalive = []
    for operand in experiment.operands:
        tensor = upload(session, operand.blob, operand.dtype, operand.shape)
        tensors[operand.role] = tensor
        keepalive.append(tensor)
    operand_roles = {operand.role for operand in experiment.operands}
    for role in experiment.outputs:
        if role not in tensors:
            # Outputs are allocated zeroed; they are not operands.
            size = _output_size(experiment, role)
            tensor = upload(session, b"\x00" * size)
            tensors[role] = tensor
            keepalive.append(tensor)
    _check_role_aliasing(experiment, tensors)

    def launch_all():
        stages = []
        blocked = False
        for stage in experiment.stages:
            if blocked:
                stages.append({"stage": stage["kind"], "rc": None,
                               "dependent_skipped": True})
                continue
            rc = _launch_stage(session, experiment, tensors, stage)
            stages.append({"stage": stage["kind"], "rc": rc,
                           "dependent_skipped": False})
            if rc != 0:
                blocked = True
        return stages

    def zero_outputs():
        # Never erase a supplied read-only operand: only produced roles that
        # are not themselves operands of this experiment are cleared.
        for role in experiment.outputs:
            if role in operand_roles:
                continue
            tensors[role].zero_()

    def run_once(repeat: int) -> dict:
        zero_outputs()
        stages = launch_all()
        session.synchronize()
        executed = all(s["rc"] == 0 for s in stages if s["rc"] is not None)
        if not executed:
            failure = _first_nonzero_stage(stages)
            return {
                "repeat": repeat,
                "stages": stages,
                "executed": False,
                "unscored": True,
                "failed_stage": failure["stage"] if failure else None,
                "roles": {},                 # no download, no comparison
                "raw_outputs_written": False,
            }
        actual = {}
        for role in experiment.outputs:
            actual[role] = download(session, tensors[role])
            (exp_dir / f"{experiment.name}_repeat{repeat}_{role}.bin"
             ).write_bytes(actual[role])
        role_results = {}
        for role, expected in experiment.expected.items():
            got = actual.get(role)
            exact = got == expected
            role_results[role] = {
                "expected_sha256": hashlib.sha256(expected).hexdigest(),
                "actual_sha256": (
                    hashlib.sha256(got).hexdigest() if got is not None else None
                ),
                "byte_exact": exact,
                "compared": True,
            }
        return {
            "repeat": repeat,
            "stages": stages,
            "executed": True,
            "unscored": False,
            "failed_stage": None,
            "roles": role_results,
            "raw_outputs_written": True,
        }

    zero_outputs()
    warmup_stages = launch_all()
    session.synchronize()
    warmup_rc = [s["rc"] for s in warmup_stages if s["rc"] is not None]
    warmup_ok = bool(warmup_rc) and all(rc == 0 for rc in warmup_rc)
    warmup_failure = _first_nonzero_stage(warmup_stages)

    repeat_records = []
    for repeat in range(repeats):
        if not warmup_ok:
            # The probe launch was rejected: do not run dependent repeats, do
            # not download or write any numeric output.
            repeat_records.append({
                "repeat": repeat, "stages": [],
                "executed": False, "unscored": True,
                "failed_stage": (
                    warmup_failure["stage"] if warmup_failure else None
                ),
                "roles": {}, "raw_outputs_written": False,
            })
            continue
        repeat_records.append(run_once(repeat))

    executed = warmup_ok and bool(repeat_records) and all(
        r["executed"] for r in repeat_records
    )
    if not executed:
        # A rejected launch is always unexecuted/unscored, for counterfactual
        # and gating experiments alike -- never a numerical mismatch or pass.
        status = "unexecuted"
    elif experiment.counterfactual:
        # Counterfactuals are evidence, never gates: no accuracy claim and no
        # oracle comparison (their expected dict is empty by construction).
        status = "executed"
    elif repeat_records and all(
        r["roles"] and all(rr["byte_exact"] for rr in r["roles"].values())
        for r in repeat_records
    ):
        status = "pass"
    else:
        status = "numerical_mismatch"

    record = {
        "name": experiment.name,
        "kind": experiment.kind,
        "counterfactual": experiment.counterfactual,
        "rows": experiment.rows,
        "slots": experiment.slots,
        "repeats": repeats,
        "warmup_stages": warmup_stages,
        "warmup_rejected": not warmup_ok,
        "executed": executed,
        "status": status,
        "gates": not experiment.counterfactual,
        "operand_hashes": {
            op.role: hashlib.sha256(op.blob).hexdigest()
            for op in experiment.operands
        },
        "operand_sources": {op.role: op.source for op in experiment.operands},
        "notes": experiment.notes,
        "repeat_records": repeat_records,
    }
    (exp_dir / f"{experiment.name}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)
    )
    return record


def _output_size(experiment: PlannedExperiment, role: str) -> int:
    sizes = {
        "output": experiment.rows * 512 * 2,
        "values": experiment.rows * 256,
        "scales": experiment.rows * 32,
        "projected": experiment.rows * 512 * 4,
        "scores": experiment.rows * 512 * 4,
    }
    if role not in sizes:
        raise RunnerError(f"unknown output role {role!r}")
    return sizes[role]


def _launch_stage(session, experiment, tensors, stage: dict) -> int:
    stream = session.stream
    kind = stage["kind"]
    if kind == "project":
        return launch_project(
            session, input_t=tensors[stage["input"]],
            weight_t=tensors[stage["weight"]],
            output_t=tensors[stage["output"]], rows=experiment.rows,
            ratio=stage.get("ratio", experiment.ratio), stream=stream,
        )
    if kind == "pool":
        return launch_pool(
            session, kv_t=tensors[stage["kv"]], scores_t=tensors[stage["scores"]],
            pending_kv_t=tensors[stage["pending_kv"]],
            pending_scores_t=tensors[stage["pending_scores"]],
            predecessors_t=tensors[stage["predecessors"]],
            norm_t=tensors[stage["norm"]], output_t=tensors[stage["output"]],
            rows=experiment.rows, slots=stage.get("slots", experiment.slots),
            stream=stream,
        )
    if kind == "pack":
        frequencies = tensors.get(stage.get("frequencies"))
        return launch_pack(
            session, input_t=tensors[stage["input"]],
            frequencies_t=frequencies, values_t=tensors[stage["values"]],
            scales_t=tensors[stage["scales"]], rows=experiment.rows,
            stream=stream,
        )
    raise RunnerError(f"unknown stage kind {kind!r}")


def run_experiments(experiments, session: CompressorSession, output_dir: Path,
                    repeats: int = 3) -> dict:
    """Execute every planned experiment; write raw outputs plus a summary.

    The summary's ``process_rc`` only means the runner completed; the
    component baseline gate is ``component_baseline_pass``: every non-
    counterfactual experiment executed all stages with rc 0 on every repeat
    and every compared role was byte-exact.
    """
    output_dir = Path(output_dir)
    records = []
    for experiment in experiments:
        records.append(
            execute_experiment(session, experiment,
                               output_dir / experiment.name, repeats=repeats)
        )
    gated = [r for r in records if r["gates"]]
    unscored = [r["name"] for r in gated if not r["executed"]]
    mismatched = [r["name"] for r in gated if r["executed"]
                  and r["status"] != "pass"]
    summary = {
        "experiments": [r["name"] for r in records],
        "records": records,
        "counterfactual_experiments": [r["name"] for r in records
                                       if r["counterfactual"]],
        "process_rc": 0,
        "process_rc_meaning": (
            "runner completed; process rc 0 alone is not a numerical pass"
        ),
        "component_baseline_pass": bool(gated) and not unscored and not mismatched,
        "unscored_experiments": unscored,
        "numerical_mismatch_experiments": mismatched,
        "status": (
            "pass" if gated and not unscored and not mismatched
            else "unexecuted" if unscored
            else "numerical_mismatch" if mismatched
            else "no_gated_experiments"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary
