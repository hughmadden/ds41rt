"""Structural validation of a planned experiment, before any device work.

``validate_experiment`` is a pure CPU check over the plan dataclasses: it never
allocates device memory, uploads bytes or calls native code, and it makes no
numeric claim.  It exists so that a mis-wired plan (a stage reading a role that
no earlier stage produces, an operand whose bytes disagree with its dtype/shape
or with the native contract, an output that would clobber an input, a baseline
without a full oracle) is rejected with an actionable error *before* the GPU
runner touches anything.

The per-stage byte contracts mirror the native signatures in ``ffi.py`` and the
fixed row extents in ``schema.py``; they are structural sizes only, never
guessed values.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

from .schema import (
    DTYPE_BYTES,
    LATENT_DIM,
    MAX_SLOTS,
    SOURCE_DIM,
)

# Native stage kind -> the exact read-role keys the stage must carry.
READ_ROLES: Dict[str, tuple] = {
    "project": ("input", "weight"),
    "pool": ("kv", "scores", "pending_kv", "pending_scores", "predecessors",
             "norm"),
    "pack": ("input", "frequencies"),
}

# Native stage kind -> the stage keys naming roles it writes.
OUTPUT_KEYS: Dict[str, tuple] = {
    "project": ("output",),
    "pool": ("output",),
    "pack": ("values", "scales"),
}

_FP32 = DTYPE_BYTES["float32"]
_BF16 = DTYPE_BYTES["bfloat16"]

# Operands are uploaded as raw bytes and only *viewed* as a dtype; uint8 is
# the runner's raw-upload view, so it has a fixed itemsize of one.
_DTYPE_ITEMSIZE = dict(DTYPE_BYTES)
_DTYPE_ITEMSIZE["uint8"] = 1


class ValidationError(ValueError):
    """A plan is structurally invalid; ``ValueError`` so the CLI catches it."""


def _positive_int(value, field: str, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(
            f"experiment {name}: {field} must be a positive integer, "
            f"got {value!r}"
        )
    return value


def _first_duplicate(values: Sequence[str]):
    seen = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _operand_blobs(experiment) -> Dict[str, bytes]:
    roles = [operand.role for operand in experiment.operands]
    duplicate = _first_duplicate(roles)
    if duplicate is not None:
        raise ValidationError(
            f"experiment {experiment.name}: duplicate operand role "
            f"{duplicate!r}; roles identify a single buffer each"
        )
    blobs: Dict[str, bytes] = {}
    for operand in experiment.operands:
        itemsize = _DTYPE_ITEMSIZE.get(operand.dtype)
        if itemsize is None:
            raise ValidationError(
                f"experiment {experiment.name}: operand {operand.role!r} has "
                f"dtype {operand.dtype!r} with no fixed itemsize; operand "
                "dtypes must be bfloat16, float32, uint64 or uint8"
            )
        shape = operand.shape or ()
        if any(isinstance(d, bool) or not isinstance(d, int) or d < 0
               for d in shape):
            raise ValidationError(
                f"experiment {experiment.name}: operand {operand.role!r} has "
                f"invalid shape {operand.shape!r}"
            )
        elements = 1
        for dim in shape:
            elements *= dim
        expected = elements * itemsize
        if expected != len(operand.blob):
            raise ValidationError(
                f"experiment {experiment.name}: operand {operand.role!r} "
                f"bytes {len(operand.blob)} != dtype {operand.dtype!r} "
                f"itemsize {itemsize} x shape {tuple(shape)} = {expected}"
            )
        blobs[operand.role] = operand.blob
    return blobs


def _read_extent(kind: str, role: str, rows: int, slots: int) -> int:
    """Contract byte extent for a stage's read role."""
    if kind == "project":
        if role == "input":
            return rows * SOURCE_DIM * _BF16
        if role == "weight":
            return LATENT_DIM * SOURCE_DIM * _BF16
    elif kind == "pool":
        if role in ("kv", "scores"):
            return rows * LATENT_DIM * _FP32
        if role in ("pending_kv", "pending_scores"):
            return slots * LATENT_DIM * _FP32
        if role == "predecessors":
            return rows * 8
        if role == "norm":
            return LATENT_DIM * _BF16
    elif kind == "pack":
        if role == "input":
            return rows * LATENT_DIM * _BF16
        if role == "frequencies":
            return rows * 32 * 2 * _FP32
    # Unreachable for the kinds/roles in READ_ROLES; kept explicit on purpose.
    raise ValidationError(
        f"unknown read role {role!r} for stage kind {kind!r}"
    )


def _output_extent(kind: str, role: str, rows: int) -> int:
    """Contract byte extent for a role produced by a stage."""
    if kind == "project":
        return rows * LATENT_DIM * _FP32
    if kind == "pool":
        return rows * LATENT_DIM * _BF16
    if kind == "pack":
        if role == "values":
            return rows * 256
        if role == "scales":
            return rows * 32
    raise ValidationError(
        f"unknown output role {role!r} for stage kind {kind!r}"
    )


def _allocated_output_bytes(experiment, role: str) -> int:
    """The executor's allocation size for a produced role (lazy import)."""
    from .runner import RunnerError, _output_size

    try:
        return _output_size(experiment, role)
    except RunnerError as error:
        raise ValidationError(
            f"experiment {experiment.name}: output role {role!r} has no "
            f"allocatable size: {error}"
        ) from error


def _validate_expected(experiment, produced: Iterable[str]) -> None:
    outputs = list(experiment.outputs)
    expected = experiment.expected or {}
    for role, blob in expected.items():
        if role not in outputs:
            raise ValidationError(
                f"experiment {experiment.name}: oracle role {role!r} is not "
                "a requested output; it would never be compared"
            )
        if not isinstance(blob, (bytes, bytearray)):
            raise ValidationError(
                f"experiment {experiment.name}: oracle {role!r} must be raw "
                f"bytes, got {type(blob).__name__}"
            )
        want = _allocated_output_bytes(experiment, role)
        if len(blob) != want:
            raise ValidationError(
                f"experiment {experiment.name}: oracle {role!r} has "
                f"{len(blob)} bytes but the output is {want} bytes"
            )
    if experiment.counterfactual:
        # A counterfactual is evidence, never a gate: an empty oracle is the
        # expected shape, and a partial one is tolerated but never required.
        return
    missing = [role for role in outputs if role not in expected]
    if missing:
        raise ValidationError(
            f"experiment {experiment.name}: non-counterfactual baseline is "
            f"missing an exact oracle for output(s) {missing}"
        )


def validate_experiment(experiment) -> None:
    """Reject a structurally invalid plan with an actionable ``ValidationError``.

    Checks, in order: positive rows/slots, unique operand roles, operand
    dtype*shape == byte length, unique outputs, and then each native stage's
    exact read roles (present in the stage and available as an operand or an
    output produced by an earlier stage), output roles (listed for allocation
    and not clobbering an operand or earlier output) and byte extents.  Finally
    it requires every requested output to be produced and every baseline output
    to carry an oracle of exactly the right byte count.
    """
    name = experiment.name
    rows = _positive_int(experiment.rows, "rows", name)
    slots = _positive_int(experiment.slots, "slots", name)
    if slots > MAX_SLOTS:
        raise ValidationError(
            f"experiment {name}: slots {slots} exceeds the native maximum "
            f"{MAX_SLOTS}"
        )

    operand_blobs = _operand_blobs(experiment)

    outputs = list(experiment.outputs)
    duplicate = _first_duplicate(outputs)
    if duplicate is not None:
        raise ValidationError(
            f"experiment {name}: duplicate output role {duplicate!r}"
        )

    available = set(operand_blobs)
    produced = set()

    for index, stage in enumerate(experiment.stages):
        kind = stage.get("kind")
        if kind not in READ_ROLES:
            raise ValidationError(
                f"experiment {name}: stage {index} has unknown kind {kind!r}"
            )
        stage_slots = stage.get("slots", slots)
        stage_slots = _positive_int(stage_slots, f"stage {index} slots", name)
        if stage_slots > MAX_SLOTS:
            raise ValidationError(
                f"experiment {name}: stage {index} slots {stage_slots} "
                f"exceeds the native maximum {MAX_SLOTS}"
            )

        # Read roles first: a stage's produced role is only registered once
        # every read it depends on has been validated.
        for role in READ_ROLES[kind]:
            if role not in stage:
                raise ValidationError(
                    f"experiment {name}: stage {index} ({kind}) is missing "
                    f"its read-role key {role!r}"
                )
            target = stage[role]
            if target not in available:
                raise ValidationError(
                    f"experiment {name}: stage {index} ({kind}) reads "
                    f"{role!r} -> {target!r}, which is neither an operand nor "
                    "produced by an earlier stage"
                )
            actual = (len(operand_blobs[target]) if target in operand_blobs
                      else _allocated_output_bytes(experiment, target))
            want = _read_extent(kind, role, rows, stage_slots)
            if actual != want:
                raise ValidationError(
                    f"experiment {name}: stage {index} ({kind}) read {role!r} "
                    f"-> {target!r} has {actual} bytes, expected {want}"
                )

        for key in OUTPUT_KEYS[kind]:
            if key not in stage:
                raise ValidationError(
                    f"experiment {name}: stage {index} ({kind}) is missing "
                    f"its output key {key!r}"
                )
            role = stage[key]
            if role in available:
                raise ValidationError(
                    f"experiment {name}: stage {index} ({kind}) output "
                    f"{role!r} would clobber an operand or an earlier output"
                )
            if role not in experiment.outputs:
                raise ValidationError(
                    f"experiment {name}: stage {index} ({kind}) produces "
                    f"{role!r}, which is not listed in experiment.outputs (it "
                    "would never be allocated or recorded)"
                )
            actual = _allocated_output_bytes(experiment, role)
            want = _output_extent(kind, role, rows)
            if actual != want:
                raise ValidationError(
                    f"experiment {name}: stage {index} ({kind}) output "
                    f"{role!r} is {actual} bytes, expected {want}"
                )
            available.add(role)
            produced.add(role)

    missing = [role for role in outputs if role not in produced]
    if missing:
        raise ValidationError(
            f"experiment {name}: requested output(s) {missing} are never "
            "produced by any stage"
        )

    _validate_expected(experiment, produced)
