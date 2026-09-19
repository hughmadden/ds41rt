"""Focused CPU tests for structural plan validation.

These cover the positive path (the full actual-style A/B/C/D/P plan validates
every experiment), the negative cases (missing projected->kv read, shape/bytes
mismatch, duplicate roles, a stage reading a future output, an output clobbering
an operand, a missing or invalid oracle) and the guarantee that rejection
happens before any FakeGPU activity.  The corrected row-case planners now emit
complete C/D bundles, so the full CLI plan succeeds and the reference-selection
rule is recorded.  No torch, GPU or native call is made.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compressor_replay_fakes as fakes  # noqa: E402

from compressor_replay import cli, experiments, fixtures, runner, schema  # noqa: E402
from compressor_replay.validation import ValidationError  # noqa: E402

ACTUAL_CAPTURE = Path(
    "/home/turq/.cache/afd-dsh-workers-20260919/"
    "flash-cap16-p2-compressor-diagnostic-01/activations"
)

# All synthetic captures share one p41 frequency row (as the actual traces
# do), so family D can combine reference and candidate row bundles.
SHARED_FREQ_ROW = bytes((i * 31 + 7) & 0xFF for i in range(32 * 2 * 4))


def _captures(tmp_path):
    root = tmp_path / "activations"
    root.mkdir()
    fixtures.write_synthetic_capture(root, "first", geometry="single",
                                     weights_writer=True, seed=1,
                                     frequency_row=SHARED_FREQ_ROW)
    fixtures.write_synthetic_capture(root, "pair", geometry="pair",
                                     weights_writer=False, seed=2,
                                     frequency_row=SHARED_FREQ_ROW)
    return schema.load_captures(root)


def _find(planned, kind, marker):
    return next(e for e in planned if e.kind == kind and marker in e.name)


def _b2(tmp_path):
    return copy.deepcopy(_find(experiments.plan_experiments(_captures(tmp_path)),
                               "B", "B2"))


def _fake_session():
    torch = fakes.FakeTorch()
    bindings = fakes.FakeBindings(torch)
    session = type("S", (), {})()
    session.torch = torch
    session.bindings = bindings
    session.stream = fakes.STREAM
    session.handle = fakes.HANDLE
    session.synchronize = lambda: None
    return session, bindings


# ---------------------------------------------------------------------------
# Positive: the A/B/P plans validate.
# ---------------------------------------------------------------------------

def test_actual_style_plans_validate_every_family(tmp_path):
    planned = experiments.plan_experiments(_captures(tmp_path))
    kinds = {e.kind for e in planned}
    assert {"A", "B", "P", "C", "D"} <= kinds
    # Every selected experiment validates before any GPU/native path: the C/D
    # replacement planners emit complete, structurally valid bundles.
    for experiment in planned:
        runner.validate_experiment(experiment)  # re-exported in runner


@pytest.mark.skipif(not ACTUAL_CAPTURE.is_dir(),
                    reason="actual compressor capture not present")
def test_actual_plan_validates_every_experiment():
    captures = schema.load_captures(ACTUAL_CAPTURE)
    planned = experiments.plan_experiments(captures)
    for experiment in planned:
        runner.validate_experiment(experiment)
    kinds = Counter(e.kind for e in planned)
    assert kinds["C"] == 14
    assert kinds["D"] == 24


# ---------------------------------------------------------------------------
# Negative cases.
# ---------------------------------------------------------------------------

def test_missing_projected_to_kv_read_is_rejected(tmp_path):
    experiment = _b2(tmp_path)
    # The B2 defect: the pool stage read the recorded "kv" role instead of the
    # fresh projection's "projected" role.
    pool = next(s for s in experiment.stages if s["kind"] == "pool")
    pool["kv"] = "kv"
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    message = str(excinfo.value)
    assert "'kv'" in message
    assert "produced by an earlier stage" in message


def test_stage_reading_future_output_is_rejected(tmp_path):
    experiment = _b2(tmp_path)
    experiment.stages = experiment.stages[2:] + experiment.stages[:2]
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "earlier stage" in str(excinfo.value)


def test_operand_shape_bytes_mismatch_is_rejected(tmp_path):
    experiment = copy.deepcopy(_find(
        experiments.plan_experiments(_captures(tmp_path)), "B", "B1"))
    operand = next(o for o in experiment.operands if o.role == "input")
    operand.blob = operand.blob[:-2]
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "itemsize" in str(excinfo.value)


def test_duplicate_operand_role_is_rejected(tmp_path):
    experiment = copy.deepcopy(_find(
        experiments.plan_experiments(_captures(tmp_path)), "B", "B1"))
    experiment.operands.append(copy.deepcopy(experiment.operands[0]))
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "duplicate operand role" in str(excinfo.value)


def test_output_clobbering_an_operand_is_rejected(tmp_path):
    planned = experiments.plan_experiments(_captures(tmp_path))
    experiment = copy.deepcopy(_find(planned, "A", "A2"))
    experiment.stages[0]["values"] = "output"  # "output" is a read operand
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "clobber" in str(excinfo.value)


def test_output_not_listed_for_allocation_is_rejected(tmp_path):
    planned = experiments.plan_experiments(_captures(tmp_path))
    experiment = copy.deepcopy(_find(planned, "B", "B1"))
    experiment.outputs = ("projected",)  # the second stage still makes scores
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "not listed in experiment.outputs" in str(excinfo.value)


def test_requested_output_never_produced_is_rejected(tmp_path):
    planned = experiments.plan_experiments(_captures(tmp_path))
    experiment = copy.deepcopy(_find(planned, "A", "A1"))
    experiment.outputs = experiment.outputs + ("projected",)
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "never produced" in str(excinfo.value)


def test_missing_baseline_oracle_is_rejected(tmp_path):
    planned = experiments.plan_experiments(_captures(tmp_path))
    experiment = copy.deepcopy(_find(planned, "A", "A1"))
    experiment.expected.pop("values")
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "missing an exact oracle" in str(excinfo.value)


def test_invalid_baseline_oracle_length_is_rejected(tmp_path):
    planned = experiments.plan_experiments(_captures(tmp_path))
    experiment = copy.deepcopy(_find(planned, "A", "A1"))
    experiment.expected["values"] = experiment.expected["values"][:-1]
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "bytes but the output is" in str(excinfo.value)


def test_missing_stage_read_key_is_rejected(tmp_path):
    experiment = _b2(tmp_path)
    pool = next(s for s in experiment.stages if s["kind"] == "pool")
    del pool["norm"]
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "missing its read-role key 'norm'" in str(excinfo.value)


def test_nonpositive_slots_is_rejected(tmp_path):
    experiment = copy.deepcopy(_find(
        experiments.plan_experiments(_captures(tmp_path)), "A", "A1"))
    experiment.slots = 0
    with pytest.raises(ValidationError) as excinfo:
        runner.validate_experiment(experiment)
    assert "positive integer" in str(excinfo.value)


def test_counterfactual_may_have_empty_oracle(tmp_path):
    experiment = copy.deepcopy(_find(
        experiments.plan_experiments(_captures(tmp_path)), "B", "B3"))
    assert experiment.counterfactual is True
    assert experiment.expected == {}
    runner.validate_experiment(experiment)


def test_invalid_plan_rejected_before_any_fake_gpu_activity(tmp_path):
    experiment = _b2(tmp_path)
    next(s for s in experiment.stages if s["kind"] == "pool")["kv"] = "kv"
    session, bindings = _fake_session()
    output = tmp_path / "untouched"
    with pytest.raises(ValidationError):
        runner.execute_experiment(session, experiment, output, repeats=1)
    assert bindings.call_names() == []
    assert not output.exists()


# ---------------------------------------------------------------------------
# CLI: the full plan validates every experiment, and an A/B/P-only root works.
# ---------------------------------------------------------------------------

def _native_pin(tmp_path):
    native = tmp_path / "libfake.so"
    native.write_bytes(b"not-a-real-native-library")
    return native, hashlib.sha256(native.read_bytes()).hexdigest()


def test_cli_full_plan_validates_every_experiment(tmp_path, capsys):
    root = tmp_path / "activations"
    root.mkdir()
    fixtures.write_synthetic_capture(root, "first", geometry="single",
                                     weights_writer=True, seed=1,
                                     frequency_row=SHARED_FREQ_ROW)
    fixtures.write_synthetic_capture(root, "pair", geometry="pair",
                                     weights_writer=False, seed=2,
                                     frequency_row=SHARED_FREQ_ROW)
    native, sha = _native_pin(tmp_path)
    output = tmp_path / "plan-out"
    rc = cli.main(["--activations", str(root), "--native-library", str(native),
                   "--native-sha256", sha, "--output", str(output)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert summary["reference_selection"]["rule"] == "unique_single_row"
    plan = json.loads((output / "plan.json").read_text())
    # Every planned experiment (C/D included) passed validation before the
    # plan was written; no experiment is emitted unvalidated.
    assert plan["reference_selection"] == {
        "rule": "unique_single_row", "refname": "first", "rows": 1,
    }
    kinds = Counter(e["kind"] for e in plan["experiments"])
    assert kinds["C"] == 7
    assert kinds["D"] == 12


def test_cli_abp_only_plan_succeeds(tmp_path, capsys):
    root = tmp_path / "activations"
    root.mkdir()
    fixtures.write_synthetic_capture(root, "only", geometry="single",
                                     weights_writer=True, seed=4)
    native, sha = _native_pin(tmp_path)
    output = tmp_path / "plan-out"
    rc = cli.main(["--activations", str(root), "--native-library", str(native),
                   "--native-sha256", sha, "--output", str(output)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert summary["reference_selection"]["rule"] == "unique_single_row"
    assert (output / "plan.json").is_file()
