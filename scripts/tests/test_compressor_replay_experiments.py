"""CPU tests for experiment planning, oracle bytes and the padded planner.

The plans are built from real-schema fixtures, so every baseline carries actual
oracle bytes and every counterfactual carries explicitly different input hashes.
The padded project-only planner is checked for its fixed rows/slots geometry,
zero neighbours and the single distinct BF16-low-bit neighbour.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compressor_replay_fakes as fakes  # noqa: E402

from compressor_replay import experiments, fixtures, runner, schema  # noqa: E402

ROW_BYTES = schema.ROW_BYTES["input"]

# One shared p41 frequency row across every synthetic capture (as the actual
# traces do), so family D can combine reference and candidate bundles.
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


def _input_sha(experiment):
    for operand in experiment.operands:
        if operand.role == "input":
            return hashlib.sha256(operand.blob).hexdigest()
    return None


def test_real_schema_baselines_have_oracle_bytes(tmp_path):
    planned = experiments.plan_experiments(_captures(tmp_path))
    baselines = [e for e in planned if not e.counterfactual]
    assert len(baselines) >= 4
    for experiment in baselines:
        assert experiment.expected, experiment.name
        assert set(experiment.expected) == set(experiment.outputs)
        for role, blob in experiment.expected.items():
            assert isinstance(blob, bytes) and blob


def test_counterfactuals_have_empty_oracles_and_distinct_input_hashes(tmp_path):
    planned = experiments.plan_experiments(_captures(tmp_path))
    baselines = [e for e in planned if not e.counterfactual]
    baseline_input = next(_input_sha(e) for e in baselines if _input_sha(e))
    assert baseline_input is not None
    counterfactuals = [e for e in planned if e.counterfactual]
    assert counterfactuals
    for experiment in counterfactuals:
        assert experiment.expected == {}
        assert experiment.counterfactual is True
    # Family D counterfactuals substitute already-projected FP32 operands, so
    # they carry no "input" role; the projection-only B/C/P counterfactuals
    # (and mutation/padding variants) must each have a distinct input hash.
    input_counterfactuals = [e for e in counterfactuals if _input_sha(e)]
    assert input_counterfactuals
    seen = set()
    for experiment in input_counterfactuals:
        digest = _input_sha(experiment)
        assert digest != baseline_input, experiment.name
        seen.add(digest)
    # Mutated/regrouped/padded inputs are explicitly different from each other
    # as well as from the baseline.
    assert len(seen) == len(input_counterfactuals)
    # No counterfactual produces an oracle or gates the baseline.
    assert all(not e.expected for e in counterfactuals)


def test_padded_planner_geometry_and_zero_neighbours(tmp_path):
    captures = _captures(tmp_path)
    capture = captures[1]  # the 2-row pair capture
    planned = experiments.plan_padded_project_only(capture)
    current = experiments.current_input_row(capture)
    source = capture.read_buffer("input")[
        current * ROW_BYTES:(current + 1) * ROW_BYTES
    ]

    rows_to_slots = {}
    for experiment in planned:
        rows_to_slots.setdefault(experiment.rows, set()).add(
            int(experiment.name.rsplit("slot", 1)[1].split("_")[0])
        )
    assert rows_to_slots == {2: {0, 1}, 16: {0, 1, 7, 15}}

    for experiment in planned:
        assert experiment.counterfactual is True
        assert experiment.expected == {}
        assert [stage["kind"] for stage in experiment.stages] == [
            "project", "project"
        ]
        assert experiment.outputs == ("projected", "scores")
        input_blob = next(op.blob for op in experiment.operands
                          if op.role == "input")
        assert len(input_blob) == experiment.rows * ROW_BYTES

    # Zero-neighbour variants: only the real placement slot is nonzero.
    for name in ("pad2_slot0", "pad2_slot1", "pad16_slot7", "pad16_slot15"):
        experiment = next(e for e in planned if name in e.name
                          and "distinct" not in e.name)
        slot = int(experiment.name.split("slot")[1].split("_")[0])
        blob = next(op.blob for op in experiment.operands if op.role == "input")
        for row in range(experiment.rows):
            segment = blob[row * ROW_BYTES:(row + 1) * ROW_BYTES]
            if row == slot:
                assert segment == source
            else:
                assert segment == bytes(ROW_BYTES), (name, row)


def test_padded_planner_distinct_neighbour_flips_one_bf16_low_bit(tmp_path):
    captures = _captures(tmp_path)
    capture = captures[1]
    planned = experiments.plan_padded_project_only(capture)
    experiment = next(e for e in planned if "distinct" in e.name)
    slot = int(experiment.name.split("slot")[1].split("_")[0])
    neighbour = int(experiment.name.rsplit("nb", 1)[1])
    blob = next(op.blob for op in experiment.operands if op.role == "input")
    source = capture.read_buffer("input")[
        experiments.current_input_row(capture) * ROW_BYTES:
        (experiments.current_input_row(capture) + 1) * ROW_BYTES
    ]
    assert blob[slot * ROW_BYTES:(slot + 1) * ROW_BYTES] == source
    neighbour_row = blob[neighbour * ROW_BYTES:(neighbour + 1) * ROW_BYTES]
    diffs = [i for i, (a, b) in enumerate(zip(neighbour_row, source)) if a != b]
    # BF16 is little-endian, so the value's low bit is the low byte at -2, not
    # the high byte at -1: exactly one byte differs, at offset 10238.
    assert diffs == [ROW_BYTES - 2]
    assert ROW_BYTES - 2 == 10238
    assert neighbour_row[-2] ^ source[-2] == 0x01
    assert neighbour_row[-1] == source[-1]  # the high byte is untouched
    # The placement slot's real target row is unchanged by the neighbour.
    assert neighbour != slot
    assert blob[slot * ROW_BYTES:(slot + 1) * ROW_BYTES] == source
    # The exact provenance hash is recorded in the notes.
    recorded = hashlib.sha256(neighbour_row).hexdigest()
    assert any(recorded in note for note in experiment.notes)
    assert any(f"offset {neighbour * ROW_BYTES + ROW_BYTES - 2}" in note
               for note in experiment.notes)


def test_padded_project_output_sizes_are_fp32_rows(tmp_path):
    captures = _captures(tmp_path)
    planned = experiments.plan_padded_project_only(captures[0])
    for experiment in planned:
        assert runner._output_size(experiment, "projected") == (
            experiment.rows * 512 * 4
        )
        assert runner._output_size(experiment, "scores") == (
            experiment.rows * 512 * 4
        )


def test_padded_project_executes_through_real_executor(tmp_path):
    captures = _captures(tmp_path)
    experiment = next(
        e for e in experiments.plan_padded_project_only(captures[0])
        if e.rows == 16 and "distinct" not in e.name
    )
    payload = bytes([0x42]) * (16 * 512 * 4)

    def writer(name, args):
        if name == "project":
            torch = session.bindings.torch
            torch.write_at(args[3], payload)
        return None

    torch = fakes.FakeTorch()
    session = type("S", (), {})()
    session.torch = torch
    session.bindings = fakes.FakeBindings(torch, writer=writer)
    session.stream = fakes.STREAM
    session.handle = fakes.HANDLE
    session.synchronize = lambda: None
    record = runner.execute_experiment(session, experiment,
                                       tmp_path / "padded", repeats=3)
    assert record["executed"] is True
    assert record["status"] == "executed"  # counterfactual, not a numeric gate
    assert len(list((tmp_path / "padded").glob("*.bin"))) == 6


# ---------------------------------------------------------------------------
# Reference selection (explicit name, unique one-row fallback, ambiguity).
# ---------------------------------------------------------------------------

def _named_capture(name, rows):
    return types.SimpleNamespace(directory=Path(name), rows=rows)


def test_select_reference_prefers_unique_explicit_name():
    captures = [
        _named_capture("lane0-batch20-Full-1rows", 1),
        _named_capture("other-reference-2rows", 2),
        _named_capture("lane1-batch82-Full-2rows", 2),
    ]
    reference = experiments.select_reference(captures)
    assert reference.directory.name == "other-reference-2rows"
    assert experiments.reference_selection(captures) == {
        "rule": "explicit_name", "refname": "other-reference-2rows", "rows": 2,
    }


def test_select_reference_falls_back_to_unique_single_row():
    captures = [
        _named_capture("lane0-batch20-Full-1rows", 1),
        _named_capture("lane0-batch81-Full-2rows", 2),
        _named_capture("lane1-batch82-Full-2rows", 2),
    ]
    assert experiments.reference_selection(captures) == {
        "rule": "unique_single_row",
        "refname": "lane0-batch20-Full-1rows",
        "rows": 1,
    }


def test_select_reference_rejects_ambiguous_single_rows():
    captures = [_named_capture("a-1rows", 1), _named_capture("b-1rows", 1)]
    with pytest.raises(ValueError, match="ambiguous reference"):
        experiments.select_reference(captures)
    with pytest.raises(ValueError, match="ambiguous reference"):
        experiments.reference_selection(captures)


def test_select_reference_non_unique_name_falls_back_to_single_row():
    captures = [_named_capture("reference-a", 2),
                _named_capture("reference-b", 2),
                _named_capture("lane0-batch20-Full-1rows", 1)]
    assert experiments.reference_selection(captures) == {
        "rule": "unique_single_row",
        "refname": "lane0-batch20-Full-1rows",
        "rows": 1,
    }


def test_select_reference_rejects_ambiguous_name_and_single_row():
    captures = [_named_capture("reference-a", 2),
                _named_capture("reference-b", 2),
                _named_capture("one-1rows", 1),
                _named_capture("two-1rows", 1)]
    with pytest.raises(ValueError, match="ambiguous reference"):
        experiments.select_reference(captures)


def test_select_reference_none_without_single_row_or_name():
    captures = [_named_capture("lane0-batch81-Full-2rows", 2)]
    assert experiments.select_reference(captures) is None
    assert experiments.reference_selection(captures) == {
        "rule": "none", "refname": None, "rows": None,
    }
