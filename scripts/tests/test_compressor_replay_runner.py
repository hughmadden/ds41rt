"""CPU executor tests: failure suppression, aliasing and unexecuted status.

These drive the real ``execute_experiment``/``run_experiments`` paths with the
fake torch/tensor/native-binding doubles, so a rejected launch is proven to
suppress downloads, byte comparisons and output writes, and a supplied
read-only operand is proven never to be erased.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compressor_replay_fakes as fakes  # noqa: E402

from compressor_replay import runner  # noqa: E402
from compressor_replay.runner import (  # noqa: E402
    PlannedExperiment,
    PlannedOperand,
)

STREAM = fakes.STREAM
HANDLE = fakes.HANDLE


class FakeSession:
    def __init__(self, bindings):
        self.torch = bindings.torch
        self.bindings = bindings
        self.stream = STREAM
        self.handle = HANDLE
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


def _session(rcs=None, writer=None):
    return FakeSession(fakes.FakeBindings(fakes.FakeTorch(), rcs=rcs,
                                          writer=writer))


def _op(role, size, dtype="uint8", shape=None):
    return PlannedOperand(role=role, blob=bytes([1]) * size, dtype=dtype,
                          shape=shape or (size,), source=f"test:{role}")


def _pool_pack(rows=1, slots=4):
    return PlannedExperiment(
        name="pool_pack", kind="A", rows=rows, slots=slots,
        operands=[
            _op("kv", rows * 512 * 4), _op("scores", rows * 512 * 4),
            _op("pending_kv", slots * 512 * 4),
            _op("pending_scores", slots * 512 * 4),
            _op("predecessors", rows * 8), _op("norm", 512 * 2),
            _op("frequencies", rows * 32 * 2 * 4),
        ],
        stages=[
            {"kind": "pool", "kv": "kv", "scores": "scores",
             "pending_kv": "pending_kv", "pending_scores": "pending_scores",
             "predecessors": "predecessors", "norm": "norm",
             "output": "output", "slots": slots},
            {"kind": "pack", "input": "output", "frequencies": "frequencies",
             "values": "values", "scales": "scales"},
        ],
        outputs=("output", "values", "scales"),
        expected={"output": bytes(rows * 512 * 2),
                  "values": bytes(rows * 256), "scales": bytes(rows * 32)},
    )


def _direct_pack(rows=1, expected=None):
    return PlannedExperiment(
        name="direct_pack", kind="A", rows=rows, slots=4,
        operands=[_op("output", rows * 512 * 2), _op("frequencies",
                                                     rows * 64 * 4)],
        stages=[{"kind": "pack", "input": "output", "frequencies":
                 "frequencies", "values": "values", "scales": "scales"}],
        outputs=("values", "scales"),
        expected=expected if expected is not None else {
            "values": bytes(rows * 256), "scales": bytes(rows * 32)},
    )


def test_upload_download_preserves_raw_bits():
    torch = fakes.FakeTorch()
    session = FakeSession(fakes.FakeBindings(torch))
    blob = bytes(range(256)) * 3
    tensor = runner.upload(session, blob, "uint8", (len(blob),))
    assert bytes(tensor.memory) == blob
    assert runner.download(session, tensor) == blob


def test_execute_failure_blocks_download_and_next_stage(tmp_path):
    session = _session(rcs=[1])  # pool rejected before any kernel launch
    out = tmp_path / "exp"
    record = runner.execute_experiment(session, _pool_pack(), out, repeats=3)
    # The dependent pack stage is never launched and no repeat runs.
    assert session.bindings.call_names() == ["pool"]
    assert record["executed"] is False
    assert record["status"] == "unexecuted"
    assert record["warmup_rejected"] is True
    for repeat in record["repeat_records"]:
        assert repeat["unscored"] is True
        assert repeat["raw_outputs_written"] is False
        assert repeat["roles"] == {}
    # No numeric output files were written, only the record json.
    assert list(out.glob("*_repeat*.bin")) == []
    assert (out / "pool_pack.json").is_file()


def test_execute_failure_stops_after_first_failing_stage(tmp_path):
    session = _session(rcs=[0, 3])  # pool ok, pack rejected
    out = tmp_path / "exp"
    record = runner.execute_experiment(session, _pool_pack(), out, repeats=3)
    # Warmup: pool then pack fails; repeats are skipped as warmup was rejected.
    assert session.bindings.call_names() == ["pool", "pack"]
    assert record["status"] == "unexecuted"
    warmup_pack = [s for s in record["warmup_stages"] if s["stage"] == "pack"]
    assert warmup_pack == [{"stage": "pack", "rc": 3,
                            "dependent_skipped": False}]


def test_run_experiments_reports_pass_and_writes_outputs(tmp_path):
    expected_values = bytes([0xAB]) * 256
    expected_scales = bytes([0xCD]) * 32

    def writer(name, args):
        if name == "pack":
            _input, _freq, values_ptr, scales_ptr = args[:4]
            torch = session.bindings.torch
            torch.write_at(values_ptr, expected_values)
            torch.write_at(scales_ptr, expected_scales)
        return None

    torch = fakes.FakeTorch()
    session = FakeSession(fakes.FakeBindings(torch, writer=writer))
    experiment = _direct_pack(expected={"values": expected_values,
                                        "scales": expected_scales})
    summary = runner.run_experiments([experiment], session,
                                     tmp_path / "run", repeats=3)
    assert summary["component_baseline_pass"] is True
    assert summary["status"] == "pass"
    assert summary["unscored_experiments"] == []
    assert summary["numerical_mismatch_experiments"] == []
    files = sorted(p.name for p in (tmp_path / "run" / "direct_pack").glob(
        "*.bin"))
    assert len(files) == 6  # 3 repeats x 2 output roles


def test_run_experiments_reports_unscored_not_mismatch(tmp_path):
    session = _session(rcs=[5])
    experiment = _direct_pack(expected={"values": bytes(256),
                                        "scales": bytes(32)})
    summary = runner.run_experiments([experiment], session,
                                     tmp_path / "run", repeats=2)
    assert summary["status"] == "unexecuted"
    assert summary["unscored_experiments"] == ["direct_pack"]
    assert summary["numerical_mismatch_experiments"] == []
    assert summary["component_baseline_pass"] is False


def test_counterfactual_failure_is_unexecuted_not_mismatch_or_pass(tmp_path):
    session = _session(rcs=[1])
    experiment = PlannedExperiment(
        name="cf", kind="C", rows=1, slots=4, counterfactual=True,
        operands=[
            _op("input", 5120 * 2), _op("wkv", 512 * 5120 * 2),
            _op("wgate", 512 * 5120 * 2),
        ],
        stages=[
            {"kind": "project", "input": "input", "weight": "wkv",
             "output": "projected"},
            {"kind": "project", "input": "input", "weight": "wgate",
             "output": "scores"},
        ],
        outputs=("projected", "scores"), expected={},
    )
    record = runner.execute_experiment(session, experiment,
                                       tmp_path / "cf", repeats=3)
    assert record["status"] == "unexecuted"
    assert record["gates"] is False
    assert runner.execute_experiment(
        _session(), experiment, tmp_path / "cf2", repeats=1
    )["status"] == "executed"


def test_read_only_operand_is_never_zeroed(tmp_path):
    recorded = bytes([0x5A]) * (1 * 512 * 2)
    seen = {}

    def writer(name, args):
        if name == "pack":
            seen["input"] = session.bindings.torch.memory_at(args[0])
        return None

    torch = fakes.FakeTorch()
    session = FakeSession(fakes.FakeBindings(torch, writer=writer))
    experiment = _direct_pack()
    # Replace the synthetic operand with the recorded bytes.
    experiment.operands[0] = PlannedOperand(
        role="output", blob=recorded, dtype="bfloat16", shape=(1, 512),
        source="test:output",
    )
    runner.execute_experiment(session, experiment, tmp_path / "pack",
                              repeats=1)
    # The direct-pack input is still the recorded operand: zero_outputs never
    # erased a supplied read-only operand.
    assert seen["input"] == recorded


def test_role_aliasing_is_rejected():
    torch = fakes.FakeTorch()
    original = torch.empty(8)
    alias = fakes.FakeTensor(torch, original.memory, original.address, "uint8")
    experiment = _direct_pack()
    with pytest.raises(runner.RunnerError):
        runner._check_role_aliasing(experiment,
                                    {"values": original, "scales": alias})


def test_session_synchronizes_before_destroy():
    torch = fakes.FakeTorch()
    bindings = fakes.FakeBindings(torch)
    session = runner.CompressorSession(torch, bindings)
    session.close()
    assert bindings.destroyed == [HANDLE]
    ordering = [event[0] for event in torch.events
                if event[0] in ("synchronize", "destroy")]
    assert ordering == ["synchronize", "destroy"]
