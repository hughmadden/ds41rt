from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat

import pytest


SCRIPT = Path(__file__).parents[1] / "validate_flash_mtp_replay.py"
SPEC = importlib.util.spec_from_file_location("validate_flash_mtp_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_structural_replay_is_explicitly_not_natural_calibration() -> None:
    assert MODULE.INPUT_KIND == "deterministic-structural-control-not-natural-calibration"


def test_atomic_report_is_world_readable(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    MODULE._atomic_json(output, {"status": "ok"})
    assert json.loads(output.read_text()) == {"status": "ok"}
    assert stat.S_IMODE(output.stat().st_mode) == 0o644


def test_run_rejects_invalid_batch_before_loading_checkpoint(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "expert_dtype": "fp4",
                "sliding_window": 128,
            }
        )
    )
    with pytest.raises(MODULE.ValidationError, match="batch_size"):
        MODULE.run(
            snapshot,
            device_name="cpu",
            batch_size=0,
            main_rows=4,
            gptqmodel_lock=None,
        )
