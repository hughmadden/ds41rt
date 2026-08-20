from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat

import pytest


SCRIPT = Path(__file__).parents[1] / "validate_flash_mtp_prefix_runtime.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_flash_mtp_prefix_runtime", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prefix_runtime_control_is_not_claimed_as_natural_calibration() -> None:
    assert MODULE.INPUT_KIND == "deterministic-runtime-control-not-natural-calibration"


def test_prefix_runtime_report_is_atomic_and_world_readable(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    MODULE._atomic_json(output, {"status": "ok"})
    assert json.loads(output.read_text()) == {"status": "ok"}
    assert stat.S_IMODE(output.stat().st_mode) == 0o644


def test_prefix_runtime_rejects_wrong_checkpoint_before_model_load(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v3",
                "expert_dtype": "fp4",
                "torch_dtype": "bfloat16",
            }
        )
    )
    with pytest.raises(MODULE.ValidationError, match="DeepSeek V4"):
        MODULE.run(snapshot, device_name="cpu", gptqmodel_lock=None)
