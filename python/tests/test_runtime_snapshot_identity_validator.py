from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from validate_ds4_runtime_snapshot_identity import (  # noqa: E402
    REQUIRED_LABELS,
    qualification_report,
)


MODEL_ID = "tpurtell/DeepSeek-V4-Flash-0731-EXL3-2bpw-v4"
REVISION = "a" * 64


def runtime_record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "ds4rt-runtime-model-snapshot-v1",
        "model_id": MODEL_ID,
        "requested_revision": REVISION,
        "selected_revision": REVISION,
        "selection": "explicit",
        "snapshot_path": f"/root/.cache/huggingface/snapshots/{REVISION}",
    }
    record.update(updates)
    return record


def write_logs(
    root: Path,
    *,
    final_updates: dict[str, object] | None = None,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for label in REQUIRED_LABELS:
        path = root / f"{label}.log"
        stale = runtime_record(selected_revision="b" * 64)
        current = runtime_record(
            **(final_updates if label == "coordinator" and final_updates else {})
        )
        path.write_text(
            "startup\n"
            f"runtime_model_snapshot {json.dumps(stale, sort_keys=True)}\n"
            f"runtime_model_snapshot {json.dumps(current, sort_keys=True)}\n",
            encoding="utf-8",
        )
        result[label] = path
    return result


def test_qualification_uses_latest_record_and_binds_logs(tmp_path: Path) -> None:
    report = qualification_report(write_logs(tmp_path), MODEL_ID, REVISION)

    assert report["status"] == "complete"
    assert report["revision"] == REVISION
    assert len(report["daemons"]) == 5
    assert len(report["report_sha256"]) == 64
    for daemon in report["daemons"]:
        assert daemon["runtime_record_count"] == 2
        assert daemon["runtime_snapshot"]["selected_revision"] == REVISION
        assert len(daemon["log_sha256"]) == 64


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"selection": "implicit"}, "explicit snapshot selection"),
        ({"requested_revision": "b" * 64}, "requested a different"),
        ({"selected_revision": "b" * 64}, "selected a different"),
        ({"model_id": "tpurtell/wrong"}, "different model ID"),
        ({"snapshot_path": "/tmp/wrong"}, "does not end"),
    ),
)
def test_qualification_rejects_identity_mismatch(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        qualification_report(
            write_logs(tmp_path, final_updates=updates),
            MODEL_ID,
            REVISION,
        )


def test_qualification_rejects_missing_daemon_record(tmp_path: Path) -> None:
    logs = write_logs(tmp_path)
    logs["kiwi"].write_text("no identity here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="kiwi has no"):
        qualification_report(logs, MODEL_ID, REVISION)
