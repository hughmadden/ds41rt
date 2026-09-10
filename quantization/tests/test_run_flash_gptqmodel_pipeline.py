from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


QUANTIZATION = Path(__file__).parents[1]
if str(QUANTIZATION) not in sys.path:
    sys.path.insert(0, str(QUANTIZATION))
SCRIPT = QUANTIZATION / "run_flash_gptqmodel_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_flash_gptqmodel_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _argv(tmp_path: Path) -> list[str]:
    output = tmp_path / "models" / "flash-k3"
    return [
        str(SCRIPT),
        "--snapshot",
        str(tmp_path / "snapshot"),
        "--calibration-jsonl",
        str(tmp_path / "calibration.jsonl"),
        "--calibration-manifest",
        str(tmp_path / "manifest.json"),
        "--route-screen-report",
        str(tmp_path / "routes.json"),
        "--preflight-report",
        str(tmp_path / "preflight.json"),
        "--base-output",
        str(output),
        "--base-offload-dir",
        str(tmp_path / "base-offload"),
        "--mtp-prefix-store",
        str(tmp_path / "mtp-prefix"),
        "--mtp-overlay-output",
        str(tmp_path / "mtp-overlay"),
        "--mtp-overlay-offload-dir",
        str(tmp_path / "mtp-overlay-offload"),
        "--base-block-audit-dir",
        str(tmp_path / "base-audits"),
        "--canonical-output",
        str(output),
        "--bits",
        "3",
    ]


def test_canonical_work_defaults_to_output_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    args = MODULE.parse_args()

    assert args.canonical_work_dir == args.canonical_output.with_name(
        f".{args.canonical_output.name}.ds41rt-assembly"
    )


def test_canonical_work_rejects_non_atomic_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(tmp_path)
    argv.extend(("--canonical-work-dir", str(tmp_path / "other" / "assembly")))
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="2"):
        MODULE.parse_args()


def test_overlay_command_passes_hessian_owner_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(tmp_path)
    argv.extend(("--mtp-hessian-owner-device-weights", "1", "2"))
    monkeypatch.setattr(sys, "argv", argv)

    args = MODULE.parse_args()
    command = MODULE._overlay_command(args, resume=False)

    index = command.index("--mtp-hessian-owner-device-weights")
    assert command[index + 1 : index + 3] == ["1", "2"]
