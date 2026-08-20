from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat

import pytest


SCRIPT = Path(__file__).parents[1] / "preflight.py"
SPEC = importlib.util.spec_from_file_location("quantization_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_role_contracts_are_native_and_architecture_specific() -> None:
    assert MODULE.ROLE_CONTRACTS == {
        "coordinator": ("linux/amd64", "120"),
        "expert": ("linux/arm64", "121"),
    }


@pytest.mark.parametrize(
    ("target", "machine"),
    [("linux/amd64", "x86_64"), ("linux/arm64", "aarch64")],
)
def test_platform_validation_accepts_native_machine(target: str, machine: str) -> None:
    MODULE._validate_platform(target, machine)


def test_platform_validation_rejects_cross_architecture() -> None:
    with pytest.raises(MODULE.PreflightError, match="requires"):
        MODULE._validate_platform("linux/arm64", "x86_64")


@pytest.mark.parametrize("value, expected", [("120", (12, 0)), ("121", (12, 1))])
def test_cuda_arch_parser(value: str, expected: tuple[int, int]) -> None:
    assert MODULE._cuda_capability(value) == expected


def test_nvidia_smi_parser_retains_provenance_fields() -> None:
    rows = MODULE._parse_nvidia_smi(
        "1, GPU-b, NVIDIA GB10, 595.84, 140.00, 12.1\n"
        "0, GPU-a, NVIDIA GB10, 595.84, 140.00, 12.1\n"
    )
    assert [row["index"] for row in rows] == [0, 1]
    assert rows[0]["uuid"] == "GPU-a"
    assert rows[0]["driver_version"] == "595.84"
    assert rows[0]["power_limit_watts"] == 140.0
    assert rows[0]["compute_capability"] == [12, 1]


def test_nvidia_smi_parser_accepts_unreported_spark_power_limit() -> None:
    rows = MODULE._parse_nvidia_smi(
        "0, GPU-a, NVIDIA GB10, 580.159.03, [N/A], 12.1\n"
    )
    assert rows[0]["power_limit_watts"] is None


def test_image_digest_requires_content_address() -> None:
    assert MODULE.IMAGE_DIGEST_RE.fullmatch("sha256:" + "a" * 64)
    assert not MODULE.IMAGE_DIGEST_RE.fullmatch("latest")


def test_report_identity_excludes_only_observation_time() -> None:
    first = {
        "generated_at": "2026-08-17T00:00:00+00:00",
        "image_digest": "sha256:" + "a" * 64,
        "gpus": [{"uuid": "GPU-a", "driver_version": "595.71.05"}],
    }
    second = {**first, "generated_at": "2026-08-18T00:00:00+00:00"}

    assert MODULE.report_identity_sha256(first) == MODULE.report_identity_sha256(
        second
    )
    second["gpus"] = [{"uuid": "GPU-b", "driver_version": "595.71.05"}]
    assert MODULE.report_identity_sha256(first) != MODULE.report_identity_sha256(
        second
    )


def test_preflight_report_is_world_readable(tmp_path: Path) -> None:
    report = tmp_path / "preflight.json"
    MODULE._atomic_json(report, {"status": "qualified"})
    assert json.loads(report.read_text(encoding="utf-8")) == {"status": "qualified"}
    assert stat.S_IMODE(report.stat().st_mode) == 0o644
