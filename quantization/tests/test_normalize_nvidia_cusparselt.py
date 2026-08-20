from __future__ import annotations

import csv
import importlib.util
import io
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "normalize_nvidia_cusparselt.py"
SPEC = importlib.util.spec_from_file_location("normalize_nvidia_cusparselt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_install(tmp_path: Path, *, elf_machine: int = 183) -> Path:
    site = tmp_path / "site-packages"
    dist = site / "nvidia_cusparselt_cu13-0.8.1.dist-info"
    library = site / "nvidia/cusparselt/lib/libcusparseLt.so.0"
    dist.mkdir(parents=True)
    library.parent.mkdir(parents=True)
    header = bytearray(20)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = elf_machine.to_bytes(2, "little")
    library.write_bytes(header + b"payload")
    wheel = (
        "Wheel-Version: 1.0\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-manylinux2014_sbsa\n"
    ).encode()
    (dist / "WHEEL").write_bytes(wheel)
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(
        [
            [
                "nvidia_cusparselt_cu13-0.8.1.dist-info/WHEEL",
                MODULE._hash_record(wheel),
                str(len(wheel)),
            ],
            ["nvidia_cusparselt_cu13-0.8.1.dist-info/RECORD", "", ""],
        ]
    )
    (dist / "RECORD").write_text(output.getvalue(), encoding="utf-8")
    return site


def test_arm64_normalization_is_audited_idempotent_and_verifiable(tmp_path: Path) -> None:
    site = make_install(tmp_path)
    report = MODULE.normalize(site, apply=True, machine="aarch64")
    assert report["status"] == "normalized-and-verified"
    assert report["source_wheel_sha256"] == MODULE.SOURCE_WHEEL_SHA256
    assert MODULE.NORMALIZED_TAG in next(site.glob("*.dist-info/WHEEL")).read_text()
    assert MODULE.normalize(site, apply=True, machine="aarch64") == report
    assert MODULE.normalize(site, apply=False, machine="aarch64") == report


def test_normalization_rejects_non_aarch64_payload(tmp_path: Path) -> None:
    site = make_install(tmp_path, elf_machine=62)
    with pytest.raises(MODULE.NormalizationError, match="e_machine=62"):
        MODULE.normalize(site, apply=True, machine="aarch64")


def test_non_arm64_is_explicit_noop(tmp_path: Path) -> None:
    assert MODULE.normalize(tmp_path, apply=True, machine="x86_64") == {
        "status": "not-applicable",
        "machine": "x86_64",
    }
