from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "verify-gptqmodel-source.py"
SPEC = importlib.util.spec_from_file_location("verify_gptqmodel_source", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_source(tmp_path: Path) -> Path:
    source = tmp_path / "gptqmodel"
    for relative in (
        "gptqmodel/models/definitions/deepseek_v4.py",
        "gptqmodel/looper/exllamav3_processor.py",
        "gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py",
        "gptqmodel/__init__.py",
        "LICENSE",
        "pyproject.toml",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    return source


def make_lock(source: Path, path: Path, **overrides: object) -> Path:
    value = {
        "schema": 1,
        "repository": "https://github.com/tpurtell/GPTQModel.git",
        "revision": "1" * 40,
        "source_tree_sha256": MODULE.source_tree_sha256(source),
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_content_digest_is_stable_and_detects_source_changes(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    first = MODULE.source_tree_sha256(source)
    assert first == MODULE.source_tree_sha256(source)
    (source / "gptqmodel/looper/exllamav3_processor.py").write_text("changed\n", encoding="utf-8")
    assert MODULE.source_tree_sha256(source) != first


def test_generated_editable_install_metadata_is_ignored(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    first = MODULE.source_tree_sha256(source)
    egg_info = source / "gptqmodel.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated\n", encoding="utf-8")
    assert MODULE.source_tree_sha256(source) == first


def test_archive_source_verifies_without_git_metadata(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    lock = make_lock(source, tmp_path / "gptqmodel.lock.json")
    assert MODULE.verify(source, lock)["revision"] == "1" * 40


def test_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    lock = make_lock(
        source,
        tmp_path / "gptqmodel.lock.json",
        source_tree_sha256="0" * 64,
    )
    with pytest.raises(MODULE.VerificationError, match="does not match the lock"):
        MODULE.verify(source, lock)


def test_incomplete_source_is_rejected(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    (source / "gptqmodel/models/definitions/deepseek_v4.py").unlink()
    lock = make_lock(source, tmp_path / "gptqmodel.lock.json")
    with pytest.raises(MODULE.VerificationError, match="source is incomplete"):
        MODULE.verify(source, lock)


def test_checked_in_pin_verifies() -> None:
    root = SCRIPT.parents[1]
    lock = json.loads(
        (root / "third_party/gptqmodel.lock.json").read_text(encoding="utf-8")
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(root / "third_party/gptqmodel"),
            "--lock",
            str(root / "third_party/gptqmodel.lock.json"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"revision={lock['revision']}" in result.stdout
    assert f"source_tree_sha256={lock['source_tree_sha256']}" in result.stdout
