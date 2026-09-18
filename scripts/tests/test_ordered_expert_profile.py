"""Exercise profile selection without importing CUDA or invoking a compiler."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
WIDTHS = "1:64,16:192,80:192,256:192,1024:192,4096:192"
CAPACITIES = (1, 16, 80, 256, 1024, 4096)


def load_exporter(monkeypatch, name):
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", ModuleType("_pinned_sparkinfer"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "python/tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standard_recipe_still_selects_existing_atomic_threshold(monkeypatch, tmp_path):
    standard = load_exporter(monkeypatch, "export_b12x_v41_experts_aot")
    slices = ModuleType("export_b12x_v41_slices_aot")
    slices.export = Mock()
    monkeypatch.setitem(sys.modules, slices.__name__, slices)
    standard.export(tmp_path, "spark", CAPACITIES, "fp8_k32")
    slices.export.assert_called_once_with(
        tmp_path, CAPACITIES, {c: 64 if c == 1 else 192 for c in CAPACITIES},
        atomic_min_capacity=256, role="spark", standard_names=True,
    )


@pytest.mark.parametrize("threshold", [None, 256, 1024, 4096, 4097])
def test_slice_cli_passes_exact_profile_to_existing_pipeline(monkeypatch, tmp_path, threshold):
    module = load_exporter(monkeypatch, "export_b12x_v41_slices_aot")
    export = Mock()
    monkeypatch.setattr(module, "export", export)
    argv = ["export", "--output-dir", str(tmp_path), "--rows", ",".join(map(str, CAPACITIES)),
            "--width", WIDTHS, "--role", "spark"]
    if threshold is not None:
        argv += ["--atomic-min-capacity", str(threshold)]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    export.assert_called_once_with(
        tmp_path, CAPACITIES, {c: 64 if c == 1 else 192 for c in CAPACITIES},
        threshold, "spark", standard_names=False,
    )
    if threshold == 4097:
        assert all(capacity < threshold for capacity in CAPACITIES)


@pytest.mark.parametrize("value", ["0", "81", "4098", "ordered"])
def test_invalid_cli_threshold_fails_before_any_export(monkeypatch, tmp_path, value):
    module = load_exporter(monkeypatch, "export_b12x_v41_slices_aot")
    export = Mock()
    monkeypatch.setattr(module, "export", export)
    monkeypatch.setattr(sys, "argv", ["export", "--output-dir", str(tmp_path),
                                      "--width", WIDTHS, "--atomic-min-capacity", value])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    export.assert_not_called()


def configure(tmp_path, *, width="", threshold="", architecture="121"):
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(profile NONE)\n"
        "set(DS41RT_ENABLE_CUDA ON)\n"
        "add_custom_target(ds41rt_verify_sparkinfer_source)\n"
        f'include("{ROOT}/native/cmake/v41_experts.cmake")\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/selection.txt" '
        '"${DS41RT_V41_EXPERT_EXPORT_SCRIPT}\\n${DS41RT_V41_EXPERT_EXPORT_ARGS}\\n")\n'
    )
    build = tmp_path / "build"
    result = subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build),
         f"-DDS41RT_CUDA_ARCHITECTURES={architecture}",
         f"-DDS41RT_V41_EXPERT_SLICE_WIDTH={width}",
         f"-DDS41RT_V41_EXPERT_ATOMIC_MIN_CAPACITY={threshold}"],
        capture_output=True, text=True, timeout=30,
    )
    return result, build


@pytest.mark.parametrize("threshold", ["", "256", "4097"])
def test_cmake_selects_explicit_ordered_or_atomic_slice_profile(tmp_path, threshold):
    result, build = configure(tmp_path, width=WIDTHS, threshold=threshold)
    assert result.returncode == 0, result.stderr
    script, args = (build / "selection.txt").read_text().splitlines()
    assert script == "export_b12x_v41_slices_aot.py"
    expected = ["--width", WIDTHS, "--rows", ",".join(map(str, CAPACITIES)), "--role", "spark"]
    if threshold:
        expected += ["--atomic-min-capacity", threshold]
    assert args.split(";") == expected


def test_cmake_default_keeps_recipe_exporter(tmp_path):
    result, build = configure(tmp_path)
    assert result.returncode == 0, result.stderr
    script, args = (build / "selection.txt").read_text().splitlines()
    assert script == "export_b12x_v41_experts_aot.py"
    assert args.split(";") == ["--role", "spark", "--input-format", "fp8_k32"]


@pytest.mark.parametrize("width,threshold,architecture", [
    ("", "4097", "121"),
    (WIDTHS, "4097", "120"),
    (WIDTHS, "4098", "121"),
])
def test_cmake_rejects_inapplicable_thresholds(tmp_path, width, threshold, architecture):
    result, _ = configure(tmp_path, width=width, threshold=threshold, architecture=architecture)
    assert result.returncode != 0
    assert "Direct token accumulation requires Spark slices" in result.stderr


@pytest.mark.parametrize("script", ["build-release-artifacts.sh", "build-wip-artifacts.sh"])
@pytest.mark.parametrize("ordered", [False, True])
def test_artifact_builder_forwards_profile_to_cmake(tmp_path, script, ordered):
    # Run the real shell builder until its first CMake call. Stub only external
    # compilation/verification tools; exit there before any artifacts are installed.
    source = tmp_path / "source"
    (source / "rust").mkdir(parents=True)
    (source / "native").mkdir()
    (source / "rust/Cargo.toml").write_text("")
    (source / "native/CMakeLists.txt").write_text("")
    (source / "THIRD_PARTY_NOTICES.md").write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python3", "cargo"):
        path = bin_dir / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    cmake = bin_dir / "cmake"
    cmake.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                     "open(os.environ['PROFILE_TEST_ARGV'], 'w').write(json.dumps(sys.argv[1:]))\n"
                     "sys.exit(73)\n")
    cmake.chmod(0o755)
    env = os.environ.copy()
    env.pop("DS41RT_V41_EXPERT_SLICE_WIDTH", None)
    env.pop("DS41RT_V41_EXPERT_ATOMIC_MIN_CAPACITY", None)
    env.update(PATH=f"{bin_dir}:{env['PATH']}", PROFILE_TEST_ARGV=str(tmp_path / "argv.json"),
               PYTHONDONTWRITEBYTECODE="1")
    if ordered:
        env.update(DS41RT_V41_EXPERT_SLICE_WIDTH=WIDTHS,
                   DS41RT_V41_EXPERT_ATOMIC_MIN_CAPACITY="4097")
    command = ["bash", str(ROOT / "scripts" / script), str(source), "expert", "121"]
    if script == "build-wip-artifacts.sh":
        command.append(str(tmp_path / "build"))
    command.append(str(tmp_path / "output"))
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 73, result.stderr
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert f"-DDS41RT_V41_EXPERT_SLICE_WIDTH={WIDTHS if ordered else ''}" in argv
    assert f"-DDS41RT_V41_EXPERT_ATOMIC_MIN_CAPACITY={'4097' if ordered else ''}" in argv
