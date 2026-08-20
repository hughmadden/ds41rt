from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


QUANTIZATION = Path(__file__).parents[1]
if str(QUANTIZATION) not in sys.path:
    sys.path.insert(0, str(QUANTIZATION))
SCRIPT = QUANTIZATION / "validate_exl3_remote_real_family.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_exl3_remote_real_family", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REPORT = (
    SCRIPT.parents[1]
    / "reports"
    / "quantization"
    / "flash-real-family-remote-parity.json"
)
REPORT_SHA256 = "1434a7098c05c3007d4fcd6a60fa8def0b7893d71dc80d8b1c36476c3daa771b"


def write_lock(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "revision": "a" * 40,
                "source_tree_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )


def write_preflight(path: Path, *, image: str = "c") -> None:
    path.write_text(
        json.dumps(
            {
                "status": "qualified",
                "role": "expert",
                "target_platform": "linux/arm64",
                "cuda_arch": "121",
                "image_digest": "sha256:" + image * 64,
                "gptqmodel": {
                    "revision": "a" * 40,
                    "source_tree_sha256": "b" * 64,
                },
                "python": {"gil_enabled": False},
                "gpus": [{"index": 0}],
            }
        ),
        encoding="utf-8",
    )


def declarations(tmp_path: Path) -> list[tuple[str, str, str]]:
    result = []
    for index, name in enumerate(("ostrich", "dodo", "emu", "kiwi"), 1):
        preflight = tmp_path / f"{name}.json"
        write_preflight(preflight)
        result.append((name, f"http://10.55.0.{index}:17841", str(preflight)))
    return result


def test_worker_specs_bind_four_workers_to_one_locked_image(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    write_lock(lock)
    specs = MODULE.load_worker_specs(declarations(tmp_path), lock_path=lock)
    assert [spec.name for spec in specs] == ["dodo", "emu", "kiwi", "ostrich"]
    assert len({spec.image_digest for spec in specs}) == 1
    assert all(len(spec.preflight_sha256) == 64 for spec in specs)


def test_worker_specs_reject_image_drift_and_incomplete_topology(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    write_lock(lock)
    values = declarations(tmp_path)
    write_preflight(Path(values[-1][2]), image="d")
    with pytest.raises(MODULE.ValidationError, match="one image digest"):
        MODULE.load_worker_specs(values, lock_path=lock)
    with pytest.raises(MODULE.ValidationError, match="exactly 4 Sparks"):
        MODULE.load_worker_specs(values[:3], lock_path=lock)


@pytest.mark.parametrize(
    "url",
    [
        "https://10.55.0.1:17841",
        "http://user@10.55.0.1:17841",
        "http://10.55.0.1",
        "http://10.55.0.1:17841/path",
    ],
)
def test_worker_specs_reject_non_internal_http_shape(tmp_path: Path, url: str) -> None:
    lock = tmp_path / "lock.json"
    write_lock(lock)
    values = declarations(tmp_path)
    values[0] = (values[0][0], url, values[0][2])
    with pytest.raises(MODULE.ValidationError, match="worker URL"):
        MODULE.load_worker_specs(values, lock_path=lock)


def test_projection_names_cover_complete_expert_family() -> None:
    assert [
        MODULE.checkpoint_projection_base(7, 31, stem)
        for stem, _alias, _hessian in MODULE.PROJECTIONS
    ] == [
        "layers.7.ffn.experts.31.w1",
        "layers.7.ffn.experts.31.w3",
        "layers.7.ffn.experts.31.w2",
    ]
    assert MODULE.runtime_module_name(7, 31, "down_proj") == (
        "model.layers.7.mlp.experts.31.down_proj"
    )


def test_committed_real_family_report_is_the_passing_four_spark_evidence() -> None:
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA256
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["family"]["assigned_worker"] == "emu"
    assert report["family_output"]["equivalent"] is True
    assert len(report["projections"]) == 3
    assert all(
        projection["local"]["repeat"]["packed_byte_equal"]
        and projection["reconstructed_equivalent"]
        and all(
            control["packed_byte_equal_to_assigned"]
            for control in projection["spark_controls"]
        )
        for projection in report["projections"]
    )
