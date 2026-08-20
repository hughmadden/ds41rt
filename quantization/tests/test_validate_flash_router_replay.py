from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import struct

import pytest


SCRIPT = Path(__file__).parents[1] / "validate_flash_router_replay.py"
SPEC = importlib.util.spec_from_file_location("validate_flash_router_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest(tmp_path: Path) -> Path:
    activation = tmp_path / "activation.bf16"
    activation.write_bytes(b"\0" * 16)
    routes = tmp_path / "routes.bin"
    routes.write_bytes(
        b"".join(struct.pack("<Hf", expert, weight) for expert, weight in ((2, 0.7), (1, 0.3)))
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": MODULE.INPUT_SCHEMA,
                "purpose": MODULE.REQUIRED_PURPOSE,
                "checkpoint_revision": "a" * 40,
                "records": [
                    {
                        "layer": 3,
                        "rows": 1,
                        "activation": activation.name,
                        "activation_sha256": MODULE.sha256_file(activation),
                        "routes": routes.name,
                        "routes_sha256": MODULE.sha256_file(routes),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_and_files_are_fail_closed(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    manifest, records = MODULE.load_manifest(path)
    assert manifest["checkpoint_revision"] == "a" * 40
    assert records[0]["layer"] == 3
    assert MODULE.verify_record_files(records[0], hidden_size=8, top_k=2) == {
        "activation_bytes": 16,
        "route_bytes": 12,
    }
    (tmp_path / "activation.bf16").write_bytes(b"corrupt")
    with pytest.raises(MODULE.ValidationError, match="bytes"):
        MODULE.verify_record_files(records[0], hidden_size=8, top_k=2)


def test_manifest_rejects_calibration_use_and_path_escape(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    value = json.loads(path.read_text())
    value["purpose"] = "GPTQ calibration"
    path.write_text(json.dumps(value))
    with pytest.raises(MODULE.ValidationError, match="non-calibration"):
        MODULE.load_manifest(path)
    value["purpose"] = MODULE.REQUIRED_PURPOSE
    value["records"][0]["activation"] = "../activation.bf16"
    path.write_text(json.dumps(value))
    with pytest.raises(MODULE.ValidationError, match="basename"):
        MODULE.load_manifest(path)


def test_route_decoder_preserves_interleaved_ids_and_weights(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    _, records = MODULE.load_manifest(path)
    indices, weights = MODULE.decode_route_file(Path(records[0]["routes"]), rows=1, top_k=2)
    assert indices == [[2, 1]]
    assert weights[0] == pytest.approx([0.7, 0.3])


def test_route_decoder_rejects_duplicate_experts(tmp_path: Path) -> None:
    routes = tmp_path / "routes.bin"
    routes.write_bytes(struct.pack("<HfHf", 2, 0.7, 2, 0.3))
    with pytest.raises(MODULE.ValidationError, match="duplicate"):
        MODULE.decode_route_file(routes, rows=1, top_k=2)


def test_atomic_report_is_world_readable(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    MODULE._atomic_json(output, {"status": "ok"})
    assert json.loads(output.read_text()) == {"status": "ok"}
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
