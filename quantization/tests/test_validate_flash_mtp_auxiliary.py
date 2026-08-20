from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat

import pytest


SCRIPT = Path(__file__).parents[1] / "validate_flash_mtp_auxiliary.py"
SPEC = importlib.util.spec_from_file_location("validate_flash_mtp_auxiliary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_cases_cover_every_mtp_block_and_projection_direction() -> None:
    assert MODULE.parse_cases(None) == [
        (0, 0, "w1"),
        (1, 127, "w2"),
        (2, 255, "w3"),
    ]
    assert MODULE.parse_case("mtp.2:9.w3") == (2, 9, "w3")


@pytest.mark.parametrize(
    "value",
    ["", "layers.0:1.w1", "mtp.-1:0.w1", "mtp.0:-1.w2", "mtp.0:1.gate_proj"],
)
def test_case_parser_rejects_malformed_values(value: str) -> None:
    with pytest.raises(MODULE.ValidationError):
        MODULE.parse_case(value)


def test_case_parser_rejects_duplicate_work() -> None:
    with pytest.raises(MODULE.ValidationError, match="unique"):
        MODULE.parse_cases(["mtp.0:1.w1", "mtp.0:1.w1"])


def test_mapping_digest_is_order_independent_and_directional() -> None:
    entries = [("mtp.0.mlp.gate", "mtp.0.ffn.gate"), ("a", "b")]
    assert MODULE.mapping_sha256(entries) == MODULE.mapping_sha256(list(reversed(entries)))
    assert MODULE.mapping_sha256(entries) != MODULE.mapping_sha256(
        [(checkpoint, runtime) for runtime, checkpoint in entries]
    )


def test_atomic_report_is_world_readable(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    MODULE._atomic_json(output, {"status": "ok"})
    assert json.loads(output.read_text()) == {"status": "ok"}
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
