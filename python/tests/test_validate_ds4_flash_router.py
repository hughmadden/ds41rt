from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_ds4_flash_router as validator  # noqa: E402


def test_parse_layer_selection_accepts_ranges_and_deduplicates() -> None:
    assert validator.parse_layer_selection("3,5-7,6,42") == [3, 5, 6, 7, 42]


@pytest.mark.parametrize("value", ["", "2", "3-2", "42-43", "3,,4"])
def test_parse_layer_selection_rejects_invalid_learned_layers(value: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        validator.parse_layer_selection(value)


def test_load_capture_records_binds_progress_paths_and_geometry(tmp_path: Path) -> None:
    capture_dir = tmp_path / "captures" / "000-cal"
    capture_dir.mkdir(parents=True)
    base = "capture_0000000123_000000000004_layer_03_rows_2_expert_input"
    activation = capture_dir / f"{base}.bf16"
    routes = capture_dir / f"{base}_routes_u16_f32.bin"
    activation.write_bytes(b"activation")
    routes.write_bytes(b"routes")
    progress = {
        "prompts": [
            {
                "capture_files": [
                    {
                        "layer_id": 3,
                        "rows": 2,
                        "path": str(activation.relative_to(tmp_path)),
                        "route_path": str(routes.relative_to(tmp_path)),
                        "sha256": "activation-sha",
                        "route_sha256": "route-sha",
                    }
                ]
            }
        ]
    }
    (tmp_path / "progress.json").write_text(json.dumps(progress), encoding="utf-8")

    records = validator.load_capture_records(tmp_path, None, [3])

    assert records == [
        {
            "layer": 3,
            "rows": 2,
            "path": activation,
            "route_path": routes,
            "sha256": "activation-sha",
            "route_sha256": "route-sha",
        }
    ]


def test_load_capture_records_requires_every_selected_layer(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text(
        json.dumps({"prompts": []}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"no retained rows for layers \[3\]"):
        validator.load_capture_records(tmp_path, None, [3])
