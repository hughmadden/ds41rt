from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_ds4_flash_route_coverage_extension as coverage  # noqa: E402


def test_greedy_extension_replays_natural_routes_to_the_floor(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"id": "first", "prompt": "first prompt", "max_tokens": 8},
        {"id": "second", "prompt": "second prompt", "max_tokens": 8},
    ]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    captures = tmp_path / "captures"
    captures.mkdir()
    route_payloads = [
        b"".join(struct.pack("<Hf", 0, 1.5) for _ in range(2)),
        struct.pack("<Hf", 1, 1.5),
    ]
    prompt_records = []
    for index, (record, payload) in enumerate(zip(records, route_payloads, strict=True)):
        route_path = captures / f"{index}.bin"
        route_path.write_bytes(payload)
        prompt_records.append(
            {
                "index": index,
                "id": record["id"],
                "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                "max_tokens": 8,
                "observed_layers": [
                    {"layer_id": 0, "rows": 2 - index, "routes": 2 - index}
                ],
                "capture_files": [
                    {
                        "layer_id": 0,
                        "route_path": str(route_path.relative_to(tmp_path)),
                        "route_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        )
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "prompts": prompt_records,
                "observed_rows_by_layer": [3],
                "observed_route_counts": [[2, 1]],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "extended.jsonl"
    manifest_path = tmp_path / "extension.json"

    manifest = coverage.build_extension(
        corpus_path=corpus,
        progress_path=progress,
        output_path=output,
        manifest_path=manifest_path,
        minimum_routes=2,
        maximum_extension_fraction=1.0,
    )

    extended = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["id"] for record in extended] == [
        "first",
        "second",
        "coverage-0000",
    ]
    assert extended[-1]["coverage_replay_of"] == "second"
    assert manifest["extension_tokens"] == 1
    assert manifest["deficient_pairs_before"] == 1
    assert manifest["forced_expert_activation"] is False
    assert json.loads(manifest_path.read_text()) == manifest


def test_extension_preflights_both_outputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "extension.json"
    manifest_path.write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        coverage.build_extension(
            corpus_path=tmp_path / "missing-corpus.jsonl",
            progress_path=tmp_path / "missing-progress.json",
            output_path=tmp_path / "extended.jsonl",
            manifest_path=manifest_path,
            minimum_routes=2,
            maximum_extension_fraction=1.0,
        )

    assert not (tmp_path / "extended.jsonl").exists()
