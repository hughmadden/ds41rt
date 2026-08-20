from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_ds4_flash_route_alignment_extension as alignment  # noqa: E402


def test_alignment_extension_selects_each_natural_pair_once(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"id": "broad", "prompt": "broad prompt", "max_tokens": 8},
        {"id": "narrow", "prompt": "narrow prompt", "max_tokens": 8},
    ]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    captures = tmp_path / "captures"
    captures.mkdir()
    payloads = [
        struct.pack("<Hf", 0, 1.5) + struct.pack("<Hf", 1, 1.5),
        struct.pack("<Hf", 1, 1.5),
    ]
    prompts = []
    for index, (record, payload) in enumerate(zip(records, payloads, strict=True)):
        route_path = captures / f"{index}.bin"
        route_path.write_bytes(payload)
        prompts.append(
            {
                "index": index,
                "id": record["id"],
                "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                "observed_layers": [{"layer_id": 0, "rows": 2 - index}],
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
                "prompts": prompts,
                "observed_rows_by_layer": [3],
                "observed_route_counts": [[1, 2]],
            }
        ),
        encoding="utf-8",
    )

    manifest = alignment.build_alignment_extension(
        corpus_path=corpus,
        progress_path=progress,
        output_path=tmp_path / "extended.jsonl",
        manifest_path=tmp_path / "manifest.json",
        minimum_routes=3,
        maximum_extension_fraction=1.0,
    )

    output = [
        json.loads(line)
        for line in (tmp_path / "extended.jsonl").read_text().splitlines()
    ]
    assert [record["id"] for record in output] == [
        "broad",
        "narrow",
        "alignment-0000",
    ]
    assert output[-1]["route_alignment_of"] == "broad"
    assert manifest["extension_records"] == 1
    assert manifest["forced_expert_activation"] is False
