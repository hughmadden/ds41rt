from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

from tokenizers import Tokenizer, models, pre_tokenizers


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_ds4_flash_token_route_coverage_extension as coverage  # noqa: E402
from deepseek_v4_benchmark import (  # noqa: E402
    DS4_ASSISTANT,
    DS4_BOS,
    DS4_THINK_CLOSE,
    DS4_USER,
)


def test_token_extension_selects_a_stable_naturally_routed_id(tmp_path: Path) -> None:
    vocab = {
        "[UNK]": 0,
        DS4_BOS: 1,
        DS4_USER: 2,
        DS4_ASSISTANT: 3,
        DS4_THINK_CLOSE: 4,
        "alpha": 5,
        "beta": 6,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens([DS4_BOS, DS4_USER, DS4_ASSISTANT, DS4_THINK_CLOSE])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    corpus = tmp_path / "corpus.jsonl"
    record = {"id": "aligned", "prompt": "alpha beta", "max_tokens": 2}
    corpus.write_text(json.dumps(record) + "\n", encoding="utf-8")
    prompt_ids, content_start, _ = coverage.content_token_bounds(
        tokenizer, record["prompt"]
    )
    assert prompt_ids[content_start] == 5

    captures = tmp_path / "captures"
    captures.mkdir()
    route_payload = struct.pack("<Hf", 0, 1.5)
    position_payload = struct.pack("<Q", content_start)
    route_path = captures / "route.bin"
    position_path = captures / "position.bin"
    route_path.write_bytes(route_payload)
    position_path.write_bytes(position_payload)
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "top_k": 1,
                "routed_scaling_factor": 1.5,
                "prompts": [
                    {
                        "index": 0,
                        "id": "aligned",
                        "prompt_sha256": hashlib.sha256(
                            record["prompt"].encode()
                        ).hexdigest(),
                        "capture_files": [
                            {
                                "layer_id": 0,
                                "rows": 1,
                                "routes_per_row": 1,
                                "route_path": str(route_path.relative_to(tmp_path)),
                                "route_sha256": hashlib.sha256(route_payload).hexdigest(),
                                "route_record_format": coverage.ROUTE_RECORD_FORMAT,
                                "position_path": str(
                                    position_path.relative_to(tmp_path)
                                ),
                                "position_sha256": hashlib.sha256(
                                    position_payload
                                ).hexdigest(),
                                "position_record_format": coverage.POSITION_RECORD_FORMAT,
                            }
                        ],
                    }
                ],
                "observed_rows_by_layer": [10],
                "observed_route_counts": [[1, 9]],
            }
        ),
        encoding="utf-8",
    )

    manifest = coverage.build_token_coverage_extension(
        corpus_path=corpus,
        progress_path=progress,
        tokenizer_path=tokenizer_path,
        output_path=tmp_path / "extended.jsonl",
        manifest_path=tmp_path / "manifest.json",
        minimum_routes=2,
        token_repetitions=2,
        maximum_extension_fraction=1.0,
    )

    output = [
        json.loads(line)
        for line in (tmp_path / "extended.jsonl").read_text().splitlines()
    ]
    assert output[-1]["coverage_token_id"] == 5
    assert output[-1]["coverage_token_repetitions"] == 2
    assert manifest["selected_token_ids"] == 1
    assert manifest["forced_expert_activation"] is False


def test_token_extension_selects_untried_retry_alignment_prompt(
    tmp_path: Path,
) -> None:
    vocab = {
        "[UNK]": 0,
        DS4_BOS: 1,
        DS4_USER: 2,
        DS4_ASSISTANT: 3,
        DS4_THINK_CLOSE: 4,
        "alpha": 5,
        "beta": 6,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens([DS4_BOS, DS4_USER, DS4_ASSISTANT, DS4_THINK_CLOSE])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    records = [
        {"id": "weak", "prompt": "alpha", "max_tokens": 2},
        {"id": "strong", "prompt": "beta", "max_tokens": 2},
        {
            "id": "alignment-0000",
            "prompt": "alpha",
            "max_tokens": 2,
            "route_alignment_of": "weak",
            "route_alignment_source_index": 0,
        },
    ]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    captures = tmp_path / "captures"
    captures.mkdir()
    prompts = []
    for index, (record, rows) in enumerate(
        zip(records, (1, 2, 1), strict=True)
    ):
        route_payload = struct.pack("<Hf", 0, 1.5) * rows
        route_path = captures / f"route-{index}.bin"
        route_path.write_bytes(route_payload)
        capture = {
            "layer_id": 0,
            "rows": rows,
            "routes_per_row": 1,
            "route_path": str(route_path.relative_to(tmp_path)),
            "route_sha256": hashlib.sha256(route_payload).hexdigest(),
            "route_record_format": coverage.ROUTE_RECORD_FORMAT,
        }
        if index == 2:
            position_payload = struct.pack("<Q", 0)
            position_path = captures / "position-2.bin"
            position_path.write_bytes(position_payload)
            capture.update(
                {
                    "position_path": str(position_path.relative_to(tmp_path)),
                    "position_sha256": hashlib.sha256(position_payload).hexdigest(),
                    "position_record_format": coverage.POSITION_RECORD_FORMAT,
                }
            )
        prompts.append(
            {
                "index": index,
                "id": record["id"],
                "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                "observed_layers": [{"layer_id": 0, "rows": rows}],
                "capture_files": [capture],
            }
        )
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "top_k": 1,
                "routed_scaling_factor": 1.5,
                "prompts": prompts,
                "observed_rows_by_layer": [4],
                "observed_route_counts": [[4]],
            }
        ),
        encoding="utf-8",
    )

    retry_output = tmp_path / "retry.jsonl"
    manifest = coverage.build_token_coverage_extension(
        corpus_path=corpus,
        progress_path=progress,
        tokenizer_path=tokenizer_path,
        output_path=tmp_path / "tokens.jsonl",
        manifest_path=tmp_path / "tokens-manifest.json",
        minimum_routes=5,
        token_repetitions=5,
        maximum_extension_fraction=1.0,
        retry_alignment_output_path=retry_output,
        retry_alignment_manifest_path=tmp_path / "retry-manifest.json",
    )

    output = [json.loads(line) for line in retry_output.read_text().splitlines()]
    assert output[-1]["id"] == "alignment-0001"
    assert output[-1]["route_alignment_of"] == "strong"
    assert output[-1]["route_alignment_source_index"] == 1
    assert manifest["selected_source_indices"] == [1]
    assert manifest["forced_expert_activation"] is False


def test_token_extension_excludes_measured_token_and_continues_ids(
    tmp_path: Path,
) -> None:
    vocab = {
        "[UNK]": 0,
        DS4_BOS: 1,
        DS4_USER: 2,
        DS4_ASSISTANT: 3,
        DS4_THINK_CLOSE: 4,
        "alpha": 5,
        "beta": 6,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens([DS4_BOS, DS4_USER, DS4_ASSISTANT, DS4_THINK_CLOSE])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    records = [
        {"id": "base", "prompt": "alpha beta", "max_tokens": 2},
        {
            "id": "token-coverage-0000",
            "prompt": "alpha",
            "max_tokens": 2,
            "coverage_token_id": 5,
            "coverage_token_repetitions": 2,
        },
    ]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    base_ids, content_start, _ = coverage.content_token_bounds(
        tokenizer, records[0]["prompt"]
    )
    assert base_ids[content_start : content_start + 2] == [5, 6]
    coverage_ids, coverage_start, _ = coverage.content_token_bounds(
        tokenizer, records[1]["prompt"]
    )
    assert coverage_ids[coverage_start] == 5

    captures = tmp_path / "captures"
    captures.mkdir()
    prompts = []
    for index, (record, positions, observed_rows) in enumerate(
        (
            (records[0], [content_start, content_start + 1], 9),
            (records[1], [coverage_start], 1),
        )
    ):
        rows = len(positions)
        route_payload = struct.pack("<Hf", 0, 1.5) * rows
        position_payload = b"".join(struct.pack("<Q", value) for value in positions)
        route_path = captures / f"route-{index}.bin"
        position_path = captures / f"position-{index}.bin"
        route_path.write_bytes(route_payload)
        position_path.write_bytes(position_payload)
        prompts.append(
            {
                "index": index,
                "id": record["id"],
                "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                "observed_layers": [{"layer_id": 0, "rows": observed_rows}],
                "capture_files": [
                    {
                        "layer_id": 0,
                        "rows": rows,
                        "routes_per_row": 1,
                        "route_path": str(route_path.relative_to(tmp_path)),
                        "route_sha256": hashlib.sha256(route_payload).hexdigest(),
                        "route_record_format": coverage.ROUTE_RECORD_FORMAT,
                        "position_path": str(position_path.relative_to(tmp_path)),
                        "position_sha256": hashlib.sha256(position_payload).hexdigest(),
                        "position_record_format": coverage.POSITION_RECORD_FORMAT,
                    }
                ],
            }
        )
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "top_k": 1,
                "routed_scaling_factor": 1.5,
                "prompts": prompts,
                "observed_rows_by_layer": [10],
                "observed_route_counts": [[3, 7]],
            }
        ),
        encoding="utf-8",
    )

    manifest = coverage.build_token_coverage_extension(
        corpus_path=corpus,
        progress_path=progress,
        tokenizer_path=tokenizer_path,
        output_path=tmp_path / "extended.jsonl",
        manifest_path=tmp_path / "manifest.json",
        minimum_routes=4,
        token_repetitions=4,
        maximum_extension_fraction=1.0,
    )

    output = [
        json.loads(line)
        for line in (tmp_path / "extended.jsonl").read_text().splitlines()
    ]
    assert output[-1]["id"] == "token-coverage-0001"
    assert output[-1]["coverage_token_id"] == 6
    assert manifest["prior_coverage_records"] == 1
    assert manifest["tried_token_ids"] == 1


def test_token_extension_uses_natural_ngram_after_single_token_is_measured(
    tmp_path: Path,
) -> None:
    vocab = {
        "[UNK]": 0,
        DS4_BOS: 1,
        DS4_USER: 2,
        DS4_ASSISTANT: 3,
        DS4_THINK_CLOSE: 4,
        "alpha": 5,
        "beta": 6,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens([DS4_BOS, DS4_USER, DS4_ASSISTANT, DS4_THINK_CLOSE])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    records = [
        {"id": "base", "prompt": "alpha beta", "max_tokens": 2},
        {
            "id": "token-coverage-0000",
            "prompt": "alpha alpha",
            "max_tokens": 2,
            "coverage_token_id": 5,
            "coverage_token_repetitions": 2,
        },
    ]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    base_ids, content_start, _ = coverage.content_token_bounds(
        tokenizer, records[0]["prompt"]
    )
    assert base_ids[content_start : content_start + 2] == [5, 6]

    captures = tmp_path / "captures"
    captures.mkdir()
    route_payloads = [
        struct.pack("<Hf", 0, 1.5) + struct.pack("<Hf", 1, 1.5),
        struct.pack("<Hf", 1, 1.5),
    ]
    position_values = [[content_start, content_start + 1], [content_start]]
    prompts = []
    for index, (record, route_payload, positions, observed_rows) in enumerate(
        zip(records, route_payloads, position_values, (9, 1), strict=True)
    ):
        position_payload = b"".join(struct.pack("<Q", value) for value in positions)
        route_path = captures / f"route-{index}.bin"
        position_path = captures / f"position-{index}.bin"
        route_path.write_bytes(route_payload)
        position_path.write_bytes(position_payload)
        prompts.append(
            {
                "index": index,
                "id": record["id"],
                "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                "observed_layers": [{"layer_id": 0, "rows": observed_rows}],
                "capture_files": [
                    {
                        "layer_id": 0,
                        "rows": len(positions),
                        "routes_per_row": 1,
                        "route_path": str(route_path.relative_to(tmp_path)),
                        "route_sha256": hashlib.sha256(route_payload).hexdigest(),
                        "route_record_format": coverage.ROUTE_RECORD_FORMAT,
                        "position_path": str(position_path.relative_to(tmp_path)),
                        "position_sha256": hashlib.sha256(position_payload).hexdigest(),
                        "position_record_format": coverage.POSITION_RECORD_FORMAT,
                    }
                ],
            }
        )
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "top_k": 1,
                "routed_scaling_factor": 1.5,
                "prompts": prompts,
                "observed_rows_by_layer": [10],
                "observed_route_counts": [[1, 9]],
            }
        ),
        encoding="utf-8",
    )

    manifest = coverage.build_token_coverage_extension(
        corpus_path=corpus,
        progress_path=progress,
        tokenizer_path=tokenizer_path,
        output_path=tmp_path / "extended.jsonl",
        manifest_path=tmp_path / "manifest.json",
        minimum_routes=2,
        token_repetitions=2,
        maximum_extension_fraction=1.0,
    )

    output = [
        json.loads(line)
        for line in (tmp_path / "extended.jsonl").read_text().splitlines()
    ]
    assert output[-1]["id"] == "token-coverage-0001"
    assert output[-1]["coverage_token_ids"] == [5, 6]
    assert "coverage_token_id" not in output[-1]
    assert manifest["selected_token_ids"] == 0
    assert manifest["selected_token_sequences"] == 1
