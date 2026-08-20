from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from collect_deepseek_v4_route_screen import (
    PROGRESS_SCHEMA,
    RouteScreenError,
    _RouteCollector,
    _layer_summary,
    _load_corpus,
)
from gptqmodel.looper.module_looper import StopMainLoop


def test_load_corpus_binds_order_and_prompt_digest(tmp_path: Path) -> None:
    corpus = tmp_path / "screening.jsonl"
    corpus.write_text(
        '\n'.join(
            (
                json.dumps({"id": "one", "prompt": "alpha"}),
                json.dumps({"id": "two", "prompt": "beta"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    texts, prompts = _load_corpus(corpus)

    assert texts == ["alpha", "beta"]
    assert [record["id"] for record in prompts] == ["one", "two"]
    assert [record["schedule_index"] for record in prompts] == [0, 1]
    assert all(record["mtp_verify_cycles"] == 0 for record in prompts)


def test_load_corpus_rejects_duplicate_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "screening.jsonl"
    corpus.write_text(
        '{"id":"same","prompt":"one"}\n'
        '{"id":"same","prompt":"two"}\n',
        encoding="utf-8",
    )

    with pytest.raises(RouteScreenError, match="duplicate corpus record"):
        _load_corpus(corpus)


def test_layer_summary_requires_exact_top_k_routes() -> None:
    summary = _layer_summary(3, [2, 4, 6, 8], rows=5, top_k=4)
    assert summary["routes"] == 20
    assert summary["zero_hit_experts"] == 0
    assert summary["min"] == 2
    assert summary["max"] == 8

    with pytest.raises(RouteScreenError, match="route totals"):
        _layer_summary(3, [2, 4, 6, 7], rows=5, top_k=4)


def test_route_collector_commits_progress_before_probe_stop(tmp_path: Path) -> None:
    progress = tmp_path / "screen.progress.json"
    pruned: list[int] = []
    collector = _RouteCollector(
        layers=2,
        experts=4,
        top_k=2,
        progress_path=progress,
        identity={"corpus_sha256": "abc"},
        stop_after_layer=0,
        prune_completed_layer=pruned.append,
    )
    collector.hook(0)(None, None, (None, None, torch.tensor([[0, 1], [1, 3]])))

    assert collector.layer_complete(layer_idx=0, submodule_finalized=True) is StopMainLoop
    assert pruned == [0]
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload == {
        "schema": PROGRESS_SCHEMA,
        "identity": {"corpus_sha256": "abc"},
        "completed_through": 0,
        "rows": [2],
        "counts": [[1, 2, 0, 1]],
    }
