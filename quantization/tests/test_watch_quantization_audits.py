from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[2] / "scripts" / "watch-quantization-audits.py"
SPEC = importlib.util.spec_from_file_location("watch_quantization_audits", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PLAN_SHA256 = "3" * 64


def bound_report(block: object, **updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": MODULE.AUDIT_SCHEMA,
        "status": "complete",
        "block_namespace": block.namespace,
        "logical_layer": block.logical_layer,
        "plan_sha256": PLAN_SHA256,
        "projection_count": 768,
        "expected_projection_count": 768,
        "missing_projection_count": 0,
        "complete_expert_families": 256,
    }
    value.update(updates)
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    value["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return value


def report_matches(value: object, block: object) -> bool:
    return MODULE.complete_report_matches(
        value,
        block,
        plan_sha256=PLAN_SHA256,
        projections_per_block=768,
    )


def test_blocks_use_serial_base_then_mtp_commit_thresholds() -> None:
    work = MODULE.blocks(43, 3, 768)
    assert len(work) == 46
    assert work[0] == MODULE.Block("base", 0, 768)
    assert work[25] == MODULE.Block("base", 25, 26 * 768)
    assert work[42] == MODULE.Block("base", 42, 43 * 768)
    assert work[43] == MODULE.Block("mtp", 0, 44 * 768)
    assert work[-1] == MODULE.Block("mtp", 2, 46 * 768)


def test_mtp_overlay_scope_starts_commit_thresholds_at_one_block() -> None:
    counts = MODULE.counts_for_scope(43, 3, "mtp-routed-experts-only")
    assert counts == (0, 3)
    assert MODULE.blocks(*counts, 768) == [
        MODULE.Block("mtp", 0, 768),
        MODULE.Block("mtp", 1, 2 * 768),
        MODULE.Block("mtp", 2, 3 * 768),
    ]


def test_full_plan_scope_retains_base_and_mtp_blocks() -> None:
    assert MODULE.counts_for_scope(43, 3, None) == (43, 3)


def test_unknown_plan_scope_fails_closed() -> None:
    try:
        MODULE.counts_for_scope(43, 3, "base-only")
    except ValueError as error:
        assert "unsupported quantization plan scope" in str(error)
    else:
        raise AssertionError("unknown scope did not fail closed")


def test_journal_ready_waits_for_the_regular_file(monkeypatch) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    assert not MODULE.journal_ready("quantizer", "/run/state")
    assert commands == [
        [
            "docker",
            "exec",
            "quantizer",
            "test",
            "-f",
            "/run/state/.ds4rt-exl3-error-journal.jsonl",
        ]
    ]


def test_complete_report_requires_plan_counts_and_canonical_digest() -> None:
    block = MODULE.Block("base", 25, 26 * 768)
    assert report_matches(bound_report(block), block)

    for updates in (
        {"status": "partial"},
        {"plan_sha256": "4" * 64},
        {"projection_count": 767},
        {"expected_projection_count": 767},
        {"missing_projection_count": 1},
        {"complete_expert_families": 255},
        {"block_namespace": "mtp"},
        {"logical_layer": 24},
    ):
        assert not report_matches(bound_report(block, **updates), block)

    drifted = bound_report(block)
    drifted["encoded_bytes"] = 1
    assert not report_matches(drifted, block)


def test_partial_recovery_report_cannot_satisfy_completed_block() -> None:
    block = MODULE.Block("base", 26, 27 * 768)
    partial = bound_report(
        block,
        status="partial",
        projection_count=504,
        missing_projection_count=264,
        complete_expert_families=0,
    )
    assert not report_matches(partial, block)
