from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_expert_reduction_replay as analyze  # noqa: E402
import collect_expert_route_bank as collect  # noqa: E402
import plan_expert_reduction_replay as plan  # noqa: E402


def flash_manifest() -> dict[str, object]:
    return {
        "record": "manifest",
        "schema": "ds4rt-expert-route-bank-v2",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "hidden_size": 4096,
        "top_k": 6,
        "routed_experts": 256,
        "quantization_recipe": "deepseek_v4_native_fp4_fp8_mixed_v1",
        "routed_layers": [0, 1],
    }


def test_route_trace_uses_deepseek_top6_and_expert_bounds() -> None:
    line = (
        "protocol_v2_expert_queue_row_routes request_id_base=1 layer_id=0 "
        "rows=1 source_kinds=MtpVerifyBlock row_routes=0:0+1+2+3+4+5"
    )
    fragment = collect.parse_route_line(line, top_k=6, routed_experts=256)
    assert fragment is not None
    assert fragment.routes == ((0, 1, 2, 3, 4, 5),)

    with pytest.raises(ValueError, match="top-8"):
        collect.parse_route_line(line, top_k=8, routed_experts=256)
    with pytest.raises(ValueError, match="outside 0..4"):
        collect.parse_route_line(line, top_k=6, routed_experts=5)
    with pytest.raises(ValueError, match="repeats row index 0"):
        collect.parse_route_line(
            line + ",0:6+7+8+9+10+11", top_k=6, routed_experts=256
        )


def test_plan_preserves_explicit_deepseek_geometry(tmp_path: Path) -> None:
    bank = tmp_path / "routes.jsonl"
    records = [flash_manifest()]
    for layer_id in (0, 1):
        records.append(
            {
                "record": "fragment",
                "output_id": "code-000",
                "case": "code",
                "cycle": 0,
                "physical_m": 1,
                "layer_id": layer_id,
                "routes": [[0, 1, 2, 3, 4, 5]],
            }
        )
    bank.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    _, geometry, cycles = plan.load_cycles(bank)
    assert geometry.hidden_size == 4096
    assert geometry.top_k == 6
    assert geometry.routed_layers == (0, 1)
    chain = plan.chain_record(
        geometry,
        "semantic-m001-0000",
        "semantic",
        1,
        [plan.CyclePiece(cycles[0], 0, 1)],
    )
    assert [layer["layer_id"] for layer in chain["layers"]] == [0, 1]
    assert all(len(row) == 6 for layer in chain["layers"] for row in layer["routes"])


def test_plan_rejects_legacy_implicit_geometry() -> None:
    legacy = flash_manifest()
    legacy["schema"] = "ds4rt-expert-route-bank-v1"
    with pytest.raises(ValueError, match="explicit DeepSeek model geometry"):
        plan.geometry_from_manifest(legacy)


def test_plan_rejects_duplicate_experts_in_a_route(tmp_path: Path) -> None:
    bank = tmp_path / "duplicate-routes.jsonl"
    records = [flash_manifest()]
    for layer_id in (0, 1):
        records.append(
            {
                "record": "fragment",
                "output_id": "code-000",
                "case": "code",
                "cycle": 0,
                "physical_m": 1,
                "layer_id": layer_id,
                "routes": [[0, 1, 2, 3, 4, 4]],
            }
        )
    bank.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repeats an expert ID"):
        plan.load_cycles(bank)


def test_reduction_summary_uses_recorded_layer_samples() -> None:
    measurements = [
        {
            "physical_m": 12,
            "chain_id": "chain",
            "path": "coordinator",
            "path_order": 0,
            "dispatch_ms": 4.0,
            "layer_ms": [1.0, 3.0],
        },
        {
            "physical_m": 12,
            "chain_id": "chain",
            "path": "spark-row-sharded",
            "path_order": 1,
            "dispatch_ms": 6.0,
            "layer_ms": [2.0, 4.0],
        },
    ]
    pairs = analyze.paired_by_m(measurements)
    summary = analyze.summarize(
        {
            "schema": "ds4rt-expert-reduction-replay-result-v2",
            "cohort": "semantic",
            **{
                key: value
                for key, value in flash_manifest().items()
                if key not in {"record", "schema"}
            },
        },
        pairs,
        bootstrap_samples=8,
        seed=1,
    )
    row = summary["rows"][0]
    assert row["coordinator_layer_ms_mean"] == 2.0
    assert row["spark_layer_ms_mean"] == 3.0
    assert summary["schema"] == "ds4rt-expert-reduction-replay-summary-v2"
    assert summary["hidden_size"] == 4096
    assert summary["routed_layers"] == [0, 1]
