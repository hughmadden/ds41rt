from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ds4rt_runtime.exl3_mix import (
    MIX_PLAN_META_KEY,
    MIX_SCORE_KIND,
    build_mixed_quantization_config,
    mixed_expert_tensor_identity,
    score_k2_k3_ledgers,
)
from ds4rt_runtime.exl3_tiers import build_expert_bit_plan


def _record(bits: int, layer: int, expert: int, projection: str, error: float) -> dict:
    stable = {
        "calibration_evidence": {"id": "cal"},
        "codebook": "mcg",
        "corpus": {"id": "corpus"},
        "gptqmodel": {"revision": "same"},
        "operator_contract": "operator",
        "quantizer_numerics": {"sigma": 0.025},
        "quantizer_seed": 787,
        "route_evidence_contract": "natural",
        "source": {
            "revision": "source",
            "geometry": {"num_hidden_layers": 2},
        },
        "zero_route_recovery_contract": "recovery",
    }
    return {
        "record_kind": "projection",
        "bits": bits,
        "codebook": "mcg",
        "block_namespace": "base",
        "logical_layer": layer,
        "processor_layer_index": layer,
        "expert": expert,
        "projection": projection,
        "provenance": {"family_join": {**stable, "bits": bits}},
        "quantizer_metrics": {
            "hessian_metric_status": "ok",
            "reported_metric_kind": "hessian_weighted_relative_error",
            "hessian_weighted_relative_error": error,
        },
        "route_evidence": {
            "expert_gate_squared_mass_fraction": (expert + 1) / 3,
        },
    }


def _ledger(path: Path, bits: int, errors: list[list[float]]) -> None:
    records = []
    for layer, row in enumerate(errors):
        for expert, error in enumerate(row):
            for projection in ("w1", "w2", "w3"):
                records.append(_record(bits, layer, expert, projection, error))
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_scores_and_balances_exact_family_plan(tmp_path: Path) -> None:
    k2 = tmp_path / "k2.jsonl"
    k3 = tmp_path / "k3.jsonl"
    _ledger(k2, 2, [[0.4, 0.3], [0.2, 0.1]])
    _ledger(k3, 3, [[0.1, 0.2], [0.1, 0.05]])

    evidence = score_k2_k3_ledgers(k2, k3, layer_count=2, experts_per_layer=2)
    assert evidence.scores[0] == pytest.approx((0.3, 0.2))
    assert evidence.scores[1] == pytest.approx((0.1, 0.1))
    plan = evidence.plan(target_bpw="2.5")
    assert plan.score_kind == MIX_SCORE_KIND
    assert plan.k3_expert_families == 2
    assert [len(row) for row in plan.k3_experts_by_layer] == [1, 1]
    report = evidence.summary(plan)
    assert report["plan_sha256"] == plan.sha256
    assert report["k2_ledger_sha256"] == hashlib.sha256(k2.read_bytes()).hexdigest()


def test_rejects_route_evidence_mismatch(tmp_path: Path) -> None:
    k2 = tmp_path / "k2.jsonl"
    k3 = tmp_path / "k3.jsonl"
    _ledger(k2, 2, [[0.4]])
    _ledger(k3, 3, [[0.1]])
    lines = k3.read_text().splitlines()
    record = json.loads(lines[0])
    record["route_evidence"]["router_token_count"] = 99
    lines[0] = json.dumps(record)
    k3.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="route-selection identity differs"):
        score_k2_k3_ledgers(k2, k3, layer_count=1, experts_per_layer=1)


def test_builds_per_module_mixed_tensor_storage() -> None:
    plan = build_expert_bit_plan(
        layer_count=2,
        experts_per_layer=2,
        target_bpw="2.5",
        k3_scores=((4.0, 1.0), (3.0, 2.0)),
        score_kind=MIX_SCORE_KIND,
    )
    selection = {
        "plan": plan.to_dict(),
        "plan_sha256": plan.sha256,
        "k2_ledger_sha256": "2" * 64,
        "k3_ledger_sha256": "3" * 64,
    }
    common = {
        "quant_method": "exl3",
        "method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "codebook": "mcg",
        "meta": {"fallback": None},
    }
    storage = {}
    for layer in range(2):
        for expert in range(2):
            for projection in ("gate_proj", "down_proj", "up_proj"):
                storage[f"model.layers.{layer}.mlp.experts.{expert}.{projection}"] = {
                    "quant_format": "exl3"
                }
    tiers = {
        bits: {
            **common,
            "bits": float(bits),
            "tensor_storage": {
                module: {**entry, "bits_per_weight": bits}
                for module, entry in storage.items()
            },
        }
        for bits in (2, 3)
    }
    mixed = build_mixed_quantization_config(
        k2_quant=tiers[2], k3_quant=tiers[3], plan=plan, selection=selection
    )
    assert mixed["bits"] == 2.5
    assert mixed["meta"][MIX_PLAN_META_KEY]["selection"]["plan_sha256"] == plan.sha256
    for module, entry in mixed["tensor_storage"].items():
        identity = mixed_expert_tensor_identity(f"{module}.trellis", hidden_layers=2)
        assert identity is not None
        assert entry["bits_per_weight"] == plan.bits_for(identity[0], identity[1])


def test_maps_disjoint_mtp_tensor_namespace() -> None:
    assert mixed_expert_tensor_identity(
        "mtp.2.mlp.experts.7.down_proj.trellis", hidden_layers=43
    ) == (45, 7, "down_proj", "trellis")
