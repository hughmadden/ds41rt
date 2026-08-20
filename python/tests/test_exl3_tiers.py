from __future__ import annotations

import re

import pytest

from ds4rt_runtime.exl3_tiers import ExpertBitPlan, build_expert_bit_plan


def test_flash_215_plan_has_the_nearest_exact_balanced_mix() -> None:
    plan = build_expert_bit_plan(
        layer_count=46,
        experts_per_layer=256,
        target_bpw="2.15",
        allow_unscored_structural_plan=True,
    )

    assert plan.total_expert_families == 11_776
    assert plan.k3_expert_families == 1_766
    assert plan.k2_expert_families == 10_010
    assert plan.realized_bpw == pytest.approx(2.1499660326086956)
    quotas = [len(expert_ids) for expert_ids in plan.k3_experts_by_layer]
    assert quotas.count(39) == 18
    assert quotas.count(38) == 28


def test_current_pro_preview_215_plan_has_the_nearest_exact_mix() -> None:
    plan = build_expert_bit_plan(
        layer_count=61,
        experts_per_layer=384,
        target_bpw="2.15",
        allow_unscored_structural_plan=True,
    )

    assert plan.total_expert_families == 23_424
    assert plan.k3_expert_families == 3_514
    assert plan.realized_bpw == pytest.approx(2.150017076502732)
    quotas = [len(expert_ids) for expert_ids in plan.k3_experts_by_layer]
    assert quotas.count(58) == 37
    assert quotas.count(57) == 24


def test_mixed_production_plan_requires_scores_and_uses_k3_gain() -> None:
    with pytest.raises(ValueError, match="require per-expert K3 benefit scores"):
        build_expert_bit_plan(
            layer_count=2,
            experts_per_layer=4,
            target_bpw="2.25",
        )

    # Two promotions total, balanced one per layer. Layer 1 receives the extra
    # marginal slot only in larger examples; here the best expert in each layer
    # demonstrates quality-ranked selection and deterministic tie breaking.
    plan = build_expert_bit_plan(
        layer_count=2,
        experts_per_layer=4,
        target_bpw="2.25",
        k3_scores=((0.1, 9.0, 1.0, 0.0), (3.0, 2.0, 1.0, 0.0)),
        score_kind="measured_activation_weighted_k3_minus_k2_proxy_gain",
    )
    assert plan.k3_experts_by_layer == ((1,), (0,))
    assert plan.score_sha256 is not None
    assert plan.bits_for(0, 1) == 3
    assert plan.bits_for(0, 0) == 2


def test_extra_layer_quota_uses_strongest_marginal_gain() -> None:
    # Five promotions across two layers means quotas 2 and 3. The third-ranked
    # candidate in layer 1 is stronger, so it receives the extra K3 slot.
    plan = build_expert_bit_plan(
        layer_count=2,
        experts_per_layer=4,
        target_bpw="2.625",
        k3_scores=((10.0, 9.0, 0.1, 0.0), (8.0, 7.0, 6.0, 0.0)),
        score_kind="test_gain",
    )
    assert plan.k3_experts_by_layer == ((0, 1), (0, 1, 2))


def test_plan_round_trip_hash_and_gptqmodel_family_overrides() -> None:
    plan = ExpertBitPlan(
        layer_count=3,
        experts_per_layer=4,
        target_bpw="2.1666666666666667",
        k3_experts_by_layer=((2,), (), (1,)),
        selection_method="test",
    )
    payload = plan.to_dict()
    restored = ExpertBitPlan.from_dict(payload)
    assert restored == plan
    assert restored.sha256 == plan.sha256

    overrides = plan.gptqmodel_dynamic_overrides(hidden_layers=2)
    assert len(overrides) == 2
    compiled = [(re.compile(pattern), value) for pattern, value in overrides.items()]

    def bits_for(name: str) -> int:
        matches = [value["bits"] for pattern, value in compiled if pattern.match(name)]
        assert len(matches) <= 1
        return matches[0] if matches else 2

    for projection in ("gate_proj", "up_proj", "down_proj"):
        assert bits_for(f"model.layers.0.mlp.experts.2.{projection}") == 3
        assert bits_for(f"mtp.0.mlp.experts.1.{projection}") == 3
    assert bits_for("model.layers.0.mlp.experts.1.gate_proj") == 2
    assert bits_for("model.layers.0.mlp.shared_experts.gate_proj") == 2


def test_plan_round_trip_rejects_derived_count_tampering() -> None:
    plan = build_expert_bit_plan(
        layer_count=2,
        experts_per_layer=4,
        target_bpw=2,
    )
    payload = plan.to_dict()
    payload["k2_expert_families"] = 7
    with pytest.raises(ValueError, match="derived field k2_expert_families"):
        ExpertBitPlan.from_dict(payload)
