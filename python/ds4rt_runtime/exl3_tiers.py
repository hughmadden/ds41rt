"""Deterministic expert-family bitrate plans for mixed EXL3 Trellis artifacts.

An expert family is the indivisible ``(w1, w2, w3)`` unit.  Keeping all three
projections at the same integer Trellis width lets SparkInfer partition a MoE
block into dense bitrate tiers and execute it in one cooperative grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
import re
from typing import Any, Sequence


EXL3_TIER_PLAN_SCHEMA = "ds4rt.exl3.expert-family-bit-plan"
EXL3_TIER_PLAN_VERSION = 1
EXL3_BASE_BITS = 2
EXL3_PROMOTED_BITS = 3


def _nearest_integer(value: Decimal) -> int:
    """Round a non-negative Decimal to nearest, with exact halves upward."""

    if value < 0:
        raise ValueError("cannot round a negative expert count")
    return int((value + Decimal("0.5")).to_integral_value(rounding=ROUND_FLOOR))


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True)
class ExpertBitPlan:
    """Exact K2/K3 assignment for every routed expert family."""

    layer_count: int
    experts_per_layer: int
    target_bpw: str
    k3_experts_by_layer: tuple[tuple[int, ...], ...]
    selection_method: str
    score_kind: str | None = None
    score_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.layer_count <= 0 or self.experts_per_layer <= 0:
            raise ValueError("expert bit plan geometry must be positive")
        if len(self.k3_experts_by_layer) != self.layer_count:
            raise ValueError("expert bit plan must contain exactly one entry per layer")
        target = Decimal(self.target_bpw)
        if not Decimal(EXL3_BASE_BITS) <= target <= Decimal(EXL3_PROMOTED_BITS):
            raise ValueError("expert bit plan target_bpw must be in [2, 3]")
        if not self.selection_method:
            raise ValueError("expert bit plan selection_method must be non-empty")
        for layer_id, expert_ids in enumerate(self.k3_experts_by_layer):
            if tuple(sorted(set(expert_ids))) != expert_ids:
                raise ValueError(
                    f"layer {layer_id} K3 expert IDs must be sorted and unique"
                )
            if any(
                expert_id < 0 or expert_id >= self.experts_per_layer
                for expert_id in expert_ids
            ):
                raise ValueError(f"layer {layer_id} contains an out-of-range expert ID")
        if (self.score_kind is None) != (self.score_sha256 is None):
            raise ValueError("score_kind and score_sha256 must be present together")
        if self.score_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.score_sha256
        ):
            raise ValueError("expert bit plan score_sha256 must be lowercase SHA-256")

    @property
    def total_expert_families(self) -> int:
        return self.layer_count * self.experts_per_layer

    @property
    def k3_expert_families(self) -> int:
        return sum(len(expert_ids) for expert_ids in self.k3_experts_by_layer)

    @property
    def k2_expert_families(self) -> int:
        return self.total_expert_families - self.k3_expert_families

    @property
    def realized_bpw(self) -> float:
        return EXL3_BASE_BITS + self.k3_expert_families / self.total_expert_families

    def bits_for(self, layer_id: int, expert_id: int) -> int:
        if not 0 <= layer_id < self.layer_count:
            raise ValueError(f"layer {layer_id} is outside 0..{self.layer_count}")
        if not 0 <= expert_id < self.experts_per_layer:
            raise ValueError(
                f"expert {expert_id} is outside 0..{self.experts_per_layer}"
            )
        return (
            EXL3_PROMOTED_BITS
            if expert_id in self.k3_experts_by_layer[layer_id]
            else EXL3_BASE_BITS
        )

    def tier_experts(self, layer_id: int, bits: int) -> tuple[int, ...]:
        if bits not in {EXL3_BASE_BITS, EXL3_PROMOTED_BITS}:
            raise ValueError(f"unsupported EXL3 expert tier K={bits}")
        promoted = self.k3_experts_by_layer[layer_id]
        if bits == EXL3_PROMOTED_BITS:
            return promoted
        promoted_set = set(promoted)
        return tuple(
            expert_id
            for expert_id in range(self.experts_per_layer)
            if expert_id not in promoted_set
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXL3_TIER_PLAN_SCHEMA,
            "schema_version": EXL3_TIER_PLAN_VERSION,
            "base_bits": EXL3_BASE_BITS,
            "promoted_bits": EXL3_PROMOTED_BITS,
            "target_bpw": self.target_bpw,
            "realized_bpw": self.realized_bpw,
            "layer_count": self.layer_count,
            "experts_per_layer": self.experts_per_layer,
            "total_expert_families": self.total_expert_families,
            "k2_expert_families": self.k2_expert_families,
            "k3_expert_families": self.k3_expert_families,
            "selection_method": self.selection_method,
            "score_kind": self.score_kind,
            "score_sha256": self.score_sha256,
            "layers": [
                {
                    "layer_id": layer_id,
                    "k2_count": self.experts_per_layer - len(expert_ids),
                    "k3_count": len(expert_ids),
                    "k3_expert_ids": list(expert_ids),
                }
                for layer_id, expert_ids in enumerate(self.k3_experts_by_layer)
            ],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "ExpertBitPlan":
        if not isinstance(value, dict):
            raise ValueError("expert bit plan must be a JSON object")
        if value.get("schema") != EXL3_TIER_PLAN_SCHEMA:
            raise ValueError("unsupported expert bit plan schema")
        if value.get("schema_version") != EXL3_TIER_PLAN_VERSION:
            raise ValueError("unsupported expert bit plan schema version")
        if value.get("base_bits") != EXL3_BASE_BITS:
            raise ValueError("expert bit plan base tier must be K2")
        if value.get("promoted_bits") != EXL3_PROMOTED_BITS:
            raise ValueError("expert bit plan promoted tier must be K3")
        layer_count = int(value["layer_count"])
        layers = value.get("layers")
        if not isinstance(layers, list) or len(layers) != layer_count:
            raise ValueError("expert bit plan layers do not match layer_count")
        ids_by_layer: list[tuple[int, ...]] = []
        for layer_id, layer in enumerate(layers):
            if not isinstance(layer, dict) or layer.get("layer_id") != layer_id:
                raise ValueError(f"invalid expert bit plan layer {layer_id}")
            ids = tuple(int(expert_id) for expert_id in layer["k3_expert_ids"])
            if int(layer.get("k3_count", -1)) != len(ids):
                raise ValueError(f"layer {layer_id} K3 count does not match its IDs")
            ids_by_layer.append(ids)
        plan = cls(
            layer_count=layer_count,
            experts_per_layer=int(value["experts_per_layer"]),
            target_bpw=str(value["target_bpw"]),
            k3_experts_by_layer=tuple(ids_by_layer),
            selection_method=str(value["selection_method"]),
            score_kind=(
                None if value.get("score_kind") is None else str(value["score_kind"])
            ),
            score_sha256=(
                None
                if value.get("score_sha256") is None
                else str(value["score_sha256"])
            ),
        )
        expected = {
            "realized_bpw": plan.realized_bpw,
            "total_expert_families": plan.total_expert_families,
            "k2_expert_families": plan.k2_expert_families,
            "k3_expert_families": plan.k3_expert_families,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise ValueError(f"expert bit plan derived field {key} is inconsistent")
        return plan

    def gptqmodel_dynamic_overrides(self, *, hidden_layers: int) -> dict[str, dict[str, int]]:
        """Return exact integer-K module overrides for GPTQModel.

        GPTQModel traverses target blocks under ``model.layers`` and the
        disjoint dSpark auxiliary under ``mtp``. One anchored expression per
        promoted family covers all three projections and nothing else.
        """

        if not 0 <= hidden_layers <= self.layer_count:
            raise ValueError("hidden_layers is incompatible with the tier plan")
        overrides: dict[str, dict[str, int]] = {}
        projections = r"(?:gate_proj|up_proj|down_proj)"
        for layer_id, expert_ids in enumerate(self.k3_experts_by_layer):
            if layer_id < hidden_layers:
                block = rf"model\.layers\.{layer_id}"
            else:
                block = rf"mtp\.{layer_id - hidden_layers}"
            for expert_id in expert_ids:
                pattern = (
                    rf"^{block}\.mlp\.experts\.{expert_id}\.{projections}$"
                )
                overrides[pattern] = {"bits": EXL3_PROMOTED_BITS}
        return overrides


def build_expert_bit_plan(
    *,
    layer_count: int,
    experts_per_layer: int,
    target_bpw: float | str | Decimal,
    k3_scores: Sequence[Sequence[float]] | None = None,
    score_kind: str | None = None,
    allow_unscored_structural_plan: bool = False,
) -> ExpertBitPlan:
    """Allocate a balanced exact K2/K3 plan.

    Production mixed plans require one quality score per expert family. The
    score should estimate the validation loss reduction from K2 to K3; larger
    values are promoted first. Per-layer quotas differ by at most one, and the
    extra slots go to layers with the strongest next candidate.
    """

    if layer_count <= 0 or experts_per_layer <= 0:
        raise ValueError("expert bit plan geometry must be positive")
    target = Decimal(str(target_bpw))
    if not Decimal(EXL3_BASE_BITS) <= target <= Decimal(EXL3_PROMOTED_BITS):
        raise ValueError("target_bpw must be in [2, 3]")
    total = layer_count * experts_per_layer
    promoted_count = _nearest_integer(
        (target - Decimal(EXL3_BASE_BITS)) * Decimal(total)
    )

    if promoted_count == 0:
        return ExpertBitPlan(
            layer_count=layer_count,
            experts_per_layer=experts_per_layer,
            target_bpw=str(target),
            k3_experts_by_layer=tuple(() for _ in range(layer_count)),
            selection_method="uniform_k2",
        )

    if k3_scores is None and not allow_unscored_structural_plan:
        raise ValueError(
            "mixed K2/K3 production plans require per-expert K3 benefit scores"
        )
    if k3_scores is not None:
        if score_kind is None or not score_kind.strip():
            raise ValueError("scored expert bit plans require score_kind")
        if len(k3_scores) != layer_count:
            raise ValueError("K3 score matrix does not match layer_count")
        normalized_scores: tuple[tuple[float, ...], ...] = tuple(
            tuple(float(score) for score in layer) for layer in k3_scores
        )
        for layer_id, layer in enumerate(normalized_scores):
            if len(layer) != experts_per_layer:
                raise ValueError(
                    f"K3 scores for layer {layer_id} do not match experts_per_layer"
                )
            if any(not math.isfinite(score) for score in layer):
                raise ValueError(f"K3 scores for layer {layer_id} are non-finite")
        score_sha256 = hashlib.sha256(
            _canonical_json([list(layer) for layer in normalized_scores])
        ).hexdigest()
    else:
        normalized_scores = tuple(
            tuple(float(-expert_id) for expert_id in range(experts_per_layer))
            for _ in range(layer_count)
        )
        score_sha256 = None

    base_quota, extra_layers = divmod(promoted_count, layer_count)
    if base_quota > experts_per_layer or (
        base_quota == experts_per_layer and extra_layers
    ):
        raise AssertionError("computed K3 quota exceeds expert geometry")
    ranked_by_layer = tuple(
        tuple(
            sorted(
                range(experts_per_layer),
                key=lambda expert_id: (-layer[expert_id], expert_id),
            )
        )
        for layer in normalized_scores
    )
    extra_layer_ids: set[int] = set()
    if extra_layers:
        marginal = sorted(
            range(layer_count),
            key=lambda layer_id: (
                -normalized_scores[layer_id][ranked_by_layer[layer_id][base_quota]],
                layer_id,
            ),
        )
        extra_layer_ids.update(marginal[:extra_layers])
    selected = tuple(
        tuple(
            sorted(
                ranked_by_layer[layer_id][
                    : base_quota + (1 if layer_id in extra_layer_ids else 0)
                ]
            )
        )
        for layer_id in range(layer_count)
    )
    return ExpertBitPlan(
        layer_count=layer_count,
        experts_per_layer=experts_per_layer,
        target_bpw=str(target),
        k3_experts_by_layer=selected,
        selection_method=(
            "activation_weighted_k3_minus_k2_gain_balanced_by_layer"
            if k3_scores is not None
            else "structural_only_balanced_by_layer_not_for_production"
        ),
        score_kind=None if k3_scores is None else score_kind.strip(),
        score_sha256=score_sha256,
    )
