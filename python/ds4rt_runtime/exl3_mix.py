"""Provenance-bound K2/K3 expert-family scoring for mixed EXL3 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from ds4rt_runtime.exl3_tiers import ExpertBitPlan, build_expert_bit_plan


MIX_SCORE_KIND = (
    "hessian-relative-k3-gain-times-k2-prefix-natural-gate-squared-mass-v1"
)
MIX_PLAN_META_KEY = "ds4rt_expert_bit_plan"
MIX_RECIPE = "deepseek_v4_exl3_trellis_mixed_k2_k3_v1"
_EXPERT_TENSOR_RE = re.compile(
    r"^(?:(?:model\.)?layers\.(?P<base>\d+)|mtp\.(?P<mtp>\d+))\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<suffix>trellis|suh|svh|mcg)$"
)
_FAMILY_JOIN_KEYS = (
    "calibration_evidence",
    "codebook",
    "corpus",
    "quantizer_numerics",
    "quantizer_seed",
    "route_evidence_contract",
    "source",
    "zero_route_recovery_contract",
)
_ROUTE_IDENTITY_KEYS = (
    "schema",
    "schema_version",
    "block_namespace",
    "logical_layer",
    "expert",
    "router_calls",
    "router_selected_route_count",
    "router_token_count",
    "router_top_k",
    "mask_modes",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_family_join(record: dict[str, Any]) -> dict[str, Any]:
    try:
        family = record["provenance"]["family_join"]
    except (KeyError, TypeError) as error:
        raise ValueError("EXL3 ledger record has no family-join provenance") from error
    if not isinstance(family, dict):
        raise ValueError("EXL3 family-join provenance must be an object")
    # The separately calibrated MTP namespace has no base router-screen
    # calibration_evidence object; its corpus/prefix/anchor identities are
    # carried elsewhere. Preserve the distinction while comparing the fields
    # shared by like namespaces across K2 and K3.
    return {key: family.get(key) for key in _FAMILY_JOIN_KEYS}


@dataclass(frozen=True)
class MixScoreEvidence:
    """Complete score matrix and its source identities."""

    scores: tuple[tuple[float, ...], ...]
    k2_ledger_sha256: str
    k3_ledger_sha256: str
    family_join_sha256: str
    positive_gain_families: int
    total_families: int
    total_weighted_gain: float

    @property
    def score_sha256(self) -> str:
        return hashlib.sha256(_canonical([list(row) for row in self.scores])).hexdigest()

    def plan(self, *, target_bpw: float | str) -> ExpertBitPlan:
        return build_expert_bit_plan(
            layer_count=len(self.scores),
            experts_per_layer=len(self.scores[0]),
            target_bpw=target_bpw,
            k3_scores=self.scores,
            score_kind=MIX_SCORE_KIND,
        )

    def summary(self, plan: ExpertBitPlan) -> dict[str, Any]:
        selected_gain = sum(
            self.scores[layer_id][expert_id]
            for layer_id, expert_ids in enumerate(plan.k3_experts_by_layer)
            for expert_id in expert_ids
        )
        unconstrained = sorted(
            (score for row in self.scores for score in row), reverse=True
        )[: plan.k3_expert_families]
        unconstrained_gain = sum(unconstrained)
        return {
            "schema": "ds4rt.exl3-mix-selection-v1",
            "score_kind": MIX_SCORE_KIND,
            "score_sha256": self.score_sha256,
            "k2_ledger_sha256": self.k2_ledger_sha256,
            "k3_ledger_sha256": self.k3_ledger_sha256,
            "family_join_sha256": self.family_join_sha256,
            "positive_gain_families": self.positive_gain_families,
            "total_families": self.total_families,
            "total_weighted_gain": self.total_weighted_gain,
            "selected_weighted_gain": selected_gain,
            "selected_gain_fraction": (
                selected_gain / self.total_weighted_gain
                if self.total_weighted_gain
                else 0.0
            ),
            "unconstrained_selected_weighted_gain": unconstrained_gain,
            "balanced_vs_unconstrained_fraction": (
                selected_gain / unconstrained_gain if unconstrained_gain else 0.0
            ),
            "plan": plan.to_dict(),
            "plan_sha256": plan.sha256,
        }


def score_k2_k3_ledgers(
    k2_ledger: str | Path,
    k3_ledger: str | Path,
    *,
    layer_count: int,
    experts_per_layer: int,
) -> MixScoreEvidence:
    """Join exact projection records and compute the preferred family score.

    The three projection relative errors are summed before multiplying by the
    family's natural squared-gate-mass fraction.  This avoids promoting an
    expert merely because it was routed often or merely because one large but
    practically cold matrix has high reconstruction error.
    """

    paths = {2: Path(k2_ledger).resolve(), 3: Path(k3_ledger).resolve()}
    records: dict[int, dict[tuple[int, int, str], tuple[float, dict[str, Any]]]] = {
        2: {},
        3: {},
    }
    stable_joins: dict[int, dict[str, dict[str, Any]]] = {2: {}, 3: {}}
    for bits, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing K{bits} EXL3 ledger: {path}")
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error
                if record.get("record_kind") != "projection":
                    continue
                if record.get("bits") != bits or record.get("codebook") != "mcg":
                    raise ValueError(f"{path}:{line_number} is not an MCG K{bits} record")
                stable = _stable_family_join(record)
                try:
                    base_layers = int(stable["source"]["geometry"]["num_hidden_layers"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{line_number} has no source base-layer geometry"
                    ) from error
                logical_layer = record.get("logical_layer")
                namespace = record.get("block_namespace")
                # GPTQModel's logical and processor layer indices both restart
                # at zero in the disjoint MTP namespace.
                layer_id = (
                    logical_layer
                    if namespace == "base"
                    else base_layers + logical_layer
                    if namespace == "mtp" and isinstance(logical_layer, int)
                    else None
                )
                expert_id = record.get("expert")
                projection = record.get("projection")
                if (
                    isinstance(layer_id, bool)
                    or not isinstance(layer_id, int)
                    or not 0 <= layer_id < layer_count
                    or isinstance(expert_id, bool)
                    or not isinstance(expert_id, int)
                    or not 0 <= expert_id < experts_per_layer
                    or projection not in {"w1", "w2", "w3"}
                ):
                    raise ValueError(f"{path}:{line_number} has invalid family identity")
                metrics = record.get("quantizer_metrics")
                routes = record.get("route_evidence")
                if not isinstance(metrics, dict) or not isinstance(routes, dict):
                    raise ValueError(f"{path}:{line_number} has incomplete metrics")
                relative_error = metrics.get("hessian_weighted_relative_error")
                if (
                    metrics.get("hessian_metric_status") != "ok"
                    or metrics.get("reported_metric_kind")
                    != "hessian_weighted_relative_error"
                    or not isinstance(relative_error, (int, float))
                    or not math.isfinite(float(relative_error))
                    or float(relative_error) < 0
                ):
                    raise ValueError(f"{path}:{line_number} has unusable Hessian error")
                identity = (layer_id, expert_id, projection)
                if identity in records[bits]:
                    raise ValueError(f"duplicate K{bits} projection record {identity}")
                records[bits][identity] = (float(relative_error), routes)
                previous_stable = stable_joins[bits].get(str(namespace))
                if previous_stable is None:
                    stable_joins[bits][str(namespace)] = stable
                elif previous_stable != stable:
                    raise ValueError(
                        f"K{bits} ledger changes {namespace} family-join provenance"
                    )

    expected = layer_count * experts_per_layer * 3
    if any(len(tier) != expected for tier in records.values()):
        raise ValueError("K2/K3 projection ledgers do not cover the requested geometry")
    if records[2].keys() != records[3].keys():
        raise ValueError("K2/K3 projection ledger identities differ")
    if not stable_joins[2] or stable_joins[2] != stable_joins[3]:
        raise ValueError("K2/K3 ledgers do not share calibration/source provenance")

    scores = [[0.0 for _ in range(experts_per_layer)] for _ in range(layer_count)]
    positive = 0
    total_gain = 0.0
    for layer_id in range(layer_count):
        for expert_id in range(experts_per_layer):
            gain = 0.0
            route_evidence = None
            for projection in ("w1", "w2", "w3"):
                k2_error, k2_routes = records[2][(layer_id, expert_id, projection)]
                k3_error, k3_routes = records[3][(layer_id, expert_id, projection)]
                if any(
                    k2_routes.get(key) != k3_routes.get(key)
                    for key in _ROUTE_IDENTITY_KEYS
                ):
                    raise ValueError(
                        "K2/K3 route-selection identity differs for "
                        f"{(layer_id, expert_id, projection)}"
                    )
                if route_evidence is None:
                    route_evidence = k2_routes
                elif route_evidence != k2_routes:
                    raise ValueError(
                        f"projection route evidence differs within family {(layer_id, expert_id)}"
                    )
                gain += k2_error - k3_error
            # The final 2.1 model is 90% K2 families, so K2-prefix routing is
            # the less biased deployment prior. K3's per-route gate values can
            # differ slightly even when the exact route identities/counts are
            # unchanged because its replayed hidden prefix is higher quality.
            mass = route_evidence.get("expert_gate_squared_mass_fraction")
            if (
                not isinstance(mass, (int, float))
                or not math.isfinite(float(mass))
                or not 0.0 <= float(mass) <= 1.0
            ):
                raise ValueError(f"family {(layer_id, expert_id)} has invalid gate-squared mass")
            score = gain * float(mass)
            scores[layer_id][expert_id] = score
            total_gain += score
            positive += score > 0

    return MixScoreEvidence(
        scores=tuple(tuple(row) for row in scores),
        k2_ledger_sha256=_sha256_file(paths[2]),
        k3_ledger_sha256=_sha256_file(paths[3]),
        family_join_sha256=hashlib.sha256(_canonical(stable_joins[2])).hexdigest(),
        positive_gain_families=positive,
        total_families=layer_count * experts_per_layer,
        total_weighted_gain=total_gain,
    )


def mixed_expert_tensor_identity(
    name: str, *, hidden_layers: int
) -> tuple[int, int, str, str] | None:
    """Return global layer, expert, projection and suffix for one EXL3 tensor."""

    match = _EXPERT_TENSOR_RE.fullmatch(name)
    if match is None:
        return None
    layer_id = (
        int(match.group("base"))
        if match.group("base") is not None
        else hidden_layers + int(match.group("mtp"))
    )
    return (
        layer_id,
        int(match.group("expert")),
        match.group("projection"),
        match.group("suffix"),
    )


def build_mixed_quantization_config(
    *,
    k2_quant: dict[str, Any],
    k3_quant: dict[str, Any],
    plan: ExpertBitPlan,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Join two uniform GPTQModel configs under an exact family bit plan."""

    if selection.get("plan") != plan.to_dict() or selection.get("plan_sha256") != plan.sha256:
        raise ValueError("mixed selection manifest does not bind its expert bit plan")
    for bits, quant in ((2, k2_quant), (3, k3_quant)):
        if (
            not isinstance(quant, dict)
            or float(quant.get("bits", -1)) != bits
            or quant.get("codebook") != "mcg"
            or not isinstance(quant.get("tensor_storage"), dict)
            or not quant["tensor_storage"]
        ):
            raise ValueError(f"mixed source K{bits} quantization config is invalid")
    if set(k2_quant["tensor_storage"]) != set(k3_quant["tensor_storage"]):
        raise ValueError("mixed K2/K3 tensor-storage module sets differ")

    # The module keys themselves make the boundary unambiguous: base layers
    # are numbered densely from zero, followed by the disjoint MTP namespace.
    base_ids = {
        int(match.group("base"))
        for module in k2_quant["tensor_storage"]
        if (match := _EXPERT_TENSOR_RE.fullmatch(f"{module}.trellis")) is not None
        and match.group("base") is not None
    }
    if not base_ids or base_ids != set(range(max(base_ids) + 1)):
        raise ValueError("mixed tensor storage has non-contiguous base layers")
    hidden_layers = max(base_ids) + 1

    storage: dict[str, Any] = {}
    for module in sorted(k2_quant["tensor_storage"]):
        identity = mixed_expert_tensor_identity(
            f"{module}.trellis", hidden_layers=hidden_layers
        )
        if identity is None:
            raise ValueError(f"mixed tensor-storage module is not routed: {module}")
        layer_id, expert_id, _, _ = identity
        bits = plan.bits_for(layer_id, expert_id)
        source = k2_quant if bits == 2 else k3_quant
        entry = source["tensor_storage"][module]
        if entry.get("bits_per_weight") != bits:
            raise ValueError(f"K{bits} tensor storage disagrees for {module}")
        storage[module] = entry

    result = {
        key: value
        for key, value in k2_quant.items()
        if key not in {"bits", "tensor_storage", "meta"}
    }
    result["bits"] = plan.realized_bpw
    result["tensor_storage"] = storage
    result["meta"] = {
        "fallback": None,
        MIX_PLAN_META_KEY: {
            "schema": "ds4rt.exl3-mixed-k2-k3-v1",
            "recipe": MIX_RECIPE,
            "selection_sha256": hashlib.sha256(_canonical(selection)).hexdigest(),
            "selection": selection,
        },
        "ds4rt_error_ledger": {
            "schema": "ds4rt.exl3-mixed-error-ledger-provenance-v1",
            "selection_sha256": hashlib.sha256(_canonical(selection)).hexdigest(),
            "k2_ledger_sha256": selection["k2_ledger_sha256"],
            "k3_ledger_sha256": selection["k3_ledger_sha256"],
        },
    }
    return result


__all__ = [
    "MIX_PLAN_META_KEY",
    "MIX_RECIPE",
    "MIX_SCORE_KIND",
    "MixScoreEvidence",
    "build_mixed_quantization_config",
    "mixed_expert_tensor_identity",
    "score_k2_k3_ledgers",
]
