"""Calibrated integer-tier EXL3 DeepSeek V4 expert execution with strict TP=4.

The artifact stores native EXL3 tensors, and SparkInfer consumes those tensors
without repacking them.  Every Spark keeps every global expert and slices only
the intermediate dimension: FC1 slices the trellis N tiles and FC2 slices its
K tiles.  The three intermediate rotation vectors are sliced identically, so
the four rank-local FP32 outputs add to the unsharded expert result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

from ds41rt_runtime.exl3_artifact_contract import (
    is_gptqmodel_native_exl3,
    validate_gptqmodel_native_exl3,
    validate_gptqmodel_publication,
)
from ds41rt_runtime.exl3_quantizer import (
    EXL3_ACTIVATION_RECIPE,
    EXL3_BITS,
    EXL3_CODEBOOK,
    EXL3_RECIPE,
    EXL3_FORCED_ACTIVATION_RECIPE,
    EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME,
    EXL3_SCHEMA,
    EXL3_SCHEMA_VERSION,
    EXL3_TENSOR_FORMAT,
    EXLLAMAV3_REPOSITORY,
    EXLLAMAV3_REVISION,
    EXLLAMAV3_SOURCE_TREE_SHA256,
    EXLLAMAV3_VERSION,
    EXPERT_TP_WORLD_SIZE,
    MCG_MARKER,
    MIN_NATURAL_ROUTES_PER_EXPERT,
    NATIVE_SOURCE_FORMAT,
)
from ds41rt_runtime.native_experts import (
    CHECKPOINT_NATIVE_EXPERT_LAYOUT,
    GPTQMODEL_EXPERT_LAYOUT,
    NativeExpertConfig,
    TpIntermediateSlice,
    replicated_expert_ids,
    tp_intermediate_slice,
)


SPARKINFER_SOURCE_FORMAT = "exl3_trellis_mcg"
EXL3_ROTATION_BLOCK = 128
# Projection-major Trellis is consumed directly by the kernel. Keep this
# separate from native FP4's `[up; gate]` pre-repack source contract.
EXL3_W13_PROJECTION_LAYOUT = (("gate", 0, 0), ("up", 1, 1))


def _native_expert_config_from_raw(
    raw: dict[str, Any],
    *,
    expert_tensor_layout: str = CHECKPOINT_NATIVE_EXPERT_LAYOUT,
) -> NativeExpertConfig:
    hidden = int(raw["hidden_size"])
    intermediate = int(raw["moe_intermediate_size"])
    if hidden % EXL3_ROTATION_BLOCK:
        raise ValueError(f"hidden_size={hidden} must be divisible by EXL3 H128")
    if intermediate % (EXPERT_TP_WORLD_SIZE * EXL3_ROTATION_BLOCK):
        raise ValueError(
            f"intermediate_size={intermediate} must be divisible by TP4 H128"
        )
    target_layers = tuple(int(value) for value in raw["dspark_target_layer_ids"])
    config = NativeExpertConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=int(raw["num_hidden_layers"]),
        dspark_blocks=len(target_layers),
        global_experts=int(raw["n_routed_experts"]),
        top_k=int(raw["num_experts_per_tok"]),
        swiglu_limit=float(raw["swiglu_limit"]),
        expert_tensor_layout=expert_tensor_layout,
    )
    if config.global_experts <= 0 or config.top_k != 6:
        raise ValueError(
            f"DeepSeek V4 EXL3 runtime requires routed experts and top-k 6, got "
            f"experts={config.global_experts} top_k={config.top_k}"
        )
    return config


@dataclass
class Exl3ExpertTpLayer:
    config: NativeExpertConfig
    layer_id: int
    tp_slice: TpIntermediateSlice
    global_expert_ids: tuple[int, ...]
    experts: Any
    route_expert_map: Any
    weight_plan: Any
    source_bytes: int
    load_seconds: float
    projection_mixed: bool = False

    @property
    def local_experts(self) -> int:
        return len(self.global_expert_ids)

    @property
    def local_intermediate_size(self) -> int:
        return self.tp_slice.size

    def plan_tp(self, *, max_tokens: int, route_ids_dtype=None):
        """Allocate fixed launch scratch before graph capture."""

        torch = _torch()
        from b12x.moe import fused_moe

        mixed_plan_hints = {}
        if self.projection_mixed and route_ids_dtype is not None:
            prepared = self.experts.representation_for("w4a16")
            mixed_plan_hints = {
                "mixed_trellis_route_id_dtypes": (route_ids_dtype,),
                "mixed_trellis_broadcast_suh": (
                    int(prepared.rotations.gate_suh.shape[0]) == 1,
                ),
                "mixed_trellis_broadcast_svh": (
                    int(prepared.rotations.down_svh.shape[0]) == 1,
                ),
            }

        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=max_tokens,
                num_topk=self.config.top_k,
                route_num_experts=(
                    self.local_experts
                    if self.projection_mixed
                    else self.config.global_experts
                ),
                device=self.route_expert_map.device,
                weight_plan=self.weight_plan,
                quant_mode="w4a16",
                swiglu_limit=self.config.swiglu_limit,
                **mixed_plan_hints,
            )
        )
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        return plan, scratch

    def run_partial(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        *,
        plan=None,
        scratch=None,
        output=None,
        fast_math: bool = True,
    ):
        """Execute the identical global routes on this TP intermediate slice."""

        torch = _torch()
        from b12x.moe import fused_moe

        if plan is None or scratch is None:
            plan, scratch = self.plan_tp(
                max_tokens=int(hidden_states.shape[0]),
                route_ids_dtype=topk_ids.dtype,
            )
        if output is None:
            # Full EXL3 rotations accumulate the route sum in FP32.  Transport
            # policy decides whether that rank partial is encoded as BF16 or
            # FP8; it must not silently narrow inside the expert kernel.
            output = torch.empty(
                hidden_states.shape,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        route_expert_map = self.route_expert_map
        if self.projection_mixed:
            # Production owns all 256 replicated experts and therefore takes
            # this identity fast path.  Small validation oracles load only six
            # experts, so remap their global route IDs before entering the
            # projection-mixed kernel, whose descriptor table is layer-local.
            identity = (
                self.local_experts == self.config.global_experts
                and self.global_expert_ids
                == tuple(range(self.config.global_experts))
            )
            if identity:
                route_expert_map = None
            else:
                local_ids = route_expert_map.index_select(
                    0, topk_ids.reshape(-1).to(torch.int64)
                ).view_as(topk_ids)
                if torch.any(local_ids < 0).item():
                    raise ValueError("routes reference an expert absent from this layer")
                topk_ids = local_ids.to(dtype=topk_ids.dtype)
                route_expert_map = None
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=hidden_states,
            experts=self.experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            route_expert_map=route_expert_map,
            output=output,
            input_scales_static=True,
            fast_math=fast_math,
        )
        return fused_moe.run(binding=binding)


@dataclass(frozen=True)
class ValidatedExl3ExpertSnapshot:
    """One process-local proof that an immutable expert snapshot was audited."""

    path: Path
    config: NativeExpertConfig
    bits: int
    projection_bits: dict[str, int] | None = None


def validate_exl3_expert_snapshot(
    snapshot: str | Path,
) -> ValidatedExl3ExpertSnapshot:
    """Read and fail closed once on a calibrated DS41RT EXL3 snapshot."""

    snapshot = Path(snapshot).resolve()
    raw = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if raw.get("model_type") != "deepseek_v4":
        raise ValueError(f"expected model_type=deepseek_v4, got {raw.get('model_type')!r}")
    quant = raw.get("quantization_config") or {}
    if is_gptqmodel_native_exl3(quant):
        validate_gptqmodel_native_exl3(quant, model_config=raw)
        validate_gptqmodel_publication(
            snapshot,
            raw,
            verify_all_hashes=False,
            require_canonical=True,
        )
        return ValidatedExl3ExpertSnapshot(
            path=snapshot,
            bits=int(quant["bits"]),
            projection_bits={
                str(module): int(entry["bits_per_weight"])
                for module, entry in quant["tensor_storage"].items()
            },
            config=_native_expert_config_from_raw(
                raw,
                expert_tensor_layout=GPTQMODEL_EXPERT_LAYOUT,
            ),
        )
    expected_top = {
        "quant_method": "exl3",
        "version": EXLLAMAV3_VERSION,
        "bits": float(EXL3_BITS),
        "codebook": EXL3_CODEBOOK,
    }
    for key, expected in expected_top.items():
        if quant.get(key) != expected:
            raise ValueError(
                f"DS41RT EXL3 requires quantization_config.{key}={expected!r}, "
                f"got {quant.get(key)!r}"
            )
    calibration = quant.get("calibration") or {}
    swiglu_limit = float(raw["swiglu_limit"])
    ds41rt = quant.get("ds41rt") or {}
    recipe = ds41rt.get("recipe")
    common_calibration_valid = (
        calibration.get("device") == "cuda:0"
        and int(calibration.get("rows", 0)) > 0
        and "seed" in calibration
        and calibration.get("activation") == "silu"
        and calibration.get("swiglu_limit") == swiglu_limit
        and calibration.get("gate_clamp") == [None, swiglu_limit]
        and calibration.get("up_clamp") == [-swiglu_limit, swiglu_limit]
    )
    if recipe == EXL3_RECIPE:
        calibration_valid = (
            common_calibration_valid
            and calibration.get("method") == "layerwise_simulated_isotropic"
            and calibration.get("hessian") == "analytic_identity"
            and calibration.get("distribution") == "rms_normalized_isotropic"
        )
    elif recipe == EXL3_FORCED_ACTIVATION_RECIPE:
        digest = calibration.get("activation_corpus_sha256")
        expert_weight = calibration.get("forced_down_expert_weight")
        pool_weight = calibration.get("forced_down_pool_weight")
        calibration_valid = (
            common_calibration_valid
            and calibration.get("method")
            == "layerwise_native_activation_forced_down"
            and calibration.get("hessian")
            == "native_sample_covariance_with_forced_down_shrinkage"
            and calibration.get("distribution")
            == "checkpoint_bound_native_expert_inputs"
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and int(calibration.get("activation_base_layers", 0))
            == int(raw["num_hidden_layers"])
            and int(calibration.get("forced_down_rows_per_expert", 0)) > 0
            and isinstance(expert_weight, (int, float))
            and isinstance(pool_weight, (int, float))
            and 0.0 <= float(expert_weight) <= 1.0
            and abs(float(expert_weight) + float(pool_weight) - 1.0) <= 1.0e-12
            and calibration.get("mtp_calibration")
            == "analytic_identity_pilot_only"
        )
    elif recipe == EXL3_ACTIVATION_RECIPE:
        digest = calibration.get("activation_corpus_sha256")
        replay_digest = calibration.get("route_replay_report_sha256")
        replay_path = snapshot / EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME
        replay_file_valid = (
            calibration.get("route_replay_report")
            == EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME
            and replay_path.is_file()
            and isinstance(replay_digest, str)
            and hashlib.sha256(replay_path.read_bytes()).hexdigest() == replay_digest
        )
        calibration_valid = (
            common_calibration_valid
            and calibration.get("method") == "layerwise_native_natural_routes"
            and calibration.get("hessian")
            == "per_expert_natural_route_gate_squared_covariance"
            and calibration.get("distribution")
            == "checkpoint_bound_native_expert_inputs"
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and isinstance(replay_digest, str)
            and len(replay_digest) == 64
            and all(
                character in "0123456789abcdef" for character in replay_digest
            )
            and replay_file_valid
            and int(calibration.get("activation_base_layers", 0))
            == int(raw["num_hidden_layers"])
            and int(calibration.get("minimum_natural_routes_per_expert", 0))
            >= MIN_NATURAL_ROUTES_PER_EXPERT
            and calibration.get("natural_routing") is True
            and calibration.get("forced_expert_activation") is False
            and calibration.get("route_gate_weighting") == "squared_unit_rms"
            and calibration.get("mtp_calibration")
            == "analytic_identity_pilot_only"
        )
    else:
        calibration_valid = False
    if not calibration_valid:
        raise ValueError(
            "DS41RT EXL3 requires a recognized checkpoint-bound bounded DeepSeek "
            "SwiGLU calibration contract on cuda:0"
        )
    expected_ds41rt = {
        "schema": EXL3_SCHEMA,
        "schema_version": EXL3_SCHEMA_VERSION,
        "calibrated": True,
        "scope": "routed_experts",
        "source_format": NATIVE_SOURCE_FORMAT,
        "tensor_format": EXL3_TENSOR_FORMAT,
        "expert_tp_world_size": EXPERT_TP_WORLD_SIZE,
    }
    for key, expected in expected_ds41rt.items():
        if ds41rt.get(key) != expected:
            raise ValueError(
                f"DS41RT EXL3 requires ds41rt.{key}={expected!r}, "
                f"got {ds41rt.get(key)!r}"
            )
    source = ds41rt.get("quantizer_source") or {}
    expected_source = {
        "repository": EXLLAMAV3_REPOSITORY,
        "revision": EXLLAMAV3_REVISION,
        "source_tree_sha256": EXLLAMAV3_SOURCE_TREE_SHA256,
    }
    if source != expected_source:
        raise ValueError("DS41RT EXL3 quantizer provenance does not match its recipe")

    return ValidatedExl3ExpertSnapshot(
        path=snapshot,
        bits=EXL3_BITS,
        config=_native_expert_config_from_raw(raw),
    )


def read_exl3_expert_config(snapshot: str | Path) -> NativeExpertConfig:
    """Read and fail closed on a calibrated DS41RT EXL3 contract."""

    return validate_exl3_expert_snapshot(snapshot).config


def checkpoint_tensor_names(
    config: NativeExpertConfig,
    layer_id: int,
    expert_id: int,
) -> dict[str, str]:
    if not 0 <= layer_id < config.total_blocks:
        raise ValueError(f"layer {layer_id} is outside 0..{config.total_blocks}")
    if not 0 <= expert_id < config.global_experts:
        raise ValueError(f"expert {expert_id} is outside 0..{config.global_experts}")
    if config.expert_tensor_layout == CHECKPOINT_NATIVE_EXPERT_LAYOUT:
        block = (
            f"layers.{layer_id}.ffn"
            if layer_id < config.num_hidden_layers
            else f"mtp.{layer_id - config.num_hidden_layers}.ffn"
        )
        projection_stems = (("gate", "w1"), ("down", "w2"), ("up", "w3"))
    elif config.expert_tensor_layout == GPTQMODEL_EXPERT_LAYOUT:
        block = (
            f"model.layers.{layer_id}.mlp"
            if layer_id < config.num_hidden_layers
            else f"mtp.{layer_id - config.num_hidden_layers}.mlp"
        )
        projection_stems = (
            ("gate", "gate_proj"),
            ("down", "down_proj"),
            ("up", "up_proj"),
        )
    else:
        raise ValueError(
            "unsupported EXL3 expert tensor layout "
            f"{config.expert_tensor_layout!r}"
        )
    result: dict[str, str] = {}
    for projection, stem in projection_stems:
        base = f"{block}.experts.{expert_id}.{stem}"
        for suffix in ("trellis", "suh", "svh", "mcg"):
            result[f"{projection}_{suffix}"] = f"{base}.{suffix}"
    return result


def load_exl3_expert_tp_layer(
    snapshot: str | Path | ValidatedExl3ExpertSnapshot,
    layer_id: int,
    *,
    tp_rank: int,
    expert_ids: Iterable[int] | None = None,
    device: str = "cuda:0",
    quant_mode: str = "w4a16",
) -> Exl3ExpertTpLayer:
    """Load one production TP=4 slice of every requested EXL3 expert."""

    return _load_exl3_expert_layer(
        snapshot,
        layer_id,
        tp_rank=tp_rank,
        tp_world_size=EXPERT_TP_WORLD_SIZE,
        expert_ids=expert_ids,
        device=device,
        quant_mode=quant_mode,
    )


def load_exl3_expert_reference_layer(
    snapshot: str | Path | ValidatedExl3ExpertSnapshot,
    layer_id: int,
    expert_ids: Iterable[int],
    *,
    device: str = "cuda:0",
    quant_mode: str = "w4a16",
) -> Exl3ExpertTpLayer:
    """Load an unsharded EXL3 layer for a single-GPU oracle."""

    return _load_exl3_expert_layer(
        snapshot,
        layer_id,
        tp_rank=0,
        tp_world_size=1,
        expert_ids=expert_ids,
        device=device,
        quant_mode=quant_mode,
    )


def _load_exl3_expert_layer(
    snapshot: str | Path | ValidatedExl3ExpertSnapshot,
    layer_id: int,
    *,
    tp_rank: int,
    tp_world_size: int,
    expert_ids: Iterable[int] | None,
    device: str,
    quant_mode: str,
) -> Exl3ExpertTpLayer:
    if device not in {"cuda", "cuda:0"}:
        raise ValueError(f"DS41RT permits only physical GPU 0, got {device!r}")
    if quant_mode != "w4a16":
        raise ValueError("EXL3 Spark expert TP requires quant_mode='w4a16'")
    if "_pinned_sparkinfer" not in sys.modules:
        raise RuntimeError(
            "import python/tools/_pinned_sparkinfer.py before loading runtime experts"
        )

    torch = _torch()
    from safetensors import safe_open
    from b12x.moe import fused_moe

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "EXL3 expert validation requires CUDA_VISIBLE_DEVICES=0 and exactly one visible GPU"
        )
    resolved_device = torch.device("cuda:0")
    validated = (
        snapshot
        if isinstance(snapshot, ValidatedExl3ExpertSnapshot)
        else validate_exl3_expert_snapshot(snapshot)
    )
    snapshot = validated.path
    config = validated.config
    bits = validated.bits
    tp_slice = tp_intermediate_slice(
        config.intermediate_size,
        tp_rank,
        tp_world_size,
    )
    if tp_slice.size % EXL3_ROTATION_BLOCK:
        raise ValueError(
            f"EXL3 TP intermediate slice {tp_slice.size} must be H128 aligned"
        )
    if expert_ids is None:
        expert_ids = replicated_expert_ids(config.global_experts)
    expert_ids = tuple(int(expert_id) for expert_id in expert_ids)
    if not expert_ids or len(set(expert_ids)) != len(expert_ids):
        raise ValueError("expert_ids must be non-empty and unique")
    if tuple(sorted(expert_ids)) != expert_ids:
        raise ValueError("expert_ids must be sorted for stable local indexing")
    for expert_id in expert_ids:
        checkpoint_tensor_names(config, layer_id, expert_id)

    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    local_e = len(expert_ids)
    h = config.hidden_size
    n = config.intermediate_size
    local_n = tp_slice.size
    from b12x.moe.fused_moe.trellis import ProjectionTrellisTierWeights

    tier_bits = (bits, bits + 1)
    projection_bits: dict[tuple[int, str], int] = {}
    for local_id, global_id in enumerate(expert_ids):
        names = checkpoint_tensor_names(config, layer_id, global_id)
        for projection in ("gate", "up", "down"):
            module = names[f"{projection}_trellis"].removesuffix(".trellis")
            projection_bits[(local_id, projection)] = (
                bits
                if validated.projection_bits is None
                else validated.projection_bits[module]
            )
    unexpected_bits = sorted(set(projection_bits.values()) - set(tier_bits))
    if unexpected_bits:
        raise ValueError(
            f"EXL3 projection tiers {unexpected_bits} are outside planned {tier_bits}"
        )

    native_tiers = []
    trellis_destinations: dict[tuple[int, str], tuple[Any, Any]] = {}
    for tier in tier_bits:
        gate_ids = tuple(
            local_id
            for local_id in range(local_e)
            if projection_bits[(local_id, "gate")] == tier
        )
        up_ids = tuple(
            local_id
            for local_id in range(local_e)
            if projection_bits[(local_id, "up")] == tier
        )
        down_ids = tuple(
            local_id
            for local_id in range(local_e)
            if projection_bits[(local_id, "down")] == tier
        )
        fc1_slots = len(gate_ids) + len(up_ids)
        if fc1_slots:
            w13 = torch.empty(
                (fc1_slots, h // 16, local_n // 16, 16 * tier),
                dtype=torch.int16,
                device=resolved_device,
            )
        else:
            w13 = torch.empty(
                (1, h // 16, local_n // 16, 16 * tier),
                dtype=torch.int16,
                device=resolved_device,
            )
        w2 = torch.empty(
            (max(len(down_ids), 1), local_n // 16, h // 16, 16 * tier),
            dtype=torch.int16,
            device=resolved_device,
        )
        for slot, local_id in enumerate(gate_ids):
            trellis_destinations[(local_id, "gate")] = (w13, slot)
        for offset, local_id in enumerate(up_ids, start=len(gate_ids)):
            trellis_destinations[(local_id, "up")] = (w13, offset)
        for slot, local_id in enumerate(down_ids):
            trellis_destinations[(local_id, "down")] = (w2, slot)
        native_tiers.append(
            ProjectionTrellisTierWeights(
                bits=tier,
                w13=w13,
                w2=w2,
                gate_experts=gate_ids,
                up_experts=up_ids,
                down_experts=down_ids,
            )
        )
    gate_suh = torch.empty((local_e, h), dtype=torch.float16, device=resolved_device)
    up_suh = torch.empty_like(gate_suh)
    intermediate_rotations = torch.empty(
        (local_e, 3 * local_n), dtype=torch.float16, device=resolved_device
    )
    down_svh = torch.empty_like(gate_suh)
    markers = torch.empty((local_e, 3), dtype=torch.int32, device=resolved_device)

    # name -> (destination, location, expected shape, source slice, dtype)
    destinations: dict[
        str,
        tuple[Any, Any, tuple[int, ...], tuple[slice, ...] | None, Any],
    ] = {}
    tile_start = tp_slice.start // 16
    tile_stop = tp_slice.stop // 16
    for local_id, global_id in enumerate(expert_ids):
        names = checkpoint_tensor_names(config, layer_id, global_id)
        # Projection-major Trellis storage is already in the kernel's logical
        # FC1 order: gate first, then up.  Unlike native FP4 W13, it is not
        # passed through the ModelOpt row-rotation repacker.
        for projection, plane, rotation_slot in EXL3_W13_PROJECTION_LAYOUT:
            input_size, output_size = h, n
            destination, location = trellis_destinations[(local_id, projection)]
            projection_streams = 16 * projection_bits[(local_id, projection)]
            destinations[names[f"{projection}_trellis"]] = (
                destination,
                location,
                (h // 16, n // 16, projection_streams),
                (slice(None), slice(tile_start, tile_stop), slice(None)),
                torch.int16,
            )
            destinations[names[f"{projection}_suh"]] = (
                up_suh if projection == "up" else gate_suh,
                local_id,
                (input_size,),
                None,
                torch.float16,
            )
            destinations[names[f"{projection}_svh"]] = (
                intermediate_rotations,
                (local_id, slice(rotation_slot * local_n, (rotation_slot + 1) * local_n)),
                (output_size,),
                (slice(tp_slice.start, tp_slice.stop),),
                torch.float16,
            )
            destinations[names[f"{projection}_mcg"]] = (
                markers,
                (local_id, rotation_slot),
                (),
                None,
                torch.int32,
            )
        down_destination, down_location = trellis_destinations[(local_id, "down")]
        down_streams = 16 * projection_bits[(local_id, "down")]
        destinations[names["down_trellis"]] = (
            down_destination,
            down_location,
            (n // 16, h // 16, down_streams),
            (slice(tile_start, tile_stop), slice(None), slice(None)),
            torch.int16,
        )
        destinations[names["down_suh"]] = (
            intermediate_rotations,
            (local_id, slice(2 * local_n, 3 * local_n)),
            (n,),
            (slice(tp_slice.start, tp_slice.stop),),
            torch.float16,
        )
        destinations[names["down_svh"]] = (
            down_svh,
            local_id,
            (h,),
            None,
            torch.float16,
        )
        destinations[names["down_mcg"]] = (
            markers,
            (local_id, 2),
            (),
            None,
            torch.int32,
        )

    by_file: dict[str, list[str]] = defaultdict(list)
    for name in destinations:
        try:
            by_file[str(index[name])].append(name)
        except KeyError as exc:
            raise ValueError(f"checkpoint index is missing EXL3 tensor {name}") from exc

    source_bytes = 0
    load_started = time.perf_counter()
    for file_name, names in sorted(by_file.items()):
        with safe_open(snapshot / file_name, framework="pt", device="cuda:0") as shard:
            for name in names:
                destination, location, expected_shape, source_slice, expected_dtype = (
                    destinations[name]
                )
                source_view = shard.get_slice(name)
                if tuple(source_view.get_shape()) != expected_shape:
                    raise ValueError(
                        f"tensor {name} has shape {tuple(source_view.get_shape())}, "
                        f"expected {expected_shape}"
                    )
                source = (
                    shard.get_tensor(name)
                    if not expected_shape
                    else source_view[source_slice or tuple(slice(None) for _ in expected_shape)]
                )
                if source.dtype != expected_dtype:
                    raise TypeError(
                        f"tensor {name} must be {expected_dtype}, got {source.dtype}"
                    )
                destination[location].copy_(source)
                source_bytes += source.numel() * source.element_size()
    torch.cuda.synchronize()
    marker_values = markers.cpu().reshape(-1).tolist()
    if any((int(value) & 0xFFFFFFFF) != MCG_MARKER for value in marker_values):
        raise ValueError("EXL3 routed expert tensors contain a non-MCG marker")
    load_seconds = time.perf_counter() - load_started

    # Direct GPTQModel artifacts contain one adjacent K pair.  The generic
    # routed planner uses its capture-safe K128/N128 geometry; DS41RT serving's
    # AOT decode path retains the independently tuned direct K64/N128 variant.
    tile_config = (128, 128, 128, 128)
    weight_plan = fused_moe.plan_weights(
        quant_modes=quant_mode,
        source_format=SPARKINFER_SOURCE_FORMAT,
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=local_e,
        hidden_size=h,
        intermediate_size=local_n,
        w13_layout="w13",
        trellis_bits=bits,
        trellis_tile_config=tile_config,
        trellis_codebook="mcg",
        trellis_rate_granularity="per_expert_projection",
    )
    experts = fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.bfloat16,
        projection_tiers=tuple(native_tiers),
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
    )
    route_expert_map = torch.full(
        (config.global_experts,), -1, dtype=torch.int32, device=resolved_device
    )
    route_expert_map[
        torch.tensor(expert_ids, dtype=torch.int64, device=resolved_device)
    ] = torch.arange(local_e, dtype=torch.int32, device=resolved_device)
    return Exl3ExpertTpLayer(
        config=config,
        layer_id=layer_id,
        tp_slice=tp_slice,
        global_expert_ids=expert_ids,
        experts=experts,
        route_expert_map=route_expert_map,
        weight_plan=weight_plan,
        source_bytes=source_bytes,
        load_seconds=load_seconds,
        projection_mixed=True,
    )


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise RuntimeError("EXL3 expert execution requires the torch runtime extra") from exc
    return torch
