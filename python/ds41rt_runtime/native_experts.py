"""Checkpoint-native DeepSeek V4 FP4/E8M0 TP=4 expert integration.

Every Spark stores the same expert IDs and one quarter of every expert's
intermediate dimension. All top-k routes execute on all four ranks and the
hidden-size partial outputs are summed by DS41RT's transport reduction. This is
the GLMRT serving architecture: expert parallelism is not used here.

SparkInfer consumes W13 in ``[up; gate]`` order, which is checkpoint ``w3``
followed by ``w1``. FC1 shards by output rows and FC2 shards by packed input
columns, so the nonlinearity remains local and the four FC2 results sum to the
unsharded expert result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable


NATIVE_RECIPE = "deepseek_v4_native_fp4_fp8_mixed_v1"
NATIVE_SOURCE_FORMAT = "fp4_e8m0_k32"
NATIVE_K_BLOCK = 32
EXPERT_TP_WORLD_SIZE = 4
CHECKPOINT_NATIVE_EXPERT_LAYOUT = "checkpoint-native"
GPTQMODEL_EXPERT_LAYOUT = "gptqmodel"


@dataclass(frozen=True)
class NativeExpertConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    dspark_blocks: int
    global_experts: int
    top_k: int
    swiglu_limit: float
    # GPTQModel publishes quantized modules under the Transformers execution
    # namespace, while the original DeepSeek checkpoint uses its flatter
    # checkpoint namespace.  Keep that distinction bound to the validated
    # artifact configuration so tensor lookup never guesses from a suffix.
    expert_tensor_layout: str = CHECKPOINT_NATIVE_EXPERT_LAYOUT

    @property
    def total_blocks(self) -> int:
        return self.num_hidden_layers + self.dspark_blocks


@dataclass(frozen=True)
class TpIntermediateSlice:
    rank: int
    world_size: int
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def packed_byte_start(self) -> int:
        return self.start // 2

    @property
    def packed_byte_stop(self) -> int:
        return self.stop // 2

    @property
    def scale_start(self) -> int:
        return self.start // NATIVE_K_BLOCK

    @property
    def scale_stop(self) -> int:
        return self.stop // NATIVE_K_BLOCK


@dataclass
class NativeExpertTpLayer:
    config: NativeExpertConfig
    layer_id: int
    tp_slice: TpIntermediateSlice
    global_expert_ids: tuple[int, ...]
    experts: Any
    route_expert_map: Any
    weight_plan: Any
    source_bytes: int
    load_seconds: float

    @property
    def local_experts(self) -> int:
        return len(self.global_expert_ids)

    @property
    def local_intermediate_size(self) -> int:
        return self.tp_slice.size

    def plan_tp(self, *, max_tokens: int):
        """Allocate this rank's fixed TP scratch outside graph capture."""

        torch = _torch()
        from b12x.moe import fused_moe

        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=max_tokens,
                num_topk=self.config.top_k,
                route_num_experts=self.config.global_experts,
                device=self.route_expert_map.device,
                weight_plan=self.weight_plan,
                quant_mode="w4a16",
                swiglu_limit=self.config.swiglu_limit,
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
        """Run all global routes through this rank's intermediate slice."""

        torch = _torch()
        from b12x.moe import fused_moe

        if plan is None or scratch is None:
            plan, scratch = self.plan_tp(max_tokens=int(hidden_states.shape[0]))
        if output is None:
            output = torch.empty_like(hidden_states)
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=hidden_states,
            experts=self.experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            route_expert_map=self.route_expert_map,
            output=output,
            input_scales_static=True,
            fast_math=fast_math,
        )
        return fused_moe.run(binding=binding)


def read_native_expert_config(snapshot: str | Path) -> NativeExpertConfig:
    snapshot = Path(snapshot)
    raw = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if raw.get("model_type") != "deepseek_v4":
        raise ValueError(f"expected model_type=deepseek_v4, got {raw.get('model_type')!r}")
    if raw.get("expert_dtype") != "fp4":
        raise ValueError(f"expected expert_dtype=fp4, got {raw.get('expert_dtype')!r}")
    quant = raw.get("quantization_config") or {}
    if str(quant.get("quant_method", "")).lower() != "fp8":
        raise ValueError("native expert loader requires the FP4/FP8 Flash checkpoint")
    hidden = int(raw["hidden_size"])
    intermediate = int(raw["moe_intermediate_size"])
    if hidden % NATIVE_K_BLOCK or intermediate % NATIVE_K_BLOCK:
        raise ValueError(
            f"native dimensions hidden={hidden} intermediate={intermediate} "
            f"must be divisible by {NATIVE_K_BLOCK}"
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
    )
    if config.top_k != 6:
        raise ValueError(f"DeepSeek V4 runtime requires top-k 6, got {config.top_k}")
    return config


def replicated_expert_ids(global_experts: int) -> tuple[int, ...]:
    if global_experts <= 0:
        raise ValueError("global_experts must be positive")
    return tuple(range(global_experts))


def tp_intermediate_slice(
    intermediate_size: int,
    rank: int,
    world_size: int = EXPERT_TP_WORLD_SIZE,
) -> TpIntermediateSlice:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid expert TP rank {rank}/{world_size}")
    alignment = world_size * NATIVE_K_BLOCK
    if intermediate_size <= 0 or intermediate_size % alignment:
        raise ValueError(
            f"intermediate_size={intermediate_size} must be divisible by "
            f"TP world size times K/32 scale block ({alignment})"
        )
    shard = intermediate_size // world_size
    return TpIntermediateSlice(
        rank=rank,
        world_size=world_size,
        start=rank * shard,
        stop=(rank + 1) * shard,
    )


def checkpoint_tensor_names(
    config: NativeExpertConfig,
    layer_id: int,
    expert_id: int,
) -> dict[str, str]:
    if not 0 <= layer_id < config.total_blocks:
        raise ValueError(f"layer {layer_id} is outside 0..{config.total_blocks}")
    if not 0 <= expert_id < config.global_experts:
        raise ValueError(f"expert {expert_id} is outside 0..{config.global_experts}")
    if layer_id < config.num_hidden_layers:
        block = f"layers.{layer_id}"
    else:
        block = f"mtp.{layer_id - config.num_hidden_layers}"
    prefix = f"{block}.ffn.experts.{expert_id}"
    return {
        "gate_weight": f"{prefix}.w1.weight",
        "gate_scale": f"{prefix}.w1.scale",
        "down_weight": f"{prefix}.w2.weight",
        "down_scale": f"{prefix}.w2.scale",
        "up_weight": f"{prefix}.w3.weight",
        "up_scale": f"{prefix}.w3.scale",
    }


def load_native_expert_tp_layer(
    snapshot: str | Path,
    layer_id: int,
    *,
    tp_rank: int,
    expert_ids: Iterable[int] | None = None,
    device: str = "cuda:0",
    quant_mode: str = "w4a16",
) -> NativeExpertTpLayer:
    """Load one production TP=4 slice of every requested native expert."""

    return _load_native_expert_layer(
        snapshot,
        layer_id,
        tp_rank=tp_rank,
        tp_world_size=EXPERT_TP_WORLD_SIZE,
        expert_ids=expert_ids,
        device=device,
        quant_mode=quant_mode,
    )


def load_native_expert_reference_layer(
    snapshot: str | Path,
    layer_id: int,
    expert_ids: Iterable[int],
    *,
    device: str = "cuda:0",
    quant_mode: str = "w4a16",
) -> NativeExpertTpLayer:
    """Load an unsharded layer for the single-GPU correctness oracle only."""

    return _load_native_expert_layer(
        snapshot,
        layer_id,
        tp_rank=0,
        tp_world_size=1,
        expert_ids=expert_ids,
        device=device,
        quant_mode=quant_mode,
    )


def _load_native_expert_layer(
    snapshot: str | Path,
    layer_id: int,
    *,
    tp_rank: int,
    tp_world_size: int,
    expert_ids: Iterable[int] | None,
    device: str,
    quant_mode: str,
) -> NativeExpertTpLayer:
    """Load one intermediate-dimension slice on physical GPU 0."""

    if device not in {"cuda", "cuda:0"}:
        raise ValueError(f"DS41RT permits only physical GPU 0, got {device!r}")
    if quant_mode != "w4a16":
        raise ValueError("native Spark expert TP currently requires quant_mode='w4a16'")
    if "_pinned_sparkinfer" not in sys.modules:
        raise RuntimeError(
            "import python/tools/_pinned_sparkinfer.py before loading runtime experts"
        )

    torch = _torch()
    from safetensors import safe_open
    from b12x.moe import fused_moe

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "native expert validation requires CUDA_VISIBLE_DEVICES=0 and exactly one visible GPU"
        )
    resolved_device = torch.device("cuda:0")
    snapshot = Path(snapshot).resolve()
    config = read_native_expert_config(snapshot)
    tp_slice = tp_intermediate_slice(
        config.intermediate_size,
        tp_rank,
        tp_world_size,
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
    w13 = torch.empty(
        (local_e, 2 * local_n, h // 2),
        dtype=torch.uint8,
        device=resolved_device,
    )
    w13_scale = torch.empty(
        (local_e, 2 * local_n, h // NATIVE_K_BLOCK),
        dtype=torch.uint8,
        device=resolved_device,
    )
    w2 = torch.empty(
        (local_e, h, local_n // 2),
        dtype=torch.uint8,
        device=resolved_device,
    )
    w2_scale = torch.empty(
        (local_e, h, local_n // NATIVE_K_BLOCK),
        dtype=torch.uint8,
        device=resolved_device,
    )

    # SparkInfer logical W13 is [up; gate], hence w3 occupies the first half.
    # Each entry carries destination, destination index, full checkpoint shape,
    # and a source slice. Safetensors reads only the rank-local rows/columns.
    destinations: dict[
        str,
        tuple[Any, Any, tuple[int, ...], tuple[slice, ...]],
    ] = {}
    for local_id, global_id in enumerate(expert_ids):
        names = checkpoint_tensor_names(config, layer_id, global_id)
        destinations[names["up_weight"]] = (
            w13,
            (local_id, slice(0, local_n)),
            (n, h // 2),
            (slice(tp_slice.start, tp_slice.stop), slice(None)),
        )
        destinations[names["gate_weight"]] = (
            w13,
            (local_id, slice(local_n, 2 * local_n)),
            (n, h // 2),
            (slice(tp_slice.start, tp_slice.stop), slice(None)),
        )
        destinations[names["up_scale"]] = (
            w13_scale,
            (local_id, slice(0, local_n)),
            (n, h // NATIVE_K_BLOCK),
            (slice(tp_slice.start, tp_slice.stop), slice(None)),
        )
        destinations[names["gate_scale"]] = (
            w13_scale,
            (local_id, slice(local_n, 2 * local_n)),
            (n, h // NATIVE_K_BLOCK),
            (slice(tp_slice.start, tp_slice.stop), slice(None)),
        )
        destinations[names["down_weight"]] = (
            w2,
            local_id,
            (h, n // 2),
            (
                slice(None),
                slice(tp_slice.packed_byte_start, tp_slice.packed_byte_stop),
            ),
        )
        destinations[names["down_scale"]] = (
            w2_scale,
            local_id,
            (h, n // NATIVE_K_BLOCK),
            (
                slice(None),
                slice(tp_slice.scale_start, tp_slice.scale_stop),
            ),
        )

    by_file: dict[str, list[str]] = defaultdict(list)
    for name in destinations:
        try:
            by_file[index[name]].append(name)
        except KeyError as exc:
            raise ValueError(f"checkpoint index is missing native expert tensor {name}") from exc

    load_started = time.perf_counter()
    source_bytes = 0
    for file_name, names in sorted(by_file.items()):
        with safe_open(snapshot / file_name, framework="pt", device="cuda:0") as shard:
            for name in names:
                source_view = shard.get_slice(name)
                destination, location, expected_shape, source_slice = destinations[name]
                if tuple(source_view.get_shape()) != expected_shape:
                    raise ValueError(
                        f"tensor {name} has shape {tuple(source_view.get_shape())}, "
                        f"expected {expected_shape}"
                    )
                source = source_view[source_slice]
                if name.endswith(".weight") and source.dtype != torch.int8:
                    raise TypeError(f"tensor {name} must be packed int8, got {source.dtype}")
                e8m0 = getattr(torch, "float8_e8m0fnu", None)
                if name.endswith(".scale") and source.dtype not in {torch.uint8, e8m0}:
                    raise TypeError(f"tensor {name} must be E8M0 bytes, got {source.dtype}")
                source_u8 = source.view(torch.uint8)
                destination[location].copy_(source_u8)
                source_bytes += source_u8.numel()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    ones = torch.ones(local_e, dtype=torch.float32, device=resolved_device)
    weight_plan = fused_moe.plan_weights(
        quant_modes=quant_mode,
        source_format=NATIVE_SOURCE_FORMAT,
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=local_e,
        hidden_size=h,
        intermediate_size=local_n,
        w13_layout="w13",
    )
    experts = fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.bfloat16,
        w1_fp4=w13,
        w1_blockscale=w13_scale,
        w1_global_scale=ones,
        a1_gscale=ones,
        w2_fp4=w2,
        w2_blockscale=w2_scale,
        w2_global_scale=ones,
        a2_gscale=ones,
    )

    route_expert_map = torch.full(
        (config.global_experts,), -1, dtype=torch.int32, device=resolved_device
    )
    route_expert_map[
        torch.tensor(expert_ids, dtype=torch.int64, device=resolved_device)
    ] = torch.arange(local_e, dtype=torch.int32, device=resolved_device)
    return NativeExpertTpLayer(
        config=config,
        layer_id=layer_id,
        tp_slice=tp_slice,
        global_expert_ids=expert_ids,
        experts=experts,
        route_expert_map=route_expert_map,
        weight_plan=weight_plan,
        source_bytes=source_bytes,
        load_seconds=load_seconds,
    )


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise RuntimeError("native expert execution requires the torch runtime extra") from exc
    return torch
