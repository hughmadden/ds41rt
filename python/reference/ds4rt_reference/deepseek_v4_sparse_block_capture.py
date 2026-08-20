from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from ds4rt_reference.deepseek_v4_attention_layer_capture import (
    DS4_EXPERT_TP,
    DS4_LAYER_ARENA_ALIGNMENT,
    DeepseekV4AttentionLayerArenaBinding,
    DeepseekV4AttentionLayerContract,
    bind_deepseek_v4_attention_layer,
    plan_deepseek_v4_attention_layer,
    qualify_deepseek_v4_attention_layer_contract,
)
from ds4rt_reference.deepseek_v4_attention_capture import DS4_SWA_TOKENS
from ds4rt_reference.deepseek_v4_mhc_capture import (
    DeepseekV4MHCContract,
    plan_deepseek_v4_mhc,
    qualify_deepseek_v4_mhc_contract,
)

DS4_GLOBAL_TOP_K = 6
DS4_SHARED_EXPERT_WORKSPACE_ROWS = 8


@dataclass(frozen=True)
class DeepseekV4SparseBlockArenaLayout:
    attention_offset: int
    attention_bytes: int
    route_indices_offset: int
    route_indices_bytes: int
    route_scores_offset: int
    route_scores_bytes: int
    route_weights_offset: int
    route_weights_bytes: int
    shared_gate_offset: int
    shared_gate_bytes: int
    shared_up_offset: int
    shared_up_bytes: int
    shared_activated_offset: int
    shared_activated_bytes: int
    shared_delta_offset: int
    shared_delta_bytes: int
    reduction_f32_offset: int
    reduction_f32_bytes: int
    ffn_delta_offset: int
    ffn_delta_bytes: int
    total_bytes: int

    def regions(self) -> tuple[tuple[str, int, int], ...]:
        return (
            ("attention", self.attention_offset, self.attention_bytes),
            (
                "route_indices",
                self.route_indices_offset,
                self.route_indices_bytes,
            ),
            (
                "route_scores",
                self.route_scores_offset,
                self.route_scores_bytes,
            ),
            (
                "route_weights",
                self.route_weights_offset,
                self.route_weights_bytes,
            ),
            ("shared_gate", self.shared_gate_offset, self.shared_gate_bytes),
            ("shared_up", self.shared_up_offset, self.shared_up_bytes),
            (
                "shared_activated",
                self.shared_activated_offset,
                self.shared_activated_bytes,
            ),
            ("shared_delta", self.shared_delta_offset, self.shared_delta_bytes),
            (
                "reduction_f32",
                self.reduction_f32_offset,
                self.reduction_f32_bytes,
            ),
            ("ffn_delta", self.ffn_delta_offset, self.ffn_delta_bytes),
        )


@dataclass(frozen=True)
class DeepseekV4SparseBlockContract:
    attention: DeepseekV4AttentionLayerContract
    ffn_mhc: DeepseekV4MHCContract
    routed_experts: int
    arena: DeepseekV4SparseBlockArenaLayout
    graph_segments: tuple[str, ...] = (
        "attention-through-ffn-pre",
        "post-dispatch-reduction-through-next-attention-pre",
    )
    owner: str = "coordinator-local-sparse-block-with-tp4-dispatch-barrier"
    status: str = "qualified-split-graph-sparse-block-not-active"

    @property
    def variant(self) -> str:
        return self.attention.variant

    @property
    def max_rows(self) -> int:
        return self.attention.max_rows

    @property
    def hidden(self) -> int:
        return self.attention.hidden

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def cuda_graph_segments(self) -> int:
        return 2

    @property
    def dispatch_barrier_between_graphs(self) -> bool:
        return True

    @property
    def expert_tensor_parallel(self) -> int:
        return DS4_EXPERT_TP

    @property
    def expert_parallel(self) -> bool:
        return False

    @property
    def expert_global_top_k(self) -> int:
        return DS4_GLOBAL_TOP_K

    @property
    def one_route_buffer_fans_out_to_all_ranks(self) -> bool:
        return True

    @property
    def router_outputs_are_caller_owned(self) -> bool:
        return True

    @property
    def router_requires_host_readback(self) -> bool:
        return False

    @property
    def router_score_scratch_dtype(self) -> str:
        return "float32"

    @property
    def shared_expert_intermediate(self) -> int:
        return 2_048 if self.variant == "flash" else 3_072

    @property
    def shared_expert_workspace_rows(self) -> int:
        return DS4_SHARED_EXPERT_WORKSPACE_ROWS

    @property
    def shared_expert_outputs_are_caller_owned(self) -> bool:
        return True

    @property
    def shared_expert_requires_host_readback(self) -> bool:
        return False

    @property
    def shared_expert_output_shape(self) -> tuple[int, int]:
        return (self.max_rows, self.hidden)

    @property
    def shared_expert_output_dtype(self) -> str:
        return "bfloat16"

    @property
    def expert_local_intermediate_fraction(self) -> tuple[int, int]:
        return (1, DS4_EXPERT_TP)

    @property
    def expert_partial_output_shape(self) -> tuple[int, int, int]:
        return (DS4_EXPERT_TP, self.max_rows, self.hidden)

    @property
    def expert_partial_dtype(self) -> str:
        return "bfloat16"

    @property
    def expert_route_weights_applied_on_sparks(self) -> bool:
        return True

    @property
    def shared_expert_owner(self) -> str:
        return "coordinator"

    @property
    def reduction_accumulator_dtype(self) -> str:
        return "float32"

    @property
    def reduction_order(self) -> tuple[str, ...]:
        return ("shared", "spark-0", "spark-1", "spark-2", "spark-3")

    @property
    def mhc_scratch_reused_across_graph_segments(self) -> bool:
        return True


@dataclass(frozen=True)
class DeepseekV4SparseBlockArenaBinding:
    serving_allocates: ClassVar[bool] = False
    views_only: ClassVar[bool] = True

    contract: DeepseekV4SparseBlockContract
    tokens: int
    arena: Any
    attention_arena: Any
    route_indices: Any
    route_scores: Any
    route_weights: Any
    shared_gate: Any
    shared_up: Any
    shared_activated: Any
    shared_delta: Any
    reduction_f32: Any
    ffn_delta: Any

    def region_views(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("attention", self.attention_arena),
            ("route_indices", self.route_indices),
            ("route_scores", self.route_scores),
            ("route_weights", self.route_weights),
            ("shared_gate", self.shared_gate),
            ("shared_up", self.shared_up),
            ("shared_activated", self.shared_activated),
            ("shared_delta", self.shared_delta),
            ("reduction_f32", self.reduction_f32),
            ("ffn_delta", self.ffn_delta),
        )


@dataclass(frozen=True)
class DeepseekV4SparseBlockBinding:
    """Two graph segments joined by the fixed four-Spark TP dispatch barrier."""

    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    cuda_graph_segments: ClassVar[int] = 2
    dispatch_barrier_between_graphs: ClassVar[bool] = True
    one_route_buffer_fans_out_to_all_ranks: ClassVar[bool] = True
    router_outputs_are_caller_owned: ClassVar[bool] = True
    router_requires_host_readback: ClassVar[bool] = False
    shared_expert_outputs_are_caller_owned: ClassVar[bool] = True
    shared_expert_requires_host_readback: ClassVar[bool] = False
    rank_partials_are_hidden_width_bf16: ClassVar[bool] = True
    reduction_is_fp32: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    arena_binding: DeepseekV4SparseBlockArenaBinding
    attention_binding: Any
    attention_runner: Callable[[Any], tuple[Any, Any, Any, Any]]
    shared_delta: Any
    rank_partials: tuple[Any, Any, Any, Any]
    ffn_mhc_binding: Any
    attention_residual: Any
    attention_post: Any
    attention_comb: Any
    dispatch_hidden: Any
    ffn_fn: Any
    ffn_hc_scale: Any
    ffn_hc_base: Any
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    next_norm_weight: Any
    next_norm_eps: float


def plan_deepseek_v4_sparse_block(
    *,
    variant: str,
    mode: str,
    compression: int,
    max_rows: int,
    source_pages: int,
    max_positions: int,
    swa_width: int = DS4_SWA_TOKENS,
    cache_format: str = "fp8",
) -> DeepseekV4SparseBlockContract:
    attention = plan_deepseek_v4_attention_layer(
        variant=variant,
        mode=mode,
        compression=compression,
        max_rows=max_rows,
        source_pages=source_pages,
        max_positions=max_positions,
        swa_width=swa_width,
        cache_format=cache_format,
    )
    ffn_mhc = plan_deepseek_v4_mhc(variant=variant, max_rows=max_rows)
    routed_experts = 256 if attention.variant == "flash" else 384

    offset = 0
    attention_offset = offset
    attention_bytes = attention.arena.total_bytes
    offset = _align_up(attention_offset + attention_bytes)
    route_indices_offset = offset
    route_indices_bytes = max_rows * DS4_GLOBAL_TOP_K * 4
    offset = _align_up(route_indices_offset + route_indices_bytes)
    route_scores_offset = offset
    # Learned routing writes raw rows-by-experts scores after the public top-k
    # prefix, then selects deterministically in a stream-ordered second pass.
    # The same arena is reused by hash-routed and learned layers, so reserve the
    # larger learned-router workspace for every sparse block graph.
    route_scores_bytes = max_rows * (DS4_GLOBAL_TOP_K + routed_experts) * 4
    offset = _align_up(route_scores_offset + route_scores_bytes)
    route_weights_offset = offset
    route_weights_bytes = max_rows * DS4_GLOBAL_TOP_K * 4
    offset = _align_up(route_weights_offset + route_weights_bytes)
    shared_workspace_bytes = (
        DS4_SHARED_EXPERT_WORKSPACE_ROWS
        * (2_048 if attention.variant == "flash" else 3_072)
        * 2
    )
    shared_gate_offset = offset
    shared_gate_bytes = shared_workspace_bytes
    offset = _align_up(shared_gate_offset + shared_gate_bytes)
    shared_up_offset = offset
    shared_up_bytes = shared_workspace_bytes
    offset = _align_up(shared_up_offset + shared_up_bytes)
    shared_activated_offset = offset
    shared_activated_bytes = shared_workspace_bytes
    offset = _align_up(shared_activated_offset + shared_activated_bytes)
    shared_delta_offset = offset
    shared_delta_bytes = max_rows * attention.hidden * 2
    offset = _align_up(shared_delta_offset + shared_delta_bytes)
    reduction_f32_offset = offset
    reduction_f32_bytes = max_rows * attention.hidden * 4
    offset = _align_up(reduction_f32_offset + reduction_f32_bytes)
    ffn_delta_offset = offset
    ffn_delta_bytes = max_rows * attention.hidden * 2
    total_bytes = _align_up(ffn_delta_offset + ffn_delta_bytes)
    return DeepseekV4SparseBlockContract(
        attention=attention,
        ffn_mhc=ffn_mhc,
        routed_experts=routed_experts,
        arena=DeepseekV4SparseBlockArenaLayout(
            attention_offset=attention_offset,
            attention_bytes=attention_bytes,
            route_indices_offset=route_indices_offset,
            route_indices_bytes=route_indices_bytes,
            route_scores_offset=route_scores_offset,
            route_scores_bytes=route_scores_bytes,
            route_weights_offset=route_weights_offset,
            route_weights_bytes=route_weights_bytes,
            shared_gate_offset=shared_gate_offset,
            shared_gate_bytes=shared_gate_bytes,
            shared_up_offset=shared_up_offset,
            shared_up_bytes=shared_up_bytes,
            shared_activated_offset=shared_activated_offset,
            shared_activated_bytes=shared_activated_bytes,
            shared_delta_offset=shared_delta_offset,
            shared_delta_bytes=shared_delta_bytes,
            reduction_f32_offset=reduction_f32_offset,
            reduction_f32_bytes=reduction_f32_bytes,
            ffn_delta_offset=ffn_delta_offset,
            ffn_delta_bytes=ffn_delta_bytes,
            total_bytes=total_bytes,
        ),
    )


def bind_deepseek_v4_sparse_block_arena(
    contract: DeepseekV4SparseBlockContract,
    *,
    arena: Any,
    tokens: int,
) -> DeepseekV4SparseBlockArenaBinding:
    import torch

    if not isinstance(contract, DeepseekV4SparseBlockContract):
        raise TypeError("sparse block arena requires a planned contract")
    if not isinstance(arena, torch.Tensor) or arena.ndim != 1:
        raise TypeError("sparse block arena must be a rank-1 tensor")
    if arena.dtype != torch.uint8 or not arena.is_contiguous():
        raise ValueError("sparse block arena must be contiguous torch.uint8")
    tokens = int(tokens)
    if not 1 <= tokens <= contract.max_rows:
        raise ValueError(
            f"sparse block tokens must be in [1, {contract.max_rows}], got {tokens}"
        )
    if arena.numel() < contract.arena.total_bytes:
        raise ValueError(
            "sparse block arena is too small: "
            f"need={contract.arena.total_bytes}, have={arena.numel()}"
        )
    arena = arena.narrow(0, 0, contract.arena.total_bytes)

    def view(offset: int, nbytes: int, *, dtype: Any, shape: tuple[int, ...]):
        return arena.narrow(0, offset, nbytes).view(dtype).view(shape)

    layout = contract.arena
    return DeepseekV4SparseBlockArenaBinding(
        contract=contract,
        tokens=tokens,
        arena=arena,
        attention_arena=view(
            layout.attention_offset,
            layout.attention_bytes,
            dtype=torch.uint8,
            shape=(layout.attention_bytes,),
        ),
        route_indices=view(
            layout.route_indices_offset,
            tokens * DS4_GLOBAL_TOP_K * 4,
            dtype=torch.int32,
            shape=(tokens, DS4_GLOBAL_TOP_K),
        ),
        route_scores=view(
            layout.route_scores_offset,
            tokens * DS4_GLOBAL_TOP_K * 4,
            dtype=torch.float32,
            shape=(tokens, DS4_GLOBAL_TOP_K),
        ),
        route_weights=view(
            layout.route_weights_offset,
            tokens * DS4_GLOBAL_TOP_K * 4,
            dtype=torch.float32,
            shape=(tokens, DS4_GLOBAL_TOP_K),
        ),
        shared_gate=view(
            layout.shared_gate_offset,
            layout.shared_gate_bytes,
            dtype=torch.bfloat16,
            shape=(DS4_SHARED_EXPERT_WORKSPACE_ROWS, contract.shared_expert_intermediate),
        ),
        shared_up=view(
            layout.shared_up_offset,
            layout.shared_up_bytes,
            dtype=torch.bfloat16,
            shape=(DS4_SHARED_EXPERT_WORKSPACE_ROWS, contract.shared_expert_intermediate),
        ),
        shared_activated=view(
            layout.shared_activated_offset,
            layout.shared_activated_bytes,
            dtype=torch.bfloat16,
            shape=(DS4_SHARED_EXPERT_WORKSPACE_ROWS, contract.shared_expert_intermediate),
        ),
        shared_delta=view(
            layout.shared_delta_offset,
            tokens * contract.hidden * 2,
            dtype=torch.bfloat16,
            shape=(tokens, contract.hidden),
        ),
        reduction_f32=view(
            layout.reduction_f32_offset,
            tokens * contract.hidden * 4,
            dtype=torch.float32,
            shape=(tokens, contract.hidden),
        ),
        ffn_delta=view(
            layout.ffn_delta_offset,
            tokens * contract.hidden * 2,
            dtype=torch.bfloat16,
            shape=(tokens, contract.hidden),
        ),
    )


def bind_deepseek_v4_sparse_block(
    contract: DeepseekV4SparseBlockContract,
    *,
    arena_binding: DeepseekV4SparseBlockArenaBinding,
    attention_binding: Any,
    attention_runner: Callable[[Any], tuple[Any, Any, Any, Any]],
    rank_partials: tuple[Any, Any, Any, Any],
    ffn_fn: Any,
    ffn_hc_scale: Any,
    ffn_hc_base: Any,
    next_norm_weight: Any,
    next_residual_out: Any,
    next_y_out: Any,
    next_post_out: Any,
    next_comb_out: Any,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
    next_norm_eps: float = 1.0e-6,
) -> DeepseekV4SparseBlockBinding:
    import torch
    from b12x.norm import mhc

    if arena_binding.contract is not contract:
        raise ValueError("sparse block arena binding belongs to another contract")
    if not callable(attention_runner):
        raise TypeError("sparse block attention_runner must be callable")
    attention_arena_binding = getattr(attention_binding, "arena_binding", None)
    if not isinstance(attention_arena_binding, DeepseekV4AttentionLayerArenaBinding):
        raise TypeError("sparse block requires a bound DS4 attention composite")
    if attention_arena_binding.contract != contract.attention:
        raise ValueError("sparse block attention binding contract mismatch")
    if (
        attention_arena_binding.arena.data_ptr()
        != arena_binding.attention_arena.data_ptr()
    ):
        raise ValueError("sparse block attention binding must use its arena prefix")
    mhc_binding = getattr(attention_binding, "mhc_binding", None)
    if mhc_binding is None:
        raise TypeError("sparse block attention binding is missing mHC outputs")
    attention_residual = mhc_binding.out
    attention_post = mhc_binding.post_buffer
    attention_comb = mhc_binding.comb_buffer
    dispatch_hidden = mhc_binding.y
    expected_shape = (arena_binding.tokens, contract.hidden)
    _require_bf16_matrix(
        "shared_delta",
        arena_binding.shared_delta,
        expected_shape,
        dispatch_hidden.device,
    )
    if not isinstance(rank_partials, tuple) or len(rank_partials) != DS4_EXPERT_TP:
        raise ValueError("sparse block requires exactly four rank partial tensors")
    for rank, partial in enumerate(rank_partials):
        _require_bf16_matrix(
            f"rank_partials[{rank}]",
            partial,
            expected_shape,
            dispatch_hidden.device,
        )

    mhc_plan = mhc.plan(
        mhc.Caps(
            device=dispatch_hidden.device,
            max_tokens=contract.max_rows,
            hidden_size=contract.hidden,
            split_k=contract.ffn_mhc.geometry.split_k,
            dtype=torch.bfloat16,
        )
    )
    ffn_mhc_binding = mhc.bind(
        mhc_plan,
        scratch=attention_arena_binding.mhc_scratch,
        tokens=arena_binding.tokens,
        expected_m=contract.max_rows,
        y=next_y_out,
        post=next_post_out,
        comb=next_comb_out,
        out=next_residual_out,
    )
    return DeepseekV4SparseBlockBinding(
        arena_binding=arena_binding,
        attention_binding=attention_binding,
        attention_runner=attention_runner,
        shared_delta=arena_binding.shared_delta,
        rank_partials=rank_partials,
        ffn_mhc_binding=ffn_mhc_binding,
        attention_residual=attention_residual,
        attention_post=attention_post,
        attention_comb=attention_comb,
        dispatch_hidden=dispatch_hidden,
        ffn_fn=ffn_fn,
        ffn_hc_scale=ffn_hc_scale,
        ffn_hc_base=ffn_hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        next_norm_weight=next_norm_weight,
        next_norm_eps=float(next_norm_eps),
    )


def run_deepseek_v4_sparse_block_attention(
    binding: DeepseekV4SparseBlockBinding,
) -> tuple[Any, Any, Any, Any]:
    """Run graph segment one through the normalized expert dispatch input."""

    if not isinstance(binding, DeepseekV4SparseBlockBinding):
        raise TypeError("sparse block attention requires its fixed binding")
    outputs = binding.attention_runner(binding.attention_binding)
    expected = (
        binding.attention_residual,
        binding.attention_post,
        binding.attention_comb,
        binding.dispatch_hidden,
    )
    if len(outputs) != 4 or any(
        actual is not wanted for actual, wanted in zip(outputs, expected)
    ):
        raise RuntimeError("attention runner lost the bound sparse-block outputs")
    return outputs


def run_deepseek_v4_sparse_block_post_dispatch(
    binding: DeepseekV4SparseBlockBinding,
) -> tuple[Any, Any, Any, Any]:
    """Reduce four fixed TP partials plus shared delta and cross the FFN boundary."""

    import torch
    from b12x.norm import mhc

    if not isinstance(binding, DeepseekV4SparseBlockBinding):
        raise TypeError("sparse block post-dispatch requires its fixed binding")
    reduction = binding.arena_binding.reduction_f32
    reduction.copy_(binding.shared_delta)
    for partial in binding.rank_partials:
        torch.add(reduction, partial, out=reduction)
    binding.arena_binding.ffn_delta.copy_(reduction)
    return mhc.run_post_pre(
        binding.arena_binding.ffn_delta,
        binding.attention_residual,
        binding.attention_post,
        binding.attention_comb,
        binding.ffn_fn,
        binding.ffn_hc_scale,
        binding.ffn_hc_base,
        rms_eps=binding.rms_eps,
        hc_eps=binding.hc_eps,
        sinkhorn_iters=binding.sinkhorn_iters,
        norm_weight=binding.next_norm_weight,
        norm_eps=binding.next_norm_eps,
        binding=binding.ffn_mhc_binding,
    )


def qualify_deepseek_v4_sparse_block_contract(**kwargs: Any) -> bool:
    contract = plan_deepseek_v4_sparse_block(
        variant=str(kwargs["variant"]),
        mode=str(kwargs["mode"]),
        compression=int(kwargs["compression"]),
        max_rows=int(kwargs["max_rows"]),
        source_pages=int(kwargs["source_pages"]),
        max_positions=int(kwargs["max_positions"]),
        swa_width=int(kwargs.get("swa_width", DS4_SWA_TOKENS)),
        cache_format=str(kwargs.get("cache_format", "fp8")),
    )
    if not qualify_deepseek_v4_attention_layer_contract(
        variant=contract.variant,
        mode=contract.attention.mode,
        compression=contract.attention.compression,
        max_rows=contract.max_rows,
        source_pages=contract.attention.source_pages,
        max_positions=contract.attention.max_positions,
        swa_width=contract.attention.swa_width,
        cache_format=contract.attention.cache_format,
    ):
        return False
    if not qualify_deepseek_v4_mhc_contract(
        variant=contract.variant,
        max_rows=contract.max_rows,
    ):
        return False
    expected_experts = 256 if contract.variant == "flash" else 384
    if (
        contract.serving_allocates
        or contract.routed_experts != expected_experts
        or contract.cuda_graph_segments != 2
        or not contract.dispatch_barrier_between_graphs
        or contract.expert_tensor_parallel != DS4_EXPERT_TP
        or contract.expert_parallel
        or contract.expert_global_top_k != DS4_GLOBAL_TOP_K
        or not contract.one_route_buffer_fans_out_to_all_ranks
        or not contract.router_outputs_are_caller_owned
        or contract.router_requires_host_readback
        or contract.router_score_scratch_dtype != "float32"
        or contract.shared_expert_intermediate
        != (2_048 if contract.variant == "flash" else 3_072)
        or contract.shared_expert_workspace_rows != DS4_SHARED_EXPERT_WORKSPACE_ROWS
        or not contract.shared_expert_outputs_are_caller_owned
        or contract.shared_expert_requires_host_readback
        or contract.shared_expert_output_shape != (contract.max_rows, contract.hidden)
        or contract.shared_expert_output_dtype != "bfloat16"
        or contract.expert_local_intermediate_fraction != (1, DS4_EXPERT_TP)
        or contract.expert_partial_output_shape
        != (DS4_EXPERT_TP, contract.max_rows, contract.hidden)
        or contract.expert_partial_dtype != "bfloat16"
        or not contract.expert_route_weights_applied_on_sparks
        or contract.shared_expert_owner != "coordinator"
        or contract.reduction_accumulator_dtype != "float32"
        or contract.reduction_order
        != ("shared", "spark-0", "spark-1", "spark-2", "spark-3")
        or not contract.mhc_scratch_reused_across_graph_segments
    ):
        raise RuntimeError("sparse block changed the fixed GLMRT TP=4 handoff")
    previous_end = 0
    for name, offset, nbytes in contract.arena.regions():
        if offset % DS4_LAYER_ARENA_ALIGNMENT:
            raise RuntimeError(f"sparse block arena region {name} lost alignment")
        if offset < previous_end:
            raise RuntimeError(f"sparse block arena region {name} overlaps")
        previous_end = offset + nbytes
    if _align_up(previous_end) != contract.arena.total_bytes:
        raise RuntimeError("sparse block arena total drifted")

    import torch

    arena = torch.empty((contract.arena.total_bytes,), dtype=torch.uint8)
    binding = bind_deepseek_v4_sparse_block_arena(contract, arena=arena, tokens=1)
    offsets = {name: offset for name, offset, _ in contract.arena.regions()}
    for name, tensor in binding.region_views():
        if tensor.data_ptr() != arena.data_ptr() + offsets[name]:
            raise RuntimeError(f"sparse block arena view {name} lost its offset")
    attention_arena = bind_deepseek_v4_attention_layer(
        contract.attention,
        arena=binding.attention_arena,
        tokens=1,
    )
    if (
        attention_arena.arena.data_ptr() != arena.data_ptr()
        or DeepseekV4SparseBlockBinding.serving_allocates
        or not DeepseekV4SparseBlockBinding.cuda_graph_safe
        or DeepseekV4SparseBlockBinding.cuda_graph_segments != 2
        or not DeepseekV4SparseBlockBinding.dispatch_barrier_between_graphs
        or not DeepseekV4SparseBlockBinding.one_route_buffer_fans_out_to_all_ranks
        or not DeepseekV4SparseBlockBinding.router_outputs_are_caller_owned
        or DeepseekV4SparseBlockBinding.router_requires_host_readback
        or not DeepseekV4SparseBlockBinding.shared_expert_outputs_are_caller_owned
        or DeepseekV4SparseBlockBinding.shared_expert_requires_host_readback
        or not DeepseekV4SparseBlockBinding.rank_partials_are_hidden_width_bf16
        or not DeepseekV4SparseBlockBinding.reduction_is_fp32
        or DeepseekV4SparseBlockBinding.expert_tensor_parallel != DS4_EXPERT_TP
        or DeepseekV4SparseBlockBinding.expert_parallel
        or not callable(bind_deepseek_v4_sparse_block)
        or not callable(run_deepseek_v4_sparse_block_attention)
        or not callable(run_deepseek_v4_sparse_block_post_dispatch)
    ):
        raise RuntimeError("sparse block binding lifecycle drifted")
    return True


def _require_bf16_matrix(
    name: str,
    tensor: Any,
    shape: tuple[int, int],
    device: Any,
) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.shape != shape or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must have shape {shape} and dtype torch.bfloat16")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _align_up(value: int) -> int:
    return (
        (int(value) + DS4_LAYER_ARENA_ALIGNMENT - 1)
        // DS4_LAYER_ARENA_ALIGNMENT
        * DS4_LAYER_ARENA_ALIGNMENT
    )


__all__ = [
    "DeepseekV4SparseBlockArenaBinding",
    "DeepseekV4SparseBlockArenaLayout",
    "DeepseekV4SparseBlockBinding",
    "DeepseekV4SparseBlockContract",
    "bind_deepseek_v4_sparse_block",
    "bind_deepseek_v4_sparse_block_arena",
    "plan_deepseek_v4_sparse_block",
    "qualify_deepseek_v4_sparse_block_contract",
    "run_deepseek_v4_sparse_block_attention",
    "run_deepseek_v4_sparse_block_post_dispatch",
]
