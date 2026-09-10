from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ds41rt_reference.deepseek_v4_attention_capture import (
    DS4_HEAD_DIM,
    DS4_SM120_PREFILL_SELECTION_TILE,
    DS4_SWA_TOKENS,
    plan_deepseek_v4_compressed_mla,
    qualify_deepseek_v4_compressed_mla_contract,
)
from ds41rt_reference.deepseek_v4_attention_compressor_capture import (
    plan_deepseek_v4_attention_compressor,
    qualify_deepseek_v4_attention_compressor_contract,
)
from ds41rt_reference.deepseek_v4_attention_output_capture import (
    plan_deepseek_v4_attention_output,
    qualify_deepseek_v4_attention_output_contract,
)
from ds41rt_reference.deepseek_v4_attention_producer_capture import (
    DS4_INDEX_HEAD_DIM,
    DS4_INDEX_HEADS,
    DS4_NOPE_DIM,
    DS4_ROPE_DIM,
    DS4_SOURCE_PAGE_TOKENS,
    deepseek_v4_producer_geometry,
    plan_deepseek_v4_attention_indexer,
    plan_deepseek_v4_attention_producer,
    qualify_deepseek_v4_attention_indexer_contract,
    qualify_deepseek_v4_attention_producer_contract,
)
from ds41rt_reference.deepseek_v4_mhc_capture import (
    plan_deepseek_v4_mhc,
    qualify_deepseek_v4_mhc_contract,
)

DS4_LAYER_ARENA_ALIGNMENT = 1_024
DS4_EXPERT_TP = 4
_SLIDING_CAPTURE_STATE: dict[tuple[int, ...], tuple[Any, Any]] = {}
_SLIDING_CAPTURE_PREPARED: set[tuple[int, ...]] = set()
_C128_CAPTURE_STATE: dict[tuple[int, ...], tuple[Any, Any, Any]] = {}
_C128_CAPTURE_PREPARED: set[tuple[int, ...]] = set()
_C4_CAPTURE_STATE: dict[tuple[int, ...], tuple[Any, Any, Any, Any]] = {}
_C4_CAPTURE_PREPARED: set[tuple[int, ...]] = set()
# Capture repeatedly views the same Rust-owned lane buffers at the same shapes
# across layers. Reusing those Python tensor wrappers avoids reconstructing
# external-storage metadata; every CUDA graph is still captured independently.
_TARGET_CAPTURE_TENSORS: dict[tuple[Any, ...], Any] = {}
_TARGET_CAPTURE_PLANS: dict[tuple[Any, Any], Any] = {}


def _cached_target_capture_plan(plan_function: Any, caps: Any) -> Any:
    """Reuse immutable B12X plans while keeping every graph binding distinct."""

    key = (plan_function, caps)
    try:
        plan = _TARGET_CAPTURE_PLANS.get(key)
    except TypeError:
        # Unit-test doubles and third-party extensions may use unhashable caps.
        return plan_function(caps)
    if plan is None:
        plan = plan_function(caps)
        _TARGET_CAPTURE_PLANS[key] = plan
    return plan


def _cached_target_capture_tensor(
    raw_tensor: Any,
    buffer: dict[str, Any],
    shape: tuple[int, ...],
    dtype: Any,
    element_bytes: int,
    *,
    name: str,
) -> Any:
    key = (
        int(buffer["ptr"]),
        int(buffer["bytes"]),
        int(buffer["device_id"]),
        shape,
        dtype,
        element_bytes,
    )
    tensor = _TARGET_CAPTURE_TENSORS.get(key)
    if tensor is None:
        tensor = raw_tensor(
            buffer,
            shape,
            dtype,
            element_bytes,
            name=name,
        )
        _TARGET_CAPTURE_TENSORS[key] = tensor
    return tensor


@dataclass(frozen=True)
class DeepseekV4AttentionLayerArenaLayout:
    producer_scratch_offset: int
    producer_scratch_bytes: int
    query_offset: int
    query_bytes: int
    compressor_scratch_offset: int
    compressor_scratch_bytes: int
    indexer_producer_scratch_offset: int
    indexer_producer_scratch_bytes: int
    index_query_offset: int
    index_query_bytes: int
    index_head_weights_offset: int
    index_head_weights_bytes: int
    selected_indices_offset: int
    selected_indices_bytes: int
    attention_scratch_offset: int
    attention_scratch_bytes: int
    attention_output_offset: int
    attention_output_bytes: int
    output_projection_offset: int
    output_projection_bytes: int
    mhc_scratch_offset: int
    mhc_scratch_bytes: int
    total_bytes: int

    def regions(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (name, offset, nbytes)
            for name, offset, nbytes in (
                (
                    "producer_scratch",
                    self.producer_scratch_offset,
                    self.producer_scratch_bytes,
                ),
                ("query", self.query_offset, self.query_bytes),
                (
                    "compressor_scratch",
                    self.compressor_scratch_offset,
                    self.compressor_scratch_bytes,
                ),
                (
                    "indexer_producer_scratch",
                    self.indexer_producer_scratch_offset,
                    self.indexer_producer_scratch_bytes,
                ),
                ("index_query", self.index_query_offset, self.index_query_bytes),
                (
                    "index_head_weights",
                    self.index_head_weights_offset,
                    self.index_head_weights_bytes,
                ),
                (
                    "selected_indices",
                    self.selected_indices_offset,
                    self.selected_indices_bytes,
                ),
                (
                    "attention_scratch",
                    self.attention_scratch_offset,
                    self.attention_scratch_bytes,
                ),
                (
                    "attention_output",
                    self.attention_output_offset,
                    self.attention_output_bytes,
                ),
                (
                    "output_projection",
                    self.output_projection_offset,
                    self.output_projection_bytes,
                ),
                ("mhc_scratch", self.mhc_scratch_offset, self.mhc_scratch_bytes),
            )
            if nbytes
        )


@dataclass(frozen=True)
class DeepseekV4AttentionLayerArenaBinding:
    serving_allocates: ClassVar[bool] = False
    views_only: ClassVar[bool] = True
    outputs_are_arena_views: ClassVar[bool] = True

    contract: DeepseekV4AttentionLayerContract
    tokens: int
    arena: Any
    producer_scratch: Any
    query: Any
    compressor_scratch: Any | None
    indexer_producer_scratch: Any | None
    index_query: Any | None
    index_head_weights: Any | None
    selected_indices: Any | None
    attention_scratch: Any
    attention_output: Any
    output_projection_scratch: Any
    mhc_scratch: Any

    def region_views(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (name, tensor)
            for name, tensor in (
                ("producer_scratch", self.producer_scratch),
                ("query", self.query),
                ("compressor_scratch", self.compressor_scratch),
                (
                    "indexer_producer_scratch",
                    self.indexer_producer_scratch,
                ),
                ("index_query", self.index_query),
                ("index_head_weights", self.index_head_weights),
                ("selected_indices", self.selected_indices),
                ("attention_scratch", self.attention_scratch),
                ("attention_output", self.attention_output),
                ("output_projection", self.output_projection_scratch),
                ("mhc_scratch", self.mhc_scratch),
            )
            if tensor is not None
        )


@dataclass(frozen=True)
class DeepseekV4SlidingAttentionLayerBinding:
    """Allocation-free bound C0 attention half-layer.

    Scheduler-owned selection metadata, the main KV cache, checkpoint weights,
    and the mHC ping-pong state deliberately remain outside the execution-lane
    arena.  Every transient produced by the four kernels is a view into the one
    arena binding shared across layers on that lane.
    """

    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    uses_one_lane_arena: ClassVar[bool] = True
    persistent_state_is_external: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    arena_binding: DeepseekV4AttentionLayerArenaBinding
    producer_binding: Any
    attention_binding: Any
    output_binding: Any
    mhc_binding: Any
    main_kv_cache: Any
    attn_sink: Any | None
    sm_scale: float
    residual: Any
    prev_post: Any
    prev_comb: Any
    fn: Any
    hc_scale: Any
    hc_base: Any
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_weight: Any
    norm_eps: float

    @property
    def output_projection(self) -> Any:
        return self.output_binding.output


@dataclass(frozen=True)
class DeepseekV4C128DecodeAttentionLayerBinding:
    """Allocation-free bound C128 decode attention half-layer."""

    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    uses_one_lane_arena: ClassVar[bool] = True
    persistent_state_is_external: ClassVar[bool] = True
    sequence_unique_rows: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    arena_binding: DeepseekV4AttentionLayerArenaBinding
    producer_binding: Any
    compressor_binding: Any
    attention_binding: Any
    output_binding: Any
    mhc_binding: Any
    main_kv_cache: Any
    compressed_main_cache: Any
    attn_sink: Any | None
    sm_scale: float
    residual: Any
    prev_post: Any
    prev_comb: Any
    fn: Any
    hc_scale: Any
    hc_base: Any
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_weight: Any
    norm_eps: float

    @property
    def output_projection(self) -> Any:
        return self.output_binding.output


@dataclass(frozen=True)
class DeepseekV4C128PrefillAttentionLayerBinding:
    """Allocation-free bound C128 initial-prefill attention half-layer."""

    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    uses_one_lane_arena: ClassVar[bool] = True
    persistent_state_is_external: ClassVar[bool] = True
    initial_prefill_only: ClassVar[bool] = True
    scheduler_owns_physical_selection: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    arena_binding: DeepseekV4AttentionLayerArenaBinding
    producer_binding: Any
    compressor_binding: Any
    attention_binding: Any
    output_binding: Any
    mhc_binding: Any
    main_kv_cache: Any
    compressed_main_cache: Any
    attn_sink: Any | None
    sm_scale: float
    residual: Any
    prev_post: Any
    prev_comb: Any
    fn: Any
    hc_scale: Any
    hc_base: Any
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_weight: Any
    norm_eps: float

    @property
    def output_projection(self) -> Any:
        return self.output_binding.output


@dataclass(frozen=True)
class DeepseekV4C128ContinuationAttentionLayerBinding:
    """Allocation-free bound C128 ordered-continuation attention half-layer."""

    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    uses_one_lane_arena: ClassVar[bool] = True
    persistent_state_is_external: ClassVar[bool] = True
    ordered_chunks_only: ClassVar[bool] = True
    scheduler_owns_state_transactions: ClassVar[bool] = True
    scheduler_owns_physical_selection: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    arena_binding: DeepseekV4AttentionLayerArenaBinding
    producer_binding: Any
    compressor_binding: Any
    attention_binding: Any
    output_binding: Any
    mhc_binding: Any
    main_kv_cache: Any
    compressed_main_cache: Any
    attn_sink: Any | None
    sm_scale: float
    residual: Any
    prev_post: Any
    prev_comb: Any
    fn: Any
    hc_scale: Any
    hc_base: Any
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_weight: Any
    norm_eps: float

    @property
    def output_projection(self) -> Any:
        return self.output_binding.output


@dataclass(frozen=True)
class DeepseekV4C4DecodeAttentionLayerBinding:
    """Allocation-free bound C4 learned-selection decode half-layer."""

    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    uses_one_lane_arena: ClassVar[bool] = True
    persistent_state_is_external: ClassVar[bool] = True
    selector_scratch_is_external: ClassVar[bool] = True
    selector_outputs_physical_slots: ClassVar[bool] = True
    sequence_unique_rows: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    arena_binding: DeepseekV4AttentionLayerArenaBinding
    producer_binding: Any
    compressor_binding: Any
    indexer_producer_binding: Any
    selector_binding: Any
    attention_binding: Any
    output_binding: Any
    mhc_binding: Any
    main_kv_cache: Any
    compressed_main_cache: Any
    index_cache: Any
    attn_sink: Any | None
    sm_scale: float
    residual: Any
    prev_post: Any
    prev_comb: Any
    fn: Any
    hc_scale: Any
    hc_base: Any
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_weight: Any
    norm_eps: float

    @property
    def output_projection(self) -> Any:
        return self.output_binding.output


@dataclass(frozen=True)
class DeepseekV4C4PrefillAttentionLayerBinding:
    """Allocation-free bound C4 learned-selection initial-prefill half-layer."""

    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    uses_one_lane_arena: ClassVar[bool] = True
    persistent_state_is_external: ClassVar[bool] = True
    selector_scratch_is_external: ClassVar[bool] = True
    selector_outputs_physical_slots: ClassVar[bool] = True
    selector_uses_shared_page_table: ClassVar[bool] = True
    selector_uses_causal_lengths: ClassVar[bool] = True
    initial_prefill_only: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    arena_binding: DeepseekV4AttentionLayerArenaBinding
    producer_binding: Any
    compressor_binding: Any
    indexer_producer_binding: Any
    selector_binding: Any
    attention_binding: Any
    output_binding: Any
    mhc_binding: Any
    main_kv_cache: Any
    compressed_main_cache: Any
    index_cache: Any
    attn_sink: Any | None
    sm_scale: float
    residual: Any
    prev_post: Any
    prev_comb: Any
    fn: Any
    hc_scale: Any
    hc_base: Any
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_weight: Any
    norm_eps: float

    @property
    def output_projection(self) -> Any:
        return self.output_binding.output


@dataclass(frozen=True)
class DeepseekV4C4ContinuationAttentionLayerBinding:
    """Allocation-free bound C4 learned-selection ordered continuation."""

    serving_allocates: ClassVar[bool] = False
    cuda_graph_safe: ClassVar[bool] = True
    uses_one_lane_arena: ClassVar[bool] = True
    persistent_state_is_external: ClassVar[bool] = True
    selector_scratch_is_external: ClassVar[bool] = True
    selector_outputs_physical_slots: ClassVar[bool] = True
    selector_uses_shared_page_table: ClassVar[bool] = True
    selector_uses_causal_lengths: ClassVar[bool] = True
    ordered_chunks_only: ClassVar[bool] = True
    scheduler_owns_state_transactions: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    arena_binding: DeepseekV4AttentionLayerArenaBinding
    producer_binding: Any
    compressor_binding: Any
    indexer_producer_binding: Any
    selector_binding: Any
    attention_binding: Any
    output_binding: Any
    mhc_binding: Any
    main_kv_cache: Any
    compressed_main_cache: Any
    index_cache: Any
    attn_sink: Any | None
    sm_scale: float
    residual: Any
    prev_post: Any
    prev_comb: Any
    fn: Any
    hc_scale: Any
    hc_base: Any
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_weight: Any
    norm_eps: float

    @property
    def output_projection(self) -> Any:
        return self.output_binding.output


@dataclass(frozen=True)
class DeepseekV4AttentionLayerContract:
    variant: str
    mode: str
    compression: int
    max_rows: int
    source_pages: int
    max_page_table_width: int
    max_positions: int
    hidden: int
    heads: int
    swa_width: int
    indexed_width: int
    cache_format: str
    main_page_bytes: int
    compressed_page_bytes: int
    arena: DeepseekV4AttentionLayerArenaLayout
    stages: tuple[str, ...]
    owner: str = "coordinator-local-layer-graph"
    status: str = "qualified-composite-arena-not-active"

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def workspace_reuse_scope(self) -> str:
        return "one-per-coordinator-execution-lane-across-layers"

    @property
    def selector_scratch_has_separate_owner(self) -> bool:
        return self.compression == 4

    @property
    def selector_outputs_physical_slots(self) -> bool:
        return self.compression == 4

    @property
    def output_projection_feeds_mhc_directly(self) -> bool:
        return True

    @property
    def expert_tensor_parallel(self) -> int:
        return DS4_EXPERT_TP

    @property
    def expert_parallel(self) -> bool:
        return False

    @property
    def changes_expert_tp(self) -> bool:
        return False

    @property
    def expert_global_top_k(self) -> int:
        return 6

    @property
    def expert_routes_identical_across_ranks(self) -> bool:
        return True

    @property
    def expert_local_intermediate_fraction(self) -> tuple[int, int]:
        return (1, DS4_EXPERT_TP)

    @property
    def expert_partial_output_width(self) -> int:
        return self.hidden


def plan_deepseek_v4_attention_layer(
    *,
    variant: str,
    mode: str,
    compression: int,
    max_rows: int,
    source_pages: int,
    max_page_table_width: int | None = None,
    max_positions: int,
    swa_width: int = DS4_SWA_TOKENS,
    cache_format: str = "fp8",
) -> DeepseekV4AttentionLayerContract:
    variant = str(variant).strip().lower()
    mode = str(mode).strip().lower()
    compression = int(compression)
    max_rows = int(max_rows)
    source_pages = int(source_pages)
    max_page_table_width = (
        source_pages
        if max_page_table_width is None
        else int(max_page_table_width)
    )
    max_positions = int(max_positions)
    swa_width = int(swa_width)
    if max_positions <= 0:
        raise ValueError("DeepSeek V4 attention-layer max_positions must be positive")

    producer = plan_deepseek_v4_attention_producer(
        variant=variant,
        max_rows=max_rows,
        cache_format=cache_format,
    )
    geometry = producer.geometry
    if compression == 4:
        indexer = plan_deepseek_v4_attention_indexer(
            variant=variant,
            max_rows=max_rows,
        )
        indexed_width = indexer.geometry.top_k
    else:
        indexer = None
        indexed_width = 0 if compression == 0 else max_page_table_width * 2
        if compression == 128 and mode in {"extend", "verify", "draft_extend"}:
            indexed_width = max(indexed_width, DS4_SM120_PREFILL_SELECTION_TILE)
    compressor = (
        plan_deepseek_v4_attention_compressor(
            variant=variant,
            compress_ratio=compression,
            max_rows=max_rows,
            cache_format=cache_format,
        )
        if compression in (4, 128)
        else None
    )
    attention = plan_deepseek_v4_compressed_mla(
        mode=mode,
        rows=max_rows,
        heads=geometry.heads,
        source_pages=source_pages,
        max_page_table_width=max_page_table_width,
        swa_width=swa_width,
        compression=compression,
        indexed_width=indexed_width,
        cache_format=cache_format,
    )
    output = plan_deepseek_v4_attention_output(
        variant=variant,
        max_rows=max_rows,
    )
    mhc = plan_deepseek_v4_mhc(variant=variant, max_rows=max_rows)

    cursor = 0

    def reserve(nbytes: int) -> tuple[int, int]:
        nonlocal cursor
        nbytes = int(nbytes)
        offset = _align_up(cursor)
        if nbytes:
            cursor = offset + nbytes
        return offset, nbytes

    producer_scratch_offset, producer_scratch_bytes = reserve(
        producer.scratch.total_bytes
    )
    query_offset, query_bytes = reserve(max_rows * geometry.heads * DS4_HEAD_DIM * 2)
    compressor_scratch_offset, compressor_scratch_bytes = reserve(
        0 if compressor is None else compressor.scratch.total_bytes
    )
    indexer_producer_scratch_offset, indexer_producer_scratch_bytes = reserve(
        0 if indexer is None else indexer.scratch.total_bytes
    )
    index_query_offset, index_query_bytes = reserve(
        0 if indexer is None else max_rows * indexer.geometry.query_width
    )
    index_head_weights_offset, index_head_weights_bytes = reserve(
        0 if indexer is None else max_rows * indexer.geometry.heads * 4
    )
    selected_indices_offset, selected_indices_bytes = reserve(
        0 if indexer is None else max_rows * indexed_width * 4
    )
    attention_scratch_offset, attention_scratch_bytes = reserve(attention.scratch_bytes)
    attention_output_offset, attention_output_bytes = reserve(
        max_rows * geometry.heads * DS4_HEAD_DIM * 2
    )
    output_projection_offset, output_projection_bytes = reserve(
        output.scratch.total_bytes
    )
    mhc_scratch_offset, mhc_scratch_bytes = reserve(mhc.scratch.total_bytes)
    arena = DeepseekV4AttentionLayerArenaLayout(
        producer_scratch_offset=producer_scratch_offset,
        producer_scratch_bytes=producer_scratch_bytes,
        query_offset=query_offset,
        query_bytes=query_bytes,
        compressor_scratch_offset=compressor_scratch_offset,
        compressor_scratch_bytes=compressor_scratch_bytes,
        indexer_producer_scratch_offset=indexer_producer_scratch_offset,
        indexer_producer_scratch_bytes=indexer_producer_scratch_bytes,
        index_query_offset=index_query_offset,
        index_query_bytes=index_query_bytes,
        index_head_weights_offset=index_head_weights_offset,
        index_head_weights_bytes=index_head_weights_bytes,
        selected_indices_offset=selected_indices_offset,
        selected_indices_bytes=selected_indices_bytes,
        attention_scratch_offset=attention_scratch_offset,
        attention_scratch_bytes=attention_scratch_bytes,
        attention_output_offset=attention_output_offset,
        attention_output_bytes=attention_output_bytes,
        output_projection_offset=output_projection_offset,
        output_projection_bytes=output_projection_bytes,
        mhc_scratch_offset=mhc_scratch_offset,
        mhc_scratch_bytes=mhc_scratch_bytes,
        total_bytes=_align_up(cursor),
    )
    stages = ["mhc-attention-pre-normalized-input", "main-query-and-kv-producer"]
    if compressor is not None:
        stages.append(f"c{compression}-compressor-state-and-cache-update")
    if indexer is not None:
        stages.extend(["c4-index-query-producer", "c4-learned-physical-slot-selection"])
    elif compression == 128:
        stages.append("c128-all-completed-physical-slots")
    stages.extend(
        [
            "compressed-mla-with-sliding-and-indexed-cache",
            "grouped-output-projection",
            "mhc-attention-post-ffn-pre",
        ]
    )
    return DeepseekV4AttentionLayerContract(
        variant=variant,
        mode=mode,
        compression=compression,
        max_rows=max_rows,
        source_pages=source_pages,
        max_page_table_width=max_page_table_width,
        max_positions=max_positions,
        hidden=geometry.hidden,
        heads=geometry.heads,
        swa_width=swa_width,
        indexed_width=indexed_width,
        cache_format=attention.cache_format,
        main_page_bytes=attention.main_page_bytes,
        compressed_page_bytes=attention.indexed_page_bytes,
        arena=arena,
        stages=tuple(stages),
    )


def bind_deepseek_v4_attention_layer(
    contract: DeepseekV4AttentionLayerContract,
    *,
    arena: Any,
    tokens: int,
) -> DeepseekV4AttentionLayerArenaBinding:
    import torch

    tokens = int(tokens)
    if not 1 <= tokens <= contract.max_rows:
        raise ValueError(
            f"composite layer tokens must be in [1, {contract.max_rows}], got {tokens}"
        )
    if not isinstance(arena, torch.Tensor):
        raise TypeError("composite layer arena must be a torch.Tensor")
    if arena.dtype != torch.uint8 or arena.ndim != 1 or not arena.is_contiguous():
        raise ValueError("composite layer arena must be contiguous rank-1 torch.uint8")
    if arena.numel() < contract.arena.total_bytes:
        raise ValueError(
            "composite layer arena needs "
            f"{contract.arena.total_bytes} bytes, got {arena.numel()}"
        )
    storage = arena[: contract.arena.total_bytes]

    def view(
        offset: int,
        nbytes: int,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ):
        if not nbytes:
            return None
        raw = storage.narrow(0, int(offset), int(nbytes))
        tensor = raw if dtype == torch.uint8 else raw.view(dtype)
        return tensor.view(shape)

    layout = contract.arena
    return DeepseekV4AttentionLayerArenaBinding(
        contract=contract,
        tokens=tokens,
        arena=storage,
        producer_scratch=view(
            layout.producer_scratch_offset,
            layout.producer_scratch_bytes,
            dtype=torch.uint8,
            shape=(layout.producer_scratch_bytes,),
        ),
        query=view(
            layout.query_offset,
            tokens * contract.heads * DS4_HEAD_DIM * 2,
            dtype=torch.bfloat16,
            shape=(tokens, contract.heads, DS4_HEAD_DIM),
        ),
        compressor_scratch=view(
            layout.compressor_scratch_offset,
            layout.compressor_scratch_bytes,
            dtype=torch.uint8,
            shape=(layout.compressor_scratch_bytes,),
        ),
        indexer_producer_scratch=view(
            layout.indexer_producer_scratch_offset,
            layout.indexer_producer_scratch_bytes,
            dtype=torch.uint8,
            shape=(layout.indexer_producer_scratch_bytes,),
        ),
        index_query=view(
            layout.index_query_offset,
            tokens * DS4_INDEX_HEADS * DS4_INDEX_HEAD_DIM
            if contract.compression == 4
            else 0,
            dtype=torch.float8_e4m3fn,
            shape=(tokens, DS4_INDEX_HEADS, DS4_INDEX_HEAD_DIM),
        ),
        index_head_weights=view(
            layout.index_head_weights_offset,
            tokens * DS4_INDEX_HEADS * 4 if contract.compression == 4 else 0,
            dtype=torch.float32,
            shape=(tokens, DS4_INDEX_HEADS),
        ),
        selected_indices=view(
            layout.selected_indices_offset,
            tokens * contract.indexed_width * 4 if contract.compression == 4 else 0,
            dtype=torch.int32,
            shape=(tokens, contract.indexed_width),
        ),
        attention_scratch=view(
            layout.attention_scratch_offset,
            layout.attention_scratch_bytes,
            dtype=torch.uint8,
            shape=(layout.attention_scratch_bytes,),
        ),
        attention_output=view(
            layout.attention_output_offset,
            tokens * contract.heads * DS4_HEAD_DIM * 2,
            dtype=torch.bfloat16,
            shape=(tokens, contract.heads, DS4_HEAD_DIM),
        ),
        output_projection_scratch=view(
            layout.output_projection_offset,
            layout.output_projection_bytes,
            dtype=torch.uint8,
            shape=(layout.output_projection_bytes,),
        ),
        mhc_scratch=view(
            layout.mhc_scratch_offset,
            layout.mhc_scratch_bytes,
            dtype=torch.uint8,
            shape=(layout.mhc_scratch_bytes,),
        ),
    )


def bind_deepseek_v4_sliding_attention_layer(
    contract: DeepseekV4AttentionLayerContract,
    *,
    arena: Any,
    hidden_states: Any,
    positions: Any,
    main_slots: Any,
    cos_sin_cache: Any,
    main_kv_cache: Any,
    swa_indices: Any,
    swa_lengths: Any,
    producer_weights: Any,
    output_weights: Any,
    residual: Any,
    prev_post: Any,
    prev_comb: Any,
    fn: Any,
    hc_scale: Any,
    hc_base: Any,
    norm_weight: Any,
    residual_out: Any,
    y_out: Any,
    post_out: Any,
    comb_out: Any,
    attn_sink: Any | None = None,
    sm_scale: float | None = None,
    producer_eps: float = 1.0e-6,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1.0e-6,
) -> DeepseekV4SlidingAttentionLayerBinding:
    """Bind the producer -> sliding MLA -> WO -> mHC chain without allocations.

    This is intentionally C0-only.  C4/C128 add persistent compressor state and
    selection lifecycles and must qualify independently before sharing this run
    path.
    """

    import torch
    from b12x.attention import compressed_sparse_mla as compressed_mla, dsv4_producer
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if contract.compression != 0:
        raise ValueError(
            "bound sliding attention layer requires compression=0; "
            f"got {contract.compression}"
        )
    if contract.mode not in {"decode", "extend", "verify", "draft_extend"}:
        raise ValueError(f"unsupported sliding attention mode {contract.mode!r}")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("sliding attention hidden_states must be a rank-2 tensor")
    tokens = int(hidden_states.shape[0])
    arena_binding = bind_deepseek_v4_attention_layer(
        contract,
        arena=arena,
        tokens=tokens,
    )
    geometry = deepseek_v4_producer_geometry(contract.variant)
    expected_m = contract.max_rows

    producer_plan = _cached_target_capture_plan(
        dsv4_producer.plan,
        dsv4_producer.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=geometry.heads,
            head_dim=DS4_HEAD_DIM,
            nope_dim=DS4_NOPE_DIM,
            rope_dim=DS4_ROPE_DIM,
            page_size=DS4_SOURCE_PAGE_TOKENS,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    producer_binding = dsv4_producer.bind(
        producer_plan,
        scratch=arena_binding.producer_scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=main_slots,
        cos_sin_cache=cos_sin_cache,
        main_kv_cache=main_kv_cache,
        query=arena_binding.query,
        weights=producer_weights,
        eps=float(producer_eps),
        expected_m=expected_m,
    )

    attention_contract = plan_deepseek_v4_compressed_mla(
        mode=contract.mode,
        rows=contract.max_rows,
        heads=contract.heads,
        source_pages=contract.source_pages,
        max_page_table_width=contract.max_page_table_width,
        swa_width=contract.swa_width,
        compression=0,
        cache_format=contract.cache_format,
    )
    attention_plan = _cached_target_capture_plan(
        compressed_mla.plan,
        compressed_mla.Caps(
            device=hidden_states.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=contract.heads,
            head_dim=DS4_HEAD_DIM,
            v_head_dim=DS4_HEAD_DIM,
            max_width=attention_contract.total_width,
            max_page_table_width=contract.max_page_table_width,
            max_q_rows=contract.max_rows,
            max_batch=contract.max_rows,
            max_kv_rows=0,
            max_chunks_per_row=attention_contract.max_chunks_per_row,
            page_size=DS4_SOURCE_PAGE_TOKENS,
        ),
    )
    attention_binding = compressed_mla.bind(
        attention_plan,
        scratch=arena_binding.attention_scratch,
        q=arena_binding.query,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
    )
    attention_binding.scratch.mode = contract.mode
    attention_binding.scratch.fixed_capacity = True
    attention_binding.scratch.use_cuda_graph = True

    output_contract = plan_deepseek_v4_attention_output(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    output_geometry = output_contract.geometry
    output_plan = _cached_target_capture_plan(
        wo_projection.plan,
        wo_projection.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            groups=output_geometry.groups,
            group_width=output_geometry.group_width,
            rank=output_geometry.rank,
            hidden=output_geometry.hidden,
            dtype=torch.bfloat16,
        ),
    )
    output_binding = wo_projection.bind_inv_rope(
        output_plan,
        scratch=arena_binding.output_projection_scratch,
        o=arena_binding.attention_output,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=output_weights,
        heads_per_group=output_geometry.heads_per_group,
        nope_dim=DS4_NOPE_DIM,
        rope_dim=DS4_ROPE_DIM,
        expected_m=expected_m,
    )

    mhc_contract = plan_deepseek_v4_mhc(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    mhc_plan = _cached_target_capture_plan(
        mhc.plan,
        mhc.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden_size=contract.hidden,
            split_k=mhc_contract.geometry.split_k,
            dtype=torch.bfloat16,
        ),
    )
    mhc_binding = mhc.bind(
        mhc_plan,
        scratch=arena_binding.mhc_scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=y_out,
        post=post_out,
        comb=comb_out,
        out=residual_out,
    )
    return DeepseekV4SlidingAttentionLayerBinding(
        arena_binding=arena_binding,
        producer_binding=producer_binding,
        attention_binding=attention_binding,
        output_binding=output_binding,
        mhc_binding=mhc_binding,
        main_kv_cache=main_kv_cache,
        attn_sink=attn_sink,
        sm_scale=(DS4_HEAD_DIM**-0.5 if sm_scale is None else float(sm_scale)),
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_weight=norm_weight,
        norm_eps=float(norm_eps),
    )


def run_deepseek_v4_sliding_attention_layer(
    binding: DeepseekV4SlidingAttentionLayerBinding,
) -> tuple[Any, Any, Any, Any]:
    """Run a bound C0 half-layer and return residual, post, comb, normalized y."""

    from b12x.attention import compressed_sparse_mla as compressed_mla, dsv4_producer
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if not isinstance(binding, DeepseekV4SlidingAttentionLayerBinding):
        raise TypeError("sliding attention layer requires its fixed binding")
    dsv4_producer.run(binding=binding.producer_binding)
    compressed_mla.run(
        binding=binding.attention_binding,
        swa_k_cache=binding.main_kv_cache,
        sm_scale=binding.sm_scale,
        swa_page_size=DS4_SOURCE_PAGE_TOKENS,
        attn_sink=binding.attn_sink,
        expected_num_q_heads=binding.arena_binding.contract.heads,
        backend="sm120",
        out=binding.arena_binding.attention_output,
    )
    delta = wo_projection.run_inv_rope(binding=binding.output_binding)
    return mhc.run_post_pre(
        delta,
        binding.residual,
        binding.prev_post,
        binding.prev_comb,
        binding.fn,
        binding.hc_scale,
        binding.hc_base,
        rms_eps=binding.rms_eps,
        hc_eps=binding.hc_eps,
        sinkhorn_iters=binding.sinkhorn_iters,
        norm_weight=binding.norm_weight,
        norm_eps=binding.norm_eps,
        binding=binding.mhc_binding,
    )


def deepseek_v4_attention_layer_arena_nbytes(
    *,
    variant: str,
    mode: str,
    compression: int,
    max_rows: int,
    source_pages: int,
    max_page_table_width: int | None = None,
    max_positions: int,
    swa_width: int = DS4_SWA_TOKENS,
    cache_format: str = "fp8",
) -> int:
    """Return the fixed composite arena size for Rust-side allocation."""

    return plan_deepseek_v4_attention_layer(
        variant=variant,
        mode=mode,
        compression=compression,
        max_rows=max_rows,
        source_pages=source_pages,
        max_page_table_width=max_page_table_width,
        max_positions=max_positions,
        swa_width=swa_width,
        cache_format=cache_format,
    ).arena.total_bytes


def deepseek_v4_c4_selector_scratch_nbytes(
    *,
    variant: str,
    mode: str,
    max_rows: int,
    source_pages: int,
    max_page_table_width: int | None = None,
) -> int:
    """Return the fixed external learned-selector scratch size for C4.

    The composite attention arena deliberately excludes this scratch because
    the scheduler owns C4 physical-slot selection and its page-table
    transaction. Rust queries the exact SparkInfer plan at startup and reserves
    the maximum once per execution lane; serving only reuses that storage.
    """

    import math
    import torch
    from b12x.attention import dsa_indexer

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"decode", "prefill"}:
        raise ValueError(
            "DeepSeek V4 C4 selector mode must be decode or prefill, got "
            f"{mode!r}"
        )
    contract = plan_deepseek_v4_attention_layer(
        variant=variant,
        mode="decode" if normalized_mode == "decode" else "extend",
        compression=4,
        max_rows=max_rows,
        source_pages=source_pages,
        max_page_table_width=max_page_table_width,
        max_positions=max_rows,
        swa_width=DS4_SWA_TOKENS,
    )
    plan = _cached_target_capture_plan(
        dsa_indexer.plan,
        dsa_indexer.Caps(
            device=torch.device("cuda:0"),
            source_layout=dsa_indexer.SOURCE_LAYOUT_PAGED,
            num_q_heads=DS4_INDEX_HEADS,
            max_q_rows=max_rows,
            max_page_table_width=contract.max_page_table_width,
            topk=contract.indexed_width,
            mode=normalized_mode,
            page_size=64,
            shared_page_table=normalized_mode == "prefill",
        ),
    )
    specs = plan.scratch_specs()
    if len(specs) != 1:
        raise RuntimeError(
            "DeepSeek V4 C4 selector expected one fixed scratch buffer, got "
            f"{len(specs)}"
        )
    spec = specs[0]
    return math.prod(spec.shape) * torch.empty((), dtype=spec.dtype).element_size()


def prepare_deepseek_v4_sliding_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_sliding_attention_layer_capture(
        ctx, prepare_only=True, **kwargs
    )


def capture_deepseek_v4_sliding_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_sliding_attention_layer_capture(
        ctx, prepare_only=False, **kwargs
    )


def _run_deepseek_v4_sliding_attention_layer_capture(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    """Run a target C0 half-layer against caller-owned serving storage.

    The graph stops at normalized FFN input, immediately before the unchanged
    strict-TP4 expert handoff.  The four-lane residual plus post/comb state
    remain caller-owned so the matching post-dispatch graph can fuse the FFN
    boundary into the next attention pre-mix.
    """

    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.attention import dsv4_producer
    from b12x.gemm import wo_projection

    variant = str(kwargs["variant"]).strip().lower()
    mode = str(kwargs["mode"]).strip().lower()
    rows = int(kwargs["rows"])
    max_rows = int(kwargs["max_rows"])
    source_pages = int(kwargs["source_pages"])
    max_page_table_width = int(
        kwargs.get("max_page_table_width", source_pages)
    )
    max_positions = int(kwargs["max_positions"])
    swa_width = int(kwargs.get("swa_width", DS4_SWA_TOKENS))
    cache_format = str(kwargs.get("cache_format", "fp8"))
    if not 1 <= rows <= max_rows:
        raise ValueError(
            f"DeepSeek V4 sliding capture rows must be in [1, {max_rows}], got {rows}"
        )
    contract = plan_deepseek_v4_attention_layer(
        variant=variant,
        mode=mode,
        compression=0,
        max_rows=max_rows,
        source_pages=source_pages,
        max_page_table_width=max_page_table_width,
        max_positions=max_positions,
        swa_width=swa_width,
        cache_format=cache_format,
    )
    geometry = deepseek_v4_producer_geometry(variant)
    output_geometry = plan_deepseek_v4_attention_output(
        variant=variant, max_rows=max_rows
    ).geometry
    hidden = geometry.hidden
    heads = geometry.heads
    q_rank = geometry.q_lora_rank
    query_width = geometry.query_width
    output_projected_width = output_geometry.groups * output_geometry.rank
    hc_mult = 4
    hc_mixes = (2 + hc_mult) * hc_mult
    buffers = ctx["buffers"]
    required = {
        "workspace": contract.arena.total_bytes,
        "hidden_states": rows * hidden * 2,
        "normalized_output": rows * hidden * 2,
        "positions": rows * 4,
        "main_slots": rows * 4,
        "cos_sin_cache": max_positions * DS4_ROPE_DIM * 4,
        "main_kv_cache": source_pages * contract.main_page_bytes,
        "selected_indices": rows * swa_width * 4,
        "selected_lengths": rows * 4,
        "residual": rows * hc_mult * hidden * 2,
        "prev_post": rows * hc_mult * 4,
        "prev_comb": rows * hc_mult * hc_mult * 4,
        "residual_out": rows * hc_mult * hidden * 2,
        "post_out": rows * hc_mult * 4,
        "comb_out": rows * hc_mult * hc_mult * 4,
        "wq_a_weight": q_rank * hidden,
        "wq_a_scale": _ceil_div(q_rank, 128) * _ceil_div(hidden, 128),
        "wq_b_weight": query_width * q_rank,
        "wq_b_scale": _ceil_div(query_width, 128) * _ceil_div(q_rank, 128),
        "wkv_weight": DS4_HEAD_DIM * hidden,
        "wkv_scale": _ceil_div(DS4_HEAD_DIM, 128) * _ceil_div(hidden, 128),
        "q_norm_weight": q_rank * 2,
        "kv_norm_weight": DS4_HEAD_DIM * 2,
        "wo_a_weight": output_projected_width * output_geometry.group_width,
        "wo_a_scale": _ceil_div(output_projected_width, 128)
        * _ceil_div(output_geometry.group_width, 128),
        "wo_b_weight": hidden * output_projected_width,
        "wo_b_scale": _ceil_div(hidden, 128)
        * _ceil_div(output_projected_width, 128),
        "attn_sink": heads * 4,
        "hc_fn": hc_mixes * hc_mult * hidden * 4,
        "hc_scale": 3 * 4,
        "hc_base": hc_mixes * 4,
        "norm_weight": hidden * 2,
    }
    device_id = _validate_capture_buffers(
        buffers, required, anchor="hidden_states"
    )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    weight_names = (
        "wq_a_weight",
        "wq_a_scale",
        "wq_b_weight",
        "wq_b_scale",
        "wkv_weight",
        "wkv_scale",
        "q_norm_weight",
        "kv_norm_weight",
        "wo_a_weight",
        "wo_a_scale",
        "wo_b_weight",
        "wo_b_scale",
    )
    state_key = (
        device_id,
        *map(ord, variant),
        *map(ord, contract.cache_format),
        *(int(buffers[name]["ptr"]) for name in weight_names),
    )
    prepared_key = (
        device_id,
        *map(ord, mode),
        *map(ord, contract.cache_format),
        rows,
        max_rows,
        source_pages,
        max_page_table_width,
        max_positions,
        swa_width,
        *(int(buffers[name]["ptr"]) for name in required),
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        state = _SLIDING_CAPTURE_STATE.get(state_key)
        if state is None:
            fp8 = torch.float8_e4m3fn
            e8m0 = torch.float8_e8m0fnu
            producer_weights = dsv4_producer.pack_weights(
                _raw_tensor(
                    buffers["wq_a_weight"],
                    (q_rank, hidden),
                    fp8,
                    1,
                    name="target attention wq_a weight",
                ),
                _raw_tensor(
                    buffers["wq_a_scale"],
                    (_ceil_div(q_rank, 128), _ceil_div(hidden, 128)),
                    e8m0,
                    1,
                    name="target attention wq_a scale",
                ),
                _raw_tensor(
                    buffers["wq_b_weight"],
                    (query_width, q_rank),
                    fp8,
                    1,
                    name="target attention wq_b weight",
                ),
                _raw_tensor(
                    buffers["wq_b_scale"],
                    (_ceil_div(query_width, 128), _ceil_div(q_rank, 128)),
                    e8m0,
                    1,
                    name="target attention wq_b scale",
                ),
                _raw_tensor(
                    buffers["wkv_weight"],
                    (DS4_HEAD_DIM, hidden),
                    fp8,
                    1,
                    name="target attention wkv weight",
                ),
                _raw_tensor(
                    buffers["wkv_scale"],
                    (_ceil_div(DS4_HEAD_DIM, 128), _ceil_div(hidden, 128)),
                    e8m0,
                    1,
                    name="target attention wkv scale",
                ),
                _raw_tensor(
                    buffers["q_norm_weight"],
                    (q_rank,),
                    torch.bfloat16,
                    2,
                    name="target attention q norm",
                ),
                _raw_tensor(
                    buffers["kv_norm_weight"],
                    (DS4_HEAD_DIM,),
                    torch.bfloat16,
                    2,
                    name="target attention kv norm",
                ),
            )
            output_weights = wo_projection.pack_weights(
                _raw_tensor(
                    buffers["wo_a_weight"],
                    (output_projected_width, output_geometry.group_width),
                    fp8,
                    1,
                    name="target attention wo_a weight",
                ),
                _raw_tensor(
                    buffers["wo_a_scale"],
                    (
                        _ceil_div(output_projected_width, 128),
                        _ceil_div(output_geometry.group_width, 128),
                    ),
                    e8m0,
                    1,
                    name="target attention wo_a scale",
                ),
                _raw_tensor(
                    buffers["wo_b_weight"],
                    (hidden, output_projected_width),
                    fp8,
                    1,
                    name="target attention wo_b weight",
                ),
                _raw_tensor(
                    buffers["wo_b_scale"],
                    (
                        _ceil_div(hidden, 128),
                        _ceil_div(output_projected_width, 128),
                    ),
                    e8m0,
                    1,
                    name="target attention wo_b scale",
                ),
                groups=output_geometry.groups,
                group_width=output_geometry.group_width,
                rank=output_geometry.rank,
                hidden=hidden,
            )
            state = (producer_weights, output_weights)
            _SLIDING_CAPTURE_STATE[state_key] = state
        producer_weights, output_weights = state
        binding = bind_deepseek_v4_sliding_attention_layer(
            contract,
            arena=_raw_tensor(
                buffers["workspace"],
                (contract.arena.total_bytes,),
                torch.uint8,
                1,
                name="target sliding attention workspace",
            ),
            hidden_states=_raw_tensor(
                buffers["hidden_states"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="target normalized attention input",
            ),
            positions=_raw_tensor(
                buffers["positions"],
                (rows,),
                torch.int32,
                4,
                name="target attention positions",
            ),
            main_slots=_raw_tensor(
                buffers["main_slots"],
                (rows,),
                torch.int32,
                4,
                name="target main KV slots",
            ),
            cos_sin_cache=_raw_tensor(
                buffers["cos_sin_cache"],
                (max_positions, DS4_ROPE_DIM),
                torch.float32,
                4,
                name="target attention cos/sin cache",
            ),
            main_kv_cache=_raw_tensor(
                buffers["main_kv_cache"],
                (source_pages, contract.main_page_bytes),
                torch.uint8,
                1,
                name="target paged main KV cache",
            ),
            swa_indices=_raw_tensor(
                buffers["selected_indices"],
                (rows, swa_width),
                torch.int32,
                4,
                name="target sliding physical selection",
            ),
            swa_lengths=_raw_tensor(
                buffers["selected_lengths"],
                (rows,),
                torch.int32,
                4,
                name="target sliding selection lengths",
            ),
            producer_weights=producer_weights,
            output_weights=output_weights,
            residual=_raw_tensor(
                buffers["residual"],
                (rows, hc_mult, hidden),
                torch.bfloat16,
                2,
                name="target attention HC residual",
            ),
            prev_post=_raw_tensor(
                buffers["prev_post"],
                (rows, hc_mult),
                torch.float32,
                4,
                name="target attention previous post mix",
            ),
            prev_comb=_raw_tensor(
                buffers["prev_comb"],
                (rows, hc_mult, hc_mult),
                torch.float32,
                4,
                name="target attention previous combination mix",
            ),
            fn=_raw_tensor(
                buffers["hc_fn"],
                (hc_mixes, hc_mult * hidden),
                torch.float32,
                4,
                name="target FFN HC function",
            ),
            hc_scale=_raw_tensor(
                buffers["hc_scale"],
                (3,),
                torch.float32,
                4,
                name="target FFN HC scale",
            ),
            hc_base=_raw_tensor(
                buffers["hc_base"],
                (hc_mixes,),
                torch.float32,
                4,
                name="target FFN HC base",
            ),
            norm_weight=_raw_tensor(
                buffers["norm_weight"],
                (hidden,),
                torch.bfloat16,
                2,
                name="target FFN norm",
            ),
            residual_out=_raw_tensor(
                buffers["residual_out"],
                (rows, hc_mult, hidden),
                torch.bfloat16,
                2,
                name="target post-attention residual",
            ),
            y_out=_raw_tensor(
                buffers["normalized_output"],
                (rows, hidden),
                torch.bfloat16,
                2,
                name="target normalized TP4 dispatch hidden",
            ),
            post_out=_raw_tensor(
                buffers["post_out"],
                (rows, hc_mult),
                torch.float32,
                4,
                name="target FFN post mix",
            ),
            comb_out=_raw_tensor(
                buffers["comb_out"],
                (rows, hc_mult, hc_mult),
                torch.float32,
                4,
                name="target FFN combination mix",
            ),
            attn_sink=_raw_tensor(
                buffers["attn_sink"],
                (heads,),
                torch.float32,
                4,
                name="target attention sink",
            ),
            producer_eps=float(kwargs.get("producer_eps", 1.0e-6)),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            sinkhorn_iters=int(kwargs.get("sinkhorn_iters", 20)),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
        )
        if not prepare_only and prepared_key not in _SLIDING_CAPTURE_PREPARED:
            raise RuntimeError(
                "DeepSeek V4 target sliding attention was not prepared for its fixed graph"
            )
        run_deepseek_v4_sliding_attention_layer(binding)
        if prepare_only:
            _SLIDING_CAPTURE_PREPARED.add(prepared_key)


def prepare_deepseek_v4_c128_prefill_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_c128_attention_layer_capture(
        ctx, lifecycle="prefill", prepare_only=True, **kwargs
    )


def capture_deepseek_v4_c128_prefill_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_c128_attention_layer_capture(
        ctx, lifecycle="prefill", prepare_only=False, **kwargs
    )


def prepare_deepseek_v4_c128_continuation_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_c128_attention_layer_capture(
        ctx, lifecycle="continuation", prepare_only=True, **kwargs
    )


def capture_deepseek_v4_c128_continuation_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_c128_attention_layer_capture(
        ctx, lifecycle="continuation", prepare_only=False, **kwargs
    )


def _run_deepseek_v4_c128_attention_layer_capture(
    ctx: dict[str, Any], *, lifecycle: str, prepare_only: bool, **kwargs: Any
) -> None:
    """Run one C128 half-layer against fixed scheduler-owned serving storage."""

    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.attention import dsv4_compressor, dsv4_producer
    from b12x.gemm import wo_projection

    if lifecycle not in {"prefill", "continuation"}:
        raise ValueError(f"unsupported DeepSeek V4 C128 lifecycle {lifecycle!r}")
    variant = str(kwargs["variant"]).strip().lower()
    rows = int(kwargs["rows"])
    max_rows = int(kwargs["max_rows"])
    source_pages = int(kwargs["source_pages"])
    max_page_table_width = int(
        kwargs.get("max_page_table_width", source_pages)
    )
    max_positions = int(kwargs["max_positions"])
    swa_width = int(kwargs.get("swa_width", DS4_SWA_TOKENS))
    cache_format = str(kwargs.get("cache_format", "fp8"))
    if not 1 <= rows <= max_rows:
        raise ValueError(
            f"DeepSeek V4 C128 capture rows must be in [1, {max_rows}], got {rows}"
        )
    contract = plan_deepseek_v4_attention_layer(
        variant=variant,
        mode="extend",
        compression=128,
        max_rows=max_rows,
        source_pages=source_pages,
        max_page_table_width=max_page_table_width,
        max_positions=max_positions,
        swa_width=swa_width,
        cache_format=cache_format,
    )
    geometry = deepseek_v4_producer_geometry(variant)
    output_geometry = plan_deepseek_v4_attention_output(
        variant=variant, max_rows=max_rows
    ).geometry
    hidden = geometry.hidden
    heads = geometry.heads
    q_rank = geometry.q_lora_rank
    query_width = geometry.query_width
    output_projected_width = output_geometry.groups * output_geometry.rank
    hc_mult = 4
    hc_mixes = (2 + hc_mult) * hc_mult
    group_capacity = (
        max_rows // 128
        if lifecycle == "prefill"
        else _ceil_div(max_rows, 128) + 1
    )
    compressed_page_bytes = contract.compressed_page_bytes
    # Two position-indexed windows retain accepted history while a speculative
    # suffix occupies distinct ring slots.
    state_rows = 256
    state_width = 512
    indexed_width = contract.indexed_width
    buffers = ctx["buffers"]
    required = {
        "workspace": contract.arena.total_bytes,
        "hidden_states": rows * hidden * 2,
        "normalized_output": rows * hidden * 2,
        "positions": rows * 4,
        "main_slots": rows * 4,
        "cos_sin_cache": (max_rows + 1) * DS4_ROPE_DIM * 4,
        "main_kv_cache": source_pages * contract.main_page_bytes,
        "swa_indices": rows * swa_width * 4,
        "swa_lengths": rows * 4,
        "active_groups": 4,
        "group_source_starts": group_capacity * 4,
        "group_sequence_slots": group_capacity * 4,
        "group_rope_positions": group_capacity * 4,
        "compressed_slots": group_capacity * 4,
        "active_sequences": 4,
        "sequence_offsets": 2 * 4,
        "sequence_start_positions": 4,
        "state_sequence_ids": 4,
        "compressed_main_cache": source_pages * compressed_page_bytes,
        "main_kv_state": state_rows * state_width * 4,
        "main_score_state": state_rows * state_width * 4,
        "indexed_indices": indexed_width * 4,
        "indexed_lengths": rows * 4,
        "residual": rows * hc_mult * hidden * 2,
        "prev_post": rows * hc_mult * 4,
        "prev_comb": rows * hc_mult * hc_mult * 4,
        "residual_out": rows * hc_mult * hidden * 2,
        "post_out": rows * hc_mult * 4,
        "comb_out": rows * hc_mult * hc_mult * 4,
        "wq_a_weight": q_rank * hidden,
        "wq_a_scale": _ceil_div(q_rank, 128) * _ceil_div(hidden, 128),
        "wq_b_weight": query_width * q_rank,
        "wq_b_scale": _ceil_div(query_width, 128) * _ceil_div(q_rank, 128),
        "wkv_weight": DS4_HEAD_DIM * hidden,
        "wkv_scale": _ceil_div(DS4_HEAD_DIM, 128) * _ceil_div(hidden, 128),
        "q_norm_weight": q_rank * 2,
        "kv_norm_weight": DS4_HEAD_DIM * 2,
        "compressor_main_wkv": state_width * hidden * 2,
        "compressor_main_wgate": state_width * hidden * 2,
        "compressor_main_ape": 128 * state_width * 4,
        "compressor_main_norm": state_width * 2,
        "wo_a_weight": output_projected_width * output_geometry.group_width,
        "wo_a_scale": _ceil_div(output_projected_width, 128)
        * _ceil_div(output_geometry.group_width, 128),
        "wo_b_weight": hidden * output_projected_width,
        "wo_b_scale": _ceil_div(hidden, 128)
        * _ceil_div(output_projected_width, 128),
        "attn_sink": heads * 4,
        "hc_fn": hc_mixes * hc_mult * hidden * 4,
        "hc_scale": 3 * 4,
        "hc_base": hc_mixes * 4,
        "norm_weight": hidden * 2,
    }
    device_id = _validate_capture_buffers(
        buffers, required, anchor="hidden_states"
    )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    weight_names = (
        "wq_a_weight",
        "wq_a_scale",
        "wq_b_weight",
        "wq_b_scale",
        "wkv_weight",
        "wkv_scale",
        "q_norm_weight",
        "kv_norm_weight",
        "compressor_main_wkv",
        "compressor_main_wgate",
        "compressor_main_ape",
        "compressor_main_norm",
        "wo_a_weight",
        "wo_a_scale",
        "wo_b_weight",
        "wo_b_scale",
    )
    state_key = (
        device_id,
        *map(ord, variant),
        *map(ord, contract.cache_format),
        *(int(buffers[name]["ptr"]) for name in weight_names),
    )
    # Kernel shape warmup is device-specific, but not stream- or
    # weight-pointer-specific. Every layer still gets its own packed-weight
    # state and every stream gets its own captured graph.
    prepared_key = (
        device_id,
        *map(ord, lifecycle),
        *map(ord, contract.cache_format),
        rows,
        max_rows,
        source_pages,
        max_page_table_width,
        max_positions,
        swa_width,
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        state = _C128_CAPTURE_STATE.get(state_key)
        if state is None and not prepare_only:
            raise RuntimeError(
                "DeepSeek V4 target C128 weights were not prepared before graph capture"
            )
        if state is None:
            fp8 = torch.float8_e4m3fn
            e8m0 = torch.float8_e8m0fnu
            producer_weights = dsv4_producer.pack_weights(
                _raw_tensor(buffers["wq_a_weight"], (q_rank, hidden), fp8, 1),
                _raw_tensor(
                    buffers["wq_a_scale"],
                    (_ceil_div(q_rank, 128), _ceil_div(hidden, 128)),
                    e8m0,
                    1,
                ),
                _raw_tensor(buffers["wq_b_weight"], (query_width, q_rank), fp8, 1),
                _raw_tensor(
                    buffers["wq_b_scale"],
                    (_ceil_div(query_width, 128), _ceil_div(q_rank, 128)),
                    e8m0,
                    1,
                ),
                _raw_tensor(
                    buffers["wkv_weight"], (DS4_HEAD_DIM, hidden), fp8, 1
                ),
                _raw_tensor(
                    buffers["wkv_scale"],
                    (_ceil_div(DS4_HEAD_DIM, 128), _ceil_div(hidden, 128)),
                    e8m0,
                    1,
                ),
                _raw_tensor(
                    buffers["q_norm_weight"], (q_rank,), torch.bfloat16, 2
                ),
                _raw_tensor(
                    buffers["kv_norm_weight"],
                    (DS4_HEAD_DIM,),
                    torch.bfloat16,
                    2,
                ),
            )
            compressor_weights = dsv4_compressor.pack_weights(
                _raw_tensor(
                    buffers["compressor_main_wkv"],
                    (state_width, hidden),
                    torch.bfloat16,
                    2,
                ),
                _raw_tensor(
                    buffers["compressor_main_wgate"],
                    (state_width, hidden),
                    torch.bfloat16,
                    2,
                ),
                _raw_tensor(
                    buffers["compressor_main_ape"],
                    (128, state_width),
                    torch.float32,
                    4,
                ),
                _raw_tensor(
                    buffers["compressor_main_norm"],
                    (state_width,),
                    torch.bfloat16,
                    2,
                ),
            )
            output_weights = wo_projection.pack_weights(
                _raw_tensor(
                    buffers["wo_a_weight"],
                    (output_projected_width, output_geometry.group_width),
                    fp8,
                    1,
                ),
                _raw_tensor(
                    buffers["wo_a_scale"],
                    (
                        _ceil_div(output_projected_width, 128),
                        _ceil_div(output_geometry.group_width, 128),
                    ),
                    e8m0,
                    1,
                ),
                _raw_tensor(
                    buffers["wo_b_weight"],
                    (hidden, output_projected_width),
                    fp8,
                    1,
                ),
                _raw_tensor(
                    buffers["wo_b_scale"],
                    (
                        _ceil_div(hidden, 128),
                        _ceil_div(output_projected_width, 128),
                    ),
                    e8m0,
                    1,
                ),
                groups=output_geometry.groups,
                group_width=output_geometry.group_width,
                rank=output_geometry.rank,
                hidden=hidden,
            )
            state = (producer_weights, compressor_weights, output_weights)
            _C128_CAPTURE_STATE[state_key] = state
        producer_weights, compressor_weights, output_weights = state

        def tensor(name: str, shape: tuple[int, ...], dtype: Any, element_bytes: int):
            return _cached_target_capture_tensor(
                _raw_tensor,
                buffers[name],
                shape,
                dtype,
                element_bytes,
                name=f"target C128 {name}",
            )

        common = dict(
            arena=tensor("workspace", (contract.arena.total_bytes,), torch.uint8, 1),
            hidden_states=tensor("hidden_states", (rows, hidden), torch.bfloat16, 2),
            positions=tensor("positions", (rows,), torch.int32, 4),
            main_slots=tensor("main_slots", (rows,), torch.int32, 4),
            cos_sin_cache=tensor(
                "cos_sin_cache", (max_rows + 1, DS4_ROPE_DIM), torch.float32, 4
            ),
            main_kv_cache=tensor(
                "main_kv_cache", (source_pages, contract.main_page_bytes), torch.uint8, 1
            ),
            swa_indices=tensor("swa_indices", (rows, swa_width), torch.int32, 4),
            swa_lengths=tensor("swa_lengths", (rows,), torch.int32, 4),
            producer_weights=producer_weights,
            active_groups=tensor("active_groups", (1,), torch.int32, 4),
            group_rope_positions=tensor(
                "group_rope_positions", (group_capacity,), torch.int32, 4
            ),
            compressed_slots=tensor(
                "compressed_slots", (group_capacity,), torch.int32, 4
            ),
            active_sequences=tensor("active_sequences", (1,), torch.int32, 4),
            sequence_offsets=tensor("sequence_offsets", (2,), torch.int32, 4),
            state_sequence_ids=tensor("state_sequence_ids", (1,), torch.int32, 4),
            compressed_cos_sin_cache=tensor(
                "cos_sin_cache", (max_rows + 1, DS4_ROPE_DIM), torch.float32, 4
            ),
            compressed_main_cache=tensor(
                "compressed_main_cache",
                (source_pages, compressed_page_bytes),
                torch.uint8,
                1,
            ),
            main_kv_state=tensor(
                "main_kv_state", (1, state_rows, state_width), torch.float32, 4
            ),
            main_score_state=tensor(
                "main_score_state", (1, state_rows, state_width), torch.float32, 4
            ),
            compressor_weights=compressor_weights,
            indexed_indices=tensor(
                "indexed_indices", (1, indexed_width), torch.int32, 4
            ).expand(rows, -1),
            indexed_lengths=tensor("indexed_lengths", (rows,), torch.int32, 4),
            output_weights=output_weights,
            residual=tensor(
                "residual", (rows, hc_mult, hidden), torch.bfloat16, 2
            ),
            prev_post=tensor("prev_post", (rows, hc_mult), torch.float32, 4),
            prev_comb=tensor(
                "prev_comb", (rows, hc_mult, hc_mult), torch.float32, 4
            ),
            fn=tensor("hc_fn", (hc_mixes, hc_mult * hidden), torch.float32, 4),
            hc_scale=tensor("hc_scale", (3,), torch.float32, 4),
            hc_base=tensor("hc_base", (hc_mixes,), torch.float32, 4),
            norm_weight=tensor("norm_weight", (hidden,), torch.bfloat16, 2),
            residual_out=tensor(
                "residual_out", (rows, hc_mult, hidden), torch.bfloat16, 2
            ),
            y_out=tensor("normalized_output", (rows, hidden), torch.bfloat16, 2),
            post_out=tensor("post_out", (rows, hc_mult), torch.float32, 4),
            comb_out=tensor(
                "comb_out", (rows, hc_mult, hc_mult), torch.float32, 4
            ),
            attn_sink=tensor("attn_sink", (heads,), torch.float32, 4),
            producer_eps=float(kwargs.get("producer_eps", 1.0e-6)),
            compressor_eps=float(kwargs.get("compressor_eps", 1.0e-6)),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            sinkhorn_iters=int(kwargs.get("sinkhorn_iters", 20)),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
        )
        if lifecycle == "prefill":
            binding = bind_deepseek_v4_c128_prefill_attention_layer(
                contract,
                group_source_starts=tensor(
                    "group_source_starts", (group_capacity,), torch.int32, 4
                ),
                **common,
            )
            run = run_deepseek_v4_c128_prefill_attention_layer
        else:
            binding = bind_deepseek_v4_c128_continuation_attention_layer(
                contract,
                group_sequence_slots=tensor(
                    "group_sequence_slots", (group_capacity,), torch.int32, 4
                ),
                group_source_positions=tensor(
                    "group_source_starts", (group_capacity,), torch.int32, 4
                ),
                sequence_start_positions=tensor(
                    "sequence_start_positions", (1,), torch.int32, 4
                ),
                **common,
            )
            run = run_deepseek_v4_c128_continuation_attention_layer
        if not prepare_only and prepared_key not in _C128_CAPTURE_PREPARED:
            raise RuntimeError(
                f"DeepSeek V4 target C128 {lifecycle} was not prepared for its fixed graph"
            )
        run(binding)
        if prepare_only:
            _C128_CAPTURE_PREPARED.add(prepared_key)


def prepare_deepseek_v4_c4_prefill_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_c4_attention_layer_capture(
        ctx, lifecycle="prefill", prepare_only=True, **kwargs
    )


def capture_deepseek_v4_c4_prefill_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_c4_attention_layer_capture(
        ctx, lifecycle="prefill", prepare_only=False, **kwargs
    )


def prepare_deepseek_v4_c4_continuation_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_c4_attention_layer_capture(
        ctx, lifecycle="continuation", prepare_only=True, **kwargs
    )


def capture_deepseek_v4_c4_continuation_attention_layer(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_c4_attention_layer_capture(
        ctx, lifecycle="continuation", prepare_only=False, **kwargs
    )


def _run_deepseek_v4_c4_attention_layer_capture(
    ctx: dict[str, Any], *, lifecycle: str, prepare_only: bool, **kwargs: Any
) -> None:
    """Run one C4 half-layer against fixed scheduler-owned serving storage."""

    import torch
    from ds41rt_reference.b12x_spark_capture import _raw_tensor
    from b12x.attention import dsv4_compressor, dsv4_producer
    from b12x.gemm import wo_projection

    if lifecycle not in {"prefill", "continuation"}:
        raise ValueError(f"unsupported DeepSeek V4 C4 lifecycle {lifecycle!r}")
    variant = str(kwargs["variant"]).strip().lower()
    rows = int(kwargs["rows"])
    max_rows = int(kwargs["max_rows"])
    source_pages = int(kwargs["source_pages"])
    max_page_table_width = int(
        kwargs.get("max_page_table_width", source_pages)
    )
    max_positions = int(kwargs["max_positions"])
    swa_width = int(kwargs.get("swa_width", DS4_SWA_TOKENS))
    cache_format = str(kwargs.get("cache_format", "fp8"))
    if not 1 <= rows <= max_rows:
        raise ValueError(
            f"DeepSeek V4 C4 capture rows must be in [1, {max_rows}], got {rows}"
        )
    contract = plan_deepseek_v4_attention_layer(
        variant=variant,
        mode="extend",
        compression=4,
        max_rows=max_rows,
        source_pages=source_pages,
        max_page_table_width=max_page_table_width,
        max_positions=max_positions,
        swa_width=swa_width,
        cache_format=cache_format,
    )
    geometry = deepseek_v4_producer_geometry(variant)
    output_geometry = plan_deepseek_v4_attention_output(
        variant=variant, max_rows=max_rows
    ).geometry
    hidden = geometry.hidden
    heads = geometry.heads
    q_rank = geometry.q_lora_rank
    query_width = geometry.query_width
    output_projected_width = output_geometry.groups * output_geometry.rank
    hc_mult = 4
    hc_mixes = (2 + hc_mult) * hc_mult
    group_capacity = (
        max_rows // 4 if lifecycle == "prefill" else _ceil_div(max_rows, 4) + 1
    )
    compressed_page_bytes = contract.compressed_page_bytes
    index_page_bytes = (DS4_SOURCE_PAGE_TOKENS // 4) * (DS4_INDEX_HEAD_DIM + 4)
    # Eight C4 history rows plus an equally sized speculative guard.
    main_state_rows = 16
    main_state_width = 1_024
    index_state_width = 256
    selector_bytes = deepseek_v4_c4_selector_scratch_nbytes(
        variant=variant,
        mode="prefill",
        max_rows=max_rows,
        source_pages=source_pages,
        max_page_table_width=max_page_table_width,
    )
    buffers = ctx["buffers"]
    required = {
        "workspace": contract.arena.total_bytes,
        "selector_scratch": selector_bytes,
        "hidden_states": rows * hidden * 2,
        "normalized_output": rows * hidden * 2,
        "positions": rows * 4,
        "main_slots": rows * 4,
        "cos_sin_cache": (max_rows + 1) * DS4_ROPE_DIM * 4,
        "main_kv_cache": source_pages * contract.main_page_bytes,
        "swa_indices": rows * swa_width * 4,
        "swa_lengths": rows * 4,
        "active_groups": 4,
        "group_source_starts": group_capacity * 4,
        "group_sequence_slots": group_capacity * 4,
        "group_rope_positions": group_capacity * 4,
        "compressed_slots": group_capacity * 4,
        "active_sequences": 4,
        "sequence_offsets": 2 * 4,
        "sequence_start_positions": 4,
        "state_sequence_ids": 4,
        "compressed_main_cache": source_pages * compressed_page_bytes,
        "main_kv_state": main_state_rows * main_state_width * 4,
        "main_score_state": main_state_rows * main_state_width * 4,
        "index_cache": source_pages * index_page_bytes,
        "index_kv_state": main_state_rows * index_state_width * 4,
        "index_score_state": main_state_rows * index_state_width * 4,
        "real_page_table": max_page_table_width * 4,
        "index_cache_seqlens": rows * 4,
        "residual": rows * hc_mult * hidden * 2,
        "prev_post": rows * hc_mult * 4,
        "prev_comb": rows * hc_mult * hc_mult * 4,
        "residual_out": rows * hc_mult * hidden * 2,
        "post_out": rows * hc_mult * 4,
        "comb_out": rows * hc_mult * hc_mult * 4,
        "wq_a_weight": q_rank * hidden,
        "wq_a_scale": _ceil_div(q_rank, 128) * _ceil_div(hidden, 128),
        "wq_b_weight": query_width * q_rank,
        "wq_b_scale": _ceil_div(query_width, 128) * _ceil_div(q_rank, 128),
        "wkv_weight": DS4_HEAD_DIM * hidden,
        "wkv_scale": _ceil_div(DS4_HEAD_DIM, 128) * _ceil_div(hidden, 128),
        "q_norm_weight": q_rank * 2,
        "kv_norm_weight": DS4_HEAD_DIM * 2,
        "compressor_main_wkv": 1_024 * hidden * 2,
        "compressor_main_wgate": 1_024 * hidden * 2,
        "compressor_main_ape": 4 * 1_024 * 4,
        "compressor_main_norm": 512 * 2,
        "compressor_index_wkv": 256 * hidden * 2,
        "compressor_index_wgate": 256 * hidden * 2,
        "compressor_index_ape": 4 * 256 * 4,
        "compressor_index_norm": 128 * 2,
        "indexer_wq_weight": DS4_INDEX_HEADS * DS4_INDEX_HEAD_DIM * q_rank,
        "indexer_wq_scale": _ceil_div(
            DS4_INDEX_HEADS * DS4_INDEX_HEAD_DIM, 128
        )
        * _ceil_div(q_rank, 128),
        "indexer_weights_projection": DS4_INDEX_HEADS * hidden * 2,
        "wo_a_weight": output_projected_width * output_geometry.group_width,
        "wo_a_scale": _ceil_div(output_projected_width, 128)
        * _ceil_div(output_geometry.group_width, 128),
        "wo_b_weight": hidden * output_projected_width,
        "wo_b_scale": _ceil_div(hidden, 128)
        * _ceil_div(output_projected_width, 128),
        "attn_sink": heads * 4,
        "hc_fn": hc_mixes * hc_mult * hidden * 4,
        "hc_scale": 3 * 4,
        "hc_base": hc_mixes * 4,
        "norm_weight": hidden * 2,
    }
    device_id = _validate_capture_buffers(
        buffers, required, anchor="hidden_states"
    )
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    weight_names = (
        "wq_a_weight",
        "wq_a_scale",
        "wq_b_weight",
        "wq_b_scale",
        "wkv_weight",
        "wkv_scale",
        "q_norm_weight",
        "kv_norm_weight",
        "compressor_main_wkv",
        "compressor_main_wgate",
        "compressor_main_ape",
        "compressor_main_norm",
        "compressor_index_wkv",
        "compressor_index_wgate",
        "compressor_index_ape",
        "compressor_index_norm",
        "indexer_wq_weight",
        "indexer_wq_scale",
        "indexer_weights_projection",
        "wo_a_weight",
        "wo_a_scale",
        "wo_b_weight",
        "wo_b_scale",
    )
    state_key = (
        device_id,
        *map(ord, variant),
        *map(ord, contract.cache_format),
        *(int(buffers[name]["ptr"]) for name in weight_names),
    )
    # Kernel shape warmup is device-specific, but not stream- or
    # weight-pointer-specific. Every layer still gets its own packed-weight
    # state and every stream gets its own captured graph.
    prepared_key = (
        device_id,
        *map(ord, lifecycle),
        *map(ord, contract.cache_format),
        rows,
        max_rows,
        source_pages,
        max_page_table_width,
        max_positions,
        swa_width,
    )
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        state = _C4_CAPTURE_STATE.get(state_key)
        if state is None and not prepare_only:
            raise RuntimeError(
                "DeepSeek V4 target C4 weights were not prepared before graph capture"
            )
        if state is None:
            fp8 = torch.float8_e4m3fn
            e8m0 = torch.float8_e8m0fnu
            producer_weights = dsv4_producer.pack_weights(
                _raw_tensor(buffers["wq_a_weight"], (q_rank, hidden), fp8, 1),
                _raw_tensor(
                    buffers["wq_a_scale"],
                    (_ceil_div(q_rank, 128), _ceil_div(hidden, 128)),
                    e8m0,
                    1,
                ),
                _raw_tensor(buffers["wq_b_weight"], (query_width, q_rank), fp8, 1),
                _raw_tensor(
                    buffers["wq_b_scale"],
                    (_ceil_div(query_width, 128), _ceil_div(q_rank, 128)),
                    e8m0,
                    1,
                ),
                _raw_tensor(
                    buffers["wkv_weight"], (DS4_HEAD_DIM, hidden), fp8, 1
                ),
                _raw_tensor(
                    buffers["wkv_scale"],
                    (_ceil_div(DS4_HEAD_DIM, 128), _ceil_div(hidden, 128)),
                    e8m0,
                    1,
                ),
                _raw_tensor(
                    buffers["q_norm_weight"], (q_rank,), torch.bfloat16, 2
                ),
                _raw_tensor(
                    buffers["kv_norm_weight"],
                    (DS4_HEAD_DIM,),
                    torch.bfloat16,
                    2,
                ),
            )
            compressor_weights = dsv4_compressor.pack_weights(
                _raw_tensor(
                    buffers["compressor_main_wkv"],
                    (1_024, hidden),
                    torch.bfloat16,
                    2,
                ),
                _raw_tensor(
                    buffers["compressor_main_wgate"],
                    (1_024, hidden),
                    torch.bfloat16,
                    2,
                ),
                _raw_tensor(
                    buffers["compressor_main_ape"],
                    (4, 1_024),
                    torch.float32,
                    4,
                ),
                _raw_tensor(
                    buffers["compressor_main_norm"],
                    (512,),
                    torch.bfloat16,
                    2,
                ),
                index_wkv=_raw_tensor(
                    buffers["compressor_index_wkv"],
                    (256, hidden),
                    torch.bfloat16,
                    2,
                ),
                index_wgate=_raw_tensor(
                    buffers["compressor_index_wgate"],
                    (256, hidden),
                    torch.bfloat16,
                    2,
                ),
                index_ape=_raw_tensor(
                    buffers["compressor_index_ape"],
                    (4, 256),
                    torch.float32,
                    4,
                ),
                index_norm=_raw_tensor(
                    buffers["compressor_index_norm"],
                    (128,),
                    torch.bfloat16,
                    2,
                ),
            )
            indexer_weights = dsv4_producer.pack_indexer_weights(
                _raw_tensor(
                    buffers["indexer_wq_weight"],
                    (DS4_INDEX_HEADS * DS4_INDEX_HEAD_DIM, q_rank),
                    fp8,
                    1,
                ),
                _raw_tensor(
                    buffers["indexer_wq_scale"],
                    (
                        _ceil_div(DS4_INDEX_HEADS * DS4_INDEX_HEAD_DIM, 128),
                        _ceil_div(q_rank, 128),
                    ),
                    e8m0,
                    1,
                ),
                _raw_tensor(
                    buffers["indexer_weights_projection"],
                    (DS4_INDEX_HEADS, hidden),
                    torch.bfloat16,
                    2,
                ),
            )
            output_weights = wo_projection.pack_weights(
                _raw_tensor(
                    buffers["wo_a_weight"],
                    (output_projected_width, output_geometry.group_width),
                    fp8,
                    1,
                ),
                _raw_tensor(
                    buffers["wo_a_scale"],
                    (
                        _ceil_div(output_projected_width, 128),
                        _ceil_div(output_geometry.group_width, 128),
                    ),
                    e8m0,
                    1,
                ),
                _raw_tensor(
                    buffers["wo_b_weight"],
                    (hidden, output_projected_width),
                    fp8,
                    1,
                ),
                _raw_tensor(
                    buffers["wo_b_scale"],
                    (
                        _ceil_div(hidden, 128),
                        _ceil_div(output_projected_width, 128),
                    ),
                    e8m0,
                    1,
                ),
                groups=output_geometry.groups,
                group_width=output_geometry.group_width,
                rank=output_geometry.rank,
                hidden=hidden,
            )
            state = (
                producer_weights,
                compressor_weights,
                indexer_weights,
                output_weights,
            )
            _C4_CAPTURE_STATE[state_key] = state
        producer_weights, compressor_weights, indexer_weights, output_weights = state

        def tensor(name: str, shape: tuple[int, ...], dtype: Any, element_bytes: int):
            return _cached_target_capture_tensor(
                _raw_tensor,
                buffers[name],
                shape,
                dtype,
                element_bytes,
                name=f"target C4 {name}",
            )

        common = dict(
            arena=tensor("workspace", (contract.arena.total_bytes,), torch.uint8, 1),
            selector_scratch=tensor(
                "selector_scratch", (selector_bytes,), torch.uint8, 1
            ),
            hidden_states=tensor("hidden_states", (rows, hidden), torch.bfloat16, 2),
            positions=tensor("positions", (rows,), torch.int32, 4),
            main_slots=tensor("main_slots", (rows,), torch.int32, 4),
            cos_sin_cache=tensor(
                "cos_sin_cache", (max_rows + 1, DS4_ROPE_DIM), torch.float32, 4
            ),
            main_kv_cache=tensor(
                "main_kv_cache", (source_pages, contract.main_page_bytes), torch.uint8, 1
            ),
            swa_indices=tensor("swa_indices", (rows, swa_width), torch.int32, 4),
            swa_lengths=tensor("swa_lengths", (rows,), torch.int32, 4),
            producer_weights=producer_weights,
            active_groups=tensor("active_groups", (1,), torch.int32, 4),
            group_rope_positions=tensor(
                "group_rope_positions", (group_capacity,), torch.int32, 4
            ),
            compressed_slots=tensor(
                "compressed_slots", (group_capacity,), torch.int32, 4
            ),
            active_sequences=tensor("active_sequences", (1,), torch.int32, 4),
            sequence_offsets=tensor("sequence_offsets", (2,), torch.int32, 4),
            state_sequence_ids=tensor("state_sequence_ids", (1,), torch.int32, 4),
            compressed_cos_sin_cache=tensor(
                "cos_sin_cache", (max_rows + 1, DS4_ROPE_DIM), torch.float32, 4
            ),
            compressed_main_cache=tensor(
                "compressed_main_cache",
                (source_pages, compressed_page_bytes),
                torch.uint8,
                1,
            ),
            main_kv_state=tensor(
                "main_kv_state",
                (1, main_state_rows, main_state_width),
                torch.float32,
                4,
            ),
            main_score_state=tensor(
                "main_score_state",
                (1, main_state_rows, main_state_width),
                torch.float32,
                4,
            ),
            index_cache=tensor(
                "index_cache", (source_pages, index_page_bytes), torch.uint8, 1
            ),
            index_kv_state=tensor(
                "index_kv_state",
                (1, main_state_rows, index_state_width),
                torch.float32,
                4,
            ),
            index_score_state=tensor(
                "index_score_state",
                (1, main_state_rows, index_state_width),
                torch.float32,
                4,
            ),
            compressor_weights=compressor_weights,
            indexer_weights=indexer_weights,
            real_page_table=tensor(
                "real_page_table",
                (1, max_page_table_width),
                torch.int32,
                4,
            ).expand(rows, -1),
            index_cache_seqlens=tensor(
                "index_cache_seqlens", (rows,), torch.int32, 4
            ),
            output_weights=output_weights,
            residual=tensor(
                "residual", (rows, hc_mult, hidden), torch.bfloat16, 2
            ),
            prev_post=tensor("prev_post", (rows, hc_mult), torch.float32, 4),
            prev_comb=tensor(
                "prev_comb", (rows, hc_mult, hc_mult), torch.float32, 4
            ),
            fn=tensor(
                "hc_fn", (hc_mixes, hc_mult * hidden), torch.float32, 4
            ),
            hc_scale=tensor("hc_scale", (3,), torch.float32, 4),
            hc_base=tensor("hc_base", (hc_mixes,), torch.float32, 4),
            norm_weight=tensor("norm_weight", (hidden,), torch.bfloat16, 2),
            residual_out=tensor(
                "residual_out", (rows, hc_mult, hidden), torch.bfloat16, 2
            ),
            y_out=tensor(
                "normalized_output", (rows, hidden), torch.bfloat16, 2
            ),
            post_out=tensor("post_out", (rows, hc_mult), torch.float32, 4),
            comb_out=tensor(
                "comb_out", (rows, hc_mult, hc_mult), torch.float32, 4
            ),
            attn_sink=tensor("attn_sink", (heads,), torch.float32, 4),
            producer_eps=float(kwargs.get("producer_eps", 1.0e-6)),
            compressor_eps=float(kwargs.get("compressor_eps", 1.0e-6)),
            rms_eps=float(kwargs.get("rms_eps", 1.0e-6)),
            hc_eps=float(kwargs.get("hc_eps", 1.0e-6)),
            sinkhorn_iters=int(kwargs.get("sinkhorn_iters", 20)),
            norm_eps=float(kwargs.get("norm_eps", 1.0e-6)),
        )
        if lifecycle == "prefill":
            binding = bind_deepseek_v4_c4_prefill_attention_layer(
                contract,
                group_source_starts=tensor(
                    "group_source_starts", (group_capacity,), torch.int32, 4
                ),
                **common,
            )
            run = run_deepseek_v4_c4_prefill_attention_layer
        else:
            binding = bind_deepseek_v4_c4_continuation_attention_layer(
                contract,
                group_sequence_slots=tensor(
                    "group_sequence_slots", (group_capacity,), torch.int32, 4
                ),
                group_source_positions=tensor(
                    "group_source_starts", (group_capacity,), torch.int32, 4
                ),
                sequence_start_positions=tensor(
                    "sequence_start_positions", (1,), torch.int32, 4
                ),
                **common,
            )
            run = run_deepseek_v4_c4_continuation_attention_layer
        if not prepare_only and prepared_key not in _C4_CAPTURE_PREPARED:
            raise RuntimeError(
                f"DeepSeek V4 target C4 {lifecycle} was not prepared for its fixed graph"
            )
        run(binding)
        if prepare_only:
            _C4_CAPTURE_PREPARED.add(prepared_key)


def _validate_capture_buffers(
    buffers: dict[str, dict[str, Any]],
    required: dict[str, int],
    *,
    anchor: str,
) -> int:
    missing = sorted(set(required) - set(buffers))
    if missing:
        raise ValueError(
            "DeepSeek V4 target capture is missing buffers: " + ", ".join(missing)
        )
    device_id = int(buffers[anchor]["device_id"])
    for name, required_bytes in required.items():
        buffer = buffers[name]
        if int(buffer["device_id"]) != device_id:
            raise ValueError(
                "DeepSeek V4 target capture buffers must share one device; "
                f"{anchor} is cuda:{device_id}, {name} is cuda:{buffer['device_id']}"
            )
        if int(buffer["bytes"]) < required_bytes:
            raise ValueError(
                f"DeepSeek V4 target capture {name} needs {required_bytes} bytes, "
                f"got {buffer['bytes']}"
            )
    return device_id


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def bind_deepseek_v4_c128_decode_attention_layer(
    contract: DeepseekV4AttentionLayerContract,
    *,
    arena: Any,
    hidden_states: Any,
    positions: Any,
    main_slots: Any,
    cos_sin_cache: Any,
    main_kv_cache: Any,
    swa_indices: Any,
    swa_lengths: Any,
    producer_weights: Any,
    sequence_ids: Any,
    compressed_slots: Any,
    compressed_cos_sin_cache: Any,
    compressed_main_cache: Any,
    main_kv_state: Any,
    main_score_state: Any,
    compressor_weights: Any,
    indexed_indices: Any,
    indexed_lengths: Any,
    output_weights: Any,
    residual: Any,
    prev_post: Any,
    prev_comb: Any,
    fn: Any,
    hc_scale: Any,
    hc_base: Any,
    norm_weight: Any,
    residual_out: Any,
    y_out: Any,
    post_out: Any,
    comb_out: Any,
    attn_sink: Any | None = None,
    sm_scale: float | None = None,
    producer_eps: float = 1.0e-6,
    compressor_eps: float = 1.0e-6,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1.0e-6,
) -> DeepseekV4C128DecodeAttentionLayerBinding:
    """Bind the sequence-unique C128 decode chain into one lane arena."""

    import torch
    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if contract.compression != 128 or contract.mode != "decode":
        raise ValueError(
            "bound C128 decode attention layer requires mode=decode and "
            f"compression=128; got mode={contract.mode!r} "
            f"compression={contract.compression}"
        )
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("C128 decode hidden_states must be a rank-2 tensor")
    tokens = int(hidden_states.shape[0])
    arena_binding = bind_deepseek_v4_attention_layer(
        contract,
        arena=arena,
        tokens=tokens,
    )
    geometry = deepseek_v4_producer_geometry(contract.variant)
    expected_m = tokens

    producer_plan = _cached_target_capture_plan(
        dsv4_producer.plan,
        dsv4_producer.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=geometry.heads,
            head_dim=DS4_HEAD_DIM,
            nope_dim=DS4_NOPE_DIM,
            rope_dim=DS4_ROPE_DIM,
            page_size=DS4_SOURCE_PAGE_TOKENS,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    producer_binding = dsv4_producer.bind(
        producer_plan,
        scratch=arena_binding.producer_scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=main_slots,
        cos_sin_cache=cos_sin_cache,
        main_kv_cache=main_kv_cache,
        query=arena_binding.query,
        weights=producer_weights,
        eps=float(producer_eps),
        expected_m=expected_m,
    )

    compressor_plan = _cached_target_capture_plan(
        dsv4_compressor.plan,
        dsv4_compressor.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=contract.hidden,
            compress_ratio=128,
            with_indexer=False,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    compressor_binding = dsv4_compressor.bind_decode(
        compressor_plan,
        scratch=arena_binding.compressor_scratch,
        hidden_states=hidden_states,
        positions=positions,
        sequence_ids=sequence_ids,
        compressed_slots=compressed_slots,
        compressed_cos_sin_cache=compressed_cos_sin_cache,
        compressed_main_cache=compressed_main_cache,
        main_kv_state=main_kv_state,
        main_score_state=main_score_state,
        weights=compressor_weights,
        eps=float(compressor_eps),
        expected_m=tokens,
        rows_are_sequence_unique=True,
    )

    attention_contract = plan_deepseek_v4_compressed_mla(
        mode="decode",
        rows=contract.max_rows,
        heads=contract.heads,
        source_pages=contract.source_pages,
        max_page_table_width=contract.max_page_table_width,
        swa_width=contract.swa_width,
        compression=128,
        indexed_width=contract.indexed_width,
        cache_format=contract.cache_format,
    )
    attention_plan = _cached_target_capture_plan(
        compressed_mla.plan,
        compressed_mla.Caps(
            device=hidden_states.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=contract.heads,
            head_dim=DS4_HEAD_DIM,
            v_head_dim=DS4_HEAD_DIM,
            max_width=attention_contract.total_width,
            max_page_table_width=contract.max_page_table_width,
            max_q_rows=contract.max_rows,
            max_batch=contract.max_rows,
            max_kv_rows=0,
            max_chunks_per_row=attention_contract.max_chunks_per_row,
            page_size=DS4_SOURCE_PAGE_TOKENS,
        ),
    )
    attention_binding = compressed_mla.bind(
        attention_plan,
        scratch=arena_binding.attention_scratch,
        q=arena_binding.query,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
    )
    attention_binding.scratch.mode = "decode"
    attention_binding.scratch.fixed_capacity = True
    attention_binding.scratch.use_cuda_graph = True

    output_contract = plan_deepseek_v4_attention_output(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    output_geometry = output_contract.geometry
    output_plan = _cached_target_capture_plan(
        wo_projection.plan,
        wo_projection.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            groups=output_geometry.groups,
            group_width=output_geometry.group_width,
            rank=output_geometry.rank,
            hidden=output_geometry.hidden,
            dtype=torch.bfloat16,
        ),
    )
    output_binding = wo_projection.bind_inv_rope(
        output_plan,
        scratch=arena_binding.output_projection_scratch,
        o=arena_binding.attention_output,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=output_weights,
        heads_per_group=output_geometry.heads_per_group,
        nope_dim=DS4_NOPE_DIM,
        rope_dim=DS4_ROPE_DIM,
        expected_m=expected_m,
    )

    mhc_contract = plan_deepseek_v4_mhc(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    mhc_plan = _cached_target_capture_plan(
        mhc.plan,
        mhc.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden_size=contract.hidden,
            split_k=mhc_contract.geometry.split_k,
            dtype=torch.bfloat16,
        ),
    )
    mhc_binding = mhc.bind(
        mhc_plan,
        scratch=arena_binding.mhc_scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=y_out,
        post=post_out,
        comb=comb_out,
        out=residual_out,
    )
    return DeepseekV4C128DecodeAttentionLayerBinding(
        arena_binding=arena_binding,
        producer_binding=producer_binding,
        compressor_binding=compressor_binding,
        attention_binding=attention_binding,
        output_binding=output_binding,
        mhc_binding=mhc_binding,
        main_kv_cache=main_kv_cache,
        compressed_main_cache=compressed_main_cache,
        attn_sink=attn_sink,
        sm_scale=(DS4_HEAD_DIM**-0.5 if sm_scale is None else float(sm_scale)),
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_weight=norm_weight,
        norm_eps=float(norm_eps),
    )


def run_deepseek_v4_c128_decode_attention_layer(
    binding: DeepseekV4C128DecodeAttentionLayerBinding,
) -> tuple[Any, Any, Any, Any]:
    """Run C128 decode and return residual, post, comb, normalized y."""

    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if not isinstance(binding, DeepseekV4C128DecodeAttentionLayerBinding):
        raise TypeError("C128 decode attention layer requires its fixed binding")
    dsv4_producer.run(binding=binding.producer_binding)
    dsv4_compressor.run_decode(binding=binding.compressor_binding)
    compressed_mla.run(
        binding=binding.attention_binding,
        swa_k_cache=binding.main_kv_cache,
        sm_scale=binding.sm_scale,
        swa_page_size=DS4_SOURCE_PAGE_TOKENS,
        indexed_k_cache=binding.compressed_main_cache,
        indexed_page_size=2,
        attn_sink=binding.attn_sink,
        expected_num_q_heads=binding.arena_binding.contract.heads,
        backend="sm120",
        out=binding.arena_binding.attention_output,
    )
    delta = wo_projection.run_inv_rope(binding=binding.output_binding)
    return mhc.run_post_pre(
        delta,
        binding.residual,
        binding.prev_post,
        binding.prev_comb,
        binding.fn,
        binding.hc_scale,
        binding.hc_base,
        rms_eps=binding.rms_eps,
        hc_eps=binding.hc_eps,
        sinkhorn_iters=binding.sinkhorn_iters,
        norm_weight=binding.norm_weight,
        norm_eps=binding.norm_eps,
        binding=binding.mhc_binding,
    )


def bind_deepseek_v4_c128_prefill_attention_layer(
    contract: DeepseekV4AttentionLayerContract,
    *,
    arena: Any,
    hidden_states: Any,
    positions: Any,
    main_slots: Any,
    cos_sin_cache: Any,
    main_kv_cache: Any,
    swa_indices: Any,
    swa_lengths: Any,
    producer_weights: Any,
    active_groups: Any,
    group_source_starts: Any,
    group_rope_positions: Any,
    compressed_slots: Any,
    active_sequences: Any,
    sequence_offsets: Any,
    state_sequence_ids: Any,
    compressed_cos_sin_cache: Any,
    compressed_main_cache: Any,
    main_kv_state: Any,
    main_score_state: Any,
    compressor_weights: Any,
    indexed_indices: Any,
    indexed_lengths: Any,
    output_weights: Any,
    residual: Any,
    prev_post: Any,
    prev_comb: Any,
    fn: Any,
    hc_scale: Any,
    hc_base: Any,
    norm_weight: Any,
    residual_out: Any,
    y_out: Any,
    post_out: Any,
    comb_out: Any,
    attn_sink: Any | None = None,
    sm_scale: float | None = None,
    producer_eps: float = 1.0e-6,
    compressor_eps: float = 1.0e-6,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1.0e-6,
) -> DeepseekV4C128PrefillAttentionLayerBinding:
    """Bind one initial C128 prefill chunk into the fixed lane arena.

    Completed compressed slots and their per-query causal lengths stay under
    scheduler ownership.  This function only composes the already-qualified
    producer, initial-prefill compressor, MLA, grouped WO, and mHC leaves.
    """

    import torch
    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if contract.compression != 128 or contract.mode != "extend":
        raise ValueError(
            "bound C128 initial-prefill attention layer requires mode=extend and "
            f"compression=128; got mode={contract.mode!r} "
            f"compression={contract.compression}"
        )
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("C128 prefill hidden_states must be a rank-2 tensor")
    tokens = int(hidden_states.shape[0])
    arena_binding = bind_deepseek_v4_attention_layer(
        contract,
        arena=arena,
        tokens=tokens,
    )
    geometry = deepseek_v4_producer_geometry(contract.variant)
    expected_m = tokens

    producer_plan = _cached_target_capture_plan(
        dsv4_producer.plan,
        dsv4_producer.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=geometry.heads,
            head_dim=DS4_HEAD_DIM,
            nope_dim=DS4_NOPE_DIM,
            rope_dim=DS4_ROPE_DIM,
            page_size=DS4_SOURCE_PAGE_TOKENS,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    producer_binding = dsv4_producer.bind(
        producer_plan,
        scratch=arena_binding.producer_scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=main_slots,
        cos_sin_cache=cos_sin_cache,
        main_kv_cache=main_kv_cache,
        query=arena_binding.query,
        weights=producer_weights,
        eps=float(producer_eps),
        expected_m=expected_m,
    )

    compressor_plan = _cached_target_capture_plan(
        dsv4_compressor.plan,
        dsv4_compressor.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=contract.hidden,
            compress_ratio=128,
            with_indexer=False,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    compressor_binding = dsv4_compressor.bind_prefill(
        compressor_plan,
        scratch=arena_binding.compressor_scratch,
        hidden_states=hidden_states,
        active_groups=active_groups,
        group_source_starts=group_source_starts,
        group_rope_positions=group_rope_positions,
        compressed_slots=compressed_slots,
        active_sequences=active_sequences,
        sequence_offsets=sequence_offsets,
        state_sequence_ids=state_sequence_ids,
        compressed_cos_sin_cache=compressed_cos_sin_cache,
        compressed_main_cache=compressed_main_cache,
        main_kv_state=main_kv_state,
        main_score_state=main_score_state,
        weights=compressor_weights,
        eps=float(compressor_eps),
        expected_m=expected_m,
        initial_prefill=True,
    )

    attention_contract = plan_deepseek_v4_compressed_mla(
        mode="extend",
        rows=contract.max_rows,
        heads=contract.heads,
        source_pages=contract.source_pages,
        max_page_table_width=contract.max_page_table_width,
        swa_width=contract.swa_width,
        compression=128,
        indexed_width=contract.indexed_width,
        cache_format=contract.cache_format,
    )
    attention_plan = _cached_target_capture_plan(
        compressed_mla.plan,
        compressed_mla.Caps(
            device=hidden_states.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=contract.heads,
            head_dim=DS4_HEAD_DIM,
            v_head_dim=DS4_HEAD_DIM,
            max_width=attention_contract.total_width,
            max_page_table_width=contract.max_page_table_width,
            max_q_rows=contract.max_rows,
            max_batch=contract.max_rows,
            max_kv_rows=0,
            max_chunks_per_row=attention_contract.max_chunks_per_row,
            page_size=DS4_SOURCE_PAGE_TOKENS,
        ),
    )
    attention_binding = compressed_mla.bind(
        attention_plan,
        scratch=arena_binding.attention_scratch,
        q=arena_binding.query,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
    )
    attention_binding.scratch.mode = "extend"
    attention_binding.scratch.fixed_capacity = True
    attention_binding.scratch.use_cuda_graph = True

    output_contract = plan_deepseek_v4_attention_output(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    output_geometry = output_contract.geometry
    output_plan = _cached_target_capture_plan(
        wo_projection.plan,
        wo_projection.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            groups=output_geometry.groups,
            group_width=output_geometry.group_width,
            rank=output_geometry.rank,
            hidden=output_geometry.hidden,
            dtype=torch.bfloat16,
        ),
    )
    output_binding = wo_projection.bind_inv_rope(
        output_plan,
        scratch=arena_binding.output_projection_scratch,
        o=arena_binding.attention_output,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=output_weights,
        heads_per_group=output_geometry.heads_per_group,
        nope_dim=DS4_NOPE_DIM,
        rope_dim=DS4_ROPE_DIM,
        expected_m=expected_m,
    )

    mhc_contract = plan_deepseek_v4_mhc(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    mhc_plan = _cached_target_capture_plan(
        mhc.plan,
        mhc.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden_size=contract.hidden,
            split_k=mhc_contract.geometry.split_k,
            dtype=torch.bfloat16,
        ),
    )
    mhc_binding = mhc.bind(
        mhc_plan,
        scratch=arena_binding.mhc_scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=y_out,
        post=post_out,
        comb=comb_out,
        out=residual_out,
    )
    return DeepseekV4C128PrefillAttentionLayerBinding(
        arena_binding=arena_binding,
        producer_binding=producer_binding,
        compressor_binding=compressor_binding,
        attention_binding=attention_binding,
        output_binding=output_binding,
        mhc_binding=mhc_binding,
        main_kv_cache=main_kv_cache,
        compressed_main_cache=compressed_main_cache,
        attn_sink=attn_sink,
        sm_scale=(DS4_HEAD_DIM**-0.5 if sm_scale is None else float(sm_scale)),
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_weight=norm_weight,
        norm_eps=float(norm_eps),
    )


def run_deepseek_v4_c128_prefill_attention_layer(
    binding: DeepseekV4C128PrefillAttentionLayerBinding,
) -> tuple[Any, Any, Any, Any]:
    """Run initial C128 prefill and return residual, post, comb, normalized y."""

    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if not isinstance(binding, DeepseekV4C128PrefillAttentionLayerBinding):
        raise TypeError("C128 prefill attention layer requires its fixed binding")
    dsv4_producer.run(binding=binding.producer_binding)
    dsv4_compressor.run_prefill(binding=binding.compressor_binding)
    compressed_mla.run(
        binding=binding.attention_binding,
        swa_k_cache=binding.main_kv_cache,
        sm_scale=binding.sm_scale,
        swa_page_size=DS4_SOURCE_PAGE_TOKENS,
        indexed_k_cache=binding.compressed_main_cache,
        indexed_page_size=2,
        attn_sink=binding.attn_sink,
        expected_num_q_heads=binding.arena_binding.contract.heads,
        backend="sm120",
        out=binding.arena_binding.attention_output,
    )
    delta = wo_projection.run_inv_rope(binding=binding.output_binding)
    return mhc.run_post_pre(
        delta,
        binding.residual,
        binding.prev_post,
        binding.prev_comb,
        binding.fn,
        binding.hc_scale,
        binding.hc_base,
        rms_eps=binding.rms_eps,
        hc_eps=binding.hc_eps,
        sinkhorn_iters=binding.sinkhorn_iters,
        norm_weight=binding.norm_weight,
        norm_eps=binding.norm_eps,
        binding=binding.mhc_binding,
    )


def bind_deepseek_v4_c128_continuation_attention_layer(
    contract: DeepseekV4AttentionLayerContract,
    *,
    arena: Any,
    hidden_states: Any,
    positions: Any,
    main_slots: Any,
    cos_sin_cache: Any,
    main_kv_cache: Any,
    swa_indices: Any,
    swa_lengths: Any,
    producer_weights: Any,
    active_groups: Any,
    group_sequence_slots: Any,
    group_source_positions: Any,
    group_rope_positions: Any,
    compressed_slots: Any,
    active_sequences: Any,
    sequence_offsets: Any,
    sequence_start_positions: Any,
    state_sequence_ids: Any,
    compressed_cos_sin_cache: Any,
    compressed_main_cache: Any,
    main_kv_state: Any,
    main_score_state: Any,
    compressor_weights: Any,
    indexed_indices: Any,
    indexed_lengths: Any,
    output_weights: Any,
    residual: Any,
    prev_post: Any,
    prev_comb: Any,
    fn: Any,
    hc_scale: Any,
    hc_base: Any,
    norm_weight: Any,
    residual_out: Any,
    y_out: Any,
    post_out: Any,
    comb_out: Any,
    attn_sink: Any | None = None,
    sm_scale: float | None = None,
    producer_eps: float = 1.0e-6,
    compressor_eps: float = 1.0e-6,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1.0e-6,
) -> DeepseekV4C128ContinuationAttentionLayerBinding:
    """Bind one ordered C128 continuation chunk into the fixed lane arena."""

    import torch
    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if contract.compression != 128 or contract.mode != "extend":
        raise ValueError(
            "bound C128 continuation attention layer requires mode=extend and "
            f"compression=128; got mode={contract.mode!r} "
            f"compression={contract.compression}"
        )
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("C128 continuation hidden_states must be a rank-2 tensor")
    tokens = int(hidden_states.shape[0])
    arena_binding = bind_deepseek_v4_attention_layer(
        contract,
        arena=arena,
        tokens=tokens,
    )
    geometry = deepseek_v4_producer_geometry(contract.variant)
    expected_m = tokens

    producer_plan = _cached_target_capture_plan(
        dsv4_producer.plan,
        dsv4_producer.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=geometry.heads,
            head_dim=DS4_HEAD_DIM,
            nope_dim=DS4_NOPE_DIM,
            rope_dim=DS4_ROPE_DIM,
            page_size=DS4_SOURCE_PAGE_TOKENS,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    producer_binding = dsv4_producer.bind(
        producer_plan,
        scratch=arena_binding.producer_scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=main_slots,
        cos_sin_cache=cos_sin_cache,
        main_kv_cache=main_kv_cache,
        query=arena_binding.query,
        weights=producer_weights,
        eps=float(producer_eps),
        expected_m=expected_m,
    )

    compressor_plan = _cached_target_capture_plan(
        dsv4_compressor.plan,
        dsv4_compressor.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=contract.hidden,
            compress_ratio=128,
            with_indexer=False,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    compressor_binding = dsv4_compressor.bind_continuation(
        compressor_plan,
        scratch=arena_binding.compressor_scratch,
        hidden_states=hidden_states,
        active_groups=active_groups,
        group_sequence_slots=group_sequence_slots,
        group_source_positions=group_source_positions,
        group_rope_positions=group_rope_positions,
        compressed_slots=compressed_slots,
        active_sequences=active_sequences,
        sequence_offsets=sequence_offsets,
        sequence_start_positions=sequence_start_positions,
        state_sequence_ids=state_sequence_ids,
        compressed_cos_sin_cache=compressed_cos_sin_cache,
        compressed_main_cache=compressed_main_cache,
        main_kv_state=main_kv_state,
        main_score_state=main_score_state,
        weights=compressor_weights,
        eps=float(compressor_eps),
        expected_m=expected_m,
        ordered_continuation=True,
    )

    attention_contract = plan_deepseek_v4_compressed_mla(
        mode="extend",
        rows=contract.max_rows,
        heads=contract.heads,
        source_pages=contract.source_pages,
        max_page_table_width=contract.max_page_table_width,
        swa_width=contract.swa_width,
        compression=128,
        indexed_width=contract.indexed_width,
        cache_format=contract.cache_format,
    )
    attention_plan = _cached_target_capture_plan(
        compressed_mla.plan,
        compressed_mla.Caps(
            device=hidden_states.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=contract.heads,
            head_dim=DS4_HEAD_DIM,
            v_head_dim=DS4_HEAD_DIM,
            max_width=attention_contract.total_width,
            max_page_table_width=contract.max_page_table_width,
            max_q_rows=contract.max_rows,
            max_batch=contract.max_rows,
            max_kv_rows=0,
            max_chunks_per_row=attention_contract.max_chunks_per_row,
            page_size=DS4_SOURCE_PAGE_TOKENS,
        ),
    )
    attention_binding = compressed_mla.bind(
        attention_plan,
        scratch=arena_binding.attention_scratch,
        q=arena_binding.query,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
    )
    attention_binding.scratch.mode = "extend"
    attention_binding.scratch.fixed_capacity = True
    attention_binding.scratch.use_cuda_graph = True

    output_contract = plan_deepseek_v4_attention_output(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    output_geometry = output_contract.geometry
    output_plan = _cached_target_capture_plan(
        wo_projection.plan,
        wo_projection.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            groups=output_geometry.groups,
            group_width=output_geometry.group_width,
            rank=output_geometry.rank,
            hidden=output_geometry.hidden,
            dtype=torch.bfloat16,
        ),
    )
    output_binding = wo_projection.bind_inv_rope(
        output_plan,
        scratch=arena_binding.output_projection_scratch,
        o=arena_binding.attention_output,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=output_weights,
        heads_per_group=output_geometry.heads_per_group,
        nope_dim=DS4_NOPE_DIM,
        rope_dim=DS4_ROPE_DIM,
        expected_m=expected_m,
    )

    mhc_contract = plan_deepseek_v4_mhc(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    mhc_plan = _cached_target_capture_plan(
        mhc.plan,
        mhc.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden_size=contract.hidden,
            split_k=mhc_contract.geometry.split_k,
            dtype=torch.bfloat16,
        ),
    )
    mhc_binding = mhc.bind(
        mhc_plan,
        scratch=arena_binding.mhc_scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=y_out,
        post=post_out,
        comb=comb_out,
        out=residual_out,
    )
    return DeepseekV4C128ContinuationAttentionLayerBinding(
        arena_binding=arena_binding,
        producer_binding=producer_binding,
        compressor_binding=compressor_binding,
        attention_binding=attention_binding,
        output_binding=output_binding,
        mhc_binding=mhc_binding,
        main_kv_cache=main_kv_cache,
        compressed_main_cache=compressed_main_cache,
        attn_sink=attn_sink,
        sm_scale=(DS4_HEAD_DIM**-0.5 if sm_scale is None else float(sm_scale)),
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_weight=norm_weight,
        norm_eps=float(norm_eps),
    )


def run_deepseek_v4_c128_continuation_attention_layer(
    binding: DeepseekV4C128ContinuationAttentionLayerBinding,
) -> tuple[Any, Any, Any, Any]:
    """Run ordered C128 continuation and return its next FFN input."""

    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if not isinstance(binding, DeepseekV4C128ContinuationAttentionLayerBinding):
        raise TypeError("C128 continuation attention layer requires its fixed binding")
    dsv4_producer.run(binding=binding.producer_binding)
    dsv4_compressor.run_continuation(binding=binding.compressor_binding)
    compressed_mla.run(
        binding=binding.attention_binding,
        swa_k_cache=binding.main_kv_cache,
        sm_scale=binding.sm_scale,
        swa_page_size=DS4_SOURCE_PAGE_TOKENS,
        indexed_k_cache=binding.compressed_main_cache,
        indexed_page_size=2,
        attn_sink=binding.attn_sink,
        expected_num_q_heads=binding.arena_binding.contract.heads,
        backend="sm120",
        out=binding.arena_binding.attention_output,
    )
    delta = wo_projection.run_inv_rope(binding=binding.output_binding)
    return mhc.run_post_pre(
        delta,
        binding.residual,
        binding.prev_post,
        binding.prev_comb,
        binding.fn,
        binding.hc_scale,
        binding.hc_base,
        rms_eps=binding.rms_eps,
        hc_eps=binding.hc_eps,
        sinkhorn_iters=binding.sinkhorn_iters,
        norm_weight=binding.norm_weight,
        norm_eps=binding.norm_eps,
        binding=binding.mhc_binding,
    )


def bind_deepseek_v4_c4_prefill_attention_layer(
    contract: DeepseekV4AttentionLayerContract,
    *,
    arena: Any,
    selector_scratch: Any,
    hidden_states: Any,
    positions: Any,
    main_slots: Any,
    cos_sin_cache: Any,
    main_kv_cache: Any,
    swa_indices: Any,
    swa_lengths: Any,
    producer_weights: Any,
    active_groups: Any,
    group_source_starts: Any,
    group_rope_positions: Any,
    compressed_slots: Any,
    active_sequences: Any,
    sequence_offsets: Any,
    state_sequence_ids: Any,
    compressed_cos_sin_cache: Any,
    compressed_main_cache: Any,
    main_kv_state: Any,
    main_score_state: Any,
    index_cache: Any,
    index_kv_state: Any,
    index_score_state: Any,
    compressor_weights: Any,
    indexer_weights: Any,
    real_page_table: Any,
    index_cache_seqlens: Any,
    output_weights: Any,
    residual: Any,
    prev_post: Any,
    prev_comb: Any,
    fn: Any,
    hc_scale: Any,
    hc_base: Any,
    norm_weight: Any,
    residual_out: Any,
    y_out: Any,
    post_out: Any,
    comb_out: Any,
    attn_sink: Any | None = None,
    sm_scale: float | None = None,
    producer_eps: float = 1.0e-6,
    compressor_eps: float = 1.0e-6,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1.0e-6,
) -> DeepseekV4C4PrefillAttentionLayerBinding:
    """Bind the initial C4 producer/compressor/learned-selector chain."""

    import torch
    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
        dsa_indexer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if contract.compression != 4 or contract.mode != "extend":
        raise ValueError(
            "bound C4 initial-prefill attention layer requires mode=extend and "
            f"compression=4; got mode={contract.mode!r} "
            f"compression={contract.compression}"
        )
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("C4 prefill hidden_states must be a rank-2 tensor")
    tokens = int(hidden_states.shape[0])
    arena_binding = bind_deepseek_v4_attention_layer(
        contract,
        arena=arena,
        tokens=tokens,
    )
    geometry = deepseek_v4_producer_geometry(contract.variant)
    expected_m = tokens

    producer_plan = _cached_target_capture_plan(
        dsv4_producer.plan,
        dsv4_producer.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=geometry.heads,
            head_dim=DS4_HEAD_DIM,
            nope_dim=DS4_NOPE_DIM,
            rope_dim=DS4_ROPE_DIM,
            page_size=DS4_SOURCE_PAGE_TOKENS,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    producer_binding = dsv4_producer.bind(
        producer_plan,
        scratch=arena_binding.producer_scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=main_slots,
        cos_sin_cache=cos_sin_cache,
        main_kv_cache=main_kv_cache,
        query=arena_binding.query,
        weights=producer_weights,
        eps=float(producer_eps),
        expected_m=expected_m,
    )

    compressor_plan = _cached_target_capture_plan(
        dsv4_compressor.plan,
        dsv4_compressor.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=contract.hidden,
            compress_ratio=4,
            with_indexer=True,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    compressor_binding = dsv4_compressor.bind_prefill(
        compressor_plan,
        scratch=arena_binding.compressor_scratch,
        hidden_states=hidden_states,
        active_groups=active_groups,
        group_source_starts=group_source_starts,
        group_rope_positions=group_rope_positions,
        compressed_slots=compressed_slots,
        active_sequences=active_sequences,
        sequence_offsets=sequence_offsets,
        state_sequence_ids=state_sequence_ids,
        compressed_cos_sin_cache=compressed_cos_sin_cache,
        compressed_main_cache=compressed_main_cache,
        main_kv_state=main_kv_state,
        main_score_state=main_score_state,
        index_cache=index_cache,
        index_kv_state=index_kv_state,
        index_score_state=index_score_state,
        weights=compressor_weights,
        eps=float(compressor_eps),
        expected_m=expected_m,
        initial_prefill=True,
    )

    indexer_plan = dsv4_producer.plan_indexer(
        dsv4_producer.IndexerCaps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=DS4_INDEX_HEADS,
            head_dim=DS4_INDEX_HEAD_DIM,
            rope_dim=DS4_ROPE_DIM,
            dtype=torch.bfloat16,
        )
    )
    indexer_producer_binding = dsv4_producer.bind_indexer(
        indexer_plan,
        scratch=arena_binding.indexer_producer_scratch,
        q_rank=producer_binding.q_rank,
        hidden_states=hidden_states,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        query=arena_binding.index_query,
        head_weights=arena_binding.index_head_weights,
        weights=indexer_weights,
        expected_m=expected_m,
    )

    selector_plan = _cached_target_capture_plan(
        dsa_indexer.plan,
        dsa_indexer.Caps(
            device=hidden_states.device,
            source_layout=dsa_indexer.SOURCE_LAYOUT_PAGED,
            num_q_heads=DS4_INDEX_HEADS,
            max_q_rows=contract.max_rows,
            max_page_table_width=contract.max_page_table_width,
            topk=contract.indexed_width,
            mode="prefill",
            page_size=64,
            shared_page_table=True,
        ),
    )
    selector_binding = selector_plan.bind(
        scratch=selector_scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=index_cache_seqlens,
        expected_num_q_heads=DS4_INDEX_HEADS,
        shared_page_table=True,
        output_physical_slots=True,
    )

    attention_contract = plan_deepseek_v4_compressed_mla(
        mode="extend",
        rows=contract.max_rows,
        heads=contract.heads,
        source_pages=contract.source_pages,
        max_page_table_width=contract.max_page_table_width,
        swa_width=contract.swa_width,
        compression=4,
        indexed_width=contract.indexed_width,
        cache_format=contract.cache_format,
    )
    attention_plan = _cached_target_capture_plan(
        compressed_mla.plan,
        compressed_mla.Caps(
            device=hidden_states.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=contract.heads,
            head_dim=DS4_HEAD_DIM,
            v_head_dim=DS4_HEAD_DIM,
            max_width=attention_contract.total_width,
            max_page_table_width=contract.max_page_table_width,
            max_q_rows=contract.max_rows,
            max_batch=contract.max_rows,
            max_kv_rows=0,
            max_chunks_per_row=attention_contract.max_chunks_per_row,
            page_size=DS4_SOURCE_PAGE_TOKENS,
        ),
    )
    attention_binding = compressed_mla.bind(
        attention_plan,
        scratch=arena_binding.attention_scratch,
        q=arena_binding.query,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=arena_binding.selected_indices,
        # The sparse-MLA kernel already clamps replay-time lengths to the
        # fixed selected-index width. Consume the selector's source lengths
        # directly instead of launching a separate clamp/copy kernel into a
        # second arena vector on every C4 layer.
        indexed_lengths=index_cache_seqlens,
    )
    attention_binding.scratch.mode = "extend"
    attention_binding.scratch.fixed_capacity = True
    attention_binding.scratch.use_cuda_graph = True

    output_contract = plan_deepseek_v4_attention_output(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    output_geometry = output_contract.geometry
    output_plan = _cached_target_capture_plan(
        wo_projection.plan,
        wo_projection.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            groups=output_geometry.groups,
            group_width=output_geometry.group_width,
            rank=output_geometry.rank,
            hidden=output_geometry.hidden,
            dtype=torch.bfloat16,
        ),
    )
    output_binding = wo_projection.bind_inv_rope(
        output_plan,
        scratch=arena_binding.output_projection_scratch,
        o=arena_binding.attention_output,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=output_weights,
        heads_per_group=output_geometry.heads_per_group,
        nope_dim=DS4_NOPE_DIM,
        rope_dim=DS4_ROPE_DIM,
        expected_m=expected_m,
    )

    mhc_contract = plan_deepseek_v4_mhc(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    mhc_plan = _cached_target_capture_plan(
        mhc.plan,
        mhc.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden_size=contract.hidden,
            split_k=mhc_contract.geometry.split_k,
            dtype=torch.bfloat16,
        ),
    )
    mhc_binding = mhc.bind(
        mhc_plan,
        scratch=arena_binding.mhc_scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=y_out,
        post=post_out,
        comb=comb_out,
        out=residual_out,
    )
    return DeepseekV4C4PrefillAttentionLayerBinding(
        arena_binding=arena_binding,
        producer_binding=producer_binding,
        compressor_binding=compressor_binding,
        indexer_producer_binding=indexer_producer_binding,
        selector_binding=selector_binding,
        attention_binding=attention_binding,
        output_binding=output_binding,
        mhc_binding=mhc_binding,
        main_kv_cache=main_kv_cache,
        compressed_main_cache=compressed_main_cache,
        index_cache=index_cache,
        attn_sink=attn_sink,
        sm_scale=(DS4_HEAD_DIM**-0.5 if sm_scale is None else float(sm_scale)),
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_weight=norm_weight,
        norm_eps=float(norm_eps),
    )


def run_deepseek_v4_c4_prefill_attention_layer(
    binding: DeepseekV4C4PrefillAttentionLayerBinding,
) -> tuple[Any, Any, Any, Any]:
    """Run initial C4 prefill and return residual, post, comb, normalized y."""

    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
        dsa_indexer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if not isinstance(binding, DeepseekV4C4PrefillAttentionLayerBinding):
        raise TypeError("C4 prefill attention layer requires its fixed binding")
    dsv4_producer.run(binding=binding.producer_binding)
    dsv4_compressor.run_prefill(binding=binding.compressor_binding)
    dsv4_producer.run_indexer(binding=binding.indexer_producer_binding)
    dsa_indexer.index_topk_fp8(
        q_fp8=binding.arena_binding.index_query,
        weights=binding.arena_binding.index_head_weights,
        index_k_cache=binding.index_cache,
        binding=binding.selector_binding,
        out_indices=binding.arena_binding.selected_indices,
    )
    compressed_mla.run(
        binding=binding.attention_binding,
        swa_k_cache=binding.main_kv_cache,
        sm_scale=binding.sm_scale,
        swa_page_size=DS4_SOURCE_PAGE_TOKENS,
        indexed_k_cache=binding.compressed_main_cache,
        indexed_page_size=64,
        attn_sink=binding.attn_sink,
        expected_num_q_heads=binding.arena_binding.contract.heads,
        backend="sm120",
        out=binding.arena_binding.attention_output,
    )
    delta = wo_projection.run_inv_rope(binding=binding.output_binding)
    return mhc.run_post_pre(
        delta,
        binding.residual,
        binding.prev_post,
        binding.prev_comb,
        binding.fn,
        binding.hc_scale,
        binding.hc_base,
        rms_eps=binding.rms_eps,
        hc_eps=binding.hc_eps,
        sinkhorn_iters=binding.sinkhorn_iters,
        norm_weight=binding.norm_weight,
        norm_eps=binding.norm_eps,
        binding=binding.mhc_binding,
    )


def bind_deepseek_v4_c4_continuation_attention_layer(
    contract: DeepseekV4AttentionLayerContract,
    *,
    arena: Any,
    selector_scratch: Any,
    hidden_states: Any,
    positions: Any,
    main_slots: Any,
    cos_sin_cache: Any,
    main_kv_cache: Any,
    swa_indices: Any,
    swa_lengths: Any,
    producer_weights: Any,
    active_groups: Any,
    group_sequence_slots: Any,
    group_source_positions: Any,
    group_rope_positions: Any,
    compressed_slots: Any,
    active_sequences: Any,
    sequence_offsets: Any,
    sequence_start_positions: Any,
    state_sequence_ids: Any,
    compressed_cos_sin_cache: Any,
    compressed_main_cache: Any,
    main_kv_state: Any,
    main_score_state: Any,
    index_cache: Any,
    index_kv_state: Any,
    index_score_state: Any,
    compressor_weights: Any,
    indexer_weights: Any,
    real_page_table: Any,
    index_cache_seqlens: Any,
    output_weights: Any,
    residual: Any,
    prev_post: Any,
    prev_comb: Any,
    fn: Any,
    hc_scale: Any,
    hc_base: Any,
    norm_weight: Any,
    residual_out: Any,
    y_out: Any,
    post_out: Any,
    comb_out: Any,
    attn_sink: Any | None = None,
    sm_scale: float | None = None,
    producer_eps: float = 1.0e-6,
    compressor_eps: float = 1.0e-6,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1.0e-6,
) -> DeepseekV4C4ContinuationAttentionLayerBinding:
    """Bind one ordered C4 continuation chunk into the fixed lane arena."""

    import torch
    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
        dsa_indexer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if contract.compression != 4 or contract.mode != "extend":
        raise ValueError(
            "bound C4 continuation attention layer requires mode=extend and "
            f"compression=4; got mode={contract.mode!r} "
            f"compression={contract.compression}"
        )
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("C4 continuation hidden_states must be a rank-2 tensor")
    tokens = int(hidden_states.shape[0])
    arena_binding = bind_deepseek_v4_attention_layer(
        contract,
        arena=arena,
        tokens=tokens,
    )
    geometry = deepseek_v4_producer_geometry(contract.variant)
    expected_m = tokens

    producer_plan = _cached_target_capture_plan(
        dsv4_producer.plan,
        dsv4_producer.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=geometry.heads,
            head_dim=DS4_HEAD_DIM,
            nope_dim=DS4_NOPE_DIM,
            rope_dim=DS4_ROPE_DIM,
            page_size=DS4_SOURCE_PAGE_TOKENS,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    producer_binding = dsv4_producer.bind(
        producer_plan,
        scratch=arena_binding.producer_scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=main_slots,
        cos_sin_cache=cos_sin_cache,
        main_kv_cache=main_kv_cache,
        query=arena_binding.query,
        weights=producer_weights,
        eps=float(producer_eps),
        expected_m=expected_m,
    )

    compressor_plan = _cached_target_capture_plan(
        dsv4_compressor.plan,
        dsv4_compressor.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=contract.hidden,
            compress_ratio=4,
            with_indexer=True,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    compressor_binding = dsv4_compressor.bind_continuation(
        compressor_plan,
        scratch=arena_binding.compressor_scratch,
        hidden_states=hidden_states,
        active_groups=active_groups,
        group_sequence_slots=group_sequence_slots,
        group_source_positions=group_source_positions,
        group_rope_positions=group_rope_positions,
        compressed_slots=compressed_slots,
        active_sequences=active_sequences,
        sequence_offsets=sequence_offsets,
        sequence_start_positions=sequence_start_positions,
        state_sequence_ids=state_sequence_ids,
        compressed_cos_sin_cache=compressed_cos_sin_cache,
        compressed_main_cache=compressed_main_cache,
        main_kv_state=main_kv_state,
        main_score_state=main_score_state,
        index_cache=index_cache,
        index_kv_state=index_kv_state,
        index_score_state=index_score_state,
        weights=compressor_weights,
        eps=float(compressor_eps),
        expected_m=expected_m,
        ordered_continuation=True,
    )

    indexer_plan = dsv4_producer.plan_indexer(
        dsv4_producer.IndexerCaps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=DS4_INDEX_HEADS,
            head_dim=DS4_INDEX_HEAD_DIM,
            rope_dim=DS4_ROPE_DIM,
            dtype=torch.bfloat16,
        )
    )
    indexer_producer_binding = dsv4_producer.bind_indexer(
        indexer_plan,
        scratch=arena_binding.indexer_producer_scratch,
        q_rank=producer_binding.q_rank,
        hidden_states=hidden_states,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        query=arena_binding.index_query,
        head_weights=arena_binding.index_head_weights,
        weights=indexer_weights,
        expected_m=expected_m,
    )

    selector_plan = _cached_target_capture_plan(
        dsa_indexer.plan,
        dsa_indexer.Caps(
            device=hidden_states.device,
            source_layout=dsa_indexer.SOURCE_LAYOUT_PAGED,
            num_q_heads=DS4_INDEX_HEADS,
            max_q_rows=contract.max_rows,
            max_page_table_width=contract.max_page_table_width,
            topk=contract.indexed_width,
            mode="decode",
            page_size=64,
            shared_page_table=True,
        ),
    )
    selector_binding = selector_plan.bind(
        scratch=selector_scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=index_cache_seqlens,
        expected_num_q_heads=DS4_INDEX_HEADS,
        shared_page_table=True,
        output_physical_slots=True,
    )

    attention_contract = plan_deepseek_v4_compressed_mla(
        mode="extend",
        rows=contract.max_rows,
        heads=contract.heads,
        source_pages=contract.source_pages,
        max_page_table_width=contract.max_page_table_width,
        swa_width=contract.swa_width,
        compression=4,
        indexed_width=contract.indexed_width,
        cache_format=contract.cache_format,
    )
    attention_plan = _cached_target_capture_plan(
        compressed_mla.plan,
        compressed_mla.Caps(
            device=hidden_states.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=contract.heads,
            head_dim=DS4_HEAD_DIM,
            v_head_dim=DS4_HEAD_DIM,
            max_width=attention_contract.total_width,
            max_page_table_width=contract.max_page_table_width,
            max_q_rows=contract.max_rows,
            max_batch=contract.max_rows,
            max_kv_rows=0,
            max_chunks_per_row=attention_contract.max_chunks_per_row,
            page_size=DS4_SOURCE_PAGE_TOKENS,
        ),
    )
    attention_binding = compressed_mla.bind(
        attention_plan,
        scratch=arena_binding.attention_scratch,
        q=arena_binding.query,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=arena_binding.selected_indices,
        indexed_lengths=index_cache_seqlens,
    )
    attention_binding.scratch.mode = "extend"
    attention_binding.scratch.fixed_capacity = True
    attention_binding.scratch.use_cuda_graph = True

    output_contract = plan_deepseek_v4_attention_output(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    output_geometry = output_contract.geometry
    output_plan = _cached_target_capture_plan(
        wo_projection.plan,
        wo_projection.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            groups=output_geometry.groups,
            group_width=output_geometry.group_width,
            rank=output_geometry.rank,
            hidden=output_geometry.hidden,
            dtype=torch.bfloat16,
        ),
    )
    output_binding = wo_projection.bind_inv_rope(
        output_plan,
        scratch=arena_binding.output_projection_scratch,
        o=arena_binding.attention_output,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=output_weights,
        heads_per_group=output_geometry.heads_per_group,
        nope_dim=DS4_NOPE_DIM,
        rope_dim=DS4_ROPE_DIM,
        expected_m=expected_m,
    )

    mhc_contract = plan_deepseek_v4_mhc(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    mhc_plan = _cached_target_capture_plan(
        mhc.plan,
        mhc.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden_size=contract.hidden,
            split_k=mhc_contract.geometry.split_k,
            dtype=torch.bfloat16,
        ),
    )
    mhc_binding = mhc.bind(
        mhc_plan,
        scratch=arena_binding.mhc_scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=y_out,
        post=post_out,
        comb=comb_out,
        out=residual_out,
    )
    return DeepseekV4C4ContinuationAttentionLayerBinding(
        arena_binding=arena_binding,
        producer_binding=producer_binding,
        compressor_binding=compressor_binding,
        indexer_producer_binding=indexer_producer_binding,
        selector_binding=selector_binding,
        attention_binding=attention_binding,
        output_binding=output_binding,
        mhc_binding=mhc_binding,
        main_kv_cache=main_kv_cache,
        compressed_main_cache=compressed_main_cache,
        index_cache=index_cache,
        attn_sink=attn_sink,
        sm_scale=(DS4_HEAD_DIM**-0.5 if sm_scale is None else float(sm_scale)),
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_weight=norm_weight,
        norm_eps=float(norm_eps),
    )


def run_deepseek_v4_c4_continuation_attention_layer(
    binding: DeepseekV4C4ContinuationAttentionLayerBinding,
) -> tuple[Any, Any, Any, Any]:
    """Run ordered C4 continuation and return its next FFN input."""

    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
        dsa_indexer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if not isinstance(binding, DeepseekV4C4ContinuationAttentionLayerBinding):
        raise TypeError("C4 continuation attention layer requires its fixed binding")
    dsv4_producer.run(binding=binding.producer_binding)
    dsv4_compressor.run_continuation(binding=binding.compressor_binding)
    dsv4_producer.run_indexer(binding=binding.indexer_producer_binding)
    dsa_indexer.index_topk_fp8(
        q_fp8=binding.arena_binding.index_query,
        weights=binding.arena_binding.index_head_weights,
        index_k_cache=binding.index_cache,
        binding=binding.selector_binding,
        out_indices=binding.arena_binding.selected_indices,
    )
    compressed_mla.run(
        binding=binding.attention_binding,
        swa_k_cache=binding.main_kv_cache,
        sm_scale=binding.sm_scale,
        swa_page_size=DS4_SOURCE_PAGE_TOKENS,
        indexed_k_cache=binding.compressed_main_cache,
        indexed_page_size=64,
        attn_sink=binding.attn_sink,
        expected_num_q_heads=binding.arena_binding.contract.heads,
        backend="sm120",
        out=binding.arena_binding.attention_output,
    )
    delta = wo_projection.run_inv_rope(binding=binding.output_binding)
    return mhc.run_post_pre(
        delta,
        binding.residual,
        binding.prev_post,
        binding.prev_comb,
        binding.fn,
        binding.hc_scale,
        binding.hc_base,
        rms_eps=binding.rms_eps,
        hc_eps=binding.hc_eps,
        sinkhorn_iters=binding.sinkhorn_iters,
        norm_weight=binding.norm_weight,
        norm_eps=binding.norm_eps,
        binding=binding.mhc_binding,
    )


def bind_deepseek_v4_c4_decode_attention_layer(
    contract: DeepseekV4AttentionLayerContract,
    *,
    arena: Any,
    selector_scratch: Any,
    hidden_states: Any,
    positions: Any,
    main_slots: Any,
    cos_sin_cache: Any,
    main_kv_cache: Any,
    swa_indices: Any,
    swa_lengths: Any,
    producer_weights: Any,
    sequence_ids: Any,
    compressed_slots: Any,
    compressed_cos_sin_cache: Any,
    compressed_main_cache: Any,
    main_kv_state: Any,
    main_score_state: Any,
    index_cache: Any,
    index_kv_state: Any,
    index_score_state: Any,
    compressor_weights: Any,
    indexer_weights: Any,
    real_page_table: Any,
    index_cache_seqlens: Any,
    output_weights: Any,
    residual: Any,
    prev_post: Any,
    prev_comb: Any,
    fn: Any,
    hc_scale: Any,
    hc_base: Any,
    norm_weight: Any,
    residual_out: Any,
    y_out: Any,
    post_out: Any,
    comb_out: Any,
    attn_sink: Any | None = None,
    sm_scale: float | None = None,
    producer_eps: float = 1.0e-6,
    compressor_eps: float = 1.0e-6,
    rms_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1.0e-6,
) -> DeepseekV4C4DecodeAttentionLayerBinding:
    """Bind the C4 producer/compressor/learned-selector decode chain."""

    import torch
    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
        dsa_indexer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if contract.compression != 4 or contract.mode != "decode":
        raise ValueError(
            "bound C4 decode attention layer requires mode=decode and "
            f"compression=4; got mode={contract.mode!r} "
            f"compression={contract.compression}"
        )
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("C4 decode hidden_states must be a rank-2 tensor")
    tokens = int(hidden_states.shape[0])
    arena_binding = bind_deepseek_v4_attention_layer(
        contract,
        arena=arena,
        tokens=tokens,
    )
    geometry = deepseek_v4_producer_geometry(contract.variant)
    expected_m = contract.max_rows

    producer_plan = _cached_target_capture_plan(
        dsv4_producer.plan,
        dsv4_producer.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=geometry.heads,
            head_dim=DS4_HEAD_DIM,
            nope_dim=DS4_NOPE_DIM,
            rope_dim=DS4_ROPE_DIM,
            page_size=DS4_SOURCE_PAGE_TOKENS,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    producer_binding = dsv4_producer.bind(
        producer_plan,
        scratch=arena_binding.producer_scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=main_slots,
        cos_sin_cache=cos_sin_cache,
        main_kv_cache=main_kv_cache,
        query=arena_binding.query,
        weights=producer_weights,
        eps=float(producer_eps),
        expected_m=expected_m,
    )

    compressor_plan = _cached_target_capture_plan(
        dsv4_compressor.plan,
        dsv4_compressor.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=contract.hidden,
            compress_ratio=4,
            with_indexer=True,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        ),
    )
    compressor_binding = dsv4_compressor.bind_decode(
        compressor_plan,
        scratch=arena_binding.compressor_scratch,
        hidden_states=hidden_states,
        positions=positions,
        sequence_ids=sequence_ids,
        compressed_slots=compressed_slots,
        compressed_cos_sin_cache=compressed_cos_sin_cache,
        compressed_main_cache=compressed_main_cache,
        main_kv_state=main_kv_state,
        main_score_state=main_score_state,
        index_cache=index_cache,
        index_kv_state=index_kv_state,
        index_score_state=index_score_state,
        weights=compressor_weights,
        eps=float(compressor_eps),
        expected_m=tokens,
        rows_are_sequence_unique=True,
    )

    indexer_plan = dsv4_producer.plan_indexer(
        dsv4_producer.IndexerCaps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=DS4_INDEX_HEADS,
            head_dim=DS4_INDEX_HEAD_DIM,
            rope_dim=DS4_ROPE_DIM,
            dtype=torch.bfloat16,
        )
    )
    indexer_producer_binding = dsv4_producer.bind_indexer(
        indexer_plan,
        scratch=arena_binding.indexer_producer_scratch,
        q_rank=producer_binding.q_rank,
        hidden_states=hidden_states,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        query=arena_binding.index_query,
        head_weights=arena_binding.index_head_weights,
        weights=indexer_weights,
        expected_m=expected_m,
    )

    selector_plan = _cached_target_capture_plan(
        dsa_indexer.plan,
        dsa_indexer.Caps(
            device=hidden_states.device,
            source_layout=dsa_indexer.SOURCE_LAYOUT_PAGED,
            num_q_heads=DS4_INDEX_HEADS,
            max_q_rows=contract.max_rows,
            max_page_table_width=contract.max_page_table_width,
            topk=contract.indexed_width,
            mode="decode",
            page_size=64,
        ),
    )
    selector_binding = selector_plan.bind(
        scratch=selector_scratch,
        real_page_table=real_page_table,
        cache_seqlens_int32=index_cache_seqlens,
        expected_num_q_heads=DS4_INDEX_HEADS,
        output_physical_slots=True,
    )

    attention_contract = plan_deepseek_v4_compressed_mla(
        mode="decode",
        rows=contract.max_rows,
        heads=contract.heads,
        source_pages=contract.source_pages,
        max_page_table_width=contract.max_page_table_width,
        swa_width=contract.swa_width,
        compression=4,
        indexed_width=contract.indexed_width,
        cache_format=contract.cache_format,
    )
    attention_plan = _cached_target_capture_plan(
        compressed_mla.plan,
        compressed_mla.Caps(
            device=hidden_states.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=contract.heads,
            head_dim=DS4_HEAD_DIM,
            v_head_dim=DS4_HEAD_DIM,
            max_width=attention_contract.total_width,
            max_page_table_width=contract.max_page_table_width,
            max_q_rows=contract.max_rows,
            max_batch=contract.max_rows,
            max_kv_rows=0,
            max_chunks_per_row=attention_contract.max_chunks_per_row,
            page_size=DS4_SOURCE_PAGE_TOKENS,
        ),
    )
    attention_binding = compressed_mla.bind(
        attention_plan,
        scratch=arena_binding.attention_scratch,
        q=arena_binding.query,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=arena_binding.selected_indices,
        indexed_lengths=index_cache_seqlens,
    )
    attention_binding.scratch.mode = "decode"
    attention_binding.scratch.fixed_capacity = True
    attention_binding.scratch.use_cuda_graph = True

    output_contract = plan_deepseek_v4_attention_output(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    output_geometry = output_contract.geometry
    output_plan = _cached_target_capture_plan(
        wo_projection.plan,
        wo_projection.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            groups=output_geometry.groups,
            group_width=output_geometry.group_width,
            rank=output_geometry.rank,
            hidden=output_geometry.hidden,
            dtype=torch.bfloat16,
        ),
    )
    output_binding = wo_projection.bind_inv_rope(
        output_plan,
        scratch=arena_binding.output_projection_scratch,
        o=arena_binding.attention_output,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=output_weights,
        heads_per_group=output_geometry.heads_per_group,
        nope_dim=DS4_NOPE_DIM,
        rope_dim=DS4_ROPE_DIM,
        expected_m=expected_m,
    )

    mhc_contract = plan_deepseek_v4_mhc(
        variant=contract.variant,
        max_rows=contract.max_rows,
    )
    mhc_plan = _cached_target_capture_plan(
        mhc.plan,
        mhc.Caps(
            device=hidden_states.device,
            max_tokens=contract.max_rows,
            hidden_size=contract.hidden,
            split_k=mhc_contract.geometry.split_k,
            dtype=torch.bfloat16,
        ),
    )
    mhc_binding = mhc.bind(
        mhc_plan,
        scratch=arena_binding.mhc_scratch,
        tokens=tokens,
        expected_m=expected_m,
        y=y_out,
        post=post_out,
        comb=comb_out,
        out=residual_out,
    )
    return DeepseekV4C4DecodeAttentionLayerBinding(
        arena_binding=arena_binding,
        producer_binding=producer_binding,
        compressor_binding=compressor_binding,
        indexer_producer_binding=indexer_producer_binding,
        selector_binding=selector_binding,
        attention_binding=attention_binding,
        output_binding=output_binding,
        mhc_binding=mhc_binding,
        main_kv_cache=main_kv_cache,
        compressed_main_cache=compressed_main_cache,
        index_cache=index_cache,
        attn_sink=attn_sink,
        sm_scale=(DS4_HEAD_DIM**-0.5 if sm_scale is None else float(sm_scale)),
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=float(rms_eps),
        hc_eps=float(hc_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_weight=norm_weight,
        norm_eps=float(norm_eps),
    )


def run_deepseek_v4_c4_decode_attention_layer(
    binding: DeepseekV4C4DecodeAttentionLayerBinding,
) -> tuple[Any, Any, Any, Any]:
    """Run C4 decode and return residual, post, comb, normalized y."""

    from b12x.attention import (
        compressed_sparse_mla as compressed_mla,
        dsv4_compressor,
        dsv4_producer,
        dsa_indexer,
    )
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    if not isinstance(binding, DeepseekV4C4DecodeAttentionLayerBinding):
        raise TypeError("C4 decode attention layer requires its fixed binding")
    dsv4_producer.run(binding=binding.producer_binding)
    dsv4_compressor.run_decode(binding=binding.compressor_binding)
    dsv4_producer.run_indexer(binding=binding.indexer_producer_binding)
    dsa_indexer.index_topk_fp8(
        q_fp8=binding.arena_binding.index_query,
        weights=binding.arena_binding.index_head_weights,
        index_k_cache=binding.index_cache,
        binding=binding.selector_binding,
        out_indices=binding.arena_binding.selected_indices,
    )
    compressed_mla.run(
        binding=binding.attention_binding,
        swa_k_cache=binding.main_kv_cache,
        sm_scale=binding.sm_scale,
        swa_page_size=DS4_SOURCE_PAGE_TOKENS,
        indexed_k_cache=binding.compressed_main_cache,
        indexed_page_size=64,
        attn_sink=binding.attn_sink,
        expected_num_q_heads=binding.arena_binding.contract.heads,
        backend="sm120",
        out=binding.arena_binding.attention_output,
    )
    delta = wo_projection.run_inv_rope(binding=binding.output_binding)
    return mhc.run_post_pre(
        delta,
        binding.residual,
        binding.prev_post,
        binding.prev_comb,
        binding.fn,
        binding.hc_scale,
        binding.hc_base,
        rms_eps=binding.rms_eps,
        hc_eps=binding.hc_eps,
        sinkhorn_iters=binding.sinkhorn_iters,
        norm_weight=binding.norm_weight,
        norm_eps=binding.norm_eps,
        binding=binding.mhc_binding,
    )


def qualify_deepseek_v4_attention_layer_contract(**kwargs: Any) -> bool:
    contract = plan_deepseek_v4_attention_layer(
        variant=str(kwargs["variant"]),
        mode=str(kwargs["mode"]),
        compression=int(kwargs["compression"]),
        max_rows=int(kwargs["max_rows"]),
        source_pages=int(kwargs["source_pages"]),
        max_page_table_width=int(
            kwargs.get("max_page_table_width", kwargs["source_pages"])
        ),
        max_positions=int(kwargs["max_positions"]),
        swa_width=int(kwargs.get("swa_width", DS4_SWA_TOKENS)),
        cache_format=str(kwargs.get("cache_format", "fp8")),
    )
    common = {
        "variant": contract.variant,
        "max_rows": contract.max_rows,
        "cache_format": contract.cache_format,
    }
    if not qualify_deepseek_v4_attention_producer_contract(**common):
        return False
    if contract.compression in (4, 128) and not (
        qualify_deepseek_v4_attention_compressor_contract(
            **common,
            compress_ratio=contract.compression,
        )
    ):
        return False
    if contract.compression == 4 and not (
        qualify_deepseek_v4_attention_indexer_contract(**common)
    ):
        return False
    if not qualify_deepseek_v4_compressed_mla_contract(
        mode=contract.mode,
        rows=contract.max_rows,
        heads=contract.heads,
        source_pages=contract.source_pages,
        max_page_table_width=contract.max_page_table_width,
        swa_width=contract.swa_width,
        compression=contract.compression,
        indexed_width=contract.indexed_width,
        cache_format=contract.cache_format,
    ):
        return False
    if not qualify_deepseek_v4_attention_output_contract(**common):
        return False
    if not qualify_deepseek_v4_mhc_contract(**common):
        return False

    previous_end = 0
    for name, offset, nbytes in contract.arena.regions():
        if offset % DS4_LAYER_ARENA_ALIGNMENT:
            raise RuntimeError(f"composite arena region {name} lost alignment")
        if offset < previous_end:
            raise RuntimeError(
                f"composite arena region {name} overlaps its predecessor"
            )
        previous_end = offset + nbytes
    if _align_up(previous_end) != contract.arena.total_bytes:
        raise RuntimeError("composite attention-layer arena total drifted")
    if (
        contract.serving_allocates
        or contract.workspace_reuse_scope
        != "one-per-coordinator-execution-lane-across-layers"
        or contract.expert_tensor_parallel != DS4_EXPERT_TP
        or contract.expert_parallel
        or contract.changes_expert_tp
        or contract.expert_global_top_k != 6
        or not contract.expert_routes_identical_across_ranks
        or contract.expert_local_intermediate_fraction != (1, DS4_EXPERT_TP)
        or contract.expert_partial_output_width != contract.hidden
    ):
        raise RuntimeError("composite attention layer changed the fixed TP=4 lifecycle")

    import torch

    arena = torch.empty((contract.arena.total_bytes,), dtype=torch.uint8)
    binding = bind_deepseek_v4_attention_layer(
        contract,
        arena=arena,
        tokens=min(contract.max_rows, 1),
    )
    offsets = {name: offset for name, offset, _ in contract.arena.regions()}
    for name, tensor in binding.region_views():
        if tensor.data_ptr() != arena.data_ptr() + offsets[name]:
            raise RuntimeError(f"composite arena view {name} lost its bound offset")
    if (
        binding.serving_allocates
        or not binding.views_only
        or not binding.outputs_are_arena_views
    ):
        raise RuntimeError("composite arena binding lifecycle drifted")
    if contract.compression == 0 and (
        DeepseekV4SlidingAttentionLayerBinding.serving_allocates
        or not DeepseekV4SlidingAttentionLayerBinding.cuda_graph_safe
        or not DeepseekV4SlidingAttentionLayerBinding.uses_one_lane_arena
        or not DeepseekV4SlidingAttentionLayerBinding.persistent_state_is_external
        or DeepseekV4SlidingAttentionLayerBinding.expert_tensor_parallel
        != DS4_EXPERT_TP
        or DeepseekV4SlidingAttentionLayerBinding.expert_parallel
        or not callable(bind_deepseek_v4_sliding_attention_layer)
        or not callable(run_deepseek_v4_sliding_attention_layer)
    ):
        raise RuntimeError("composite sliding execution lifecycle drifted")
    if (
        contract.compression == 128
        and contract.mode == "decode"
        and (
            DeepseekV4C128DecodeAttentionLayerBinding.serving_allocates
            or not DeepseekV4C128DecodeAttentionLayerBinding.cuda_graph_safe
            or not DeepseekV4C128DecodeAttentionLayerBinding.uses_one_lane_arena
            or not DeepseekV4C128DecodeAttentionLayerBinding.persistent_state_is_external
            or not DeepseekV4C128DecodeAttentionLayerBinding.sequence_unique_rows
            or DeepseekV4C128DecodeAttentionLayerBinding.expert_tensor_parallel
            != DS4_EXPERT_TP
            or DeepseekV4C128DecodeAttentionLayerBinding.expert_parallel
            or not callable(bind_deepseek_v4_c128_decode_attention_layer)
            or not callable(run_deepseek_v4_c128_decode_attention_layer)
        )
    ):
        raise RuntimeError("composite C128 decode execution lifecycle drifted")
    if (
        contract.compression == 128
        and contract.mode == "extend"
        and (
            DeepseekV4C128PrefillAttentionLayerBinding.serving_allocates
            or not DeepseekV4C128PrefillAttentionLayerBinding.cuda_graph_safe
            or not DeepseekV4C128PrefillAttentionLayerBinding.uses_one_lane_arena
            or not DeepseekV4C128PrefillAttentionLayerBinding.persistent_state_is_external
            or not DeepseekV4C128PrefillAttentionLayerBinding.initial_prefill_only
            or not DeepseekV4C128PrefillAttentionLayerBinding.scheduler_owns_physical_selection
            or DeepseekV4C128PrefillAttentionLayerBinding.expert_tensor_parallel
            != DS4_EXPERT_TP
            or DeepseekV4C128PrefillAttentionLayerBinding.expert_parallel
            or not callable(bind_deepseek_v4_c128_prefill_attention_layer)
            or not callable(run_deepseek_v4_c128_prefill_attention_layer)
        )
    ):
        raise RuntimeError("composite C128 prefill execution lifecycle drifted")
    if (
        contract.compression == 128
        and contract.mode == "extend"
        and (
            DeepseekV4C128ContinuationAttentionLayerBinding.serving_allocates
            or not DeepseekV4C128ContinuationAttentionLayerBinding.cuda_graph_safe
            or not DeepseekV4C128ContinuationAttentionLayerBinding.uses_one_lane_arena
            or not DeepseekV4C128ContinuationAttentionLayerBinding.persistent_state_is_external
            or not DeepseekV4C128ContinuationAttentionLayerBinding.ordered_chunks_only
            or not DeepseekV4C128ContinuationAttentionLayerBinding.scheduler_owns_state_transactions
            or not DeepseekV4C128ContinuationAttentionLayerBinding.scheduler_owns_physical_selection
            or DeepseekV4C128ContinuationAttentionLayerBinding.expert_tensor_parallel
            != DS4_EXPERT_TP
            or DeepseekV4C128ContinuationAttentionLayerBinding.expert_parallel
            or not callable(bind_deepseek_v4_c128_continuation_attention_layer)
            or not callable(run_deepseek_v4_c128_continuation_attention_layer)
        )
    ):
        raise RuntimeError("composite C128 continuation execution lifecycle drifted")
    if (
        contract.compression == 4
        and contract.mode == "extend"
        and (
            DeepseekV4C4PrefillAttentionLayerBinding.serving_allocates
            or not DeepseekV4C4PrefillAttentionLayerBinding.cuda_graph_safe
            or not DeepseekV4C4PrefillAttentionLayerBinding.uses_one_lane_arena
            or not DeepseekV4C4PrefillAttentionLayerBinding.persistent_state_is_external
            or not DeepseekV4C4PrefillAttentionLayerBinding.selector_scratch_is_external
            or not DeepseekV4C4PrefillAttentionLayerBinding.selector_outputs_physical_slots
            or not DeepseekV4C4PrefillAttentionLayerBinding.selector_uses_shared_page_table
            or not DeepseekV4C4PrefillAttentionLayerBinding.selector_uses_causal_lengths
            or not DeepseekV4C4PrefillAttentionLayerBinding.initial_prefill_only
            or DeepseekV4C4PrefillAttentionLayerBinding.expert_tensor_parallel
            != DS4_EXPERT_TP
            or DeepseekV4C4PrefillAttentionLayerBinding.expert_parallel
            or not callable(bind_deepseek_v4_c4_prefill_attention_layer)
            or not callable(run_deepseek_v4_c4_prefill_attention_layer)
        )
    ):
        raise RuntimeError("composite C4 prefill execution lifecycle drifted")
    if (
        contract.compression == 4
        and contract.mode == "extend"
        and (
            DeepseekV4C4ContinuationAttentionLayerBinding.serving_allocates
            or not DeepseekV4C4ContinuationAttentionLayerBinding.cuda_graph_safe
            or not DeepseekV4C4ContinuationAttentionLayerBinding.uses_one_lane_arena
            or not DeepseekV4C4ContinuationAttentionLayerBinding.persistent_state_is_external
            or not DeepseekV4C4ContinuationAttentionLayerBinding.selector_scratch_is_external
            or not DeepseekV4C4ContinuationAttentionLayerBinding.selector_outputs_physical_slots
            or not DeepseekV4C4ContinuationAttentionLayerBinding.selector_uses_shared_page_table
            or not DeepseekV4C4ContinuationAttentionLayerBinding.selector_uses_causal_lengths
            or not DeepseekV4C4ContinuationAttentionLayerBinding.ordered_chunks_only
            or not DeepseekV4C4ContinuationAttentionLayerBinding.scheduler_owns_state_transactions
            or DeepseekV4C4ContinuationAttentionLayerBinding.expert_tensor_parallel
            != DS4_EXPERT_TP
            or DeepseekV4C4ContinuationAttentionLayerBinding.expert_parallel
            or not callable(bind_deepseek_v4_c4_continuation_attention_layer)
            or not callable(run_deepseek_v4_c4_continuation_attention_layer)
        )
    ):
        raise RuntimeError("composite C4 continuation execution lifecycle drifted")
    if (
        contract.compression == 4
        and contract.mode == "decode"
        and (
            DeepseekV4C4DecodeAttentionLayerBinding.serving_allocates
            or not DeepseekV4C4DecodeAttentionLayerBinding.cuda_graph_safe
            or not DeepseekV4C4DecodeAttentionLayerBinding.uses_one_lane_arena
            or not DeepseekV4C4DecodeAttentionLayerBinding.persistent_state_is_external
            or not DeepseekV4C4DecodeAttentionLayerBinding.selector_scratch_is_external
            or not DeepseekV4C4DecodeAttentionLayerBinding.selector_outputs_physical_slots
            or not DeepseekV4C4DecodeAttentionLayerBinding.sequence_unique_rows
            or DeepseekV4C4DecodeAttentionLayerBinding.expert_tensor_parallel
            != DS4_EXPERT_TP
            or DeepseekV4C4DecodeAttentionLayerBinding.expert_parallel
            or not callable(bind_deepseek_v4_c4_decode_attention_layer)
            or not callable(run_deepseek_v4_c4_decode_attention_layer)
        )
    ):
        raise RuntimeError("composite C4 decode execution lifecycle drifted")
    return True


def _align_up(value: int) -> int:
    return (
        (int(value) + DS4_LAYER_ARENA_ALIGNMENT - 1)
        // DS4_LAYER_ARENA_ALIGNMENT
        * DS4_LAYER_ARENA_ALIGNMENT
    )


__all__ = [
    "DeepseekV4AttentionLayerArenaBinding",
    "DeepseekV4AttentionLayerArenaLayout",
    "DeepseekV4AttentionLayerContract",
    "DeepseekV4C4ContinuationAttentionLayerBinding",
    "DeepseekV4C4DecodeAttentionLayerBinding",
    "DeepseekV4C4PrefillAttentionLayerBinding",
    "DeepseekV4C128ContinuationAttentionLayerBinding",
    "DeepseekV4C128DecodeAttentionLayerBinding",
    "DeepseekV4C128PrefillAttentionLayerBinding",
    "DeepseekV4SlidingAttentionLayerBinding",
    "bind_deepseek_v4_attention_layer",
    "bind_deepseek_v4_c4_continuation_attention_layer",
    "bind_deepseek_v4_c4_decode_attention_layer",
    "bind_deepseek_v4_c4_prefill_attention_layer",
    "bind_deepseek_v4_c128_continuation_attention_layer",
    "bind_deepseek_v4_c128_decode_attention_layer",
    "bind_deepseek_v4_c128_prefill_attention_layer",
    "bind_deepseek_v4_sliding_attention_layer",
    "capture_deepseek_v4_c4_continuation_attention_layer",
    "capture_deepseek_v4_c4_prefill_attention_layer",
    "capture_deepseek_v4_c128_continuation_attention_layer",
    "capture_deepseek_v4_c128_prefill_attention_layer",
    "capture_deepseek_v4_sliding_attention_layer",
    "deepseek_v4_c4_selector_scratch_nbytes",
    "deepseek_v4_attention_layer_arena_nbytes",
    "plan_deepseek_v4_attention_layer",
    "prepare_deepseek_v4_c4_continuation_attention_layer",
    "prepare_deepseek_v4_c4_prefill_attention_layer",
    "prepare_deepseek_v4_c128_continuation_attention_layer",
    "prepare_deepseek_v4_c128_prefill_attention_layer",
    "prepare_deepseek_v4_sliding_attention_layer",
    "qualify_deepseek_v4_attention_layer_contract",
    "run_deepseek_v4_c4_continuation_attention_layer",
    "run_deepseek_v4_c4_decode_attention_layer",
    "run_deepseek_v4_c4_prefill_attention_layer",
    "run_deepseek_v4_c128_continuation_attention_layer",
    "run_deepseek_v4_c128_decode_attention_layer",
    "run_deepseek_v4_c128_prefill_attention_layer",
    "run_deepseek_v4_sliding_attention_layer",
]
