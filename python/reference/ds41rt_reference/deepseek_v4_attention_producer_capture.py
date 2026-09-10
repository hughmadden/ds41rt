from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ds41rt_reference.deepseek_v4_kv_format import (
    DS4_INDEX_PAGE_BYTES,
    DS4_SOURCE_PAGE_TOKENS,
    FP8_UE8M0,
    deepseek_v4_kv_format,
)


DS4_HEAD_DIM = 512
DS4_NOPE_DIM = 448
DS4_ROPE_DIM = 64
DS4_MAIN_PAGE_BYTES = FP8_UE8M0.main_page_bytes
DS4_ACTIVATION_BLOCK = 128
DS4_INDEX_HEADS = 64
DS4_INDEX_HEAD_DIM = 128
DS4_INDEX_PAGE_TOKENS = 64
DS4_MAX_PRODUCER_ROWS = 8_192
_SCRATCH_ALIGNMENT = 1_024


@dataclass(frozen=True)
class DeepseekV4ProducerGeometry:
    variant: str
    hidden: int
    q_lora_rank: int
    heads: int

    @property
    def query_width(self) -> int:
        return self.heads * DS4_HEAD_DIM

    @property
    def joint_qkv_rank_width(self) -> int:
        return self.q_lora_rank + DS4_HEAD_DIM


@dataclass(frozen=True)
class DeepseekV4ProducerScratchLayout:
    qkv_linear_offset: int
    qkv_linear_bytes: int
    q_linear_offset: int
    q_linear_bytes: int
    qkv_output_offset: int
    qkv_output_bytes: int
    q_rank_offset: int
    q_rank_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class DeepseekV4AttentionProducerContract:
    geometry: DeepseekV4ProducerGeometry
    max_rows: int
    cache_format: str
    main_page_bytes: int
    scratch: DeepseekV4ProducerScratchLayout
    stages: tuple[str, ...] = (
        "joint-wq-a-wkv-block-fp8-k128",
        "fused-qrank-rmsnorm-kv-rmsnorm-rope-main-page-pack",
        "wq-b-block-fp8-k128-direct-query-output",
        "in-place-per-head-rmsnorm-partial-rope",
    )
    status: str = "qualified-main-q-kv-not-active"

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def stages_bf16_kv(self) -> bool:
        return False

    @property
    def touches_compressor_state(self) -> bool:
        return False

    def required_buffer_bytes(
        self, *, source_pages: int, max_positions: int
    ) -> dict[str, int]:
        source_pages = int(source_pages)
        max_positions = int(max_positions)
        if source_pages <= 0 or max_positions <= 0:
            raise ValueError("source_pages and max_positions must be positive")
        geometry = self.geometry
        return {
            "hidden_states": self.max_rows * geometry.hidden * 2,
            "positions": self.max_rows * 4,
            "main_slots": self.max_rows * 4,
            "cos_sin_cache": max_positions * DS4_ROPE_DIM * 4,
            "main_kv_cache": source_pages * self.main_page_bytes,
            "query": self.max_rows * geometry.query_width * 2,
            "scratch": self.scratch.total_bytes,
        }


@dataclass(frozen=True)
class DeepseekV4IndexerGeometry:
    variant: str
    hidden: int
    q_lora_rank: int
    top_k: int
    heads: int = DS4_INDEX_HEADS
    head_dim: int = DS4_INDEX_HEAD_DIM

    @property
    def query_width(self) -> int:
        return self.heads * self.head_dim


@dataclass(frozen=True)
class DeepseekV4IndexerProducerScratchLayout:
    q_linear_offset: int
    q_linear_bytes: int
    q_output_offset: int
    q_output_bytes: int
    weights_output_offset: int
    weights_output_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class DeepseekV4AttentionIndexerContract:
    geometry: DeepseekV4IndexerGeometry
    max_rows: int
    scratch: DeepseekV4IndexerProducerScratchLayout
    stages: tuple[str, ...] = (
        "index-wq-b-block-fp8-k128-from-shared-qrank",
        "partial-rope-randomized-hadamard-e2m1-qat-direct-fp8-query",
        "bf16-head-weight-projection-scaled-to-fp32",
        "paged-fp8-score-relu-learned-head-reduction",
        "causal-completed-c4-group-lengths",
        "direct-physical-slot-topk",
    )
    selector_owner: str = "b12x.attention.dsa_indexer"
    status: str = "qualified-index-query-and-physical-selection-not-active"

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def consumes_shared_q_rank(self) -> bool:
        return True

    @property
    def output_physical_slots(self) -> bool:
        return True

    @property
    def selection_scratch_planned_by_owner(self) -> bool:
        return True

    @property
    def supports_initial_prefill(self) -> bool:
        return True

    @property
    def supports_ordered_continuation(self) -> bool:
        return True

    def completed_groups(self, logical_tokens: int) -> int:
        logical_tokens = int(logical_tokens)
        if logical_tokens < 0:
            raise ValueError("logical_tokens must be non-negative")
        return logical_tokens // 4

    def selected_length(self, logical_tokens: int) -> int:
        return min(self.geometry.top_k, self.completed_groups(logical_tokens))

    def required_buffer_bytes(
        self, *, source_pages: int, max_positions: int
    ) -> dict[str, int]:
        source_pages = int(source_pages)
        max_positions = int(max_positions)
        if source_pages <= 0 or max_positions <= 0:
            raise ValueError("source_pages and max_positions must be positive")
        geometry = self.geometry
        return {
            "q_rank": self.max_rows * geometry.q_lora_rank * 2,
            "hidden_states": self.max_rows * geometry.hidden * 2,
            "positions": self.max_rows * 4,
            "cos_sin_cache": max_positions * DS4_ROPE_DIM * 4,
            "index_query": self.max_rows * geometry.query_width,
            "head_weights": self.max_rows * geometry.heads * 4,
            "index_k_cache": source_pages * DS4_INDEX_PAGE_BYTES,
            "real_page_table": self.max_rows * source_pages * 4,
            "completed_group_lengths": self.max_rows * 4,
            "selected_indices": self.max_rows * geometry.top_k * 4,
            "selected_lengths": self.max_rows * 4,
            "producer_scratch": self.scratch.total_bytes,
        }


def deepseek_v4_producer_geometry(variant: str) -> DeepseekV4ProducerGeometry:
    variant = str(variant).strip().lower()
    if variant == "flash":
        return DeepseekV4ProducerGeometry(
            variant="flash", hidden=4_096, q_lora_rank=1_024, heads=64
        )
    if variant == "pro":
        return DeepseekV4ProducerGeometry(
            variant="pro", hidden=7_168, q_lora_rank=1_536, heads=128
        )
    raise ValueError(f"DeepSeek V4 producer variant must be flash or pro, got {variant!r}")


def plan_deepseek_v4_attention_producer(
    *, variant: str, max_rows: int, cache_format: str = "fp8"
) -> DeepseekV4AttentionProducerContract:
    geometry = deepseek_v4_producer_geometry(variant)
    format_plan = deepseek_v4_kv_format(cache_format)
    max_rows = int(max_rows)
    if not 1 <= max_rows <= DS4_MAX_PRODUCER_ROWS:
        raise ValueError(
            f"DeepSeek V4 producer max_rows must be in [1, {DS4_MAX_PRODUCER_ROWS}], "
            f"got {max_rows}"
        )

    cursor = 0
    qkv_linear_offset = _align_up(cursor)
    qkv_linear_bytes = _block_fp8_linear_scratch_bytes(
        rows=max_rows, input_width=geometry.hidden
    )
    cursor = qkv_linear_offset + qkv_linear_bytes
    q_linear_offset = _align_up(cursor)
    q_linear_bytes = _block_fp8_linear_scratch_bytes(
        rows=max_rows, input_width=geometry.q_lora_rank
    )
    cursor = q_linear_offset + q_linear_bytes
    qkv_output_offset = _align_up(cursor)
    qkv_output_bytes = max_rows * geometry.joint_qkv_rank_width * 2
    cursor = qkv_output_offset + qkv_output_bytes
    q_rank_offset = _align_up(cursor)
    q_rank_bytes = max_rows * geometry.q_lora_rank * 2
    cursor = q_rank_offset + q_rank_bytes
    scratch = DeepseekV4ProducerScratchLayout(
        qkv_linear_offset=qkv_linear_offset,
        qkv_linear_bytes=qkv_linear_bytes,
        q_linear_offset=q_linear_offset,
        q_linear_bytes=q_linear_bytes,
        qkv_output_offset=qkv_output_offset,
        qkv_output_bytes=qkv_output_bytes,
        q_rank_offset=q_rank_offset,
        q_rank_bytes=q_rank_bytes,
        total_bytes=_align_up(cursor),
    )
    return DeepseekV4AttentionProducerContract(
        geometry=geometry,
        max_rows=max_rows,
        cache_format=format_plan.name,
        main_page_bytes=format_plan.main_page_bytes,
        scratch=scratch,
    )


def deepseek_v4_indexer_geometry(variant: str) -> DeepseekV4IndexerGeometry:
    producer = deepseek_v4_producer_geometry(variant)
    return DeepseekV4IndexerGeometry(
        variant=producer.variant,
        hidden=producer.hidden,
        q_lora_rank=producer.q_lora_rank,
        top_k=512 if producer.variant == "flash" else 1_024,
    )


def plan_deepseek_v4_attention_indexer(
    *, variant: str, max_rows: int
) -> DeepseekV4AttentionIndexerContract:
    geometry = deepseek_v4_indexer_geometry(variant)
    max_rows = int(max_rows)
    if not 1 <= max_rows <= DS4_MAX_PRODUCER_ROWS:
        raise ValueError(
            f"DeepSeek V4 indexer max_rows must be in [1, {DS4_MAX_PRODUCER_ROWS}], "
            f"got {max_rows}"
        )
    cursor = 0
    q_linear_offset = _align_up(cursor)
    q_linear_bytes = _block_fp8_linear_scratch_bytes(
        rows=max_rows, input_width=geometry.q_lora_rank
    )
    cursor = q_linear_offset + q_linear_bytes
    q_output_offset = _align_up(cursor)
    q_output_bytes = max_rows * geometry.query_width * 2
    cursor = q_output_offset + q_output_bytes
    weights_output_offset = _align_up(cursor)
    weights_output_bytes = max_rows * geometry.heads * 2
    cursor = weights_output_offset + weights_output_bytes
    scratch = DeepseekV4IndexerProducerScratchLayout(
        q_linear_offset=q_linear_offset,
        q_linear_bytes=q_linear_bytes,
        q_output_offset=q_output_offset,
        q_output_bytes=q_output_bytes,
        weights_output_offset=weights_output_offset,
        weights_output_bytes=weights_output_bytes,
        total_bytes=_align_up(cursor),
    )
    return DeepseekV4AttentionIndexerContract(
        geometry=geometry,
        max_rows=max_rows,
        scratch=scratch,
    )


def qualify_deepseek_v4_attention_producer_contract(**kwargs: Any) -> bool:
    """Cross-check DS41RT's independent plan against the pinned SparkInfer op."""

    contract = plan_deepseek_v4_attention_producer(
        variant=str(kwargs["variant"]),
        max_rows=int(kwargs["max_rows"]),
        cache_format=str(kwargs.get("cache_format", "fp8")),
    )
    import torch
    from b12x.attention import dsv4_producer

    geometry = contract.geometry
    sparkinfer_plan = dsv4_producer.plan(
        dsv4_producer.Caps(
            device="cpu",
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
        )
    )
    (scratch_spec,) = sparkinfer_plan.scratch_specs()
    if int(scratch_spec.nbytes) != contract.scratch.total_bytes:
        raise RuntimeError(
            "pinned SparkInfer DSV4 producer scratch ABI drifted: "
            f"DS41RT={contract.scratch.total_bytes} SparkInfer={scratch_spec.nbytes}"
        )
    layout = sparkinfer_plan.layout
    expected_layout = (
        contract.scratch.qkv_linear_offset,
        contract.scratch.qkv_linear_bytes,
        contract.scratch.q_linear_offset,
        contract.scratch.q_linear_bytes,
        contract.scratch.qkv_output_offset,
        contract.scratch.q_rank_offset,
    )
    actual_layout = (
        int(layout.qkv_linear_offset),
        int(layout.qkv_linear_bytes),
        int(layout.q_linear_offset),
        int(layout.q_linear_bytes),
        int(layout.qkv_output_offset),
        int(layout.q_rank_offset),
    )
    if actual_layout != expected_layout:
        raise RuntimeError(
            "pinned SparkInfer DSV4 producer arena offsets drifted: "
            f"DS41RT={expected_layout} SparkInfer={actual_layout}"
        )
    return True


def qualify_deepseek_v4_attention_indexer_contract(**kwargs: Any) -> bool:
    """Pin the dedicated query producer and the existing physical selector."""

    contract = plan_deepseek_v4_attention_indexer(
        variant=str(kwargs["variant"]), max_rows=int(kwargs["max_rows"])
    )
    import inspect
    import torch
    from b12x.attention import dsa_indexer, dsv4_producer

    geometry = contract.geometry
    sparkinfer_plan = dsv4_producer.plan_indexer(
        dsv4_producer.IndexerCaps(
            device="cpu",
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            q_lora_rank=geometry.q_lora_rank,
            heads=geometry.heads,
            head_dim=geometry.head_dim,
            rope_dim=DS4_ROPE_DIM,
            dtype=torch.bfloat16,
        )
    )
    (scratch_spec,) = sparkinfer_plan.scratch_specs()
    if int(scratch_spec.nbytes) != contract.scratch.total_bytes:
        raise RuntimeError(
            "pinned SparkInfer DSV4 indexer-producer scratch ABI drifted: "
            f"DS41RT={contract.scratch.total_bytes} SparkInfer={scratch_spec.nbytes}"
        )
    layout = sparkinfer_plan.layout
    expected_layout = (
        contract.scratch.q_linear_offset,
        contract.scratch.q_linear_bytes,
        contract.scratch.q_output_offset,
        contract.scratch.q_output_bytes,
        contract.scratch.weights_output_offset,
        contract.scratch.weights_output_bytes,
    )
    actual_layout = (
        int(layout.q_linear_offset),
        int(layout.q_linear_bytes),
        int(layout.q_output_offset),
        int(layout.q_output_bytes),
        int(layout.weights_output_offset),
        int(layout.weights_output_bytes),
    )
    if actual_layout != expected_layout:
        raise RuntimeError(
            "pinned SparkInfer DSV4 indexer-producer arena offsets drifted: "
            f"DS41RT={expected_layout} SparkInfer={actual_layout}"
        )
    required_producer_surface = {
        "IndexerCaps",
        "IndexerPlan",
        "IndexerBinding",
        "IndexerWeights",
        "plan_indexer",
        "bind_indexer",
        "pack_indexer_weights",
        "run_indexer",
    }
    missing_producer = required_producer_surface.difference(
        dsv4_producer.META.entry_points
    )
    if missing_producer:
        raise RuntimeError(
            "pinned SparkInfer DSV4 indexer producer surface is missing: "
            + ", ".join(sorted(missing_producer))
        )
    required_selector_surface = {"plan", "bind_paged", "index_topk_fp8"}
    missing_selector = required_selector_surface.difference(
        dsa_indexer.META.entry_points
    )
    if missing_selector:
        raise RuntimeError(
            "pinned SparkInfer physical index selector surface is missing: "
            + ", ".join(sorted(missing_selector))
        )
    if (
        int(dsa_indexer.INDEX_HEAD_DIM) != DS4_INDEX_HEAD_DIM
        or int(dsa_indexer.PAGED_INDEX_PAGE_SIZE) != DS4_INDEX_PAGE_TOKENS
    ):
        raise RuntimeError("pinned SparkInfer DSV4 index-cache geometry drifted")
    if "output_physical_slots" not in inspect.signature(
        dsa_indexer.bind_paged
    ).parameters:
        raise RuntimeError(
            "pinned SparkInfer paged index selector lost physical-slot output"
        )
    return True


def _block_fp8_linear_scratch_bytes(*, rows: int, input_width: int) -> int:
    if rows <= 0 or input_width <= 0 or input_width % 128:
        raise ValueError("block-FP8 rows must be positive and input_width divisible by 128")
    cursor = 0
    cursor = _align_up(cursor) + rows * input_width
    cursor = _align_up(cursor) + rows * (input_width // 32)
    scale_mma_bytes = (
        _ceil_div(rows, 128) * _ceil_div(input_width, 128) * 32 * 4 * 4
    )
    return _align_up(cursor) + scale_mma_bytes


def _align_up(value: int) -> int:
    return ((int(value) + _SCRATCH_ALIGNMENT - 1) // _SCRATCH_ALIGNMENT) * (
        _SCRATCH_ALIGNMENT
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)
