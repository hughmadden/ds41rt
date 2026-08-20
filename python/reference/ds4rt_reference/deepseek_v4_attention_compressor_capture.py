from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ds4rt_reference.deepseek_v4_kv_format import (
    DS4_INDEX_PAGE_BYTES,
    FP8_UE8M0,
    deepseek_v4_kv_format,
)

DS4_HEAD_DIM = 512
DS4_INDEX_HEAD_DIM = 128
DS4_ROPE_DIM = 64
DS4_SOURCE_PAGE_TOKENS = 256
DS4_C4_MAIN_PAGE_BYTES = FP8_UE8M0.c4_page_bytes
DS4_C128_MAIN_PAGE_BYTES = FP8_UE8M0.c128_page_bytes
DS4_MAX_COMPRESSOR_ROWS = 8_192
_SCRATCH_ALIGNMENT = 1_024


@dataclass(frozen=True)
class DeepseekV4CompressorGeometry:
    variant: str
    hidden: int
    compress_ratio: int
    with_indexer: bool

    @property
    def overlap(self) -> bool:
        return self.compress_ratio == 4

    @property
    def coefficient(self) -> int:
        return 2 if self.overlap else 1

    @property
    def state_rows(self) -> int:
        return self.coefficient * self.compress_ratio

    @property
    def main_projected_width(self) -> int:
        return self.coefficient * DS4_HEAD_DIM

    @property
    def index_projected_width(self) -> int:
        return self.coefficient * DS4_INDEX_HEAD_DIM if self.with_indexer else 0

    @property
    def joint_projection_width(self) -> int:
        return 2 * (self.main_projected_width + self.index_projected_width)

    @property
    def main_page_bytes(self) -> int:
        return (
            DS4_C4_MAIN_PAGE_BYTES
            if self.compress_ratio == 4
            else DS4_C128_MAIN_PAGE_BYTES
        )


@dataclass(frozen=True)
class DeepseekV4CompressorScratchLayout:
    projection_offset: int
    projection_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class DeepseekV4AttentionCompressorContract:
    geometry: DeepseekV4CompressorGeometry
    max_rows: int
    cache_format: str
    main_page_bytes: int
    scratch: DeepseekV4CompressorScratchLayout
    stages: tuple[str, ...] = (
        "joint-bf16-wkv-wgate-projection-caller-arena",
        "decode-fp32-sequence-state-update-and-gated-pool",
        "initial-prefill-parallel-group-pool-and-direct-pack",
        "initial-prefill-single-terminal-state-finalize-per-sequence",
        "continuation-carried-state-plus-ordered-chunk-pool-and-pack",
        "continuation-sequence-safe-terminal-state-finalize",
        "direct-rmsnorm-rope-compressed-main-page-pack",
        "c4-direct-hadamard-fp4-qat-index-page-pack",
    )
    status: str = "qualified-decode-and-prefill-continuation-not-active"

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def stages_compressed_output(self) -> bool:
        return False

    @property
    def sequence_unique_decode_only(self) -> bool:
        return True

    @property
    def supports_initial_prefill(self) -> bool:
        return True

    @property
    def supports_ordered_prefill(self) -> bool:
        return True

    @property
    def supports_mtp_state_transactions(self) -> bool:
        return False

    @property
    def initial_prefill_group_capacity(self) -> int:
        return self.max_rows // self.geometry.compress_ratio

    def required_initial_prefill_metadata_bytes(
        self, *, sequence_capacity: int
    ) -> dict[str, int]:
        sequence_capacity = int(sequence_capacity)
        if not 1 <= sequence_capacity <= self.max_rows:
            raise ValueError(
                f"sequence_capacity must be in [1,{self.max_rows}], "
                f"got {sequence_capacity}"
            )
        group_capacity = self.initial_prefill_group_capacity
        return {
            "active_groups": 4,
            "group_source_starts": group_capacity * 4,
            "group_rope_positions": group_capacity * 4,
            "prefill_compressed_slots": group_capacity * 4,
            "active_sequences": 4,
            "sequence_offsets": (sequence_capacity + 1) * 4,
            "state_sequence_ids": sequence_capacity * 4,
        }

    def required_continuation_metadata_bytes(
        self, *, sequence_capacity: int, group_capacity: int
    ) -> dict[str, int]:
        sequence_capacity = int(sequence_capacity)
        group_capacity = int(group_capacity)
        if not 1 <= sequence_capacity <= self.max_rows:
            raise ValueError(
                f"sequence_capacity must be in [1,{self.max_rows}], "
                f"got {sequence_capacity}"
            )
        if not 0 <= group_capacity <= self.max_rows:
            raise ValueError(
                f"group_capacity must be in [0,{self.max_rows}], got {group_capacity}"
            )
        return {
            "active_groups": 4,
            "group_sequence_slots": group_capacity * 4,
            "group_source_positions": group_capacity * 4,
            "group_rope_positions": group_capacity * 4,
            "continuation_compressed_slots": group_capacity * 4,
            "active_sequences": 4,
            "sequence_offsets": (sequence_capacity + 1) * 4,
            "sequence_start_positions": sequence_capacity * 4,
            "state_sequence_ids": sequence_capacity * 4,
        }

    def required_buffer_bytes(
        self, *, source_pages: int, state_sequences: int, max_positions: int
    ) -> dict[str, int]:
        source_pages = int(source_pages)
        state_sequences = int(state_sequences)
        max_positions = int(max_positions)
        if source_pages <= 0 or state_sequences <= 0 or max_positions <= 0:
            raise ValueError(
                "source_pages, state_sequences, and max_positions must be positive"
            )
        geometry = self.geometry
        main_state = (
            state_sequences * geometry.state_rows * geometry.main_projected_width * 4
        )
        required = {
            "hidden_states": self.max_rows * geometry.hidden * 2,
            "positions": self.max_rows * 4,
            "sequence_ids": self.max_rows * 4,
            "compressed_slots": self.max_rows * 4,
            "compressed_cos_sin_cache": max_positions * DS4_ROPE_DIM * 4,
            "compressed_main_cache": source_pages * self.main_page_bytes,
            "main_kv_state": main_state,
            "main_score_state": main_state,
            "scratch": self.scratch.total_bytes,
        }
        if geometry.with_indexer:
            index_state = (
                state_sequences
                * geometry.state_rows
                * geometry.index_projected_width
                * 4
            )
            required.update(
                {
                    "index_cache": source_pages * DS4_INDEX_PAGE_BYTES,
                    "index_kv_state": index_state,
                    "index_score_state": index_state,
                }
            )
        return required


def deepseek_v4_compressor_geometry(
    *, variant: str, compress_ratio: int
) -> DeepseekV4CompressorGeometry:
    variant = str(variant).strip().lower()
    if variant == "flash":
        hidden = 4_096
    elif variant == "pro":
        hidden = 7_168
    else:
        raise ValueError(
            f"DeepSeek V4 compressor variant must be flash or pro, got {variant!r}"
        )
    compress_ratio = int(compress_ratio)
    if compress_ratio not in (4, 128):
        raise ValueError(
            f"DeepSeek V4 compressor ratio must be 4 or 128, got {compress_ratio}"
        )
    return DeepseekV4CompressorGeometry(
        variant=variant,
        hidden=hidden,
        compress_ratio=compress_ratio,
        with_indexer=compress_ratio == 4,
    )


def plan_deepseek_v4_attention_compressor(
    *, variant: str, compress_ratio: int, max_rows: int, cache_format: str = "fp8"
) -> DeepseekV4AttentionCompressorContract:
    geometry = deepseek_v4_compressor_geometry(
        variant=variant, compress_ratio=compress_ratio
    )
    format_plan = deepseek_v4_kv_format(cache_format)
    max_rows = int(max_rows)
    if not 1 <= max_rows <= DS4_MAX_COMPRESSOR_ROWS:
        raise ValueError(
            f"DeepSeek V4 compressor max_rows must be in [1, "
            f"{DS4_MAX_COMPRESSOR_ROWS}], got {max_rows}"
        )
    projection_bytes = max_rows * geometry.joint_projection_width * 2
    scratch = DeepseekV4CompressorScratchLayout(
        projection_offset=0,
        projection_bytes=projection_bytes,
        total_bytes=_align_up(projection_bytes),
    )
    return DeepseekV4AttentionCompressorContract(
        geometry=geometry,
        max_rows=max_rows,
        cache_format=format_plan.name,
        main_page_bytes=(
            format_plan.c4_page_bytes
            if geometry.compress_ratio == 4
            else format_plan.c128_page_bytes
        ),
        scratch=scratch,
    )


def qualify_deepseek_v4_attention_compressor_contract(**kwargs: Any) -> bool:
    """Cross-check DS4RT's independent plan against the pinned SparkInfer op."""

    contract = plan_deepseek_v4_attention_compressor(
        variant=str(kwargs["variant"]),
        compress_ratio=int(kwargs["compress_ratio"]),
        max_rows=int(kwargs["max_rows"]),
        cache_format=str(kwargs.get("cache_format", "fp8")),
    )
    import torch
    from b12x.attention import dsv4_compressor

    geometry = contract.geometry
    sparkinfer_plan = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cpu",
            max_tokens=contract.max_rows,
            hidden=geometry.hidden,
            compress_ratio=geometry.compress_ratio,
            with_indexer=geometry.with_indexer,
            dtype=torch.bfloat16,
            cache_format=contract.cache_format,
        )
    )
    (scratch_spec,) = sparkinfer_plan.scratch_specs()
    if int(scratch_spec.nbytes) != contract.scratch.total_bytes:
        raise RuntimeError(
            "pinned SparkInfer DSV4 compressor scratch ABI drifted: "
            f"DS4RT={contract.scratch.total_bytes} SparkInfer={scratch_spec.nbytes}"
        )
    layout = sparkinfer_plan.layout
    expected_layout = (
        contract.scratch.projection_offset,
        contract.scratch.projection_bytes,
        contract.scratch.total_bytes,
    )
    actual_layout = (
        int(layout.projection_offset),
        int(layout.projection_bytes),
        int(layout.nbytes),
    )
    if actual_layout != expected_layout:
        raise RuntimeError(
            "pinned SparkInfer DSV4 compressor arena offsets drifted: "
            f"DS4RT={expected_layout} SparkInfer={actual_layout}"
        )
    required_prefill_surface = {
        "PrefillBinding",
        "bind_prefill",
        "run_prefill",
        "ContinuationBinding",
        "bind_continuation",
        "run_continuation",
    }
    missing_prefill_surface = required_prefill_surface.difference(
        dsv4_compressor.META.entry_points
    )
    if (
        missing_prefill_surface
        or not callable(getattr(sparkinfer_plan, "bind_prefill", None))
        or not callable(getattr(sparkinfer_plan, "bind_continuation", None))
    ):
        raise RuntimeError(
            "pinned SparkInfer DSV4 initial-prefill surface drifted: "
            f"missing={sorted(missing_prefill_surface)}"
        )
    return True


def _align_up(value: int) -> int:
    return ((int(value) + _SCRATCH_ALIGNMENT - 1) // _SCRATCH_ALIGNMENT) * (
        _SCRATCH_ALIGNMENT
    )
