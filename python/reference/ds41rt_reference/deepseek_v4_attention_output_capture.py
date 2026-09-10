from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DS4_HEAD_DIM = 512
DS4_NOPE_DIM = 448
DS4_ROPE_DIM = 64
DS4_O_LORA_RANK = 1_024
DS4_MXFP8_SCALE_VEC = 32
DS4_MXFP8_SCALE_ROW_TILE = 128
DS4_MXFP8_SCALE_K_TILE = 4
DS4_MAX_OUTPUT_ROWS = 8_192
_SCRATCH_ALIGNMENT = 1_024


@dataclass(frozen=True)
class DeepseekV4AttentionOutputGeometry:
    variant: str
    hidden: int
    heads: int
    groups: int
    rank: int = DS4_O_LORA_RANK
    head_dim: int = DS4_HEAD_DIM

    @property
    def heads_per_group(self) -> int:
        return self.heads // self.groups

    @property
    def group_width(self) -> int:
        return self.heads_per_group * self.head_dim

    @property
    def projected_width(self) -> int:
        return self.groups * self.rank

    @property
    def checkpoint_weight_bytes(self) -> int:
        wo_a_weight = self.projected_width * self.group_width
        wo_a_scale = _ceil_div(self.projected_width, 128) * _ceil_div(
            self.group_width, 128
        )
        wo_b_weight = self.hidden * self.projected_width
        wo_b_scale = _ceil_div(self.hidden, 128) * _ceil_div(
            self.projected_width, 128
        )
        return wo_a_weight + wo_a_scale + wo_b_weight + wo_b_scale


@dataclass(frozen=True)
class DeepseekV4AttentionOutputScratchLayout:
    x_q_values_offset: int
    x_q_scale_rows_offset: int
    x_q_scale_mma_offset: int
    tmp_offset: int
    tmp_q_values_offset: int
    tmp_q_scale_rows_offset: int
    tmp_q_scale_mma_offset: int
    output_offset: int
    total_bytes: int


@dataclass(frozen=True)
class DeepseekV4AttentionOutputContract:
    geometry: DeepseekV4AttentionOutputGeometry
    max_rows: int
    scratch: DeepseekV4AttentionOutputScratchLayout
    stages: tuple[str, ...] = (
        "inverse-rope-plus-mxfp8-activation-pack-from-attention-output",
        "grouped-wo-a-mxfp8-gemm",
        "group-major-rank-concatenation-and-mxfp8-pack",
        "wo-b-mxfp8-gemm-to-arena-output",
    )
    owner: str = "coordinator-local-attention"
    status: str = "qualified-grouped-output-projection-not-active"

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def fuses_inverse_rope(self) -> bool:
        return True

    @property
    def output_is_arena_view(self) -> bool:
        return True

    @property
    def changes_expert_tp(self) -> bool:
        return False

    def required_buffer_bytes(self, *, max_positions: int) -> dict[str, int]:
        max_positions = int(max_positions)
        if max_positions <= 0:
            raise ValueError("max_positions must be positive")
        geometry = self.geometry
        return {
            "attention_output": self.max_rows
            * geometry.heads
            * geometry.head_dim
            * 2,
            "positions": self.max_rows * 8,
            "cos_sin_cache": max_positions * DS4_ROPE_DIM * 4,
            "scratch_and_output": self.scratch.total_bytes,
        }


def deepseek_v4_attention_output_geometry(
    variant: str,
) -> DeepseekV4AttentionOutputGeometry:
    variant = str(variant).strip().lower()
    if variant == "flash":
        return DeepseekV4AttentionOutputGeometry(
            variant="flash", hidden=4_096, heads=64, groups=8
        )
    if variant == "pro":
        return DeepseekV4AttentionOutputGeometry(
            variant="pro", hidden=7_168, heads=128, groups=16
        )
    raise ValueError(
        f"DeepSeek V4 attention output variant must be flash or pro, got {variant!r}"
    )


def plan_deepseek_v4_attention_output(
    *, variant: str, max_rows: int
) -> DeepseekV4AttentionOutputContract:
    geometry = deepseek_v4_attention_output_geometry(variant)
    max_rows = int(max_rows)
    if not 1 <= max_rows <= DS4_MAX_OUTPUT_ROWS:
        raise ValueError(
            f"DeepSeek V4 attention output max_rows must be in [1, "
            f"{DS4_MAX_OUTPUT_ROWS}], got {max_rows}"
        )

    cursor = 0
    x_q_values_offset = _align_up(cursor)
    cursor = x_q_values_offset + max_rows * geometry.group_width * geometry.groups
    x_q_scale_rows_offset = _align_up(cursor)
    cursor = x_q_scale_rows_offset + (
        geometry.groups
        * max_rows
        * (geometry.group_width // DS4_MXFP8_SCALE_VEC)
    )
    x_q_scale_mma_offset = _align_up(cursor)
    cursor = x_q_scale_mma_offset + _scale_mma_bytes(
        rows=max_rows,
        width=geometry.group_width,
        groups=geometry.groups,
    )
    tmp_offset = _align_up(cursor)
    cursor = tmp_offset + max_rows * geometry.rank * geometry.groups * 2
    tmp_q_values_offset = _align_up(cursor)
    cursor = tmp_q_values_offset + max_rows * geometry.projected_width
    tmp_q_scale_rows_offset = _align_up(cursor)
    cursor = tmp_q_scale_rows_offset + (
        max_rows * (geometry.projected_width // DS4_MXFP8_SCALE_VEC)
    )
    tmp_q_scale_mma_offset = _align_up(cursor)
    cursor = tmp_q_scale_mma_offset + _scale_mma_bytes(
        rows=max_rows,
        width=geometry.projected_width,
        groups=1,
    )
    output_offset = _align_up(cursor)
    cursor = output_offset + max_rows * geometry.hidden * 2
    scratch = DeepseekV4AttentionOutputScratchLayout(
        x_q_values_offset=x_q_values_offset,
        x_q_scale_rows_offset=x_q_scale_rows_offset,
        x_q_scale_mma_offset=x_q_scale_mma_offset,
        tmp_offset=tmp_offset,
        tmp_q_values_offset=tmp_q_values_offset,
        tmp_q_scale_rows_offset=tmp_q_scale_rows_offset,
        tmp_q_scale_mma_offset=tmp_q_scale_mma_offset,
        output_offset=output_offset,
        total_bytes=cursor,
    )
    return DeepseekV4AttentionOutputContract(
        geometry=geometry,
        max_rows=max_rows,
        scratch=scratch,
    )


def qualify_deepseek_v4_attention_output_contract(**kwargs: Any) -> bool:
    """Cross-check the fixed output arena and non-allocating bound lifecycle."""

    contract = plan_deepseek_v4_attention_output(
        variant=str(kwargs["variant"]), max_rows=int(kwargs["max_rows"])
    )
    import torch
    from b12x.gemm import wo_projection

    geometry = contract.geometry
    sparkinfer_plan = wo_projection.plan(
        wo_projection.Caps(
            device="cpu",
            max_tokens=contract.max_rows,
            groups=geometry.groups,
            group_width=geometry.group_width,
            rank=geometry.rank,
            hidden=geometry.hidden,
            dtype=torch.bfloat16,
        )
    )
    (scratch_spec,) = sparkinfer_plan.scratch_specs()
    if int(scratch_spec.nbytes) != contract.scratch.total_bytes:
        raise RuntimeError(
            "pinned SparkInfer DSV4 output-projection scratch ABI drifted: "
            f"DS41RT={contract.scratch.total_bytes} SparkInfer={scratch_spec.nbytes}"
        )
    layout = sparkinfer_plan.layout
    expected_layout = (
        contract.scratch.x_q_values_offset,
        contract.scratch.x_q_scale_rows_offset,
        contract.scratch.x_q_scale_mma_offset,
        contract.scratch.tmp_offset,
        contract.scratch.tmp_q_values_offset,
        contract.scratch.tmp_q_scale_rows_offset,
        contract.scratch.tmp_q_scale_mma_offset,
        contract.scratch.output_offset,
    )
    actual_layout = (
        int(layout.x_q_values_offset_bytes),
        int(layout.x_q_scale_rows_offset_bytes),
        int(layout.x_q_scale_mma_offset_bytes),
        int(layout.tmp_offset_bytes),
        int(layout.tmp_q_values_offset_bytes),
        int(layout.tmp_q_scale_rows_offset_bytes),
        int(layout.tmp_q_scale_mma_offset_bytes),
        int(layout.output_offset_bytes),
    )
    if actual_layout != expected_layout:
        raise RuntimeError(
            "pinned SparkInfer DSV4 output-projection arena offsets drifted: "
            f"DS41RT={expected_layout} SparkInfer={actual_layout}"
        )
    required_surface = {
        "Caps",
        "Plan",
        "InvRopeBinding",
        "Weights",
        "plan",
        "bind_inv_rope",
        "run_inv_rope",
        "pack_weights",
    }
    missing_surface = required_surface.difference(wo_projection.META.entry_points)
    if missing_surface:
        raise RuntimeError(
            "pinned SparkInfer DSV4 output-projection surface is missing: "
            + ", ".join(sorted(missing_surface))
        )
    if (
        wo_projection.InvRopeBinding.serving_allocates is not False
        or wo_projection.InvRopeBinding.output_is_bound_arena is not True
    ):
        raise RuntimeError(
            "pinned SparkInfer inverse-RoPE WO lost its caller-owned arena lifecycle"
        )
    return True


def _scale_mma_bytes(*, rows: int, width: int, groups: int) -> int:
    return (
        groups
        * _ceil_div(rows, DS4_MXFP8_SCALE_ROW_TILE)
        * _ceil_div(
            width // DS4_MXFP8_SCALE_VEC,
            DS4_MXFP8_SCALE_K_TILE,
        )
        * 32
        * 4
        * 4
    )


def _align_up(value: int) -> int:
    return _ceil_div(value, _SCRATCH_ALIGNMENT) * _SCRATCH_ALIGNMENT


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


__all__ = [
    "DeepseekV4AttentionOutputContract",
    "DeepseekV4AttentionOutputGeometry",
    "DeepseekV4AttentionOutputScratchLayout",
    "deepseek_v4_attention_output_geometry",
    "plan_deepseek_v4_attention_output",
    "qualify_deepseek_v4_attention_output_contract",
]
