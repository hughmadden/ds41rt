from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ds41rt_reference.deepseek_v4_attention_layer_capture import (
    DS4_EXPERT_TP,
    DS4_LAYER_ARENA_ALIGNMENT,
)
from ds41rt_reference.deepseek_v4_spark_rank_capture import (
    DS4_FLASH_GLOBAL_TOP_K,
    DS4_FLASH_HIDDEN,
    DS4_FLASH_LOCAL_INTERMEDIATE,
    DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS,
    DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS,
    DS4_FLASH_PREFILL_MAX_ROWS,
    DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS,
    DS4_FLASH_ROUTED_EXPERTS,
    DS4_FLASH_SPARK_RESIDENT_WEIGHT_BYTES,
)

DS4_FLASH_PREFILL_SCRATCH_ELEMENTS = 3_145_728
DS4_FLASH_LOCK_ELEMENTS = 194


@dataclass(frozen=True)
class DeepseekV4FlashSparkPrefillArenaRegion:
    name: str
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class DeepseekV4FlashSparkPrefillArenaLayout:
    regions: tuple[DeepseekV4FlashSparkPrefillArenaRegion, ...]
    total_bytes: int

    def region(self, name: str) -> DeepseekV4FlashSparkPrefillArenaRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)


@dataclass(frozen=True)
class DeepseekV4FlashSparkPrefillContract:
    rank: int
    arena: DeepseekV4FlashSparkPrefillArenaLayout
    variant: str = "flash"
    max_rows: int = DS4_FLASH_PREFILL_MAX_ROWS
    hidden: int = DS4_FLASH_HIDDEN
    routed_experts: int = DS4_FLASH_ROUTED_EXPERTS
    global_top_k: int = DS4_FLASH_GLOBAL_TOP_K
    expert_tensor_parallel: int = DS4_EXPERT_TP
    expert_parallel: bool = False
    local_intermediate: int = DS4_FLASH_LOCAL_INTERMEDIATE
    route_block_rows: int = DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS
    max_packed_route_slots: int = DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS
    max_route_blocks: int = DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS
    status: str = "qualified-flash-spark-prefill-rank-not-active"

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def resident_weights_are_external(self) -> bool:
        return True

    @property
    def resident_weight_bytes(self) -> int:
        return DS4_FLASH_SPARK_RESIDENT_WEIGHT_BYTES

    @property
    def owns_same_expert_ids_as_every_rank(self) -> bool:
        return True

    @property
    def consumes_identical_global_routes(self) -> bool:
        return True

    @property
    def applies_expert_ownership_map(self) -> bool:
        return False

    @property
    def applies_route_weights_locally(self) -> bool:
        return True

    @property
    def output_capacity_shape(self) -> tuple[int, int]:
        return (self.max_rows, self.hidden)

    @property
    def output_dtype(self) -> str:
        return "bfloat16"

    @property
    def output_is_rank_partial(self) -> bool:
        return True

    @property
    def requires_coordinator_reduction(self) -> bool:
        return True

    @property
    def minimum_cuda_graph_kernel_nodes(self) -> int:
        return 5

    @property
    def minimum_cuda_graph_memset_nodes(self) -> int:
        return 2

    @property
    def cuda_graph_memcpy_nodes(self) -> int:
        return 0


@dataclass(frozen=True)
class DeepseekV4FlashSparkPrefillBinding:
    serving_allocates: ClassVar[bool] = False
    views_only: ClassVar[bool] = True
    cuda_graph_safe: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    contract: DeepseekV4FlashSparkPrefillContract
    arena: Any
    input: Any
    topk_ids: Any
    topk_weights: Any
    packed_route_indices: Any
    block_expert_ids: Any
    packed_route_count: Any
    expert_counts: Any
    expert_offsets: Any
    fc1: Any
    activated: Any
    routed_output: Any
    output: Any
    fc1_scratch: Any
    fc2_scratch: Any
    locks: Any

    def region_views(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (region.name, getattr(self, region.name))
            for region in self.contract.arena.regions
        )


def plan_deepseek_v4_flash_spark_prefill(
    *, rank: int
) -> DeepseekV4FlashSparkPrefillContract:
    rank = int(rank)
    if not 0 <= rank < DS4_EXPERT_TP:
        raise ValueError(
            f"Flash Spark prefill rank must be in [0, {DS4_EXPERT_TP}), got {rank}"
        )

    regions: list[DeepseekV4FlashSparkPrefillArenaRegion] = []
    offset = 0

    def reserve(
        name: str, nbytes: int, *, dtype: str, shape: tuple[int, ...]
    ) -> None:
        nonlocal offset
        start = _align_up(offset)
        regions.append(
            DeepseekV4FlashSparkPrefillArenaRegion(
                name=name,
                offset=start,
                nbytes=nbytes,
                dtype=dtype,
                shape=shape,
            )
        )
        offset = start + nbytes

    routed_rows = DS4_FLASH_PREFILL_MAX_ROWS * DS4_FLASH_GLOBAL_TOP_K
    reserve(
        "input",
        DS4_FLASH_PREFILL_MAX_ROWS * DS4_FLASH_HIDDEN * 2,
        dtype="bfloat16",
        shape=(DS4_FLASH_PREFILL_MAX_ROWS, DS4_FLASH_HIDDEN),
    )
    reserve(
        "topk_ids",
        routed_rows * 4,
        dtype="int32",
        shape=(DS4_FLASH_PREFILL_MAX_ROWS, DS4_FLASH_GLOBAL_TOP_K),
    )
    reserve(
        "topk_weights",
        routed_rows * 4,
        dtype="float32",
        shape=(DS4_FLASH_PREFILL_MAX_ROWS, DS4_FLASH_GLOBAL_TOP_K),
    )
    reserve(
        "packed_route_indices",
        DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS * 4,
        dtype="int32",
        shape=(DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS,),
    )
    reserve(
        "block_expert_ids",
        DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS * 4,
        dtype="int32",
        shape=(DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS,),
    )
    reserve("packed_route_count", 4, dtype="int32", shape=(1,))
    reserve(
        "expert_counts",
        DS4_FLASH_ROUTED_EXPERTS * 4,
        dtype="int32",
        shape=(DS4_FLASH_ROUTED_EXPERTS,),
    )
    reserve(
        "expert_offsets",
        (DS4_FLASH_ROUTED_EXPERTS + 1) * 4,
        dtype="int32",
        shape=(DS4_FLASH_ROUTED_EXPERTS + 1,),
    )
    reserve(
        "fc1",
        routed_rows * 2 * DS4_FLASH_LOCAL_INTERMEDIATE * 2,
        dtype="bfloat16",
        shape=(routed_rows, 2 * DS4_FLASH_LOCAL_INTERMEDIATE),
    )
    reserve(
        "activated",
        routed_rows * DS4_FLASH_LOCAL_INTERMEDIATE * 2,
        dtype="bfloat16",
        shape=(routed_rows, DS4_FLASH_LOCAL_INTERMEDIATE),
    )
    reserve(
        "routed_output",
        routed_rows * DS4_FLASH_HIDDEN * 2,
        dtype="bfloat16",
        shape=(routed_rows, DS4_FLASH_HIDDEN),
    )
    reserve(
        "output",
        DS4_FLASH_PREFILL_MAX_ROWS * DS4_FLASH_HIDDEN * 2,
        dtype="bfloat16",
        shape=(DS4_FLASH_PREFILL_MAX_ROWS, DS4_FLASH_HIDDEN),
    )
    reserve(
        "fc1_scratch",
        DS4_FLASH_PREFILL_SCRATCH_ELEMENTS * 4,
        dtype="float32",
        shape=(DS4_FLASH_PREFILL_SCRATCH_ELEMENTS,),
    )
    reserve(
        "fc2_scratch",
        DS4_FLASH_PREFILL_SCRATCH_ELEMENTS * 4,
        dtype="float32",
        shape=(DS4_FLASH_PREFILL_SCRATCH_ELEMENTS,),
    )
    reserve(
        "locks",
        DS4_FLASH_LOCK_ELEMENTS * 4,
        dtype="int32",
        shape=(DS4_FLASH_LOCK_ELEMENTS,),
    )
    return DeepseekV4FlashSparkPrefillContract(
        rank=rank,
        arena=DeepseekV4FlashSparkPrefillArenaLayout(
            regions=tuple(regions), total_bytes=_align_up(offset)
        ),
    )


def bind_deepseek_v4_flash_spark_prefill(
    contract: DeepseekV4FlashSparkPrefillContract,
    *,
    arena: Any,
) -> DeepseekV4FlashSparkPrefillBinding:
    import torch

    if not isinstance(contract, DeepseekV4FlashSparkPrefillContract):
        raise TypeError("Flash Spark prefill arena requires its planned contract")
    if not isinstance(arena, torch.Tensor) or arena.ndim != 1:
        raise TypeError("Flash Spark prefill arena must be a rank-1 tensor")
    if arena.dtype != torch.uint8 or not arena.is_contiguous():
        raise ValueError("Flash Spark prefill arena must be contiguous torch.uint8")
    if arena.numel() < contract.arena.total_bytes:
        raise ValueError(
            "Flash Spark prefill arena is too small: "
            f"need={contract.arena.total_bytes}, have={arena.numel()}"
        )
    arena = arena.narrow(0, 0, contract.arena.total_bytes)
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
    }
    views = {
        region.name: arena.narrow(0, region.offset, region.nbytes)
        .view(dtypes[region.dtype])
        .view(region.shape)
        for region in contract.arena.regions
    }
    return DeepseekV4FlashSparkPrefillBinding(
        contract=contract, arena=arena, **views
    )


def qualify_deepseek_v4_flash_spark_prefill_contract() -> bool:
    contracts = tuple(
        plan_deepseek_v4_flash_spark_prefill(rank=rank)
        for rank in range(DS4_EXPERT_TP)
    )
    first = contracts[0]
    expected_names = (
        "input",
        "topk_ids",
        "topk_weights",
        "packed_route_indices",
        "block_expert_ids",
        "packed_route_count",
        "expert_counts",
        "expert_offsets",
        "fc1",
        "activated",
        "routed_output",
        "output",
        "fc1_scratch",
        "fc2_scratch",
        "locks",
    )
    for contract in contracts:
        if (
            contract.rank not in range(DS4_EXPERT_TP)
            or contract.variant != "flash"
            or contract.max_rows != DS4_FLASH_PREFILL_MAX_ROWS
            or contract.hidden != DS4_FLASH_HIDDEN
            or contract.routed_experts != DS4_FLASH_ROUTED_EXPERTS
            or contract.global_top_k != DS4_FLASH_GLOBAL_TOP_K
            or contract.expert_tensor_parallel != DS4_EXPERT_TP
            or contract.expert_parallel
            or contract.local_intermediate != DS4_FLASH_LOCAL_INTERMEDIATE
            or contract.route_block_rows != DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS
            or contract.max_packed_route_slots
            != DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS
            or contract.max_route_blocks != DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS
            or contract.serving_allocates
            or not contract.resident_weights_are_external
            or contract.resident_weight_bytes != DS4_FLASH_SPARK_RESIDENT_WEIGHT_BYTES
            or not contract.owns_same_expert_ids_as_every_rank
            or not contract.consumes_identical_global_routes
            or contract.applies_expert_ownership_map
            or not contract.applies_route_weights_locally
            or contract.output_capacity_shape
            != (DS4_FLASH_PREFILL_MAX_ROWS, DS4_FLASH_HIDDEN)
            or contract.output_dtype != "bfloat16"
            or not contract.output_is_rank_partial
            or not contract.requires_coordinator_reduction
            or contract.minimum_cuda_graph_kernel_nodes != 5
            or contract.minimum_cuda_graph_memset_nodes != 2
            or contract.cuda_graph_memcpy_nodes != 0
            or contract.arena.total_bytes != 197_319_680
            or contract.arena != first.arena
            or tuple(region.name for region in contract.arena.regions)
            != expected_names
        ):
            raise RuntimeError(
                "Flash Spark prefill lost the fixed replicated TP=4 contract"
            )
        previous_end = 0
        for region in contract.arena.regions:
            if region.offset % DS4_LAYER_ARENA_ALIGNMENT:
                raise RuntimeError(
                    f"Flash Spark prefill arena region {region.name} lost alignment"
                )
            if region.offset < previous_end:
                raise RuntimeError(
                    f"Flash Spark prefill arena region {region.name} overlaps"
                )
            previous_end = region.offset + region.nbytes
        if _align_up(previous_end) != contract.arena.total_bytes:
            raise RuntimeError("Flash Spark prefill arena total drifted")
    if (
        DeepseekV4FlashSparkPrefillBinding.serving_allocates
        or not DeepseekV4FlashSparkPrefillBinding.views_only
        or not DeepseekV4FlashSparkPrefillBinding.cuda_graph_safe
        or DeepseekV4FlashSparkPrefillBinding.expert_tensor_parallel != DS4_EXPERT_TP
        or DeepseekV4FlashSparkPrefillBinding.expert_parallel
    ):
        raise RuntimeError("Flash Spark prefill binding lifecycle drifted")
    return True


def _align_up(value: int) -> int:
    return (
        (int(value) + DS4_LAYER_ARENA_ALIGNMENT - 1)
        // DS4_LAYER_ARENA_ALIGNMENT
        * DS4_LAYER_ARENA_ALIGNMENT
    )


__all__ = [
    "DeepseekV4FlashSparkPrefillArenaLayout",
    "DeepseekV4FlashSparkPrefillArenaRegion",
    "DeepseekV4FlashSparkPrefillBinding",
    "DeepseekV4FlashSparkPrefillContract",
    "bind_deepseek_v4_flash_spark_prefill",
    "plan_deepseek_v4_flash_spark_prefill",
    "qualify_deepseek_v4_flash_spark_prefill_contract",
]
