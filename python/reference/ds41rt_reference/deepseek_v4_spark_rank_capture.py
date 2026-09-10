from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ds41rt_reference.deepseek_v4_attention_layer_capture import (
    DS4_EXPERT_TP,
    DS4_LAYER_ARENA_ALIGNMENT,
)

DS4_FLASH_HIDDEN = 4_096
DS4_FLASH_ROUTED_EXPERTS = 256
DS4_FLASH_GLOBAL_TOP_K = 6
DS4_FLASH_LOCAL_INTERMEDIATE = 512
DS4_FLASH_FC1_SCRATCH_ELEMENTS = 98_304
DS4_FLASH_FC2_SCRATCH_ELEMENTS = 393_216
DS4_FLASH_LOCK_ELEMENTS = 194
DS4_FLASH_SPARK_RESIDENT_WEIGHT_BYTES = 855_640_064
DS4_FLASH_PREFILL_MAX_ROWS = 2_048
DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS = 32
DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS = 20_224
DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS = 632


@dataclass(frozen=True)
class DeepseekV4FlashSparkRankDecodeM1ArenaLayout:
    input_offset: int
    input_bytes: int
    route_indices_offset: int
    route_indices_bytes: int
    route_weights_offset: int
    route_weights_bytes: int
    fc1_offset: int
    fc1_bytes: int
    activated_offset: int
    activated_bytes: int
    output_offset: int
    output_bytes: int
    block_expert_ids_offset: int
    block_expert_ids_bytes: int
    packed_route_count_offset: int
    packed_route_count_bytes: int
    fc1_scratch_offset: int
    fc1_scratch_bytes: int
    fc2_scratch_offset: int
    fc2_scratch_bytes: int
    locks_offset: int
    locks_bytes: int
    total_bytes: int

    def regions(self) -> tuple[tuple[str, int, int], ...]:
        return (
            ("input", self.input_offset, self.input_bytes),
            (
                "route_indices",
                self.route_indices_offset,
                self.route_indices_bytes,
            ),
            (
                "route_weights",
                self.route_weights_offset,
                self.route_weights_bytes,
            ),
            ("fc1", self.fc1_offset, self.fc1_bytes),
            ("activated", self.activated_offset, self.activated_bytes),
            ("output", self.output_offset, self.output_bytes),
            (
                "block_expert_ids",
                self.block_expert_ids_offset,
                self.block_expert_ids_bytes,
            ),
            (
                "packed_route_count",
                self.packed_route_count_offset,
                self.packed_route_count_bytes,
            ),
            (
                "fc1_scratch",
                self.fc1_scratch_offset,
                self.fc1_scratch_bytes,
            ),
            (
                "fc2_scratch",
                self.fc2_scratch_offset,
                self.fc2_scratch_bytes,
            ),
            ("locks", self.locks_offset, self.locks_bytes),
        )


@dataclass(frozen=True)
class DeepseekV4FlashSparkRankDecodeM1Contract:
    rank: int
    arena: DeepseekV4FlashSparkRankDecodeM1ArenaLayout
    variant: str = "flash"
    rows: int = 1
    hidden: int = DS4_FLASH_HIDDEN
    routed_experts: int = DS4_FLASH_ROUTED_EXPERTS
    global_top_k: int = DS4_FLASH_GLOBAL_TOP_K
    expert_tensor_parallel: int = DS4_EXPERT_TP
    expert_parallel: bool = False
    local_intermediate: int = DS4_FLASH_LOCAL_INTERMEDIATE
    status: str = "qualified-flash-spark-rank-decode-m1-not-active"

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
    def requires_host_route_pack(self) -> bool:
        return False

    @property
    def applies_route_weights_locally(self) -> bool:
        return True

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.rows, self.hidden)

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
    def cuda_graph_memcpy_nodes(self) -> int:
        return 0


@dataclass(frozen=True)
class DeepseekV4FlashSparkRankDecodeM1Binding:
    serving_allocates: ClassVar[bool] = False
    views_only: ClassVar[bool] = True
    cuda_graph_safe: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    contract: DeepseekV4FlashSparkRankDecodeM1Contract
    arena: Any
    input: Any
    route_indices: Any
    route_weights: Any
    fc1: Any
    activated: Any
    output: Any
    block_expert_ids: Any
    packed_route_count: Any
    fc1_scratch: Any
    fc2_scratch: Any
    locks: Any

    def region_views(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("input", self.input),
            ("route_indices", self.route_indices),
            ("route_weights", self.route_weights),
            ("fc1", self.fc1),
            ("activated", self.activated),
            ("output", self.output),
            ("block_expert_ids", self.block_expert_ids),
            ("packed_route_count", self.packed_route_count),
            ("fc1_scratch", self.fc1_scratch),
            ("fc2_scratch", self.fc2_scratch),
            ("locks", self.locks),
        )


@dataclass(frozen=True)
class DeepseekV4FlashSparkPrefillRoutePackArenaLayout:
    topk_ids_offset: int
    topk_ids_bytes: int
    packed_route_indices_offset: int
    packed_route_indices_bytes: int
    block_expert_ids_offset: int
    block_expert_ids_bytes: int
    packed_route_count_offset: int
    packed_route_count_bytes: int
    expert_counts_offset: int
    expert_counts_bytes: int
    expert_offsets_offset: int
    expert_offsets_bytes: int
    total_bytes: int

    def regions(self) -> tuple[tuple[str, int, int], ...]:
        return (
            ("topk_ids", self.topk_ids_offset, self.topk_ids_bytes),
            (
                "packed_route_indices",
                self.packed_route_indices_offset,
                self.packed_route_indices_bytes,
            ),
            (
                "block_expert_ids",
                self.block_expert_ids_offset,
                self.block_expert_ids_bytes,
            ),
            (
                "packed_route_count",
                self.packed_route_count_offset,
                self.packed_route_count_bytes,
            ),
            (
                "expert_counts",
                self.expert_counts_offset,
                self.expert_counts_bytes,
            ),
            (
                "expert_offsets",
                self.expert_offsets_offset,
                self.expert_offsets_bytes,
            ),
        )


@dataclass(frozen=True)
class DeepseekV4FlashSparkPrefillRoutePackContract:
    rank: int
    arena: DeepseekV4FlashSparkPrefillRoutePackArenaLayout
    variant: str = "flash"
    max_rows: int = DS4_FLASH_PREFILL_MAX_ROWS
    global_top_k: int = DS4_FLASH_GLOBAL_TOP_K
    routed_experts: int = DS4_FLASH_ROUTED_EXPERTS
    route_block_rows: int = DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS
    max_packed_route_slots: int = DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS
    max_route_blocks: int = DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS
    expert_tensor_parallel: int = DS4_EXPERT_TP
    expert_parallel: bool = False
    status: str = "qualified-flash-spark-prefill-route-pack-not-active"

    @property
    def serving_allocates(self) -> bool:
        return False

    @property
    def consumes_identical_global_routes(self) -> bool:
        return True

    @property
    def applies_expert_ownership_map(self) -> bool:
        return False

    @property
    def cuda_graph_kernel_nodes(self) -> int:
        return 3

    @property
    def cuda_graph_memset_nodes(self) -> int:
        return 1

    @property
    def cuda_graph_memcpy_nodes(self) -> int:
        return 0


@dataclass(frozen=True)
class DeepseekV4FlashSparkPrefillRoutePackBinding:
    serving_allocates: ClassVar[bool] = False
    views_only: ClassVar[bool] = True
    cuda_graph_safe: ClassVar[bool] = True
    expert_tensor_parallel: ClassVar[int] = DS4_EXPERT_TP
    expert_parallel: ClassVar[bool] = False

    contract: DeepseekV4FlashSparkPrefillRoutePackContract
    arena: Any
    topk_ids: Any
    packed_route_indices: Any
    block_expert_ids: Any
    packed_route_count: Any
    expert_counts: Any
    expert_offsets: Any

    def region_views(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("topk_ids", self.topk_ids),
            ("packed_route_indices", self.packed_route_indices),
            ("block_expert_ids", self.block_expert_ids),
            ("packed_route_count", self.packed_route_count),
            ("expert_counts", self.expert_counts),
            ("expert_offsets", self.expert_offsets),
        )


def plan_deepseek_v4_flash_spark_rank_decode_m1(
    *, rank: int
) -> DeepseekV4FlashSparkRankDecodeM1Contract:
    rank = int(rank)
    if not 0 <= rank < DS4_EXPERT_TP:
        raise ValueError(f"Flash Spark rank must be in [0, {DS4_EXPERT_TP}), got {rank}")

    regions: list[tuple[str, int, int]] = []
    offset = 0

    def reserve(name: str, nbytes: int) -> tuple[int, int]:
        nonlocal offset
        start = _align_up(offset)
        regions.append((name, start, nbytes))
        offset = start + nbytes
        return start, nbytes

    input_offset, input_bytes = reserve("input", DS4_FLASH_HIDDEN * 2)
    route_indices_offset, route_indices_bytes = reserve(
        "route_indices", DS4_FLASH_GLOBAL_TOP_K * 4
    )
    route_weights_offset, route_weights_bytes = reserve(
        "route_weights", DS4_FLASH_GLOBAL_TOP_K * 4
    )
    fc1_offset, fc1_bytes = reserve(
        "fc1", DS4_FLASH_GLOBAL_TOP_K * 2 * DS4_FLASH_LOCAL_INTERMEDIATE * 2
    )
    activated_offset, activated_bytes = reserve(
        "activated", DS4_FLASH_GLOBAL_TOP_K * DS4_FLASH_LOCAL_INTERMEDIATE * 2
    )
    output_offset, output_bytes = reserve("output", DS4_FLASH_HIDDEN * 2)
    block_expert_ids_offset, block_expert_ids_bytes = reserve("block_expert_ids", 4)
    packed_route_count_offset, packed_route_count_bytes = reserve(
        "packed_route_count", 4
    )
    fc1_scratch_offset, fc1_scratch_bytes = reserve(
        "fc1_scratch", DS4_FLASH_FC1_SCRATCH_ELEMENTS * 4
    )
    fc2_scratch_offset, fc2_scratch_bytes = reserve(
        "fc2_scratch", DS4_FLASH_FC2_SCRATCH_ELEMENTS * 4
    )
    locks_offset, locks_bytes = reserve("locks", DS4_FLASH_LOCK_ELEMENTS * 4)
    return DeepseekV4FlashSparkRankDecodeM1Contract(
        rank=rank,
        arena=DeepseekV4FlashSparkRankDecodeM1ArenaLayout(
            input_offset=input_offset,
            input_bytes=input_bytes,
            route_indices_offset=route_indices_offset,
            route_indices_bytes=route_indices_bytes,
            route_weights_offset=route_weights_offset,
            route_weights_bytes=route_weights_bytes,
            fc1_offset=fc1_offset,
            fc1_bytes=fc1_bytes,
            activated_offset=activated_offset,
            activated_bytes=activated_bytes,
            output_offset=output_offset,
            output_bytes=output_bytes,
            block_expert_ids_offset=block_expert_ids_offset,
            block_expert_ids_bytes=block_expert_ids_bytes,
            packed_route_count_offset=packed_route_count_offset,
            packed_route_count_bytes=packed_route_count_bytes,
            fc1_scratch_offset=fc1_scratch_offset,
            fc1_scratch_bytes=fc1_scratch_bytes,
            fc2_scratch_offset=fc2_scratch_offset,
            fc2_scratch_bytes=fc2_scratch_bytes,
            locks_offset=locks_offset,
            locks_bytes=locks_bytes,
            total_bytes=_align_up(offset),
        ),
    )


def plan_deepseek_v4_flash_spark_prefill_route_pack(
    *, rank: int
) -> DeepseekV4FlashSparkPrefillRoutePackContract:
    rank = int(rank)
    if not 0 <= rank < DS4_EXPERT_TP:
        raise ValueError(
            f"Flash Spark prefill route-pack rank must be in [0, {DS4_EXPERT_TP}), "
            f"got {rank}"
        )

    offset = 0

    def reserve(nbytes: int) -> tuple[int, int]:
        nonlocal offset
        start = _align_up(offset)
        offset = start + nbytes
        return start, nbytes

    topk_ids_offset, topk_ids_bytes = reserve(
        DS4_FLASH_PREFILL_MAX_ROWS * DS4_FLASH_GLOBAL_TOP_K * 4
    )
    packed_route_indices_offset, packed_route_indices_bytes = reserve(
        DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS * 4
    )
    block_expert_ids_offset, block_expert_ids_bytes = reserve(
        DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS * 4
    )
    packed_route_count_offset, packed_route_count_bytes = reserve(4)
    expert_counts_offset, expert_counts_bytes = reserve(DS4_FLASH_ROUTED_EXPERTS * 4)
    expert_offsets_offset, expert_offsets_bytes = reserve(
        (DS4_FLASH_ROUTED_EXPERTS + 1) * 4
    )
    return DeepseekV4FlashSparkPrefillRoutePackContract(
        rank=rank,
        arena=DeepseekV4FlashSparkPrefillRoutePackArenaLayout(
            topk_ids_offset=topk_ids_offset,
            topk_ids_bytes=topk_ids_bytes,
            packed_route_indices_offset=packed_route_indices_offset,
            packed_route_indices_bytes=packed_route_indices_bytes,
            block_expert_ids_offset=block_expert_ids_offset,
            block_expert_ids_bytes=block_expert_ids_bytes,
            packed_route_count_offset=packed_route_count_offset,
            packed_route_count_bytes=packed_route_count_bytes,
            expert_counts_offset=expert_counts_offset,
            expert_counts_bytes=expert_counts_bytes,
            expert_offsets_offset=expert_offsets_offset,
            expert_offsets_bytes=expert_offsets_bytes,
            total_bytes=_align_up(offset),
        ),
    )


def bind_deepseek_v4_flash_spark_rank_decode_m1(
    contract: DeepseekV4FlashSparkRankDecodeM1Contract,
    *,
    arena: Any,
) -> DeepseekV4FlashSparkRankDecodeM1Binding:
    import torch

    if not isinstance(contract, DeepseekV4FlashSparkRankDecodeM1Contract):
        raise TypeError("Flash Spark decode M1 arena requires its planned contract")
    if not isinstance(arena, torch.Tensor) or arena.ndim != 1:
        raise TypeError("Flash Spark decode M1 arena must be a rank-1 tensor")
    if arena.dtype != torch.uint8 or not arena.is_contiguous():
        raise ValueError("Flash Spark decode M1 arena must be contiguous torch.uint8")
    if arena.numel() < contract.arena.total_bytes:
        raise ValueError(
            "Flash Spark decode M1 arena is too small: "
            f"need={contract.arena.total_bytes}, have={arena.numel()}"
        )
    arena = arena.narrow(0, 0, contract.arena.total_bytes)
    layout = contract.arena

    def view(offset: int, nbytes: int, *, dtype: Any, shape: tuple[int, ...]):
        return arena.narrow(0, offset, nbytes).view(dtype).view(shape)

    return DeepseekV4FlashSparkRankDecodeM1Binding(
        contract=contract,
        arena=arena,
        input=view(
            layout.input_offset,
            layout.input_bytes,
            dtype=torch.bfloat16,
            shape=(1, DS4_FLASH_HIDDEN),
        ),
        route_indices=view(
            layout.route_indices_offset,
            layout.route_indices_bytes,
            dtype=torch.int32,
            shape=(1, DS4_FLASH_GLOBAL_TOP_K),
        ),
        route_weights=view(
            layout.route_weights_offset,
            layout.route_weights_bytes,
            dtype=torch.float32,
            shape=(1, DS4_FLASH_GLOBAL_TOP_K),
        ),
        fc1=view(
            layout.fc1_offset,
            layout.fc1_bytes,
            dtype=torch.bfloat16,
            shape=(DS4_FLASH_GLOBAL_TOP_K, 2 * DS4_FLASH_LOCAL_INTERMEDIATE),
        ),
        activated=view(
            layout.activated_offset,
            layout.activated_bytes,
            dtype=torch.bfloat16,
            shape=(DS4_FLASH_GLOBAL_TOP_K, DS4_FLASH_LOCAL_INTERMEDIATE),
        ),
        output=view(
            layout.output_offset,
            layout.output_bytes,
            dtype=torch.bfloat16,
            shape=(1, DS4_FLASH_HIDDEN),
        ),
        block_expert_ids=view(
            layout.block_expert_ids_offset,
            layout.block_expert_ids_bytes,
            dtype=torch.int32,
            shape=(1,),
        ),
        packed_route_count=view(
            layout.packed_route_count_offset,
            layout.packed_route_count_bytes,
            dtype=torch.int32,
            shape=(1,),
        ),
        fc1_scratch=view(
            layout.fc1_scratch_offset,
            layout.fc1_scratch_bytes,
            dtype=torch.float32,
            shape=(DS4_FLASH_FC1_SCRATCH_ELEMENTS,),
        ),
        fc2_scratch=view(
            layout.fc2_scratch_offset,
            layout.fc2_scratch_bytes,
            dtype=torch.float32,
            shape=(DS4_FLASH_FC2_SCRATCH_ELEMENTS,),
        ),
        locks=view(
            layout.locks_offset,
            layout.locks_bytes,
            dtype=torch.int32,
            shape=(DS4_FLASH_LOCK_ELEMENTS,),
        ),
    )


def bind_deepseek_v4_flash_spark_prefill_route_pack(
    contract: DeepseekV4FlashSparkPrefillRoutePackContract,
    *,
    arena: Any,
) -> DeepseekV4FlashSparkPrefillRoutePackBinding:
    import torch

    if not isinstance(contract, DeepseekV4FlashSparkPrefillRoutePackContract):
        raise TypeError("Flash Spark prefill route-pack arena requires its contract")
    if not isinstance(arena, torch.Tensor) or arena.ndim != 1:
        raise TypeError("Flash Spark prefill route-pack arena must be rank-1")
    if arena.dtype != torch.uint8 or not arena.is_contiguous():
        raise ValueError(
            "Flash Spark prefill route-pack arena must be contiguous torch.uint8"
        )
    if arena.numel() < contract.arena.total_bytes:
        raise ValueError(
            "Flash Spark prefill route-pack arena is too small: "
            f"need={contract.arena.total_bytes}, have={arena.numel()}"
        )
    arena = arena.narrow(0, 0, contract.arena.total_bytes)
    layout = contract.arena

    def i32_view(offset: int, nbytes: int, shape: tuple[int, ...]):
        return arena.narrow(0, offset, nbytes).view(torch.int32).view(shape)

    return DeepseekV4FlashSparkPrefillRoutePackBinding(
        contract=contract,
        arena=arena,
        topk_ids=i32_view(
            layout.topk_ids_offset,
            layout.topk_ids_bytes,
            (DS4_FLASH_PREFILL_MAX_ROWS, DS4_FLASH_GLOBAL_TOP_K),
        ),
        packed_route_indices=i32_view(
            layout.packed_route_indices_offset,
            layout.packed_route_indices_bytes,
            (DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS,),
        ),
        block_expert_ids=i32_view(
            layout.block_expert_ids_offset,
            layout.block_expert_ids_bytes,
            (DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS,),
        ),
        packed_route_count=i32_view(
            layout.packed_route_count_offset,
            layout.packed_route_count_bytes,
            (1,),
        ),
        expert_counts=i32_view(
            layout.expert_counts_offset,
            layout.expert_counts_bytes,
            (DS4_FLASH_ROUTED_EXPERTS,),
        ),
        expert_offsets=i32_view(
            layout.expert_offsets_offset,
            layout.expert_offsets_bytes,
            (DS4_FLASH_ROUTED_EXPERTS + 1,),
        ),
    )


def qualify_deepseek_v4_flash_spark_rank_decode_m1_contract() -> bool:
    contracts = tuple(
        plan_deepseek_v4_flash_spark_rank_decode_m1(rank=rank)
        for rank in range(DS4_EXPERT_TP)
    )
    first = contracts[0]
    for contract in contracts:
        if (
            contract.rank not in range(DS4_EXPERT_TP)
            or contract.variant != "flash"
            or contract.rows != 1
            or contract.hidden != DS4_FLASH_HIDDEN
            or contract.routed_experts != DS4_FLASH_ROUTED_EXPERTS
            or contract.global_top_k != DS4_FLASH_GLOBAL_TOP_K
            or contract.expert_tensor_parallel != DS4_EXPERT_TP
            or contract.expert_parallel
            or contract.local_intermediate != DS4_FLASH_LOCAL_INTERMEDIATE
            or contract.serving_allocates
            or not contract.resident_weights_are_external
            or contract.resident_weight_bytes != DS4_FLASH_SPARK_RESIDENT_WEIGHT_BYTES
            or not contract.owns_same_expert_ids_as_every_rank
            or not contract.consumes_identical_global_routes
            or contract.requires_host_route_pack
            or not contract.applies_route_weights_locally
            or contract.output_shape != (1, DS4_FLASH_HIDDEN)
            or contract.output_dtype != "bfloat16"
            or not contract.output_is_rank_partial
            or not contract.requires_coordinator_reduction
            or contract.cuda_graph_memcpy_nodes != 0
            or contract.arena != first.arena
        ):
            raise RuntimeError("Flash Spark decode M1 lost the fixed TP=4 rank contract")
        previous_end = 0
        for name, offset, nbytes in contract.arena.regions():
            if offset % DS4_LAYER_ARENA_ALIGNMENT:
                raise RuntimeError(f"Flash Spark rank arena region {name} lost alignment")
            if offset < previous_end:
                raise RuntimeError(f"Flash Spark rank arena region {name} overlaps")
            previous_end = offset + nbytes
        if _align_up(previous_end) != contract.arena.total_bytes:
            raise RuntimeError("Flash Spark rank arena total drifted")

    import torch

    arena = torch.empty((first.arena.total_bytes,), dtype=torch.uint8)
    binding = bind_deepseek_v4_flash_spark_rank_decode_m1(first, arena=arena)
    offsets = {name: offset for name, offset, _ in first.arena.regions()}
    for name, tensor in binding.region_views():
        if tensor.data_ptr() != arena.data_ptr() + offsets[name]:
            raise RuntimeError(f"Flash Spark rank arena view {name} lost its offset")
    if (
        DeepseekV4FlashSparkRankDecodeM1Binding.serving_allocates
        or not DeepseekV4FlashSparkRankDecodeM1Binding.views_only
        or not DeepseekV4FlashSparkRankDecodeM1Binding.cuda_graph_safe
        or DeepseekV4FlashSparkRankDecodeM1Binding.expert_tensor_parallel
        != DS4_EXPERT_TP
        or DeepseekV4FlashSparkRankDecodeM1Binding.expert_parallel
    ):
        raise RuntimeError("Flash Spark rank decode M1 binding lifecycle drifted")
    return True


def qualify_deepseek_v4_flash_spark_prefill_route_pack_contract() -> bool:
    contracts = tuple(
        plan_deepseek_v4_flash_spark_prefill_route_pack(rank=rank)
        for rank in range(DS4_EXPERT_TP)
    )
    first = contracts[0]
    for contract in contracts:
        if (
            contract.rank not in range(DS4_EXPERT_TP)
            or contract.variant != "flash"
            or contract.max_rows != DS4_FLASH_PREFILL_MAX_ROWS
            or contract.global_top_k != DS4_FLASH_GLOBAL_TOP_K
            or contract.routed_experts != DS4_FLASH_ROUTED_EXPERTS
            or contract.route_block_rows != DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS
            or contract.max_packed_route_slots
            != DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS
            or contract.max_route_blocks != DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS
            or contract.expert_tensor_parallel != DS4_EXPERT_TP
            or contract.expert_parallel
            or contract.serving_allocates
            or not contract.consumes_identical_global_routes
            or contract.applies_expert_ownership_map
            or contract.cuda_graph_kernel_nodes != 3
            or contract.cuda_graph_memset_nodes != 1
            or contract.cuda_graph_memcpy_nodes != 0
            or contract.arena.total_bytes != 137_216
            or contract.arena != first.arena
        ):
            raise RuntimeError(
                "Flash Spark prefill route pack lost the replicated TP=4 contract"
            )
        previous_end = 0
        for name, offset, nbytes in contract.arena.regions():
            if offset % DS4_LAYER_ARENA_ALIGNMENT:
                raise RuntimeError(
                    f"Flash Spark prefill route-pack region {name} lost alignment"
                )
            if offset < previous_end:
                raise RuntimeError(
                    f"Flash Spark prefill route-pack region {name} overlaps"
                )
            previous_end = offset + nbytes
        if _align_up(previous_end) != contract.arena.total_bytes:
            raise RuntimeError("Flash Spark prefill route-pack arena total drifted")

    import torch

    arena = torch.empty((first.arena.total_bytes,), dtype=torch.uint8)
    binding = bind_deepseek_v4_flash_spark_prefill_route_pack(first, arena=arena)
    offsets = {name: offset for name, offset, _ in first.arena.regions()}
    for name, tensor in binding.region_views():
        if tensor.data_ptr() != arena.data_ptr() + offsets[name]:
            raise RuntimeError(
                f"Flash Spark prefill route-pack view {name} lost its offset"
            )
    if (
        DeepseekV4FlashSparkPrefillRoutePackBinding.serving_allocates
        or not DeepseekV4FlashSparkPrefillRoutePackBinding.views_only
        or not DeepseekV4FlashSparkPrefillRoutePackBinding.cuda_graph_safe
        or DeepseekV4FlashSparkPrefillRoutePackBinding.expert_tensor_parallel
        != DS4_EXPERT_TP
        or DeepseekV4FlashSparkPrefillRoutePackBinding.expert_parallel
    ):
        raise RuntimeError("Flash Spark prefill route-pack lifecycle drifted")
    return True


def _align_up(value: int) -> int:
    return (
        (int(value) + DS4_LAYER_ARENA_ALIGNMENT - 1)
        // DS4_LAYER_ARENA_ALIGNMENT
        * DS4_LAYER_ARENA_ALIGNMENT
    )


__all__ = [
    "DeepseekV4FlashSparkPrefillRoutePackArenaLayout",
    "DeepseekV4FlashSparkPrefillRoutePackBinding",
    "DeepseekV4FlashSparkPrefillRoutePackContract",
    "DeepseekV4FlashSparkRankDecodeM1ArenaLayout",
    "DeepseekV4FlashSparkRankDecodeM1Binding",
    "DeepseekV4FlashSparkRankDecodeM1Contract",
    "bind_deepseek_v4_flash_spark_prefill_route_pack",
    "bind_deepseek_v4_flash_spark_rank_decode_m1",
    "plan_deepseek_v4_flash_spark_prefill_route_pack",
    "plan_deepseek_v4_flash_spark_rank_decode_m1",
    "qualify_deepseek_v4_flash_spark_prefill_route_pack_contract",
    "qualify_deepseek_v4_flash_spark_rank_decode_m1_contract",
]
