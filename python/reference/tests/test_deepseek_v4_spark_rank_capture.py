import pytest
import torch

from ds41rt_reference.deepseek_v4_spark_rank_capture import (
    DS4_FLASH_SPARK_RESIDENT_WEIGHT_BYTES,
    DeepseekV4FlashSparkPrefillRoutePackBinding,
    DeepseekV4FlashSparkRankDecodeM1Binding,
    bind_deepseek_v4_flash_spark_prefill_route_pack,
    bind_deepseek_v4_flash_spark_rank_decode_m1,
    plan_deepseek_v4_flash_spark_prefill_route_pack,
    plan_deepseek_v4_flash_spark_rank_decode_m1,
    qualify_deepseek_v4_flash_spark_prefill_route_pack_contract,
    qualify_deepseek_v4_flash_spark_rank_decode_m1_contract,
)


def test_flash_decode_m1_all_four_ranks_have_identical_tp_workspace() -> None:
    contracts = [
        plan_deepseek_v4_flash_spark_rank_decode_m1(rank=rank) for rank in range(4)
    ]

    assert [contract.rank for contract in contracts] == [0, 1, 2, 3]
    assert all(contract.arena == contracts[0].arena for contract in contracts)
    assert contracts[0].arena.total_bytes == 2_006_016
    for contract in contracts:
        assert contract.resident_weight_bytes == DS4_FLASH_SPARK_RESIDENT_WEIGHT_BYTES
        assert contract.routed_experts == 256
        assert contract.global_top_k == 6
        assert contract.expert_tensor_parallel == 4
        assert not contract.expert_parallel
        assert contract.local_intermediate == 512
        assert contract.owns_same_expert_ids_as_every_rank
        assert contract.consumes_identical_global_routes
        assert not contract.requires_host_route_pack
        assert contract.applies_route_weights_locally
        assert contract.output_shape == (1, 4_096)
        assert contract.output_dtype == "bfloat16"
        assert contract.output_is_rank_partial
        assert contract.requires_coordinator_reduction
        assert contract.cuda_graph_memcpy_nodes == 0
        assert contract.status == "qualified-flash-spark-rank-decode-m1-not-active"


def test_flash_decode_m1_rank_arena_binds_native_caller_owned_buffers() -> None:
    contract = plan_deepseek_v4_flash_spark_rank_decode_m1(rank=2)
    arena = torch.empty((contract.arena.total_bytes + 17,), dtype=torch.uint8)
    binding = bind_deepseek_v4_flash_spark_rank_decode_m1(contract, arena=arena)

    assert binding.arena.numel() == contract.arena.total_bytes
    assert binding.input.shape == (1, 4_096)
    assert binding.input.dtype == torch.bfloat16
    assert binding.route_indices.shape == (1, 6)
    assert binding.route_indices.dtype == torch.int32
    assert binding.route_weights.shape == (1, 6)
    assert binding.route_weights.dtype == torch.float32
    assert binding.fc1.shape == (6, 1_024)
    assert binding.fc1.dtype == torch.bfloat16
    assert binding.activated.shape == (6, 512)
    assert binding.activated.dtype == torch.bfloat16
    assert binding.output.shape == (1, 4_096)
    assert binding.output.dtype == torch.bfloat16
    assert binding.fc1_scratch.shape == (98_304,)
    assert binding.fc2_scratch.shape == (393_216,)
    assert binding.locks.shape == (194,)
    offsets = {name: offset for name, offset, _ in contract.arena.regions()}
    for name, tensor in binding.region_views():
        assert tensor.data_ptr() == arena.data_ptr() + offsets[name]


def test_flash_decode_m1_rank_contract_is_startup_qualified() -> None:
    assert qualify_deepseek_v4_flash_spark_rank_decode_m1_contract()
    assert not DeepseekV4FlashSparkRankDecodeM1Binding.serving_allocates
    assert DeepseekV4FlashSparkRankDecodeM1Binding.views_only
    assert DeepseekV4FlashSparkRankDecodeM1Binding.cuda_graph_safe
    assert DeepseekV4FlashSparkRankDecodeM1Binding.expert_tensor_parallel == 4
    assert not DeepseekV4FlashSparkRankDecodeM1Binding.expert_parallel


@pytest.mark.parametrize("rank", [-1, 4])
def test_flash_decode_m1_rank_contract_rejects_non_tp4_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="rank must be"):
        plan_deepseek_v4_flash_spark_rank_decode_m1(rank=rank)


def test_flash_decode_m1_rank_arena_rejects_short_storage() -> None:
    contract = plan_deepseek_v4_flash_spark_rank_decode_m1(rank=0)
    with pytest.raises(ValueError, match="too small"):
        bind_deepseek_v4_flash_spark_rank_decode_m1(
            contract,
            arena=torch.empty((contract.arena.total_bytes - 1,), dtype=torch.uint8),
        )


def test_flash_prefill_route_pack_is_identical_on_all_four_tp_ranks() -> None:
    contracts = [
        plan_deepseek_v4_flash_spark_prefill_route_pack(rank=rank)
        for rank in range(4)
    ]

    assert [contract.rank for contract in contracts] == [0, 1, 2, 3]
    assert all(contract.arena == contracts[0].arena for contract in contracts)
    assert contracts[0].arena.total_bytes == 137_216
    for contract in contracts:
        assert contract.max_rows == 2_048
        assert contract.global_top_k == 6
        assert contract.routed_experts == 256
        assert contract.route_block_rows == 32
        assert contract.max_packed_route_slots == 20_224
        assert contract.max_route_blocks == 632
        assert contract.expert_tensor_parallel == 4
        assert not contract.expert_parallel
        assert contract.consumes_identical_global_routes
        assert not contract.applies_expert_ownership_map
        assert contract.cuda_graph_kernel_nodes == 3
        assert contract.cuda_graph_memset_nodes == 1
        assert contract.cuda_graph_memcpy_nodes == 0


def test_flash_prefill_route_pack_arena_binds_native_caller_owned_buffers() -> None:
    contract = plan_deepseek_v4_flash_spark_prefill_route_pack(rank=1)
    arena = torch.empty((contract.arena.total_bytes,), dtype=torch.uint8)
    binding = bind_deepseek_v4_flash_spark_prefill_route_pack(
        contract, arena=arena
    )

    assert binding.topk_ids.shape == (2_048, 6)
    assert binding.packed_route_indices.shape == (20_224,)
    assert binding.block_expert_ids.shape == (632,)
    assert binding.packed_route_count.shape == (1,)
    assert binding.expert_counts.shape == (256,)
    assert binding.expert_offsets.shape == (257,)
    assert all(tensor.dtype == torch.int32 for _, tensor in binding.region_views())
    offsets = {name: offset for name, offset, _ in contract.arena.regions()}
    for name, tensor in binding.region_views():
        assert tensor.data_ptr() == arena.data_ptr() + offsets[name]


def test_flash_prefill_route_pack_contract_is_startup_qualified() -> None:
    assert qualify_deepseek_v4_flash_spark_prefill_route_pack_contract()
    assert not DeepseekV4FlashSparkPrefillRoutePackBinding.serving_allocates
    assert DeepseekV4FlashSparkPrefillRoutePackBinding.views_only
    assert DeepseekV4FlashSparkPrefillRoutePackBinding.cuda_graph_safe
    assert DeepseekV4FlashSparkPrefillRoutePackBinding.expert_tensor_parallel == 4
    assert not DeepseekV4FlashSparkPrefillRoutePackBinding.expert_parallel


def test_flash_prefill_route_pack_rejects_non_tp4_rank() -> None:
    with pytest.raises(ValueError, match="rank must be"):
        plan_deepseek_v4_flash_spark_prefill_route_pack(rank=4)
