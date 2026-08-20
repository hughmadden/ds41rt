import pytest
import torch

from ds4rt_reference.deepseek_v4_spark_prefill_capture import (
    DeepseekV4FlashSparkPrefillBinding,
    bind_deepseek_v4_flash_spark_prefill,
    plan_deepseek_v4_flash_spark_prefill,
    qualify_deepseek_v4_flash_spark_prefill_contract,
)


def test_flash_prefill_all_four_ranks_have_identical_max_capacity_tp_workspace() -> None:
    contracts = [
        plan_deepseek_v4_flash_spark_prefill(rank=rank) for rank in range(4)
    ]

    assert [contract.rank for contract in contracts] == [0, 1, 2, 3]
    assert all(contract.arena == contracts[0].arena for contract in contracts)
    assert contracts[0].arena.total_bytes == 197_319_680
    for contract in contracts:
        assert contract.max_rows == 2_048
        assert contract.hidden == 4_096
        assert contract.routed_experts == 256
        assert contract.global_top_k == 6
        assert contract.expert_tensor_parallel == 4
        assert not contract.expert_parallel
        assert contract.local_intermediate == 512
        assert contract.owns_same_expert_ids_as_every_rank
        assert contract.consumes_identical_global_routes
        assert not contract.applies_expert_ownership_map
        assert contract.applies_route_weights_locally
        assert contract.output_capacity_shape == (2_048, 4_096)
        assert contract.output_is_rank_partial
        assert contract.requires_coordinator_reduction
        assert contract.minimum_cuda_graph_kernel_nodes == 5
        assert contract.minimum_cuda_graph_memset_nodes == 2
        assert contract.cuda_graph_memcpy_nodes == 0


def test_flash_prefill_arena_regions_match_native_fixed_buffers() -> None:
    contract = plan_deepseek_v4_flash_spark_prefill(rank=0)
    regions = {region.name: region for region in contract.arena.regions}

    assert regions["input"].shape == (2_048, 4_096)
    assert regions["input"].nbytes == 16_777_216
    assert regions["topk_ids"].shape == (2_048, 6)
    assert regions["topk_ids"].nbytes == 49_152
    assert regions["topk_weights"].shape == (2_048, 6)
    assert regions["packed_route_indices"].shape == (20_224,)
    assert regions["block_expert_ids"].shape == (632,)
    assert regions["expert_counts"].shape == (256,)
    assert regions["expert_offsets"].shape == (257,)
    assert regions["fc1"].shape == (12_288, 1_024)
    assert regions["activated"].shape == (12_288, 512)
    assert regions["routed_output"].shape == (12_288, 4_096)
    assert regions["routed_output"].nbytes == 100_663_296
    assert regions["output"].shape == (2_048, 4_096)
    assert regions["fc1_scratch"].shape == (3_145_728,)
    assert regions["fc2_scratch"].shape == (3_145_728,)
    assert regions["locks"].shape == (194,)


def test_flash_prefill_arena_binds_all_native_caller_owned_views() -> None:
    contract = plan_deepseek_v4_flash_spark_prefill(rank=3)
    arena = torch.empty((contract.arena.total_bytes,), dtype=torch.uint8)
    binding = bind_deepseek_v4_flash_spark_prefill(contract, arena=arena)

    assert binding.arena.numel() == 197_319_680
    for region, (name, tensor) in zip(
        contract.arena.regions, binding.region_views(), strict=True
    ):
        assert name == region.name
        assert tuple(tensor.shape) == region.shape
        assert tensor.data_ptr() == arena.data_ptr() + region.offset


def test_flash_prefill_contract_is_startup_qualified() -> None:
    assert qualify_deepseek_v4_flash_spark_prefill_contract()
    assert not DeepseekV4FlashSparkPrefillBinding.serving_allocates
    assert DeepseekV4FlashSparkPrefillBinding.views_only
    assert DeepseekV4FlashSparkPrefillBinding.cuda_graph_safe
    assert DeepseekV4FlashSparkPrefillBinding.expert_tensor_parallel == 4
    assert not DeepseekV4FlashSparkPrefillBinding.expert_parallel


@pytest.mark.parametrize("rank", [-1, 4])
def test_flash_prefill_contract_rejects_non_tp4_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="rank must be"):
        plan_deepseek_v4_flash_spark_prefill(rank=rank)
