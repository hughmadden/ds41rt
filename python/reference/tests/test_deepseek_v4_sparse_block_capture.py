import pytest
import torch
from ds41rt_reference.deepseek_v4_sparse_block_capture import (
    DS4_GLOBAL_TOP_K,
    DeepseekV4SparseBlockBinding,
    bind_deepseek_v4_sparse_block_arena,
    plan_deepseek_v4_sparse_block,
    qualify_deepseek_v4_sparse_block_contract,
)


@pytest.mark.parametrize(
    ("variant", "hidden", "experts", "expected_bytes"),
    [
        ("flash", 4_096, 256, 61_648_896),
        ("pro", 7_168, 384, 210_895_872),
    ],
)
def test_sparse_block_freezes_split_graph_tp4_handoff(
    variant: str,
    hidden: int,
    experts: int,
    expected_bytes: int,
) -> None:
    contract = plan_deepseek_v4_sparse_block(
        variant=variant,
        mode="decode",
        compression=4,
        max_rows=16,
        source_pages=512,
        max_positions=1_048_576,
    )

    assert contract.hidden == hidden
    assert contract.routed_experts == experts
    assert contract.arena.total_bytes == expected_bytes
    assert not contract.serving_allocates
    assert contract.cuda_graph_segments == 2
    assert contract.dispatch_barrier_between_graphs
    assert contract.expert_tensor_parallel == 4
    assert not contract.expert_parallel
    assert contract.expert_global_top_k == DS4_GLOBAL_TOP_K == 6
    assert contract.one_route_buffer_fans_out_to_all_ranks
    assert contract.router_outputs_are_caller_owned
    assert not contract.router_requires_host_readback
    assert contract.router_score_scratch_dtype == "float32"
    assert contract.shared_expert_intermediate == (
        2_048 if variant == "flash" else 3_072
    )
    assert contract.shared_expert_workspace_rows == 8
    assert contract.shared_expert_outputs_are_caller_owned
    assert not contract.shared_expert_requires_host_readback
    assert contract.shared_expert_output_shape == (16, hidden)
    assert contract.shared_expert_output_dtype == "bfloat16"
    assert contract.expert_local_intermediate_fraction == (1, 4)
    assert contract.expert_partial_output_shape == (4, 16, hidden)
    assert contract.expert_partial_dtype == "bfloat16"
    assert contract.expert_route_weights_applied_on_sparks
    assert contract.shared_expert_owner == "coordinator"
    assert contract.reduction_accumulator_dtype == "float32"
    assert contract.reduction_order == (
        "shared",
        "spark-0",
        "spark-1",
        "spark-2",
        "spark-3",
    )
    assert contract.mhc_scratch_reused_across_graph_segments
    assert contract.status == "qualified-split-graph-sparse-block-not-active"


def test_sparse_block_arena_keeps_attention_prefix_and_handoff_views_fixed() -> None:
    contract = plan_deepseek_v4_sparse_block(
        variant="flash",
        mode="decode",
        compression=0,
        max_rows=1,
        source_pages=1,
        max_positions=1,
    )
    arena = torch.empty((contract.arena.total_bytes + 31,), dtype=torch.uint8)
    binding = bind_deepseek_v4_sparse_block_arena(
        contract,
        arena=arena,
        tokens=1,
    )

    assert not binding.serving_allocates
    assert binding.views_only
    assert binding.arena.numel() == contract.arena.total_bytes
    assert binding.attention_arena.numel() == contract.attention.arena.total_bytes
    assert binding.route_indices.shape == (1, 6)
    assert binding.route_indices.dtype == torch.int32
    assert binding.route_scores.shape == (1, 6)
    assert binding.route_scores.dtype == torch.float32
    assert binding.route_weights.shape == (1, 6)
    assert binding.route_weights.dtype == torch.float32
    for workspace in (
        binding.shared_gate,
        binding.shared_up,
        binding.shared_activated,
    ):
        assert workspace.shape == (8, 2_048)
        assert workspace.dtype == torch.bfloat16
    assert binding.shared_delta.shape == (1, 4_096)
    assert binding.shared_delta.dtype == torch.bfloat16
    assert binding.reduction_f32.shape == (1, 4_096)
    assert binding.reduction_f32.dtype == torch.float32
    assert binding.ffn_delta.shape == (1, 4_096)
    assert binding.ffn_delta.dtype == torch.bfloat16
    offsets = {name: offset for name, offset, _ in contract.arena.regions()}
    for name, tensor in binding.region_views():
        assert tensor.data_ptr() == arena.data_ptr() + offsets[name]


@pytest.mark.parametrize(
    ("variant", "expected_bytes"),
    [("flash", 83_351_552), ("pro", 165_277_696)],
)
def test_dspark_width_reuses_exact_sparse_block_workspace(
    variant: str, expected_bytes: int
) -> None:
    contract = plan_deepseek_v4_sparse_block(
        variant=variant,
        mode="decode",
        compression=0,
        max_rows=80,
        source_pages=16,
        max_positions=1_048_576,
        swa_width=133,
    )

    assert contract.attention.swa_width == 133
    assert contract.arena.attention_bytes == contract.attention.arena.total_bytes
    assert contract.arena.total_bytes == expected_bytes


def test_sparse_block_binding_lifecycle_is_startup_qualified() -> None:
    assert qualify_deepseek_v4_sparse_block_contract(
        variant="flash",
        mode="decode",
        compression=0,
        max_rows=1,
        source_pages=1,
        max_positions=1,
    )
    assert not DeepseekV4SparseBlockBinding.serving_allocates
    assert DeepseekV4SparseBlockBinding.cuda_graph_safe
    assert DeepseekV4SparseBlockBinding.cuda_graph_segments == 2
    assert DeepseekV4SparseBlockBinding.dispatch_barrier_between_graphs
    assert DeepseekV4SparseBlockBinding.one_route_buffer_fans_out_to_all_ranks
    assert DeepseekV4SparseBlockBinding.router_outputs_are_caller_owned
    assert not DeepseekV4SparseBlockBinding.router_requires_host_readback
    assert DeepseekV4SparseBlockBinding.shared_expert_outputs_are_caller_owned
    assert not DeepseekV4SparseBlockBinding.shared_expert_requires_host_readback
    assert DeepseekV4SparseBlockBinding.rank_partials_are_hidden_width_bf16
    assert DeepseekV4SparseBlockBinding.reduction_is_fp32
    assert DeepseekV4SparseBlockBinding.expert_tensor_parallel == 4
    assert not DeepseekV4SparseBlockBinding.expert_parallel


@pytest.mark.parametrize(
    ("tokens", "extra_bytes", "match"),
    [
        (0, 0, "tokens must be"),
        (2, 0, "tokens must be"),
        (1, -1, "too small"),
    ],
)
def test_sparse_block_arena_fails_closed(
    tokens: int,
    extra_bytes: int,
    match: str,
) -> None:
    contract = plan_deepseek_v4_sparse_block(
        variant="flash",
        mode="decode",
        compression=0,
        max_rows=1,
        source_pages=1,
        max_positions=1,
    )
    arena = torch.empty(
        (contract.arena.total_bytes + extra_bytes,),
        dtype=torch.uint8,
    )
    with pytest.raises(ValueError, match=match):
        bind_deepseek_v4_sparse_block_arena(
            contract,
            arena=arena,
            tokens=tokens,
        )
