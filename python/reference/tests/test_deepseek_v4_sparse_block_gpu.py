from __future__ import annotations

import os

import pytest
import torch
from ds41rt_reference.deepseek_v4_attention_layer_capture import (
    bind_deepseek_v4_sliding_attention_layer,
    run_deepseek_v4_sliding_attention_layer,
)
from ds41rt_reference.deepseek_v4_sparse_block_capture import (
    bind_deepseek_v4_sparse_block,
    bind_deepseek_v4_sparse_block_arena,
    plan_deepseek_v4_sparse_block,
    run_deepseek_v4_sparse_block_attention,
    run_deepseek_v4_sparse_block_post_dispatch,
)
from b12x.attention import dsv4_producer
from b12x.gemm import wo_projection

pytestmark = pytest.mark.skipif(
    os.environ.get("DS41RT_RUN_GPU_QUALIFICATION") != "1",
    reason="set DS41RT_RUN_GPU_QUALIFICATION=1 for the explicit GPU0 qualification",
)


def _block_fp8_weight(
    rows: int,
    cols: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (torch.randn((rows, cols), device=device, dtype=torch.bfloat16) / 32).to(
        torch.float8_e4m3fn
    )
    scale = torch.full(
        ((rows + 127) // 128, (cols + 127) // 128),
        127,
        device=device,
        dtype=torch.uint8,
    ).view(torch.float8_e8m0fnu)
    return weight, scale


def _flash_attention_weights(device: torch.device):
    hidden, q_rank, heads = 4_096, 1_024, 64
    wq_a, wq_a_scale = _block_fp8_weight(q_rank, hidden, device=device)
    wq_b, wq_b_scale = _block_fp8_weight(heads * 512, q_rank, device=device)
    wkv, wkv_scale = _block_fp8_weight(512, hidden, device=device)
    producer_weights = dsv4_producer.pack_weights(
        wq_a,
        wq_a_scale,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        torch.ones((q_rank,), device=device, dtype=torch.bfloat16),
        torch.ones((512,), device=device, dtype=torch.bfloat16),
    )
    groups, group_width, rank = 8, 4_096, 1_024
    wo_a, wo_a_scale = _block_fp8_weight(
        groups * rank,
        group_width,
        device=device,
    )
    wo_b, wo_b_scale = _block_fp8_weight(
        hidden,
        groups * rank,
        device=device,
    )
    output_weights = wo_projection.pack_weights(
        wo_a,
        wo_a_scale,
        wo_b,
        wo_b_scale,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )
    return producer_weights, output_weights


def test_flash_m1_sparse_block_replays_both_tp4_graph_segments_on_gpu0() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        pytest.skip(f"GPU0 must be SM120, got sm_{capability[0]}{capability[1]}")

    with torch.cuda.device(device):
        torch.manual_seed(20260813)
        hidden, heads, tokens = 4_096, 64, 1
        contract = plan_deepseek_v4_sparse_block(
            variant="flash",
            mode="decode",
            compression=0,
            max_rows=tokens,
            source_pages=1,
            max_positions=1,
        )
        arena = torch.empty(
            (contract.arena.total_bytes,), device=device, dtype=torch.uint8
        )
        block_arena = bind_deepseek_v4_sparse_block_arena(
            contract,
            arena=arena,
            tokens=tokens,
        )
        producer_weights, output_weights = _flash_attention_weights(device)
        hidden_states = (
            torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16) / 4
        ).contiguous()
        positions = torch.zeros((tokens,), device=device, dtype=torch.int32)
        cos_sin = torch.zeros((1, 64), device=device, dtype=torch.float32)
        cos_sin[:, :32] = 1
        main_cache = torch.zeros((1, 149_760), device=device, dtype=torch.uint8)
        swa_indices = torch.zeros((tokens, 128), device=device, dtype=torch.int32)
        swa_lengths = torch.ones((tokens,), device=device, dtype=torch.int32)
        residual = (
            torch.randn((tokens, 4, hidden), device=device, dtype=torch.bfloat16) / 8
        ).contiguous()
        prev_post = torch.zeros((tokens, 4), device=device, dtype=torch.float32)
        prev_comb = (
            torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).contiguous()
        )
        attention_fn = (
            torch.randn((24, 4 * hidden), device=device, dtype=torch.float32) / 128
        ).contiguous()
        attention_hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        attention_hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        ffn_norm_weight = torch.ones(
            (hidden,), device=device, dtype=torch.bfloat16
        )
        attention_residual = torch.empty_like(residual)
        dispatch_hidden = torch.empty_like(hidden_states)
        attention_post = torch.empty_like(prev_post)
        attention_comb = torch.empty_like(prev_comb)
        attention = bind_deepseek_v4_sliding_attention_layer(
            contract.attention,
            arena=block_arena.attention_arena,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=positions,
            cos_sin_cache=cos_sin,
            main_kv_cache=main_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            producer_weights=producer_weights,
            output_weights=output_weights,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=attention_fn,
            hc_scale=attention_hc_scale,
            hc_base=attention_hc_base,
            norm_weight=ffn_norm_weight,
            residual_out=attention_residual,
            y_out=dispatch_hidden,
            post_out=attention_post,
            comb_out=attention_comb,
            attn_sink=torch.zeros((heads,), device=device, dtype=torch.float32),
        )

        block_arena.shared_delta.copy_(
            torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16) / 16
        )
        rank_partials = tuple(
            (
                torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16)
                / 32
            ).contiguous()
            for _ in range(4)
        )
        ffn_fn = (
            torch.randn((24, 4 * hidden), device=device, dtype=torch.float32) / 128
        ).contiguous()
        ffn_hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        ffn_hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        next_norm_weight = torch.ones(
            (hidden,), device=device, dtype=torch.bfloat16
        )
        next_residual = torch.empty_like(residual)
        next_hidden = torch.empty_like(hidden_states)
        next_post = torch.empty_like(prev_post)
        next_comb = torch.empty_like(prev_comb)
        block = bind_deepseek_v4_sparse_block(
            contract,
            arena_binding=block_arena,
            attention_binding=attention,
            attention_runner=run_deepseek_v4_sliding_attention_layer,
            rank_partials=rank_partials,
            ffn_fn=ffn_fn,
            ffn_hc_scale=ffn_hc_scale,
            ffn_hc_base=ffn_hc_base,
            next_norm_weight=next_norm_weight,
            next_residual_out=next_residual,
            next_y_out=next_hidden,
            next_post_out=next_post,
            next_comb_out=next_comb,
        )
        block_arena.route_indices.copy_(
            torch.arange(6, device=device, dtype=torch.int32).view(1, 6)
        )
        block_arena.route_scores.zero_()
        block_arena.route_weights.fill_(1.0 / 6.0)

        assert block.dispatch_hidden.data_ptr() == dispatch_hidden.data_ptr()
        assert block.ffn_mhc_binding.partials.data_ptr() == (
            attention.mhc_binding.partials.data_ptr()
        )
        assert block_arena.route_indices.data_ptr() == (
            arena.data_ptr() + contract.arena.route_indices_offset
        )
        assert block.shared_delta.data_ptr() == (
            arena.data_ptr() + contract.arena.shared_delta_offset
        )
        assert block_arena.shared_gate.data_ptr() == (
            arena.data_ptr() + contract.arena.shared_gate_offset
        )
        attention_outputs = run_deepseek_v4_sparse_block_attention(block)
        next_outputs = run_deepseek_v4_sparse_block_post_dispatch(block)
        torch.cuda.synchronize(device)
        assert tuple(output.data_ptr() for output in attention_outputs) == tuple(
            output.data_ptr()
            for output in (
                attention_residual,
                attention_post,
                attention_comb,
                dispatch_hidden,
            )
        )
        assert tuple(output.data_ptr() for output in next_outputs) == tuple(
            output.data_ptr()
            for output in (next_residual, next_post, next_comb, next_hidden)
        )
        expected_reduction = block_arena.shared_delta.float()
        for partial in rank_partials:
            expected_reduction.add_(partial.float())
        torch.testing.assert_close(
            block_arena.reduction_f32,
            expected_reduction,
            rtol=0,
            atol=0,
        )
        assert torch.equal(block_arena.ffn_delta, expected_reduction.bfloat16())
        eager_dispatch = dispatch_hidden.clone()
        eager_reduction = block_arena.reduction_f32.clone()
        eager = tuple(
            output.clone()
            for output in (next_residual, next_post, next_comb, next_hidden)
        )
        assert all(bool(torch.isfinite(output.float()).all()) for output in eager)

        attention_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(attention_graph):
            graph_attention_outputs = run_deepseek_v4_sparse_block_attention(block)
        post_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(post_graph):
            graph_next_outputs = run_deepseek_v4_sparse_block_post_dispatch(block)
        attention_ptrs = tuple(output.data_ptr() for output in graph_attention_outputs)
        next_ptrs = tuple(output.data_ptr() for output in graph_next_outputs)
        for _ in range(3):
            attention_graph.replay()
            post_graph.replay()
        torch.cuda.synchronize(device)
        assert attention_ptrs == tuple(output.data_ptr() for output in attention_outputs)
        assert next_ptrs == tuple(output.data_ptr() for output in next_outputs)
        assert torch.equal(dispatch_hidden, eager_dispatch)
        assert torch.equal(block_arena.reduction_f32, eager_reduction)
        for actual, expected in zip(
            (next_residual, next_post, next_comb, next_hidden), eager
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        hidden_states.mul_(-0.5).add_(0.01171875)
        residual.mul_(0.625).sub_(0.01953125)
        block_arena.shared_delta.mul_(0.75).add_(0.0078125)
        for rank, partial in enumerate(rank_partials):
            partial.mul_(0.5 + rank / 8).sub_(0.00390625 * (rank + 1))
        attention_graph.replay()
        post_graph.replay()
        torch.cuda.synchronize(device)
        replayed_dispatch = dispatch_hidden.clone()
        replayed_reduction = block_arena.reduction_f32.clone()
        replayed = tuple(
            output.clone()
            for output in (next_residual, next_post, next_comb, next_hidden)
        )
        run_deepseek_v4_sparse_block_attention(block)
        run_deepseek_v4_sparse_block_post_dispatch(block)
        torch.cuda.synchronize(device)
        assert torch.equal(dispatch_hidden, replayed_dispatch)
        assert torch.equal(block_arena.reduction_f32, replayed_reduction)
        for actual, expected in zip(
            replayed,
            (next_residual, next_post, next_comb, next_hidden),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
