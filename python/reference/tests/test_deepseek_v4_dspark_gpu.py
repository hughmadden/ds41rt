import os

import pytest
import torch

from ds41rt_reference.deepseek_v4_dspark_capture import (
    bind_deepseek_v4_dspark_proposal_entry,
    capture_deepseek_v4_dspark_block_attention,
    capture_deepseek_v4_dspark_block_post_dispatch,
    capture_deepseek_v4_dspark_terminal_collapse,
    plan_deepseek_v4_dspark,
    prepare_deepseek_v4_dspark_block_attention,
    prepare_deepseek_v4_dspark_block_post_dispatch,
    prepare_deepseek_v4_dspark_terminal_collapse,
    run_deepseek_v4_dspark_proposal_entry,
)
from ds41rt_reference.deepseek_v4_mhc_capture import plan_deepseek_v4_mhc


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for dSpark graph tests"
)


def test_flash_proposal_entry_replays_block_zero_mhc_pre_on_gpu0() -> None:
    device = torch.device("cuda", 0)
    contract = plan_deepseek_v4_dspark(
        variant="flash", max_batch=1, max_main_rows=1
    )
    rows = contract.geometry.proposal_tokens
    hidden = contract.geometry.hidden
    scratch_bytes = plan_deepseek_v4_mhc(
        variant="flash", max_rows=rows
    ).scratch.total_bytes
    generator = torch.Generator(device=device).manual_seed(7)
    residual_input = torch.randn(
        (rows, hidden), device=device, dtype=torch.bfloat16, generator=generator
    )
    residual_output = torch.empty(
        (rows, 4, hidden), device=device, dtype=torch.bfloat16
    )
    normalized_output = torch.empty(
        (rows, hidden), device=device, dtype=torch.bfloat16
    )
    post_output = torch.empty((rows, 4), device=device, dtype=torch.float32)
    comb_output = torch.empty((rows, 4, 4), device=device, dtype=torch.float32)
    binding = bind_deepseek_v4_dspark_proposal_entry(
        contract,
        scratch=torch.empty((scratch_bytes,), device=device, dtype=torch.uint8),
        residual_input=residual_input,
        residual_output=residual_output,
        normalized_output=normalized_output,
        post_output=post_output,
        comb_output=comb_output,
        hc_fn=torch.zeros((24, hidden), device=device, dtype=torch.float32),
        hc_scale=torch.ones((3,), device=device, dtype=torch.float32),
        hc_base=torch.zeros((24,), device=device, dtype=torch.float32),
        norm_weight=torch.ones((hidden,), device=device, dtype=torch.bfloat16),
    )

    run_deepseek_v4_dspark_proposal_entry(binding)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_deepseek_v4_dspark_proposal_entry(binding)
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(
        residual_output,
        residual_input.unsqueeze(1).expand(-1, 4, -1),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(normalized_output).all()
    assert torch.isfinite(post_output).all()
    assert torch.isfinite(comb_output).all()


def _device_buffer(tensor: torch.Tensor) -> dict[str, int]:
    return {
        "ptr": tensor.data_ptr(),
        "bytes": tensor.numel() * tensor.element_size(),
        "device_id": tensor.device.index,
    }


def test_flash_terminal_collapse_retains_raw_and_normalized_hidden() -> None:
    device = torch.device("cuda", 0)
    contract = plan_deepseek_v4_dspark(
        variant="flash", max_batch=1, max_main_rows=1
    )
    rows, hidden = 5, 4_096
    workspace = torch.empty(
        (contract.arena.region("reused_sparse_block_workspace").nbytes,),
        device=device,
        dtype=torch.uint8,
    )
    block_arena = contract.blocks[2].expert_handoff.arena
    ffn_delta = (
        workspace.narrow(0, block_arena.ffn_delta_offset, rows * hidden * 2)
        .view(torch.bfloat16)
        .view(rows, hidden)
    )
    ffn_delta.copy_(torch.randn_like(ffn_delta))
    residual = torch.randn(
        (rows, 4, hidden), device=device, dtype=torch.bfloat16
    )
    prev_post = torch.randn((rows, 4), device=device, dtype=torch.float32)
    prev_comb = torch.randn((rows, 4, 4), device=device, dtype=torch.float32)
    terminal_residual = torch.empty_like(residual)
    collapsed = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)
    normalized = torch.empty_like(collapsed)
    hc_fn = torch.randn(
        (4, 4 * hidden), device=device, dtype=torch.float32
    ) / 128
    tensors = {
        "workspace": workspace,
        "ffn_delta": ffn_delta,
        "residual": residual,
        "prev_post": prev_post,
        "prev_comb": prev_comb,
        "terminal_residual": terminal_residual,
        "collapsed_hidden": collapsed,
        "normalized_hidden": normalized,
        "hc_head_fn": hc_fn,
        "hc_head_scale": torch.tensor([0.8], device=device),
        "hc_head_base": torch.linspace(-0.2, 0.2, 4, device=device),
        "norm_weight": torch.linspace(
            0.8, 1.2, hidden, device=device, dtype=torch.float32
        ).to(torch.bfloat16),
    }
    source_ffn_delta = ffn_delta.clone()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    ctx = {
        "buffers": {name: _device_buffer(value) for name, value in tensors.items()},
        "cuda_stream": stream.cuda_stream,
    }
    kwargs = {"variant": "flash", "max_batch": 1, "max_main_rows": 1}
    prepare_deepseek_v4_dspark_terminal_collapse(ctx, **kwargs)
    stream.synchronize()
    eager_terminal = terminal_residual.clone()
    eager_collapsed = collapsed.clone()
    eager_normalized = normalized.clone()

    ffn_delta.copy_(source_ffn_delta)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        capture_deepseek_v4_dspark_terminal_collapse(ctx, **kwargs)
    ffn_delta.copy_(source_ffn_delta)
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(terminal_residual, eager_terminal, rtol=0, atol=0)
    torch.testing.assert_close(collapsed, eager_collapsed, rtol=0, atol=0)
    torch.testing.assert_close(normalized, eager_normalized, rtol=0, atol=0)
    assert not torch.equal(collapsed, normalized)

    residual.add_(torch.tensor(0.125, device=device, dtype=torch.bfloat16))
    ffn_delta.copy_(source_ffn_delta)
    graph.replay()
    torch.cuda.synchronize(device)
    assert not torch.equal(terminal_residual, eager_terminal)
    assert torch.isfinite(collapsed).all()
    assert torch.isfinite(normalized).all()


@pytest.mark.parametrize(("max_batch", "request_slot"), [(1, 0), (16, 1)])
@pytest.mark.skipif(
    os.environ.get("DS41RT_RUN_GPU_QUALIFICATION") != "1",
    reason="set DS41RT_RUN_GPU_QUALIFICATION=1 for full block capture",
)
def test_flash_block_capture_replays_across_tp4_dispatch_boundary(
    max_batch: int, request_slot: int
) -> None:
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("dSpark attention graph qualification requires SM120 GPU0")
    contract = plan_deepseek_v4_dspark(
        variant="flash", max_batch=max_batch, max_main_rows=1
    )
    rows, hidden, heads, q_rank = 5, 4_096, 64, 1_024
    groups, output_rank, group_width = 8, 1_024, 4_096
    query_width = heads * 512

    def fp8_weight(shape: tuple[int, int]) -> torch.Tensor:
        return torch.zeros(shape, device=device, dtype=torch.float8_e4m3fn)

    def e8m0_scale(rows_: int, cols_: int) -> torch.Tensor:
        return torch.full(
            ((rows_ + 127) // 128, (cols_ + 127) // 128),
            127,
            device=device,
            dtype=torch.uint8,
        ).view(torch.float8_e8m0fnu)

    workspace = torch.empty(
        (contract.arena.region("reused_sparse_block_workspace").nbytes,),
        device=device,
        dtype=torch.uint8,
    )
    hidden_states = torch.randn(
        (rows, hidden), device=device, dtype=torch.bfloat16
    )
    positions = torch.arange(1, rows + 1, device=device, dtype=torch.int32)
    page_base = request_slot * 256
    main_slots = torch.arange(
        page_base + 128,
        page_base + 128 + rows,
        device=device,
        dtype=torch.int32,
    )
    cos_sin = torch.zeros((rows, 64), device=device, dtype=torch.float32)
    cos_sin[:, :32] = 1
    main_kv_cache = torch.zeros(
        (max_batch, 149_760), device=device, dtype=torch.uint8
    )
    selected_indices = (
        torch.arange(page_base, page_base + 133, device=device, dtype=torch.int32)
        .expand(rows, -1)
        .contiguous()
    )
    selected_lengths = torch.full(
        (rows,), 133, device=device, dtype=torch.int32
    )
    residual = torch.randn(
        (rows, 4, hidden), device=device, dtype=torch.bfloat16
    )
    prev_post = torch.zeros((rows, 4), device=device, dtype=torch.float32)
    prev_comb = torch.eye(4, device=device, dtype=torch.float32).expand(
        rows, -1, -1
    ).contiguous()
    residual_out = torch.empty_like(residual)
    post_out = torch.empty_like(prev_post)
    comb_out = torch.empty_like(prev_comb)
    tensors = {
        "workspace": workspace,
        "hidden_states": hidden_states,
        "positions": positions,
        "main_slots": main_slots,
        "cos_sin_cache": cos_sin,
        "main_kv_cache": main_kv_cache,
        "selected_indices": selected_indices,
        "selected_lengths": selected_lengths,
        "residual": residual,
        "prev_post": prev_post,
        "prev_comb": prev_comb,
        "residual_out": residual_out,
        "post_out": post_out,
        "comb_out": comb_out,
        "wq_a_weight": fp8_weight((q_rank, hidden)),
        "wq_a_scale": e8m0_scale(q_rank, hidden),
        "wq_b_weight": fp8_weight((query_width, q_rank)),
        "wq_b_scale": e8m0_scale(query_width, q_rank),
        "wkv_weight": fp8_weight((512, hidden)),
        "wkv_scale": e8m0_scale(512, hidden),
        "q_norm_weight": torch.ones(
            (q_rank,), device=device, dtype=torch.bfloat16
        ),
        "kv_norm_weight": torch.ones(
            (512,), device=device, dtype=torch.bfloat16
        ),
        "wo_a_weight": fp8_weight((groups * output_rank, group_width)),
        "wo_a_scale": e8m0_scale(groups * output_rank, group_width),
        "wo_b_weight": fp8_weight((hidden, groups * output_rank)),
        "wo_b_scale": e8m0_scale(hidden, groups * output_rank),
        "attn_sink": torch.zeros((heads,), device=device, dtype=torch.float32),
        "hc_fn": torch.zeros(
            (24, 4 * hidden), device=device, dtype=torch.float32
        ),
        "hc_scale": torch.ones((3,), device=device, dtype=torch.float32),
        "hc_base": torch.zeros((24,), device=device, dtype=torch.float32),
        "norm_weight": torch.ones(
            (hidden,), device=device, dtype=torch.bfloat16
        ),
    }
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    ctx = {
        "buffers": {name: _device_buffer(tensor) for name, tensor in tensors.items()},
        "cuda_stream": capture_stream.cuda_stream,
    }
    kwargs = {
        "variant": "flash",
        "block_index": 0,
        "max_batch": max_batch,
        "max_main_rows": 1,
    }
    prepare_deepseek_v4_dspark_block_attention(ctx, **kwargs)
    capture_stream.synchronize()
    eager_hidden = hidden_states.clone()
    eager_residual = residual_out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        capture_deepseek_v4_dspark_block_attention(ctx, **kwargs)
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(hidden_states, eager_hidden, rtol=0, atol=0)
    torch.testing.assert_close(residual_out, eager_residual, rtol=0, atol=0)
    assert torch.isfinite(hidden_states).all()
    assert torch.isfinite(post_out).all()
    assert torch.isfinite(comb_out).all()

    block_arena = contract.blocks[0].expert_handoff.arena
    ffn_delta_bytes = rows * hidden * 2
    ffn_delta = (
        workspace.narrow(
            0, block_arena.ffn_delta_offset, ffn_delta_bytes
        )
        .view(torch.bfloat16)
        .view(rows, hidden)
    )
    ffn_delta.copy_(
        torch.randn((rows, hidden), device=device, dtype=torch.bfloat16)
    )
    post_tensors = {
        "workspace": workspace,
        "ffn_delta": ffn_delta,
        "residual": residual_out,
        "prev_post": post_out,
        "prev_comb": comb_out,
        "residual_out": residual,
        "hidden_states": hidden_states,
        "post_out": prev_post,
        "comb_out": prev_comb,
        "next_hc_fn": torch.zeros(
            (24, 4 * hidden), device=device, dtype=torch.float32
        ),
        "next_hc_scale": torch.ones(
            (3,), device=device, dtype=torch.float32
        ),
        "next_hc_base": torch.zeros(
            (24,), device=device, dtype=torch.float32
        ),
        "next_norm_weight": torch.ones(
            (hidden,), device=device, dtype=torch.bfloat16
        ),
    }
    post_ctx = {
        "buffers": {
            name: _device_buffer(tensor) for name, tensor in post_tensors.items()
        },
        "cuda_stream": capture_stream.cuda_stream,
    }
    prepare_deepseek_v4_dspark_block_post_dispatch(post_ctx, **kwargs)
    capture_stream.synchronize()
    eager_next_hidden = hidden_states.clone()
    eager_next_residual = residual.clone()
    post_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(post_graph, stream=capture_stream):
        capture_deepseek_v4_dspark_block_post_dispatch(post_ctx, **kwargs)
    post_graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(hidden_states, eager_next_hidden, rtol=0, atol=0)
    torch.testing.assert_close(residual, eager_next_residual, rtol=0, atol=0)
    assert torch.isfinite(prev_post).all()
    assert torch.isfinite(prev_comb).all()
