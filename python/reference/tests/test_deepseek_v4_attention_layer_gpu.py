from __future__ import annotations

import os

import pytest
import torch
from ds4rt_reference.deepseek_v4_attention_layer_capture import (
    bind_deepseek_v4_c4_continuation_attention_layer,
    bind_deepseek_v4_c4_decode_attention_layer,
    bind_deepseek_v4_c4_prefill_attention_layer,
    bind_deepseek_v4_c128_continuation_attention_layer,
    bind_deepseek_v4_c128_decode_attention_layer,
    bind_deepseek_v4_c128_prefill_attention_layer,
    bind_deepseek_v4_sliding_attention_layer,
    plan_deepseek_v4_attention_layer,
    run_deepseek_v4_c4_continuation_attention_layer,
    run_deepseek_v4_c4_decode_attention_layer,
    run_deepseek_v4_c4_prefill_attention_layer,
    run_deepseek_v4_c128_continuation_attention_layer,
    run_deepseek_v4_c128_decode_attention_layer,
    run_deepseek_v4_c128_prefill_attention_layer,
    run_deepseek_v4_sliding_attention_layer,
)
from b12x.attention import dsv4_compressor, dsv4_producer, nsa_indexer
from b12x.attention._shared.mla.compressed_reference import (
    pack_compressed_mla_kv_cache_reference,
)
from b12x.attention.nsa_indexer.reference import pack_index_k_cache_reference
from b12x.gemm import wo_projection

pytestmark = pytest.mark.skipif(
    os.environ.get("DS4RT_RUN_GPU_QUALIFICATION") != "1",
    reason="set DS4RT_RUN_GPU_QUALIFICATION=1 for the explicit GPU0 qualification",
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


@pytest.mark.parametrize("cache_format", ("fp8", "nvfp4"))
def test_flash_m1_sliding_half_layer_replays_one_fixed_arena_on_gpu0(
    cache_format: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        pytest.skip(f"GPU0 must be SM120, got sm_{capability[0]}{capability[1]}")

    with torch.cuda.device(device):
        torch.manual_seed(20260804)
        hidden, heads = 4_096, 64
        tokens = 1
        contract = plan_deepseek_v4_attention_layer(
            variant="flash",
            mode="decode",
            compression=0,
            max_rows=tokens,
            source_pages=1,
            max_positions=1,
            cache_format=cache_format,
        )

        producer_weights, output_weights = _flash_attention_weights(device)

        arena = torch.empty(
            (contract.arena.total_bytes,),
            device=device,
            dtype=torch.uint8,
        )
        hidden_states = (
            torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16) / 4
        ).contiguous()
        positions = torch.zeros((tokens,), device=device, dtype=torch.int32)
        main_slots = torch.zeros_like(positions)
        cos_sin = torch.zeros((1, 64), device=device, dtype=torch.float32)
        cos_sin[:, :32] = 1
        main_cache = torch.zeros(
            (1, contract.main_page_bytes), device=device, dtype=torch.uint8
        )
        swa_indices = torch.zeros((tokens, 128), device=device, dtype=torch.int32)
        swa_lengths = torch.ones((tokens,), device=device, dtype=torch.int32)
        residual = (
            torch.randn((tokens, 4, hidden), device=device, dtype=torch.bfloat16) / 8
        ).contiguous()
        prev_post = torch.zeros((tokens, 4), device=device, dtype=torch.float32)
        prev_comb = (
            torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).contiguous()
        )
        fn = (
            torch.randn((24, 4 * hidden), device=device, dtype=torch.float32) / 128
        ).contiguous()
        hc_scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)
        hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        norm_weight = torch.ones((hidden,), device=device, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual)
        y_out = torch.empty_like(hidden_states)
        post_out = torch.empty_like(prev_post)
        comb_out = torch.empty_like(prev_comb)
        layer = bind_deepseek_v4_sliding_attention_layer(
            contract,
            arena=arena,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin,
            main_kv_cache=main_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            producer_weights=producer_weights,
            output_weights=output_weights,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            norm_weight=norm_weight,
            residual_out=residual_out,
            y_out=y_out,
            post_out=post_out,
            comb_out=comb_out,
            attn_sink=torch.zeros((heads,), device=device, dtype=torch.float32),
        )

        base = arena.data_ptr()
        layout = contract.arena
        assert layer.producer_binding.query.data_ptr() == base + layout.query_offset
        assert layer.attention_binding.q.data_ptr() == base + layout.query_offset
        assert (
            layer.output_binding.o.data_ptr() == base + layout.attention_output_offset
        )
        assert (
            layer.mhc_binding.partials.untyped_storage().data_ptr()
            == arena.untyped_storage().data_ptr()
        )
        assert layer.output_projection.untyped_storage().data_ptr() == base

        run_deepseek_v4_sliding_attention_layer(layer)
        torch.cuda.synchronize(device)
        eager = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        assert all(bool(torch.isfinite(output.float()).all()) for output in eager)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = run_deepseek_v4_sliding_attention_layer(layer)
        output_ptrs = tuple(output.data_ptr() for output in graph_outputs)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)
        assert output_ptrs == tuple(
            output.data_ptr() for output in (residual_out, post_out, comb_out, y_out)
        )
        for actual, expected in zip((residual_out, post_out, comb_out, y_out), eager):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        hidden_states.mul_(-0.5).add_(0.01171875)
        residual.mul_(0.625).sub_(0.01953125)
        graph.replay()
        torch.cuda.synchronize(device)
        replayed = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        run_deepseek_v4_sliding_attention_layer(layer)
        torch.cuda.synchronize(device)
        for actual, expected in zip(
            replayed,
            (residual_out, post_out, comb_out, y_out),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cache_format", ("fp8", "nvfp4"))
def test_flash_m1_c128_decode_replays_one_fixed_arena_on_gpu0(
    cache_format: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        pytest.skip(f"GPU0 must be SM120, got sm_{capability[0]}{capability[1]}")

    with torch.cuda.device(device):
        torch.manual_seed(20260805)
        hidden, heads, tokens = 4_096, 64, 1
        contract = plan_deepseek_v4_attention_layer(
            variant="flash",
            mode="decode",
            compression=128,
            max_rows=tokens,
            source_pages=1,
            max_positions=256,
            cache_format=cache_format,
        )
        producer_weights, output_weights = _flash_attention_weights(device)
        compressor_weights = dsv4_compressor.pack_weights(
            (
                torch.randn((512, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((512, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((128, 512), device=device, dtype=torch.float32) / 16
            ).contiguous(),
            (
                torch.randn((512,), device=device, dtype=torch.bfloat16) / 16 + 1
            ).contiguous(),
        )

        arena = torch.empty(
            (contract.arena.total_bytes,), device=device, dtype=torch.uint8
        )
        hidden_states = (
            torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16) / 4
        ).contiguous()
        positions = torch.full((tokens,), 255, device=device, dtype=torch.int32)
        main_slots = positions.clone()
        cos_sin = torch.zeros((256, 64), device=device, dtype=torch.float32)
        cos_sin[:, :32] = 1
        main_cache = torch.zeros(
            (1, contract.main_page_bytes), device=device, dtype=torch.uint8
        )
        swa_indices = torch.full((tokens, 128), 255, device=device, dtype=torch.int32)
        swa_lengths = torch.ones((tokens,), device=device, dtype=torch.int32)
        compressed_cache = torch.zeros(
            (1, contract.compressed_page_bytes), device=device, dtype=torch.uint8
        )
        compressor_kv_state = (
            torch.randn((1, 128, 512), device=device, dtype=torch.float32) / 4
        ).contiguous()
        compressor_score_state = (
            torch.randn((1, 128, 512), device=device, dtype=torch.float32) / 4
        ).contiguous()
        indexed_indices = torch.zeros(
            (tokens, contract.indexed_width), device=device, dtype=torch.int32
        )
        indexed_lengths = torch.ones((tokens,), device=device, dtype=torch.int32)
        residual = (
            torch.randn((tokens, 4, hidden), device=device, dtype=torch.bfloat16) / 8
        ).contiguous()
        prev_post = torch.zeros((tokens, 4), device=device, dtype=torch.float32)
        prev_comb = (
            torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).contiguous()
        )
        fn = (
            torch.randn((24, 4 * hidden), device=device, dtype=torch.float32) / 128
        ).contiguous()
        hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        norm_weight = torch.ones((hidden,), device=device, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual)
        y_out = torch.empty_like(hidden_states)
        post_out = torch.empty_like(prev_post)
        comb_out = torch.empty_like(prev_comb)
        layer = bind_deepseek_v4_c128_decode_attention_layer(
            contract,
            arena=arena,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin,
            main_kv_cache=main_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            producer_weights=producer_weights,
            sequence_ids=torch.zeros((tokens,), device=device, dtype=torch.int32),
            compressed_slots=torch.ones((tokens,), device=device, dtype=torch.int32),
            compressed_cos_sin_cache=cos_sin,
            compressed_main_cache=compressed_cache,
            main_kv_state=compressor_kv_state,
            main_score_state=compressor_score_state,
            compressor_weights=compressor_weights,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            output_weights=output_weights,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            norm_weight=norm_weight,
            residual_out=residual_out,
            y_out=y_out,
            post_out=post_out,
            comb_out=comb_out,
            attn_sink=torch.zeros((heads,), device=device, dtype=torch.float32),
        )

        base = arena.data_ptr()
        layout = contract.arena
        assert layer.producer_binding.query.data_ptr() == base + layout.query_offset
        assert (
            layer.compressor_binding.projection.data_ptr()
            == base + layout.compressor_scratch_offset
        )
        assert layer.attention_binding.q.data_ptr() == base + layout.query_offset
        assert (
            layer.output_binding.o.data_ptr() == base + layout.attention_output_offset
        )

        run_deepseek_v4_c128_decode_attention_layer(layer)
        torch.cuda.synchronize(device)
        eager = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        eager_compressed = compressed_cache.clone()
        assert all(bool(torch.isfinite(output.float()).all()) for output in eager)
        assert bool(torch.count_nonzero(eager_compressed))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = run_deepseek_v4_c128_decode_attention_layer(layer)
        output_ptrs = tuple(output.data_ptr() for output in graph_outputs)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)
        assert output_ptrs == tuple(
            output.data_ptr() for output in (residual_out, post_out, comb_out, y_out)
        )
        assert torch.equal(compressed_cache, eager_compressed)
        for actual, expected in zip((residual_out, post_out, comb_out, y_out), eager):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        hidden_states.mul_(-0.5).add_(0.01171875)
        residual.mul_(0.625).sub_(0.01953125)
        graph.replay()
        torch.cuda.synchronize(device)
        replayed = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        replayed_compressed = compressed_cache.clone()
        run_deepseek_v4_c128_decode_attention_layer(layer)
        torch.cuda.synchronize(device)
        assert torch.equal(compressed_cache, replayed_compressed)
        for actual, expected in zip(
            replayed,
            (residual_out, post_out, comb_out, y_out),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cache_format", ("fp8", "nvfp4"))
def test_flash_m128_c128_initial_prefill_replays_one_fixed_arena_on_gpu0(
    cache_format: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        pytest.skip(f"GPU0 must be SM120, got sm_{capability[0]}{capability[1]}")

    with torch.cuda.device(device):
        torch.manual_seed(20260807)
        hidden, heads, tokens = 4_096, 64, 128
        contract = plan_deepseek_v4_attention_layer(
            variant="flash",
            mode="extend",
            compression=128,
            max_rows=tokens,
            source_pages=1,
            max_positions=tokens,
            cache_format=cache_format,
        )
        producer_weights, output_weights = _flash_attention_weights(device)
        compressor_weights = dsv4_compressor.pack_weights(
            (
                torch.randn((512, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((512, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((128, 512), device=device, dtype=torch.float32) / 16
            ).contiguous(),
            (
                torch.randn((512,), device=device, dtype=torch.bfloat16) / 16 + 1
            ).contiguous(),
        )

        arena = torch.empty(
            (contract.arena.total_bytes,), device=device, dtype=torch.uint8
        )
        hidden_states = (
            torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16) / 4
        ).contiguous()
        positions = torch.arange(tokens, device=device, dtype=torch.int32)
        cos_sin = torch.zeros((tokens, 64), device=device, dtype=torch.float32)
        cos_sin[:, :32] = 1
        # Deliberately allocate the compressed page first. Main and compressed
        # cache pools have independent owners, so the dual-cache kernel must not
        # rely on either device-address ordering.
        compressed_cache = torch.zeros(
            (1, contract.compressed_page_bytes), device=device, dtype=torch.uint8
        )
        main_cache = torch.zeros(
            (1, contract.main_page_bytes), device=device, dtype=torch.uint8
        )
        swa_indices = torch.zeros(
            (tokens, 128), device=device, dtype=torch.int32
        )
        swa_indices[:, 0].copy_(positions)
        swa_lengths = torch.ones((tokens,), device=device, dtype=torch.int32)
        compressor_kv_state = (
            torch.randn((1, 128, 512), device=device, dtype=torch.float32) / 4
        ).contiguous()
        compressor_score_state = (
            torch.randn((1, 128, 512), device=device, dtype=torch.float32) / 4
        ).contiguous()
        indexed_indices = torch.zeros(
            (tokens, contract.indexed_width), device=device, dtype=torch.int32
        )
        indexed_lengths = torch.zeros(
            (tokens,), device=device, dtype=torch.int32
        )
        indexed_lengths[-1] = 1
        residual = (
            torch.randn((tokens, 4, hidden), device=device, dtype=torch.bfloat16) / 8
        ).contiguous()
        prev_post = torch.zeros((tokens, 4), device=device, dtype=torch.float32)
        prev_comb = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(tokens, -1, -1)
            .contiguous()
        )
        fn = (
            torch.randn((24, 4 * hidden), device=device, dtype=torch.float32) / 128
        ).contiguous()
        hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        norm_weight = torch.ones((hidden,), device=device, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual)
        y_out = torch.empty_like(hidden_states)
        post_out = torch.empty_like(prev_post)
        comb_out = torch.empty_like(prev_comb)
        layer = bind_deepseek_v4_c128_prefill_attention_layer(
            contract,
            arena=arena,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=positions,
            cos_sin_cache=cos_sin,
            main_kv_cache=main_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            producer_weights=producer_weights,
            active_groups=torch.ones((1,), device=device, dtype=torch.int32),
            group_source_starts=torch.zeros(
                (1,), device=device, dtype=torch.int32
            ),
            group_rope_positions=torch.zeros(
                (1,), device=device, dtype=torch.int32
            ),
            compressed_slots=torch.zeros((1,), device=device, dtype=torch.int32),
            active_sequences=torch.ones((1,), device=device, dtype=torch.int32),
            sequence_offsets=torch.tensor(
                [0, tokens], device=device, dtype=torch.int32
            ),
            state_sequence_ids=torch.zeros(
                (1,), device=device, dtype=torch.int32
            ),
            compressed_cos_sin_cache=cos_sin,
            compressed_main_cache=compressed_cache,
            main_kv_state=compressor_kv_state,
            main_score_state=compressor_score_state,
            compressor_weights=compressor_weights,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            output_weights=output_weights,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            norm_weight=norm_weight,
            residual_out=residual_out,
            y_out=y_out,
            post_out=post_out,
            comb_out=comb_out,
            attn_sink=torch.zeros((heads,), device=device, dtype=torch.float32),
        )

        base = arena.data_ptr()
        layout = contract.arena
        assert layer.producer_binding.query.data_ptr() == base + layout.query_offset
        assert (
            layer.compressor_binding.projection.data_ptr()
            == base + layout.compressor_scratch_offset
        )
        assert layer.attention_binding.q.data_ptr() == base + layout.query_offset
        assert (
            layer.output_binding.o.data_ptr() == base + layout.attention_output_offset
        )
        assert layer.expert_tensor_parallel == 4
        assert not layer.expert_parallel

        run_deepseek_v4_c128_prefill_attention_layer(layer)
        torch.cuda.synchronize(device)
        eager = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        eager_compressed = compressed_cache.clone()
        assert all(bool(torch.isfinite(output.float()).all()) for output in eager)
        assert bool(torch.count_nonzero(eager_compressed))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = run_deepseek_v4_c128_prefill_attention_layer(layer)
        output_ptrs = tuple(output.data_ptr() for output in graph_outputs)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)
        assert output_ptrs == tuple(
            output.data_ptr() for output in (residual_out, post_out, comb_out, y_out)
        )
        assert torch.equal(compressed_cache, eager_compressed)
        for actual, expected in zip((residual_out, post_out, comb_out, y_out), eager):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        hidden_states.mul_(-0.5).add_(0.01171875)
        residual.mul_(0.625).sub_(0.01953125)
        graph.replay()
        torch.cuda.synchronize(device)
        replayed = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        replayed_compressed = compressed_cache.clone()
        run_deepseek_v4_c128_prefill_attention_layer(layer)
        torch.cuda.synchronize(device)
        assert torch.equal(compressed_cache, replayed_compressed)
        for actual, expected in zip(
            replayed,
            (residual_out, post_out, comb_out, y_out),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cache_format", ("fp8", "nvfp4"))
def test_flash_m32_c128_continuation_replays_transactional_state_on_gpu0(
    cache_format: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        pytest.skip(f"GPU0 must be SM120, got sm_{capability[0]}{capability[1]}")

    with torch.cuda.device(device):
        torch.manual_seed(20260808)
        hidden, heads, tokens, pages = 4_096, 64, 32, 3
        contract = plan_deepseek_v4_attention_layer(
            variant="flash",
            mode="extend",
            compression=128,
            max_rows=tokens,
            source_pages=pages,
            max_positions=768,
            cache_format=cache_format,
        )
        producer_weights, output_weights = _flash_attention_weights(device)
        compressor_weights = dsv4_compressor.pack_weights(
            (
                torch.randn((512, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((512, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((128, 512), device=device, dtype=torch.float32) / 16
            ).contiguous(),
            (
                torch.randn((512,), device=device, dtype=torch.bfloat16) / 16 + 1
            ).contiguous(),
        )

        arena = torch.empty(
            (contract.arena.total_bytes,), device=device, dtype=torch.uint8
        )
        hidden_states = (
            torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16) / 4
        ).contiguous()
        positions = torch.cat(
            (
                torch.arange(250, 270, device=device, dtype=torch.int32),
                torch.arange(260, 264, device=device, dtype=torch.int32),
                torch.arange(8, device=device, dtype=torch.int32),
            )
        ).contiguous()
        main_slots = torch.cat(
            (
                torch.arange(20, device=device, dtype=torch.int32),
                torch.arange(256, 260, device=device, dtype=torch.int32),
                torch.arange(512, 520, device=device, dtype=torch.int32),
            )
        ).contiguous()
        cos_sin = torch.zeros((384, 64), device=device, dtype=torch.float32)
        cos_sin[:, :32] = 1
        compressed_cache = torch.zeros(
            (1, contract.compressed_page_bytes), device=device, dtype=torch.uint8
        )
        main_cache = torch.zeros(
            (pages, contract.main_page_bytes), device=device, dtype=torch.uint8
        )
        swa_indices = torch.zeros(
            (tokens, 128), device=device, dtype=torch.int32
        )
        swa_indices[:, 0].copy_(main_slots)
        swa_lengths = torch.ones((tokens,), device=device, dtype=torch.int32)
        compressor_kv_state = (
            torch.randn((2, 128, 512), device=device, dtype=torch.float32) / 4
        ).contiguous()
        compressor_score_state = (
            torch.randn((2, 128, 512), device=device, dtype=torch.float32) / 4
        ).contiguous()
        initial_kv_state = compressor_kv_state.clone()
        initial_score_state = compressor_score_state.clone()
        indexed_indices = torch.zeros(
            (tokens, contract.indexed_width), device=device, dtype=torch.int32
        )
        indexed_lengths = torch.zeros(
            (tokens,), device=device, dtype=torch.int32
        )
        indexed_lengths[5:20] = 1
        residual = (
            torch.randn((tokens, 4, hidden), device=device, dtype=torch.bfloat16) / 8
        ).contiguous()
        prev_post = torch.zeros((tokens, 4), device=device, dtype=torch.float32)
        prev_comb = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(tokens, -1, -1)
            .contiguous()
        )
        fn = (
            torch.randn((24, 4 * hidden), device=device, dtype=torch.float32) / 128
        ).contiguous()
        hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        norm_weight = torch.ones((hidden,), device=device, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual)
        y_out = torch.empty_like(hidden_states)
        post_out = torch.empty_like(prev_post)
        comb_out = torch.empty_like(prev_comb)
        layer = bind_deepseek_v4_c128_continuation_attention_layer(
            contract,
            arena=arena,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin,
            main_kv_cache=main_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            producer_weights=producer_weights,
            active_groups=torch.ones((1,), device=device, dtype=torch.int32),
            group_sequence_slots=torch.tensor(
                [0, 0, 0, 0], device=device, dtype=torch.int32
            ),
            group_source_positions=torch.tensor(
                [128, 0, 0, 0], device=device, dtype=torch.int32
            ),
            group_rope_positions=torch.tensor(
                [128, 0, 0, 0], device=device, dtype=torch.int32
            ),
            compressed_slots=torch.tensor(
                [0, 1, 1, 1], device=device, dtype=torch.int32
            ),
            active_sequences=torch.tensor(
                [2], device=device, dtype=torch.int32
            ),
            sequence_offsets=torch.tensor(
                [0, 20, 24], device=device, dtype=torch.int32
            ),
            sequence_start_positions=torch.tensor(
                [250, 260], device=device, dtype=torch.int32
            ),
            state_sequence_ids=torch.tensor(
                [0, 1], device=device, dtype=torch.int32
            ),
            compressed_cos_sin_cache=cos_sin,
            compressed_main_cache=compressed_cache,
            main_kv_state=compressor_kv_state,
            main_score_state=compressor_score_state,
            compressor_weights=compressor_weights,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            output_weights=output_weights,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            norm_weight=norm_weight,
            residual_out=residual_out,
            y_out=y_out,
            post_out=post_out,
            comb_out=comb_out,
            attn_sink=torch.zeros((heads,), device=device, dtype=torch.float32),
        )

        base = arena.data_ptr()
        layout = contract.arena
        assert layer.producer_binding.query.data_ptr() == base + layout.query_offset
        assert (
            layer.compressor_binding.projection.data_ptr()
            == base + layout.compressor_scratch_offset
        )
        assert layer.attention_binding.q.data_ptr() == base + layout.query_offset
        assert layer.scheduler_owns_state_transactions
        assert layer.expert_tensor_parallel == 4
        assert not layer.expert_parallel

        def reset_compressor_state() -> None:
            compressor_kv_state.copy_(initial_kv_state)
            compressor_score_state.copy_(initial_score_state)
            compressed_cache.zero_()

        run_deepseek_v4_c128_continuation_attention_layer(layer)
        torch.cuda.synchronize(device)
        eager = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        eager_compressed = compressed_cache.clone()
        eager_kv_state = compressor_kv_state.clone()
        eager_score_state = compressor_score_state.clone()
        assert all(bool(torch.isfinite(output.float()).all()) for output in eager)
        assert bool(torch.count_nonzero(eager_compressed))
        assert not torch.equal(eager_kv_state, initial_kv_state)
        assert not torch.equal(eager_score_state, initial_score_state)

        reset_compressor_state()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = run_deepseek_v4_c128_continuation_attention_layer(layer)
        output_ptrs = tuple(output.data_ptr() for output in graph_outputs)
        for _ in range(3):
            reset_compressor_state()
            graph.replay()
        torch.cuda.synchronize(device)
        assert output_ptrs == tuple(
            output.data_ptr() for output in (residual_out, post_out, comb_out, y_out)
        )
        assert torch.equal(compressed_cache, eager_compressed)
        assert torch.equal(compressor_kv_state, eager_kv_state)
        assert torch.equal(compressor_score_state, eager_score_state)
        for actual, expected in zip((residual_out, post_out, comb_out, y_out), eager):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        hidden_states.mul_(-0.5).add_(0.01171875)
        residual.mul_(0.625).sub_(0.01953125)
        reset_compressor_state()
        graph.replay()
        torch.cuda.synchronize(device)
        replayed = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        replayed_compressed = compressed_cache.clone()
        replayed_kv_state = compressor_kv_state.clone()
        replayed_score_state = compressor_score_state.clone()
        reset_compressor_state()
        run_deepseek_v4_c128_continuation_attention_layer(layer)
        torch.cuda.synchronize(device)
        assert torch.equal(compressed_cache, replayed_compressed)
        assert torch.equal(compressor_kv_state, replayed_kv_state)
        assert torch.equal(compressor_score_state, replayed_score_state)
        for actual, expected in zip(
            replayed,
            (residual_out, post_out, comb_out, y_out),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cache_format", ("fp8", "nvfp4"))
def test_flash_m1_c4_decode_replays_learned_selection_on_gpu0(
    cache_format: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        pytest.skip(f"GPU0 must be SM120, got sm_{capability[0]}{capability[1]}")

    with torch.cuda.device(device):
        torch.manual_seed(20260806)
        hidden, heads, tokens, pages = 4_096, 64, 1, 12
        contract = plan_deepseek_v4_attention_layer(
            variant="flash",
            mode="decode",
            compression=4,
            max_rows=tokens,
            source_pages=pages,
            max_positions=3_072,
            cache_format=cache_format,
        )
        producer_weights, output_weights = _flash_attention_weights(device)
        compressor_weights = dsv4_compressor.pack_weights(
            (
                torch.randn((1_024, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((1_024, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((4, 1_024), device=device, dtype=torch.float32) / 16
            ).contiguous(),
            (
                torch.randn((512,), device=device, dtype=torch.bfloat16) / 16 + 1
            ).contiguous(),
            index_wkv=(
                torch.randn((256, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            index_wgate=(
                torch.randn((256, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            index_ape=(
                torch.randn((4, 256), device=device, dtype=torch.float32) / 16
            ).contiguous(),
            index_norm=(
                torch.randn((128,), device=device, dtype=torch.bfloat16) / 16 + 1
            ).contiguous(),
        )
        index_wq, index_wq_scale = _block_fp8_weight(
            64 * 128,
            1_024,
            device=device,
        )
        indexer_weights = dsv4_producer.pack_indexer_weights(
            index_wq,
            index_wq_scale,
            (
                torch.randn((64, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
        )
        selector_plan = nsa_indexer.plan(
            nsa_indexer.Caps(
                device=device,
                source_layout=nsa_indexer.SOURCE_LAYOUT_PAGED,
                num_q_heads=64,
                max_q_rows=tokens,
                max_page_table_width=pages,
                topk=contract.indexed_width,
                mode="decode",
                page_size=64,
            )
        )
        (selector_spec,) = selector_plan.scratch_specs()
        selector_scratch = torch.empty(
            selector_spec.shape,
            device=device,
            dtype=selector_spec.dtype,
        )

        arena = torch.empty(
            (contract.arena.total_bytes,), device=device, dtype=torch.uint8
        )
        hidden_states = (
            torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16) / 4
        ).contiguous()
        position = 3_071 if cache_format == "nvfp4" else 3_068
        positions = torch.full((tokens,), position, device=device, dtype=torch.int32)
        main_slots = positions.clone()
        cos_sin = torch.zeros((3_072, 64), device=device, dtype=torch.float32)
        cos_sin[:, :32] = 1
        main_cache = torch.zeros(
            (pages, contract.main_page_bytes), device=device, dtype=torch.uint8
        )
        swa_indices = torch.full(
            (tokens, 128), position, device=device, dtype=torch.int32
        )
        swa_lengths = torch.ones((tokens,), device=device, dtype=torch.int32)
        compressed_values = (
            torch.randn((pages * 64, 512), device=device, dtype=torch.bfloat16) / 4
        )
        compressed_cache = (
            torch.zeros(
                (pages, contract.compressed_page_bytes),
                device=device,
                dtype=torch.uint8,
            )
            if cache_format == "nvfp4"
            else pack_compressed_mla_kv_cache_reference(
                compressed_values[:, :448],
                compressed_values[:, 448:],
                page_size=64,
                num_pages=pages,
            )
        )
        index_cache = pack_index_k_cache_reference(
            torch.randn((pages * 64, 128), device=device, dtype=torch.float32)
        )
        main_kv_state = (
            torch.randn((1, 16, 1_024), device=device, dtype=torch.float32) / 4
        ).contiguous()
        main_score_state = (
            torch.randn((1, 16, 1_024), device=device, dtype=torch.float32) / 4
        ).contiguous()
        index_kv_state = (
            torch.randn((1, 16, 256), device=device, dtype=torch.float32) / 4
        ).contiguous()
        index_score_state = (
            torch.randn((1, 16, 256), device=device, dtype=torch.float32) / 4
        ).contiguous()
        residual = (
            torch.randn((tokens, 4, hidden), device=device, dtype=torch.bfloat16) / 8
        ).contiguous()
        prev_post = torch.zeros((tokens, 4), device=device, dtype=torch.float32)
        prev_comb = (
            torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).contiguous()
        )
        fn = (
            torch.randn((24, 4 * hidden), device=device, dtype=torch.float32) / 128
        ).contiguous()
        hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        norm_weight = torch.ones((hidden,), device=device, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual)
        y_out = torch.empty_like(hidden_states)
        post_out = torch.empty_like(prev_post)
        comb_out = torch.empty_like(prev_comb)
        index_cache_seqlens = torch.full(
            (tokens,), 768, device=device, dtype=torch.int32
        )
        layer = bind_deepseek_v4_c4_decode_attention_layer(
            contract,
            arena=arena,
            selector_scratch=selector_scratch,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin,
            main_kv_cache=main_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            producer_weights=producer_weights,
            sequence_ids=torch.zeros((tokens,), device=device, dtype=torch.int32),
            compressed_slots=torch.full(
                (tokens,), 767, device=device, dtype=torch.int32
            ),
            compressed_cos_sin_cache=cos_sin,
            compressed_main_cache=compressed_cache,
            main_kv_state=main_kv_state,
            main_score_state=main_score_state,
            index_cache=index_cache,
            index_kv_state=index_kv_state,
            index_score_state=index_score_state,
            compressor_weights=compressor_weights,
            indexer_weights=indexer_weights,
            real_page_table=torch.arange(
                pages, device=device, dtype=torch.int32
            ).unsqueeze(0),
            index_cache_seqlens=index_cache_seqlens,
            output_weights=output_weights,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            norm_weight=norm_weight,
            residual_out=residual_out,
            y_out=y_out,
            post_out=post_out,
            comb_out=comb_out,
            attn_sink=torch.zeros((heads,), device=device, dtype=torch.float32),
        )

        base = arena.data_ptr()
        layout = contract.arena
        assert layer.indexer_producer_binding.query.data_ptr() == (
            base + layout.index_query_offset
        )
        assert layer.indexer_producer_binding.head_weights.data_ptr() == (
            base + layout.index_head_weights_offset
        )
        assert layer.attention_binding.indexed_indices.data_ptr() == (
            base + layout.selected_indices_offset
        )

        mutable_cache_state = (
            main_cache,
            compressed_cache,
            index_cache,
            main_kv_state,
            main_score_state,
            index_kv_state,
            index_score_state,
        )
        initial_cache_state = tuple(value.clone() for value in mutable_cache_state)

        def restore_cache_state() -> None:
            for value, initial in zip(mutable_cache_state, initial_cache_state):
                value.copy_(initial)

        run_deepseek_v4_c4_decode_attention_layer(layer)
        torch.cuda.synchronize(device)
        eager = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        eager_selected = layer.arena_binding.selected_indices.clone()
        assert all(bool(torch.isfinite(output.float()).all()) for output in eager)
        assert bool(torch.count_nonzero(compressed_cache))
        assert bool(torch.count_nonzero(index_cache))
        assert layer.attention_binding.indexed_lengths is index_cache_seqlens
        assert int(torch.unique(eager_selected).numel()) == 512
        assert int(eager_selected.min()) >= 0
        assert int(eager_selected.max()) < 768

        if cache_format == "nvfp4":
            # This case deliberately completes a C4 group. Restore the recurrent
            # compressor state so eager and captured executions start at the
            # same boundary instead of accumulating the synthetic row twice.
            restore_cache_state()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = run_deepseek_v4_c4_decode_attention_layer(layer)
        output_ptrs = tuple(output.data_ptr() for output in graph_outputs)
        for _ in range(3):
            if cache_format == "nvfp4":
                restore_cache_state()
            graph.replay()
        torch.cuda.synchronize(device)
        assert output_ptrs == tuple(
            output.data_ptr() for output in (residual_out, post_out, comb_out, y_out)
        )
        assert torch.equal(
            torch.sort(layer.arena_binding.selected_indices).values,
            torch.sort(eager_selected).values,
        )
        for actual, expected in zip((residual_out, post_out, comb_out, y_out), eager):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        hidden_states.mul_(-0.5).add_(0.01171875)
        residual.mul_(0.625).sub_(0.01953125)
        if cache_format == "nvfp4":
            restore_cache_state()
        graph.replay()
        torch.cuda.synchronize(device)
        replayed = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        replayed_selected = layer.arena_binding.selected_indices.clone()
        if cache_format == "nvfp4":
            restore_cache_state()
        run_deepseek_v4_c4_decode_attention_layer(layer)
        torch.cuda.synchronize(device)
        assert torch.equal(
            torch.sort(layer.arena_binding.selected_indices).values,
            torch.sort(replayed_selected).values,
        )
        for actual, expected in zip(
            replayed,
            (residual_out, post_out, comb_out, y_out),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cache_format", ("fp8", "nvfp4"))
def test_flash_m512_c4_initial_prefill_replays_causal_selection_on_gpu0(
    cache_format: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        pytest.skip(f"GPU0 must be SM120, got sm_{capability[0]}{capability[1]}")

    with torch.cuda.device(device):
        torch.manual_seed(20260811)
        hidden, heads, tokens, pages = 4_096, 64, 512, 2
        compressed_rows = tokens // 4
        contract = plan_deepseek_v4_attention_layer(
            variant="flash",
            mode="extend",
            compression=4,
            max_rows=tokens,
            source_pages=pages,
            max_positions=tokens,
            cache_format=cache_format,
        )
        producer_weights, output_weights = _flash_attention_weights(device)
        compressor_weights = dsv4_compressor.pack_weights(
            (
                torch.randn((1_024, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((1_024, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((4, 1_024), device=device, dtype=torch.float32) / 16
            ).contiguous(),
            (
                torch.randn((512,), device=device, dtype=torch.bfloat16) / 16 + 1
            ).contiguous(),
            index_wkv=(
                torch.randn((256, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            index_wgate=(
                torch.randn((256, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            index_ape=(
                torch.randn((4, 256), device=device, dtype=torch.float32) / 16
            ).contiguous(),
            index_norm=(
                torch.randn((128,), device=device, dtype=torch.bfloat16) / 16 + 1
            ).contiguous(),
        )
        index_wq, index_wq_scale = _block_fp8_weight(
            64 * 128,
            1_024,
            device=device,
        )
        indexer_weights = dsv4_producer.pack_indexer_weights(
            index_wq,
            index_wq_scale,
            (
                torch.randn((64, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
        )
        selector_plan = nsa_indexer.plan(
            nsa_indexer.Caps(
                device=device,
                source_layout=nsa_indexer.SOURCE_LAYOUT_PAGED,
                num_q_heads=64,
                max_q_rows=tokens,
                max_page_table_width=pages,
                topk=contract.indexed_width,
                mode="prefill",
                page_size=64,
                shared_page_table=True,
            )
        )
        (selector_spec,) = selector_plan.scratch_specs()
        selector_scratch = torch.empty(
            selector_spec.shape,
            device=device,
            dtype=selector_spec.dtype,
        )

        arena = torch.empty(
            (contract.arena.total_bytes,), device=device, dtype=torch.uint8
        )
        hidden_states = (
            torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16) / 4
        ).contiguous()
        positions = torch.arange(tokens, device=device, dtype=torch.int32)
        main_slots = positions.clone()
        cos_sin = torch.zeros((tokens, 64), device=device, dtype=torch.float32)
        cos_sin[:, :32] = 1
        main_cache = torch.zeros(
            (pages, contract.main_page_bytes), device=device, dtype=torch.uint8
        )
        swa_indices = positions.view(tokens, 1).expand(tokens, 128).contiguous()
        swa_lengths = torch.ones((tokens,), device=device, dtype=torch.int32)
        compressed_cache = (
            torch.zeros(
                (pages, contract.compressed_page_bytes),
                device=device,
                dtype=torch.uint8,
            )
            if cache_format == "nvfp4"
            else pack_compressed_mla_kv_cache_reference(
                torch.zeros(
                    (compressed_rows, 448), device=device, dtype=torch.bfloat16
                ),
                torch.zeros(
                    (compressed_rows, 64), device=device, dtype=torch.bfloat16
                ),
                page_size=64,
                num_pages=pages,
            )
        )
        index_cache = pack_index_k_cache_reference(
            torch.zeros(
                (compressed_rows, 128), device=device, dtype=torch.float32
            )
        )
        main_kv_state = torch.randn(
            (1, 16, 1_024), device=device, dtype=torch.float32
        ).contiguous()
        main_score_state = torch.randn(
            (1, 16, 1_024), device=device, dtype=torch.float32
        ).contiguous()
        index_kv_state = torch.randn(
            (1, 16, 256), device=device, dtype=torch.float32
        ).contiguous()
        index_score_state = torch.randn(
            (1, 16, 256), device=device, dtype=torch.float32
        ).contiguous()
        residual = (
            torch.randn((tokens, 4, hidden), device=device, dtype=torch.bfloat16) / 8
        ).contiguous()
        prev_post = torch.zeros((tokens, 4), device=device, dtype=torch.float32)
        prev_comb = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(tokens, -1, -1)
            .contiguous()
        )
        fn = (
            torch.randn((24, 4 * hidden), device=device, dtype=torch.float32) / 128
        ).contiguous()
        hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        norm_weight = torch.ones((hidden,), device=device, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual)
        y_out = torch.empty_like(hidden_states)
        post_out = torch.empty_like(prev_post)
        comb_out = torch.empty_like(prev_comb)
        group_starts = torch.arange(
            0, tokens, 4, device=device, dtype=torch.int32
        )
        index_cache_seqlens = torch.div(
            positions + 1,
            4,
            rounding_mode="floor",
        ).to(torch.int32)
        real_page_table = torch.arange(
            pages, device=device, dtype=torch.int32
        ).view(1, pages).expand(tokens, -1)
        assert real_page_table.stride(0) == 0
        layer = bind_deepseek_v4_c4_prefill_attention_layer(
            contract,
            arena=arena,
            selector_scratch=selector_scratch,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin,
            main_kv_cache=main_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            producer_weights=producer_weights,
            active_groups=torch.tensor(
                [compressed_rows], device=device, dtype=torch.int32
            ),
            group_source_starts=group_starts,
            group_rope_positions=group_starts,
            compressed_slots=torch.arange(
                compressed_rows, device=device, dtype=torch.int32
            ),
            active_sequences=torch.ones((1,), device=device, dtype=torch.int32),
            sequence_offsets=torch.tensor(
                [0, tokens], device=device, dtype=torch.int32
            ),
            state_sequence_ids=torch.zeros((1,), device=device, dtype=torch.int32),
            compressed_cos_sin_cache=cos_sin,
            compressed_main_cache=compressed_cache,
            main_kv_state=main_kv_state,
            main_score_state=main_score_state,
            index_cache=index_cache,
            index_kv_state=index_kv_state,
            index_score_state=index_score_state,
            compressor_weights=compressor_weights,
            indexer_weights=indexer_weights,
            real_page_table=real_page_table,
            index_cache_seqlens=index_cache_seqlens,
            output_weights=output_weights,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            norm_weight=norm_weight,
            residual_out=residual_out,
            y_out=y_out,
            post_out=post_out,
            comb_out=comb_out,
            attn_sink=torch.zeros((heads,), device=device, dtype=torch.float32),
        )

        assert layer.selector_binding.shared_page_table
        assert layer.selector_binding.real_page_table.stride(0) == 0
        assert layer.arena_binding.selected_indices.data_ptr() == (
            arena.data_ptr() + contract.arena.selected_indices_offset
        )
        run_deepseek_v4_c4_prefill_attention_layer(layer)
        torch.cuda.synchronize(device)
        eager = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        eager_selected = layer.arena_binding.selected_indices.clone()
        eager_compressed = compressed_cache.clone()
        eager_index = index_cache.clone()
        eager_states = tuple(
            state.clone()
            for state in (
                main_kv_state,
                main_score_state,
                index_kv_state,
                index_score_state,
            )
        )
        assert all(bool(torch.isfinite(output.float()).all()) for output in eager)
        assert bool(torch.count_nonzero(compressed_cache))
        assert bool(torch.count_nonzero(index_cache))
        assert layer.attention_binding.indexed_lengths is index_cache_seqlens
        assert torch.equal(eager_selected[0], torch.full_like(eager_selected[0], -1))
        assert torch.equal(
            eager_selected[3, 1:],
            torch.full_like(eager_selected[3, 1:], -1),
        )
        assert int(eager_selected[3, 0]) == 0
        assert int(eager_selected[-1, :compressed_rows].min()) >= 0
        assert int(eager_selected[-1, :compressed_rows].max()) < compressed_rows
        assert torch.equal(
            eager_selected[-1, compressed_rows:],
            torch.full_like(eager_selected[-1, compressed_rows:], -1),
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = run_deepseek_v4_c4_prefill_attention_layer(layer)
        output_ptrs = tuple(output.data_ptr() for output in graph_outputs)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)
        assert output_ptrs == tuple(
            output.data_ptr() for output in (residual_out, post_out, comb_out, y_out)
        )
        assert torch.equal(compressed_cache, eager_compressed)
        assert torch.equal(index_cache, eager_index)
        assert torch.equal(layer.arena_binding.selected_indices, eager_selected)
        for actual, expected in zip(
            (main_kv_state, main_score_state, index_kv_state, index_score_state),
            eager_states,
        ):
            assert torch.equal(actual, expected)
        for actual, expected in zip((residual_out, post_out, comb_out, y_out), eager):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        hidden_states.mul_(-0.5).add_(0.01171875)
        residual.mul_(0.625).sub_(0.01953125)
        graph.replay()
        torch.cuda.synchronize(device)
        replayed = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        replayed_compressed = compressed_cache.clone()
        replayed_index = index_cache.clone()
        replayed_selected = layer.arena_binding.selected_indices.clone()
        run_deepseek_v4_c4_prefill_attention_layer(layer)
        torch.cuda.synchronize(device)
        assert torch.equal(compressed_cache, replayed_compressed)
        assert torch.equal(index_cache, replayed_index)
        assert torch.equal(layer.arena_binding.selected_indices, replayed_selected)
        for actual, expected in zip(
            replayed,
            (residual_out, post_out, comb_out, y_out),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cache_format", ("fp8", "nvfp4"))
def test_flash_m8_c4_continuation_replays_shared_selection_on_gpu0(
    cache_format: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        pytest.skip(f"GPU0 must be SM120, got sm_{capability[0]}{capability[1]}")

    with torch.cuda.device(device):
        torch.manual_seed(20260812)
        hidden, heads, tokens, pages = 4_096, 64, 8, 3
        contract = plan_deepseek_v4_attention_layer(
            variant="flash",
            mode="extend",
            compression=4,
            max_rows=tokens,
            source_pages=pages,
            max_positions=768,
            cache_format=cache_format,
        )
        producer_weights, output_weights = _flash_attention_weights(device)
        compressor_weights = dsv4_compressor.pack_weights(
            (
                torch.randn((1_024, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((1_024, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            (
                torch.randn((4, 1_024), device=device, dtype=torch.float32) / 16
            ).contiguous(),
            (
                torch.randn((512,), device=device, dtype=torch.bfloat16) / 16 + 1
            ).contiguous(),
            index_wkv=(
                torch.randn((256, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            index_wgate=(
                torch.randn((256, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
            index_ape=(
                torch.randn((4, 256), device=device, dtype=torch.float32) / 16
            ).contiguous(),
            index_norm=(
                torch.randn((128,), device=device, dtype=torch.bfloat16) / 16 + 1
            ).contiguous(),
        )
        index_wq, index_wq_scale = _block_fp8_weight(
            64 * 128,
            1_024,
            device=device,
        )
        indexer_weights = dsv4_producer.pack_indexer_weights(
            index_wq,
            index_wq_scale,
            (
                torch.randn((64, hidden), device=device, dtype=torch.bfloat16) / 64
            ).contiguous(),
        )
        selector_plan = nsa_indexer.plan(
            nsa_indexer.Caps(
                device=device,
                source_layout=nsa_indexer.SOURCE_LAYOUT_PAGED,
                num_q_heads=64,
                max_q_rows=tokens,
                max_page_table_width=pages,
                topk=contract.indexed_width,
                mode="decode",
                page_size=64,
                shared_page_table=True,
            )
        )
        (selector_spec,) = selector_plan.scratch_specs()
        selector_scratch = torch.empty(
            selector_spec.shape,
            device=device,
            dtype=selector_spec.dtype,
        )

        arena = torch.empty(
            (contract.arena.total_bytes,), device=device, dtype=torch.uint8
        )
        hidden_states = (
            torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16) / 4
        ).contiguous()
        positions = torch.tensor(
            [6, 7, 3, 512, 513, 514, 515, 516],
            device=device,
            dtype=torch.int32,
        )
        main_slots = torch.tensor(
            [6, 7, 259, 512, 513, 514, 515, 516],
            device=device,
            dtype=torch.int32,
        )
        cos_sin = torch.zeros((768, 64), device=device, dtype=torch.float32)
        cos_sin[:, :32] = 1
        main_cache = torch.zeros(
            (pages, contract.main_page_bytes), device=device, dtype=torch.uint8
        )
        swa_indices = main_slots.view(tokens, 1).expand(tokens, 128).contiguous()
        swa_lengths = torch.ones((tokens,), device=device, dtype=torch.int32)
        compressed_values = (
            torch.randn((pages * 64, 512), device=device, dtype=torch.bfloat16) / 4
        )
        compressed_cache = (
            torch.zeros(
                (pages, contract.compressed_page_bytes),
                device=device,
                dtype=torch.uint8,
            )
            if cache_format == "nvfp4"
            else pack_compressed_mla_kv_cache_reference(
                compressed_values[:, :448],
                compressed_values[:, 448:],
                page_size=64,
                num_pages=pages,
            )
        )
        index_cache = pack_index_k_cache_reference(
            torch.randn((pages * 64, 128), device=device, dtype=torch.float32)
        )
        main_kv_state = (
            torch.randn((2, 16, 1_024), device=device, dtype=torch.float32) / 4
        ).contiguous()
        main_score_state = (
            torch.randn((2, 16, 1_024), device=device, dtype=torch.float32) / 4
        ).contiguous()
        index_kv_state = (
            torch.randn((2, 16, 256), device=device, dtype=torch.float32) / 4
        ).contiguous()
        index_score_state = (
            torch.randn((2, 16, 256), device=device, dtype=torch.float32) / 4
        ).contiguous()
        initial_compressed = compressed_cache.clone()
        initial_index = index_cache.clone()
        initial_states = tuple(
            state.clone()
            for state in (
                main_kv_state,
                main_score_state,
                index_kv_state,
                index_score_state,
            )
        )
        residual = (
            torch.randn((tokens, 4, hidden), device=device, dtype=torch.bfloat16) / 8
        ).contiguous()
        prev_post = torch.zeros((tokens, 4), device=device, dtype=torch.float32)
        prev_comb = (
            torch.eye(4, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(tokens, -1, -1)
            .contiguous()
        )
        fn = (
            torch.randn((24, 4 * hidden), device=device, dtype=torch.float32) / 128
        ).contiguous()
        hc_scale = torch.ones((3,), device=device, dtype=torch.float32)
        hc_base = torch.zeros((24,), device=device, dtype=torch.float32)
        norm_weight = torch.ones((hidden,), device=device, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual)
        y_out = torch.empty_like(hidden_states)
        post_out = torch.empty_like(prev_post)
        comb_out = torch.empty_like(prev_comb)
        real_page_table = torch.tensor(
            [[0, 2, 2]],
            device=device,
            dtype=torch.int32,
        ).expand(tokens, -1)
        index_cache_seqlens = torch.tensor(
            [1, 2, 1, 0, 0, 0, 0, 0],
            device=device,
            dtype=torch.int32,
        )
        layer = bind_deepseek_v4_c4_continuation_attention_layer(
            contract,
            arena=arena,
            selector_scratch=selector_scratch,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin,
            main_kv_cache=main_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            producer_weights=producer_weights,
            active_groups=torch.tensor([2], device=device, dtype=torch.int32),
            group_sequence_slots=torch.tensor(
                [0, 1, 0, 0], device=device, dtype=torch.int32
            ),
            group_source_positions=torch.tensor(
                [4, 0, 0, 0], device=device, dtype=torch.int32
            ),
            group_rope_positions=torch.tensor(
                [4, 0, 0, 0], device=device, dtype=torch.int32
            ),
            compressed_slots=torch.tensor(
                [1, 64, 191, 191], device=device, dtype=torch.int32
            ),
            active_sequences=torch.tensor([2], device=device, dtype=torch.int32),
            sequence_offsets=torch.tensor(
                [0, 2, 3], device=device, dtype=torch.int32
            ),
            sequence_start_positions=torch.tensor(
                [6, 3], device=device, dtype=torch.int32
            ),
            state_sequence_ids=torch.tensor(
                [0, 1], device=device, dtype=torch.int32
            ),
            compressed_cos_sin_cache=cos_sin,
            compressed_main_cache=compressed_cache,
            main_kv_state=main_kv_state,
            main_score_state=main_score_state,
            index_cache=index_cache,
            index_kv_state=index_kv_state,
            index_score_state=index_score_state,
            compressor_weights=compressor_weights,
            indexer_weights=indexer_weights,
            real_page_table=real_page_table,
            index_cache_seqlens=index_cache_seqlens,
            output_weights=output_weights,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            norm_weight=norm_weight,
            residual_out=residual_out,
            y_out=y_out,
            post_out=post_out,
            comb_out=comb_out,
            attn_sink=torch.zeros((heads,), device=device, dtype=torch.float32),
        )

        def restore_transaction() -> None:
            compressed_cache.copy_(initial_compressed)
            index_cache.copy_(initial_index)
            for state, initial in zip(
                (main_kv_state, main_score_state, index_kv_state, index_score_state),
                initial_states,
            ):
                state.copy_(initial)

        assert layer.selector_binding.shared_page_table
        assert layer.selector_binding.scratch.route == "packed_contiguous"
        run_deepseek_v4_c4_continuation_attention_layer(layer)
        torch.cuda.synchronize(device)
        eager = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        eager_compressed = compressed_cache.clone()
        eager_index = index_cache.clone()
        eager_selected = layer.arena_binding.selected_indices.clone()
        eager_states = tuple(
            state.clone()
            for state in (
                main_kv_state,
                main_score_state,
                index_kv_state,
                index_score_state,
            )
        )
        assert all(bool(torch.isfinite(output.float()).all()) for output in eager)
        assert layer.attention_binding.indexed_lengths is index_cache_seqlens
        assert int(eager_selected[0, 0]) == 0
        assert torch.equal(
            torch.sort(eager_selected[1, :2]).values,
            torch.tensor([0, 1], device=device, dtype=torch.int32),
        )
        assert int(eager_selected[2, 0]) == 0
        assert torch.equal(
            eager_selected[3:],
            torch.full_like(eager_selected[3:], -1),
        )
        assert not torch.equal(compressed_cache, initial_compressed)
        assert not torch.equal(index_cache, initial_index)

        restore_transaction()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = run_deepseek_v4_c4_continuation_attention_layer(layer)
        output_ptrs = tuple(output.data_ptr() for output in graph_outputs)
        for _ in range(3):
            restore_transaction()
            graph.replay()
        torch.cuda.synchronize(device)
        assert output_ptrs == tuple(
            output.data_ptr() for output in (residual_out, post_out, comb_out, y_out)
        )
        assert torch.equal(compressed_cache, eager_compressed)
        assert torch.equal(index_cache, eager_index)
        assert torch.equal(layer.arena_binding.selected_indices, eager_selected)
        for actual, expected in zip(
            (main_kv_state, main_score_state, index_kv_state, index_score_state),
            eager_states,
        ):
            assert torch.equal(actual, expected)
        for actual, expected in zip((residual_out, post_out, comb_out, y_out), eager):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        hidden_states.mul_(-0.5).add_(0.01171875)
        residual.mul_(0.625).sub_(0.01953125)
        restore_transaction()
        graph.replay()
        torch.cuda.synchronize(device)
        replayed = tuple(
            output.clone() for output in (residual_out, post_out, comb_out, y_out)
        )
        replayed_compressed = compressed_cache.clone()
        replayed_index = index_cache.clone()
        replayed_selected = layer.arena_binding.selected_indices.clone()
        replayed_states = tuple(
            state.clone()
            for state in (
                main_kv_state,
                main_score_state,
                index_kv_state,
                index_score_state,
            )
        )
        restore_transaction()
        run_deepseek_v4_c4_continuation_attention_layer(layer)
        torch.cuda.synchronize(device)
        assert torch.equal(compressed_cache, replayed_compressed)
        assert torch.equal(index_cache, replayed_index)
        assert torch.equal(layer.arena_binding.selected_indices, replayed_selected)
        for actual, expected in zip(
            (main_kv_state, main_score_state, index_kv_state, index_score_state),
            replayed_states,
        ):
            assert torch.equal(actual, expected)
        for actual, expected in zip(
            replayed,
            (residual_out, post_out, comb_out, y_out),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
