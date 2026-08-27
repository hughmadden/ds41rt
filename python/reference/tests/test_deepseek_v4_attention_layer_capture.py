from types import SimpleNamespace

import pytest
import torch
from ds4rt_reference.deepseek_v4_attention_layer_capture import (
    DS4_LAYER_ARENA_ALIGNMENT,
    bind_deepseek_v4_attention_layer,
    bind_deepseek_v4_sliding_attention_layer,
    capture_deepseek_v4_sliding_attention_layer,
    deepseek_v4_attention_layer_arena_nbytes,
    deepseek_v4_c4_selector_scratch_nbytes,
    plan_deepseek_v4_attention_layer,
    prepare_deepseek_v4_sliding_attention_layer,
    qualify_deepseek_v4_attention_layer_contract,
    run_deepseek_v4_sliding_attention_layer,
)


def _assert_fixed_non_overlapping_arena(contract) -> None:
    previous_end = 0
    for _, offset, nbytes in contract.arena.regions():
        assert offset % DS4_LAYER_ARENA_ALIGNMENT == 0
        assert offset >= previous_end
        previous_end = offset + nbytes
    assert contract.arena.total_bytes % DS4_LAYER_ARENA_ALIGNMENT == 0
    assert contract.arena.total_bytes >= previous_end


@pytest.mark.parametrize(
    ("variant", "mode", "compression", "rows", "expected_arena_bytes"),
    [
        ("flash", "decode", 0, 1, 1_130_496),
        ("flash", "decode", 4, 16, 61_006_848),
        ("flash", "extend", 128, 2_048, 716_835_840),
        ("pro", "decode", 4, 16, 209_803_264),
    ],
)
def test_composite_layer_arena_preserves_one_fixed_tp4_path(
    variant: str,
    mode: str,
    compression: int,
    rows: int,
    expected_arena_bytes: int,
) -> None:
    contract = plan_deepseek_v4_attention_layer(
        variant=variant,
        mode=mode,
        compression=compression,
        max_rows=rows,
        source_pages=512,
        max_positions=1_048_576,
    )

    assert not contract.serving_allocates
    assert (
        contract.workspace_reuse_scope
        == "one-per-coordinator-execution-lane-across-layers"
    )
    assert contract.expert_tensor_parallel == 4
    assert not contract.expert_parallel
    assert not contract.changes_expert_tp
    assert contract.expert_global_top_k == 6
    assert contract.expert_routes_identical_across_ranks
    assert contract.expert_local_intermediate_fraction == (1, 4)
    assert contract.expert_partial_output_width == contract.hidden
    assert contract.output_projection_feeds_mhc_directly
    assert contract.selector_scratch_has_separate_owner == (compression == 4)
    assert contract.selector_outputs_physical_slots == (compression == 4)
    assert contract.status == "qualified-composite-arena-not-active"
    assert contract.arena.total_bytes == expected_arena_bytes
    _assert_fixed_non_overlapping_arena(contract)


@pytest.mark.parametrize(
    ("compression", "compressed_page_bytes"),
    [(0, 0), (4, 64 * 432), (128, 2 * 432)],
)
def test_composite_layer_threads_native_nvfp4_page_geometry(
    compression: int, compressed_page_bytes: int
) -> None:
    contract = plan_deepseek_v4_attention_layer(
        variant="flash",
        mode="decode",
        compression=compression,
        max_rows=1,
        source_pages=4,
        max_positions=256,
        cache_format="nvfp4",
    )

    assert contract.cache_format == "nvfp4"
    assert contract.main_page_bytes == 256 * 432
    assert contract.compressed_page_bytes == compressed_page_bytes


def test_target_sliding_serving_exports_use_the_composite_arena() -> None:
    kwargs = {
        "variant": "flash",
        "mode": "extend",
        "compression": 0,
        "max_rows": 2_048,
        "source_pages": 64,
        "max_positions": 16_384,
    }
    assert deepseek_v4_attention_layer_arena_nbytes(
        **kwargs
    ) == plan_deepseek_v4_attention_layer(**kwargs).arena.total_bytes
    assert callable(prepare_deepseek_v4_sliding_attention_layer)
    assert callable(capture_deepseek_v4_sliding_attention_layer)


@pytest.mark.parametrize(
    ("mode", "rows", "expected_bytes"),
    [("decode", 16, 1_413_120), ("prefill", 2_048, 42_501_120)],
)
def test_c4_selector_scratch_query_matches_the_pinned_sparkinfer_plan(
    mode: str, rows: int, expected_bytes: int
) -> None:
    assert (
        deepseek_v4_c4_selector_scratch_nbytes(
            variant="flash",
            mode=mode,
            max_rows=rows,
            source_pages=64,
        )
        == expected_bytes
    )


def test_shared_cache_size_does_not_expand_per_sequence_graph_storage() -> None:
    one_context = plan_deepseek_v4_attention_layer(
        variant="flash",
        mode="extend",
        compression=128,
        max_rows=2_048,
        source_pages=64,
        max_page_table_width=64,
        max_positions=16_384,
    )
    shared_pool = plan_deepseek_v4_attention_layer(
        variant="flash",
        mode="extend",
        compression=128,
        max_rows=2_048,
        source_pages=512,
        max_page_table_width=64,
        max_positions=16_384,
    )

    assert shared_pool.source_pages == 512
    assert shared_pool.max_page_table_width == 64
    assert shared_pool.indexed_width == one_context.indexed_width == 128
    assert shared_pool.arena.total_bytes == one_context.arena.total_bytes
    assert deepseek_v4_c4_selector_scratch_nbytes(
        variant="flash",
        mode="prefill",
        max_rows=2_048,
        source_pages=512,
        max_page_table_width=64,
    ) == deepseek_v4_c4_selector_scratch_nbytes(
        variant="flash",
        mode="prefill",
        max_rows=2_048,
        source_pages=64,
        max_page_table_width=64,
    )


def test_composite_layer_arena_specializes_flash_and_pro_without_branching_tp() -> None:
    flash = plan_deepseek_v4_attention_layer(
        variant="flash",
        mode="decode",
        compression=4,
        max_rows=16,
        source_pages=512,
        max_positions=1_048_576,
    )
    pro = plan_deepseek_v4_attention_layer(
        variant="pro",
        mode="decode",
        compression=4,
        max_rows=16,
        source_pages=512,
        max_positions=1_048_576,
    )

    assert (flash.hidden, flash.heads, flash.indexed_width) == (4_096, 64, 512)
    assert (pro.hidden, pro.heads, pro.indexed_width) == (7_168, 128, 1_024)
    assert flash.stages == pro.stages
    assert flash.expert_tensor_parallel == pro.expert_tensor_parallel == 4
    assert pro.arena.total_bytes > flash.arena.total_bytes


@pytest.mark.parametrize(
    ("variant", "scratch_bytes", "arena_bytes"),
    [
        ("flash", 63_183_872, 80_543_744),
        ("pro", 126_364_672, 160_413_696),
    ],
)
def test_dspark_width_reuses_decode_layer_arena(
    variant: str, scratch_bytes: int, arena_bytes: int
) -> None:
    contract = plan_deepseek_v4_attention_layer(
        variant=variant,
        mode="decode",
        compression=0,
        max_rows=80,
        source_pages=16,
        max_positions=1_048_576,
        swa_width=133,
    )

    assert contract.swa_width == 133
    assert contract.indexed_width == 0
    assert contract.arena.attention_scratch_bytes == scratch_bytes
    assert contract.arena.total_bytes == arena_bytes
    _assert_fixed_non_overlapping_arena(contract)


def test_composite_layer_binding_materializes_only_fixed_arena_views() -> None:
    contract = plan_deepseek_v4_attention_layer(
        variant="flash",
        mode="decode",
        compression=4,
        max_rows=16,
        source_pages=512,
        max_positions=1_048_576,
    )
    arena = torch.empty((contract.arena.total_bytes + 37,), dtype=torch.uint8)
    binding = bind_deepseek_v4_attention_layer(contract, arena=arena, tokens=3)

    assert not binding.serving_allocates
    assert binding.views_only
    assert binding.outputs_are_arena_views
    assert binding.arena.numel() == contract.arena.total_bytes
    assert binding.query.shape == (3, 64, 512)
    assert binding.index_query.shape == (3, 64, 128)
    assert binding.index_query.dtype == torch.float8_e4m3fn
    assert binding.index_head_weights.shape == (3, 64)
    assert binding.index_head_weights.dtype == torch.float32
    assert binding.selected_indices.shape == (3, 512)
    assert binding.attention_output.shape == (3, 64, 512)
    offsets = {name: offset for name, offset, _ in contract.arena.regions()}
    for name, tensor in binding.region_views():
        assert tensor.data_ptr() == arena.data_ptr() + offsets[name]


@pytest.mark.parametrize(
    ("tokens", "dtype", "extra_bytes", "match"),
    [
        (0, torch.uint8, 0, "tokens must be"),
        (17, torch.uint8, 0, "tokens must be"),
        (1, torch.float32, 0, "rank-1 torch.uint8"),
        (1, torch.uint8, -1, "needs"),
    ],
)
def test_composite_layer_binding_fails_closed(
    tokens: int,
    dtype: torch.dtype,
    extra_bytes: int,
    match: str,
) -> None:
    contract = plan_deepseek_v4_attention_layer(
        variant="flash",
        mode="decode",
        compression=0,
        max_rows=16,
        source_pages=512,
        max_positions=1_048_576,
    )
    nbytes = contract.arena.total_bytes + extra_bytes
    arena = torch.empty((nbytes,), dtype=dtype)
    with pytest.raises((ValueError, TypeError), match=match):
        bind_deepseek_v4_attention_layer(contract, arena=arena, tokens=tokens)


def test_sliding_layer_binding_runs_four_bound_stages_in_order(monkeypatch) -> None:
    from b12x.attention import compressed_sparse_mla as compressed_mla, dsv4_producer
    from b12x.gemm import wo_projection
    from b12x.norm import mhc

    contract = plan_deepseek_v4_attention_layer(
        variant="flash",
        mode="decode",
        compression=0,
        max_rows=1,
        source_pages=1,
        max_positions=1,
    )
    arena = torch.empty((contract.arena.total_bytes,), dtype=torch.uint8)
    hidden = torch.empty((1, contract.hidden), dtype=torch.bfloat16)
    calls = []

    monkeypatch.setattr(dsv4_producer, "Caps", lambda **kwargs: kwargs)
    monkeypatch.setattr(dsv4_producer, "plan", lambda caps: caps)
    monkeypatch.setattr(
        dsv4_producer,
        "bind",
        lambda plan, **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        dsv4_producer,
        "run",
        lambda *, binding: calls.append(("producer", binding.query)),
    )
    monkeypatch.setattr(compressed_mla, "Caps", lambda **kwargs: kwargs)
    monkeypatch.setattr(compressed_mla, "plan", lambda caps: caps)

    def bind_attention(plan, **kwargs):
        kwargs.pop("scratch")
        return SimpleNamespace(scratch=SimpleNamespace(), **kwargs)

    monkeypatch.setattr(compressed_mla, "bind", bind_attention)

    def run_attention(*, binding, out, **kwargs):
        assert out is layer.arena_binding.attention_output
        calls.append(("attention", binding.q))
        return out

    monkeypatch.setattr(compressed_mla, "run", run_attention)
    monkeypatch.setattr(wo_projection, "Caps", lambda **kwargs: kwargs)
    monkeypatch.setattr(wo_projection, "plan", lambda caps: caps)

    def bind_output(plan, **kwargs):
        return SimpleNamespace(output=hidden.clone(), **kwargs)

    monkeypatch.setattr(wo_projection, "bind_inv_rope", bind_output)

    def run_output(*, binding):
        calls.append(("output", binding.o))
        return binding.output

    monkeypatch.setattr(wo_projection, "run_inv_rope", run_output)
    monkeypatch.setattr(mhc, "Caps", lambda **kwargs: kwargs)
    monkeypatch.setattr(mhc, "plan", lambda caps: caps)
    monkeypatch.setattr(
        mhc,
        "bind",
        lambda plan, **kwargs: SimpleNamespace(**kwargs),
    )

    residual_out = torch.empty((1, 4, contract.hidden), dtype=torch.bfloat16)
    y_out = torch.empty_like(hidden)
    post_out = torch.empty((1, 4), dtype=torch.float32)
    comb_out = torch.empty((1, 4, 4), dtype=torch.float32)
    expected = (residual_out, post_out, comb_out, y_out)

    def run_mhc(x, *args, binding, **kwargs):
        assert x is layer.output_projection
        calls.append(("mhc", x))
        return expected

    monkeypatch.setattr(mhc, "run_post_pre", run_mhc)
    positions = torch.zeros((1,), dtype=torch.int32)
    layer = bind_deepseek_v4_sliding_attention_layer(
        contract,
        arena=arena,
        hidden_states=hidden,
        positions=positions,
        main_slots=positions,
        cos_sin_cache=torch.empty((1, 64), dtype=torch.float32),
        main_kv_cache=torch.empty((1, 149_760), dtype=torch.uint8),
        swa_indices=torch.zeros((1, 128), dtype=torch.int32),
        swa_lengths=torch.ones((1,), dtype=torch.int32),
        producer_weights=object(),
        output_weights=object(),
        residual=residual_out,
        prev_post=post_out,
        prev_comb=comb_out,
        fn=torch.empty((24, 4 * contract.hidden), dtype=torch.float32),
        hc_scale=torch.empty((3,), dtype=torch.float32),
        hc_base=torch.empty((24,), dtype=torch.float32),
        norm_weight=torch.empty((contract.hidden,), dtype=torch.bfloat16),
        residual_out=residual_out,
        y_out=y_out,
        post_out=post_out,
        comb_out=comb_out,
    )

    assert not layer.serving_allocates
    assert layer.cuda_graph_safe
    assert layer.uses_one_lane_arena
    assert layer.persistent_state_is_external
    assert layer.expert_tensor_parallel == 4
    assert not layer.expert_parallel
    outputs = run_deepseek_v4_sliding_attention_layer(layer)
    assert all(actual is wanted for actual, wanted in zip(outputs, expected))
    assert [name for name, _ in calls] == ["producer", "attention", "output", "mhc"]
    assert calls[0][1] is layer.arena_binding.query
    assert calls[1][1] is layer.arena_binding.query
    assert calls[2][1] is layer.arena_binding.attention_output


def test_sliding_layer_binding_rejects_indexed_compression() -> None:
    contract = plan_deepseek_v4_attention_layer(
        variant="flash",
        mode="decode",
        compression=4,
        max_rows=1,
        source_pages=1,
        max_positions=1,
    )
    with pytest.raises(ValueError, match="requires compression=0"):
        bind_deepseek_v4_sliding_attention_layer(
            contract,
            arena=torch.empty((contract.arena.total_bytes,), dtype=torch.uint8),
            hidden_states=torch.empty((1, contract.hidden), dtype=torch.bfloat16),
            positions=None,
            main_slots=None,
            cos_sin_cache=None,
            main_kv_cache=None,
            swa_indices=None,
            swa_lengths=None,
            producer_weights=None,
            output_weights=None,
            residual=None,
            prev_post=None,
            prev_comb=None,
            fn=None,
            hc_scale=None,
            hc_base=None,
            norm_weight=None,
            residual_out=None,
            y_out=None,
            post_out=None,
            comb_out=None,
        )


def test_c128_decode_execution_lifecycle_is_startup_qualified() -> None:
    assert qualify_deepseek_v4_attention_layer_contract(
        variant="flash",
        mode="decode",
        compression=128,
        max_rows=1,
        source_pages=1,
        max_positions=256,
    )


def test_c128_initial_prefill_execution_lifecycle_is_startup_qualified() -> None:
    from ds4rt_reference.deepseek_v4_attention_layer_capture import (
        DeepseekV4C128PrefillAttentionLayerBinding,
        bind_deepseek_v4_c128_prefill_attention_layer,
        run_deepseek_v4_c128_prefill_attention_layer,
    )

    assert qualify_deepseek_v4_attention_layer_contract(
        variant="flash",
        mode="extend",
        compression=128,
        max_rows=128,
        source_pages=1,
        max_positions=256,
    )
    assert not DeepseekV4C128PrefillAttentionLayerBinding.serving_allocates
    assert DeepseekV4C128PrefillAttentionLayerBinding.cuda_graph_safe
    assert DeepseekV4C128PrefillAttentionLayerBinding.uses_one_lane_arena
    assert DeepseekV4C128PrefillAttentionLayerBinding.persistent_state_is_external
    assert DeepseekV4C128PrefillAttentionLayerBinding.initial_prefill_only
    assert (
        DeepseekV4C128PrefillAttentionLayerBinding.scheduler_owns_physical_selection
    )
    assert DeepseekV4C128PrefillAttentionLayerBinding.expert_tensor_parallel == 4
    assert not DeepseekV4C128PrefillAttentionLayerBinding.expert_parallel
    assert callable(bind_deepseek_v4_c128_prefill_attention_layer)
    assert callable(run_deepseek_v4_c128_prefill_attention_layer)


def test_c128_continuation_execution_lifecycle_is_startup_qualified() -> None:
    from ds4rt_reference.deepseek_v4_attention_layer_capture import (
        DeepseekV4C128ContinuationAttentionLayerBinding,
        bind_deepseek_v4_c128_continuation_attention_layer,
        run_deepseek_v4_c128_continuation_attention_layer,
    )

    assert qualify_deepseek_v4_attention_layer_contract(
        variant="flash",
        mode="extend",
        compression=128,
        max_rows=32,
        source_pages=3,
        max_positions=768,
    )
    binding = DeepseekV4C128ContinuationAttentionLayerBinding
    assert not binding.serving_allocates
    assert binding.cuda_graph_safe
    assert binding.uses_one_lane_arena
    assert binding.persistent_state_is_external
    assert binding.ordered_chunks_only
    assert binding.scheduler_owns_state_transactions
    assert binding.scheduler_owns_physical_selection
    assert binding.expert_tensor_parallel == 4
    assert not binding.expert_parallel
    assert callable(bind_deepseek_v4_c128_continuation_attention_layer)
    assert callable(run_deepseek_v4_c128_continuation_attention_layer)


def test_c4_initial_prefill_execution_lifecycle_is_startup_qualified() -> None:
    from ds4rt_reference.deepseek_v4_attention_layer_capture import (
        DeepseekV4C4PrefillAttentionLayerBinding,
        bind_deepseek_v4_c4_prefill_attention_layer,
        run_deepseek_v4_c4_prefill_attention_layer,
    )

    assert qualify_deepseek_v4_attention_layer_contract(
        variant="flash",
        mode="extend",
        compression=4,
        max_rows=512,
        source_pages=2,
        max_positions=512,
    )
    binding = DeepseekV4C4PrefillAttentionLayerBinding
    assert not binding.serving_allocates
    assert binding.cuda_graph_safe
    assert binding.uses_one_lane_arena
    assert binding.persistent_state_is_external
    assert binding.selector_scratch_is_external
    assert binding.selector_outputs_physical_slots
    assert binding.selector_uses_shared_page_table
    assert binding.selector_uses_causal_lengths
    assert binding.initial_prefill_only
    assert binding.expert_tensor_parallel == 4
    assert not binding.expert_parallel
    assert callable(bind_deepseek_v4_c4_prefill_attention_layer)
    assert callable(run_deepseek_v4_c4_prefill_attention_layer)


def test_c4_continuation_execution_lifecycle_is_startup_qualified() -> None:
    from ds4rt_reference.deepseek_v4_attention_layer_capture import (
        DeepseekV4C4ContinuationAttentionLayerBinding,
        bind_deepseek_v4_c4_continuation_attention_layer,
        run_deepseek_v4_c4_continuation_attention_layer,
    )

    assert qualify_deepseek_v4_attention_layer_contract(
        variant="flash",
        mode="extend",
        compression=4,
        max_rows=8,
        source_pages=3,
        max_positions=768,
    )
    binding = DeepseekV4C4ContinuationAttentionLayerBinding
    assert not binding.serving_allocates
    assert binding.cuda_graph_safe
    assert binding.uses_one_lane_arena
    assert binding.persistent_state_is_external
    assert binding.selector_scratch_is_external
    assert binding.selector_outputs_physical_slots
    assert binding.selector_uses_shared_page_table
    assert binding.selector_uses_causal_lengths
    assert binding.ordered_chunks_only
    assert binding.scheduler_owns_state_transactions
    assert binding.expert_tensor_parallel == 4
    assert not binding.expert_parallel
    assert callable(bind_deepseek_v4_c4_continuation_attention_layer)
    assert callable(run_deepseek_v4_c4_continuation_attention_layer)


@pytest.mark.parametrize("variant", ["flash", "pro"])
def test_composite_layer_contract_matches_every_pinned_leaf(variant: str) -> None:
    assert qualify_deepseek_v4_attention_layer_contract(
        variant=variant,
        mode="decode",
        compression=4,
        max_rows=16,
        source_pages=512,
        max_positions=1_048_576,
    )


def test_composite_layer_contract_fails_closed_on_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="max_positions must be positive"):
        plan_deepseek_v4_attention_layer(
            variant="flash",
            mode="decode",
            compression=0,
            max_rows=1,
            source_pages=512,
            max_positions=0,
        )
