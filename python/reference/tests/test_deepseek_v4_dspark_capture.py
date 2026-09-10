import pytest
import torch

import ds41rt_reference.deepseek_v4_dspark_capture as dspark_capture
from ds41rt_reference.deepseek_v4_dspark_capture import (
    DS4_DSPARK_CACHE_PAGE_BYTES,
    DS4_DSPARK_ENTRY_ACTIVATION_BLOCK_SIZE,
    DS4_DSPARK_NOISE_TOKEN_ID,
    DeepseekV4DsparkBlockBinding,
    DeepseekV4DsparkContract,
    DeepseekV4DsparkPromptPrimeBlockBinding,
    DeepseekV4DsparkProposalEntryBinding,
    bind_deepseek_v4_dspark_block,
    bind_deepseek_v4_dspark_prompt_prime_block,
    bind_deepseek_v4_dspark_proposal_entry,
    deepseek_v4_dspark_arena_nbytes,
    deepseek_v4_dspark_arena_region_nbytes,
    deepseek_v4_dspark_arena_region_offset,
    deepseek_v4_dspark_block_buffer_nbytes,
    deepseek_v4_dspark_block_buffer_offset,
    deepseek_v4_dspark_entry_buffer_nbytes,
    deepseek_v4_dspark_entry_buffer_offset,
    deepseek_v4_dspark_persistent_kv_nbytes,
    deepseek_v4_dspark_prompt_buffer_nbytes,
    deepseek_v4_dspark_prompt_buffer_offset,
    deepseek_v4_dspark_proposal_buffer_nbytes,
    deepseek_v4_dspark_proposal_buffer_offset,
    plan_deepseek_v4_dspark,
    qualify_deepseek_v4_dspark_contract,
    run_deepseek_v4_dspark_block_post_dispatch,
    run_deepseek_v4_dspark_block_pre_dispatch,
    run_deepseek_v4_dspark_prompt_prime_block,
    run_deepseek_v4_dspark_proposal_entry,
    run_deepseek_v4_dspark_terminal_reference,
)


@pytest.mark.parametrize(
    (
        "variant",
        "hidden",
        "target_layers",
        "target_taps",
        "physical_blocks",
        "experts",
        "intermediate",
        "heads",
        "markov_rank",
    ),
    [
        ("flash", 4_096, 43, (40, 41, 42), (43, 44, 45), 256, 2_048, 64, 256),
        ("pro", 7_168, 61, (58, 59, 60), (61, 62, 63), 384, 3_072, 128, 512),
    ],
)
def test_dspark_geometry_uses_mtp_only_as_checkpoint_namespace(
    variant: str,
    hidden: int,
    target_layers: int,
    target_taps: tuple[int, int, int],
    physical_blocks: tuple[int, int, int],
    experts: int,
    intermediate: int,
    heads: int,
    markov_rank: int,
) -> None:
    contract = plan_deepseek_v4_dspark(
        variant=variant, max_batch=4, max_main_rows=2_048
    )
    geometry = contract.geometry
    assert geometry.hidden == hidden
    assert geometry.target_layers == target_layers
    assert geometry.target_taps == target_taps
    assert geometry.physical_block_ids == physical_blocks
    assert geometry.storage_prefixes == ("mtp.0", "mtp.1", "mtp.2")
    assert geometry.routed_experts == experts
    assert geometry.expert_intermediate == intermediate
    assert geometry.attention_heads == heads
    assert geometry.markov_rank == markov_rank
    assert geometry.target_projection_input_width == 3 * hidden
    assert geometry.proposal_hc_width == 4 * hidden
    for index, block in enumerate(contract.blocks):
        assert block.physical_layer_id == physical_blocks[index]
        assert block.target_tap_id == target_taps[index]
        assert block.storage_prefix == f"mtp.{index}"
        assert not block.checkpoint_namespace_is_algorithm_name
        assert block.attention_semantics == "dual-source-dspark-sliding-attention"


def test_dspark_lifecycle_is_integrated_and_keeps_strict_tp4() -> None:
    contract = plan_deepseek_v4_dspark(
        variant="flash", max_batch=2, max_main_rows=128
    )
    assert isinstance(contract, DeepseekV4DsparkContract)
    assert not contract.serving_allocates
    assert contract.target_taps_are_caller_owned
    assert contract.target_projection_is_shared_by_all_blocks
    assert contract.cache_is_request_local
    assert contract.cache_device_id == 0
    assert contract.proposal_input_token_pattern == (
        "anchor",
        "noise",
        "noise",
        "noise",
        "noise",
    )
    assert contract.geometry.noise_token_id == DS4_DSPARK_NOISE_TOKEN_ID
    assert contract.dispatch_barriers_per_decode == 3
    assert contract.cuda_graph_segments_per_decode == 8
    assert contract.sparse_workspace_reused_across_blocks
    assert contract.entry_scratch_reuses_sparse_workspace
    assert contract.entry_activation_block_size == DS4_DSPARK_ENTRY_ACTIVATION_BLOCK_SIZE
    assert contract.entry_capture_surface == (
        "prepare_deepseek_v4_dspark_entry_projection",
        "capture_deepseek_v4_dspark_entry_projection",
    )
    assert contract.target_main_kv_producer_scratch_reused_across_blocks
    assert contract.proposal_scratch_reuses_markov_storage
    assert not contract.prompt_prime_computes_query
    assert contract.prompt_prime_producer_surface == ("plan_kv", "bind_kv", "run_kv")
    assert contract.prompt_prime_capture_surface == (
        "prepare_deepseek_v4_dspark_prompt_prime",
        "capture_deepseek_v4_dspark_prompt_prime",
    )
    assert contract.proposal_entry_capture_surface == (
        "prepare_deepseek_v4_dspark_proposal_entry",
        "capture_deepseek_v4_dspark_proposal_entry",
    )
    assert contract.block_attention_capture_surface == (
        "prepare_deepseek_v4_dspark_block_attention",
        "capture_deepseek_v4_dspark_block_attention",
    )
    assert contract.block_post_dispatch_capture_surface == (
        "prepare_deepseek_v4_dspark_block_post_dispatch",
        "capture_deepseek_v4_dspark_block_post_dispatch",
    )
    assert contract.terminal_collapse_capture_surface == (
        "prepare_deepseek_v4_dspark_terminal_collapse",
        "capture_deepseek_v4_dspark_terminal_collapse",
    )
    assert contract.decode_attention.target_ring_slots == 128
    assert contract.decode_attention.proposal_slots == (128, 129, 130, 131, 132)
    assert contract.decode_attention.selection_width == 133
    assert contract.decode_attention.cache_page_tokens == 256
    assert contract.decode_attention.cache_page_bytes == 149_760
    assert contract.decode_attention.all_proposal_kv_visible_to_every_query
    assert not contract.decode_attention.proposal_attention_is_causal
    assert contract.expert_tensor_parallel == 4
    assert not contract.expert_parallel
    assert not contract.recurrent_mtp_state
    assert contract.terminal_batch_axis == "active-requests-times-five-proposal-rows"
    assert contract.terminal_serial_dependency_axis == "five-markov-positions-only"
    assert contract.terminal_preserves_unnormalized_hidden
    assert contract.terminal_jointly_issues_active_requests
    assert contract.status == "qualified-integrated-dspark-composite-not-active"
    for block in contract.blocks:
        assert block.uses_target_main_kv
        assert block.proposal_queries_include_proposal_kv
        assert block.expert_handoff.expert_tensor_parallel == 4
        assert not block.expert_handoff.expert_parallel
        assert block.expert_handoff.dispatch_barrier_between_graphs
        assert block.expert_handoff.one_route_buffer_fans_out_to_all_ranks
        assert block.expert_handoff.attention.swa_width == 133


def test_dspark_terminal_oracle_batches_requests_and_serializes_only_positions() -> None:
    requests, proposal_tokens, hc_mult, hidden = 3, 5, 4, 8
    vocab, markov_rank = 11, 3
    generator = torch.Generator(device="cpu")
    generator.manual_seed(94_205)
    residual = torch.randn(
        (requests, proposal_tokens, hc_mult, hidden),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    anchors = torch.tensor([1, 4, 7], dtype=torch.long)
    hc_fn = torch.randn(
        (hc_mult, hc_mult * hidden), generator=generator, dtype=torch.float32
    ) / 8
    hc_scale = torch.tensor([0.7], dtype=torch.float32)
    hc_base = torch.linspace(-0.2, 0.2, hc_mult, dtype=torch.float32)
    norm_weight = torch.linspace(0.8, 1.2, hidden, dtype=torch.float32).to(
        torch.bfloat16
    )
    shared_head = torch.randn(
        (vocab, hidden), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    markov_w1 = torch.randn(
        (vocab, markov_rank), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    markov_w2 = torch.randn(
        (vocab, markov_rank), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    confidence = torch.randn(
        (1, hidden + markov_rank), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)

    actual = run_deepseek_v4_dspark_terminal_reference(
        residual,
        anchors,
        hc_head_fn=hc_fn,
        hc_head_scale=hc_scale,
        hc_head_base=hc_base,
        norm_weight=norm_weight,
        shared_head_weight=shared_head,
        markov_w1=markov_w1,
        markov_w2=markov_w2,
        confidence_weight=confidence,
    )

    assert actual.output_token_ids.shape == (requests, proposal_tokens + 1)
    assert torch.equal(actual.output_token_ids[:, 0], anchors)
    assert actual.logits.shape == (requests, proposal_tokens, vocab)
    assert actual.collapsed_hidden.shape == (requests, proposal_tokens, hidden)
    assert actual.normalized_hidden.shape == (requests, proposal_tokens, hidden)
    assert actual.markov_embeddings.shape == (
        requests,
        proposal_tokens,
        markov_rank,
    )
    assert actual.confidence_logits.shape == (requests, proposal_tokens)
    assert bool(torch.all((0.0 <= actual.conditional_confidence)))
    assert bool(torch.all((actual.conditional_confidence <= 1.0)))

    for position in range(proposal_tokens):
        conditioning_ids = actual.output_token_ids[:, position]
        expected_markov = markov_w1[conditioning_ids]
        torch.testing.assert_close(
            actual.markov_embeddings[:, position], expected_markov
        )
        expected_bias = expected_markov.float() @ markov_w2.float().T
        base_logits = (
            actual.normalized_hidden[:, position].float()
            @ shared_head.float().T
        )
        torch.testing.assert_close(
            actual.logits[:, position], base_logits + expected_bias
        )
        assert torch.equal(
            actual.output_token_ids[:, position + 1],
            actual.logits[:, position].argmax(dim=-1),
        )

    confidence_inputs = torch.cat(
        [actual.collapsed_hidden.float(), actual.markov_embeddings.float()],
        dim=-1,
    )
    expected_confidence_logits = (
        confidence_inputs @ confidence.float().T
    ).squeeze(-1)
    torch.testing.assert_close(actual.confidence_logits, expected_confidence_logits)
    torch.testing.assert_close(
        actual.conditional_confidence, torch.sigmoid(expected_confidence_logits)
    )


def test_dspark_arena_is_preallocated_and_reuses_one_sparse_workspace() -> None:
    contract = plan_deepseek_v4_dspark(
        variant="flash", max_batch=2, max_main_rows=2_048
    )
    names = tuple(region.name for region in contract.arena.regions)
    assert names == (
        "projected_target_main",
        "reused_target_main_kv_producer_scratch",
        "draft_input_token_ids",
        "proposal_residual_ping_pong",
        "proposal_collapsed_hidden",
        "proposal_normalized_hidden",
        "terminal_compact_normalized_hidden",
        "shared_lm_logits",
        "terminal_markov_logits",
        "markov_embeddings",
        "confidence",
        "terminal_active_slot_ids",
        "terminal_anchor_token_ids",
        "terminal_output_token_ids",
        "reused_dspark_attention_selected_indices",
        "reused_dspark_attention_selected_lengths",
        "reused_sparse_block_workspace",
    )
    previous_end = 0
    for region in contract.arena.regions:
        assert region.offset % 256 == 0
        assert region.offset >= previous_end
        assert region.nbytes > 0
        previous_end = region.offset + region.nbytes
    assert contract.arena.total_bytes % 256 == 0
    assert contract.arena.total_bytes >= previous_end
    assert (
        contract.arena.region("projected_target_main").nbytes
        == contract.max_main_rows * contract.geometry.hidden * 2
    )
    assert contract.arena.persistent_kv_bytes == 3 * 2 * DS4_DSPARK_CACHE_PAGE_BYTES
    assert (
        contract.arena.region("proposal_normalized_hidden").nbytes
        == 2 * 5 * contract.geometry.hidden * 2
    )
    assert (
        contract.arena.region("terminal_compact_normalized_hidden").nbytes
        == 2 * 5 * contract.geometry.hidden * 2
    )
    assert (
        contract.arena.region("terminal_markov_logits").nbytes
        == 2 * contract.geometry.vocab_size * 4
    )
    assert contract.arena.region("terminal_active_slot_ids").nbytes == 2 * 4
    assert contract.arena.region("terminal_anchor_token_ids").nbytes == 2 * 4
    assert contract.arena.region("terminal_output_token_ids").nbytes == 2 * 6 * 4
    assert (
        contract.arena.region("reused_sparse_block_workspace").nbytes
        == max(
            contract.entry_scratch.total_bytes,
            contract.blocks[0].expert_handoff.arena.total_bytes
            + contract.decode_attention.workspace_growth_bytes,
        )
    )
    kv_scratch = contract.target_main_kv_producer_scratch
    assert kv_scratch.kv_linear_offset == 0
    assert kv_scratch.kv_linear_bytes == 8_912_896
    assert kv_scratch.kv_output_offset == 8_912_896
    assert kv_scratch.kv_output_bytes == 2_097_152
    assert kv_scratch.total_bytes == 11_010_048
    assert (
        contract.arena.region("reused_target_main_kv_producer_scratch").nbytes
        == kv_scratch.total_bytes
    )
    entry = contract.entry_scratch
    assert entry.target_tap_concat_offset == 0
    assert entry.target_tap_concat_bytes == 50_331_648
    assert entry.projection_scratch_offset == 50_331_648
    assert entry.projection_scratch_bytes == 26_738_688
    assert entry.prompt_positions_offset == 0
    assert entry.prompt_positions_bytes == 8_192
    assert entry.prompt_main_slots_offset == 8_192
    assert entry.prompt_main_slots_bytes == 8_192
    assert entry.prompt_cos_sin_offset == 16_384
    assert entry.prompt_cos_sin_bytes == 524_288
    assert (
        entry.prompt_cos_sin_offset + entry.prompt_cos_sin_bytes
        <= entry.target_tap_concat_bytes
    )
    assert entry.total_bytes == 77_070_336
    assert (
        entry.total_bytes
        <= contract.arena.region("reused_sparse_block_workspace").nbytes
    )


@pytest.mark.parametrize("max_batch", [1, 2, 4, 16])
def test_dspark_proposal_scratch_overlays_pre_head_storage(max_batch: int) -> None:
    kwargs = {"variant": "flash", "max_batch": max_batch, "max_main_rows": 2_048}
    contract = plan_deepseek_v4_dspark(**kwargs)
    scratch = contract.proposal_scratch
    rows = max_batch * contract.geometry.proposal_tokens

    token_ids = contract.arena.region("draft_input_token_ids")
    residual = contract.arena.region("proposal_residual_ping_pong")
    collapsed = contract.arena.region("proposal_collapsed_hidden")
    markov = contract.arena.region("markov_embeddings")
    selected_indices = contract.arena.region(
        "reused_dspark_attention_selected_indices"
    )
    selected_lengths = contract.arena.region(
        "reused_dspark_attention_selected_lengths"
    )

    assert scratch.buffer("draft_input_token_ids") == (token_ids.offset, rows * 4)
    assert scratch.buffer("residual_ping") == (
        residual.offset,
        rows * 4 * contract.geometry.hidden * 2,
    )
    assert scratch.residual_pong_offset == (
        scratch.residual_ping_offset + scratch.residual_ping_bytes
    )
    assert scratch.residual_pong_offset + scratch.residual_pong_bytes == (
        residual.offset + residual.nbytes
    )
    assert scratch.buffer("collapsed_hidden") == (
        collapsed.offset,
        collapsed.nbytes,
    )
    assert scratch.buffer("selected_indices") == (
        selected_indices.offset,
        selected_indices.nbytes,
    )
    assert scratch.buffer("selected_lengths") == (
        selected_lengths.offset,
        selected_lengths.nbytes,
    )

    overlay_names = (
        "positions",
        "main_slots",
        "cos_sin_cache",
        "post_ping",
        "comb_ping",
        "post_pong",
        "comb_pong",
    )
    previous_end = markov.offset
    for name in overlay_names:
        offset, nbytes = scratch.buffer(name)
        assert offset % 16 == 0
        assert offset >= previous_end
        assert nbytes > 0
        previous_end = offset + nbytes
        assert deepseek_v4_dspark_proposal_buffer_offset(
            **kwargs, buffer=name
        ) == offset
        assert deepseek_v4_dspark_proposal_buffer_nbytes(
            **kwargs, buffer=name
        ) == nbytes
    assert previous_end <= markov.offset + markov.nbytes


def test_flash_dspark_proposal_scratch_has_pinned_production_layout() -> None:
    contract = plan_deepseek_v4_dspark(
        variant="flash", max_batch=16, max_main_rows=2_048
    )
    scratch = contract.proposal_scratch
    markov = contract.arena.region("markov_embeddings")
    expected_markov_views = {
        "positions": (0, 320),
        "main_slots": (320, 320),
        "cos_sin_cache": (640, 20_480),
        "post_ping": (21_120, 1_280),
        "comb_ping": (22_400, 5_120),
        "post_pong": (27_520, 1_280),
        "comb_pong": (28_800, 5_120),
    }
    for name, (relative_offset, nbytes) in expected_markov_views.items():
        assert scratch.buffer(name) == (markov.offset + relative_offset, nbytes)
    assert markov.nbytes == 40_960


def test_pro_dspark_pins_kv_only_prompt_prime_workspace() -> None:
    contract = plan_deepseek_v4_dspark(
        variant="pro", max_batch=16, max_main_rows=2_048
    )

    scratch = contract.target_main_kv_producer_scratch
    assert scratch.kv_linear_offset == 0
    assert scratch.kv_linear_bytes == 15_597_568
    assert scratch.kv_output_offset == 15_597_568
    assert scratch.kv_output_bytes == 2_097_152
    assert scratch.total_bytes == 17_694_720
    entry = contract.entry_scratch
    assert entry.target_tap_concat_bytes == 88_080_384
    assert entry.projection_scratch_bytes == 46_792_704
    assert entry.total_bytes == 134_873_088
    assert (
        entry.total_bytes
        <= contract.arena.region("reused_sparse_block_workspace").nbytes
    )


@pytest.mark.parametrize(
    ("variant", "arena_bytes"),
    [("flash", 168_077_824), ("pro", 274_719_232)],
)
def test_dspark_startup_exports_exact_device_storage_sizes(
    variant: str, arena_bytes: int
) -> None:
    kwargs = {"variant": variant, "max_batch": 16, "max_main_rows": 2_048}
    assert deepseek_v4_dspark_arena_nbytes(**kwargs) == arena_bytes
    assert deepseek_v4_dspark_persistent_kv_nbytes(**kwargs) == 3 * 16 * 149_760
    contract = plan_deepseek_v4_dspark(**kwargs)
    workspace = contract.arena.region("reused_sparse_block_workspace")
    assert (
        deepseek_v4_dspark_arena_region_offset(
            **kwargs, region="reused_sparse_block_workspace"
        )
        == workspace.offset
    )
    assert (
        deepseek_v4_dspark_arena_region_nbytes(
            **kwargs, region="reused_sparse_block_workspace"
        )
        == workspace.nbytes
    )
    assert (
        deepseek_v4_dspark_entry_buffer_offset(
            **kwargs, buffer="projection_scratch"
        )
        == contract.entry_scratch.projection_scratch_offset
    )
    assert (
        deepseek_v4_dspark_entry_buffer_nbytes(
            **kwargs, buffer="projection_scratch"
        )
        == contract.entry_scratch.projection_scratch_bytes
    )
    for name, offset, nbytes in (
        (
            "positions",
            contract.entry_scratch.prompt_positions_offset,
            contract.entry_scratch.prompt_positions_bytes,
        ),
        (
            "main_slots",
            contract.entry_scratch.prompt_main_slots_offset,
            contract.entry_scratch.prompt_main_slots_bytes,
        ),
        (
            "cos_sin_cache",
            contract.entry_scratch.prompt_cos_sin_offset,
            contract.entry_scratch.prompt_cos_sin_bytes,
        ),
    ):
        assert deepseek_v4_dspark_prompt_buffer_offset(**kwargs, buffer=name) == offset
        assert deepseek_v4_dspark_prompt_buffer_nbytes(**kwargs, buffer=name) == nbytes

    sparse_regions = contract.blocks[0].expert_handoff.arena.regions()
    for name, offset, nbytes in sparse_regions:
        assert deepseek_v4_dspark_block_buffer_offset(
            **kwargs, buffer=name
        ) == offset
        assert deepseek_v4_dspark_block_buffer_nbytes(
            **kwargs, buffer=name
        ) == nbytes


def test_dspark_native_nvfp4_persistent_pages_use_exact_record_width() -> None:
    contract = plan_deepseek_v4_dspark(
        variant="flash",
        max_batch=16,
        max_main_rows=2_048,
        cache_format="nvfp4",
    )

    assert contract.decode_attention.cache_format == "nvfp4"
    assert contract.decode_attention.cache_page_bytes == 256 * 432
    assert contract.arena.persistent_kv_bytes == 3 * 16 * 256 * 432
    assert deepseek_v4_dspark_persistent_kv_nbytes(
        variant="flash",
        max_batch=16,
        max_main_rows=2_048,
        cache_format="nvfp4",
    ) == 3 * 16 * 256 * 432


@pytest.mark.parametrize(
    (
        "variant",
        "heads",
        "scratch_bytes",
        "base_scratch_bytes",
        "workspace_growth_bytes",
    ),
    [
        ("flash", 64, 63_183_872, 63_183_872, 0),
        ("pro", 128, 126_364_672, 126_364_672, 0),
    ],
)
def test_dspark_decode_attention_pins_full_dual_source_selection(
    variant: str,
    heads: int,
    scratch_bytes: int,
    base_scratch_bytes: int,
    workspace_growth_bytes: int,
) -> None:
    contract = plan_deepseek_v4_dspark(
        variant=variant, max_batch=16, max_main_rows=2_048
    )
    attention = contract.decode_attention

    assert attention.rows == 80
    assert attention.heads == heads
    assert attention.max_chunks_per_row == 12
    assert attention.scratch_bytes == scratch_bytes
    assert attention.base_attention_scratch_bytes == base_scratch_bytes
    assert attention.workspace_growth_bytes == workspace_growth_bytes
    assert attention.selected_indices_bytes == 80 * 133 * 4
    assert attention.selected_lengths_bytes == 80 * 4
    assert (
        contract.arena.region("reused_dspark_attention_selected_indices").nbytes
        == attention.selected_indices_bytes
    )
    assert (
        contract.arena.region("reused_dspark_attention_selected_lengths").nbytes
        == attention.selected_lengths_bytes
    )
    assert contract.arena.persistent_kv_bytes == 3 * 16 * 149_760


def test_dspark_selection_uses_request_page_and_noncausal_proposal_suffix() -> None:
    attention = plan_deepseek_v4_dspark(
        variant="flash", max_batch=4, max_main_rows=128
    ).decode_attention

    partial = attention.physical_selection(
        request_index=1,
        cache_window_start=6,
        main_context_end=9,
    )
    assert partial == (262, 263, 264, 384, 385, 386, 387, 388)
    full = attention.physical_selection(
        request_index=2,
        cache_window_start=130,
        main_context_end=258,
    )
    assert full[:128] == tuple(range(512, 640))
    assert full[128:] == (640, 641, 642, 643, 644)
    assert len(full) == 133

    with pytest.raises(ValueError, match="request index"):
        attention.physical_selection(
            request_index=4,
            cache_window_start=0,
            main_context_end=1,
        )
    with pytest.raises(ValueError, match="exceeds"):
        attention.physical_selection(
            request_index=0,
            cache_window_start=0,
            main_context_end=129,
        )


def test_dspark_prompt_prime_binds_and_runs_kv_only_producer(monkeypatch) -> None:
    from types import SimpleNamespace

    from b12x.attention import dsv4_producer

    contract = plan_deepseek_v4_dspark(
        variant="flash", max_batch=2, max_main_rows=8
    )
    calls = []
    monkeypatch.setattr(dsv4_producer, "Caps", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        dsv4_producer,
        "plan_kv",
        lambda caps: calls.append(("plan", caps)) or SimpleNamespace(caps=caps),
    )
    monkeypatch.setattr(
        dsv4_producer,
        "bind_kv",
        lambda plan, **kwargs: calls.append(("bind", kwargs))
        or SimpleNamespace(**kwargs),
    )
    expected = object()
    monkeypatch.setattr(
        dsv4_producer,
        "run_kv",
        lambda *, binding: calls.append(("run", binding)) or expected,
    )
    projected = torch.empty((3, 4_096), dtype=torch.bfloat16)
    scratch = torch.empty(
        (contract.target_main_kv_producer_scratch.total_bytes,), dtype=torch.uint8
    )
    binding = bind_deepseek_v4_dspark_prompt_prime_block(
        contract,
        block_index=1,
        scratch=scratch,
        projected_target_main=projected,
        positions=object(),
        main_slots=object(),
        cos_sin_cache=object(),
        main_kv_cache=object(),
        producer_weights=object(),
    )

    assert isinstance(binding, DeepseekV4DsparkPromptPrimeBlockBinding)
    assert binding.block.storage_prefix == "mtp.1"
    assert not binding.serving_allocates
    assert binding.cuda_graph_safe
    assert not binding.computes_query
    assert calls[0][1]["max_tokens"] == 8
    assert calls[1][1]["expected_m"] == 8
    assert run_deepseek_v4_dspark_prompt_prime_block(binding) is expected
    assert [call[0] for call in calls] == ["plan", "bind", "run"]


def test_dspark_proposal_entry_is_block_zero_mhc_pre_not_recurrent_mtp(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from b12x.norm import mhc

    contract = plan_deepseek_v4_dspark(
        variant="flash", max_batch=2, max_main_rows=8
    )
    rows = contract.geometry.proposal_tokens
    hidden = contract.geometry.hidden
    calls = []
    monkeypatch.setattr(mhc, "Caps", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        mhc,
        "plan",
        lambda caps: calls.append(("plan", caps)) or SimpleNamespace(caps=caps),
    )
    expected_binding = SimpleNamespace()
    monkeypatch.setattr(
        mhc,
        "bind",
        lambda plan, **kwargs: calls.append(("bind", kwargs)) or expected_binding,
    )
    expected = (object(), object(), object(), object())
    monkeypatch.setattr(
        mhc,
        "run_pre",
        lambda *args, **kwargs: calls.append(("run-pre", args, kwargs)) or expected,
    )
    residual_in = torch.empty((rows, hidden), dtype=torch.bfloat16)
    residual_out = torch.empty((rows, 4, hidden), dtype=torch.bfloat16)
    normalized = torch.empty((rows, hidden), dtype=torch.bfloat16)
    post = torch.empty((rows, 4), dtype=torch.float32)
    comb = torch.empty((rows, 4, 4), dtype=torch.float32)
    binding = bind_deepseek_v4_dspark_proposal_entry(
        contract,
        scratch=torch.empty((32_000,), dtype=torch.uint8),
        residual_input=residual_in,
        residual_output=residual_out,
        normalized_output=normalized,
        post_output=post,
        comb_output=comb,
        hc_fn=torch.empty((24, hidden), dtype=torch.float32),
        hc_scale=torch.empty((3,), dtype=torch.float32),
        hc_base=torch.empty((24,), dtype=torch.float32),
        norm_weight=torch.empty((hidden,), dtype=torch.bfloat16),
    )

    assert isinstance(binding, DeepseekV4DsparkProposalEntryBinding)
    assert not binding.serving_allocates
    assert binding.cuda_graph_safe
    assert not binding.recurrent_mtp_state
    assert calls[0][1]["max_tokens"] == 5
    assert calls[1][1]["expected_m"] == 5
    assert calls[1][1]["out"] is residual_out
    assert run_deepseek_v4_dspark_proposal_entry(binding) is expected
    assert [call[0] for call in calls] == ["plan", "bind", "run-pre"]


def test_dspark_block_binds_normal_split_tp4_graph(monkeypatch) -> None:
    from types import SimpleNamespace

    contract = plan_deepseek_v4_dspark(
        variant="flash", max_batch=2, max_main_rows=8
    )
    block = contract.blocks[2]
    arena = torch.empty((block.expert_handoff.arena.total_bytes,), dtype=torch.uint8)
    hidden = torch.empty((10, 4_096), dtype=torch.bfloat16)
    calls = []

    def bind_attention(attention_contract, **kwargs):
        calls.append(("bind-attention", attention_contract, kwargs))
        return SimpleNamespace()

    def bind_sparse(sparse_contract, **kwargs):
        calls.append(("bind-sparse", sparse_contract, kwargs))
        return SimpleNamespace(
            dispatch_hidden=hidden,
            arena_binding=kwargs["arena_binding"],
        )

    monkeypatch.setattr(
        dspark_capture, "bind_deepseek_v4_sliding_attention_layer", bind_attention
    )
    monkeypatch.setattr(dspark_capture, "bind_deepseek_v4_sparse_block", bind_sparse)
    pre = (object(), object(), object(), object())
    post = (object(), object(), object(), object())
    monkeypatch.setattr(
        dspark_capture,
        "run_deepseek_v4_sparse_block_attention",
        lambda sparse: calls.append(("pre", sparse)) or pre,
    )
    monkeypatch.setattr(
        dspark_capture,
        "run_deepseek_v4_sparse_block_post_dispatch",
        lambda sparse: calls.append(("post", sparse)) or post,
    )

    binding = bind_deepseek_v4_dspark_block(
        contract,
        block_index=2,
        arena=arena,
        attention_kwargs={"hidden_states": hidden, "marker": "attention"},
        sparse_kwargs={"marker": "sparse"},
    )

    assert isinstance(binding, DeepseekV4DsparkBlockBinding)
    assert binding.block.storage_prefix == "mtp.2"
    assert binding.dispatch_hidden is hidden
    assert binding.route_indices.shape == (10, 6)
    assert binding.route_weights.shape == (10, 6)
    assert not binding.serving_allocates
    assert binding.cuda_graph_safe
    assert binding.cuda_graph_segments == 2
    assert binding.dispatch_barrier_between_graphs
    assert binding.expert_tensor_parallel == 4
    assert not binding.expert_parallel
    assert calls[0][1] is block.expert_handoff.attention
    assert calls[0][2]["arena"].data_ptr() == arena.data_ptr()
    assert calls[1][1] is block.expert_handoff
    assert calls[1][2]["attention_runner"] is dspark_capture.run_deepseek_v4_sliding_attention_layer
    assert run_deepseek_v4_dspark_block_pre_dispatch(binding) is pre
    assert run_deepseek_v4_dspark_block_post_dispatch(binding) is post

    with pytest.raises(ValueError, match="five-row proposal group"):
        bind_deepseek_v4_dspark_block(
            contract,
            block_index=0,
            arena=arena,
            attention_kwargs={
                "hidden_states": torch.empty((7, 4_096), dtype=torch.bfloat16)
            },
            sparse_kwargs={},
        )
@pytest.mark.parametrize("variant", ["flash", "pro"])
def test_dspark_contract_is_startup_qualified(variant: str) -> None:
    assert qualify_deepseek_v4_dspark_contract(
        variant=variant, max_batch=16, max_main_rows=2_048
    )


@pytest.mark.parametrize(
    ("variant", "max_batch", "max_main_rows", "match"),
    [
        ("unknown", 1, 1, "variant"),
        ("flash", 0, 1, "max_batch"),
        ("flash", 129, 1, "max_batch"),
        ("flash", 1, 0, "max_main_rows"),
    ],
)
def test_dspark_contract_fails_closed(
    variant: str, max_batch: int, max_main_rows: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        plan_deepseek_v4_dspark(
            variant=variant,
            max_batch=max_batch,
            max_main_rows=max_main_rows,
        )
