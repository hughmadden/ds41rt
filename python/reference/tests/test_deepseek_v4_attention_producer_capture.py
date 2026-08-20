import pytest

from ds4rt_reference.deepseek_v4_attention_producer_capture import (
    DS4_INDEX_PAGE_BYTES,
    DS4_MAIN_PAGE_BYTES,
    plan_deepseek_v4_attention_indexer,
    plan_deepseek_v4_attention_producer,
)


def test_flash_producer_plan_joins_first_projection_and_owns_no_kv_staging() -> None:
    contract = plan_deepseek_v4_attention_producer(variant="flash", max_rows=2_048)

    assert contract.geometry.hidden == 4_096
    assert contract.geometry.q_lora_rank == 1_024
    assert contract.geometry.heads == 64
    assert contract.geometry.query_width == 32_768
    assert contract.geometry.joint_qkv_rank_width == 1_536
    assert contract.scratch.total_bytes == 21_626_880
    assert contract.stages[0] == "joint-wq-a-wkv-block-fp8-k128"
    assert not contract.serving_allocates
    assert not contract.stages_bf16_kv
    assert not contract.touches_compressor_state
    assert contract.status == "qualified-main-q-kv-not-active"


def test_pro_plan_keeps_native_preview_geometry_and_fixed_arena() -> None:
    contract = plan_deepseek_v4_attention_producer(variant="pro", max_rows=2_048)

    assert contract.geometry.hidden == 7_168
    assert contract.geometry.q_lora_rank == 1_536
    assert contract.geometry.heads == 128
    assert contract.geometry.query_width == 65_536
    assert contract.scratch.total_bytes == 33_619_968
    assert contract.scratch.q_linear_offset % 1_024 == 0
    assert contract.scratch.qkv_output_offset % 1_024 == 0
    assert contract.scratch.q_rank_offset % 1_024 == 0


def test_required_buffers_leave_persistent_cache_and_final_query_caller_owned() -> None:
    contract = plan_deepseek_v4_attention_producer(variant="flash", max_rows=16)
    required = contract.required_buffer_bytes(source_pages=512, max_positions=131_072)

    assert required["hidden_states"] == 16 * 4_096 * 2
    assert required["positions"] == 16 * 4
    assert required["main_slots"] == 16 * 4
    assert required["query"] == 16 * 64 * 512 * 2
    assert required["main_kv_cache"] == 512 * DS4_MAIN_PAGE_BYTES
    assert required["cos_sin_cache"] == 131_072 * 64 * 4
    assert required["scratch"] == contract.scratch.total_bytes


def test_native_nvfp4_producer_uses_exact_432_byte_token_records() -> None:
    contract = plan_deepseek_v4_attention_producer(
        variant="flash", max_rows=16, cache_format="nvfp4"
    )
    required = contract.required_buffer_bytes(source_pages=3, max_positions=256)

    assert contract.cache_format == "nvfp4"
    assert contract.main_page_bytes == 256 * 432
    assert required["main_kv_cache"] == 3 * 256 * 432


def test_flash_indexer_plan_reuses_qrank_and_selects_physical_c4_slots() -> None:
    contract = plan_deepseek_v4_attention_indexer(variant="flash", max_rows=2_048)

    assert contract.geometry.hidden == 4_096
    assert contract.geometry.q_lora_rank == 1_024
    assert contract.geometry.heads == 64
    assert contract.geometry.head_dim == 128
    assert contract.geometry.query_width == 8_192
    assert contract.geometry.top_k == 512
    assert contract.scratch.total_bytes == 36_044_800
    assert contract.consumes_shared_q_rank
    assert contract.output_physical_slots
    assert contract.selection_scratch_planned_by_owner
    assert contract.selector_owner == "sparkinfer.attention.nsa_indexer"
    assert contract.supports_initial_prefill
    assert contract.supports_ordered_continuation
    assert not contract.serving_allocates
    assert contract.status == "qualified-index-query-and-physical-selection-not-active"


def test_pro_indexer_plan_preserves_preview_topk_and_fixed_arena() -> None:
    contract = plan_deepseek_v4_attention_indexer(variant="pro", max_rows=2_048)

    assert contract.geometry.hidden == 7_168
    assert contract.geometry.q_lora_rank == 1_536
    assert contract.geometry.top_k == 1_024
    assert contract.scratch.total_bytes == 37_158_912
    assert contract.scratch.q_output_offset % 1_024 == 0
    assert contract.scratch.weights_output_offset % 1_024 == 0


@pytest.mark.parametrize(
    "logical_tokens, completed_groups",
    [(0, 0), (3, 0), (4, 1), (2_048, 512), (4_096, 1_024)],
)
def test_indexer_exposes_only_completed_c4_groups(
    logical_tokens: int, completed_groups: int
) -> None:
    contract = plan_deepseek_v4_attention_indexer(variant="flash", max_rows=1)

    assert contract.completed_groups(logical_tokens) == completed_groups
    assert contract.selected_length(logical_tokens) == min(512, completed_groups)


def test_indexer_required_buffers_keep_cache_selection_and_arenas_caller_owned() -> None:
    contract = plan_deepseek_v4_attention_indexer(variant="pro", max_rows=16)
    required = contract.required_buffer_bytes(source_pages=512, max_positions=131_072)

    assert required["q_rank"] == 16 * 1_536 * 2
    assert required["hidden_states"] == 16 * 7_168 * 2
    assert required["positions"] == 16 * 4
    assert required["index_query"] == 16 * 64 * 128
    assert required["head_weights"] == 16 * 64 * 4
    assert required["index_k_cache"] == 512 * DS4_INDEX_PAGE_BYTES
    assert required["real_page_table"] == 16 * 512 * 4
    assert required["completed_group_lengths"] == 16 * 4
    assert required["selected_indices"] == 16 * 1_024 * 4
    assert required["selected_lengths"] == 16 * 4
    assert required["producer_scratch"] == contract.scratch.total_bytes


@pytest.mark.parametrize("variant", ["flash", "pro"])
@pytest.mark.parametrize("cache_format", ["fp8", "nvfp4"])
def test_contract_matches_pinned_sparkinfer_plan(
    variant: str, cache_format: str
) -> None:
    from ds4rt_reference.deepseek_v4_attention_producer_capture import (
        qualify_deepseek_v4_attention_producer_contract,
    )

    assert qualify_deepseek_v4_attention_producer_contract(
        variant=variant, max_rows=2_048, cache_format=cache_format
    )


@pytest.mark.parametrize("variant", ["flash", "pro"])
def test_indexer_contract_matches_pinned_sparkinfer_producer_and_selector(
    variant: str,
) -> None:
    from ds4rt_reference.deepseek_v4_attention_producer_capture import (
        qualify_deepseek_v4_attention_indexer_contract,
    )

    assert qualify_deepseek_v4_attention_indexer_contract(
        variant=variant, max_rows=2_048
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"variant": "preview", "max_rows": 1}, "flash or pro"),
        ({"variant": "flash", "max_rows": 0}, "must be in"),
        ({"variant": "pro", "max_rows": 8_193}, "must be in"),
    ],
)
def test_producer_contract_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plan_deepseek_v4_attention_producer(**kwargs)

    with pytest.raises(ValueError, match=match):
        plan_deepseek_v4_attention_indexer(**kwargs)


def test_indexer_rejects_negative_logical_length() -> None:
    contract = plan_deepseek_v4_attention_indexer(variant="flash", max_rows=1)

    with pytest.raises(ValueError, match="non-negative"):
        contract.completed_groups(-1)
