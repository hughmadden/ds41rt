import pytest
from ds4rt_reference.deepseek_v4_attention_compressor_capture import (
    DS4_C4_MAIN_PAGE_BYTES,
    DS4_C128_MAIN_PAGE_BYTES,
    DS4_INDEX_PAGE_BYTES,
    plan_deepseek_v4_attention_compressor,
)


def test_c4_plan_joins_main_and_index_projection_without_output_staging() -> None:
    contract = plan_deepseek_v4_attention_compressor(
        variant="flash", compress_ratio=4, max_rows=2_048
    )

    assert contract.geometry.hidden == 4_096
    assert contract.geometry.overlap
    assert contract.geometry.with_indexer
    assert contract.geometry.state_rows == 8
    assert contract.geometry.main_projected_width == 1_024
    assert contract.geometry.index_projected_width == 256
    assert contract.geometry.joint_projection_width == 2_560
    assert contract.scratch.projection_bytes == 10_485_760
    assert contract.scratch.total_bytes == 10_485_760
    assert not contract.serving_allocates
    assert not contract.stages_compressed_output
    assert contract.sequence_unique_decode_only
    assert contract.supports_initial_prefill
    assert contract.supports_ordered_prefill
    assert not contract.supports_mtp_state_transactions
    assert contract.initial_prefill_group_capacity == 512
    assert contract.status == "qualified-decode-and-prefill-continuation-not-active"


def test_c128_plan_keeps_single_window_and_smaller_fixed_arena() -> None:
    contract = plan_deepseek_v4_attention_compressor(
        variant="pro", compress_ratio=128, max_rows=2_048
    )

    assert contract.geometry.hidden == 7_168
    assert not contract.geometry.overlap
    assert not contract.geometry.with_indexer
    assert contract.geometry.state_rows == 128
    assert contract.geometry.main_projected_width == 512
    assert contract.geometry.joint_projection_width == 1_024
    assert contract.scratch.total_bytes == 4_194_304


def test_required_buffers_keep_cache_and_fp32_state_caller_owned() -> None:
    c4 = plan_deepseek_v4_attention_compressor(
        variant="flash", compress_ratio=4, max_rows=16
    )
    required = c4.required_buffer_bytes(
        source_pages=512, state_sequences=4, max_positions=131_072
    )

    assert required["hidden_states"] == 16 * 4_096 * 2
    assert required["positions"] == 16 * 4
    assert required["sequence_ids"] == 16 * 4
    assert required["compressed_slots"] == 16 * 4
    assert required["compressed_main_cache"] == 512 * DS4_C4_MAIN_PAGE_BYTES
    assert required["index_cache"] == 512 * DS4_INDEX_PAGE_BYTES
    assert required["main_kv_state"] == 4 * 8 * 1_024 * 4
    assert required["main_score_state"] == 4 * 8 * 1_024 * 4
    assert required["index_kv_state"] == 4 * 8 * 256 * 4
    assert required["index_score_state"] == 4 * 8 * 256 * 4

    c128 = plan_deepseek_v4_attention_compressor(
        variant="flash", compress_ratio=128, max_rows=16
    )
    required = c128.required_buffer_bytes(
        source_pages=512, state_sequences=4, max_positions=131_072
    )
    assert required["compressed_main_cache"] == 512 * DS4_C128_MAIN_PAGE_BYTES
    assert required["main_kv_state"] == 4 * 128 * 512 * 4
    assert "index_cache" not in required


@pytest.mark.parametrize(
    ("compress_ratio", "expected_page_bytes"),
    [(4, 64 * 432), (128, 2 * 432)],
)
def test_native_nvfp4_compressor_uses_exact_record_width(
    compress_ratio: int, expected_page_bytes: int
) -> None:
    contract = plan_deepseek_v4_attention_compressor(
        variant="flash",
        compress_ratio=compress_ratio,
        max_rows=16,
        cache_format="nvfp4",
    )
    required = contract.required_buffer_bytes(
        source_pages=3, state_sequences=2, max_positions=256
    )

    assert contract.cache_format == "nvfp4"
    assert contract.main_page_bytes == expected_page_bytes
    assert required["compressed_main_cache"] == 3 * expected_page_bytes


def test_initial_prefill_metadata_has_fixed_graph_capacities() -> None:
    c4 = plan_deepseek_v4_attention_compressor(
        variant="flash", compress_ratio=4, max_rows=2_048
    )
    required = c4.required_initial_prefill_metadata_bytes(sequence_capacity=32)

    assert required == {
        "active_groups": 4,
        "group_source_starts": 512 * 4,
        "group_rope_positions": 512 * 4,
        "prefill_compressed_slots": 512 * 4,
        "active_sequences": 4,
        "sequence_offsets": 33 * 4,
        "state_sequence_ids": 32 * 4,
    }

    c128 = plan_deepseek_v4_attention_compressor(
        variant="pro", compress_ratio=128, max_rows=2_048
    )
    assert c128.initial_prefill_group_capacity == 16
    assert (
        c128.required_initial_prefill_metadata_bytes(sequence_capacity=1)[
            "group_source_starts"
        ]
        == 16 * 4
    )


@pytest.mark.parametrize("sequence_capacity", [0, 2_049])
def test_initial_prefill_metadata_fails_closed_on_invalid_sequence_capacity(
    sequence_capacity: int,
) -> None:
    contract = plan_deepseek_v4_attention_compressor(
        variant="flash", compress_ratio=4, max_rows=2_048
    )
    with pytest.raises(ValueError, match="sequence_capacity"):
        contract.required_initial_prefill_metadata_bytes(
            sequence_capacity=sequence_capacity
        )


def test_continuation_metadata_pins_sequence_and_group_capacities() -> None:
    contract = plan_deepseek_v4_attention_compressor(
        variant="pro", compress_ratio=128, max_rows=2_048
    )

    assert contract.required_continuation_metadata_bytes(
        sequence_capacity=32, group_capacity=64
    ) == {
        "active_groups": 4,
        "group_sequence_slots": 64 * 4,
        "group_source_positions": 64 * 4,
        "group_rope_positions": 64 * 4,
        "continuation_compressed_slots": 64 * 4,
        "active_sequences": 4,
        "sequence_offsets": 33 * 4,
        "sequence_start_positions": 32 * 4,
        "state_sequence_ids": 32 * 4,
    }


@pytest.mark.parametrize(
    "sequence_capacity, group_capacity", [(0, 1), (1, -1), (1, 2_049)]
)
def test_continuation_metadata_fails_closed_on_invalid_capacities(
    sequence_capacity: int, group_capacity: int
) -> None:
    contract = plan_deepseek_v4_attention_compressor(
        variant="flash", compress_ratio=4, max_rows=2_048
    )
    with pytest.raises(ValueError, match="capacity"):
        contract.required_continuation_metadata_bytes(
            sequence_capacity=sequence_capacity,
            group_capacity=group_capacity,
        )


@pytest.mark.parametrize("variant", ["flash", "pro"])
@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("cache_format", ["fp8", "nvfp4"])
def test_contract_matches_pinned_sparkinfer_plan(
    variant: str, compress_ratio: int, cache_format: str
) -> None:
    from ds4rt_reference.deepseek_v4_attention_compressor_capture import (
        qualify_deepseek_v4_attention_compressor_contract,
    )

    assert qualify_deepseek_v4_attention_compressor_contract(
        variant=variant,
        compress_ratio=compress_ratio,
        max_rows=2_048,
        cache_format=cache_format,
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"variant": "preview", "compress_ratio": 4, "max_rows": 1}, "flash or pro"),
        ({"variant": "flash", "compress_ratio": 8, "max_rows": 1}, "4 or 128"),
        ({"variant": "pro", "compress_ratio": 128, "max_rows": 0}, "must be in"),
    ],
)
def test_compressor_contract_fails_closed(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        plan_deepseek_v4_attention_compressor(**kwargs)
