import pytest
from ds4rt_reference.deepseek_v4_attention_capture import (
    DS4_C4_INDEX_TOPK,
    DS4_C4_PAGE_BYTES,
    DS4_C128_PAGE_BYTES,
    DS4_MAIN_PAGE_BYTES,
    DS4_PRO_C4_INDEX_TOPK,
    DS4_SM120_PREFILL_SELECTION_TILE,
    plan_deepseek_v4_compressed_mla,
)


def test_sliding_contract_uses_native_256_token_main_pages() -> None:
    plan = plan_deepseek_v4_compressed_mla(
        mode="decode",
        rows=1,
        source_pages=512,
        compression=0,
    )

    assert plan.heads == 64
    assert plan.swa_width == 128
    assert plan.total_width == 128
    assert plan.max_chunks_per_row == 11
    assert plan.scratch_bytes == 728_064
    assert not plan.has_indexed_cache
    required = plan.required_buffer_bytes(use_sink=True)
    assert required["swa_k_cache"] == 512 * DS4_MAIN_PAGE_BYTES
    assert required["q"] == 64 * 512 * 2
    assert required["attn_sink"] == 64 * 4
    assert "indexed_k_cache" not in required


@pytest.mark.parametrize(
    ("compression", "indexed_width", "indexed_page_bytes"),
    [(0, 0, 0), (4, DS4_C4_INDEX_TOPK, 64 * 432), (128, 2, 2 * 432)],
)
def test_native_nvfp4_contract_sizes_main_and_indexed_pages_exactly(
    compression: int, indexed_width: int, indexed_page_bytes: int
) -> None:
    plan = plan_deepseek_v4_compressed_mla(
        mode="decode",
        rows=1,
        source_pages=1,
        compression=compression,
        indexed_width=indexed_width,
        cache_format="nvfp4",
    )

    assert plan.cache_format == "nvfp4"
    assert plan.main_page_bytes == 256 * 432
    assert plan.indexed_page_bytes == indexed_page_bytes


def test_integrated_dspark_decode_extends_sliding_selection_by_five_slots() -> None:
    plan = plan_deepseek_v4_compressed_mla(
        mode="decode",
        rows=80,
        heads=64,
        source_pages=16,
        swa_width=133,
        compression=0,
    )

    assert plan.swa_width == plan.total_width == 133
    assert plan.max_chunks_per_row == 12
    assert plan.scratch_bytes == 63_183_872
    assert plan.required_buffer_bytes(use_sink=False)["swa_indices"] == 80 * 133 * 4


@pytest.mark.parametrize(
    ("mode", "compression", "indexed_width"),
    [("extend", 0, 0), ("decode", 4, DS4_C4_INDEX_TOPK)],
)
def test_integrated_dspark_width_is_decode_only_c0(
    mode: str, compression: int, indexed_width: int
) -> None:
    with pytest.raises(ValueError, match="decode-only integrated dSpark C0"):
        plan_deepseek_v4_compressed_mla(
            mode=mode,
            rows=16,
            source_pages=16,
            swa_width=133,
            compression=compression,
            indexed_width=indexed_width,
        )


def test_c4_contract_keeps_fixed_learned_top512_and_planar_cache() -> None:
    plan = plan_deepseek_v4_compressed_mla(
        mode="decode",
        rows=16,
        source_pages=512,
        compression=4,
        indexed_width=DS4_C4_INDEX_TOPK,
    )

    assert plan.indexed_page_tokens == 64
    assert plan.indexed_page_bytes == DS4_C4_PAGE_BYTES
    assert plan.total_width == 640
    assert plan.max_chunks_per_row == 54
    required = plan.required_buffer_bytes(use_sink=False)
    assert required["indexed_k_cache"] == 512 * DS4_C4_PAGE_BYTES
    assert required["indexed_indices"] == 16 * 512 * 4
    assert "attn_sink" not in required


def test_c128_contract_bounds_all_completed_blocks_by_sequence_pages() -> None:
    plan = plan_deepseek_v4_compressed_mla(
        mode="extend",
        rows=2_048,
        source_pages=512,
        compression=128,
        indexed_width=1_024,
    )

    assert plan.indexed_page_tokens == 2
    assert plan.indexed_page_bytes == DS4_C128_PAGE_BYTES
    assert plan.total_width == 1_152
    assert plan.max_chunks_per_row == 2
    assert plan.required_buffer_bytes(use_sink=True)["indexed_k_cache"] == (
        512 * DS4_C128_PAGE_BYTES
    )


def test_cache_pages_are_independent_from_per_sequence_graph_width() -> None:
    plan = plan_deepseek_v4_compressed_mla(
        mode="extend",
        rows=2_048,
        source_pages=512,
        max_page_table_width=64,
        compression=128,
        indexed_width=128,
    )

    assert plan.source_pages == 512
    assert plan.max_page_table_width == 64
    assert plan.indexed_width == 128
    assert plan.required_buffer_bytes(use_sink=False)["indexed_k_cache"] == (
        512 * DS4_C128_PAGE_BYTES
    )
    with pytest.raises(ValueError, match="no larger than 128"):
        plan_deepseek_v4_compressed_mla(
            mode="extend",
            rows=2_048,
            source_pages=512,
            max_page_table_width=64,
            compression=128,
            indexed_width=129,
        )


def test_c128_small_prefill_pads_selection_to_one_sm120_tile() -> None:
    plan = plan_deepseek_v4_compressed_mla(
        mode="extend",
        rows=128,
        source_pages=1,
        compression=128,
        indexed_width=DS4_SM120_PREFILL_SELECTION_TILE,
    )

    assert plan.indexed_width == 64
    assert plan.total_width == 192
    with pytest.raises(ValueError, match="no larger than 2"):
        plan_deepseek_v4_compressed_mla(
            mode="decode",
            rows=1,
            source_pages=1,
            compression=128,
            indexed_width=DS4_SM120_PREFILL_SELECTION_TILE,
        )


def test_pro_c4_contract_is_the_same_path_with_128_heads_and_top1024() -> None:
    plan = plan_deepseek_v4_compressed_mla(
        mode="decode",
        rows=16,
        heads=128,
        source_pages=512,
        compression=4,
        indexed_width=DS4_PRO_C4_INDEX_TOPK,
    )

    assert plan.heads == 128
    assert plan.indexed_width == 1_024
    assert plan.total_width == 1_152
    assert plan.max_chunks_per_row == 96
    required = plan.required_buffer_bytes(use_sink=True)
    assert required["q"] == 16 * 128 * 512 * 2
    assert required["indexed_indices"] == 16 * 1_024 * 4
    assert required["attn_sink"] == 128 * 4


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"compression": 0, "indexed_width": 1}, "cannot carry indexed"),
        ({"compression": 4, "indexed_width": 511}, "top-512"),
        ({"compression": 128, "indexed_width": 1_025}, "no larger than 1024"),
        ({"compression": 8, "indexed_width": 0}, "0, 4, or 128"),
        (
            {"compression": 4, "indexed_width": 512, "heads": 128},
            "top-1024",
        ),
        ({"compression": 0, "indexed_width": 0, "heads": 32}, r"\(64, 128\)"),
        ({"compression": 0, "indexed_width": 0, "swa_width": 64}, "must be 128"),
    ],
)
def test_contract_fails_closed_on_non_native_geometry(
    kwargs: dict[str, int], match: str
) -> None:
    base = {
        "mode": "decode",
        "rows": 1,
        "source_pages": 512,
    }
    with pytest.raises(ValueError, match=match):
        plan_deepseek_v4_compressed_mla(**base, **kwargs)
