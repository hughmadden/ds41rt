"""Native DeepSeek V4 physical KV page formats shared by capture adapters."""

from __future__ import annotations

from dataclasses import dataclass


DS4_SOURCE_PAGE_TOKENS = 256
DS4_C4_PAGE_TOKENS = 64
DS4_C128_PAGE_TOKENS = 2
DS4_INDEX_PAGE_BYTES = 8_448


@dataclass(frozen=True)
class DeepseekV4KvFormat:
    name: str
    record_bytes: int
    main_page_bytes: int
    c4_page_bytes: int
    c128_page_bytes: int


FP8_UE8M0 = DeepseekV4KvFormat(
    name="fp8",
    record_bytes=584,
    main_page_bytes=149_760,
    c4_page_bytes=37_440,
    c128_page_bytes=1_728,
)
NVFP4 = DeepseekV4KvFormat(
    name="nvfp4",
    record_bytes=432,
    main_page_bytes=110_592,
    c4_page_bytes=27_648,
    c128_page_bytes=864,
)


def deepseek_v4_kv_format(value: str) -> DeepseekV4KvFormat:
    normalized = str(value).strip().lower()
    if normalized in {"fp8", "fp8-ue8m0"}:
        return FP8_UE8M0
    if normalized == "nvfp4":
        return NVFP4
    raise ValueError(
        f"DeepSeek V4 physical cache_format must be fp8 or nvfp4, got {value!r}"
    )


def deepseek_v4_mla_page_bytes(cache_format: str, rows: int) -> int:
    format_plan = deepseek_v4_kv_format(cache_format)
    rows = int(rows)
    if rows == DS4_SOURCE_PAGE_TOKENS:
        return format_plan.main_page_bytes
    if rows == DS4_C4_PAGE_TOKENS:
        return format_plan.c4_page_bytes
    if rows == DS4_C128_PAGE_TOKENS:
        return format_plan.c128_page_bytes
    raise ValueError(f"unsupported DeepSeek V4 MLA page row count {rows}")


__all__ = [
    "DS4_C4_PAGE_TOKENS",
    "DS4_C128_PAGE_TOKENS",
    "DS4_INDEX_PAGE_BYTES",
    "DS4_SOURCE_PAGE_TOKENS",
    "DeepseekV4KvFormat",
    "FP8_UE8M0",
    "NVFP4",
    "deepseek_v4_kv_format",
    "deepseek_v4_mla_page_bytes",
]
