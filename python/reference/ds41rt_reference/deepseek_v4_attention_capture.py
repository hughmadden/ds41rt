from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ds41rt_reference.deepseek_v4_kv_format import (
    DS4_C4_PAGE_TOKENS,
    DS4_C128_PAGE_TOKENS,
    DS4_SOURCE_PAGE_TOKENS,
    FP8_UE8M0,
    deepseek_v4_kv_format,
)

DS4_FLASH_HEADS = 64
DS4_PRO_HEADS = 128
DS4_SUPPORTED_HEADS = (DS4_FLASH_HEADS, DS4_PRO_HEADS)
DS4_HEAD_DIM = 512
DS4_SWA_TOKENS = 128
DS4_MAIN_PAGE_BYTES = FP8_UE8M0.main_page_bytes
DS4_C4_PAGE_BYTES = FP8_UE8M0.c4_page_bytes
DS4_C128_PAGE_BYTES = FP8_UE8M0.c128_page_bytes
DS4_C4_INDEX_TOPK = 512
DS4_PRO_C4_INDEX_TOPK = 1_024
DS4_SM120_PREFILL_SELECTION_TILE = 64
DS4_MAX_CAPTURE_ROWS = 8_192
DS4_MAX_SPLIT_CHUNKS = 256
_SCRATCH_ALIGNMENT = 1_024
_PREPARED: set[tuple[Any, ...]] = set()
_PLANS: dict[tuple[Any, ...], Any] = {}


@dataclass(frozen=True)
class DeepseekV4CompressedMlaContract:
    mode: str
    rows: int
    heads: int
    source_pages: int
    max_page_table_width: int
    swa_width: int
    compression: int
    indexed_width: int
    indexed_page_tokens: int
    indexed_page_bytes: int
    total_width: int
    max_chunks_per_row: int
    scratch_bytes: int
    cache_format: str
    main_page_bytes: int

    @property
    def has_indexed_cache(self) -> bool:
        return self.compression != 0

    def required_buffer_bytes(self, *, use_sink: bool) -> dict[str, int]:
        required = {
            "q": self.rows * self.heads * DS4_HEAD_DIM * 2,
            "swa_k_cache": self.source_pages * self.main_page_bytes,
            "swa_indices": self.rows * self.swa_width * 4,
            "swa_lengths": self.rows * 4,
            "scratch": self.scratch_bytes,
            "output": self.rows * self.heads * DS4_HEAD_DIM * 2,
        }
        if self.has_indexed_cache:
            required.update(
                {
                    "indexed_k_cache": self.source_pages
                    * self.indexed_page_bytes,
                    "indexed_indices": self.rows * self.indexed_width * 4,
                    "indexed_lengths": self.rows * 4,
                }
            )
        if use_sink:
            required["attn_sink"] = self.heads * 4
        return required


def plan_deepseek_v4_compressed_mla(
    *,
    mode: str,
    rows: int,
    heads: int = DS4_FLASH_HEADS,
    source_pages: int,
    max_page_table_width: int | None = None,
    swa_width: int = DS4_SWA_TOKENS,
    compression: int,
    indexed_width: int = 0,
    cache_format: str = "fp8",
) -> DeepseekV4CompressedMlaContract:
    """Build the fixed native DSV4 sparse-MLA graph contract.

    ``compression`` is zero for sliding-only layers, four for learned-indexer
    layers, or 128 for completed-block layers. Indices are raw physical slot
    IDs; no indexed page table is accepted by SparkInfer's SM120 dual-cache ABI.
    """

    mode = str(mode).strip().lower()
    if mode not in {"decode", "extend", "verify", "draft_extend"}:
        raise ValueError(
            "DeepSeek V4 compressed MLA mode must be decode, extend, verify, "
            f"or draft_extend, got {mode!r}"
        )
    rows = int(rows)
    heads = int(heads)
    source_pages = int(source_pages)
    max_page_table_width = (
        source_pages
        if max_page_table_width is None
        else int(max_page_table_width)
    )
    swa_width = int(swa_width)
    compression = int(compression)
    indexed_width = int(indexed_width)
    format_plan = deepseek_v4_kv_format(cache_format)
    if not 1 <= rows <= DS4_MAX_CAPTURE_ROWS:
        raise ValueError(
            f"DeepSeek V4 compressed MLA rows must be in [1, {DS4_MAX_CAPTURE_ROWS}], got {rows}"
        )
    if mode == "decode" and rows > 256:
        raise ValueError(
            f"DeepSeek V4 decode graph rows must be in [1, 256], got {rows}"
        )
    if heads not in DS4_SUPPORTED_HEADS:
        raise ValueError(
            "DeepSeek V4 coordinator attention requires Flash/Pro heads "
            f"{DS4_SUPPORTED_HEADS}, got {heads}"
        )
    if source_pages <= 0:
        raise ValueError(
            f"DeepSeek V4 compressed MLA source_pages must be positive, got {source_pages}"
        )
    if not 1 <= max_page_table_width <= source_pages:
        raise ValueError(
            "DeepSeek V4 compressed MLA max_page_table_width must be positive "
            f"and no larger than source_pages={source_pages}, got "
            f"{max_page_table_width}"
        )
    dspark_decode = (
        mode == "decode"
        and compression == 0
        and swa_width == DS4_SWA_TOKENS + 5
    )
    if swa_width != DS4_SWA_TOKENS and not dspark_decode:
        raise ValueError(
            "DeepSeek V4 sliding selection width must be 128, except the "
            f"decode-only integrated dSpark C0 path may use 133; got {swa_width}"
        )

    if compression == 0:
        if indexed_width != 0:
            raise ValueError(
                "DeepSeek V4 sliding-only attention cannot carry indexed slots"
            )
        indexed_page_tokens = 0
        indexed_page_bytes = 0
    elif compression == 4:
        required_top_k = (
            DS4_C4_INDEX_TOPK
            if heads == DS4_FLASH_HEADS
            else DS4_PRO_C4_INDEX_TOPK
        )
        if indexed_width != required_top_k:
            raise ValueError(
                f"DeepSeek V4 C=4 attention with {heads} heads requires learned "
                f"top-{required_top_k}, "
                f"got indexed_width={indexed_width}"
            )
        indexed_page_tokens = DS4_C4_PAGE_TOKENS
        indexed_page_bytes = format_plan.c4_page_bytes
    elif compression == 128:
        max_completed_blocks = max_page_table_width * DS4_C128_PAGE_TOKENS
        indexed_capacity = max_completed_blocks
        if mode in {"extend", "verify", "draft_extend"}:
            indexed_capacity = max(
                indexed_capacity,
                DS4_SM120_PREFILL_SELECTION_TILE,
            )
        if not 1 <= indexed_width <= indexed_capacity:
            raise ValueError(
                "DeepSeek V4 C=128 indexed width must cover a positive number "
                "of completed blocks (with one padded SM120 prefill tile when "
                f"needed) no larger than {indexed_capacity}, got {indexed_width}"
            )
        indexed_page_tokens = DS4_C128_PAGE_TOKENS
        indexed_page_bytes = format_plan.c128_page_bytes
    else:
        raise ValueError(
            f"DeepSeek V4 compressed MLA supports compression 0, 4, or 128, got {compression}"
        )

    total_width = swa_width + indexed_width
    max_chunks_per_row = _split_chunks_for_contract(rows=rows, width=total_width)
    scratch_bytes = _compressed_mla_scratch_bytes(
        rows=rows,
        heads=heads,
        chunks=max_chunks_per_row,
    )
    return DeepseekV4CompressedMlaContract(
        mode=mode,
        rows=rows,
        heads=heads,
        source_pages=source_pages,
        max_page_table_width=max_page_table_width,
        swa_width=swa_width,
        compression=compression,
        indexed_width=indexed_width,
        indexed_page_tokens=indexed_page_tokens,
        indexed_page_bytes=indexed_page_bytes,
        total_width=total_width,
        max_chunks_per_row=max_chunks_per_row,
        scratch_bytes=scratch_bytes,
        cache_format=format_plan.name,
        main_page_bytes=format_plan.main_page_bytes,
    )


def deepseek_v4_compressed_mla_scratch_nbytes(**kwargs: Any) -> int:
    return plan_deepseek_v4_compressed_mla(**kwargs).scratch_bytes


def qualify_deepseek_v4_compressed_mla_contract(**kwargs: Any) -> bool:
    """Fail closed if the pinned SparkInfer public surface drifts from DS41RT."""

    contract = plan_deepseek_v4_compressed_mla(**kwargs)
    import torch
    from b12x.attention._shared.mla.compressed_reference import (
        COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
        COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
        COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
        compressed_sparse_mla_page_nbytes,
    )
    from b12x.attention.compressed_sparse_mla import (
        Caps,
        plan,
        split_chunks_for_contract,
    )

    if (
        COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE != DS4_SOURCE_PAGE_TOKENS
        or COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE != DS4_C4_PAGE_TOKENS
        or COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE != DS4_C128_PAGE_TOKENS
    ):
        raise RuntimeError("pinned SparkInfer DSV4 compressed-page ABI drifted")
    if contract.cache_format == "fp8" and (
        compressed_sparse_mla_page_nbytes(DS4_SOURCE_PAGE_TOKENS)
        != contract.main_page_bytes
        or compressed_sparse_mla_page_nbytes(DS4_C4_PAGE_TOKENS)
        != DS4_C4_PAGE_BYTES
        or compressed_sparse_mla_page_nbytes(DS4_C128_PAGE_TOKENS)
        != DS4_C128_PAGE_BYTES
    ):
        raise RuntimeError("pinned SparkInfer DSV4 FP8 page ABI drifted")
    if contract.cache_format == "nvfp4" and (
        contract.main_page_bytes != DS4_SOURCE_PAGE_TOKENS * 432
        or contract.indexed_page_bytes
        != (contract.indexed_page_tokens * 432 if contract.has_indexed_cache else 0)
    ):
        raise RuntimeError("pinned SparkInfer DSV4 NVFP4 page ABI drifted")
    sparkinfer_chunks = split_chunks_for_contract(
        rows=contract.rows,
        width=contract.total_width,
    )
    if int(sparkinfer_chunks) != contract.max_chunks_per_row:
        raise RuntimeError(
            "pinned SparkInfer DSV4 split policy drifted: "
            f"DS41RT={contract.max_chunks_per_row} SparkInfer={sparkinfer_chunks}"
        )
    scratch_plan = plan(
        Caps(
            device="cpu",
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=contract.heads,
            head_dim=DS4_HEAD_DIM,
            v_head_dim=DS4_HEAD_DIM,
            max_width=contract.total_width,
            max_page_table_width=contract.max_page_table_width,
            max_q_rows=contract.rows,
            max_batch=contract.rows,
            max_kv_rows=0,
            max_chunks_per_row=contract.max_chunks_per_row,
            page_size=DS4_SOURCE_PAGE_TOKENS,
        )
    )
    (scratch_spec,) = scratch_plan.scratch_specs()
    if int(scratch_spec.nbytes) != contract.scratch_bytes:
        raise RuntimeError(
            "pinned SparkInfer compressed-MLA scratch ABI drifted: "
            f"DS41RT={contract.scratch_bytes} SparkInfer={scratch_spec.nbytes}"
        )
    return True


def prepare_deepseek_v4_compressed_mla(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_compressed_mla(ctx, prepare_only=True, **kwargs)


def capture_deepseek_v4_compressed_mla(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    _run_deepseek_v4_compressed_mla(ctx, prepare_only=False, **kwargs)


def _run_deepseek_v4_compressed_mla(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from b12x_mla_capture import _bf16_tensor, _f32_tensor, _i32_tensor, _u8_tensor
    from b12x.attention.compressed_sparse_mla import Caps, plan, run

    use_sink = bool(kwargs.get("use_sink", True))
    contract = plan_deepseek_v4_compressed_mla(
        mode=str(kwargs["mode"]),
        rows=int(kwargs["rows"]),
        heads=int(kwargs.get("heads", DS4_FLASH_HEADS)),
        source_pages=int(kwargs["source_pages"]),
        max_page_table_width=int(
            kwargs.get("max_page_table_width", kwargs["source_pages"])
        ),
        swa_width=int(kwargs.get("swa_width", DS4_SWA_TOKENS)),
        compression=int(kwargs["compression"]),
        indexed_width=int(kwargs.get("indexed_width", 0)),
        cache_format=str(kwargs.get("cache_format", "fp8")),
    )
    buffers = ctx["buffers"]
    required = contract.required_buffer_bytes(use_sink=use_sink)
    device_id = _validate_buffers(buffers, required)
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    plan_key = (
        device_id,
        contract.rows,
        contract.heads,
        contract.source_pages,
        contract.max_page_table_width,
        contract.total_width,
        contract.max_chunks_per_row,
        contract.cache_format,
    )

    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        scratch_plan = _PLANS.get(plan_key)
        if scratch_plan is None:
            scratch_plan = plan(
                Caps(
                    device=torch.device("cuda", device_id),
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    num_q_heads=contract.heads,
                    head_dim=DS4_HEAD_DIM,
                    v_head_dim=DS4_HEAD_DIM,
                    max_width=contract.total_width,
                    max_page_table_width=contract.max_page_table_width,
                    max_q_rows=contract.rows,
                    max_batch=contract.rows,
                    max_kv_rows=0,
                    max_chunks_per_row=contract.max_chunks_per_row,
                    page_size=DS4_SOURCE_PAGE_TOKENS,
                )
            )
            (scratch_spec,) = scratch_plan.scratch_specs()
            if int(scratch_spec.nbytes) != contract.scratch_bytes:
                raise RuntimeError(
                    "SparkInfer compressed-MLA scratch plan changed after startup "
                    f"qualification: expected {contract.scratch_bytes}, got {scratch_spec.nbytes}"
                )
            _PLANS[plan_key] = scratch_plan

        q = _bf16_tensor(buffers["q"], (contract.rows, contract.heads, DS4_HEAD_DIM))
        swa_cache = _u8_tensor(
            buffers["swa_k_cache"],
            (contract.source_pages, contract.main_page_bytes),
        )
        swa_indices = _i32_tensor(
            buffers["swa_indices"], (contract.rows, contract.swa_width)
        )
        swa_lengths = _i32_tensor(buffers["swa_lengths"], (contract.rows,))
        scratch = _u8_tensor(buffers["scratch"], (int(buffers["scratch"]["bytes"]),))
        output = _bf16_tensor(
            buffers["output"], (contract.rows, contract.heads, DS4_HEAD_DIM)
        )
        indexed_cache = None
        indexed_indices = None
        indexed_lengths = None
        if contract.has_indexed_cache:
            indexed_cache = _u8_tensor(
                buffers["indexed_k_cache"],
                (contract.source_pages, contract.indexed_page_bytes),
            )
            indexed_indices = _i32_tensor(
                buffers["indexed_indices"],
                (contract.rows, contract.indexed_width),
            )
            indexed_lengths = _i32_tensor(
                buffers["indexed_lengths"], (contract.rows,)
            )
        attn_sink = (
            _f32_tensor(buffers["attn_sink"], (contract.heads,))
            if use_sink
            else None
        )
        binding = scratch_plan.bind(
            scratch=scratch,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
        )
        binding.scratch.mode = contract.mode
        binding.scratch.fixed_capacity = True
        binding.scratch.use_cuda_graph = True

        prepared_key = plan_key + (
            contract.mode,
            contract.compression,
            contract.indexed_width,
            use_sink,
        )
        if not prepare_only and prepared_key not in _PREPARED:
            raise RuntimeError(
                "DeepSeek V4 compressed MLA capture was not prepared before graph capture"
            )
        run(
            binding=binding,
            swa_k_cache=swa_cache,
            sm_scale=float(kwargs["scale"]),
            swa_page_size=DS4_SOURCE_PAGE_TOKENS,
            indexed_k_cache=indexed_cache,
            indexed_page_size=(
                contract.indexed_page_tokens if contract.has_indexed_cache else None
            ),
            attn_sink=attn_sink,
            expected_num_q_heads=contract.heads,
            backend="sm120",
            out=output,
        )
        if prepare_only:
            _PREPARED.add(prepared_key)


def _validate_buffers(
    buffers: dict[str, dict[str, Any]], required: dict[str, int]
) -> int:
    missing = sorted(set(required) - set(buffers))
    if missing:
        raise ValueError(
            f"DeepSeek V4 compressed MLA is missing buffers: {', '.join(missing)}"
        )
    device_id = int(buffers["q"]["device_id"])
    for name, required_bytes in required.items():
        buffer = buffers[name]
        if int(buffer["device_id"]) != device_id:
            raise ValueError(
                "DeepSeek V4 compressed MLA buffers must share one device; "
                f"q is cuda:{device_id}, {name} is cuda:{buffer['device_id']}"
            )
        available = int(buffer["bytes"])
        if available < required_bytes:
            raise ValueError(
                f"DeepSeek V4 compressed MLA {name} needs {required_bytes} bytes, got {available}"
            )
    return device_id


def _split_chunks_for_contract(*, rows: int, width: int) -> int:
    if rows <= 256:
        decode_chunks = _ceil_div(width, 12)
        if decode_chunks <= DS4_MAX_SPLIT_CHUNKS:
            return decode_chunks
        wide_chunks = _ceil_div(width, 64)
        if wide_chunks <= DS4_MAX_SPLIT_CHUNKS:
            return wide_chunks
    return min(_ceil_div(width, 1_024), DS4_MAX_SPLIT_CHUNKS)


def _compressed_mla_scratch_bytes(*, rows: int, heads: int, chunks: int) -> int:
    q_chunks = rows * chunks
    cursor = 0
    cursor = _align(cursor)
    cursor += q_chunks * heads * DS4_HEAD_DIM * 2
    cursor = _align(cursor)
    cursor += q_chunks * heads * 4
    cursor = _align(cursor)
    cursor += rows * heads * 4
    cursor = _align(cursor)
    cursor += 4
    cursor = _align(cursor)
    cursor += 4
    cursor = _align(cursor)
    cursor += 4
    return _align(cursor)


def _align(value: int) -> int:
    return _ceil_div(value, _SCRATCH_ALIGNMENT) * _SCRATCH_ALIGNMENT


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


__all__ = [
    "DeepseekV4CompressedMlaContract",
    "capture_deepseek_v4_compressed_mla",
    "deepseek_v4_compressed_mla_scratch_nbytes",
    "plan_deepseek_v4_compressed_mla",
    "prepare_deepseek_v4_compressed_mla",
    "qualify_deepseek_v4_compressed_mla_contract",
]
