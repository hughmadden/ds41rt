"""DeepSeek launch-settings resolution without CUDA or model imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Mapping


MIB = 1 << 20
GIB = 1 << 30
SOURCE_PAGE_TOKENS = 256
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
EXL3_RECIPE = "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
KV_BYTES_PER_SOURCE_PAGE = {
    ("flash", "fp8"): 7_437_888,
    ("pro", "fp8"): 10_565_568,
}
# Flash includes the measured C=4 large-prefill frontier in addition to four
# execution lanes, resident native experts, and dSpark.  Pro's qualified K2
# artifact has 30.24 GB of coordinator-resident weights and 25.96 GB of target
# arena/dSpark state before physical KV. Connected CUDA graphs, the reusable
# 65,536-row cuDNN MLA suffix workspace, and the qualified C4 long-prefill
# frontier raise the measured fixed/working serving footprint to about 71 GiB.
# Reserve 72 GiB to expose the 400k logical context contract
# with 5,504 physical rows to spare, while keeping the separate headroom budget
# real during four concurrent 10k-token requests.
FIXED_RESERVE_GIB = {
    "flash": 45.0,
    "pro": 72.0,
}
GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z"
)
PCI_BUS_ID_RE = re.compile(
    r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]\Z"
)


@dataclass(frozen=True)
class ResolvedServeSettings:
    model_id: str
    model_variant: str
    expert_format: str
    dspark: bool
    dspark_draft_policy: str
    coordinator_gpu: int
    coordinator_gpu_uuid: str | None
    coordinator_gpu_pci_bus_id: str | None
    gpu_total_mib: int
    headroom_gib: float
    fixed_reserve_gib: float
    kv_dtype: str
    kv_bytes_per_source_page: int
    kv_pool_tokens: int
    kv_pool_bytes: int
    max_context_tokens: int
    max_output_tokens: int
    concurrency: int
    spark_reduction_min_rows: int
    qualification: str
    note: str
    environment: dict[str, str]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def find_hf_snapshot(
    model_id: str,
    cache_root: Path | None = None,
    revision: str | None = None,
) -> Path | None:
    root = cache_root or Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = root / ("models--" + model_id.replace("/", "--"))
    if revision:
        candidate = repo_dir / "snapshots" / revision
        return candidate if candidate.is_dir() else None
    refs_main = repo_dir / "refs" / "main"
    if refs_main.is_file():
        candidate = repo_dir / "snapshots" / refs_main.read_text().strip()
        if candidate.is_dir():
            return candidate
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return candidates[0] if candidates else None


def query_gpu_total_mib(gpu: str = "0") -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first = result.stdout.strip().splitlines()[0].strip()
    return int(first)


def _round_pool_tokens(available_bytes: int, bytes_per_page: int) -> int:
    return (available_bytes // bytes_per_page) * SOURCE_PAGE_TOKENS


def resolve_serve_settings(
    *,
    repo_root: Path,
    model_id: str = DEFAULT_MODEL_ID,
    model_variant: str = "flash",
    expert_format: str = "native",
    dspark: bool = True,
    dspark_draft_policy: str = "adaptive",
    coordinator_gpu: int = 0,
    coordinator_gpu_uuid: str | None = None,
    coordinator_gpu_pci_bus_id: str | None = None,
    headroom_gib: float = 8.0,
    gpu_total_mib: int | None = None,
    max_context_tokens: int | None = None,
    max_output_tokens: int | None = None,
    kv_pool_tokens: int | None = None,
    concurrency: int = 4,
    spark_reduction_min_rows: int = 16,
    inherited_environment: Mapping[str, str] | None = None,
) -> ResolvedServeSettings:
    del repo_root
    del inherited_environment
    if not model_id.strip() or "/" not in model_id:
        raise ValueError("model_id must be a non-empty Hugging Face repository ID")
    if model_variant not in {"flash", "pro"}:
        raise ValueError("model_variant must be flash or pro")
    if expert_format not in {"native", "exl3"}:
        raise ValueError("expert_format must be native or exl3")
    if dspark_draft_policy not in {"full", "adaptive"}:
        raise ValueError("dspark_draft_policy must be full or adaptive")
    if model_variant == "pro" and expert_format != "exl3":
        raise ValueError("DeepSeek V4 Pro requires calibrated EXL3 expert weights")
    if coordinator_gpu != 0:
        raise ValueError("coordinator_gpu must be 0; GPU 1 is outside DS41RT")
    if (
        coordinator_gpu_uuid is not None
        and GPU_UUID_RE.fullmatch(coordinator_gpu_uuid) is None
    ):
        raise ValueError("coordinator_gpu_uuid must be a physical NVIDIA GPU UUID")
    if (
        coordinator_gpu_pci_bus_id is not None
        and PCI_BUS_ID_RE.fullmatch(coordinator_gpu_pci_bus_id) is None
    ):
        raise ValueError("coordinator_gpu_pci_bus_id must be a full PCI bus ID")
    if (coordinator_gpu_uuid is None) != (coordinator_gpu_pci_bus_id is None):
        raise ValueError("coordinator GPU UUID and PCI bus ID must be set together")
    if headroom_gib < 0:
        raise ValueError("headroom_gib must be non-negative")
    if concurrency < 1 or concurrency > 4:
        raise ValueError("concurrency must be in 1..4")
    if spark_reduction_min_rows < 1:
        raise ValueError("spark_reduction_min_rows must be positive")
    if kv_pool_tokens is not None and (
        kv_pool_tokens <= 0 or kv_pool_tokens % SOURCE_PAGE_TOKENS
    ):
        raise ValueError(
            "kv_pool_tokens must be a positive multiple of "
            f"{SOURCE_PAGE_TOKENS}"
        )

    physical_gpu = coordinator_gpu_uuid or str(coordinator_gpu)
    total_mib = (
        gpu_total_mib
        if gpu_total_mib is not None
        else query_gpu_total_mib(physical_gpu)
    )
    if total_mib <= 0:
        raise ValueError("gpu_total_mib must be positive")
    fixed_reserve_gib = FIXED_RESERVE_GIB[model_variant]
    reserved_bytes = int((fixed_reserve_gib + headroom_gib) * GIB)
    total_bytes = total_mib * MIB
    if reserved_bytes >= total_bytes:
        raise ValueError(
            f"serving reserves {reserved_bytes / GIB:.2f} GiB on a "
            f"{total_bytes / GIB:.2f} GiB GPU"
        )
    bytes_per_page = KV_BYTES_PER_SOURCE_PAGE[(model_variant, "fp8")]
    calculated_pool = _round_pool_tokens(
        total_bytes - reserved_bytes, bytes_per_page
    )
    pool_tokens = calculated_pool if kv_pool_tokens is None else kv_pool_tokens
    if pool_tokens <= 0 or pool_tokens % SOURCE_PAGE_TOKENS:
        raise ValueError(
            "kv_pool_tokens must be a positive multiple of "
            f"{SOURCE_PAGE_TOKENS}"
        )
    if pool_tokens > calculated_pool:
        raise ValueError(
            f"kv_pool_tokens={pool_tokens} exceeds the headroom-safe "
            f"calculated capacity {calculated_pool}"
        )
    context_tokens = (
        min(pool_tokens, 400_000)
        if max_context_tokens is None
        else max_context_tokens
    )
    output_tokens = (
        100_000 if max_output_tokens is None else max_output_tokens
    )
    if context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")
    if context_tokens > 400_000:
        raise ValueError(
            "max_context_tokens exceeds the serving cap "
            "400000"
        )
    if context_tokens > pool_tokens:
        raise ValueError("max_context_tokens cannot exceed kv_pool_tokens")
    if output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if output_tokens > 100_000:
        raise ValueError(
            "max_output_tokens exceeds the serving cap "
            "100000"
        )
    if output_tokens >= context_tokens:
        raise ValueError("max_output_tokens must be smaller than max_context_tokens")

    environment = {
        "CUDA_VISIBLE_DEVICES": physical_gpu,
        "NVIDIA_VISIBLE_DEVICES": physical_gpu,
        "DS41RT_COORDINATOR_GPU": "0",
        "DS41RT_MODEL_ID": model_id,
        "DS41RT_MODEL_VARIANT": model_variant,
        "DS41RT_EXPERT_FORMAT": expert_format,
        "DS41RT_DSPARK_ENABLED": "1" if dspark else "0",
        "DS41RT_REAL_FULL_DSPARK": "1" if dspark else "0",
        "DS41RT_REAL_FULL_SERVE_TRANSPORT": "verbs-host",
        "DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES": str(concurrency),
        "DS41RT_REAL_FULL_MAX_EXECUTION_LANES": str(concurrency),
        "DS41RT_REAL_FULL_SERVE_KV_CACHE_DTYPE": "fp8",
        "DS41RT_REAL_FULL_SERVE_MAX_CONTEXT_TOKENS": str(context_tokens),
        "DS41RT_REAL_FULL_SERVE_MAX_OUTPUT_TOKENS": str(output_tokens),
        "DS41RT_REAL_FULL_KV_POOL_TOKENS": str(pool_tokens),
        "DS41RT_EXPERT_INTERMEDIATE_SHARDS": "4",
        # Preserve GLMRT's qualified TP4 topology: small waves return direct
        # rank partials; wide waves are row-sharded and reduced over Spark RDMA.
        "DS41RT_EXPERT_INTERMEDIATE_REDUCTION": "spark-rdma",
        "DS41RT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS": str(
            spark_reduction_min_rows
        ),
        "DS41RT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS": "8",
        "DS41RT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE": "bf16",
        "DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION": "1",
        "DS41RT_REAL_FULL_MOE_RESPONSE_DTYPE": "bf16",
        "DS41RT_REAL_FULL_MOE_OWNER_RESPONSE_DTYPE": "bf16",
    }
    if coordinator_gpu_uuid is not None:
        assert coordinator_gpu_pci_bus_id is not None
        environment["DS41RT_COORDINATOR_GPU_UUID"] = coordinator_gpu_uuid
        environment["DS41RT_COORDINATOR_GPU_PCI_BUS_ID"] = (
            coordinator_gpu_pci_bus_id
        )
    if expert_format == "exl3":
        environment["DS41RT_EXL3_RECIPE"] = EXL3_RECIPE
    if dspark and dspark_draft_policy == "full":
        environment["DS41RT_REAL_FULL_DSPARK_FIXED_DRAFTS"] = "5"
    environment["DS41RT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE"] = (
        "fp8"
    )
    warnings = (
        "DeepSeek V4 serving remains a porting candidate until the native/EXL3 "
        "gates in benchmarking.md pass.",
    )
    return ResolvedServeSettings(
        model_id=model_id,
        model_variant=model_variant,
        expert_format=expert_format,
        dspark=dspark,
        dspark_draft_policy=dspark_draft_policy,
        coordinator_gpu=coordinator_gpu,
        coordinator_gpu_uuid=coordinator_gpu_uuid,
        coordinator_gpu_pci_bus_id=coordinator_gpu_pci_bus_id,
        gpu_total_mib=total_mib,
        headroom_gib=headroom_gib,
        fixed_reserve_gib=fixed_reserve_gib,
        kv_dtype="fp8",
        kv_bytes_per_source_page=bytes_per_page,
        kv_pool_tokens=pool_tokens,
        kv_pool_bytes=(pool_tokens // SOURCE_PAGE_TOKENS) * bytes_per_page,
        max_context_tokens=context_tokens,
        max_output_tokens=output_tokens,
        concurrency=concurrency,
        spark_reduction_min_rows=spark_reduction_min_rows,
        qualification="porting",
        note="FP8 target KV.",
        environment=environment,
        blockers=(),
        warnings=warnings,
    )
