import subprocess
import sys
from pathlib import Path

import pytest

from ds41rt_reference.serve_profiles import (
    DEFAULT_MODEL_ID,
    EXL3_RECIPE,
    find_hf_snapshot,
    resolve_serve_profile,
)


def resolve(tmp_path: Path, **kwargs):
    kwargs.setdefault("gpu_total_mib", 97_887)
    return resolve_serve_profile(repo_root=tmp_path, **kwargs)


def test_flash_native_balanced_defaults_are_gpu0_only(tmp_path):
    profile = resolve(tmp_path)
    assert profile.model_id == DEFAULT_MODEL_ID
    assert profile.model_variant == "flash"
    assert profile.expert_format == "native"
    assert profile.kv_dtype == "fp8"
    assert profile.max_context_tokens == 400_000
    assert profile.max_output_tokens == 100_000
    assert profile.concurrency == 4
    assert profile.spark_reduction_min_rows == 16
    assert profile.gpu_total_mib == 97_887
    assert profile.fixed_reserve_gib == 45.0
    assert profile.kv_bytes_per_source_page == 7_437_888
    assert profile.kv_pool_tokens == 1_573_888
    assert profile.environment["DS41RT_REAL_FULL_KV_POOL_TOKENS"] == "1573888"
    assert profile.environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert profile.environment["NVIDIA_VISIBLE_DEVICES"] == "0"
    assert profile.environment["DS41RT_DSPARK_ENABLED"] == "1"
    assert profile.dspark_draft_policy == "adaptive"
    assert "DS41RT_REAL_FULL_DSPARK_FIXED_DRAFTS" not in profile.environment
    assert profile.environment["DS41RT_EXPERT_INTERMEDIATE_SHARDS"] == "4"
    assert profile.environment["DS41RT_EXPERT_INTERMEDIATE_REDUCTION"] == "spark-rdma"
    assert profile.environment["DS41RT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS"] == (
        "16"
    )
    assert profile.environment["DS41RT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE"] == "fp8"
    assert profile.environment["DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION"] == "1"
    assert "DS41RT_DSPARK_MODEL_ID" not in profile.environment


def test_flash_exl3_selects_versioned_trellis_recipe(tmp_path):
    uuid = "GPU-95f8f212-9131-df99-fd53-7535965197d7"
    profile = resolve(
        tmp_path,
        expert_format="exl3",
        coordinator_gpu_uuid=uuid,
        coordinator_gpu_pci_bus_id="00000000:11:00.0",
    )
    assert profile.environment["DS41RT_EXL3_RECIPE"] == EXL3_RECIPE
    assert profile.coordinator_gpu == 0
    assert profile.coordinator_gpu_uuid == uuid
    assert profile.coordinator_gpu_pci_bus_id == "00000000:11:00.0"
    assert profile.environment["CUDA_VISIBLE_DEVICES"] == uuid
    assert profile.environment["NVIDIA_VISIBLE_DEVICES"] == uuid
    assert profile.environment["DS41RT_COORDINATOR_GPU"] == "0"
    assert profile.environment["DS41RT_COORDINATOR_GPU_UUID"] == uuid
    assert profile.environment["DS41RT_COORDINATOR_GPU_PCI_BUS_ID"] == (
        "00000000:11:00.0"
    )


def test_pro_requires_exl3_but_accepts_future_model_id(tmp_path):
    with pytest.raises(ValueError, match="requires calibrated EXL3"):
        resolve(
            tmp_path,
            model_id="future-org/deepseek-v4-pro",
            model_variant="pro",
            expert_format="native",
        )
    profile = resolve(
        tmp_path,
        model_id="future-org/deepseek-v4-pro",
        model_variant="pro",
        expert_format="exl3",
    )
    assert profile.model_id == "future-org/deepseek-v4-pro"
    assert profile.model_variant == "pro"
    assert profile.fixed_reserve_gib == 72.0
    assert profile.kv_pool_tokens == 405_504
    assert profile.max_context_tokens == 400_000


def test_gpu1_and_concurrency_above_four_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="GPU 1"):
        resolve(tmp_path, coordinator_gpu=1)
    with pytest.raises(ValueError, match="1..4"):
        resolve(tmp_path, concurrency=5)
    with pytest.raises(ValueError, match="must be positive"):
        resolve(tmp_path, spark_reduction_min_rows=0)
    with pytest.raises(ValueError, match="physical NVIDIA GPU UUID"):
        resolve(
            tmp_path,
            coordinator_gpu_uuid="GPU-1",
            coordinator_gpu_pci_bus_id="00000000:11:00.0",
        )
    with pytest.raises(ValueError, match="must be set together"):
        resolve(
            tmp_path,
            coordinator_gpu_uuid=(
                "GPU-95f8f212-9131-df99-fd53-7535965197d7"
            ),
        )


def test_shared_kv_pool_is_memory_budgeted_not_multiplied_by_concurrency(tmp_path):
    single = resolve(tmp_path, concurrency=1)
    four = resolve(tmp_path, concurrency=4)

    assert single.kv_pool_tokens == four.kv_pool_tokens == 1_573_888
    assert single.max_context_tokens == four.max_context_tokens == 400_000


def test_long_and_accuracy_profiles_expose_ds4_limits(tmp_path):
    long = resolve(tmp_path, profile="long", dspark=False)
    assert long.kv_dtype == "nvfp4"
    assert long.kv_bytes_per_source_page == 5_530_752
    assert long.kv_pool_tokens == 2_116_608
    assert long.max_context_tokens == 1_048_576
    assert long.environment["DS41RT_REAL_FULL_SERVE_KV_CACHE_DTYPE"] == "nvfp4"
    assert long.environment["DS41RT_DSPARK_ENABLED"] == "0"
    assert "DS41RT_REAL_FULL_DSPARK_FIXED_DRAFTS" not in long.environment

    accuracy = resolve(tmp_path, profile="accuracy")
    assert accuracy.kv_dtype == "bf16"
    assert accuracy.max_context_tokens == 200_000
    assert accuracy.max_output_tokens == 50_000
    assert accuracy.environment["DS41RT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE"] == (
        "bf16"
    )
    assert accuracy.environment["DS41RT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE"] == (
        "bf16"
    )


def test_full_dspark_draft_policy_is_an_explicit_diagnostic_control(tmp_path):
    full = resolve(tmp_path, dspark_draft_policy="full")
    assert full.dspark_draft_policy == "full"
    assert full.environment["DS41RT_REAL_FULL_DSPARK_FIXED_DRAFTS"] == "5"

    with pytest.raises(ValueError, match="must be full or adaptive"):
        resolve(tmp_path, dspark_draft_policy="unknown")


def test_dry_run_rejects_profile_options_unknown_to_a_stale_resolver():
    tool = Path(__file__).resolve().parents[2] / "tools" / "resolve_serve_profile.py"
    result = subprocess.run(
        [sys.executable, str(tool), "--dry-run", "--future-profile-option"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "unrecognized profile arguments" in result.stderr


def test_pool_and_context_limits_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="multiple of 256"):
        resolve(tmp_path, kv_pool_tokens=65)
    with pytest.raises(ValueError, match="cannot exceed kv_pool"):
        resolve(tmp_path, kv_pool_tokens=100_096, max_context_tokens=100_097)
    with pytest.raises(ValueError, match="headroom-safe"):
        resolve(tmp_path, kv_pool_tokens=2_000_128)
    with pytest.raises(ValueError, match="smaller than max_context"):
        resolve(
            tmp_path,
            max_context_tokens=100_000,
            max_output_tokens=100_000,
        )


def test_find_hf_snapshot_honors_pinned_revision(tmp_path):
    model_id = "test/model"
    revision = "a" * 40
    snapshot = (
        tmp_path
        / ("models--" + model_id.replace("/", "--"))
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    assert find_hf_snapshot(model_id, tmp_path, revision) == snapshot
