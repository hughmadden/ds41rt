from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "export_ds4_flash_spark_moe_aot.py"
)
SPEC = importlib.util.spec_from_file_location("ds4_spark_aot_profiles", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_flash_and_pro_profiles_preserve_strict_tp4_without_widening() -> None:
    flash = MODULE.profile_manifest(
        MODULE.KERNEL_PROFILES["flash"], transformer_blocks=46
    )
    pro = MODULE.profile_manifest(
        MODULE.KERNEL_PROFILES["pro"], transformer_blocks=64
    )

    assert MODULE.DEFAULT_TRANSFORMER_BLOCKS == {"flash": 46, "pro": 64}
    assert MODULE.EXL3_CODEBOOK == "mcg"
    assert MODULE.EXL3_TILE_CONFIGS == {
        "flash": (64, 128, 64, 128),
        "pro": (64, 256, 64, 256),
    }
    assert flash["trellis_codebook"] == "mcg"
    assert flash["trellis_lut_bytes"] == 4_096
    assert flash["trellis_tile_config"] == [64, 128, 64, 128]
    assert flash["local_intermediate_size"] == 512
    assert flash["projection_trellis_bytes"] == 524_288
    assert flash["resident_weight_bytes_per_block"] == 402_653_184
    assert flash["resident_total_bytes_per_rank"] == 18_847_912_856

    assert pro["hidden_size"] == 7_168
    assert pro["global_intermediate_size"] == 3_072
    assert pro["local_intermediate_size"] == 768
    assert pro["num_experts"] == 384
    assert pro["top_k"] == 6
    assert pro["trellis_tile_config"] == [64, 256, 64, 256]
    assert pro["native_fp4"] is False
    assert pro["projection_trellis_bytes"] == 1_376_256
    assert pro["resident_weight_bytes_per_block"] == 1_585_446_912
    assert pro["transformer_blocks"] == 64
    assert pro["resident_total_bytes_per_rank"] == 102_639_273_216


def test_exl3_export_pins_checkpoint_codebook_at_the_compiler_call() -> None:
    source = SCRIPT.read_text()
    assert "trellis_codebook=EXL3_CODEBOOK" in source


def test_flash_exports_shape_specific_joint_dspark_direct_kernels() -> None:
    assert MODULE.FLASH_DIRECT_M4_CAPACITY == 4
    assert MODULE.FLASH_DIRECT_M4_MIN_ACTIVE == 2
    assert MODULE.FLASH_DIRECT_M4_BLOCK_SIZE == 16
    assert MODULE.FLASH_DIRECT_M6_CAPACITY == 8
    assert MODULE.FLASH_DIRECT_M6_BLOCK_SIZE == 16
    source = SCRIPT.read_text()
    assert 'label="decode_m4_direct_m4"' in source
    assert 'label="decode_m8_direct_m6"' in source
    assert 'if profile.variant == "flash":' in source
    assert "EXL3_K2_DIRECT_M4_MIN_ACTIVE" in source


def test_flash_mixed_decode_uses_direct_routes_without_splitting_tiers() -> None:
    source = SCRIPT.read_text()
    assert "direct_topk = rows <= FLASH_DIRECT_M6_CAPACITY" in source
    assert "direct_topk_routes=direct_topk" in source
    assert "EXL3_{macro}_DIRECT_TOPK" in source
    assert "tier0:K2,tier1:K3,full_rotation:1,one_grid:1" in source


def test_pro_profile_uses_distinct_aot_symbol_and_macro_names() -> None:
    profile = MODULE.KERNEL_PROFILES["pro"]
    assert profile.symbol_prefix == "ds4_pro"
    assert profile.macro_prefix == "DS4RT_DS4_PRO"
    assert MODULE.KERNEL_PROFILES["flash"].symbol_prefix == "ds4_flash"
