import pytest
from ds4rt_reference.deepseek_v4_mhc_capture import (
    capture_deepseek_v4_mhc_entry,
    capture_deepseek_v4_mhc_post_pre,
    capture_deepseek_v4_mhc_terminal,
    deepseek_v4_mhc_scratch_nbytes,
    plan_deepseek_v4_mhc,
    prepare_deepseek_v4_mhc_entry,
    prepare_deepseek_v4_mhc_post_pre,
    prepare_deepseek_v4_mhc_terminal,
    qualify_deepseek_v4_mhc_contract,
)


def test_flash_mhc_plan_preserves_fused_tp4_layer_boundaries() -> None:
    contract = plan_deepseek_v4_mhc(variant="flash", max_rows=2_048)

    assert contract.geometry.hidden == 4_096
    assert contract.geometry.function_width == 16_384
    assert contract.geometry.split_k == 64
    assert contract.geometry.checkpoint_bytes_per_layer == 3_162_328
    assert contract.scratch.total_bytes == 13_107_200
    assert contract.boundary_call_counts(layers=43) == {
        "entry_pre": 1,
        "fused_post_pre": 85,
        "terminal_post": 1,
        "terminal_head": 1,
    }
    assert not contract.serving_allocates
    assert contract.steady_state_fuses_post_pre
    assert contract.entry_pre_broadcasts_lanes
    assert contract.head_fuses_final_rmsnorm
    assert contract.head_uses_bound_normalized_output
    assert contract.head_scratch_bytes == 0
    assert contract.residual_ping_pong_buffers == 2
    assert contract.normalized_buffers == 1
    assert contract.mix_state_ping_pong_buffers == 2
    assert contract.expert_tensor_parallel == 4
    assert not contract.expert_parallel
    assert not contract.changes_expert_tp
    assert contract.expert_exchange_width == 4_096
    assert contract.owner == "coordinator-local-residual-stream"


def test_pro_mhc_plan_preserves_preview_geometry() -> None:
    contract = plan_deepseek_v4_mhc(variant="pro", max_rows=2_048)

    assert contract.geometry.hidden == 7_168
    assert contract.geometry.function_width == 28_672
    assert contract.geometry.split_k == 112
    assert contract.geometry.checkpoint_bytes_per_layer == 5_533_912
    assert contract.scratch.total_bytes == 22_937_600
    assert contract.expert_exchange_width == 7_168


def test_serving_mhc_boundary_exports_reuse_the_qualified_scratch() -> None:
    assert deepseek_v4_mhc_scratch_nbytes(
        variant="flash", max_rows=2_048
    ) == plan_deepseek_v4_mhc(
        variant="flash", max_rows=2_048
    ).scratch.total_bytes
    assert callable(prepare_deepseek_v4_mhc_entry)
    assert callable(capture_deepseek_v4_mhc_entry)
    assert callable(prepare_deepseek_v4_mhc_post_pre)
    assert callable(capture_deepseek_v4_mhc_post_pre)
    assert callable(prepare_deepseek_v4_mhc_terminal)
    assert callable(capture_deepseek_v4_mhc_terminal)


@pytest.mark.parametrize(
    "variant, expected",
    [
        (
            "flash",
            {
                "shared_partials_scratch": 13_107_200,
                "residual_lane_ping_pong": 134_217_728,
                "normalized_collapsed": 16_777_216,
                "post_comb_ping_pong": 327_680,
            },
        ),
        (
            "pro",
            {
                "shared_partials_scratch": 22_937_600,
                "residual_lane_ping_pong": 234_881_024,
                "normalized_collapsed": 29_360_128,
                "post_comb_ping_pong": 327_680,
            },
        ),
    ],
)
def test_mhc_required_buffers_are_fixed_and_caller_owned(
    variant: str, expected: dict[str, int]
) -> None:
    contract = plan_deepseek_v4_mhc(variant=variant, max_rows=2_048)
    assert contract.required_buffer_bytes() == expected


@pytest.mark.parametrize("variant", ["flash", "pro"])
def test_mhc_contract_matches_pinned_sparkinfer(variant: str) -> None:
    assert qualify_deepseek_v4_mhc_contract(variant=variant, max_rows=2_048)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"variant": "preview", "max_rows": 1}, "flash or pro"),
        ({"variant": "flash", "max_rows": 0}, "must be in"),
        ({"variant": "pro", "max_rows": 8_193}, "must be in"),
    ],
)
def test_mhc_contract_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plan_deepseek_v4_mhc(**kwargs)
