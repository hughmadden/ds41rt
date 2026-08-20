import pytest

from ds4rt_reference.deepseek_v4_attention_output_capture import (
    plan_deepseek_v4_attention_output,
    qualify_deepseek_v4_attention_output_contract,
)


def test_flash_output_plan_preserves_full_coordinator_grouping() -> None:
    contract = plan_deepseek_v4_attention_output(variant="flash", max_rows=2_048)

    assert contract.geometry.hidden == 4_096
    assert contract.geometry.heads == 64
    assert contract.geometry.groups == 8
    assert contract.geometry.heads_per_group == 8
    assert contract.geometry.group_width == 4_096
    assert contract.geometry.rank == 1_024
    assert contract.geometry.projected_width == 8_192
    assert contract.geometry.checkpoint_weight_bytes == 67_112_960
    assert contract.scratch.total_bytes == 139_460_608
    assert not contract.serving_allocates
    assert contract.fuses_inverse_rope
    assert contract.output_is_arena_view
    assert not contract.changes_expert_tp
    assert contract.owner == "coordinator-local-attention"
    assert contract.status == "qualified-grouped-output-projection-not-active"


def test_pro_output_plan_preserves_preview_grouping_and_fixed_arena() -> None:
    contract = plan_deepseek_v4_attention_output(variant="pro", max_rows=2_048)

    assert contract.geometry.hidden == 7_168
    assert contract.geometry.heads == 128
    assert contract.geometry.groups == 16
    assert contract.geometry.heads_per_group == 8
    assert contract.geometry.group_width == 4_096
    assert contract.geometry.projected_width == 16_384
    assert contract.geometry.checkpoint_weight_bytes == 184_560_640
    assert contract.scratch.total_bytes == 274_726_912
    assert contract.scratch.output_offset == 245_366_784


def test_output_required_buffers_keep_attention_and_arena_caller_owned() -> None:
    contract = plan_deepseek_v4_attention_output(variant="flash", max_rows=16)
    required = contract.required_buffer_bytes(max_positions=131_072)

    assert required["attention_output"] == 16 * 64 * 512 * 2
    assert required["positions"] == 16 * 8
    assert required["cos_sin_cache"] == 131_072 * 64 * 4
    assert required["scratch_and_output"] == 1_232_896


@pytest.mark.parametrize("variant", ["flash", "pro"])
def test_output_contract_matches_pinned_sparkinfer(variant: str) -> None:
    assert qualify_deepseek_v4_attention_output_contract(
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
def test_output_contract_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plan_deepseek_v4_attention_output(**kwargs)
