use super::*;

fn flash_compressors() -> Vec<DeepseekV4CompressorLayerExecutionPlan> {
    deepseek_v4_compressor_execution_plans(&ModelFacts::default()).unwrap()
}

#[test]
fn flash_layer_plans_match_joint_projection_and_state_geometry() {
    let plans = flash_compressors();
    let sliding = &plans[0];
    assert!(sliding.main.is_none());
    assert!(sliding.indexer.is_none());
    assert_eq!(sliding.joint_projection_width, 0);

    let c4 = plans
        .iter()
        .find(|plan| {
            plan.main
                .as_ref()
                .is_some_and(|spec| spec.compress_ratio == 4)
        })
        .unwrap();
    let main = c4.main.as_ref().unwrap();
    assert_eq!(main.kind, DeepseekV4CompressorKind::Main);
    assert!(main.overlap);
    assert_eq!(main.coefficient, 2);
    assert_eq!(main.projected_width, 1_024);
    assert_eq!(main.state_rows, 16);
    assert_eq!(main.state_width, 1_024);
    assert_eq!(main.paired_state_bytes(), Some(131_072));
    let indexer = c4.indexer.as_ref().unwrap();
    assert_eq!(indexer.projected_width, 256);
    assert_eq!(indexer.paired_state_bytes(), Some(32_768));
    assert_eq!(c4.joint_projection_width, 2_560);
    assert_eq!(c4.paired_state_bytes(), Some(163_840));

    let c128 = plans
        .iter()
        .find(|plan| {
            plan.main
                .as_ref()
                .is_some_and(|spec| spec.compress_ratio == 128)
        })
        .unwrap();
    let main = c128.main.as_ref().unwrap();
    assert!(!main.overlap);
    assert_eq!(main.coefficient, 1);
    assert_eq!(main.projected_width, 512);
    assert_eq!(main.state_rows, 256);
    assert_eq!(main.state_width, 512);
    assert_eq!(c128.joint_projection_width, 1_024);
    assert_eq!(c128.paired_state_bytes(), Some(1_048_576));
}

#[test]
fn c4_decode_writes_position_ring_without_rolling() {
    let plans = flash_compressors();
    let c4 = plans.iter().find(|plan| plan.indexer.is_some()).unwrap();

    let first = c4.decode_step(0).unwrap().unwrap();
    assert_eq!(first.ape_row, 0);
    assert_eq!(first.state_row, 0);
    assert!(!first.emits);
    assert_eq!(first.compressed_slot, None);
    assert!(!first.rolls_current_window_to_previous);

    let boundary = c4.decode_step(3).unwrap().unwrap();
    assert_eq!(boundary.ape_row, 3);
    assert_eq!(boundary.state_row, 3);
    assert!(boundary.emits);
    assert_eq!(boundary.compressed_slot, Some(0));
    assert_eq!(boundary.rope_position, Some(0));
    assert!(!boundary.rolls_current_window_to_previous);

    let next = c4.decode_step(7).unwrap().unwrap();
    assert_eq!(next.compressed_slot, Some(1));
    assert_eq!(next.rope_position, Some(4));
}

#[test]
fn c128_decode_emits_at_group_end_without_overlap_rollover() {
    let plans = flash_compressors();
    let c128 = plans
        .iter()
        .find(|plan| {
            plan.main
                .as_ref()
                .is_some_and(|spec| spec.compress_ratio == 128)
        })
        .unwrap();

    let before = c128.decode_step(126).unwrap().unwrap();
    assert_eq!(before.state_row, 126);
    assert!(!before.emits);
    let boundary = c128.decode_step(127).unwrap().unwrap();
    assert_eq!(boundary.ape_row, 127);
    assert_eq!(boundary.state_row, 127);
    assert_eq!(boundary.compressed_slot, Some(0));
    assert_eq!(boundary.rope_position, Some(0));
    assert!(!boundary.rolls_current_window_to_previous);
}

#[test]
fn c4_prefill_retains_last_complete_group_and_current_remainder() {
    let plans = flash_compressors();
    let c4 = plans.iter().find(|plan| plan.indexer.is_some()).unwrap();
    let prefill = c4.prefill(10).unwrap().unwrap();

    assert_eq!(prefill.complete_groups, 2);
    assert_eq!(prefill.remainder, 2);
    assert!(prefill.reset_state_before_projection);
    assert!(prefill.first_overlap_half_is_inactive);
    assert_eq!(prefill.output_rope_position(0), Some(0));
    assert_eq!(prefill.output_rope_position(1), Some(4));
    assert_eq!(prefill.output_rope_position(2), None);
    assert_eq!(
        prefill.previous_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 4,
            rows: 4,
            state_row_start: 0,
            ape_row_start: 0,
        })
    );
    assert_eq!(
        prefill.current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 8,
            rows: 2,
            state_row_start: 4,
            ape_row_start: 0,
        })
    );
}

#[test]
fn c128_prefill_keeps_only_incomplete_current_group_in_state() {
    let plans = flash_compressors();
    let c128 = plans
        .iter()
        .find(|plan| {
            plan.main
                .as_ref()
                .is_some_and(|spec| spec.compress_ratio == 128)
        })
        .unwrap();
    let prefill = c128.prefill(260).unwrap().unwrap();

    assert_eq!(prefill.complete_groups, 2);
    assert_eq!(prefill.remainder, 4);
    assert_eq!(prefill.previous_window, None);
    assert_eq!(
        prefill.current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 256,
            rows: 4,
            state_row_start: 0,
            ape_row_start: 0,
        })
    );
    assert_eq!(prefill.output_rope_position(1), Some(128));
}

#[test]
fn c4_continuation_maps_carried_prefix_outputs_and_terminal_windows() {
    let plans = flash_compressors();
    let c4 = plans.iter().find(|plan| plan.indexer.is_some()).unwrap();
    let continuation = c4.continuation_prefill(6, 7).unwrap().unwrap();

    assert_eq!(continuation.logical_end, 13);
    assert_eq!(continuation.complete_groups, 2);
    assert_eq!(continuation.carried_remainder, 2);
    assert_eq!(continuation.terminal_remainder, 1);
    assert_eq!(continuation.first_output_group_start, Some(4));
    assert_eq!(continuation.output_group_start(0), Some(4));
    assert_eq!(continuation.output_group_start(1), Some(8));
    assert_eq!(continuation.output_group_start(2), None);
    assert_eq!(continuation.output_slot(0), Some(1));
    assert_eq!(continuation.output_slot(1), Some(2));
    assert_eq!(continuation.output_rope_position(1), Some(8));
    assert_eq!(
        continuation.carried_previous_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 0,
            rows: 4,
            state_row_start: 0,
            ape_row_start: 0,
        })
    );
    assert_eq!(
        continuation.carried_current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 4,
            rows: 2,
            state_row_start: 4,
            ape_row_start: 0,
        })
    );
    assert_eq!(
        continuation.terminal_previous_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 8,
            rows: 4,
            state_row_start: 0,
            ape_row_start: 0,
        })
    );
    assert_eq!(
        continuation.terminal_current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 12,
            rows: 1,
            state_row_start: 4,
            ape_row_start: 0,
        })
    );
}

#[test]
fn c4_continuation_handles_no_boundary_and_first_group_boundary() {
    let plans = flash_compressors();
    let c4 = plans.iter().find(|plan| plan.indexer.is_some()).unwrap();

    let no_boundary = c4.continuation_prefill(6, 1).unwrap().unwrap();
    assert_eq!(no_boundary.complete_groups, 0);
    assert_eq!(no_boundary.first_output_group_start, None);
    assert_eq!(no_boundary.output_slot_start, None);
    assert_eq!(
        no_boundary.terminal_current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 4,
            rows: 3,
            state_row_start: 4,
            ape_row_start: 0,
        })
    );

    let first_boundary = c4.continuation_prefill(3, 1).unwrap().unwrap();
    assert_eq!(first_boundary.complete_groups, 1);
    assert_eq!(first_boundary.output_group_start(0), Some(0));
    assert_eq!(first_boundary.output_slot(0), Some(0));
    assert_eq!(first_boundary.carried_previous_window, None);
    assert_eq!(
        first_boundary.carried_current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 0,
            rows: 3,
            state_row_start: 4,
            ape_row_start: 0,
        })
    );
    assert_eq!(
        first_boundary.terminal_previous_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 0,
            rows: 4,
            state_row_start: 0,
            ape_row_start: 0,
        })
    );
    assert_eq!(first_boundary.terminal_current_window, None);
}

#[test]
fn c128_continuation_finishes_carried_group_and_retains_new_remainder() {
    let plans = flash_compressors();
    let c128 = plans
        .iter()
        .find(|plan| {
            plan.main
                .as_ref()
                .is_some_and(|spec| spec.compress_ratio == 128)
        })
        .unwrap();
    let continuation = c128.continuation_prefill(250, 20).unwrap().unwrap();

    assert_eq!(continuation.logical_end, 270);
    assert_eq!(continuation.complete_groups, 1);
    assert_eq!(continuation.carried_remainder, 122);
    assert_eq!(continuation.terminal_remainder, 14);
    assert_eq!(continuation.output_group_start(0), Some(128));
    assert_eq!(continuation.output_slot(0), Some(1));
    assert_eq!(continuation.carried_previous_window, None);
    assert_eq!(
        continuation.carried_current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 128,
            rows: 122,
            state_row_start: 0,
            ape_row_start: 0,
        })
    );
    assert_eq!(continuation.terminal_previous_window, None);
    assert_eq!(
        continuation.terminal_current_window,
        Some(DeepseekV4CompressorStateFill {
            source_start: 256,
            rows: 14,
            state_row_start: 0,
            ape_row_start: 0,
        })
    );
}

#[test]
fn continuation_rejects_empty_or_overflowing_ranges() {
    let plans = flash_compressors();
    let c4 = plans.iter().find(|plan| plan.indexer.is_some()).unwrap();
    assert!(c4.continuation_prefill(4, 0).is_err());
    assert!(c4.continuation_prefill(usize::MAX, 1).is_err());
}

#[test]
fn sliding_layers_have_no_compressor_transition() {
    let plans = flash_compressors();
    assert_eq!(plans[0].decode_step(0).unwrap(), None);
    assert_eq!(plans[0].prefill(1).unwrap(), None);
    assert_eq!(plans[0].continuation_prefill(4, 1).unwrap(), None);
}
