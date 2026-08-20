use super::*;

#[test]
fn flash_physical_kv_plan_matches_sparkinfer_page_abi() {
    let facts = ModelFacts::default();
    let plan = DeepseekV4PhysicalKvPlan::for_model(&facts, 128 * 1_024, false).unwrap();

    assert_eq!(plan.source_page_tokens, 256);
    assert_eq!(plan.source_page_count, 512);
    assert_eq!(plan.layers.len(), 43);
    assert!(!plan.includes_dspark);
    assert_eq!(plan.persistent_bytes, 3_808_198_656);
    assert_eq!(plan.compressor_state_bytes_per_sequence, 24_412_160);
    assert_eq!(plan.persistent_bytes_per_logical_token(), 29_054.25);
    assert_eq!(plan.active_sequence_bytes(4), Some(3_905_847_296));

    let sliding = plan.layer(0).unwrap();
    assert_eq!(sliding.regions.len(), 1);
    assert_eq!(
        sliding
            .region(DeepseekV4KvRegionKind::Main)
            .unwrap()
            .bytes_per_page,
        149_760
    );
    assert_eq!(sliding.compressor_state_bytes_per_sequence, 0);

    let c4 = plan.layer(2).unwrap();
    assert_eq!(c4.regions.len(), 3);
    assert_eq!(
        c4.region(DeepseekV4KvRegionKind::Compressed)
            .unwrap()
            .rows_per_page,
        64
    );
    assert_eq!(
        c4.region(DeepseekV4KvRegionKind::Compressed)
            .unwrap()
            .bytes_per_page,
        37_440
    );
    assert_eq!(
        c4.region(DeepseekV4KvRegionKind::Indexer)
            .unwrap()
            .bytes_per_page,
        8_448
    );
    assert_eq!(c4.compressor_state_bytes_per_sequence, 163_840);

    let c128 = plan.layer(3).unwrap();
    assert_eq!(c128.regions.len(), 2);
    assert_eq!(
        c128.region(DeepseekV4KvRegionKind::Compressed)
            .unwrap()
            .rows_per_page,
        2
    );
    assert_eq!(
        c128.region(DeepseekV4KvRegionKind::Compressed)
            .unwrap()
            .bytes_per_page,
        1_728
    );
    assert_eq!(c128.compressor_state_bytes_per_sequence, 1_048_576);

    let all_regions = plan
        .layers
        .iter()
        .flat_map(|layer| layer.regions.iter())
        .collect::<Vec<_>>();
    for regions in all_regions.windows(2) {
        assert_eq!(regions[0].end_offset_bytes(), regions[1].offset_bytes);
    }
    assert_eq!(
        plan.layers
            .last()
            .unwrap()
            .regions
            .last()
            .unwrap()
            .end_offset_bytes(),
        plan.persistent_bytes
    );
}

#[test]
fn flash_native_nvfp4_plan_uses_compact_row_major_pages() {
    let plan = DeepseekV4PhysicalKvPlan::for_model_with_format(
        &ModelFacts::default(),
        128 * 1_024,
        false,
        DeepseekV4KvCacheFormat::Nvfp4,
    )
    .unwrap();

    assert_eq!(plan.cache_format, DeepseekV4KvCacheFormat::Nvfp4);
    assert_eq!(plan.source_page_count, 512);
    assert_eq!(plan.persistent_bytes, 2_831_745_024);
    assert_eq!(plan.persistent_bytes_per_logical_token(), 21_604.5);
    assert_eq!(
        plan.layer(0)
            .unwrap()
            .region(DeepseekV4KvRegionKind::Main)
            .unwrap()
            .bytes_per_page,
        110_592
    );
    assert_eq!(
        plan.layer(2)
            .unwrap()
            .region(DeepseekV4KvRegionKind::Compressed)
            .unwrap()
            .bytes_per_page,
        27_648
    );
    assert_eq!(
        plan.layer(3)
            .unwrap()
            .region(DeepseekV4KvRegionKind::Compressed)
            .unwrap()
            .bytes_per_page,
        864
    );
    // The learned C=4 index cache remains FP8 plus FP32 scales.
    assert_eq!(
        plan.layer(2)
            .unwrap()
            .region(DeepseekV4KvRegionKind::Indexer)
            .unwrap()
            .bytes_per_page,
        8_448
    );

    let boundary = plan.boundary_copy_plan(14).unwrap();
    let sliding = &boundary.layers[0];
    assert_eq!(sliding.spans.len(), 1);
    assert_eq!(sliding.spans[0].plane, DeepseekV4KvPagePlane::Payload);
    assert_eq!(sliding.spans[0].length_bytes, 14 * 432);

    let c4 = &boundary.layers[2];
    assert_eq!(c4.spans.len(), 4);
    assert_eq!(c4.spans[2].region, DeepseekV4KvRegionKind::Indexer);
    assert_eq!(c4.spans[3].plane, DeepseekV4KvPagePlane::Scale);
}

#[test]
fn physical_kv_plan_maps_compressed_slots_and_rewind_replay_boundaries() {
    let plan = DeepseekV4PhysicalKvPlan::for_model(&ModelFacts::default(), 1_025, false).unwrap();
    assert_eq!(plan.source_page_count, 5);

    let c4 = plan.layer(2).unwrap();
    assert_eq!(
        c4.main_slot(256),
        Some(DeepseekV4KvSlot {
            page_index: 1,
            slot_index: 0
        })
    );
    assert_eq!(c4.main_slot(1_280), None);
    assert_eq!(c4.completed_compressed_slot(2), None);
    assert_eq!(
        c4.completed_compressed_slot(3),
        Some(DeepseekV4KvSlot {
            page_index: 0,
            slot_index: 0
        })
    );
    assert_eq!(
        c4.completed_compressed_slot(259),
        Some(DeepseekV4KvSlot {
            page_index: 1,
            slot_index: 0
        })
    );
    assert_eq!(c4.compressed_blocks_before_rewind(3), 0);
    assert_eq!(c4.compressed_blocks_before_rewind(4), 1);
    assert_eq!(c4.compressed_blocks_before_rewind(5), 1);
    assert_eq!(c4.compressor_replay_start(260), Some(256));
    assert_eq!(c4.sliding_replay_start(260), 132);

    let c128 = plan.layer(3).unwrap();
    assert_eq!(c128.completed_compressed_slot(126), None);
    assert_eq!(
        c128.completed_compressed_slot(127),
        Some(DeepseekV4KvSlot {
            page_index: 0,
            slot_index: 0
        })
    );
    assert_eq!(c128.compressor_replay_start(260), Some(256));
}

#[test]
fn physical_kv_plan_adds_three_sliding_dspark_regions_without_compressor_state() {
    let facts = ModelFacts::default();
    let target = DeepseekV4PhysicalKvPlan::for_model(&facts, 128 * 1_024, false).unwrap();
    let with_dspark = DeepseekV4PhysicalKvPlan::for_model(&facts, 128 * 1_024, true).unwrap();

    assert_eq!(with_dspark.layers.len(), 46);
    assert!(with_dspark.includes_dspark);
    assert_eq!(
        with_dspark.persistent_bytes - target.persistent_bytes,
        3 * 512 * 149_760
    );
    assert_eq!(
        with_dspark.compressor_state_bytes_per_sequence,
        target.compressor_state_bytes_per_sequence
    );
    assert!(with_dspark.layers[43].source.is_dspark());
}

#[test]
fn boundary_copy_plan_preserves_planar_sparkinfer_pages_and_replays_state() {
    let plan = DeepseekV4PhysicalKvPlan::for_model(&ModelFacts::default(), 1_024, false).unwrap();
    let boundary = plan.boundary_copy_plan(14).unwrap();

    assert_eq!(boundary.layers.len(), 43);
    assert_eq!(boundary.copied_bytes(), 396_676);

    let sliding = &boundary.layers[0];
    assert_eq!(sliding.valid_main_rows, 14);
    assert_eq!(sliding.valid_compressed_rows, 0);
    assert_eq!(sliding.compressor_replay_start_source_slot, None);
    assert_eq!(sliding.spans.len(), 2);
    assert_eq!(sliding.spans[0].plane, DeepseekV4KvPagePlane::Payload);
    assert_eq!(sliding.spans[0].offset_within_page_bytes, 0);
    assert_eq!(sliding.spans[0].length_bytes, 14 * 576);
    assert_eq!(sliding.spans[1].plane, DeepseekV4KvPagePlane::Scale);
    assert_eq!(sliding.spans[1].offset_within_page_bytes, 256 * 576);
    assert_eq!(sliding.spans[1].length_bytes, 14 * 8);

    let c4 = &boundary.layers[2];
    assert_eq!(c4.valid_compressed_rows, 3);
    assert_eq!(c4.compressor_replay_start_source_slot, Some(8));
    assert_eq!(c4.spans.len(), 6);
    assert_eq!(c4.copied_bytes(), 10_324);
    assert_eq!(c4.spans[2].region, DeepseekV4KvRegionKind::Compressed);
    assert_eq!(c4.spans[2].length_bytes, 3 * 576);
    assert_eq!(c4.spans[3].offset_within_page_bytes, 64 * 576);
    assert_eq!(c4.spans[3].length_bytes, 3 * 8);
    assert_eq!(c4.spans[4].region, DeepseekV4KvRegionKind::Indexer);
    assert_eq!(c4.spans[4].length_bytes, 3 * 128);
    assert_eq!(c4.spans[5].offset_within_page_bytes, 64 * 128);
    assert_eq!(c4.spans[5].length_bytes, 3 * 4);

    let c128 = &boundary.layers[3];
    assert_eq!(c128.valid_compressed_rows, 0);
    assert_eq!(c128.compressor_replay_start_source_slot, Some(0));
    assert_eq!(c128.spans.len(), 2);

    let source_page = 1;
    assert_eq!(
        c4.spans[3].page_offset_bytes(source_page),
        Some(c4.spans[3].region_offset_bytes + 37_440 + 64 * 576)
    );
    assert_eq!(c4.spans[3].page_offset_bytes(4), None);

    let later_boundary = plan.boundary_copy_plan(130).unwrap();
    let later_c4 = &later_boundary.layers[2];
    assert_eq!(later_c4.valid_compressed_rows, 32);
    assert_eq!(later_c4.compressor_replay_start_source_slot, Some(124));
    let later_c128 = &later_boundary.layers[3];
    assert_eq!(later_c128.valid_compressed_rows, 1);
    assert_eq!(later_c128.compressor_replay_start_source_slot, Some(128));
    assert_eq!(later_c128.spans.len(), 4);
    assert_eq!(later_c128.spans[2].length_bytes, 576);
    assert_eq!(later_c128.spans[3].offset_within_page_bytes, 2 * 576);
    assert_eq!(later_c128.spans[3].length_bytes, 8);

    assert!(plan.boundary_copy_plan(0).is_err());
    assert!(plan.boundary_copy_plan(256).is_err());
}
