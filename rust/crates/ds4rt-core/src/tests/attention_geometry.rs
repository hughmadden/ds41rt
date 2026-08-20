use super::*;

#[test]
fn flash_attention_plan_preserves_native_target_and_dspark_schedule() {
    let facts = ModelFacts::default();
    let plan = DeepseekV4AttentionPlan::from_model_facts(&facts).unwrap();

    assert_eq!(plan.target_layer_count, 43);
    assert_eq!(plan.dspark_layer_count, 3);
    assert_eq!(plan.layers.len(), 46);
    assert_eq!(plan.geometry.head_dim, 512);
    assert_eq!(plan.geometry.nope_head_dim, 448);
    assert_eq!(plan.geometry.rope_head_dim, 64);
    assert_eq!(plan.geometry.q_lora_rank, 1024);
    assert_eq!(plan.geometry.o_lora_rank, 1024);
    assert_eq!(plan.geometry.o_groups, 8);

    let target_counts = attention_kind_counts(plan.target_layers());
    assert_eq!(target_counts, (2, 21, 20));
    let all_counts = attention_kind_counts(&plan.layers);
    assert_eq!(all_counts, (5, 21, 20));

    let c4 = plan.layer(2).unwrap();
    assert_eq!(c4.kind, AttentionKind::CompressedSparse);
    assert_eq!(c4.compress_ratio, 4);
    assert!(c4.uses_compressor());
    assert!(c4.uses_indexer());
    assert!(c4.uses_yarn());
    assert_eq!(c4.compressor_state_rows(), 16);
    assert_eq!(c4.completed_compressed_blocks(1_025), 256);
    assert_eq!(c4.persistent_kv_slots(1_025), 384);
    assert_eq!(
        c4.compressed_selection,
        DeepseekV4CompressedSelection::LearnedIndexer { top_k: 512 }
    );

    let c128 = plan.layer(3).unwrap();
    assert_eq!(c128.kind, AttentionKind::HeavilyCompressed);
    assert!(!c128.uses_indexer());
    assert_eq!(c128.compressor_state_rows(), 256);
    assert_eq!(
        c128.compressed_selection,
        DeepseekV4CompressedSelection::AllCompletedBlocks
    );

    let draft = plan.layer(43).unwrap();
    assert_eq!(draft.kind, AttentionKind::Sliding);
    assert_eq!(draft.compressor_state_rows(), 0);
    assert_eq!(draft.persistent_kv_slots(1_025), 128);
    assert_eq!(
        draft.source,
        DeepseekV4AttentionLayerSource::Dspark {
            block_index: 0,
            target_layer_id: 40
        }
    );
    assert_eq!(
        plan.layer(0).unwrap().source.checkpoint_block_prefix(),
        "layers.0"
    );
    assert_eq!(draft.source.checkpoint_block_prefix(), "mtp.0");
    assert_eq!(draft.source.target_layer_id(), 40);
    assert!(draft.source.is_dspark());
    plan.validate_sparkinfer_sm120_contract().unwrap();
}

#[test]
fn pro_attention_plan_preserves_equal_kernel_geometry_and_native_schedule() {
    let mut facts = ModelFacts::default();
    facts.variant = ModelVariant::Pro;
    facts.hidden_size = DS4_PRO_HIDDEN_SIZE;
    facts.num_hidden_layers = DS4_PRO_NUM_HIDDEN_LAYERS;
    facts.attention_heads = 128;
    facts.q_lora_rank = DS4_PRO_Q_LORA_RANK;
    facts.o_groups = 16;
    facts.compress_ratios = DS4_PRO_COMPRESS_RATIOS.to_vec();
    facts.dspark_markov_rank = DS4_PRO_DSPARK_MARKOV_RANK;
    facts.dspark_target_layer_ids = vec![58, 59, 60];

    let plan = DeepseekV4AttentionPlan::from_model_facts(&facts).unwrap();
    assert_eq!(attention_kind_counts(plan.target_layers()), (0, 30, 31));
    assert_eq!(attention_kind_counts(&plan.layers), (3, 30, 31));
    assert_eq!(plan.geometry.hidden_size, 7168);
    assert_eq!(plan.geometry.attention_heads, 128);
    assert_eq!(plan.geometry.o_groups, 16);
    assert_eq!(plan.dspark_layers()[2].source.target_layer_id(), 60);
    plan.validate_sparkinfer_sm120_contract().unwrap();
}

#[test]
fn attention_plan_rejects_truncated_schedule_and_kernel_incompatible_heads() {
    let mut facts = ModelFacts::default();
    facts.compress_ratios.pop();
    let error = DeepseekV4AttentionPlan::from_model_facts(&facts)
        .unwrap_err()
        .to_string();
    assert!(error.contains("compression schedule has 45 entries"));

    let mut facts = ModelFacts::default();
    facts.kv_heads = 2;
    let plan = DeepseekV4AttentionPlan::from_model_facts(&facts).unwrap();
    let error = plan
        .validate_sparkinfer_sm120_contract()
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires one KV head"));
}

fn attention_kind_counts(layers: &[DeepseekV4AttentionLayerPlan]) -> (usize, usize, usize) {
    layers.iter().fold((0, 0, 0), |mut counts, layer| {
        match layer.kind {
            AttentionKind::Sliding => counts.0 += 1,
            AttentionKind::CompressedSparse => counts.1 += 1,
            AttentionKind::HeavilyCompressed => counts.2 += 1,
        }
        counts
    })
}
