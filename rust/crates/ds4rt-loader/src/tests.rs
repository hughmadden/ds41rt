use super::{
    build_load_plan, encode_tokenizer_text, load_tensor_bytes, load_tensor_bytes_with_options,
    load_tensor_rows, load_tensor_rows_with_options, model_cache_dir,
    native_deepseek_v4_attention_tensor_specs, read_safetensors_metadata, read_tensor_bytes_into,
    read_tensor_bytes_into_with_options, read_tensor_row_prefix_into,
    read_tensor_row_prefix_into_with_options, read_tensor_row_window_into, read_tensor_rows_into,
    read_tensor_rows_into_with_options, resolve_snapshot, resolve_snapshot_at_revision,
    streaming_token_decoder, validate_native_deepseek_v4_attention_catalog,
    validate_native_fp4_expert_catalog, NativeDeepseekV4AttentionTensorFamily,
    NativeFp4ProjectionKind, TensorLoadOptions,
};
use crate::catalog::{
    classify_tensor, is_quantization_tensor, read_model_facts, resolve_model_quantization_config,
    tensor_layer_id,
};
use ds4rt_core::{
    owner_for_expert, AttentionKind, DType, ModelFacts, ModelVariant, PlacementPolicy,
    TensorCatalog, TensorInfo, TensorRole, COORDINATOR_HOST, DS4_FLASH_NUM_HIDDEN_LAYERS,
    DS4_PRO_COMPRESS_RATIOS, GLM52_MTP_LAYER_ID,
};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[test]
fn model_cache_path_uses_hf_layout() {
    let p = model_cache_dir(Path::new("/tmp/hf"), "a/b");
    assert_eq!(p, PathBuf::from("/tmp/hf/hub/models--a--b"));
}

#[test]
fn snapshot_resolution_honors_main_ref_over_lexicographic_order() {
    let tempdir = tempfile::tempdir().unwrap();
    let model_root = model_cache_dir(tempdir.path(), "a/b");
    std::fs::create_dir_all(model_root.join("snapshots/aaa")).unwrap();
    std::fs::create_dir_all(model_root.join("snapshots/zzz")).unwrap();
    std::fs::create_dir_all(model_root.join("refs")).unwrap();
    std::fs::write(model_root.join("refs/main"), "aaa\n").unwrap();

    let resolution = resolve_snapshot("a/b", Some(tempdir.path())).unwrap();

    assert_eq!(
        resolution.snapshot_path,
        Some(model_root.join("snapshots/aaa"))
    );
    assert_eq!(resolution.snapshots.len(), 2);
}

#[test]
fn explicit_snapshot_revision_wins_over_main_ref() {
    let tempdir = tempfile::tempdir().unwrap();
    let model_root = model_cache_dir(tempdir.path(), "a/b");
    std::fs::create_dir_all(model_root.join("snapshots/aaa")).unwrap();
    std::fs::create_dir_all(model_root.join("snapshots/zzz")).unwrap();
    std::fs::create_dir_all(model_root.join("refs")).unwrap();
    std::fs::write(model_root.join("refs/main"), "aaa\n").unwrap();

    let resolution =
        resolve_snapshot_at_revision("a/b", Some(tempdir.path()), Some("zzz")).unwrap();

    assert_eq!(
        resolution.snapshot_path,
        Some(model_root.join("snapshots/zzz"))
    );
}

#[test]
fn explicit_snapshot_revision_rejects_missing_or_traversing_selection() {
    let tempdir = tempfile::tempdir().unwrap();
    let model_root = model_cache_dir(tempdir.path(), "a/b");
    std::fs::create_dir_all(model_root.join("snapshots/aaa")).unwrap();

    assert!(
        resolve_snapshot_at_revision("a/b", Some(tempdir.path()), Some("missing"))
            .unwrap_err()
            .to_string()
            .contains("missing snapshot")
    );
    assert!(
        resolve_snapshot_at_revision("a/b", Some(tempdir.path()), Some("../aaa"))
            .unwrap_err()
            .to_string()
            .contains("invalid revision")
    );
}

#[test]
fn snapshot_resolution_rejects_missing_or_traversing_main_ref() {
    let tempdir = tempfile::tempdir().unwrap();
    let model_root = model_cache_dir(tempdir.path(), "a/b");
    std::fs::create_dir_all(model_root.join("snapshots/aaa")).unwrap();
    std::fs::create_dir_all(model_root.join("refs")).unwrap();

    std::fs::write(model_root.join("refs/main"), "missing\n").unwrap();
    assert!(resolve_snapshot("a/b", Some(tempdir.path()))
        .unwrap_err()
        .to_string()
        .contains("missing snapshot"));

    std::fs::write(model_root.join("refs/main"), "../aaa\n").unwrap();
    assert!(resolve_snapshot("a/b", Some(tempdir.path()))
        .unwrap_err()
        .to_string()
        .contains("invalid revision"));

    std::fs::remove_file(model_root.join("refs/main")).unwrap();
    std::fs::create_dir(model_root.join("refs/main")).unwrap();
    assert!(resolve_snapshot("a/b", Some(tempdir.path()))
        .unwrap_err()
        .to_string()
        .contains("not a regular file"));
}

#[test]
fn compact_exl3_config_resolves_external_tensor_storage() {
    let tempdir = tempfile::tempdir().unwrap();
    let compact = serde_json::json!({
        "bits": 2.0,
        "codebook": "mcg",
        "quant_method": "exl3"
    });
    let mut full = compact.clone();
    full.as_object_mut().unwrap().insert(
        "tensor_storage".to_owned(),
        serde_json::json!({"model.layers.0.mlp.experts.0.gate_proj": {"quant_format": "exl3"}}),
    );
    std::fs::write(
        tempdir.path().join("quantize_config.json"),
        serde_json::to_vec(&full).unwrap(),
    )
    .unwrap();

    let resolved = resolve_model_quantization_config(tempdir.path(), Some(&compact)).unwrap();
    assert_eq!(resolved.as_deref(), Some(&full));
}

#[test]
fn compact_exl3_config_rejects_missing_or_mismatched_external_config() {
    let tempdir = tempfile::tempdir().unwrap();
    let compact = serde_json::json!({
        "bits": 2.0,
        "codebook": "mcg",
        "quant_method": "exl3"
    });
    assert!(
        resolve_model_quantization_config(tempdir.path(), Some(&compact))
            .unwrap_err()
            .to_string()
            .contains("requires quantize_config.json or quantization_config.json")
    );

    let mismatched = serde_json::json!({
        "bits": 3.0,
        "codebook": "mcg",
        "quant_method": "exl3",
        "tensor_storage": {"proj": {"quant_format": "exl3"}}
    });
    std::fs::write(
        tempdir.path().join("quantization_config.json"),
        serde_json::to_vec(&mismatched).unwrap(),
    )
    .unwrap();
    assert!(
        resolve_model_quantization_config(tempdir.path(), Some(&compact))
            .unwrap_err()
            .to_string()
            .contains("differs from compact")
    );
}

#[test]
fn tokenizer_is_reused_after_the_snapshot_file_changes() {
    let tempdir = tempfile::tempdir().unwrap();
    let tokenizer_path = tempdir.path().join("tokenizer.json");
    std::fs::write(
        &tokenizer_path,
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();

    let first = encode_tokenizer_text(tempdir.path(), "hello", false).unwrap();
    assert_eq!(first.token_ids, vec![1]);
    std::fs::write(&tokenizer_path, b"not valid tokenizer json").unwrap();
    let cached = encode_tokenizer_text(tempdir.path(), "hello", false).unwrap();
    assert_eq!(cached.token_ids, vec![1]);
}

#[test]
fn streaming_tokenizer_buffers_split_utf8_scalars() {
    let tempdir = tempfile::tempdir().unwrap();
    std::fs::write(
        tempdir.path().join("tokenizer.json"),
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"WordLevel","vocab":{"[UNK]":0,"ðŁ":1,"¦":2,"ľ":3," ok":4},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();

    let mut decoder = streaming_token_decoder(tempdir.path(), false).unwrap();
    assert_eq!(decoder.step(1).unwrap(), None);
    assert_eq!(decoder.step(2).unwrap(), None);
    assert_eq!(decoder.step(3).unwrap().as_deref(), Some("🦜"));
}

#[test]
fn reads_single_file_safetensors_metadata_with_absolute_offsets() {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("model.safetensors");
    let header = serde_json::json!({
        "z": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
        "a": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]},
        "__metadata__": {"format": "pt"}
    });
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }
    let data_start = 8 + header_bytes.len();
    let mut file = File::create(&path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&[0_u8; 16]).unwrap();

    let metadata = read_safetensors_metadata(&path).unwrap();
    assert_eq!(metadata.len(), 2);
    assert_eq!(metadata[0].name, "a");
    assert_eq!(metadata[0].dtype, DType::F32);
    assert_eq!(metadata[0].shape, [1]);
    assert_eq!(metadata[0].byte_offset, (data_start + 12) as u64);
    assert_eq!(metadata[0].byte_length, 4);
    assert_eq!(metadata[1].name, "z");
    assert_eq!(metadata[1].dtype, DType::Bf16);
    assert_eq!(metadata[1].shape, [2, 3]);
    assert_eq!(metadata[1].byte_offset, data_start as u64);
    assert_eq!(metadata[1].byte_length, 12);
}

#[test]
fn classifier_identifies_routed_expert_scale() {
    let facts = ModelFacts::default();
    let name = "model.layers.3.mlp.experts.17.down_proj.weight_scale";
    assert_eq!(
        classify_tensor(name, Some(3), Some(17), true, &facts),
        TensorRole::RoutedExpert
    );
    assert!(is_quantization_tensor(name));
}

#[test]
fn classifier_identifies_native_ds4_expert_scale() {
    let facts = ModelFacts::default();
    let name = "layers.0.ffn.experts.17.w1.scale";
    assert_eq!(
        classify_tensor(name, Some(0), Some(17), true, &facts),
        TensorRole::RoutedExpert
    );
    assert!(is_quantization_tensor(name));
    assert!(is_quantization_tensor("layers.2.attn.indexer.wq_b.scale"));
    assert!(is_quantization_tensor("mtp.0.main_proj.scale"));
}

#[test]
fn classifier_treats_exl3_trellis_as_weight_and_rotations_as_metadata() {
    let facts = ModelFacts::default();
    for name in [
        "model.layers.3.mlp.experts.17.gate_proj.trellis",
        "mtp.2.mlp.experts.17.down_proj.trellis",
    ] {
        assert_eq!(
            classify_tensor(name, Some(3), Some(17), false, &facts),
            TensorRole::RoutedExpert
        );
        assert!(!is_quantization_tensor(name));
    }
    for suffix in ["suh", "svh", "mcg"] {
        assert!(is_quantization_tensor(&format!(
            "model.layers.3.mlp.experts.17.gate_proj.{suffix}"
        )));
    }
}

#[test]
fn native_attention_specs_match_flash_projection_and_compression_contract() {
    let facts = ModelFacts::default();
    let specs = native_deepseek_v4_attention_tensor_specs(&facts).unwrap();

    assert_eq!(specs.len(), 909);
    let expected = [
        (
            "layers.2.attn.wq_b.weight",
            DType::F8E4M3,
            vec![32_768, 1_024],
            NativeDeepseekV4AttentionTensorFamily::Main,
        ),
        (
            "layers.2.attn.compressor.ape",
            DType::F32,
            vec![4, 1_024],
            NativeDeepseekV4AttentionTensorFamily::Compressor,
        ),
        (
            "layers.2.attn.indexer.compressor.wkv.weight",
            DType::Bf16,
            vec![256, 4_096],
            NativeDeepseekV4AttentionTensorFamily::Indexer,
        ),
        (
            "layers.2.attn.indexer.wq_b.scale",
            DType::F8E8M0,
            vec![64, 8],
            NativeDeepseekV4AttentionTensorFamily::Indexer,
        ),
        (
            "layers.3.attn.compressor.ape",
            DType::F32,
            vec![128, 512],
            NativeDeepseekV4AttentionTensorFamily::Compressor,
        ),
        (
            "mtp.2.attn.wo_b.weight",
            DType::F8E4M3,
            vec![4_096, 8_192],
            NativeDeepseekV4AttentionTensorFamily::Main,
        ),
    ];
    for (name, dtype, shape, family) in expected {
        let spec = specs.iter().find(|spec| spec.name == name).unwrap();
        assert_eq!(spec.dtype, dtype, "unexpected dtype for {name}");
        assert_eq!(spec.shape, shape, "unexpected shape for {name}");
        assert_eq!(spec.family, family, "unexpected family for {name}");
    }
    assert!(!specs.iter().any(|spec| {
        spec.name.starts_with("mtp.") && spec.family != NativeDeepseekV4AttentionTensorFamily::Main
    }));
}

#[test]
fn native_attention_catalog_validates_all_target_and_dspark_headers() {
    let catalog = native_attention_test_catalog();

    let summary = validate_native_deepseek_v4_attention_catalog(&catalog).unwrap();

    assert_eq!(summary.target_blocks, 43);
    assert_eq!(summary.dspark_blocks, 3);
    assert_eq!(summary.sliding_blocks, 5);
    assert_eq!(summary.c4_blocks, 21);
    assert_eq!(summary.c128_blocks, 20);
    assert_eq!(summary.main_tensors, 598);
    assert_eq!(summary.compressor_tensors, 164);
    assert_eq!(summary.indexer_tensors, 147);
    assert_eq!(summary.tensor_bytes, 5_721_447_936);
}

#[test]
fn native_attention_catalog_rejects_header_drift() {
    let mut wrong_shape = native_attention_test_catalog();
    let tensor = wrong_shape
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "layers.2.attn.indexer.wq_b.weight")
        .unwrap();
    tensor.shape[0] -= 1;
    let error = validate_native_deepseek_v4_attention_catalog(&wrong_shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("has shape"));

    let mut missing = native_attention_test_catalog();
    missing
        .tensors
        .retain(|tensor| tensor.name != "layers.3.attn.compressor.ape");
    let error = validate_native_deepseek_v4_attention_catalog(&missing)
        .unwrap_err()
        .to_string();
    assert!(error.contains("tensor set mismatch"));
    assert!(error.contains("layers.3.attn.compressor.ape"));

    let mut wrong_role = native_attention_test_catalog();
    wrong_role
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "mtp.0.attn.wq_a.weight")
        .unwrap()
        .role = TensorRole::Attention;
    let error = validate_native_deepseek_v4_attention_catalog(&wrong_role)
        .unwrap_err()
        .to_string();
    assert!(error.contains("inconsistent attention identity"));
}

fn native_attention_test_catalog() -> TensorCatalog {
    let facts = ModelFacts::default();
    let tensors = native_deepseek_v4_attention_tensor_specs(&facts)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(index, spec)| TensorInfo {
            byte_length: spec.byte_length().unwrap(),
            name: spec.name,
            file: "model.safetensors".to_owned(),
            dtype: spec.dtype,
            shape: spec.shape,
            byte_offset: index as u64 * 4_096,
            role: spec.role,
            layer_id: Some(spec.logical_layer_id as u32),
            expert_id: None,
            is_quantization_metadata: spec.is_quantization_metadata,
        })
        .collect();
    TensorCatalog {
        model_id: facts.model_id.clone(),
        snapshot_path: "/tmp/native-attention-test".to_owned(),
        facts,
        tensors,
    }
}

fn native_fp4_test_tensor(
    facts: &ModelFacts,
    layer_id: usize,
    expert_id: usize,
    stem: &str,
    suffix: &str,
) -> TensorInfo {
    let (rows, columns) = match stem {
        "w1" | "w3" => (facts.moe_intermediate_size, facts.hidden_size),
        "w2" => (facts.hidden_size, facts.moe_intermediate_size),
        _ => panic!("unknown test projection {stem}"),
    };
    let (dtype, width, is_quantization_metadata) = match suffix {
        "weight" => (DType::I8, columns / 2, false),
        "scale" => (DType::F8E8M0, columns / 32, true),
        _ => panic!("unknown test suffix {suffix}"),
    };
    let name = format!("layers.{layer_id}.ffn.experts.{expert_id}.{stem}.{suffix}");
    TensorInfo {
        name,
        file: "model.safetensors".to_owned(),
        dtype,
        shape: vec![rows, width],
        byte_offset: 0,
        byte_length: (rows * width) as u64,
        role: TensorRole::RoutedExpert,
        layer_id: Some(layer_id as u32),
        expert_id: Some(expert_id as u32),
        is_quantization_metadata,
    }
}

#[test]
fn native_fp4_contract_maps_checkpoint_w3_then_w1_to_sparkinfer_w13() {
    let mut facts = ModelFacts::default();
    facts.num_hidden_layers = 1;
    facts.dspark_target_layer_ids.clear();
    facts.routed_experts = 1;
    facts.hidden_size = 64;
    facts.moe_intermediate_size = 32;
    let mut tensors = Vec::new();
    for stem in ["w1", "w2", "w3"] {
        for suffix in ["weight", "scale"] {
            tensors.push(native_fp4_test_tensor(&facts, 0, 0, stem, suffix));
        }
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let catalog = TensorCatalog {
        model_id: "deepseek-ai/test-native-fp4".to_owned(),
        snapshot_path: "/tmp/test-native-fp4".to_owned(),
        facts,
        tensors,
    };

    let summary = validate_native_fp4_expert_catalog(&catalog).unwrap();
    assert_eq!(summary.transformer_blocks, 1);
    assert_eq!(summary.experts_per_block, 1);
    assert_eq!(summary.expert_tensors, 6);
    assert_eq!(summary.packed_weight_bytes, 3_072);
    assert_eq!(summary.e8m0_scale_bytes, 192);

    let expert = super::native_fp4_expert(&catalog, 0, 0).unwrap();
    let w13 = expert.sparkinfer_w13();
    assert_eq!(w13[0].kind, NativeFp4ProjectionKind::Up);
    assert_eq!(w13[0].weight.name, "layers.0.ffn.experts.0.w3.weight");
    assert_eq!(w13[1].kind, NativeFp4ProjectionKind::Gate);
    assert_eq!(w13[1].weight.name, "layers.0.ffn.experts.0.w1.weight");
    assert_eq!(expert.down.logical_rows, 64);
    assert_eq!(expert.down.logical_columns, 32);
}

fn native_fp4_tp_test_catalog(hidden_size: usize, intermediate_size: usize) -> TensorCatalog {
    let mut facts = ModelFacts::default();
    facts.num_hidden_layers = 1;
    facts.dspark_target_layer_ids.clear();
    facts.routed_experts = 1;
    facts.hidden_size = hidden_size;
    facts.moe_intermediate_size = intermediate_size;
    let mut tensors = Vec::new();
    for stem in ["w1", "w2", "w3"] {
        for suffix in ["weight", "scale"] {
            tensors.push(native_fp4_test_tensor(&facts, 0, 0, stem, suffix));
        }
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    TensorCatalog {
        model_id: "deepseek-ai/test-native-fp4-tp".to_owned(),
        snapshot_path: "/tmp/test-native-fp4-tp".to_owned(),
        facts,
        tensors,
    }
}

#[test]
fn native_fp4_flash_tp4_windows_match_gpu_oracle_memory_geometry() {
    let catalog = native_fp4_tp_test_catalog(4096, 2048);
    let expert = super::native_fp4_expert(&catalog, 0, 0).unwrap();
    for rank in 0..4 {
        let shard = expert.tp_shard(rank).unwrap();
        assert_eq!(shard.rank, rank);
        assert_eq!(shard.world_size, 4);
        assert_eq!(shard.gate.local_intermediate_size, 512);
        assert_eq!(shard.up.local_intermediate_size, 512);
        assert_eq!(shard.down.local_intermediate_size, 512);
        assert_eq!(shard.gate.weight.row_start, rank * 512);
        assert_eq!(shard.gate.weight.row_count, 512);
        assert_eq!(shard.gate.weight.column_start_bytes, 0);
        assert_eq!(shard.gate.weight.row_width_bytes, 2048);
        assert!(shard.gate.weight.is_contiguous());
        assert_eq!(shard.down.weight.row_start, 0);
        assert_eq!(shard.down.weight.row_count, 4096);
        assert_eq!(shard.down.weight.column_start_bytes, rank * 256);
        assert_eq!(shard.down.weight.row_width_bytes, 256);
        assert!(!shard.down.weight.is_contiguous());
        assert_eq!(shard.down.scale.column_start_bytes, rank * 16);
        assert_eq!(shard.down.scale.row_width_bytes, 16);
        assert_eq!(shard.source_bytes().unwrap(), 3_342_336);
    }

    let rank_2 = expert.tp_shard(2).unwrap();
    assert_eq!(
        rank_2.gate.weight.source_offset_for_row(0).unwrap(),
        2_097_152
    );
    assert_eq!(rank_2.down.weight.source_offset_for_row(0).unwrap(), 512);
    assert_eq!(rank_2.down.weight.source_offset_for_row(1).unwrap(), 1_536);
    assert_eq!(rank_2.down.scale.source_offset_for_row(1).unwrap(), 96);
}

#[test]
fn native_fp4_pro_tp4_windows_preserve_equal_rank_demand() {
    let catalog = native_fp4_tp_test_catalog(7168, 3072);
    let expert = super::native_fp4_expert(&catalog, 0, 0).unwrap();
    let source_bytes = (0..4)
        .map(|rank| {
            let shard = expert.tp_shard(rank).unwrap();
            assert_eq!(shard.gate.local_intermediate_size, 768);
            shard.source_bytes().unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(source_bytes, vec![8_773_632; 4]);
    assert!(expert.tp_shard(4).is_err());
}

#[test]
fn native_fp4_contract_fails_closed_on_legacy_nvfp4_metadata() {
    let mut facts = ModelFacts::default();
    facts.num_hidden_layers = 1;
    facts.dspark_target_layer_ids.clear();
    facts.routed_experts = 1;
    facts.hidden_size = 64;
    facts.moe_intermediate_size = 32;
    let mut tensors = Vec::new();
    for stem in ["w1", "w2", "w3"] {
        for suffix in ["weight", "scale"] {
            tensors.push(native_fp4_test_tensor(&facts, 0, 0, stem, suffix));
        }
    }
    tensors
        .iter_mut()
        .find(|tensor| tensor.name.ends_with("w1.scale"))
        .unwrap()
        .dtype = DType::F8E4M3;
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let catalog = TensorCatalog {
        model_id: "deepseek-ai/test-native-fp4".to_owned(),
        snapshot_path: "/tmp/test-native-fp4".to_owned(),
        facts,
        tensors,
    };

    let error = validate_native_fp4_expert_catalog(&catalog)
        .unwrap_err()
        .to_string();
    assert!(error.contains("validating native FP4 layer 0 expert 0"));
}

#[test]
fn classifier_maps_dspark_experts_and_envelope_separately() {
    let facts = ModelFacts::default();
    let expert = "mtp.2.ffn.experts.17.w2.weight";
    let envelope = "mtp.2.main_proj.weight";
    let layer_id = Some((DS4_FLASH_NUM_HIDDEN_LAYERS + 2) as u32);
    assert_eq!(tensor_layer_id(expert, &facts), layer_id);
    assert_eq!(
        classify_tensor(expert, layer_id, Some(17), false, &facts),
        TensorRole::RoutedExpert
    );
    assert_eq!(
        classify_tensor(envelope, layer_id, None, false, &facts),
        TensorRole::Dspark
    );
}

#[test]
fn classifier_identifies_ds4_coordinator_tensor_families() {
    let facts = ModelFacts::default();
    let cases = [
        (
            "layers.3.attn.compressor.wk.weight",
            TensorRole::AttentionCompressor,
        ),
        (
            "layers.3.attn.indexer.wq.weight",
            TensorRole::AttentionIndexer,
        ),
        (
            "layers.3.hc_attn_output.weight",
            TensorRole::HyperConnection,
        ),
        ("layers.3.attn_norm.weight", TensorRole::Norm),
        ("norm.weight", TensorRole::Norm),
    ];
    for (name, expected) in cases {
        assert_eq!(
            classify_tensor(name, Some(3), None, false, &facts),
            expected,
            "unexpected role for {name}"
        );
    }
}

#[test]
fn flash_model_facts_are_read_from_deepseek_v4_config() {
    let tempdir = tempfile::tempdir().unwrap();
    let mut config = serde_json::json!({
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_hash_layers": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "vocab_size": 129280,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 512,
        "sliding_window": 128,
        "max_position_embeddings": 1048576,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0],
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "swiglu_limit": 10.0,
        "expert_dtype": "fp4",
        "dspark_block_size": 5,
        "dspark_markov_rank": 256,
        "dspark_target_layer_ids": [40, 41, 42],
        "quantization_config": {"quant_method": "fp8"}
    });
    add_native_attention_geometry(&mut config);
    std::fs::write(
        tempdir.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();

    let facts = read_model_facts("deepseek-ai/test-flash", tempdir.path()).unwrap();
    assert_eq!(facts.variant, ModelVariant::Flash);
    assert_eq!(facts.total_transformer_blocks(), 46);
    assert_eq!(facts.attention_kind(0), Some(AttentionKind::Sliding));
    assert_eq!(facts.rope_theta, 10_000.0);
    assert_eq!(facts.compress_rope_theta, 160_000.0);
    assert_eq!(facts.rope_scaling_factor, 16.0);
    assert_eq!(facts.original_max_position_embeddings, 65_536);
    assert_eq!(facts.rope_beta_fast, 32.0);
    assert_eq!(facts.rope_beta_slow, 1.0);
    assert_eq!(facts.hyper_connection_eps, 1.0e-6);
    assert_eq!(facts.dspark_noise_token_id, 128_799);
    assert_eq!(
        facts.attention_kind(2),
        Some(AttentionKind::CompressedSparse)
    );
    assert_eq!(
        facts.attention_kind(3),
        Some(AttentionKind::HeavilyCompressed)
    );
    assert_eq!(
        facts.quantization_recipe,
        "deepseek_v4_native_fp4_fp8_mixed_v1"
    );
}

#[test]
fn pro_model_facts_are_inferred_from_dimensions_not_repository_name() {
    let tempdir = tempfile::tempdir().unwrap();
    let mut config = serde_json::json!({
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_hash_layers": 3,
        "n_routed_experts": 384,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 3072,
        "n_shared_experts": 1,
        "vocab_size": 129280,
        "num_attention_heads": 128,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1536,
        "o_lora_rank": 1024,
        "o_groups": 16,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 1024,
        "sliding_window": 128,
        "max_position_embeddings": 1048576,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "compress_ratios": DS4_PRO_COMPRESS_RATIOS.to_vec(),
        "routed_scaling_factor": 2.5,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "swiglu_limit": 10.0,
        "expert_dtype": "fp4",
        "dspark_block_size": 5,
        "dspark_markov_rank": 512,
        "dspark_target_layer_ids": [58, 59, 60],
        "quantization_config": ds4rt_exl3_quantization_config()
    });
    add_native_attention_geometry(&mut config);
    std::fs::write(
        tempdir.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();

    let facts = read_model_facts("future-org/future-pro-release", tempdir.path()).unwrap();
    assert_eq!(facts.variant, ModelVariant::Pro);
    assert_eq!(facts.hidden_size, 7168);
    assert_eq!(facts.routed_experts, 384);
    assert_eq!(facts.index_top_k, 1024);
    assert_eq!(facts.dspark_markov_rank, 512);
    assert_eq!(
        facts.quantization_recipe,
        "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route"
    );
    let specs = native_deepseek_v4_attention_tensor_specs(&facts).unwrap();
    for (name, shape) in [
        ("layers.0.attn.wq_b.weight", vec![65_536, 1_536]),
        ("layers.0.attn.wq_b.scale", vec![512, 12]),
        ("layers.0.attn.wo_a.weight", vec![16_384, 4_096]),
        ("layers.0.attn.wo_b.weight", vec![7_168, 16_384]),
        ("layers.0.attn.wo_b.scale", vec![56, 128]),
        ("mtp.2.attn.wq_a.weight", vec![1_536, 7_168]),
    ] {
        assert_eq!(
            specs.iter().find(|spec| spec.name == name).unwrap().shape,
            shape,
            "unexpected Pro attention shape for {name}"
        );
    }
}

fn add_native_attention_geometry(config: &mut serde_json::Value) {
    let object = config.as_object_mut().unwrap();
    object.insert("rope_theta".to_owned(), 10_000.0.into());
    object.insert("compress_rope_theta".to_owned(), 160_000.0.into());
    object.insert(
        "rope_scaling".to_owned(),
        serde_json::json!({
            "type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32.0,
            "beta_slow": 1.0
        }),
    );
    object.insert("hc_eps".to_owned(), 1.0e-6.into());
    object.insert("dspark_noise_token_id".to_owned(), 128_799.into());
}

fn ds4rt_exl3_quantization_config() -> serde_json::Value {
    serde_json::json!({
        "quant_method": "exl3",
        "version": "1.3.0",
        "bits": 2.0,
        "codebook": "mcg",
        "calibration": {
            "method": "layerwise_native_natural_routes",
            "device": "cuda:0",
            "rows": 43691,
            "seed": 20260806,
            "hessian": "per_expert_natural_route_gate_squared_covariance",
            "distribution": "checkpoint_bound_native_expert_inputs",
            "activation_corpus_sha256": "eacedc6122f284326f734522f4b1a5faa90e559b00fe0a8307bb29be1a7b2660",
            "activation_base_layers": 43,
            "natural_routing": true,
            "forced_expert_activation": false,
            "route_gate_weighting": "squared_unit_rms",
            "minimum_natural_routes_per_expert": 1024,
            "route_replay_report_sha256": "7da2a2608ad24fc9b0c7cb7a15d57232d189511413ad92a894087645c6f04136",
            "mtp_calibration": "analytic_identity_pilot_only",
            "activation": "silu",
            "swiglu_limit": 10.0,
            "gate_clamp": [null, 10.0],
            "up_clamp": [-10.0, 10.0]
        },
        "ds4rt": {
            "schema": "ds4rt.exl3.expert-trellis",
            "schema_version": 1,
            "recipe": "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route",
            "calibrated": true,
            "scope": "routed_experts",
            "source_format": "fp4_e8m0_k32",
            "tensor_format": "exllamav3_trellis_mcg",
            "expert_tp_world_size": 4,
            "quantizer_source": {
                "repository": "https://github.com/turboderp-org/exllamav3.git",
                "revision": "0b9745c526a13d5b30f1b58a864efc1932d3d9eb",
                "source_tree_sha256": "8c2f94e3a7335e304c47dad85a5254428950c324acf74263fb7bff59fe057dec"
            }
        }
    })
}

#[test]
fn classifier_keeps_shared_expert_on_coordinator() {
    let facts = ModelFacts::default();
    let name = "model.layers.3.mlp.shared_experts.down_proj.weight";
    assert_eq!(
        classify_tensor(name, Some(3), None, false, &facts),
        TensorRole::SharedExpert
    );
}

#[test]
fn classifier_assigns_mtp_routed_experts_to_expert_placement() {
    let facts = ModelFacts::default();
    let name = "model.layers.78.mlp.experts.17.gate_proj.weight";
    assert_eq!(
        classify_tensor(
            name,
            Some(GLM52_MTP_LAYER_ID as u32),
            Some(17),
            false,
            &facts,
        ),
        TensorRole::RoutedExpert
    );
}

#[test]
fn classifier_keeps_mtp_non_expert_tensors_on_the_mtp_role() {
    let mut facts = ModelFacts::default();
    facts.model_type = "glm4_moe_lite".to_owned();
    let name = "model.layers.78.eh_proj.weight";
    assert_eq!(
        classify_tensor(name, Some(GLM52_MTP_LAYER_ID as u32), None, false, &facts,),
        TensorRole::Mtp
    );
}

#[test]
fn load_plan_places_mtp_experts_on_sparks_and_mtp_envelope_on_coordinator() {
    let expert_hosts = vec!["ostrich".to_owned(), "dodo".to_owned()];
    let expert_name = "model.layers.78.mlp.experts.17.gate_proj.weight";
    let envelope_name = "model.layers.78.eh_proj.weight";
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: "/tmp/test-model".to_owned(),
        facts: ModelFacts::default(),
        tensors: vec![
            TensorInfo {
                name: expert_name.to_owned(),
                file: "model.safetensors".to_owned(),
                dtype: DType::U8,
                shape: vec![1],
                byte_offset: 0,
                byte_length: 1,
                role: TensorRole::RoutedExpert,
                layer_id: Some(GLM52_MTP_LAYER_ID as u32),
                expert_id: Some(17),
                is_quantization_metadata: false,
            },
            TensorInfo {
                name: envelope_name.to_owned(),
                file: "model.safetensors".to_owned(),
                dtype: DType::Bf16,
                shape: vec![1],
                byte_offset: 1,
                byte_length: 2,
                role: TensorRole::Mtp,
                layer_id: Some(GLM52_MTP_LAYER_ID as u32),
                expert_id: None,
                is_quantization_metadata: false,
            },
        ],
    };

    let plan = build_load_plan(&catalog, PlacementPolicy::Range, expert_hosts.clone()).unwrap();
    let expert = plan
        .assignments
        .iter()
        .find(|assignment| assignment.tensor_name == expert_name)
        .unwrap();
    let envelope = plan
        .assignments
        .iter()
        .find(|assignment| assignment.tensor_name == envelope_name)
        .unwrap();

    assert_eq!(
        expert.owner,
        owner_for_expert(
            GLM52_MTP_LAYER_ID,
            17,
            catalog.facts.routed_experts,
            &expert_hosts,
            PlacementPolicy::Range,
        )
        .unwrap()
    );
    assert_eq!(envelope.owner, COORDINATOR_HOST);
}

#[test]
fn load_tensor_bytes_reads_exact_range() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("shard.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard.write_all(&[9, 8, 7, 1, 2, 3, 4, 5]).unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "tensor".to_owned(),
            file: "shard.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![4],
            byte_offset: 3,
            byte_length: 4,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };

    let loaded = load_tensor_bytes(&catalog, "tensor").unwrap();
    assert_eq!(loaded.bytes, vec![1, 2, 3, 4]);
    assert_eq!(loaded.sha256, "");
    let summary = loaded.summary();
    assert_eq!(summary.bytes_requested, 4);
    assert_eq!(summary.bytes_read, 4);
    assert_eq!(summary.tensor_name, "tensor");
    assert_eq!(summary.sha256, "");

    let hashed =
        load_tensor_bytes_with_options(&catalog, "tensor", TensorLoadOptions::verify_hashes())
            .unwrap();
    assert_eq!(hashed.bytes, vec![1, 2, 3, 4]);
    assert_eq!(hashed.sha256.len(), 64);
    assert_ne!(hashed.sha256, loaded.sha256);
}

#[test]
fn read_tensor_bytes_into_uses_caller_buffer_prefix() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("shard.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard.write_all(&[9, 8, 7, 1, 2, 3, 4, 5]).unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "tensor".to_owned(),
            file: "shard.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![4],
            byte_offset: 3,
            byte_length: 4,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0xee; 8];

    let summary = read_tensor_bytes_into(&catalog, "tensor", &mut dst).unwrap();

    assert_eq!(&dst[..4], &[1, 2, 3, 4]);
    assert_eq!(&dst[4..], &[0xee, 0xee, 0xee, 0xee]);
    assert_eq!(summary.bytes_requested, 4);
    assert_eq!(summary.bytes_read, 4);
    assert_eq!(summary.byte_offset, 3);
    assert_eq!(summary.sha256, "");

    let hashed = read_tensor_bytes_into_with_options(
        &catalog,
        "tensor",
        &mut dst,
        TensorLoadOptions::verify_hashes(),
    )
    .unwrap();
    assert_eq!(hashed.sha256.len(), 64);
}

#[test]
fn read_tensor_bytes_into_rejects_small_destination() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("shard.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard.write_all(&[1, 2, 3, 4]).unwrap();
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "tensor".to_owned(),
            file: "shard.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![4],
            byte_offset: 0,
            byte_length: 4,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0_u8; 3];

    let err = read_tensor_bytes_into(&catalog, "tensor", &mut dst)
        .unwrap_err()
        .to_string();

    assert!(err.contains("destination buffer for tensor tensor has 3 bytes, needs 4"));
}

#[test]
fn load_tensor_rows_reads_exact_2d_window() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard
        .write_all(&[99, 98, 97, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        .unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 3,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };

    let rows = load_tensor_rows(&catalog, "matrix", 1, 2).unwrap();
    assert_eq!(rows.bytes, vec![20, 21, 22, 23, 30, 31, 32, 33]);
    assert_eq!(rows.sha256, "");
    let summary = rows.summary();
    assert_eq!(summary.start_row, 1);
    assert_eq!(summary.row_count, 2);
    assert_eq!(summary.row_width, 4);
    assert_eq!(summary.byte_offset, 7);
    assert_eq!(summary.bytes_read, 8);
    assert_eq!(summary.sha256, "");

    let hashed =
        load_tensor_rows_with_options(&catalog, "matrix", 1, 2, TensorLoadOptions::verify_hashes())
            .unwrap();
    assert_eq!(hashed.bytes, rows.bytes);
    assert_eq!(hashed.sha256.len(), 64);
    assert_ne!(hashed.sha256, rows.sha256);
}

#[test]
fn read_tensor_rows_into_uses_caller_buffer_prefix() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard
        .write_all(&[99, 98, 97, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        .unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 3,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0xcc; 12];

    let summary = read_tensor_rows_into(&catalog, "matrix", 1, 2, &mut dst).unwrap();

    assert_eq!(&dst[..8], &[20, 21, 22, 23, 30, 31, 32, 33]);
    assert_eq!(&dst[8..], &[0xcc, 0xcc, 0xcc, 0xcc]);
    assert_eq!(summary.start_row, 1);
    assert_eq!(summary.row_count, 2);
    assert_eq!(summary.row_width, 4);
    assert_eq!(summary.byte_offset, 7);
    assert_eq!(summary.bytes_read, 8);
    assert_eq!(summary.sha256, "");

    let hashed = read_tensor_rows_into_with_options(
        &catalog,
        "matrix",
        1,
        2,
        &mut dst,
        TensorLoadOptions::verify_hashes(),
    )
    .unwrap();
    assert_eq!(hashed.sha256.len(), 64);
}

#[test]
fn read_tensor_rows_into_rejects_small_destination() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    File::create(&shard_path)
        .unwrap()
        .write_all(&[0; 12])
        .unwrap();
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 0,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0_u8; 7];

    let err = read_tensor_rows_into(&catalog, "matrix", 1, 2, &mut dst)
        .unwrap_err()
        .to_string();

    assert!(err.contains("destination buffer for tensor matrix rows 1..3 has 7 bytes, needs 8"));
}

#[test]
fn read_tensor_row_prefix_into_compacts_leading_columns() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard
        .write_all(&[99, 98, 97, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        .unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 3,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0xcc; 8];

    let summary = read_tensor_row_prefix_into(&catalog, "matrix", 1, 2, 2, &mut dst).unwrap();

    assert_eq!(&dst[..4], &[20, 21, 30, 31]);
    assert_eq!(&dst[4..], &[0xcc, 0xcc, 0xcc, 0xcc]);
    assert_eq!(summary.start_row, 1);
    assert_eq!(summary.row_count, 2);
    assert_eq!(summary.row_width, 2);
    assert_eq!(summary.byte_offset, 7);
    assert_eq!(summary.bytes_read, 4);
    assert_eq!(summary.sha256, "");

    let hashed = read_tensor_row_prefix_into_with_options(
        &catalog,
        "matrix",
        1,
        2,
        2,
        &mut dst,
        TensorLoadOptions::verify_hashes(),
    )
    .unwrap();
    assert_eq!(hashed.sha256.len(), 64);
}

#[test]
fn read_tensor_row_window_into_compacts_middle_columns() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    File::create(&shard_path)
        .unwrap()
        .write_all(&[99, 98, 97, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        .unwrap();
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 3,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0xcc; 8];

    let summary = read_tensor_row_window_into(&catalog, "matrix", 1, 2, 1, 2, &mut dst).unwrap();

    assert_eq!(&dst[..4], &[21, 22, 31, 32]);
    assert_eq!(&dst[4..], &[0xcc, 0xcc, 0xcc, 0xcc]);
    assert_eq!(summary.start_row, 1);
    assert_eq!(summary.row_count, 2);
    assert_eq!(summary.row_width, 2);
    assert_eq!(summary.byte_offset, 8);
    assert_eq!(summary.bytes_read, 4);
}

#[test]
fn read_tensor_row_prefix_into_rejects_invalid_prefix_width() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    File::create(&shard_path)
        .unwrap()
        .write_all(&[0; 12])
        .unwrap();
    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 0,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };
    let mut dst = vec![0_u8; 16];

    let err = read_tensor_row_prefix_into(&catalog, "matrix", 0, 1, 5, &mut dst)
        .unwrap_err()
        .to_string();

    assert!(err.contains("row prefix width 5 exceeds tensor matrix row width 4"));
}

#[test]
fn load_tensor_rows_rejects_out_of_bounds_window() {
    let tempdir = tempfile::tempdir().unwrap();
    let shard_path = tempdir.path().join("rows.safetensors");
    File::create(&shard_path)
        .unwrap()
        .write_all(&[0; 12])
        .unwrap();

    let catalog = TensorCatalog {
        model_id: "test/model".to_owned(),
        snapshot_path: tempdir.path().display().to_string(),
        facts: ModelFacts::default(),
        tensors: vec![TensorInfo {
            name: "matrix".to_owned(),
            file: "rows.safetensors".to_owned(),
            dtype: DType::U8,
            shape: vec![3, 4],
            byte_offset: 0,
            byte_length: 12,
            role: TensorRole::Other,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        }],
    };

    let err = load_tensor_rows(&catalog, "matrix", 2, 2)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds tensor matrix row count 3"));
}
