use std::fs;

use ds41rt_core::ModelFacts;

use super::coordinator_kernels::cuda_reference_kernels_test_override;
use super::entry::{real_full_info_from_report, real_full_info_from_startup};
use super::preflight::{
    coordinator_resident_preload_requirement, real_ds4_full_preflight_report,
    real_full_kv_cache_config_for_model,
};
use super::residency::real_full_coordinator_resident_preload_plan;
use super::sampling::RealFullLmHeadSamplingOptions;
use super::scheduler::{real_full_scheduler_execution_for_shape, RealFullSchedulerExecutionShape};

mod assertions;
pub(super) mod fixture;

use assertions::assert_real_full_preflight_report;
use fixture::{clear_real_full_probe_env, coordinator_args, full_catalog};

#[test]
fn real_full_preflight_reports_missing_execution_components() {
    let _probe_env = clear_real_full_probe_env();
    let _cuda_reference_override = cuda_reference_kernels_test_override(false);
    let catalog = full_catalog();
    let args = coordinator_args();
    let report = real_ds4_full_preflight_report(&args, "reports/model_catalog.json", &catalog)
        .expect("real-full preflight report builds");

    assert_real_full_preflight_report(&report);

    let info = real_full_info_from_report(&report);
    let terminal_lm_head_sample = &report.scheduler_execution_dry_run.terminal_lm_head_sample;
    assert_eq!(
        info.snapshot_path.as_deref(),
        Some(report.snapshot_path.as_str())
    );
    assert_eq!(
        info.scheduler_full_context_device_attention_complete,
        report
            .scheduler_execution_dry_run
            .full_context_device_attention_complete
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_sample_status,
        terminal_lm_head_sample.status
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_sample_passed,
        terminal_lm_head_sample.passed
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_uses_final_decode_device_hidden,
        terminal_lm_head_sample.uses_final_decode_device_hidden
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_covers_full_vocabulary,
        terminal_lm_head_sample.covers_full_vocabulary
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_logits_evaluated,
        terminal_lm_head_sample.logits_evaluated
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_vocab_size,
        terminal_lm_head_sample.vocab_size
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_top_token_id,
        terminal_lm_head_sample.top_token_id
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_sampled_token_id,
        terminal_lm_head_sample.sampled_token_id
    );
    assert_eq!(info.scheduler_terminal_lm_head_sampled_text, None);
    assert_eq!(
        info.scheduler_terminal_lm_head_sample_top_k,
        terminal_lm_head_sample.sample_top_k
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_sample_top_p,
        terminal_lm_head_sample.sample_top_p
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_argmax_backend.as_deref(),
        terminal_lm_head_sample.argmax_kernel_backend
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_sampler_backend.as_deref(),
        terminal_lm_head_sample.sampler_kernel_backend
    );
    assert_eq!(
        info.scheduler_terminal_lm_head_blocker.as_deref(),
        terminal_lm_head_sample.blocker.as_deref()
    );
}

#[test]
fn deepseek_v4_serving_requires_fp8_physical_kv() {
    let mut args = coordinator_args();
    args.kv_cache_dtype = "fp8".to_owned();
    let config = real_full_kv_cache_config_for_model(&args, &ModelFacts::default())
        .expect("FP8 physical KV should be accepted");
    // The transactional mirror remains BF16; the physical cache is FP8.
    assert_eq!(config.dtype_label(), "bf16");
    for dtype in ["bf16", "nvfp4"] {
        args.kv_cache_dtype = dtype.to_owned();
        assert!(real_full_kv_cache_config_for_model(&args, &ModelFacts::default()).is_err());
    }
}

#[test]
fn real_full_kv_rejects_non_deepseek_catalogs() {
    let args = coordinator_args();
    let mut facts = ModelFacts::default();
    facts.model_type = "glm".to_owned();

    let error = real_full_kv_cache_config_for_model(&args, &facts)
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires model_type deepseek_v4"));
}

#[test]
fn real_full_info_from_report_decodes_terminal_sampled_token_text() {
    let _probe_env = clear_real_full_probe_env();
    let _cuda_reference_override = cuda_reference_kernels_test_override(false);
    let catalog = full_catalog();
    let args = coordinator_args();
    let mut report = real_ds4_full_preflight_report(&args, "reports/model_catalog.json", &catalog)
        .expect("real-full preflight report builds");
    let snapshot = tempfile::tempdir().expect("creating tokenizer snapshot");
    fs::write(
        snapshot.path().join("tokenizer.json"),
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"decoded-sample":42},"unk_token":"[UNK]"}}"#,
    )
    .expect("writing tokenizer fixture");
    report.snapshot_path = snapshot.path().display().to_string();
    report
        .scheduler_execution_dry_run
        .terminal_lm_head_sample
        .sampled_token_id = Some(42);

    let info = real_full_info_from_report(&report);

    assert_eq!(
        info.scheduler_terminal_lm_head_sampled_text.as_deref(),
        Some("decoded-sample")
    );
}

#[test]
fn real_full_kv_capacity_uses_configured_global_context_budget() {
    let mut args = coordinator_args();
    args.max_context_tokens = 256 * 1024;

    for dtype in ["fp8"] {
        args.kv_cache_dtype = dtype.to_owned();
        let config = real_full_kv_cache_config_for_model(&args, &ModelFacts::default())
            .expect("configured DeepSeek V4 KV cache builds");
        assert_eq!(config.max_tokens, 256 * 1024);
        assert_eq!(config.layout_label(), "deepseek-v4-hybrid-bf16");
        assert_eq!(
            config.capacity_bytes(),
            config.bytes_per_token() * 256 * 1024
        );
    }

    args.max_context_tokens = 0;
    assert!(real_full_kv_cache_config_for_model(&args, &ModelFacts::default()).is_err());
}

#[test]
fn real_full_scheduler_execution_accepts_request_shaped_prefill_decode_mtp_rows() {
    let _probe_env = clear_real_full_probe_env();
    let _cuda_reference_override = cuda_reference_kernels_test_override(false);
    let catalog = full_catalog();
    let shape = RealFullSchedulerExecutionShape {
        request_id: "real-full-request-shaped-test".to_owned(),
        sequence_id: "real-full-request-shaped-test-sequence".to_owned(),
        placement_version: "real-full-request-shaped-test-placement".to_owned(),
        prefix_tokens: 0,
        prefill_tokens: 9,
        prefill_chunk_tokens: 4,
        decode_rows: 2,
        mtp_rows: 3,
        mtp_accepted_rows: 1,
        prefill_token_ids: None,
        decode_token_ids: None,
        lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
    };
    let report = real_full_scheduler_execution_for_shape(
        ds41rt_core::KvCacheConfig::glm52_phase0(shape.reservation_tokens()),
        &catalog,
        shape.clone(),
    )
    .expect("request-shaped scheduler execution report builds");
    let prefill_chunks = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
    let sparse_layers = catalog.facts.num_hidden_layers - catalog.facts.first_k_dense_replace;
    let rows_per_layer = shape.prefill_tokens + shape.decode_rows + shape.mtp_rows;

    assert_eq!(report.request_prefill_tokens, shape.prefill_tokens);
    assert_eq!(report.request_prefill_chunks, prefill_chunks);
    assert_eq!(report.request_decode_rows, shape.decode_rows);
    assert_eq!(report.request_mtp_verify_rows, shape.mtp_rows);
    assert_eq!(report.request_mtp_accepted_rows, shape.mtp_accepted_rows);
    assert_eq!(
        report.iterations,
        catalog.facts.num_hidden_layers * prefill_chunks
    );
    assert_eq!(
        report.candidate_layerwaves,
        catalog.facts.num_hidden_layers
            * (prefill_chunks + shape.decode_rows + usize::from(shape.mtp_rows > 0))
    );
    assert_eq!(report.candidate_layerwaves, report.selected_layerwaves);
    assert_eq!(report.deferred_layerwaves, 0);
    assert_eq!(report.sparse_expert_batches, sparse_layers * prefill_chunks);
    assert_eq!(
        report.sparse_expert_batch_rows,
        sparse_layers * rows_per_layer
    );
    assert_eq!(
        report.sparse_expert_batch_routes,
        report.sparse_expert_batch_rows * catalog.facts.top_k
    );
    assert_eq!(
        report.sparse_expert_prefill_rows,
        sparse_layers * shape.prefill_tokens
    );
    assert_eq!(
        report.sparse_expert_decode_rows,
        sparse_layers * shape.decode_rows
    );
    assert_eq!(
        report.sparse_expert_mtp_verify_rows,
        sparse_layers * shape.mtp_rows
    );
    assert_eq!(
        report.sparse_expert_prefill_routes,
        report.sparse_expert_prefill_rows * catalog.facts.top_k
    );
    assert_eq!(
        report.sparse_expert_decode_routes,
        report.sparse_expert_decode_rows * catalog.facts.top_k
    );
    assert_eq!(
        report.sparse_expert_mtp_verify_routes,
        report.sparse_expert_mtp_verify_rows * catalog.facts.top_k
    );
    assert_eq!(
        report.sparse_expert_host_batch_sets,
        report.sparse_expert_batches
    );
    assert_eq!(
        report.sparse_expert_host_batches,
        report.sparse_expert_batches * ds41rt_core::EXPERT_HOSTS.len()
    );
    assert_eq!(
        report.sparse_expert_host_batch_rows,
        report.sparse_expert_batch_rows * ds41rt_core::EXPERT_HOSTS.len()
    );
    assert_eq!(
        report.sparse_expert_host_batch_routes,
        report.sparse_expert_batch_routes * ds41rt_core::DS4_EXPERT_TP_WORLD_SIZE
    );
    assert!(report.sparse_expert_host_batch_expert_tiles > 0);
    assert!(report.sparse_expert_host_batch_expert_tiles <= report.sparse_expert_host_batch_routes);
    assert!(report.sparse_expert_host_batch_routes_match_global);
    assert!(report.sparse_expert_host_batch_graph_counts_valid);
    assert_eq!(
        report.sparse_expert_host_request_frames,
        report.sparse_expert_host_batches
    );
    assert_eq!(
        report.sparse_expert_host_request_rows,
        report.sparse_expert_host_batch_rows
    );
    assert_eq!(
        report.sparse_expert_host_request_routes,
        report.sparse_expert_host_batch_routes
    );
    assert_eq!(
        report.sparse_expert_host_request_payload_bytes,
        report.sparse_expert_host_batch_rows
            * catalog.facts.hidden_size
            * std::mem::size_of::<u16>()
    );
    assert!(
        report.sparse_expert_host_request_wire_bytes
            > report.sparse_expert_host_request_payload_bytes
    );
    assert_eq!(
        report.sparse_expert_host_response_frames,
        report.sparse_expert_host_batches
    );
    assert_eq!(
        report.sparse_expert_host_response_rows,
        report.sparse_expert_host_batch_rows
    );
    assert_eq!(
        report.sparse_expert_host_response_payload_bytes,
        report.sparse_expert_host_batch_rows
            * catalog.facts.hidden_size
            * std::mem::size_of::<u16>()
    );
    assert!(
        report.sparse_expert_host_response_wire_bytes
            > report.sparse_expert_host_response_payload_bytes
    );
    assert!(report.sparse_expert_host_wire_envelopes_valid);
    assert_eq!(
        report.backed_kv_writes,
        report.committed_kv_writes + report.committed_mtp_writes
    );
    assert_eq!(
        report.kv_reservation_bytes,
        shape.reservation_tokens() * ds41rt_core::KvCacheConfig::glm52_phase0(1).bytes_per_token()
    );
    assert!(report.backed_bytes_after_discard <= report.kv_reservation_bytes);
    assert!(report.byte_backed_scheduler_trace);
    assert_eq!(
        report.numeric_progression_self_test.unique_source_rows,
        rows_per_layer
    );
    assert_eq!(
        report.numeric_progression_self_test.selected_prefill_rows,
        shape.prefill_tokens * catalog.facts.num_hidden_layers
    );
    assert_eq!(
        report.numeric_progression_self_test.selected_decode_rows,
        shape.decode_rows * catalog.facts.num_hidden_layers
    );
    assert_eq!(
        report.numeric_progression_self_test.selected_mtp_rows,
        shape.mtp_rows * catalog.facts.num_hidden_layers
    );
    assert_eq!(
        report
            .numeric_progression_self_test
            .mtp_accepted_rows_per_layer,
        shape.mtp_accepted_rows
    );
    assert_eq!(
        report
            .numeric_progression_self_test
            .mtp_rejected_rows_per_layer,
        shape.mtp_rows - shape.mtp_accepted_rows
    );
}

#[test]
fn coordinator_resident_loaded_preload_requirement_accepts_loaded_plan() {
    let _probe_env = clear_real_full_probe_env();
    let catalog = full_catalog();
    let mut resident_preload = real_full_coordinator_resident_preload_plan(&catalog);
    resident_preload.status = "loaded";
    resident_preload.loaded_tensor_bytes = resident_preload.selected_tensor_bytes;

    assert_eq!(resident_preload.status, "loaded");
    assert_eq!(
        resident_preload.loaded_tensor_bytes,
        resident_preload.selected_tensor_bytes
    );
    let requirement = coordinator_resident_preload_requirement(&resident_preload);

    assert_eq!(
        requirement.name,
        "coordinator_startup_resident_preload_plan_available"
    );
    assert!(requirement.passed);
    assert!(requirement.evidence.contains("status=loaded"));
    assert!(requirement.evidence.contains(&format!(
        "loaded_bytes={}",
        resident_preload.selected_tensor_bytes
    )));
}

#[test]
fn real_full_runtime_info_uses_loaded_residency_without_probe_preflight() {
    let _probe_env = clear_real_full_probe_env();
    let catalog = full_catalog();
    let args = coordinator_args();
    let mut resident_preload = real_full_coordinator_resident_preload_plan(&catalog);
    resident_preload.status = "loaded";
    resident_preload.loaded_tensor_bytes = resident_preload.selected_tensor_bytes;

    let info = real_full_info_from_startup(&args, &catalog, resident_preload)
        .expect("real-full runtime info builds from startup residency");

    assert_eq!(info.status, "blocked");
    assert_eq!(
        info.snapshot_path.as_deref(),
        Some(catalog.snapshot_path.as_str())
    );
    assert_eq!(
        info.startup_diagnostic_mode,
        "serving-startup-residency-only"
    );
    assert_eq!(info.coordinator_resident_preload_status, "loaded");
    assert_eq!(
        info.coordinator_resident_preload_loaded_bytes,
        info.coordinator_resident_preload_selected_bytes
    );
    assert!(info.coordinator_resident_preload_selected_tensors > 0);
    assert_eq!(info.layer_count, 43);
    assert_eq!(info.dense_layer_count, 0);
    assert_eq!(info.sparse_layer_count, 43);
    assert_eq!(info.scheduler_iterations, 0);
    assert_eq!(info.selected_layerwaves, 0);
    assert!(!info.scheduler_numeric_progression_passed);
    assert_eq!(info.scheduler_numeric_progression_source_rows, 0);
    assert!(!info.scheduler_full_context_device_attention_complete);
    assert_eq!(info.scheduler_terminal_lm_head_sample_status, "not-run");
    assert!(!info.scheduler_terminal_lm_head_sample_passed);
    assert!(!info.scheduler_terminal_lm_head_uses_final_decode_device_hidden);
    assert!(!info.scheduler_terminal_lm_head_covers_full_vocabulary);
    assert_eq!(info.scheduler_terminal_lm_head_logits_evaluated, 0);
    assert_eq!(info.scheduler_terminal_lm_head_vocab_size, 0);
    assert_eq!(info.scheduler_terminal_lm_head_top_token_id, None);
    assert_eq!(info.scheduler_terminal_lm_head_sampled_token_id, None);
    assert_eq!(info.scheduler_terminal_lm_head_sampled_text, None);
    assert_eq!(info.scheduler_terminal_lm_head_sample_top_k, None);
    assert_eq!(info.scheduler_terminal_lm_head_sample_top_p, None);
    assert_eq!(info.scheduler_terminal_lm_head_argmax_backend, None);
    assert_eq!(info.scheduler_terminal_lm_head_sampler_backend, None);
    assert!(info.scheduler_terminal_lm_head_blocker.is_some());
    assert!(!info.sampling_default_lm_head_chunk_passed);
    assert_eq!(info.sampling_default_lm_head_chunk_rows_scored, 0);
    assert_eq!(info.sampling_default_lm_head_chunk_lm_head_bytes_read, 0);
    assert_eq!(info.sampling_default_lm_head_chunk_top_token_id, None);
    assert_eq!(info.sampling_default_lm_head_chunk_top_logit, None);
    assert_eq!(
        info.failed_requirements,
        vec![
            "full_residual_stream_execution".to_owned(),
            "full_vocab_sampling".to_owned()
        ]
    );
}
