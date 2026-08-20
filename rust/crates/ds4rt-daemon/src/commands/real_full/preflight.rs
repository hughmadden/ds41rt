use anyhow::Result;
use ds4rt_core::{
    DeepseekV4AttentionPlan, DeepseekV4KvCacheFormat, DeepseekV4PhysicalKvPlan, KvCacheConfig,
    KvCacheDType, ModelFacts, TensorCatalog, TransportCapabilities,
};

use crate::cli::CoordinatorArgs;

mod requirements;

use super::attention::real_full_attention_kv_binding_dry_run;
use super::constants::{
    REAL_DS4_FULL_BLOCKER, REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
    REAL_FULL_PREFLIGHT_MTP_ROWS, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START,
};
use super::coverage::tensor_coverage;
use super::execution_plan::real_full_execution_plan;
use super::experts::real_full_expert_execution_dry_run;
use super::kv::{
    real_full_attention_kv_io_dry_run_for_layers, real_full_kv_backing_store_dry_run_for_layers,
};
use super::residency::real_full_coordinator_resident_preload_plan;
use super::residual::real_full_residual_stream_dry_run;
use super::sampling::real_full_sampling_dry_run;
use super::scheduler::{real_full_scheduler_dry_run, scheduler_prefill_chunk_count_for_rows};
use super::types::{
    RealDs4FullPreflightReport, RealFullCoordinatorResidentPreloadPlan, RealFullKvPlan,
    RealFullNativePhysicalKvPlan, RealFullRequirement, RealFullSchedulerExecutionDryRun,
    RealFullSparseTransportPlan,
};
use requirements::{real_full_preflight_requirements, RealFullPreflightRequirementInputs};

pub(super) fn real_ds4_full_preflight_report(
    args: &CoordinatorArgs,
    catalog_source: &str,
    catalog: &TensorCatalog,
) -> Result<RealDs4FullPreflightReport> {
    let coordinator_resident_preload = real_full_coordinator_resident_preload_plan(catalog);
    real_ds4_full_preflight_report_with_coordinator_resident_preload(
        args,
        catalog_source,
        catalog,
        coordinator_resident_preload,
    )
}

pub(super) fn coordinator_resident_preload_requirement(
    coordinator_resident_preload: &RealFullCoordinatorResidentPreloadPlan,
) -> RealFullRequirement {
    requirements::coordinator_resident_preload_requirement(coordinator_resident_preload)
}

pub(super) fn real_ds4_full_preflight_report_with_coordinator_resident_preload(
    args: &CoordinatorArgs,
    catalog_source: &str,
    catalog: &TensorCatalog,
    coordinator_resident_preload: RealFullCoordinatorResidentPreloadPlan,
) -> Result<RealDs4FullPreflightReport> {
    let coverage = tensor_coverage(catalog);
    let catalog_hash = catalog.content_hash();
    let kv_config = real_full_kv_cache_config_for_model(args, &catalog.facts)?;
    let native_cache_format = match KvCacheDType::parse_cache_dtype(&args.kv_cache_dtype) {
        Some(KvCacheDType::Nvfp4) => DeepseekV4KvCacheFormat::Nvfp4,
        _ => DeepseekV4KvCacheFormat::Fp8Ue8m0,
    };
    let native_physical_kv = if catalog.facts.model_type == "deepseek_v4" {
        Some(DeepseekV4PhysicalKvPlan::for_model_with_format(
            &catalog.facts,
            args.max_context_tokens,
            false,
            native_cache_format,
        )?)
    } else {
        None
    };
    let native_physical_kv_report = if let Some(plan) = native_physical_kv.as_ref() {
        let max_partial_boundary = plan.boundary_copy_plan(plan.source_page_tokens - 1)?;
        Some(RealFullNativePhysicalKvPlan {
            status: "active",
            layout: match plan.cache_format {
                DeepseekV4KvCacheFormat::Fp8Ue8m0 => "sparkinfer-dsv4-paged-fp8-ue8m0",
                DeepseekV4KvCacheFormat::Nvfp4 => "sparkinfer-dsv4-paged-nvfp4",
            },
            active: true,
            source_page_tokens: plan.source_page_tokens,
            source_page_count: plan.source_page_count,
            layer_count: plan.layers.len(),
            persistent_bytes: plan.persistent_bytes,
            persistent_bytes_per_logical_token: plan.persistent_bytes_per_logical_token(),
            compressor_state_bytes_per_sequence: plan.compressor_state_bytes_per_sequence,
            boundary_copy_status: match plan.cache_format {
                DeepseekV4KvCacheFormat::Fp8Ue8m0 => "planar-active",
                DeepseekV4KvCacheFormat::Nvfp4 => "row-major-active",
            },
            max_partial_boundary_copy_spans: max_partial_boundary
                .layers
                .iter()
                .map(|layer| layer.spans.len())
                .sum(),
            max_partial_boundary_copy_bytes: max_partial_boundary.copied_bytes(),
            compressor_state_replay: true,
            attention_adapter: "sparkinfer-dsv4-compressed-mla",
            attention_adapter_status: "active",
            attention_adapter_startup_qualification: "required-when-python-capture-enabled",
        })
    } else {
        None
    };
    let sparse_transport = real_full_sparse_transport_plan(args);
    let expert_hosts = args
        .expert_hosts
        .split(',')
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let execution_plan =
        real_full_execution_plan(&expert_hosts, kv_config.bytes_per_token(), &catalog.facts);
    let residual_stream_dry_run = real_full_residual_stream_dry_run(&execution_plan, catalog);
    let sampling_dry_run = real_full_sampling_dry_run(catalog, &execution_plan)?;
    let scheduler_dry_run = real_full_scheduler_dry_run(&catalog_hash, &catalog.facts)?;
    // Startup preflight must not execute the legacy scheduler's numerical
    // attention path. Live DeepSeek requests supply native target storage and
    // execute the catalog-shaped target-attention graphs; preflight records
    // only the jointly admitted row/route geometry needed for planning.
    let scheduler_execution_dry_run =
        real_full_scheduler_execution_preflight_summary(&catalog.facts);
    let expert_execution_dry_run = real_full_expert_execution_dry_run(catalog, &expert_hosts);
    let kv_backing_store_dry_run = real_full_kv_backing_store_dry_run_for_layers(
        kv_config.clone(),
        catalog.facts.num_hidden_layers,
    )?;
    let attention_kv_io_dry_run = real_full_attention_kv_io_dry_run_for_layers(
        kv_config.clone(),
        catalog.facts.num_hidden_layers,
    )?;
    let attention_kv_binding_dry_run =
        real_full_attention_kv_binding_dry_run(catalog, &kv_config, &attention_kv_io_dry_run);
    let requirements = real_full_preflight_requirements(RealFullPreflightRequirementInputs {
        catalog,
        coverage: &coverage,
        kv_config: &kv_config,
        expert_hosts: &expert_hosts,
        execution_plan: &execution_plan,
        residual_stream_dry_run: &residual_stream_dry_run,
        sampling_dry_run: &sampling_dry_run,
        expert_execution_dry_run: &expert_execution_dry_run,
        scheduler_dry_run: &scheduler_dry_run,
        scheduler_execution_dry_run: &scheduler_execution_dry_run,
        kv_backing_store_dry_run: &kv_backing_store_dry_run,
        attention_kv_io_dry_run: &attention_kv_io_dry_run,
        attention_kv_binding_dry_run: &attention_kv_binding_dry_run,
        coordinator_resident_preload: &coordinator_resident_preload,
    });
    Ok(RealDs4FullPreflightReport {
        backend: "real-ds4-full",
        status: "blocked",
        model_id: args.model_id.clone(),
        catalog_path: catalog_source.to_owned(),
        snapshot_path: catalog.snapshot_path.clone(),
        catalog_hash,
        tensor_count: catalog.tensors.len(),
        listen: args.listen.clone(),
        transport: args.transport.clone(),
        sparse_transport,
        expert_hosts,
        model_facts: catalog.facts.clone(),
        expected_facts: catalog.facts.clone(),
        role_counts: catalog.summary_by_role(),
        full_model_tensor_coverage: coverage,
        kv_plan: RealFullKvPlan {
            layout: kv_config.layout_label(),
            dtype: kv_config.dtype_label(),
            max_tokens: kv_config.max_tokens,
            bytes_per_token: kv_config.bytes_per_token(),
            capacity_bytes: kv_config.capacity_bytes(),
            native_physical: native_physical_kv_report,
        },
        execution_plan,
        residual_stream_dry_run,
        sampling_dry_run,
        expert_execution_dry_run,
        scheduler_dry_run,
        scheduler_execution_dry_run,
        kv_backing_store_dry_run,
        attention_kv_io_dry_run,
        attention_kv_binding_dry_run,
        coordinator_resident_preload,
        requirements,
        blocker: REAL_DS4_FULL_BLOCKER,
    })
}

fn real_full_scheduler_execution_preflight_summary(
    facts: &ModelFacts,
) -> RealFullSchedulerExecutionDryRun {
    let request_prefill_tokens =
        REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize + REAL_FULL_PREFLIGHT_PREFILL_ROWS;
    let request_prefill_chunks = scheduler_prefill_chunk_count_for_rows(
        request_prefill_tokens,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS,
        REAL_FULL_PREFLIGHT_DECODE_ROWS + REAL_FULL_PREFLIGHT_MTP_ROWS,
    );
    let sparse_layers = facts
        .num_hidden_layers
        .saturating_sub(facts.first_k_dense_replace);
    let sparse_expert_batch_rows = sparse_layers
        * (request_prefill_tokens + REAL_FULL_PREFLIGHT_DECODE_ROWS + REAL_FULL_PREFLIGHT_MTP_ROWS);

    RealFullSchedulerExecutionDryRun {
        status: "not-run-preflight-uses-structural-deepseek-plan",
        scope: "structural DeepSeek V4 joint prefill/decode/dSpark target planning; numerical execution occurs only in the live native-target scheduler",
        request_prefill_tokens,
        request_prefill_chunks,
        request_decode_rows: REAL_FULL_PREFLIGHT_DECODE_ROWS,
        request_mtp_verify_rows: REAL_FULL_PREFLIGHT_MTP_ROWS,
        request_mtp_accepted_rows: REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS,
        sparse_expert_batches: sparse_layers * request_prefill_chunks,
        sparse_expert_batch_rows,
        sparse_expert_batch_routes: sparse_expert_batch_rows * facts.top_k,
        sparse_expert_prefill_rows: sparse_layers * request_prefill_tokens,
        sparse_expert_decode_rows: sparse_layers * REAL_FULL_PREFLIGHT_DECODE_ROWS,
        sparse_expert_mtp_verify_rows: sparse_layers * REAL_FULL_PREFLIGHT_MTP_ROWS,
        sparse_expert_prefill_routes: sparse_layers * request_prefill_tokens * facts.top_k,
        sparse_expert_decode_routes: sparse_layers
            * REAL_FULL_PREFLIGHT_DECODE_ROWS
            * facts.top_k,
        sparse_expert_mtp_verify_routes: sparse_layers
            * REAL_FULL_PREFLIGHT_MTP_ROWS
            * facts.top_k,
        device_kv_status: "not-run",
        device_attention_status: "not-run",
        numeric_progression_self_test: super::types::RealFullSchedulerNumericProgressionSelfTest {
            status: "not-run",
            scope: "live native DeepSeek V4 target execution only",
            source_modes: ["prefill", "decode", "dspark-target-verify"],
            residual_dtype: "bf16",
            device_attention_output_delta_status: "not-run",
            attention_device_output_delta_backend: "not-run",
            attention_device_output_delta_device_prefix_backend: "not-run",
            device_delta_template_status: "not-run",
            device_mlp_delta_status: "not-run",
            device_mlp_delta_backend: "not-run",
            device_real_dense_mlp_delta_status: "not-run",
            device_real_dense_mlp_delta_backend: "not-run",
            device_real_dense_mlp_norm_backend: "not-run",
            device_real_sparse_shared_mlp_delta_status: "not-run",
            device_real_sparse_shared_mlp_delta_backend: "not-run",
            device_real_sparse_shared_mlp_norm_backend: "not-run",
            device_real_sparse_routed_mlp_delta_status: "not-run",
            device_real_sparse_routed_mlp_delta_backend: "not-run",
            device_real_sparse_routed_mlp_route_backend: "not-run",
            device_real_sparse_routed_mlp_router_backend: "not-run",
            device_hidden_segment_status: "not-run",
            device_hidden_segment_residual_add_backend: "not-run",
            ..Default::default()
        },
        terminal_lm_head_sample: super::types::RealFullSchedulerTerminalLmHeadSample {
            status: "not-run",
            scope: "live final DeepSeek V4 residual only",
            blocker: Some(
                "startup structural planning does not manufacture a terminal residual".to_owned(),
            ),
            ..Default::default()
        },
        ..Default::default()
    }
}

pub(in crate::commands::real_full) fn real_full_sparse_transport_plan(
    args: &CoordinatorArgs,
) -> RealFullSparseTransportPlan {
    match args.transport.as_str() {
        "tcp" => {
            let targets_configured = args
                .expert_hosts
                .split(',')
                .map(str::trim)
                .any(|target| !target.is_empty());
            sparse_transport_plan_from_capabilities(
                args,
                ds4rt_transport::tcp_capabilities(),
                if targets_configured {
                    "ready-tcp-protocol-v2"
                } else {
                    "blocked-missing-expert-hosts"
                },
                targets_configured,
                Some("tcp-protocol-v2-persistent-client"),
                true,
                None,
                Some(ds4rt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL),
                (!targets_configured).then_some(
                    "real-ds4-full TCP sparse dispatch requires --expert-hosts".to_owned(),
                ),
            )
        }
        "inproc" => sparse_transport_plan_from_capabilities(
            args,
            ds4rt_transport::inproc_capabilities(),
            "disabled-inproc",
            false,
            None,
            true,
            None,
            None,
            Some(
                "real-ds4-full inproc transport disables sparse expert dispatch; use --transport tcp for live sparse serving"
                    .to_owned(),
            ),
        ),
        "tcp-debug-json" | "debug-json" => sparse_transport_plan_from_capabilities(
            args,
            ds4rt_transport::tcp_capabilities(),
            "blocked-debug-json-not-supported",
            false,
            None,
            true,
            None,
            Some(ds4rt_transport::DEBUG_JSON_FRAME_PROTOCOL),
            Some(
                "real-ds4-full sparse serving requires ProtocolV2 TCP, not debug-json framing"
                    .to_owned(),
            ),
        ),
        "verbs-host" => {
            let targets_configured = args
                .expert_hosts
                .split(',')
                .map(str::trim)
                .any(|target| !target.is_empty());
            let preflight = ds4rt_transport::verbs_host_preflight();
            let (preflight_ok, preflight_error) = match preflight {
                Ok(_) => (true, None),
                Err(error) => (false, Some(error.to_string())),
            };
            let blocker = if let Some(error) = &preflight_error {
                Some(format!(
                    "real-ds4-full verbs-host sparse dispatch RDMA preflight failed: {error}"
                ))
            } else if !targets_configured {
                Some("real-ds4-full verbs-host sparse dispatch requires --expert-hosts".to_owned())
            } else {
                None
            };
            let sparse_dispatch_available = preflight_ok && targets_configured;
            sparse_transport_plan_from_capabilities(
                args,
                ds4rt_transport::verbs_host_capabilities(),
                if sparse_dispatch_available {
                    "ready-verbs-host-protocol-v2"
                } else if preflight_ok {
                    "blocked-missing-expert-hosts"
                } else {
                    "blocked-rdma-preflight-failed"
                },
                sparse_dispatch_available,
                sparse_dispatch_available.then_some("verbs-host-protocol-v2-rc-qp"),
                preflight_ok,
                preflight_error,
                Some(ds4rt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL),
                blocker,
            )
        }
        other => RealFullSparseTransportPlan {
            transport: other.to_owned(),
            status: "blocked-unsupported-transport".to_owned(),
            sparse_dispatch_available: false,
            scheduler_dispatch_backend: None,
            supports_rdma: false,
            supports_host_registered_buffers: false,
            requires_pinned_host_memory: false,
            app_transport_implemented: false,
            app_transport_status: "unsupported".to_owned(),
            preflight_ok: false,
            preflight_error: Some(format!("unsupported real-ds4-full sparse transport: {other}")),
            frame_protocol: None,
            blocker: Some(format!(
                "unsupported real-ds4-full sparse transport: {other}"
            )),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn sparse_transport_plan_from_capabilities(
    args: &CoordinatorArgs,
    capabilities: TransportCapabilities,
    status: &str,
    sparse_dispatch_available: bool,
    scheduler_dispatch_backend: Option<&str>,
    preflight_ok: bool,
    preflight_error: Option<String>,
    frame_protocol: Option<&str>,
    blocker: Option<String>,
) -> RealFullSparseTransportPlan {
    RealFullSparseTransportPlan {
        transport: args.transport.clone(),
        status: status.to_owned(),
        sparse_dispatch_available,
        scheduler_dispatch_backend: scheduler_dispatch_backend.map(str::to_owned),
        supports_rdma: capabilities.supports_rdma,
        supports_host_registered_buffers: capabilities.supports_host_registered_buffers,
        requires_pinned_host_memory: capabilities.requires_pinned_host_memory,
        app_transport_implemented: capabilities.app_transport_implemented,
        app_transport_status: capabilities.app_transport_status,
        preflight_ok,
        preflight_error,
        frame_protocol: frame_protocol.map(str::to_owned),
        blocker,
    }
}

pub(super) fn real_full_kv_cache_config_for_model(
    args: &CoordinatorArgs,
    facts: &ModelFacts,
) -> Result<KvCacheConfig> {
    anyhow::ensure!(
        args.max_context_tokens > 0,
        "real-full --max-context-tokens must be a positive integer"
    );
    let dtype = KvCacheDType::parse_cache_dtype(&args.kv_cache_dtype).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported real-full --kv-cache-dtype {}; expected bf16, fp8, or nvfp4",
            args.kv_cache_dtype
        )
    })?;
    anyhow::ensure!(
        facts.model_type == "deepseek_v4",
        "real-full requires model_type deepseek_v4, got {:?}",
        facts.model_type
    );
    anyhow::ensure!(
        matches!(
            dtype,
            KvCacheDType::Bf16 | KvCacheDType::Fp8 | KvCacheDType::Nvfp4
        ),
        "DeepSeek V4 native KV cache supports bf16, fp8, or nvfp4 profiles; {} is not implemented",
        dtype.label()
    );
    let attention = DeepseekV4AttentionPlan::from_model_facts(facts)?;
    attention.validate_sparkinfer_sm120_contract()?;
    // The active DeepSeek target owns SparkInfer-format 256-token physical
    // pages and packs FP8/UE8M0 or NVFP4 directly in its captured producer. Keep the
    // generic transactional mirror BF16: its inherited FP8 codec is the
    // GLM 512+64 layout and must never be substituted for DeepSeek's
    // checkpoint-derived 448+64 physical ABI.
    Ok(KvCacheConfig::try_deepseek_v4_hybrid_bf16(
        args.max_context_tokens,
        facts,
    )?)
}
