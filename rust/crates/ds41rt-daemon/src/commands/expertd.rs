use anyhow::{Context, Result};
use ds41rt_core::{
    owner_for_expert, PlacementPolicy, TensorCatalog, TensorInfo, TensorRole, EXPERT_HOSTS,
};
use serde::Serialize;
use std::{fs::File, path::Path, sync::Arc, time::Instant};

use crate::cli::ExpertDaemonArgs;
use crate::commands::model_artifacts::{
    build_runtime_catalog, build_runtime_owner_lookup, read_expert_loadplan,
    read_expert_serving_loadplan, validate_production_runtime_catalog,
    validate_runtime_expert_role, ExpertOwnerLookup,
};
use crate::commands::real_full::{
    real_nvfp4_cuda_reference_kernels_enabled, spark_expert_intermediate_shard_from_env,
    spark_expert_owner_reduction_config_from_env, ExpertIntermediateShard,
    RealNvfp4ProtocolV2Executor, RealNvfp4ResidentPreloadPlan,
    REAL_NVFP4_CUDA_REFERENCE_KERNELS_ENV,
};
use crate::python_graph_capture::initialize_spark_python_capture_from_env;

fn report_expertd_startup_phase(
    stage: &str,
    startup_started: Instant,
    phase_started: &mut Instant,
) {
    let now = Instant::now();
    eprintln!(
        "expertd_startup_phase stage={stage} elapsed_ms={:.3} total_ms={:.3}",
        now.duration_since(*phase_started).as_secs_f64() * 1_000.0,
        now.duration_since(startup_started).as_secs_f64() * 1_000.0,
    );
    *phase_started = now;
}

pub(crate) async fn run_expertd(args: ExpertDaemonArgs) -> Result<()> {
    let startup_started = Instant::now();
    let mut phase_started = startup_started;
    if args.preflight_only {
        let report = expertd_preflight_report(&args)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let mut serving_owner_lookup: Option<ExpertOwnerLookup> = None;
    let catalog_source = (!args.synthetic_weights).then(|| {
        args.catalog
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| format!("hf://{}", args.model_id))
    });
    let loadplan_readiness = match args.loadplan.as_deref() {
        Some(path) if args.synthetic_weights => Some(read_expert_loadplan(
            path,
            args.role_hostname.as_deref(),
            args.real_layer,
        )?),
        Some(path) => {
            let serving_loadplan =
                read_expert_serving_loadplan(path, args.role_hostname.as_deref(), args.real_layer)?;
            serving_owner_lookup = Some(serving_loadplan.owner_lookup);
            Some(serving_loadplan.readiness)
        }
        None => None,
    };
    report_expertd_startup_phase("loadplan", startup_started, &mut phase_started);
    println!(
        "starting expertd synthetic_weights={} transport={} listen={} model_id={} loadplan={:?} catalog_source={:?} real_layer={:?} role={:?}",
        args.synthetic_weights,
        args.transport,
        args.listen,
        args.model_id,
        args.loadplan,
        catalog_source,
        args.real_layer,
        args.role_hostname
    );
    if let Some(readiness) = &loadplan_readiness {
        println!(
            "expertd_loadplan_ready {}",
            serde_json::to_string(readiness).context("serializing expert loadplan readiness")?
        );
    }
    if !args.synthetic_weights {
        if !matches!(args.transport.as_str(), "tcp" | "verbs-host") {
            anyhow::bail!(
                "phase0 real-weight expertd serving supports --transport tcp or verbs-host, got {}",
                args.transport
            );
        }
        if !real_nvfp4_cuda_reference_kernels_enabled() {
            anyhow::bail!(
                "phase0 real-weight expertd serving requires {REAL_NVFP4_CUDA_REFERENCE_KERNELS_ENV}=1 and DS41RT_NATIVE_LIB pointing at native/build-cuda/libds41rt_native.so; refusing CPU fallback"
            );
        }
        if let Some(status) = initialize_spark_python_capture_from_env()
            .context("initializing Spark Python graph-capture bridge")?
        {
            println!(
                "spark_python_graph_capture status=enabled gate={} modules={}",
                status.gate_env,
                status.imported_modules.join(",")
            );
        }
        report_expertd_startup_phase("python-capture", startup_started, &mut phase_started);
        let (resolved_catalog_source, catalog) = resolve_expertd_catalog(&args)?;
        if args.loadplan.is_none() {
            validate_runtime_expert_role(args.role_hostname.as_deref())?;
            serving_owner_lookup = Some(build_runtime_owner_lookup(&catalog)?);
        }
        let intermediate_shard = real_tp4_intermediate_shard_from_env()?;
        report_expertd_startup_phase("catalog-owner-config", startup_started, &mut phase_started);
        let original_catalog_tensors = catalog.tensors.len();
        let catalog = filter_real_weight_serving_catalog(
            catalog,
            serving_owner_lookup.as_ref(),
            args.role_hostname.as_deref(),
            args.real_layer,
            true,
        );
        let (_, routed_expert_tensors) =
            validate_real_weight_catalog(&catalog, &resolved_catalog_source)?;
        println!(
            "expertd_serving_catalog_ready original_tensors={} serving_tensors={} routed_expert_tensors={}",
            original_catalog_tensors,
            catalog.tensors.len(),
            routed_expert_tensors
        );
        report_expertd_startup_phase(
            "catalog-filter-validation",
            startup_started,
            &mut phase_started,
        );
        let mut executor = RealNvfp4ProtocolV2Executor::new(
            catalog,
            args.real_layer.map(|layer_id| layer_id as usize),
            args.role_hostname.clone(),
        );
        if let Some(owner_lookup) = serving_owner_lookup {
            executor = executor.with_owner_lookup(owner_lookup);
        }
        println!(
            "expertd_intermediate_shard topology=strict-tp4-replicated-experts count={} rank={}",
            intermediate_shard.count, intermediate_shard.rank
        );
        executor = executor.with_intermediate_shard(intermediate_shard);
        if let Some(owner_reduction) =
            spark_expert_owner_reduction_config_from_env(Some(intermediate_shard))?
        {
            println!(
                "expertd_owner_reduction rank={} peers={} dtype={:?} max_rows={}",
                owner_reduction.shard.rank,
                owner_reduction.peers.len(),
                owner_reduction.dtype,
                owner_reduction.max_rows
            );
            executor = executor.with_owner_reduction(owner_reduction)?;
        }
        report_expertd_startup_phase(
            "executor-configuration",
            startup_started,
            &mut phase_started,
        );
        let resident_preload = executor.preload_assigned_projections()?;
        report_expertd_startup_phase("resident-preload", startup_started, &mut phase_started);
        if !resident_preload.cuda_reference_enabled {
            anyhow::bail!(
                "phase0 real-weight expertd CUDA resident preload was not enabled after {REAL_NVFP4_CUDA_REFERENCE_KERNELS_ENV}=1"
            );
        }
        if resident_preload.cuda_projection_groups != resident_preload.projection_groups {
            anyhow::bail!(
                "phase0 real-weight expertd CUDA resident preload incomplete: cuda_projection_groups={} projection_groups={}",
                resident_preload.cuda_projection_groups,
                resident_preload.projection_groups
            );
        }
        println!(
            "expertd_real_weight_resident_preload projection_groups={} layers={} experts={} weight_bytes={} quant_metadata_bytes={} route_cache_entries={} route_cache_loads={} route_cache_hits={} projection_row_entries={} projection_row_loads={} projection_row_hits={} cuda_reference_enabled={} cuda_projection_groups={} cuda_weight_bytes={} cuda_weight_scale_bytes={} cuda_projection_entries={} cuda_projection_uploads={} cuda_cache_hits={}",
            resident_preload.projection_groups,
            resident_preload.layers,
            resident_preload.experts,
            resident_preload.weight_bytes,
            resident_preload.quant_metadata_bytes,
            resident_preload.route_cache_entries,
            resident_preload.route_cache_loads,
            resident_preload.route_cache_hits,
            resident_preload.projection_row_entries,
            resident_preload.projection_row_loads,
            resident_preload.projection_row_hits,
            resident_preload.cuda_reference_enabled,
            resident_preload.cuda_projection_groups,
            resident_preload.cuda_weight_bytes,
            resident_preload.cuda_weight_scale_bytes,
            resident_preload.cuda_projection_entries,
            resident_preload.cuda_projection_uploads,
            resident_preload.cuda_cache_hits,
        );
        let executor = Arc::new(executor);
        report_expertd_startup_phase("service-handoff", startup_started, &mut phase_started);
        return match args.transport.as_str() {
            "tcp" => {
                ds41rt_transport::serve_protocol_v2_tcp_with_executor(&args.listen, executor).await
            }
            "verbs-host" => {
                ds41rt_transport::serve_protocol_v2_verbs_host_with_executor(&args.listen, executor)
                    .await
            }
            other => anyhow::bail!("unsupported real-weight expertd transport: {other}"),
        };
    }
    match args.transport.as_str() {
        "tcp" => ds41rt_transport::serve_synthetic_protocol_v2_tcp(&args.listen).await,
        "tcp-debug-json" | "debug-json" => {
            ds41rt_transport::serve_synthetic_debug_json_tcp(&args.listen).await
        }
        "verbs-host" => ds41rt_transport::serve_synthetic_verbs_host(&args.listen).await,
        other => anyhow::bail!("unsupported expertd transport: {other}"),
    }
}

#[derive(Debug, Serialize)]
struct ExpertDaemonPreflightReport {
    status: String,
    transport: String,
    listen: String,
    synthetic_weights: bool,
    model_id: String,
    loadplan: Option<String>,
    catalog_source: Option<String>,
    real_layer: Option<u32>,
    #[serde(rename = "role")]
    role_hostname: Option<String>,
    supports_rdma: bool,
    supports_host_registered_buffers: bool,
    app_transport_implemented: bool,
    app_transport_status: String,
    real_weight_serving_ready: bool,
    real_weight_serving_status: String,
    real_weight_resident_preload_plan: Option<ExpertDaemonResidentPreloadPlan>,
    preflight_ok: bool,
    preflight_error: Option<String>,
    infiniband_path: Option<String>,
    frame_protocol: Option<String>,
    requires_pinned_host_memory: bool,
}

#[derive(Debug, Serialize)]
struct ExpertDaemonResidentPreloadPlan {
    startup_required: bool,
    projection_groups: usize,
    layers: usize,
    experts: usize,
    complete_expert_projection_sets: usize,
    incomplete_expert_projection_sets: usize,
    weight_bytes: u64,
    weight_scale_bytes: u64,
    scalar_metadata_bytes: u64,
    missing_metadata_tensors: usize,
}

impl From<RealNvfp4ResidentPreloadPlan> for ExpertDaemonResidentPreloadPlan {
    fn from(plan: RealNvfp4ResidentPreloadPlan) -> Self {
        Self {
            startup_required: plan.startup_required,
            projection_groups: plan.projection_groups,
            layers: plan.layers,
            experts: plan.experts,
            complete_expert_projection_sets: plan.complete_expert_projection_sets,
            incomplete_expert_projection_sets: plan.incomplete_expert_projection_sets,
            weight_bytes: plan.weight_bytes,
            weight_scale_bytes: plan.weight_scale_bytes,
            scalar_metadata_bytes: plan.scalar_metadata_bytes,
            missing_metadata_tensors: plan.missing_metadata_tensors,
        }
    }
}

fn expertd_preflight_report(args: &ExpertDaemonArgs) -> Result<ExpertDaemonPreflightReport> {
    expertd_preflight_report_with_intermediate_shard(args, None)
}

fn expertd_preflight_report_with_intermediate_shard(
    args: &ExpertDaemonArgs,
    intermediate_shard_override: Option<ExpertIntermediateShard>,
) -> Result<ExpertDaemonPreflightReport> {
    let (capabilities, preflight_ok, preflight_error, infiniband_path, frame_protocol) =
        match args.transport.as_str() {
            "tcp" => (
                ds41rt_transport::tcp_capabilities(),
                true,
                None,
                None,
                Some(ds41rt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL.to_owned()),
            ),
            "tcp-debug-json" | "debug-json" => (
                ds41rt_transport::tcp_capabilities(),
                true,
                None,
                None,
                Some(ds41rt_transport::DEBUG_JSON_FRAME_PROTOCOL.to_owned()),
            ),
            "verbs-host" => {
                let capabilities = ds41rt_transport::verbs_host_capabilities();
                let preflight = ds41rt_transport::verbs_host_preflight();
                match preflight {
                    Ok(preflight) => (
                        capabilities,
                        true,
                        None,
                        Some(preflight.infiniband_path),
                        Some(preflight.frame_protocol),
                    ),
                    Err(error) => (
                        capabilities,
                        false,
                        Some(error.to_string()),
                        Some("/dev/infiniband".to_owned()),
                        Some(ds41rt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL.to_owned()),
                    ),
                }
            }
            other => anyhow::bail!("unsupported expertd transport: {other}"),
        };
    let (real_weight_serving_ready, real_weight_serving_status, real_weight_resident_preload_plan) =
        real_weight_serving_preflight(args, intermediate_shard_override);
    let status =
        if preflight_ok && capabilities.app_transport_implemented && real_weight_serving_ready {
            "ready"
        } else {
            "blocked"
        };
    Ok(ExpertDaemonPreflightReport {
        status: status.to_owned(),
        transport: args.transport.clone(),
        listen: args.listen.clone(),
        synthetic_weights: args.synthetic_weights,
        model_id: args.model_id.clone(),
        loadplan: args
            .loadplan
            .as_ref()
            .map(|path| path.display().to_string()),
        catalog_source: if args.synthetic_weights {
            None
        } else {
            Some(
                args.catalog
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| format!("hf://{}", args.model_id)),
            )
        },
        real_layer: args.real_layer,
        role_hostname: args.role_hostname.clone(),
        supports_rdma: capabilities.supports_rdma,
        supports_host_registered_buffers: capabilities.supports_host_registered_buffers,
        app_transport_implemented: capabilities.app_transport_implemented,
        app_transport_status: capabilities.app_transport_status,
        real_weight_serving_ready,
        real_weight_serving_status,
        real_weight_resident_preload_plan,
        preflight_ok,
        preflight_error,
        infiniband_path,
        frame_protocol,
        requires_pinned_host_memory: capabilities.requires_pinned_host_memory,
    })
}

fn real_weight_serving_preflight(
    args: &ExpertDaemonArgs,
    intermediate_shard_override: Option<ExpertIntermediateShard>,
) -> (bool, String, Option<ExpertDaemonResidentPreloadPlan>) {
    if args.synthetic_weights {
        return (
            true,
            "synthetic ProtocolV2 route executor selected".to_owned(),
            None,
        );
    }
    let intermediate_shard = match intermediate_shard_override
        .map(Ok)
        .unwrap_or_else(real_tp4_intermediate_shard_from_env)
    {
        Ok(shard) => shard,
        Err(error) => {
            return (
                false,
                format!("real checkpoint NVFP4 intermediate shard config failed: {error:#}"),
                None,
            )
        }
    };
    if !matches!(args.transport.as_str(), "tcp" | "verbs-host") {
        return (
            false,
            format!(
                "real checkpoint NVFP4 serving requires --transport tcp or verbs-host in phase0, got {}",
                args.transport
            ),
            None,
        );
    }
    let (catalog_source, catalog) = match resolve_expertd_catalog(args) {
        Ok(resolved) => resolved,
        Err(error) => {
            return (
                false,
                format!("real checkpoint NVFP4 serving catalog preflight failed: {error:#}"),
                None,
            )
        }
    };
    let serving_owner_lookup = match args.loadplan.as_deref() {
        Some(loadplan_path) => match read_expert_serving_loadplan(
            loadplan_path,
            args.role_hostname.as_deref(),
            args.real_layer,
        ) {
            Ok(serving_loadplan) if serving_loadplan.owner_lookup.is_empty() => {
                return (
                    false,
                    format!(
                        "real checkpoint NVFP4 serving loadplan owner lookup preflight failed: {} has no routed expert owners",
                        loadplan_path.display()
                    ),
                    None,
                )
            }
            Ok(serving_loadplan) => Some(serving_loadplan.owner_lookup),
            Err(error) => {
                return (
                    false,
                    format!(
                        "real checkpoint NVFP4 serving loadplan owner lookup preflight failed: {error:#}"
                    ),
                    None,
                )
            }
        },
        None => {
            if let Err(error) = validate_runtime_expert_role(args.role_hostname.as_deref()) {
                return (
                    false,
                    format!("real checkpoint NVFP4 inferred role preflight failed: {error:#}"),
                    None,
                );
            }
            match build_runtime_owner_lookup(&catalog) {
                Ok(owner_lookup) => Some(owner_lookup),
                Err(error) => {
                    return (
                        false,
                        format!(
                            "real checkpoint NVFP4 inferred placement preflight failed: {error:#}"
                        ),
                        None,
                    )
                }
            }
        }
    };
    let original_catalog_tensors = catalog.tensors.len();
    let catalog = filter_real_weight_serving_catalog(
        catalog,
        serving_owner_lookup.as_ref(),
        args.role_hostname.as_deref(),
        args.real_layer,
        true,
    );
    let (serving_catalog_tensors, routed_expert_tensors) =
        match validate_real_weight_catalog(&catalog, &catalog_source) {
            Ok(counts) => counts,
            Err(error) => {
                return (
                    false,
                    format!("real checkpoint NVFP4 serving catalog preflight failed: {error:#}"),
                    None,
                )
            }
        };
    let mut executor = RealNvfp4ProtocolV2Executor::new(
        catalog,
        args.real_layer.map(|layer_id| layer_id as usize),
        args.role_hostname.clone(),
    );
    executor = executor.with_intermediate_shard(intermediate_shard);
    let resident_preload_plan = match executor.resident_preload_plan() {
        Ok(plan) => plan,
        Err(error) => {
            return (
                false,
                format!("real checkpoint NVFP4 resident preload plan failed: {error:#}"),
                None,
            )
        }
    };
    let resident_preload_plan_report = ExpertDaemonResidentPreloadPlan::from(resident_preload_plan);
    if resident_preload_plan.incomplete_expert_projection_sets > 0
        || resident_preload_plan.missing_metadata_tensors > 0
    {
        return (
            false,
            format!(
                "real checkpoint NVFP4 resident preload plan incomplete projection_groups={} complete_expert_projection_sets={} incomplete_expert_projection_sets={} missing_metadata_tensors={}",
                resident_preload_plan.projection_groups,
                resident_preload_plan.complete_expert_projection_sets,
                resident_preload_plan.incomplete_expert_projection_sets,
                resident_preload_plan.missing_metadata_tensors
            ),
            Some(resident_preload_plan_report),
        );
    }
    let owner_lookup_experts = serving_owner_lookup
        .as_ref()
        .map(ExpertOwnerLookup::len)
        .unwrap_or(0);
    (
        true,
        format!(
            "real checkpoint NVFP4 ProtocolV2 executor configured topology=strict-tp4-replicated-experts shard_count={} shard_rank={} startup_resident_preload=required catalog_tensors={original_catalog_tensors} serving_catalog_tensors={serving_catalog_tensors} routed_expert_tensors={routed_expert_tensors} owner_lookup_experts={owner_lookup_experts} projection_groups={} complete_expert_projection_sets={} weight_bytes={} weight_scale_bytes={} scalar_metadata_bytes={}",
            intermediate_shard.count,
            intermediate_shard.rank,
            resident_preload_plan.projection_groups,
            resident_preload_plan.complete_expert_projection_sets,
            resident_preload_plan.weight_bytes,
            resident_preload_plan.weight_scale_bytes,
            resident_preload_plan.scalar_metadata_bytes
        ),
        Some(resident_preload_plan_report),
    )
}

fn real_tp4_intermediate_shard_from_env() -> Result<ExpertIntermediateShard> {
    require_real_tp4_intermediate_shard(spark_expert_intermediate_shard_from_env()?)
}

fn require_real_tp4_intermediate_shard(
    shard: Option<ExpertIntermediateShard>,
) -> Result<ExpertIntermediateShard> {
    shard.context(
        "real DeepSeek V4 expert serving requires DS41RT_EXPERT_INTERMEDIATE_SHARDS=4 and \
DS41RT_EXPERT_INTERMEDIATE_SHARD_RANK=0..3; logical owner metadata is only for ingestion and \
reduction, never expert placement",
    )
}

fn read_expertd_catalog(path: &Path) -> Result<TensorCatalog> {
    serde_json::from_reader(
        File::open(path)
            .with_context(|| format!("opening expert daemon catalog {}", path.display()))?,
    )
    .with_context(|| format!("parsing expert daemon catalog {}", path.display()))
}

fn resolve_expertd_catalog(args: &ExpertDaemonArgs) -> Result<(String, TensorCatalog)> {
    if let Some(path) = args.catalog.as_deref() {
        let catalog = read_expertd_catalog(path)?;
        anyhow::ensure!(
            catalog.model_id == args.model_id,
            "expert daemon catalog model_id {} does not match requested {}",
            catalog.model_id,
            args.model_id
        );
        validate_production_runtime_catalog(&catalog)?;
        return Ok((path.display().to_string(), catalog));
    }
    Ok((
        format!("hf://{}", args.model_id),
        build_runtime_catalog(&args.model_id)?,
    ))
}

fn validate_real_weight_catalog(catalog: &TensorCatalog, source: &str) -> Result<(usize, usize)> {
    if catalog.tensors.is_empty() {
        anyhow::bail!("expert daemon catalog {source} contains no tensors");
    }
    let snapshot_path = Path::new(&catalog.snapshot_path);
    if !snapshot_path.exists() {
        anyhow::bail!(
            "expert daemon catalog {} snapshot path {} does not exist",
            source,
            snapshot_path.display()
        );
    }
    let routed_expert_tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::RoutedExpert)
        .count();
    if routed_expert_tensors == 0 {
        anyhow::bail!(
            "expert daemon catalog {} contains no routed expert tensors",
            source
        );
    }
    Ok((catalog.tensors.len(), routed_expert_tensors))
}

fn filter_real_weight_serving_catalog(
    mut catalog: TensorCatalog,
    owner_lookup: Option<&ExpertOwnerLookup>,
    role_hostname: Option<&str>,
    real_layer: Option<u32>,
    serve_all_experts: bool,
) -> TensorCatalog {
    let routed_experts = catalog.facts.routed_experts;
    catalog.tensors.retain(|tensor| {
        let routed_expert = tensor.role == TensorRole::RoutedExpert
            && real_layer
                .map(|layer_id| tensor.layer_id == Some(layer_id))
                .unwrap_or(true)
            && (serve_all_experts
                || tensor_served_by_role(tensor, routed_experts, owner_lookup, role_hostname));
        routed_expert
    });
    catalog
}

fn tensor_served_by_role(
    tensor: &TensorInfo,
    routed_experts: usize,
    owner_lookup: Option<&ExpertOwnerLookup>,
    role_hostname: Option<&str>,
) -> bool {
    let Some(role_hostname) = role_hostname else {
        return true;
    };
    let (Some(layer_id), Some(expert_id)) = (tensor.layer_id, tensor.expert_id) else {
        return false;
    };
    let owner = owner_lookup
        .and_then(|lookup| {
            lookup
                .owner_for(layer_id as usize, expert_id as usize)
                .map(str::to_owned)
        })
        .or_else(|| {
            let hosts = EXPERT_HOSTS
                .iter()
                .map(|host| (*host).to_owned())
                .collect::<Vec<_>>();
            owner_for_expert(
                layer_id as usize,
                expert_id as usize,
                routed_experts,
                &hosts,
                PlacementPolicy::Modulo,
            )
        });
    owner
        .as_deref()
        .map(|owner| host_matches(owner, role_hostname))
        .unwrap_or(false)
}

fn host_matches(assignment_owner: &str, requested_owner: &str) -> bool {
    assignment_owner == requested_owner
        || assignment_owner.split('.').next() == Some(requested_owner)
        || requested_owner.split('.').next() == Some(assignment_owner)
}

#[cfg(test)]
mod tests {
    use super::{
        expertd_preflight_report, expertd_preflight_report_with_intermediate_shard,
        filter_real_weight_serving_catalog, require_real_tp4_intermediate_shard, run_expertd,
        ExpertDaemonPreflightReport,
    };
    use crate::cli::ExpertDaemonArgs;
    use crate::commands::real_full::{
        real_nvfp4_cuda_reference_kernels_enabled, ExpertIntermediateShard,
        REAL_NVFP4_PROTOCOL_V2_EXECUTOR,
    };
    use ds41rt_core::{
        DType, ExpertOwnerLookup, ModelFacts, TensorAssignment, TensorCatalog, TensorInfo,
        TensorRole,
    };
    use ds41rt_transport::{
        expert_protocol_v2_compact_id, tcp_protocol_v2_roundtrip, ExpertProtocolV2Request,
        ExpertProtocolV2ResponseView, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
        ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig,
    };
    use std::{
        fs::File,
        io::Write,
        net::{SocketAddr, TcpListener as StdTcpListener},
        path::{Path, PathBuf},
    };
    use tokio::{
        net::TcpStream,
        time::{sleep, Duration},
    };

    fn expertd_tp4_preflight_report(args: &ExpertDaemonArgs) -> ExpertDaemonPreflightReport {
        expertd_preflight_report_with_intermediate_shard(
            args,
            Some(ExpertIntermediateShard::new(4, 0).unwrap()),
        )
        .expect("strict TP4 expert-daemon preflight report")
    }

    #[test]
    fn real_weight_serving_rejects_owner_partition_fallback() {
        let error = require_real_tp4_intermediate_shard(None)
            .expect_err("real expert serving must reject missing TP4 sharding")
            .to_string();

        assert!(error.contains("requires DS41RT_EXPERT_INTERMEDIATE_SHARDS=4"));
        assert!(error.contains("never expert placement"));
    }

    #[test]
    fn tcp_preflight_reports_protocol_v2_default_frame_protocol() {
        let args = ExpertDaemonArgs {
            synthetic_weights: true,
            preflight_only: true,
            transport: "tcp".to_owned(),
            listen: "127.0.0.1:9154".to_owned(),
            loadplan: None,
            catalog: None,
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: None,
            role_hostname: None,
        };

        let report = expertd_preflight_report(&args).expect("tcp preflight report");

        assert_eq!(report.status, "ready");
        assert_eq!(report.transport, "tcp");
        assert_eq!(
            report.frame_protocol.as_deref(),
            Some(ds41rt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL)
        );
    }

    #[test]
    fn tcp_debug_json_preflight_is_explicit_legacy_protocol() {
        let args = ExpertDaemonArgs {
            synthetic_weights: true,
            preflight_only: true,
            transport: "tcp-debug-json".to_owned(),
            listen: "127.0.0.1:9155".to_owned(),
            loadplan: None,
            catalog: None,
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: None,
            role_hostname: None,
        };

        let report = expertd_preflight_report(&args).expect("tcp debug JSON preflight report");

        assert_eq!(report.status, "ready");
        assert_eq!(report.transport, "tcp-debug-json");
        assert_eq!(
            report.frame_protocol.as_deref(),
            Some(ds41rt_transport::DEBUG_JSON_FRAME_PROTOCOL)
        );
    }

    #[test]
    fn verbs_host_preflight_reports_protocol_v2_without_starting_service() {
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: true,
            transport: "verbs-host".to_owned(),
            listen: "127.0.0.1:9153".to_owned(),
            loadplan: None,
            catalog: None,
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: None,
            role_hostname: None,
        };

        let report = expertd_preflight_report(&args).expect("verbs-host preflight report");

        assert_eq!(report.transport, "verbs-host");
        assert_eq!(report.status, "blocked");
        assert!(report.supports_rdma);
        assert!(report.supports_host_registered_buffers);
        assert!(report.app_transport_implemented);
        assert_eq!(
            report.app_transport_status,
            ds41rt_transport::VERBS_HOST_APP_TRANSPORT_STATUS
        );
        assert_eq!(report.infiniband_path.as_deref(), Some("/dev/infiniband"));
        assert_eq!(
            report.frame_protocol.as_deref(),
            Some(ds41rt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL)
        );
    }

    #[test]
    fn tcp_real_weight_preflight_reports_unresolvable_model_catalog() {
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: true,
            transport: "tcp".to_owned(),
            listen: "127.0.0.1:9160".to_owned(),
            loadplan: None,
            catalog: None,
            model_id: "unsupported/checkpoint".to_owned(),
            real_layer: Some(3),
            role_hostname: Some("spark-0".to_owned()),
        };

        let report = expertd_tp4_preflight_report(&args);

        assert_eq!(report.status, "blocked");
        assert!(!report.real_weight_serving_ready);
        assert!(report.real_weight_serving_status.contains("catalog"));
        assert!(report
            .real_weight_serving_status
            .contains("unsupported/checkpoint"));
        assert_eq!(
            report.frame_protocol.as_deref(),
            Some(ds41rt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL)
        );
    }

    #[test]
    fn tcp_real_weight_preflight_validates_catalog_json() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog_path = tempdir.path().join("bad-catalog.json");
        std::fs::write(&catalog_path, "{}").unwrap();
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: true,
            transport: "tcp".to_owned(),
            listen: "127.0.0.1:9161".to_owned(),
            loadplan: None,
            catalog: Some(catalog_path),
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: Some(3),
            role_hostname: Some("spark-0".to_owned()),
        };

        let report = expertd_tp4_preflight_report(&args);

        assert_eq!(report.status, "blocked");
        assert!(!report.real_weight_serving_ready);
        assert!(report.real_weight_serving_status.contains("parsing"));
    }

    #[test]
    fn tcp_real_weight_preflight_reports_ready_with_parsed_catalog() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog_path = write_tiny_real_weight_serving_catalog(tempdir.path());
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: true,
            transport: "tcp".to_owned(),
            listen: "127.0.0.1:9162".to_owned(),
            loadplan: None,
            catalog: Some(catalog_path),
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: Some(3),
            role_hostname: Some("spark-0".to_owned()),
        };

        let report = expertd_tp4_preflight_report(&args);

        assert_eq!(report.status, "ready");
        assert!(report.real_weight_serving_ready);
        assert!(report
            .real_weight_serving_status
            .contains("catalog_tensors=12"));
        assert!(report
            .real_weight_serving_status
            .contains("routed_expert_tensors=12"));
        assert!(report
            .real_weight_serving_status
            .contains("projection_groups=3"));
        let plan = report
            .real_weight_resident_preload_plan
            .expect("ready real-weight preflight should include resident preload plan");
        assert!(plan.startup_required);
        assert_eq!(plan.projection_groups, 3);
        assert_eq!(plan.layers, 1);
        assert_eq!(plan.experts, 1);
        assert_eq!(plan.complete_expert_projection_sets, 1);
        assert_eq!(plan.incomplete_expert_projection_sets, 0);
        assert_eq!(plan.missing_metadata_tensors, 0);
        assert_eq!(plan.weight_bytes, 4);
        assert_eq!(plan.weight_scale_bytes, 4);
        assert_eq!(plan.scalar_metadata_bytes, 24);
    }

    #[test]
    fn tcp_real_weight_preflight_blocks_incomplete_resident_preload_plan() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog_path = write_tiny_real_weight_catalog(tempdir.path());
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: true,
            transport: "tcp".to_owned(),
            listen: "127.0.0.1:9164".to_owned(),
            loadplan: None,
            catalog: Some(catalog_path),
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: Some(3),
            role_hostname: Some("spark-0".to_owned()),
        };

        let report = expertd_tp4_preflight_report(&args);

        assert_eq!(report.status, "blocked");
        assert!(!report.real_weight_serving_ready);
        assert!(report
            .real_weight_serving_status
            .contains("resident preload plan incomplete"));
        let plan = report
            .real_weight_resident_preload_plan
            .expect("blocked preload plan should still be reported");
        assert_eq!(plan.projection_groups, 1);
        assert_eq!(plan.complete_expert_projection_sets, 0);
        assert_eq!(plan.incomplete_expert_projection_sets, 1);
        assert_eq!(plan.missing_metadata_tensors, 3);
    }

    #[test]
    fn tcp_real_weight_preflight_validates_loadplan_owner_lookup() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog_path = write_tiny_real_weight_catalog(tempdir.path());
        let loadplan_path = tempdir.path().join("bad-loadplan.json");
        std::fs::write(&loadplan_path, "{}").unwrap();
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: true,
            transport: "tcp".to_owned(),
            listen: "127.0.0.1:9163".to_owned(),
            loadplan: Some(loadplan_path),
            catalog: Some(catalog_path),
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: Some(3),
            role_hostname: Some("spark-0".to_owned()),
        };

        let report = expertd_tp4_preflight_report(&args);

        assert_eq!(report.status, "blocked");
        assert!(!report.real_weight_serving_ready);
        assert!(report
            .real_weight_serving_status
            .contains("owner lookup preflight failed"));
    }

    #[test]
    fn real_weight_serving_catalog_filters_role_and_layer_from_owner_lookup() {
        let catalog = TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: "/tmp/ds41rt-filtered-serving-catalog".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![
                serving_filter_tensor("model.layers.3.mlp.experts.0.gate_proj.weight", 3, 0),
                serving_filter_tensor("model.layers.3.mlp.experts.0.gate_proj.weight_scale", 3, 0),
                serving_filter_tensor("model.layers.3.mlp.experts.1.gate_proj.weight", 3, 1),
                serving_filter_tensor("model.layers.4.mlp.experts.0.gate_proj.weight", 4, 0),
                TensorInfo {
                    name: "model.layers.3.mlp.shared_experts.down_proj.weight".to_owned(),
                    file: "expert.bin".to_owned(),
                    dtype: DType::Bf16,
                    shape: vec![1, 1],
                    byte_offset: 4,
                    byte_length: 1,
                    role: TensorRole::SharedExpert,
                    layer_id: Some(3),
                    expert_id: None,
                    is_quantization_metadata: false,
                },
            ],
        };
        let owner_lookup = ExpertOwnerLookup::from_pairs([
            ((3, 0), "ostrich".to_owned()),
            ((3, 1), "dodo".to_owned()),
            ((4, 0), "ostrich".to_owned()),
        ]);

        let filtered = filter_real_weight_serving_catalog(
            catalog.clone(),
            Some(&owner_lookup),
            Some("ostrich"),
            Some(3),
            false,
        );

        assert_eq!(
            filtered
                .tensors
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "model.layers.3.mlp.experts.0.gate_proj.weight",
                "model.layers.3.mlp.experts.0.gate_proj.weight_scale"
            ]
        );
    }

    #[test]
    fn tp4_serving_keeps_every_expert_independent_of_logical_owner() {
        let catalog = TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: "/tmp/ds41rt-tp4-serving-catalog".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![
                serving_filter_tensor("model.layers.3.mlp.experts.0.gate_proj.weight", 3, 0),
                serving_filter_tensor("model.layers.3.mlp.experts.1.gate_proj.weight", 3, 1),
                serving_filter_tensor("model.layers.4.mlp.experts.0.gate_proj.weight", 4, 0),
            ],
        };
        let owner_lookup = ExpertOwnerLookup::from_pairs([
            ((3, 0), "ostrich".to_owned()),
            ((3, 1), "dodo".to_owned()),
            ((4, 0), "ostrich".to_owned()),
        ]);

        let filtered = filter_real_weight_serving_catalog(
            catalog,
            Some(&owner_lookup),
            Some("ostrich"),
            Some(3),
            true,
        );

        assert_eq!(
            filtered
                .tensors
                .iter()
                .map(|tensor| (tensor.layer_id, tensor.expert_id))
                .collect::<Vec<_>>(),
            vec![(Some(3), Some(0)), (Some(3), Some(1))]
        );
    }

    #[tokio::test]
    async fn tcp_real_weight_entrypoint_refuses_cpu_fallback_without_cuda_env() {
        if real_nvfp4_cuda_reference_kernels_enabled() {
            eprintln!(
                "skipping CPU-fallback refusal test because CUDA reference kernels are enabled"
            );
            return;
        }
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: false,
            transport: "tcp".to_owned(),
            listen: "127.0.0.1:9165".to_owned(),
            loadplan: None,
            catalog: None,
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: Some(3),
            role_hostname: Some("spark-0".to_owned()),
        };

        let error = run_expertd(args).await.unwrap_err().to_string();

        assert!(error.contains("requires DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS=1"));
        assert!(error.contains("refusing CPU fallback"));
    }

    #[tokio::test]
    async fn tcp_real_weight_entrypoint_serves_tiny_protocol_v2_request() {
        if !real_nvfp4_cuda_reference_kernels_enabled() {
            eprintln!(
                "skipping real-weight expertd TCP serving test because CUDA reference kernels are disabled"
            );
            return;
        }
        let tempdir = tempfile::tempdir().unwrap();
        let catalog_path = write_tiny_real_weight_serving_catalog(tempdir.path());
        let addr = unused_loopback_addr();
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: false,
            transport: "tcp".to_owned(),
            listen: addr.to_string(),
            loadplan: None,
            catalog: Some(catalog_path),
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: Some(3),
            role_hostname: Some("spark-0".to_owned()),
        };
        let server = tokio::spawn(async move { run_expertd(args).await });
        wait_for_tcp_listener(addr).await;

        let request = tiny_protocol_v2_request().with_debug_checksum();
        let response = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default())
            .await
            .expect("round-tripping tiny real-weight ProtocolV2 request through expertd");

        assert_eq!(response.header.request_id, request.header.request_id);
        assert_eq!(response.header.layer_id, request.header.layer_id);
        assert_eq!(response.header.output_dim, request.header.hidden_dim);
        assert_eq!(
            response.header.executor_id,
            expert_protocol_v2_compact_id(REAL_NVFP4_PROTOCOL_V2_EXECUTOR)
        );
        assert_eq!(response.partial_output_payload.len(), 4);
        let encoded = response.encode().unwrap();
        ExpertProtocolV2ResponseView::parse(&encoded)
            .unwrap()
            .verify_checksum()
            .unwrap();
        let output = bf16_values(&response.partial_output_payload);
        assert!(output.iter().all(|value| value.is_finite()));
        let output_checksum = output.iter().map(|value| *value as f64).sum::<f64>();
        eprintln!(
            "expertd_real_weight_tcp_entrypoint executor=protocol-v2-real-nvfp4-checkpoint-executor layer={} expert=0 hidden_dim={} output_values={} output_checksum={output_checksum}",
            request.header.layer_id,
            request.header.hidden_dim,
        output.len()
        );
        server.abort();
    }

    #[tokio::test]
    async fn tcp_real_weight_entrypoint_serves_tiny_protocol_v2_request_with_loadplan() {
        if !real_nvfp4_cuda_reference_kernels_enabled() {
            eprintln!(
                "skipping loadplan-backed real-weight expertd TCP serving test because CUDA reference kernels are disabled"
            );
            return;
        }
        let tempdir = tempfile::tempdir().unwrap();
        let catalog_path = write_tiny_real_weight_serving_catalog(tempdir.path());
        let loadplan_path = write_tiny_real_weight_serving_loadplan(tempdir.path());
        let addr = unused_loopback_addr();
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: false,
            transport: "tcp".to_owned(),
            listen: addr.to_string(),
            loadplan: Some(loadplan_path),
            catalog: Some(catalog_path),
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: Some(3),
            role_hostname: Some("ostrich".to_owned()),
        };
        let server = tokio::spawn(async move { run_expertd(args).await });
        wait_for_tcp_listener(addr).await;

        let request = tiny_protocol_v2_request().with_debug_checksum();
        let response = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default())
            .await
            .expect("round-tripping tiny loadplan-backed real-weight ProtocolV2 request through expertd");

        assert_eq!(response.header.request_id, request.header.request_id);
        assert_eq!(response.header.layer_id, request.header.layer_id);
        assert_eq!(response.header.output_dim, request.header.hidden_dim);
        assert_eq!(
            response.header.executor_id,
            expert_protocol_v2_compact_id(REAL_NVFP4_PROTOCOL_V2_EXECUTOR)
        );
        assert_eq!(response.partial_output_payload.len(), 4);
        let output = bf16_values(&response.partial_output_payload);
        assert!(output.iter().all(|value| value.is_finite()));
        let output_checksum = output.iter().map(|value| *value as f64).sum::<f64>();
        eprintln!(
            "expertd_real_weight_loadplan_tcp_entrypoint executor=protocol-v2-real-nvfp4-checkpoint-executor layer={} expert=0 hidden_dim={} output_values={} output_checksum={output_checksum}",
            request.header.layer_id,
            request.header.hidden_dim,
            output.len()
        );
        server.abort();
    }

    fn write_tiny_real_weight_catalog(root: &Path) -> PathBuf {
        let catalog = TensorCatalog {
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            snapshot_path: root.display().to_string(),
            facts: ModelFacts::default(),
            tensors: vec![TensorInfo {
                name: "model.layers.3.mlp.experts.0.gate_proj.weight".to_owned(),
                file: "expert.bin".to_owned(),
                dtype: DType::U8,
                shape: vec![4, 1],
                byte_offset: 0,
                byte_length: 4,
                role: TensorRole::RoutedExpert,
                layer_id: Some(3),
                expert_id: Some(0),
                is_quantization_metadata: false,
            }],
        };
        File::create(root.join("expert.bin")).unwrap();
        let catalog_path = root.join("model_catalog.json");
        serde_json::to_writer(File::create(&catalog_path).unwrap(), &catalog).unwrap();
        catalog_path
    }

    fn write_tiny_real_weight_serving_catalog(root: &Path) -> PathBuf {
        let shard_path = root.join("expert.bin");
        let mut shard_bytes = Vec::new();
        let mut tensors = Vec::new();
        for projection in ["gate_proj", "up_proj"] {
            push_serving_tensor(
                &mut shard_bytes,
                &mut tensors,
                projection,
                "weight",
                DType::U8,
                vec![4, 1],
                &[0xaa; 4],
            );
            push_serving_tensor(
                &mut shard_bytes,
                &mut tensors,
                projection,
                "weight_scale",
                DType::F8E4M3,
                vec![4, 1],
                &[0x38; 4],
            );
            push_serving_tensor(
                &mut shard_bytes,
                &mut tensors,
                projection,
                "input_scale",
                DType::F32,
                Vec::new(),
                &1.0_f32.to_le_bytes(),
            );
            push_serving_tensor(
                &mut shard_bytes,
                &mut tensors,
                projection,
                "weight_scale_2",
                DType::F32,
                Vec::new(),
                &1.0_f32.to_le_bytes(),
            );
        }
        push_serving_tensor(
            &mut shard_bytes,
            &mut tensors,
            "down_proj",
            "weight",
            DType::U8,
            vec![2, 4],
            &[0x0a; 8],
        );
        push_serving_tensor(
            &mut shard_bytes,
            &mut tensors,
            "down_proj",
            "weight_scale",
            DType::F8E4M3,
            vec![2, 4],
            &[0x38; 8],
        );
        push_serving_tensor(
            &mut shard_bytes,
            &mut tensors,
            "down_proj",
            "input_scale",
            DType::F32,
            Vec::new(),
            &1.0_f32.to_le_bytes(),
        );
        push_serving_tensor(
            &mut shard_bytes,
            &mut tensors,
            "down_proj",
            "weight_scale_2",
            DType::F32,
            Vec::new(),
            &1.0_f32.to_le_bytes(),
        );
        File::create(&shard_path)
            .unwrap()
            .write_all(&shard_bytes)
            .unwrap();
        let catalog = TensorCatalog {
            model_id: ds41rt_core::DEFAULT_MODEL_ID.to_owned(),
            snapshot_path: root.display().to_string(),
            facts: ModelFacts::default(),
            tensors,
        };
        let catalog_path = root.join("model_catalog.json");
        serde_json::to_writer(File::create(&catalog_path).unwrap(), &catalog).unwrap();
        catalog_path
    }

    fn push_serving_tensor(
        shard_bytes: &mut Vec<u8>,
        tensors: &mut Vec<TensorInfo>,
        projection: &str,
        suffix: &str,
        dtype: DType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) {
        let byte_offset = shard_bytes.len() as u64;
        shard_bytes.extend_from_slice(bytes);
        tensors.push(TensorInfo {
            name: format!("model.layers.3.mlp.experts.0.{projection}.{suffix}"),
            file: "expert.bin".to_owned(),
            dtype,
            shape,
            byte_offset,
            byte_length: bytes.len() as u64,
            role: TensorRole::RoutedExpert,
            layer_id: Some(3),
            expert_id: Some(0),
            is_quantization_metadata: suffix != "weight",
        });
    }

    fn serving_filter_tensor(name: &str, layer_id: u32, expert_id: u32) -> TensorInfo {
        TensorInfo {
            name: name.to_owned(),
            file: "expert.bin".to_owned(),
            dtype: DType::U8,
            shape: vec![1, 1],
            byte_offset: 0,
            byte_length: 1,
            role: TensorRole::RoutedExpert,
            layer_id: Some(layer_id),
            expert_id: Some(expert_id),
            is_quantization_metadata: name.ends_with("weight_scale"),
        }
    }

    fn write_tiny_real_weight_serving_loadplan(root: &Path) -> PathBuf {
        #[derive(serde::Serialize)]
        struct TinyNodeLoadPlan {
            model_id: String,
            placement_version: String,
            owner: Option<String>,
            assignments: Vec<TensorAssignment>,
        }
        let assignments = ["gate_proj", "up_proj", "down_proj"]
            .into_iter()
            .map(|projection| TensorAssignment {
                tensor_name: format!("model.layers.3.mlp.experts.0.{projection}.weight"),
                owner: "ostrich".to_owned(),
                role: TensorRole::RoutedExpert,
                layer_id: Some(3),
                expert_id: Some(0),
            })
            .collect::<Vec<_>>();
        let plan = TinyNodeLoadPlan {
            model_id: "test/model".to_owned(),
            placement_version: "phase0-tiny-serving-loadplan".to_owned(),
            owner: Some("ostrich".to_owned()),
            assignments,
        };
        let loadplan_path = root.join("loadplan.json");
        serde_json::to_writer(File::create(&loadplan_path).unwrap(), &plan).unwrap();
        loadplan_path
    }

    fn tiny_protocol_v2_request() -> ExpertProtocolV2Request {
        let rows = vec![ExpertProtocolV2RowDescriptor {
            row_id: 0,
            source_kind: ExpertV2SourceKind::Decode,
            source_request_id: 1,
            token_position: 0,
            route_offset: 0,
            route_count: 1,
        }];
        let routes = vec![ExpertProtocolV2RouteEntry {
            row_index: 0,
            expert_id: 0,
            gate_weight: 1.0,
        }];
        let mut hidden_payload = Vec::new();
        for value in [1.0_f32, 2.0] {
            hidden_payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        ExpertProtocolV2Request::new(
            501,
            0x51CE,
            3,
            2,
            ExpertV2Dtype::Bf16,
            rows,
            routes,
            hidden_payload,
        )
        .unwrap()
    }

    fn bf16_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect()
    }

    fn unused_loopback_addr() -> SocketAddr {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    }

    async fn wait_for_tcp_listener(addr: SocketAddr) {
        for _ in 0..100 {
            if TcpStream::connect(addr).await.is_ok() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for expertd TCP listener at {addr}");
    }
}
