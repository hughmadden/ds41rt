use anyhow::{Context, Result};
use ds4rt_core::{KvCacheAllocator, KvCacheConfig, KvCacheDType, ModelFacts};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::cli::CoordinatorArgs;
use crate::commands::real_full::{load_real_full_serving, run_real_ds4_full_preflight};
use crate::python_graph_capture::{
    finish_coordinator_python_capture_startup, initialize_coordinator_python_capture_from_env,
    PythonCaptureStatus,
};

type CoordinatorPythonCaptureTask = JoinHandle<Result<Option<PythonCaptureStatus>>>;

fn coordinator_backend(value: &str) -> Result<ds4rt_api::ApiBackend> {
    match value {
        "tiny" | "synthetic-ds4-layer" | "real-ds4-full" => {
            Ok(ds4rt_api::ApiBackend::parse(value).expect("matched coordinator backend parses"))
        }
        "real-ds4-slice" => anyhow::bail!(
            "real-ds4-slice coordinator probes were superseded by real-ds4-full execution stepper coverage"
        ),
        other => anyhow::bail!("unsupported coordinator backend: {other}"),
    }
}

fn finish_python_capture_task(
    task: &mut Option<CoordinatorPythonCaptureTask>,
    status: &mut Option<PythonCaptureStatus>,
    started: Instant,
) -> Result<()> {
    let join_started = Instant::now();
    let result = task
        .take()
        .context("coordinator Python graph-capture initialization task is missing")?
        .join()
        .map_err(|_| anyhow::anyhow!("coordinator Python graph-capture initialization panicked"))?
        .context("initializing coordinator Python graph-capture bridge")?;
    *status = result;
    eprintln!(
        "coordinator_startup_python_capture total_ms={:.3} barrier_wait_ms={:.3}",
        started.elapsed().as_secs_f64() * 1_000.0,
        join_started.elapsed().as_secs_f64() * 1_000.0,
    );
    Ok(())
}

pub(crate) async fn run_coordinator(args: CoordinatorArgs) -> Result<()> {
    let startup_started = Instant::now();
    if args.backend == "real-ds4-full" && args.preflight_only {
        return run_real_ds4_full_preflight(&args);
    }
    let backend = coordinator_backend(&args.backend)?;
    let transport = ds4rt_api::ApiTransport::parse(&args.transport)
        .ok_or_else(|| anyhow::anyhow!("unsupported coordinator transport: {}", args.transport))?;
    let python_capture_started = Instant::now();
    let mut python_capture_task = Some(
        std::thread::Builder::new()
            .name("ds4rt-python-capture-init".to_owned())
            .spawn(initialize_coordinator_python_capture_from_env)
            .context("spawning coordinator Python graph-capture initialization")?,
    );
    let mut python_capture = None;
    let expert_targets = args
        .expert_hosts
        .split(',')
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let serving_started = Instant::now();
    let real_full_serving = if backend == ds4rt_api::ApiBackend::RealDs4Full {
        Some(load_real_full_serving(&args, || {
            if python_capture_task.is_some() {
                finish_python_capture_task(
                    &mut python_capture_task,
                    &mut python_capture,
                    python_capture_started,
                )?;
            }
            Ok(())
        })?)
    } else {
        if python_capture_task.is_some() {
            finish_python_capture_task(
                &mut python_capture_task,
                &mut python_capture,
                python_capture_started,
            )?;
        }
        None
    };
    eprintln!(
        "coordinator_startup_phase stage=real-full-serving elapsed_ms={:.3} total_ms={:.3}",
        serving_started.elapsed().as_secs_f64() * 1_000.0,
        startup_started.elapsed().as_secs_f64() * 1_000.0,
    );
    finish_coordinator_python_capture_startup();
    let coordinator_kv_config = match real_full_serving.as_ref() {
        Some(serving) => serving.kv_config.clone(),
        None => coordinator_kv_cache_config(args.max_context_tokens, &args.kv_cache_dtype)?,
    };
    let api_config = ds4rt_api::ApiConfig {
        backend,
        transport,
        model_id: args.model_id.clone(),
        expert_targets,
        real_slice: None,
        real_full: real_full_serving
            .as_ref()
            .map(|serving| serving.info.clone()),
        real_full_executor: real_full_serving.map(|serving| serving.executor),
    };
    let api_started = Instant::now();
    let kv_allocator = KvCacheAllocator::new(coordinator_kv_config);
    let kv_snapshot = kv_allocator.snapshot();
    let listener = tokio::net::TcpListener::bind(&args.listen)
        .await
        .with_context(|| format!("binding coordinator API to {}", args.listen))?;
    eprintln!(
        "coordinator_startup_phase stage=api-bind elapsed_ms={:.3} total_ms={:.3}",
        api_started.elapsed().as_secs_f64() * 1_000.0,
        startup_started.elapsed().as_secs_f64() * 1_000.0,
    );
    println!(
        "starting coordinator backend={} transport={} model_id={} expert_hosts={} listen={}",
        args.backend, args.transport, args.model_id, args.expert_hosts, args.listen
    );
    if let Some(status) = python_capture.as_ref() {
        println!(
            "python_graph_capture status=enabled gate={} modules={}",
            status.gate_env,
            status.imported_modules.join(",")
        );
    }
    println!(
        "kv_cache layout={:?} dtype={:?} layers={} key_value_width={} dsa_indexer_layers={} dsa_index_head_dim={} max_tokens={} bytes_per_token={} capacity_bytes={}",
        kv_snapshot.config.layout,
        kv_snapshot.config.dtype,
        kv_snapshot.config.layers,
        kv_snapshot.config.key_value_width,
        kv_snapshot.config.dsa_indexer_layers,
        kv_snapshot.config.dsa_index_head_dim,
        kv_snapshot.config.max_tokens,
        kv_snapshot.bytes_per_token,
        kv_snapshot.capacity_bytes
    );
    axum::serve(listener, ds4rt_api::router_with_config(api_config)).await?;
    Ok(())
}

fn coordinator_kv_cache_config(
    max_context_tokens: usize,
    kv_cache_dtype: &str,
) -> Result<KvCacheConfig> {
    anyhow::ensure!(
        max_context_tokens > 0,
        "coordinator --max-context-tokens must be a positive integer"
    );
    let dtype = KvCacheDType::parse_cache_dtype(kv_cache_dtype).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported coordinator --kv-cache-dtype {}; expected bf16",
            kv_cache_dtype
        )
    })?;
    anyhow::ensure!(
        dtype == KvCacheDType::Bf16,
        "non-serving DeepSeek V4 coordinator backends support only bf16 KV, got {}",
        dtype.label()
    );
    KvCacheConfig::try_deepseek_v4_hybrid_bf16(max_context_tokens, &ModelFacts::default())
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::{coordinator_backend, coordinator_kv_cache_config};
    use ds4rt_core::{KvCacheDType, KvLayout, MlaKvCacheRepresentation, ModelFacts};

    #[test]
    fn diagnostic_coordinator_uses_flash_hybrid_kv_geometry() {
        let facts = ModelFacts::default();
        let config = coordinator_kv_cache_config(4096, "bf16").unwrap();

        assert_eq!(config.layout, KvLayout::DeepseekV4HybridBf16);
        assert_eq!(config.dtype, KvCacheDType::Bf16);
        assert_eq!(config.layers, facts.num_hidden_layers);
        assert_eq!(config.key_value_width, facts.head_dim);
        assert_eq!(
            config.mla_representation,
            MlaKvCacheRepresentation::NormalizedRotated
        );
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn diagnostic_coordinator_rejects_non_bf16_and_zero_capacity() {
        assert!(coordinator_kv_cache_config(4096, "fp8").is_err());
        assert!(coordinator_kv_cache_config(4096, "nvfp4").is_err());
        assert!(coordinator_kv_cache_config(0, "bf16").is_err());
    }

    #[test]
    fn coordinator_rejects_legacy_cuda_reference_serving() {
        assert!(coordinator_backend("real-ds4-full").is_ok());
        let error = coordinator_backend("cuda-reference").unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported coordinator backend: cuda-reference"));
    }
}
