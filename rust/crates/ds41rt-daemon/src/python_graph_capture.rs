use anyhow::{Context, Result};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
#[cfg(test)]
use std::cell::Cell;
use std::env;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub(crate) const DS41RT_B12X_ENV: &str = "DS41RT_B12X";
pub(crate) const DS41RT_B12X_SPARK_ENV: &str = "DS41RT_B12X_SPARK_PYTHON_CAPTURE";

const COORDINATOR_PYTHON_CAPTURE_MODULES: &[&str] = &[
    "b12x",
    "flashinfer",
    "triton",
    "b12x_mla_capture",
    "deepseek_v4_attention_capture",
    "deepseek_v4_attention_compressor_capture",
    "deepseek_v4_attention_producer_capture",
    "deepseek_v4_attention_output_capture",
    "deepseek_v4_attention_layer_capture",
    "deepseek_v4_mhc_capture",
    "deepseek_v4_sparse_block_capture",
    "deepseek_v4_dspark_capture",
    "deepseek_v4_spark_prefill_capture",
    "deepseek_v4_spark_rank_capture",
    "triton_mlp_capture",
    "triton_router_capture",
    "triton_sampling_capture",
    "triton_kv_pack_capture",
];
const SPARK_PYTHON_CAPTURE_MODULES: &[&str] = &["b12x_spark_capture"];
const DEEPSEEK_V4_ATTENTION_CAPTURE_MODULE: &str = "deepseek_v4_attention_capture";
const DEEPSEEK_V4_ATTENTION_QUALIFY_FUNCTION: &str = "qualify_deepseek_v4_compressed_mla_contract";
const DEEPSEEK_V4_ATTENTION_COMPRESSOR_CAPTURE_MODULE: &str =
    "deepseek_v4_attention_compressor_capture";
const DEEPSEEK_V4_ATTENTION_COMPRESSOR_QUALIFY_FUNCTION: &str =
    "qualify_deepseek_v4_attention_compressor_contract";
const DEEPSEEK_V4_ATTENTION_PRODUCER_CAPTURE_MODULE: &str =
    "deepseek_v4_attention_producer_capture";
const DEEPSEEK_V4_ATTENTION_PRODUCER_QUALIFY_FUNCTION: &str =
    "qualify_deepseek_v4_attention_producer_contract";
const DEEPSEEK_V4_ATTENTION_INDEXER_QUALIFY_FUNCTION: &str =
    "qualify_deepseek_v4_attention_indexer_contract";
const DEEPSEEK_V4_ATTENTION_OUTPUT_CAPTURE_MODULE: &str = "deepseek_v4_attention_output_capture";
const DEEPSEEK_V4_ATTENTION_OUTPUT_QUALIFY_FUNCTION: &str =
    "qualify_deepseek_v4_attention_output_contract";
const DEEPSEEK_V4_ATTENTION_LAYER_CAPTURE_MODULE: &str = "deepseek_v4_attention_layer_capture";
const DEEPSEEK_V4_ATTENTION_LAYER_QUALIFY_FUNCTION: &str =
    "qualify_deepseek_v4_attention_layer_contract";
const DEEPSEEK_V4_MHC_CAPTURE_MODULE: &str = "deepseek_v4_mhc_capture";
const DEEPSEEK_V4_MHC_QUALIFY_FUNCTION: &str = "qualify_deepseek_v4_mhc_contract";
const DEEPSEEK_V4_SPARSE_BLOCK_CAPTURE_MODULE: &str = "deepseek_v4_sparse_block_capture";
const DEEPSEEK_V4_SPARSE_BLOCK_QUALIFY_FUNCTION: &str = "qualify_deepseek_v4_sparse_block_contract";
const DEEPSEEK_V4_DSPARK_CAPTURE_MODULE: &str = "deepseek_v4_dspark_capture";
const DEEPSEEK_V4_DSPARK_QUALIFY_FUNCTION: &str = "qualify_deepseek_v4_dspark_contract";
const DEEPSEEK_V4_SPARK_RANK_CAPTURE_MODULE: &str = "deepseek_v4_spark_rank_capture";
const DEEPSEEK_V4_SPARK_RANK_QUALIFY_FUNCTION: &str =
    "qualify_deepseek_v4_flash_spark_rank_decode_m1_contract";
const DEEPSEEK_V4_SPARK_PREFILL_ROUTE_PACK_QUALIFY_FUNCTION: &str =
    "qualify_deepseek_v4_flash_spark_prefill_route_pack_contract";
const DEEPSEEK_V4_SPARK_PREFILL_CAPTURE_MODULE: &str = "deepseek_v4_spark_prefill_capture";
const DEEPSEEK_V4_SPARK_PREFILL_QUALIFY_FUNCTION: &str =
    "qualify_deepseek_v4_flash_spark_prefill_contract";
static COORDINATOR_PYTHON_CAPTURE_STARTUP_OPEN: AtomicBool = AtomicBool::new(true);
static DS41RT_PYTHON_REFERENCE_PATH_READY: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
thread_local! {
    static COORDINATOR_PYTHON_CAPTURE_TEST_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PythonCaptureStatus {
    pub(crate) gate_env: &'static str,
    pub(crate) imported_modules: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PythonDeviceBufferArg<'a> {
    pub(crate) name: &'a str,
    pub(crate) ptr: *mut c_void,
    pub(crate) bytes: usize,
    pub(crate) device_id: i32,
    pub(crate) flags: u64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum PythonKernelArg<'a> {
    Bool(bool),
    F64(f64),
    I64(i64),
    Str(&'a str),
    Usize(usize),
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PythonGraphCaptureLaunch<'a> {
    pub(crate) module: &'a str,
    pub(crate) function: &'a str,
    pub(crate) cuda_stream: *mut c_void,
    pub(crate) buffers: &'a [PythonDeviceBufferArg<'a>],
    pub(crate) kwargs: &'a [(&'a str, PythonKernelArg<'a>)],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PythonBoolQuery<'a> {
    pub(crate) module: &'a str,
    pub(crate) function: &'a str,
    pub(crate) kwargs: &'a [(&'a str, PythonKernelArg<'a>)],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PythonUsizeQuery<'a> {
    pub(crate) module: &'a str,
    pub(crate) function: &'a str,
    pub(crate) kwargs: &'a [(&'a str, PythonKernelArg<'a>)],
}

pub(crate) fn initialize_coordinator_python_capture_from_env() -> Result<Option<PythonCaptureStatus>>
{
    if !coordinator_python_capture_enabled() {
        return Ok(None);
    }

    let status = initialize_python_capture(
        DS41RT_B12X_ENV,
        COORDINATOR_PYTHON_CAPTURE_MODULES,
        "coordinator",
    )?;
    qualify_deepseek_v4_attention_capture()?;
    qualify_deepseek_v4_attention_compressor_capture()?;
    qualify_deepseek_v4_attention_producer_capture()?;
    qualify_deepseek_v4_attention_indexer_capture()?;
    qualify_deepseek_v4_attention_output_capture()?;
    qualify_deepseek_v4_mhc_capture()?;
    qualify_deepseek_v4_attention_layer_capture()?;
    qualify_deepseek_v4_sparse_block_capture()?;
    qualify_deepseek_v4_dspark_capture()?;
    qualify_deepseek_v4_spark_rank_capture()?;
    qualify_deepseek_v4_spark_prefill_capture()?;
    Ok(status)
}

fn qualify_deepseek_v4_attention_capture() -> Result<()> {
    let contracts = [
        ("decode", 1, 0, 0),
        ("decode", 16, 4, 512),
        ("extend", 2_048, 128, 1_024),
    ];
    for (mode, rows, compression, indexed_width) in contracts {
        let kwargs = [
            ("mode", PythonKernelArg::Str(mode)),
            ("rows", PythonKernelArg::Usize(rows)),
            ("heads", PythonKernelArg::Usize(64)),
            ("source_pages", PythonKernelArg::Usize(512)),
            ("swa_width", PythonKernelArg::Usize(128)),
            ("compression", PythonKernelArg::Usize(compression)),
            ("indexed_width", PythonKernelArg::Usize(indexed_width)),
        ];
        anyhow::ensure!(
            query_python_bool_during_startup(PythonBoolQuery {
                module: DEEPSEEK_V4_ATTENTION_CAPTURE_MODULE,
                function: DEEPSEEK_V4_ATTENTION_QUALIFY_FUNCTION,
                kwargs: &kwargs,
            })?,
            "DeepSeek V4 compressed MLA qualification returned false for mode={mode} rows={rows} compression={compression} indexed_width={indexed_width}"
        );
    }
    let pro_kwargs = [
        ("mode", PythonKernelArg::Str("decode")),
        ("rows", PythonKernelArg::Usize(16)),
        ("heads", PythonKernelArg::Usize(128)),
        ("source_pages", PythonKernelArg::Usize(512)),
        ("swa_width", PythonKernelArg::Usize(128)),
        ("compression", PythonKernelArg::Usize(4)),
        ("indexed_width", PythonKernelArg::Usize(1_024)),
    ];
    anyhow::ensure!(
        query_python_bool_during_startup(PythonBoolQuery {
            module: DEEPSEEK_V4_ATTENTION_CAPTURE_MODULE,
            function: DEEPSEEK_V4_ATTENTION_QUALIFY_FUNCTION,
            kwargs: &pro_kwargs,
        })?,
        "DeepSeek V4 Pro compressed MLA qualification returned false for heads=128 indexed_width=1024"
    );
    Ok(())
}

fn qualify_deepseek_v4_attention_producer_capture() -> Result<()> {
    for variant in ["flash", "pro"] {
        let kwargs = [
            ("variant", PythonKernelArg::Str(variant)),
            ("max_rows", PythonKernelArg::Usize(2_048)),
        ];
        anyhow::ensure!(
            query_python_bool_during_startup(PythonBoolQuery {
                module: DEEPSEEK_V4_ATTENTION_PRODUCER_CAPTURE_MODULE,
                function: DEEPSEEK_V4_ATTENTION_PRODUCER_QUALIFY_FUNCTION,
                kwargs: &kwargs,
            })?,
            "DeepSeek V4 attention producer qualification returned false for variant={variant} max_rows=2048"
        );
    }
    Ok(())
}

fn qualify_deepseek_v4_attention_indexer_capture() -> Result<()> {
    for variant in ["flash", "pro"] {
        let kwargs = [
            ("variant", PythonKernelArg::Str(variant)),
            ("max_rows", PythonKernelArg::Usize(2_048)),
        ];
        anyhow::ensure!(
            query_python_bool_during_startup(PythonBoolQuery {
                module: DEEPSEEK_V4_ATTENTION_PRODUCER_CAPTURE_MODULE,
                function: DEEPSEEK_V4_ATTENTION_INDEXER_QUALIFY_FUNCTION,
                kwargs: &kwargs,
            })?,
            "DeepSeek V4 index-query and physical-selection qualification returned false for variant={variant} max_rows=2048"
        );
    }
    Ok(())
}

fn qualify_deepseek_v4_attention_output_capture() -> Result<()> {
    for variant in ["flash", "pro"] {
        let kwargs = [
            ("variant", PythonKernelArg::Str(variant)),
            ("max_rows", PythonKernelArg::Usize(2_048)),
        ];
        anyhow::ensure!(
            query_python_bool_during_startup(PythonBoolQuery {
                module: DEEPSEEK_V4_ATTENTION_OUTPUT_CAPTURE_MODULE,
                function: DEEPSEEK_V4_ATTENTION_OUTPUT_QUALIFY_FUNCTION,
                kwargs: &kwargs,
            })?,
            "DeepSeek V4 grouped output-projection qualification returned false for variant={variant} max_rows=2048"
        );
    }
    Ok(())
}

fn qualify_deepseek_v4_mhc_capture() -> Result<()> {
    for variant in ["flash", "pro"] {
        let kwargs = [
            ("variant", PythonKernelArg::Str(variant)),
            ("max_rows", PythonKernelArg::Usize(2_048)),
        ];
        anyhow::ensure!(
            query_python_bool_during_startup(PythonBoolQuery {
                module: DEEPSEEK_V4_MHC_CAPTURE_MODULE,
                function: DEEPSEEK_V4_MHC_QUALIFY_FUNCTION,
                kwargs: &kwargs,
            })?,
            "DeepSeek V4 full mHC lifecycle qualification returned false for variant={variant} max_rows=2048"
        );
    }
    Ok(())
}

fn qualify_deepseek_v4_attention_layer_capture() -> Result<()> {
    let contracts = [
        ("flash", "decode", 0, 1),
        ("flash", "decode", 128, 1),
        ("flash", "decode", 4, 16),
        ("flash", "extend", 128, 2_048),
        ("pro", "decode", 4, 16),
    ];
    for (variant, mode, compression, max_rows) in contracts {
        let kwargs = [
            ("variant", PythonKernelArg::Str(variant)),
            ("mode", PythonKernelArg::Str(mode)),
            ("compression", PythonKernelArg::Usize(compression)),
            ("max_rows", PythonKernelArg::Usize(max_rows)),
            ("source_pages", PythonKernelArg::Usize(512)),
            ("max_positions", PythonKernelArg::Usize(1_048_576)),
        ];
        anyhow::ensure!(
            query_python_bool_during_startup(PythonBoolQuery {
                module: DEEPSEEK_V4_ATTENTION_LAYER_CAPTURE_MODULE,
                function: DEEPSEEK_V4_ATTENTION_LAYER_QUALIFY_FUNCTION,
                kwargs: &kwargs,
            })?,
            "DeepSeek V4 composite attention-layer arena qualification returned false for variant={variant} mode={mode} compression={compression} max_rows={max_rows}"
        );
    }
    Ok(())
}

fn qualify_deepseek_v4_sparse_block_capture() -> Result<()> {
    let contracts = [
        ("flash", "decode", 0, 1),
        ("flash", "decode", 4, 16),
        ("pro", "decode", 4, 16),
    ];
    for (variant, mode, compression, max_rows) in contracts {
        let kwargs = [
            ("variant", PythonKernelArg::Str(variant)),
            ("mode", PythonKernelArg::Str(mode)),
            ("compression", PythonKernelArg::Usize(compression)),
            ("max_rows", PythonKernelArg::Usize(max_rows)),
            ("source_pages", PythonKernelArg::Usize(512)),
            ("max_positions", PythonKernelArg::Usize(1_048_576)),
        ];
        anyhow::ensure!(
            query_python_bool_during_startup(PythonBoolQuery {
                module: DEEPSEEK_V4_SPARSE_BLOCK_CAPTURE_MODULE,
                function: DEEPSEEK_V4_SPARSE_BLOCK_QUALIFY_FUNCTION,
                kwargs: &kwargs,
            })?,
            "DeepSeek V4 sparse-block TP=4 qualification returned false for variant={variant} mode={mode} compression={compression} max_rows={max_rows}"
        );
    }
    Ok(())
}

fn qualify_deepseek_v4_dspark_capture() -> Result<()> {
    for variant in ["flash", "pro"] {
        let kwargs = [
            ("variant", PythonKernelArg::Str(variant)),
            ("max_batch", PythonKernelArg::Usize(16)),
            ("max_main_rows", PythonKernelArg::Usize(2_048)),
            ("validate_sparkinfer", PythonKernelArg::Bool(true)),
        ];
        anyhow::ensure!(
            query_python_bool_during_startup(PythonBoolQuery {
                module: DEEPSEEK_V4_DSPARK_CAPTURE_MODULE,
                function: DEEPSEEK_V4_DSPARK_QUALIFY_FUNCTION,
                kwargs: &kwargs,
            })?,
            "DeepSeek V4 integrated dSpark qualification returned false for variant={variant} max_batch=16 max_main_rows=2048"
        );
    }
    Ok(())
}

fn qualify_deepseek_v4_spark_rank_capture() -> Result<()> {
    anyhow::ensure!(
        query_python_bool_during_startup(PythonBoolQuery {
            module: DEEPSEEK_V4_SPARK_RANK_CAPTURE_MODULE,
            function: DEEPSEEK_V4_SPARK_RANK_QUALIFY_FUNCTION,
            kwargs: &[],
        })?,
        "DeepSeek V4 Flash Spark TP=4 decode M1 rank qualification returned false"
    );
    anyhow::ensure!(
        query_python_bool_during_startup(PythonBoolQuery {
            module: DEEPSEEK_V4_SPARK_RANK_CAPTURE_MODULE,
            function: DEEPSEEK_V4_SPARK_PREFILL_ROUTE_PACK_QUALIFY_FUNCTION,
            kwargs: &[],
        })?,
        "DeepSeek V4 Flash Spark TP=4 prefill route-pack qualification returned false"
    );
    Ok(())
}

fn qualify_deepseek_v4_spark_prefill_capture() -> Result<()> {
    anyhow::ensure!(
        query_python_bool_during_startup(PythonBoolQuery {
            module: DEEPSEEK_V4_SPARK_PREFILL_CAPTURE_MODULE,
            function: DEEPSEEK_V4_SPARK_PREFILL_QUALIFY_FUNCTION,
            kwargs: &[],
        })?,
        "DeepSeek V4 Flash Spark TP=4 prefill rank qualification returned false"
    );
    Ok(())
}

fn qualify_deepseek_v4_attention_compressor_capture() -> Result<()> {
    for variant in ["flash", "pro"] {
        for compress_ratio in [4, 128] {
            let kwargs = [
                ("variant", PythonKernelArg::Str(variant)),
                ("compress_ratio", PythonKernelArg::Usize(compress_ratio)),
                ("max_rows", PythonKernelArg::Usize(2_048)),
            ];
            anyhow::ensure!(
                query_python_bool_during_startup(PythonBoolQuery {
                    module: DEEPSEEK_V4_ATTENTION_COMPRESSOR_CAPTURE_MODULE,
                    function: DEEPSEEK_V4_ATTENTION_COMPRESSOR_QUALIFY_FUNCTION,
                    kwargs: &kwargs,
                })?,
                "DeepSeek V4 compressor qualification returned false for variant={variant} compression={compress_ratio} max_rows=2048"
            );
        }
    }
    Ok(())
}

pub(crate) fn initialize_spark_python_capture_from_env() -> Result<Option<PythonCaptureStatus>> {
    if !spark_python_capture_enabled() {
        return Ok(None);
    }

    initialize_python_capture(DS41RT_B12X_SPARK_ENV, SPARK_PYTHON_CAPTURE_MODULES, "Spark")
}

fn initialize_python_capture(
    gate_env: &'static str,
    modules: &[&str],
    label: &str,
) -> Result<Option<PythonCaptureStatus>> {
    let startup_started = Instant::now();
    let prepare_started = Instant::now();
    pyo3::prepare_freethreaded_python();
    eprintln!(
        "python_capture_startup_phase component={label:?} stage=python-runtime elapsed_ms={:.3} total_ms={:.3}",
        prepare_started.elapsed().as_secs_f64() * 1_000.0,
        startup_started.elapsed().as_secs_f64() * 1_000.0,
    );
    Python::with_gil(|py| -> PyResult<Vec<String>> {
        let path_started = Instant::now();
        ensure_ds41rt_python_reference_path(py)?;
        eprintln!(
            "python_capture_startup_phase component={label:?} stage=python-path elapsed_ms={:.3} total_ms={:.3}",
            path_started.elapsed().as_secs_f64() * 1_000.0,
            startup_started.elapsed().as_secs_f64() * 1_000.0,
        );
        let mut imported_modules = Vec::with_capacity(modules.len());
        for module in modules {
            let import_started = Instant::now();
            PyModule::import_bound(py, *module)?;
            eprintln!(
                "python_capture_startup_phase component={label:?} stage=module-import module={module} elapsed_ms={:.3} total_ms={:.3}",
                import_started.elapsed().as_secs_f64() * 1_000.0,
                startup_started.elapsed().as_secs_f64() * 1_000.0,
            );
            imported_modules.push((*module).to_owned());
        }
        Ok(imported_modules)
    })
    .map(|imported_modules| {
        Some(PythonCaptureStatus {
            gate_env,
            imported_modules,
        })
    })
    .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    .with_context(|| format!("importing {label} Python kernel modules"))
    .map(|status| {
        eprintln!(
            "python_capture_startup_phase component={label:?} stage=complete elapsed_ms={:.3} total_ms={:.3}",
            startup_started.elapsed().as_secs_f64() * 1_000.0,
            startup_started.elapsed().as_secs_f64() * 1_000.0,
        );
        status
    })
}

pub(crate) fn coordinator_python_capture_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = COORDINATOR_PYTHON_CAPTURE_TEST_OVERRIDE.with(|value| value.get()) {
        return enabled;
    }

    env::var(DS41RT_B12X_ENV)
        .map(|value| matches_env_true(&value))
        .unwrap_or(false)
}

pub(crate) fn attention_python_capture_enabled() -> bool {
    coordinator_python_capture_enabled()
}

pub(crate) fn finish_coordinator_python_capture_startup() {
    COORDINATOR_PYTHON_CAPTURE_STARTUP_OPEN.store(false, Ordering::Release);
}

pub(crate) fn coordinator_python_capture_startup_open() -> bool {
    COORDINATOR_PYTHON_CAPTURE_STARTUP_OPEN.load(Ordering::Acquire)
}

pub(crate) fn spark_python_capture_enabled() -> bool {
    env::var(DS41RT_B12X_SPARK_ENV)
        .map(|value| matches_env_true(&value))
        .unwrap_or(false)
}

#[allow(dead_code)]
pub(crate) fn launch_python_graph_capture(launch: PythonGraphCaptureLaunch<'_>) -> Result<()> {
    anyhow::ensure!(
        coordinator_python_capture_startup_open() || launch.module == "b12x_spark_capture",
        "coordinator Python graph capture is closed after startup"
    );
    anyhow::ensure!(
        !launch.cuda_stream.is_null(),
        "Python graph-capture launch requires a non-null CUDA stream"
    );

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| call_python_graph_capture(py, &launch).map(|_| ()))
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
        .with_context(|| {
            format!(
                "launching Python graph-capture kernel {}.{}",
                launch.module, launch.function
            )
        })
}

pub(crate) fn launch_python_kernel(launch: PythonGraphCaptureLaunch<'_>) -> Result<()> {
    anyhow::ensure!(
        !launch.cuda_stream.is_null(),
        "Python kernel launch requires a non-null CUDA stream"
    );

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| call_python_graph_capture(py, &launch).map(|_| ()))
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
        .with_context(|| {
            format!(
                "launching Python kernel {}.{}",
                launch.module, launch.function
            )
        })
}

pub(crate) fn query_python_bool_during_startup(query: PythonBoolQuery<'_>) -> Result<bool> {
    anyhow::ensure!(
        coordinator_python_capture_startup_open(),
        "Python planner queries are closed after coordinator startup"
    );

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| call_python_bool_query(py, &query))
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
        .with_context(|| {
            format!(
                "querying Python planner {}.{}",
                query.module, query.function
            )
        })
}

pub(crate) fn query_python_usize_during_startup(query: PythonUsizeQuery<'_>) -> Result<usize> {
    anyhow::ensure!(
        coordinator_python_capture_startup_open(),
        "Python planner queries are closed after coordinator startup"
    );

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| call_python_usize_query(py, &query))
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
        .with_context(|| {
            format!(
                "querying Python planner {}.{}",
                query.module, query.function
            )
        })
}

fn call_python_graph_capture<'py>(
    py: Python<'py>,
    launch: &PythonGraphCaptureLaunch<'_>,
) -> PyResult<pyo3::Bound<'py, PyAny>> {
    ensure_ds41rt_python_reference_path(py)?;
    let module = PyModule::import_bound(py, launch.module)?;
    let function = module.getattr(launch.function)?;
    let context = PyDict::new_bound(py);
    context.set_item("cuda_stream", launch.cuda_stream as usize)?;
    context.set_item("cuda_stream_ptr", launch.cuda_stream as usize)?;
    context.set_item("capture_phase", "cuda_graph_capture")?;

    let buffers = PyDict::new_bound(py);
    for buffer in launch.buffers {
        let py_buffer = PyDict::new_bound(py);
        py_buffer.set_item("ptr", buffer.ptr as usize)?;
        py_buffer.set_item("bytes", buffer.bytes)?;
        py_buffer.set_item("device_id", buffer.device_id)?;
        py_buffer.set_item("flags", buffer.flags)?;
        buffers.set_item(buffer.name, py_buffer)?;
    }
    context.set_item("buffers", buffers)?;

    let kwargs = PyDict::new_bound(py);
    for (name, value) in launch.kwargs {
        match value {
            PythonKernelArg::Bool(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::F64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::I64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Str(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Usize(value) => kwargs.set_item(name, value)?,
        }
    }

    function.call((context,), Some(&kwargs))
}

fn call_python_bool_query(py: Python<'_>, query: &PythonBoolQuery<'_>) -> PyResult<bool> {
    ensure_ds41rt_python_reference_path(py)?;
    let module = PyModule::import_bound(py, query.module)?;
    let function = module.getattr(query.function)?;
    let kwargs = PyDict::new_bound(py);
    for (name, value) in query.kwargs {
        match value {
            PythonKernelArg::Bool(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::F64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::I64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Str(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Usize(value) => kwargs.set_item(name, value)?,
        }
    }
    function.call((), Some(&kwargs))?.extract::<bool>()
}

fn call_python_usize_query(py: Python<'_>, query: &PythonUsizeQuery<'_>) -> PyResult<usize> {
    ensure_ds41rt_python_reference_path(py)?;
    let module = PyModule::import_bound(py, query.module)?;
    let function = module.getattr(query.function)?;
    let kwargs = PyDict::new_bound(py);
    for (name, value) in query.kwargs {
        match value {
            PythonKernelArg::Bool(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::F64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::I64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Str(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Usize(value) => kwargs.set_item(name, value)?,
        }
    }
    function.call((), Some(&kwargs))?.extract::<usize>()
}

fn add_ds41rt_python_reference_path(py: Python<'_>) -> PyResult<()> {
    let sys = PyModule::import_bound(py, "sys")?;
    let sys_path = sys.getattr("path")?;
    for path in ds41rt_python_reference_paths() {
        add_python_path(&sys_path, path)?;
    }
    for path in ds41rt_python_dynload_paths(py)? {
        add_python_path(&sys_path, path)?;
    }
    Ok(())
}

fn ensure_ds41rt_python_reference_path(py: Python<'_>) -> PyResult<()> {
    if DS41RT_PYTHON_REFERENCE_PATH_READY.load(Ordering::Acquire) {
        return Ok(());
    }
    // All callers hold the GIL, so successful initialization is serialized.
    // Avoid repeating sys.path scans, filesystem probes, and Python prefix
    // discovery for every one of the thousands of startup graph captures.
    add_ds41rt_python_reference_path(py)?;
    DS41RT_PYTHON_REFERENCE_PATH_READY.store(true, Ordering::Release);
    Ok(())
}

fn add_python_path(sys_path: &pyo3::Bound<'_, PyAny>, path: PathBuf) -> PyResult<()> {
    if !path.is_dir() {
        return Ok(());
    }
    let path = path.to_string_lossy().to_string();
    if !sys_path.contains(path.as_str())? {
        sys_path.call_method1("insert", (0, path))?;
    }
    Ok(())
}

fn ds41rt_python_reference_paths() -> [PathBuf; 2] {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("python")
        .join("reference");
    [root.clone(), root.join("ds41rt_reference")]
}

fn ds41rt_python_dynload_paths(py: Python<'_>) -> PyResult<Vec<PathBuf>> {
    let sys = PyModule::import_bound(py, "sys")?;
    let version_info = sys.getattr("version_info")?;
    let major = version_info.get_item(0)?.extract::<usize>()?;
    let minor = version_info.get_item(1)?.extract::<usize>()?;
    let python_version = format!("python{major}.{minor}");

    let mut prefixes = Vec::new();
    if let Ok(prefix) = sys.getattr("prefix")?.extract::<String>() {
        prefixes.push(PathBuf::from(prefix));
    }
    if let Ok(prefix) = sys.getattr("base_prefix")?.extract::<String>() {
        prefixes.push(PathBuf::from(prefix));
    }
    if let Some(prefix) = env::var_os("PYTHONHOME") {
        prefixes.push(PathBuf::from(prefix));
    }
    if let Some(prefix) = env::var_os("VIRTUAL_ENV") {
        prefixes.push(PathBuf::from(prefix));
    }
    prefixes.push(PathBuf::from("/usr"));
    prefixes.push(PathBuf::from("/usr/local"));

    let mut paths = Vec::new();
    for prefix in prefixes {
        for lib_dir in ["lib", "lib64"] {
            let path = prefix
                .join(lib_dir)
                .join(&python_version)
                .join("lib-dynload");
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    Ok(paths)
}

fn matches_env_true(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enabled"
    )
}

fn format_python_error(err: PyErr) -> String {
    err.to_string()
}

#[cfg(test)]
pub(crate) struct CoordinatorPythonCaptureTestOverride {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for CoordinatorPythonCaptureTestOverride {
    fn drop(&mut self) {
        set_coordinator_python_capture_test_override(self.previous);
    }
}

#[cfg(test)]
pub(crate) fn set_coordinator_python_capture_test_override(enabled: Option<bool>) -> Option<bool> {
    COORDINATOR_PYTHON_CAPTURE_TEST_OVERRIDE.with(|value| {
        let previous = value.get();
        value.set(enabled);
        previous
    })
}

#[cfg(test)]
pub(crate) fn coordinator_python_capture_test_override(
    enabled: bool,
) -> CoordinatorPythonCaptureTestOverride {
    CoordinatorPythonCaptureTestOverride {
        previous: set_coordinator_python_capture_test_override(Some(enabled)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn env_gate_defaults_to_disabled() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        env::remove_var(DS41RT_B12X_ENV);
        env::remove_var(DS41RT_B12X_SPARK_ENV);
        assert!(!coordinator_python_capture_enabled());
        assert!(!spark_python_capture_enabled());
    }

    #[test]
    fn env_gates_parse_true_values() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        env::set_var(DS41RT_B12X_ENV, "on");
        env::set_var(DS41RT_B12X_SPARK_ENV, "1");
        assert!(coordinator_python_capture_enabled());
        assert!(spark_python_capture_enabled());
        env::remove_var(DS41RT_B12X_ENV);
        env::remove_var(DS41RT_B12X_SPARK_ENV);
    }

    #[test]
    fn legacy_spark_b12x_env_does_not_enable_python_capture() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        env::remove_var(DS41RT_B12X_SPARK_ENV);
        env::set_var("DS41RT_B12X_SPARK", "1");
        assert!(!spark_python_capture_enabled());
        env::remove_var("DS41RT_B12X_SPARK");
    }

    #[test]
    fn deepseek_v4_attention_contract_imports_without_loading_gpu_runtime() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module = PyModule::import_bound(py, DEEPSEEK_V4_ATTENTION_CAPTURE_MODULE)?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("mode", "decode")?;
            kwargs.set_item("rows", 16)?;
            kwargs.set_item("source_pages", 512)?;
            kwargs.set_item("compression", 4)?;
            kwargs.set_item("indexed_width", 512)?;
            let contract = module
                .getattr("plan_deepseek_v4_compressed_mla")?
                .call((), Some(&kwargs))?;
            assert_eq!(contract.getattr("heads")?.extract::<usize>()?, 64);
            assert_eq!(contract.getattr("swa_width")?.extract::<usize>()?, 128);
            assert_eq!(
                contract
                    .getattr("indexed_page_tokens")?
                    .extract::<usize>()?,
                64
            );
            assert_eq!(contract.getattr("total_width")?.extract::<usize>()?, 640);
            assert_eq!(
                contract.getattr("max_chunks_per_row")?.extract::<usize>()?,
                54
            );
            let pro_kwargs = PyDict::new_bound(py);
            pro_kwargs.set_item("mode", "decode")?;
            pro_kwargs.set_item("rows", 16)?;
            pro_kwargs.set_item("heads", 128)?;
            pro_kwargs.set_item("source_pages", 512)?;
            pro_kwargs.set_item("compression", 4)?;
            pro_kwargs.set_item("indexed_width", 1_024)?;
            let pro_contract = module
                .getattr("plan_deepseek_v4_compressed_mla")?
                .call((), Some(&pro_kwargs))?;
            assert_eq!(pro_contract.getattr("heads")?.extract::<usize>()?, 128);
            assert_eq!(
                pro_contract.getattr("total_width")?.extract::<usize>()?,
                1_152
            );
            assert_eq!(
                pro_contract
                    .getattr("max_chunks_per_row")?
                    .extract::<usize>()?,
                96
            );
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn deepseek_v4_attention_layer_arena_preserves_fixed_tp4_lifecycle() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module = PyModule::import_bound(py, DEEPSEEK_V4_ATTENTION_LAYER_CAPTURE_MODULE)?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("variant", "flash")?;
            kwargs.set_item("mode", "decode")?;
            kwargs.set_item("compression", 4)?;
            kwargs.set_item("max_rows", 16)?;
            kwargs.set_item("source_pages", 512)?;
            kwargs.set_item("max_positions", 1_048_576)?;
            let contract = module
                .getattr("plan_deepseek_v4_attention_layer")?
                .call((), Some(&kwargs))?;
            assert_eq!(
                contract.getattr("status")?.extract::<String>()?,
                "qualified-composite-arena-not-active"
            );
            assert!(!contract.getattr("serving_allocates")?.extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("workspace_reuse_scope")?
                    .extract::<String>()?,
                "one-per-coordinator-execution-lane-across-layers"
            );
            assert_eq!(
                contract
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!contract.getattr("expert_parallel")?.extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("expert_global_top_k")?
                    .extract::<usize>()?,
                6
            );
            assert!(contract
                .getattr("expert_routes_identical_across_ranks")?
                .extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("expert_local_intermediate_fraction")?
                    .extract::<(usize, usize)>()?,
                (1, 4)
            );
            assert_eq!(
                contract
                    .getattr("expert_partial_output_width")?
                    .extract::<usize>()?,
                4_096
            );
            let sliding_binding = module.getattr("DeepseekV4SlidingAttentionLayerBinding")?;
            assert!(!sliding_binding
                .getattr("serving_allocates")?
                .extract::<bool>()?);
            assert!(sliding_binding
                .getattr("cuda_graph_safe")?
                .extract::<bool>()?);
            assert!(sliding_binding
                .getattr("uses_one_lane_arena")?
                .extract::<bool>()?);
            assert!(sliding_binding
                .getattr("persistent_state_is_external")?
                .extract::<bool>()?);
            assert_eq!(
                sliding_binding
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!sliding_binding
                .getattr("expert_parallel")?
                .extract::<bool>()?);
            assert!(module
                .getattr("bind_deepseek_v4_sliding_attention_layer")?
                .is_callable());
            assert!(module
                .getattr("run_deepseek_v4_sliding_attention_layer")?
                .is_callable());
            let c128_binding = module.getattr("DeepseekV4C128DecodeAttentionLayerBinding")?;
            assert!(!c128_binding
                .getattr("serving_allocates")?
                .extract::<bool>()?);
            assert!(c128_binding.getattr("cuda_graph_safe")?.extract::<bool>()?);
            assert!(c128_binding
                .getattr("uses_one_lane_arena")?
                .extract::<bool>()?);
            assert!(c128_binding
                .getattr("persistent_state_is_external")?
                .extract::<bool>()?);
            assert!(c128_binding
                .getattr("sequence_unique_rows")?
                .extract::<bool>()?);
            assert_eq!(
                c128_binding
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!c128_binding.getattr("expert_parallel")?.extract::<bool>()?);
            assert!(module
                .getattr("bind_deepseek_v4_c128_decode_attention_layer")?
                .is_callable());
            assert!(module
                .getattr("run_deepseek_v4_c128_decode_attention_layer")?
                .is_callable());
            let c128_prefill_binding =
                module.getattr("DeepseekV4C128PrefillAttentionLayerBinding")?;
            assert!(!c128_prefill_binding
                .getattr("serving_allocates")?
                .extract::<bool>()?);
            assert!(c128_prefill_binding
                .getattr("cuda_graph_safe")?
                .extract::<bool>()?);
            assert!(c128_prefill_binding
                .getattr("uses_one_lane_arena")?
                .extract::<bool>()?);
            assert!(c128_prefill_binding
                .getattr("persistent_state_is_external")?
                .extract::<bool>()?);
            assert!(c128_prefill_binding
                .getattr("initial_prefill_only")?
                .extract::<bool>()?);
            assert!(c128_prefill_binding
                .getattr("scheduler_owns_physical_selection")?
                .extract::<bool>()?);
            assert_eq!(
                c128_prefill_binding
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!c128_prefill_binding
                .getattr("expert_parallel")?
                .extract::<bool>()?);
            assert!(module
                .getattr("bind_deepseek_v4_c128_prefill_attention_layer")?
                .is_callable());
            assert!(module
                .getattr("run_deepseek_v4_c128_prefill_attention_layer")?
                .is_callable());
            let c128_continuation_binding =
                module.getattr("DeepseekV4C128ContinuationAttentionLayerBinding")?;
            assert!(!c128_continuation_binding
                .getattr("serving_allocates")?
                .extract::<bool>()?);
            assert!(c128_continuation_binding
                .getattr("cuda_graph_safe")?
                .extract::<bool>()?);
            assert!(c128_continuation_binding
                .getattr("uses_one_lane_arena")?
                .extract::<bool>()?);
            assert!(c128_continuation_binding
                .getattr("persistent_state_is_external")?
                .extract::<bool>()?);
            assert!(c128_continuation_binding
                .getattr("ordered_chunks_only")?
                .extract::<bool>()?);
            assert!(c128_continuation_binding
                .getattr("scheduler_owns_state_transactions")?
                .extract::<bool>()?);
            assert!(c128_continuation_binding
                .getattr("scheduler_owns_physical_selection")?
                .extract::<bool>()?);
            assert_eq!(
                c128_continuation_binding
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!c128_continuation_binding
                .getattr("expert_parallel")?
                .extract::<bool>()?);
            assert!(module
                .getattr("bind_deepseek_v4_c128_continuation_attention_layer")?
                .is_callable());
            assert!(module
                .getattr("run_deepseek_v4_c128_continuation_attention_layer")?
                .is_callable());
            let c4_prefill_binding = module.getattr("DeepseekV4C4PrefillAttentionLayerBinding")?;
            assert!(!c4_prefill_binding
                .getattr("serving_allocates")?
                .extract::<bool>()?);
            assert!(c4_prefill_binding
                .getattr("cuda_graph_safe")?
                .extract::<bool>()?);
            assert!(c4_prefill_binding
                .getattr("uses_one_lane_arena")?
                .extract::<bool>()?);
            assert!(c4_prefill_binding
                .getattr("persistent_state_is_external")?
                .extract::<bool>()?);
            assert!(c4_prefill_binding
                .getattr("selector_scratch_is_external")?
                .extract::<bool>()?);
            assert!(c4_prefill_binding
                .getattr("selector_outputs_physical_slots")?
                .extract::<bool>()?);
            assert!(c4_prefill_binding
                .getattr("selector_uses_shared_page_table")?
                .extract::<bool>()?);
            assert!(c4_prefill_binding
                .getattr("selector_uses_causal_lengths")?
                .extract::<bool>()?);
            assert!(c4_prefill_binding
                .getattr("initial_prefill_only")?
                .extract::<bool>()?);
            assert_eq!(
                c4_prefill_binding
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!c4_prefill_binding
                .getattr("expert_parallel")?
                .extract::<bool>()?);
            assert!(module
                .getattr("bind_deepseek_v4_c4_prefill_attention_layer")?
                .is_callable());
            assert!(module
                .getattr("run_deepseek_v4_c4_prefill_attention_layer")?
                .is_callable());
            let c4_continuation_binding =
                module.getattr("DeepseekV4C4ContinuationAttentionLayerBinding")?;
            assert!(!c4_continuation_binding
                .getattr("serving_allocates")?
                .extract::<bool>()?);
            assert!(c4_continuation_binding
                .getattr("cuda_graph_safe")?
                .extract::<bool>()?);
            assert!(c4_continuation_binding
                .getattr("uses_one_lane_arena")?
                .extract::<bool>()?);
            assert!(c4_continuation_binding
                .getattr("persistent_state_is_external")?
                .extract::<bool>()?);
            assert!(c4_continuation_binding
                .getattr("selector_scratch_is_external")?
                .extract::<bool>()?);
            assert!(c4_continuation_binding
                .getattr("selector_outputs_physical_slots")?
                .extract::<bool>()?);
            assert!(c4_continuation_binding
                .getattr("selector_uses_shared_page_table")?
                .extract::<bool>()?);
            assert!(c4_continuation_binding
                .getattr("selector_uses_causal_lengths")?
                .extract::<bool>()?);
            assert!(c4_continuation_binding
                .getattr("ordered_chunks_only")?
                .extract::<bool>()?);
            assert!(c4_continuation_binding
                .getattr("scheduler_owns_state_transactions")?
                .extract::<bool>()?);
            assert_eq!(
                c4_continuation_binding
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!c4_continuation_binding
                .getattr("expert_parallel")?
                .extract::<bool>()?);
            assert!(module
                .getattr("bind_deepseek_v4_c4_continuation_attention_layer")?
                .is_callable());
            assert!(module
                .getattr("run_deepseek_v4_c4_continuation_attention_layer")?
                .is_callable());
            let c4_binding = module.getattr("DeepseekV4C4DecodeAttentionLayerBinding")?;
            assert!(!c4_binding.getattr("serving_allocates")?.extract::<bool>()?);
            assert!(c4_binding.getattr("cuda_graph_safe")?.extract::<bool>()?);
            assert!(c4_binding
                .getattr("uses_one_lane_arena")?
                .extract::<bool>()?);
            assert!(c4_binding
                .getattr("persistent_state_is_external")?
                .extract::<bool>()?);
            assert!(c4_binding
                .getattr("selector_scratch_is_external")?
                .extract::<bool>()?);
            assert!(c4_binding
                .getattr("selector_outputs_physical_slots")?
                .extract::<bool>()?);
            assert!(c4_binding
                .getattr("sequence_unique_rows")?
                .extract::<bool>()?);
            assert_eq!(
                c4_binding
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!c4_binding.getattr("expert_parallel")?.extract::<bool>()?);
            assert!(module
                .getattr("bind_deepseek_v4_c4_decode_attention_layer")?
                .is_callable());
            assert!(module
                .getattr("run_deepseek_v4_c4_decode_attention_layer")?
                .is_callable());
            assert_eq!(
                contract
                    .getattr("arena")?
                    .getattr("total_bytes")?
                    .extract::<usize>()?,
                61_006_848
            );
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn deepseek_v4_sparse_block_preserves_split_graph_tp4_handoff() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module = PyModule::import_bound(py, DEEPSEEK_V4_SPARSE_BLOCK_CAPTURE_MODULE)?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("variant", "flash")?;
            kwargs.set_item("mode", "decode")?;
            kwargs.set_item("compression", 0)?;
            kwargs.set_item("max_rows", 1)?;
            kwargs.set_item("source_pages", 1)?;
            kwargs.set_item("max_positions", 1)?;
            let contract = module
                .getattr("plan_deepseek_v4_sparse_block")?
                .call((), Some(&kwargs))?;
            assert_eq!(
                contract.getattr("status")?.extract::<String>()?,
                "qualified-split-graph-sparse-block-not-active"
            );
            assert!(!contract.getattr("serving_allocates")?.extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("cuda_graph_segments")?
                    .extract::<usize>()?,
                2
            );
            assert!(contract
                .getattr("dispatch_barrier_between_graphs")?
                .extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!contract.getattr("expert_parallel")?.extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("expert_global_top_k")?
                    .extract::<usize>()?,
                6
            );
            assert!(contract
                .getattr("one_route_buffer_fans_out_to_all_ranks")?
                .extract::<bool>()?);
            assert!(contract
                .getattr("router_outputs_are_caller_owned")?
                .extract::<bool>()?);
            assert!(!contract
                .getattr("router_requires_host_readback")?
                .extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("router_score_scratch_dtype")?
                    .extract::<String>()?,
                "float32"
            );
            assert_eq!(
                contract
                    .getattr("shared_expert_intermediate")?
                    .extract::<usize>()?,
                2_048
            );
            assert_eq!(
                contract
                    .getattr("shared_expert_workspace_rows")?
                    .extract::<usize>()?,
                8
            );
            assert!(contract
                .getattr("shared_expert_outputs_are_caller_owned")?
                .extract::<bool>()?);
            assert!(!contract
                .getattr("shared_expert_requires_host_readback")?
                .extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("shared_expert_output_shape")?
                    .extract::<(usize, usize)>()?,
                (1, 4_096)
            );
            assert_eq!(
                contract
                    .getattr("shared_expert_output_dtype")?
                    .extract::<String>()?,
                "bfloat16"
            );
            assert_eq!(
                contract
                    .getattr("expert_local_intermediate_fraction")?
                    .extract::<(usize, usize)>()?,
                (1, 4)
            );
            assert_eq!(
                contract
                    .getattr("expert_partial_output_shape")?
                    .extract::<(usize, usize, usize)>()?,
                (4, 1, 4_096)
            );
            assert_eq!(
                contract
                    .getattr("expert_partial_dtype")?
                    .extract::<String>()?,
                "bfloat16"
            );
            assert_eq!(
                contract
                    .getattr("reduction_accumulator_dtype")?
                    .extract::<String>()?,
                "float32"
            );
            assert!(contract
                .getattr("mhc_scratch_reused_across_graph_segments")?
                .extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("arena")?
                    .getattr("total_bytes")?
                    .extract::<usize>()?,
                1_265_664
            );
            let binding = module.getattr("DeepseekV4SparseBlockBinding")?;
            assert!(!binding.getattr("serving_allocates")?.extract::<bool>()?);
            assert!(binding.getattr("cuda_graph_safe")?.extract::<bool>()?);
            assert_eq!(
                binding.getattr("cuda_graph_segments")?.extract::<usize>()?,
                2
            );
            assert!(binding
                .getattr("rank_partials_are_hidden_width_bf16")?
                .extract::<bool>()?);
            assert!(binding
                .getattr("router_outputs_are_caller_owned")?
                .extract::<bool>()?);
            assert!(!binding
                .getattr("router_requires_host_readback")?
                .extract::<bool>()?);
            assert!(binding
                .getattr("shared_expert_outputs_are_caller_owned")?
                .extract::<bool>()?);
            assert!(!binding
                .getattr("shared_expert_requires_host_readback")?
                .extract::<bool>()?);
            assert!(binding.getattr("reduction_is_fp32")?.extract::<bool>()?);
            assert_eq!(
                binding
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!binding.getattr("expert_parallel")?.extract::<bool>()?);
            assert!(module
                .getattr("bind_deepseek_v4_sparse_block")?
                .is_callable());
            assert!(module
                .getattr("run_deepseek_v4_sparse_block_attention")?
                .is_callable());
            assert!(module
                .getattr("run_deepseek_v4_sparse_block_post_dispatch")?
                .is_callable());
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn deepseek_v4_flash_spark_rank_freezes_decode_and_prefill_route_pack_arenas() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module = PyModule::import_bound(py, DEEPSEEK_V4_SPARK_RANK_CAPTURE_MODULE)?;
            for rank in 0..4 {
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("rank", rank)?;
                let contract = module
                    .getattr("plan_deepseek_v4_flash_spark_rank_decode_m1")?
                    .call((), Some(&kwargs))?;
                assert_eq!(contract.getattr("rank")?.extract::<usize>()?, rank);
                assert_eq!(
                    contract
                        .getattr("expert_tensor_parallel")?
                        .extract::<usize>()?,
                    4
                );
                assert!(!contract.getattr("expert_parallel")?.extract::<bool>()?);
                assert_eq!(contract.getattr("routed_experts")?.extract::<usize>()?, 256);
                assert_eq!(contract.getattr("global_top_k")?.extract::<usize>()?, 6);
                assert_eq!(
                    contract.getattr("local_intermediate")?.extract::<usize>()?,
                    512
                );
                assert!(contract
                    .getattr("owns_same_expert_ids_as_every_rank")?
                    .extract::<bool>()?);
                assert!(contract
                    .getattr("consumes_identical_global_routes")?
                    .extract::<bool>()?);
                assert!(!contract
                    .getattr("requires_host_route_pack")?
                    .extract::<bool>()?);
                assert_eq!(
                    contract
                        .getattr("arena")?
                        .getattr("total_bytes")?
                        .extract::<usize>()?,
                    2_006_016
                );
            }
            assert!(module
                .getattr(DEEPSEEK_V4_SPARK_RANK_QUALIFY_FUNCTION)?
                .call0()?
                .extract::<bool>()?);
            for rank in 0..4 {
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("rank", rank)?;
                let contract = module
                    .getattr("plan_deepseek_v4_flash_spark_prefill_route_pack")?
                    .call((), Some(&kwargs))?;
                assert_eq!(contract.getattr("rank")?.extract::<usize>()?, rank);
                assert_eq!(
                    contract
                        .getattr("expert_tensor_parallel")?
                        .extract::<usize>()?,
                    4
                );
                assert!(!contract.getattr("expert_parallel")?.extract::<bool>()?);
                assert_eq!(contract.getattr("max_rows")?.extract::<usize>()?, 2_048);
                assert_eq!(
                    contract.getattr("route_block_rows")?.extract::<usize>()?,
                    32
                );
                assert!(!contract
                    .getattr("applies_expert_ownership_map")?
                    .extract::<bool>()?);
                assert_eq!(
                    contract
                        .getattr("arena")?
                        .getattr("total_bytes")?
                        .extract::<usize>()?,
                    137_216
                );
            }
            assert!(module
                .getattr(DEEPSEEK_V4_SPARK_PREFILL_ROUTE_PACK_QUALIFY_FUNCTION)?
                .call0()?
                .extract::<bool>()?);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn deepseek_v4_flash_spark_prefill_freezes_replicated_tp4_arena() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module = PyModule::import_bound(py, DEEPSEEK_V4_SPARK_PREFILL_CAPTURE_MODULE)?;
            for rank in 0..4 {
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("rank", rank)?;
                let contract = module
                    .getattr("plan_deepseek_v4_flash_spark_prefill")?
                    .call((), Some(&kwargs))?;
                assert_eq!(contract.getattr("rank")?.extract::<usize>()?, rank);
                assert_eq!(
                    contract
                        .getattr("expert_tensor_parallel")?
                        .extract::<usize>()?,
                    4
                );
                assert!(!contract.getattr("expert_parallel")?.extract::<bool>()?);
                assert_eq!(contract.getattr("max_rows")?.extract::<usize>()?, 2_048);
                assert_eq!(
                    contract.getattr("local_intermediate")?.extract::<usize>()?,
                    512
                );
                assert!(contract
                    .getattr("consumes_identical_global_routes")?
                    .extract::<bool>()?);
                assert!(!contract
                    .getattr("applies_expert_ownership_map")?
                    .extract::<bool>()?);
                assert!(contract
                    .getattr("requires_coordinator_reduction")?
                    .extract::<bool>()?);
                let arena = contract.getattr("arena")?;
                assert_eq!(
                    arena.getattr("total_bytes")?.extract::<usize>()?,
                    197_319_680
                );
                assert_eq!(arena.getattr("regions")?.len()?, 15);
            }
            assert!(module
                .getattr(DEEPSEEK_V4_SPARK_PREFILL_QUALIFY_FUNCTION)?
                .call0()?
                .extract::<bool>()?);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn deepseek_v4_compressor_contract_imports_without_loading_gpu_runtime() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module =
                PyModule::import_bound(py, DEEPSEEK_V4_ATTENTION_COMPRESSOR_CAPTURE_MODULE)?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("variant", "flash")?;
            kwargs.set_item("compress_ratio", 4)?;
            kwargs.set_item("max_rows", 2_048)?;
            let contract = module
                .getattr("plan_deepseek_v4_attention_compressor")?
                .call((), Some(&kwargs))?;
            let geometry = contract.getattr("geometry")?;
            assert_eq!(
                geometry
                    .getattr("joint_projection_width")?
                    .extract::<usize>()?,
                2_560
            );
            assert_eq!(geometry.getattr("state_rows")?.extract::<usize>()?, 8);
            assert!(contract
                .getattr("sequence_unique_decode_only")?
                .extract::<bool>()?);
            assert!(contract
                .getattr("supports_ordered_prefill")?
                .extract::<bool>()?);
            assert!(!contract
                .getattr("supports_mtp_state_transactions")?
                .extract::<bool>()?);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn deepseek_v4_indexer_contract_imports_without_loading_gpu_runtime() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module = PyModule::import_bound(py, DEEPSEEK_V4_ATTENTION_PRODUCER_CAPTURE_MODULE)?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("variant", "flash")?;
            kwargs.set_item("max_rows", 2_048)?;
            let contract = module
                .getattr("plan_deepseek_v4_attention_indexer")?
                .call((), Some(&kwargs))?;
            let geometry = contract.getattr("geometry")?;
            assert_eq!(geometry.getattr("heads")?.extract::<usize>()?, 64);
            assert_eq!(geometry.getattr("head_dim")?.extract::<usize>()?, 128);
            assert_eq!(geometry.getattr("top_k")?.extract::<usize>()?, 512);
            assert_eq!(
                contract
                    .getattr("scratch")?
                    .getattr("total_bytes")?
                    .extract::<usize>()?,
                36_044_800
            );
            assert!(contract
                .getattr("output_physical_slots")?
                .extract::<bool>()?);
            assert!(!contract.getattr("serving_allocates")?.extract::<bool>()?);
            assert_eq!(
                contract.getattr("status")?.extract::<String>()?,
                "qualified-index-query-and-physical-selection-not-active"
            );
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn deepseek_v4_attention_output_contract_imports_without_loading_gpu_runtime() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module = PyModule::import_bound(py, DEEPSEEK_V4_ATTENTION_OUTPUT_CAPTURE_MODULE)?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("variant", "flash")?;
            kwargs.set_item("max_rows", 2_048)?;
            let contract = module
                .getattr("plan_deepseek_v4_attention_output")?
                .call((), Some(&kwargs))?;
            let geometry = contract.getattr("geometry")?;
            assert_eq!(geometry.getattr("heads")?.extract::<usize>()?, 64);
            assert_eq!(geometry.getattr("groups")?.extract::<usize>()?, 8);
            assert_eq!(geometry.getattr("group_width")?.extract::<usize>()?, 4_096);
            assert_eq!(geometry.getattr("rank")?.extract::<usize>()?, 1_024);
            assert_eq!(
                contract
                    .getattr("scratch")?
                    .getattr("total_bytes")?
                    .extract::<usize>()?,
                139_460_608
            );
            assert!(!contract.getattr("serving_allocates")?.extract::<bool>()?);
            assert!(!contract.getattr("changes_expert_tp")?.extract::<bool>()?);
            assert_eq!(
                contract.getattr("status")?.extract::<String>()?,
                "qualified-grouped-output-projection-not-active"
            );
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn deepseek_v4_mhc_contract_imports_without_loading_gpu_runtime() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module = PyModule::import_bound(py, DEEPSEEK_V4_MHC_CAPTURE_MODULE)?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("variant", "flash")?;
            kwargs.set_item("max_rows", 2_048)?;
            let contract = module
                .getattr("plan_deepseek_v4_mhc")?
                .call((), Some(&kwargs))?;
            let geometry = contract.getattr("geometry")?;
            assert_eq!(geometry.getattr("hidden")?.extract::<usize>()?, 4_096);
            assert_eq!(geometry.getattr("split_k")?.extract::<usize>()?, 64);
            assert_eq!(
                contract
                    .getattr("scratch")?
                    .getattr("total_bytes")?
                    .extract::<usize>()?,
                13_107_200
            );
            assert_eq!(
                contract
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!contract.getattr("expert_parallel")?.extract::<bool>()?);
            assert!(contract
                .getattr("steady_state_fuses_post_pre")?
                .extract::<bool>()?);
            assert_eq!(
                contract.getattr("status")?.extract::<String>()?,
                "qualified-full-mhc-lifecycle-not-active"
            );
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn deepseek_v4_dspark_contract_imports_without_loading_gpu_runtime() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            add_ds41rt_python_reference_path(py)?;
            let module = PyModule::import_bound(py, DEEPSEEK_V4_DSPARK_CAPTURE_MODULE)?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("variant", "flash")?;
            kwargs.set_item("max_batch", 2)?;
            kwargs.set_item("max_main_rows", 128)?;
            let contract = module
                .getattr("plan_deepseek_v4_dspark")?
                .call((), Some(&kwargs))?;
            let geometry = contract.getattr("geometry")?;
            assert_eq!(geometry.getattr("hidden")?.extract::<usize>()?, 4_096);
            assert_eq!(
                geometry
                    .getattr("physical_block_ids")?
                    .extract::<Vec<usize>>()?,
                vec![43, 44, 45]
            );
            assert_eq!(
                geometry
                    .getattr("storage_prefixes")?
                    .extract::<Vec<String>>()?,
                vec!["mtp.0", "mtp.1", "mtp.2"]
            );
            assert_eq!(
                contract
                    .getattr("arena")?
                    .getattr("persistent_kv_bytes")?
                    .extract::<usize>()?,
                3 * 2 * 149_760
            );
            assert_eq!(
                contract
                    .getattr("blocks")?
                    .get_item(0)?
                    .getattr("expert_handoff")?
                    .getattr("attention")?
                    .getattr("swa_width")?
                    .extract::<usize>()?,
                133
            );
            assert_eq!(
                contract
                    .getattr("decode_attention")?
                    .getattr("workspace_growth_bytes")?
                    .extract::<usize>()?,
                0
            );
            assert_eq!(
                contract
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            assert!(!contract.getattr("expert_parallel")?.extract::<bool>()?);
            assert!(!contract.getattr("recurrent_mtp_state")?.extract::<bool>()?);
            assert!(contract
                .getattr("proposal_scratch_reuses_markov_storage")?
                .extract::<bool>()?);
            assert_eq!(
                contract
                    .getattr("prompt_prime_capture_surface")?
                    .extract::<Vec<String>>()?,
                vec![
                    "prepare_deepseek_v4_dspark_prompt_prime",
                    "capture_deepseek_v4_dspark_prompt_prime",
                ]
            );
            let prompt_binding = module.getattr("DeepseekV4DsparkPromptPrimeBlockBinding")?;
            assert!(!prompt_binding
                .getattr("serving_allocates")?
                .extract::<bool>()?);
            assert!(prompt_binding
                .getattr("cuda_graph_safe")?
                .extract::<bool>()?);
            assert!(!prompt_binding
                .getattr("computes_query")?
                .extract::<bool>()?);
            let block_binding = module.getattr("DeepseekV4DsparkBlockBinding")?;
            assert!(!block_binding
                .getattr("serving_allocates")?
                .extract::<bool>()?);
            assert!(block_binding
                .getattr("cuda_graph_safe")?
                .extract::<bool>()?);
            assert_eq!(
                block_binding
                    .getattr("expert_tensor_parallel")?
                    .extract::<usize>()?,
                4
            );
            for function in [
                "prepare_deepseek_v4_dspark_entry_projection",
                "capture_deepseek_v4_dspark_entry_projection",
                "prepare_deepseek_v4_dspark_prompt_prime",
                "capture_deepseek_v4_dspark_prompt_prime",
                "bind_deepseek_v4_dspark_prompt_prime_block",
                "run_deepseek_v4_dspark_prompt_prime_block",
                "bind_deepseek_v4_dspark_block",
                "run_deepseek_v4_dspark_block_pre_dispatch",
                "run_deepseek_v4_dspark_block_post_dispatch",
                "deepseek_v4_dspark_prompt_buffer_offset",
                "deepseek_v4_dspark_prompt_buffer_nbytes",
                "deepseek_v4_dspark_proposal_buffer_offset",
                "deepseek_v4_dspark_proposal_buffer_nbytes",
            ] {
                assert!(module.getattr(function)?.is_callable());
            }
            let storage_kwargs = PyDict::new_bound(py);
            storage_kwargs.set_item("variant", "flash")?;
            storage_kwargs.set_item("max_batch", 16)?;
            storage_kwargs.set_item("max_main_rows", 2_048)?;
            assert_eq!(
                module
                    .getattr("deepseek_v4_dspark_arena_nbytes")?
                    .call((), Some(&storage_kwargs))?
                    .extract::<usize>()?,
                168_077_824
            );
            assert_eq!(
                module
                    .getattr("deepseek_v4_dspark_persistent_kv_nbytes")?
                    .call((), Some(&storage_kwargs))?
                    .extract::<usize>()?,
                3 * 16 * 149_760
            );
            assert_eq!(
                contract.getattr("status")?.extract::<String>()?,
                "qualified-integrated-dspark-composite-not-active"
            );
            kwargs.set_item("validate_sparkinfer", false)?;
            let qualified = module
                .getattr(DEEPSEEK_V4_DSPARK_QUALIFY_FUNCTION)?
                .call((), Some(&kwargs))?
                .extract::<bool>()?;
            assert!(qualified);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    }

    #[test]
    fn launch_passes_stream_buffers_and_kwargs_to_python() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::from_code_bound(
                py,
                r#"
captured = None
def capture(ctx, *, rows, label, deterministic):
    global captured
    captured = {
        "stream": ctx["cuda_stream"],
        "phase": ctx["capture_phase"],
        "x_ptr": ctx["buffers"]["x"]["ptr"],
        "x_bytes": ctx["buffers"]["x"]["bytes"],
        "device_id": ctx["buffers"]["x"]["device_id"],
        "rows": rows,
        "label": label,
        "deterministic": deterministic,
    }
"#,
                "ds41rt_test_capture.py",
                "ds41rt_test_capture",
            )?;
            let sys = PyModule::import_bound(py, "sys")?;
            sys.getattr("modules")?
                .set_item("ds41rt_test_capture", module)?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        let stream = 0x1234usize as *mut c_void;
        let ptr = 0x5678usize as *mut c_void;
        let buffers = [PythonDeviceBufferArg {
            name: "x",
            ptr,
            bytes: 4096,
            device_id: 0,
            flags: 7,
        }];
        let kwargs = [
            ("rows", PythonKernelArg::Usize(16)),
            ("label", PythonKernelArg::Str("unit-test")),
            ("deterministic", PythonKernelArg::Bool(true)),
        ];
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "ds41rt_test_capture",
            function: "capture",
            cuda_stream: stream,
            buffers: &buffers,
            kwargs: &kwargs,
        })?;

        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::import_bound(py, "ds41rt_test_capture")?;
            let captured = module.getattr("captured")?;
            assert_eq!(
                captured.get_item("stream")?.extract::<usize>()?,
                stream as usize
            );
            assert_eq!(
                captured.get_item("phase")?.extract::<String>()?,
                "cuda_graph_capture"
            );
            assert_eq!(
                captured.get_item("x_ptr")?.extract::<usize>()?,
                ptr as usize
            );
            assert_eq!(captured.get_item("x_bytes")?.extract::<usize>()?, 4096);
            assert_eq!(captured.get_item("device_id")?.extract::<i32>()?, 0);
            assert_eq!(captured.get_item("rows")?.extract::<usize>()?, 16);
            assert_eq!(
                captured.get_item("label")?.extract::<String>()?,
                "unit-test"
            );
            assert!(captured.get_item("deterministic")?.extract::<bool>()?);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        Ok(())
    }

    #[test]
    fn startup_bool_query_passes_kwargs_and_extracts_result() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::from_code_bound(
                py,
                r#"
def plan(*, workload, rows, enabled):
    return workload == "decode" and rows == 8 and enabled
"#,
                "ds41rt_test_bool_query.py",
                "ds41rt_test_bool_query",
            )?;
            let sys = PyModule::import_bound(py, "sys")?;
            sys.getattr("modules")?
                .set_item("ds41rt_test_bool_query", module)?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        let kwargs = [
            ("workload", PythonKernelArg::Str("decode")),
            ("rows", PythonKernelArg::Usize(8)),
            ("enabled", PythonKernelArg::Bool(true)),
        ];
        assert!(query_python_bool_during_startup(PythonBoolQuery {
            module: "ds41rt_test_bool_query",
            function: "plan",
            kwargs: &kwargs,
        })?);

        Ok(())
    }

    #[test]
    fn launch_imports_ds41rt_reference_b12x_adapter_with_target_override() -> Result<()> {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::from_code_bound(
                py,
                r#"
captured = None
def capture(ctx, **kwargs):
    global captured
    captured = {
        "phase": ctx["capture_phase"],
        "rows": kwargs["rows"],
    }
"#,
                "ds41rt_test_b12x_capture_target.py",
                "ds41rt_test_b12x_capture_target",
            )?;
            let sys = PyModule::import_bound(py, "sys")?;
            sys.getattr("modules")?
                .set_item("ds41rt_test_b12x_capture_target", module)?;
            let os = PyModule::import_bound(py, "os")?;
            os.getattr("environ")?.set_item(
                "DS41RT_B12X_MLA_CAPTURE_TARGET",
                "ds41rt_test_b12x_capture_target:capture",
            )?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        env::set_var(
            "DS41RT_B12X_MLA_CAPTURE_TARGET",
            "ds41rt_test_b12x_capture_target:capture",
        );
        let kwargs = [("rows", PythonKernelArg::Usize(4))];
        let result = launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "b12x_mla_capture",
            function: "capture_mla_rope_attention",
            cuda_stream: 0x1234usize as *mut c_void,
            buffers: &[],
            kwargs: &kwargs,
        });
        env::remove_var("DS41RT_B12X_MLA_CAPTURE_TARGET");
        Python::with_gil(|py| -> PyResult<()> {
            let os = PyModule::import_bound(py, "os")?;
            os.getattr("environ")?
                .call_method1("pop", ("DS41RT_B12X_MLA_CAPTURE_TARGET", py.None()))?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;
        result?;

        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::import_bound(py, "ds41rt_test_b12x_capture_target")?;
            let captured = module.getattr("captured")?;
            assert_eq!(
                captured.get_item("phase")?.extract::<String>()?,
                "cuda_graph_capture"
            );
            assert_eq!(captured.get_item("rows")?.extract::<usize>()?, 4);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        Ok(())
    }

    #[test]
    fn launch_imports_ds41rt_reference_b12x_spark_adapter_with_target_override() -> Result<()> {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::from_code_bound(
                py,
                r#"
captured = None
def capture(ctx, **kwargs):
    global captured
    captured = {
        "phase": ctx["capture_phase"],
        "rows": kwargs["rows"],
        "n": kwargs["n"],
        "k": kwargs["k"],
    }
"#,
                "ds41rt_test_b12x_spark_capture_target.py",
                "ds41rt_test_b12x_spark_capture_target",
            )?;
            let sys = PyModule::import_bound(py, "sys")?;
            sys.getattr("modules")?
                .set_item("ds41rt_test_b12x_spark_capture_target", module)?;
            let os = PyModule::import_bound(py, "os")?;
            os.getattr("environ")?.set_item(
                "DS41RT_B12X_SPARK_CAPTURE_TARGET",
                "ds41rt_test_b12x_spark_capture_target:capture",
            )?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        env::set_var(
            "DS41RT_B12X_SPARK_CAPTURE_TARGET",
            "ds41rt_test_b12x_spark_capture_target:capture",
        );
        let kwargs = [
            ("rows", PythonKernelArg::Usize(4)),
            ("n", PythonKernelArg::Usize(16)),
            ("k", PythonKernelArg::Usize(32)),
        ];
        let result = launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "b12x_spark_capture",
            function: "capture_dense_gemm",
            cuda_stream: 0x1234usize as *mut c_void,
            buffers: &[],
            kwargs: &kwargs,
        });
        env::remove_var("DS41RT_B12X_SPARK_CAPTURE_TARGET");
        Python::with_gil(|py| -> PyResult<()> {
            let os = PyModule::import_bound(py, "os")?;
            os.getattr("environ")?
                .call_method1("pop", ("DS41RT_B12X_SPARK_CAPTURE_TARGET", py.None()))?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;
        result?;

        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::import_bound(py, "ds41rt_test_b12x_spark_capture_target")?;
            let captured = module.getattr("captured")?;
            assert_eq!(
                captured.get_item("phase")?.extract::<String>()?,
                "cuda_graph_capture"
            );
            assert_eq!(captured.get_item("rows")?.extract::<usize>()?, 4);
            assert_eq!(captured.get_item("n")?.extract::<usize>()?, 16);
            assert_eq!(captured.get_item("k")?.extract::<usize>()?, 32);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        Ok(())
    }
}
