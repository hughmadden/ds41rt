//! Bounded, opt-in capture of completed wire executions for the layer-1
//! rank-1 expert-route drift diagnostic.
//!
//! Enabled only when `DS41RT_EXPERT_TRACE_DIR` is set, read once per process.
//! When disabled there are no device reads, no files and no per-request env
//! lookups. When enabled, the first successfully completed request of exactly
//! one row and the first of exactly two rows are captured per process for
//! layer 1 on executor 2 (rank 1) only; later requests, other layers, other
//! executors and larger shapes are never touched. Two serving lanes may race
//! a claim; exactly one winner per row shape is recorded and its wire
//! identity plus the monotonic trace ordinal document which lane won.
//!
//! Capture reads only completed buffers after `ExpertExecution::synchronize`
//! while the wave still owns them, before the response is emitted or storage
//! is reused. No protocol, kernel, dispatch or arithmetic state is changed;
//! capture failures are logged, never fail the request, and release the claim
//! so the first later successful request of that shape is still recorded.
//! Token-accumulating kernels expose no per-route planes, so a selected
//! request running on one is rejected instead of fabricating six route
//! planes. Consumers should treat a trace directory as complete only once
//! `metadata.json` exists: data files are written first, all with
//! `create_new`, and `metadata.json` is written last.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{ensure, Context, Result};
use ds41rt_ffi::Ds41rtDeviceBuffer;

use super::super::ExpertLayer;
use super::{ExpertExecution, HostExpertExchange};

pub(crate) const TRACE_SCHEMA: u32 = 1;
pub(crate) const TRACE_ENV_VAR: &str = "DS41RT_EXPERT_TRACE_DIR";
/// Backbone layer 1 is the layer under investigation.
pub(crate) const TRACE_LAYER: usize = 1;
/// Executor identities are `rank + 1` on the local Spark workers.
pub(crate) const TRACE_EXECUTOR_ID: u64 = 2;
pub(crate) const TRACE_ROW_CHOICES: [u32; 2] = [1, 2];
pub(crate) const TRACE_TOPK: u32 = 6;
pub(crate) const TRACE_HIDDEN: u32 = 5120;

/// Per-process bounded capture selection; the serving path and the CPU tests
/// share this exact implementation.
pub(crate) struct ExpertTraceSelection {
    dir: PathBuf,
    claimed: [AtomicBool; TRACE_ROW_CHOICES.len()],
    ordinal: AtomicU64,
}

static GLOBAL_SELECTION: OnceLock<Option<ExpertTraceSelection>> = OnceLock::new();

/// Parse one env var value; shared by the process-global cache and the
/// CPU tests so both exercise identical logic.
pub(crate) fn selection_from_env_var(value: Option<OsString>) -> Option<ExpertTraceSelection> {
    let dir = PathBuf::from(value?);
    (!dir.as_os_str().is_empty()).then(|| ExpertTraceSelection {
        dir,
        claimed: [const { AtomicBool::new(false) }; TRACE_ROW_CHOICES.len()],
        ordinal: AtomicU64::new(0),
    })
}

impl ExpertTraceSelection {
    /// Read `DS41RT_EXPERT_TRACE_DIR` once per process. Unset or empty
    /// disables capture completely.
    pub(crate) fn global() -> Option<&'static ExpertTraceSelection> {
        GLOBAL_SELECTION
            .get_or_init(|| selection_from_env_var(std::env::var_os(TRACE_ENV_VAR)))
            .as_ref()
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// True only for the exact layer/executor/rows under investigation.
    pub(crate) fn is_candidate(&self, layer: usize, executor_id: u64, rows: u32) -> bool {
        layer == TRACE_LAYER
            && executor_id == TRACE_EXECUTOR_ID
            && TRACE_ROW_CHOICES.contains(&rows)
    }

    /// Atomically claim the single capture slot for `rows`; exactly one
    /// concurrent caller wins. The claim carries the monotonic trace ordinal
    /// used as part of the unique on-disk identity.
    pub(crate) fn try_claim(&self, rows: u32) -> Option<ExpertTraceClaim<'_>> {
        let index = TRACE_ROW_CHOICES.iter().position(|&choice| choice == rows)?;
        if self.claimed[index]
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let ordinal = self.ordinal.fetch_add(1, Ordering::Relaxed) + 1;
            Some(ExpertTraceClaim {
                selection: self,
                rows,
                ordinal,
            })
        } else {
            None
        }
    }
}

/// An exclusive capture slot. Dropping after a successful write keeps the
/// slot taken; [`ExpertTraceClaim::release`] reopens it after a failed
/// capture so the first later successful request of that shape is recorded.
pub(crate) struct ExpertTraceClaim<'a> {
    selection: &'a ExpertTraceSelection,
    rows: u32,
    ordinal: u64,
}

impl ExpertTraceClaim<'_> {
    pub(crate) fn rows(&self) -> u32 {
        self.rows
    }
    pub(crate) fn ordinal(&self) -> u64 {
        self.ordinal
    }
    pub(crate) fn release(self) {
        let index = TRACE_ROW_CHOICES
            .iter()
            .position(|&choice| choice == self.rows)
            .expect("claim rows are always trace row choices");
        self.selection.claimed[index].store(false, Ordering::Release);
    }
}

/// Checked byte extents of every captured buffer for one row shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertTraceExtents {
    pub(crate) rows: u32,
    pub(crate) input_row_bytes: usize,
    pub(crate) hidden_bytes: usize,
    pub(crate) route_ids_bytes: usize,
    pub(crate) routing_bytes: usize,
    pub(crate) route_partials_bytes: usize,
    pub(crate) compact_bytes: usize,
}

impl ExpertTraceExtents {
    pub(crate) fn for_rows(rows: u32, input_row_bytes: usize, topk: u32) -> Result<Self> {
        ensure!(rows > 0, "expert trace needs at least one row");
        let row_count = rows as usize;
        let routes = row_count
            .checked_mul(topk as usize)
            .context("expert trace route count overflow")?;
        let hidden_bytes = row_count
            .checked_mul(input_row_bytes)
            .context("expert trace input overflow")?;
        let route_ids_bytes = routes.checked_mul(4).context("expert trace route ids overflow")?;
        let routing_bytes = routes.checked_mul(4).context("expert trace routing overflow")?;
        let route_partials_bytes = routes
            .checked_mul(TRACE_HIDDEN as usize)
            .and_then(|n| n.checked_mul(4))
            .context("expert trace route partials overflow")?;
        let compact_bytes = row_count
            .checked_mul(TRACE_HIDDEN as usize)
            .and_then(|n| n.checked_mul(2))
            .context("expert trace compact output overflow")?;
        Ok(Self {
            rows,
            input_row_bytes,
            hidden_bytes,
            route_ids_bytes,
            routing_bytes,
            route_partials_bytes,
            compact_bytes,
        })
    }

    /// Validate the live row count against the owning kernel's capacity and
    /// contract constants before any device bytes are read.
    pub(crate) fn validate_capacity(
        &self,
        capacity_rows: u32,
        topk: u32,
        input_row_bytes: usize,
    ) -> Result<()> {
        ensure!(
            self.rows <= capacity_rows,
            "expert trace rows exceed the kernel capacity"
        );
        ensure!(topk == TRACE_TOPK, "expert trace expects six routes per row");
        ensure!(
            input_row_bytes == self.input_row_bytes,
            "expert trace input row bytes disagree with the kernel"
        );
        ensure!(
            input_row_bytes == self.hidden_bytes / self.rows as usize,
            "expert trace input rows are not contiguous"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertTraceWireIdentity {
    pub(crate) request_id: u64,
    pub(crate) placement_version: u64,
}

/// Immutable completed-buffer views for one captured request.
pub(crate) struct ExpertTraceCapture<'a> {
    pub(crate) wire: ExpertTraceWireIdentity,
    pub(crate) layer: usize,
    pub(crate) executor_id: u64,
    pub(crate) rank: usize,
    pub(crate) rows: u32,
    pub(crate) kernel_capacity_rows: u32,
    pub(crate) input_dtype: u32,
    pub(crate) input_row_bytes: usize,
    pub(crate) registered_destination: bool,
    /// Packed native input rows exactly as uploaded from the wire request.
    pub(crate) hidden: &'a [u8],
    /// Original wire route order; `rows * 6` entries.
    pub(crate) route_ids: &'a [i32],
    /// Original wire route order; `rows * 6` entries.
    pub(crate) routing: &'a [f32],
    /// Completed FP32 route partials `[rows, 6, 5120]` in original route order.
    pub(crate) route_partials: &'a [u8],
    /// Completed compact BF16 `[rows, 5120]` from the actual output buffer.
    pub(crate) compact: &'a [u8],
}

fn input_dtype_name(code: u32) -> &'static str {
    match code {
        1 => "bfloat16",
        7 => "fp8e4m3ue8m0k32",
        _ => "unknown",
    }
}

pub(crate) fn trace_dir_name(
    ordinal: u64,
    capture: &ExpertTraceCapture<'_>,
) -> String {
    format!(
        "expert-trace-ord{ordinal:06}-L{}-E{}-R{}-rows{}-req{}-pv{}",
        capture.layer,
        capture.executor_id,
        capture.rank,
        capture.rows,
        capture.wire.request_id,
        capture.wire.placement_version,
    )
}

impl ExpertTraceCapture<'_> {
    /// Write all capture files plus `metadata.json` under `root`. Every file
    /// is created new; on any error the files written so far are removed and
    /// the error is returned honestly. `metadata.json` is written last so its
    /// presence marks a complete trace.
    pub(crate) fn write(&self, root: &Path, ordinal: u64) -> Result<PathBuf> {
        let extents = ExpertTraceExtents::for_rows(self.rows, self.input_row_bytes, TRACE_TOPK)?;
        ensure!(
            self.hidden.len() == extents.hidden_bytes,
            "trace input payload extent mismatch: {} != {}",
            self.hidden.len(),
            extents.hidden_bytes
        );
        ensure!(
            self.route_ids.len() * 4 == extents.route_ids_bytes,
            "trace route id extent mismatch"
        );
        ensure!(
            self.routing.len() * 4 == extents.routing_bytes,
            "trace routing extent mismatch"
        );
        ensure!(
            self.route_partials.len() == extents.route_partials_bytes,
            "trace route partials extent mismatch: {} != {}",
            self.route_partials.len(),
            extents.route_partials_bytes
        );
        ensure!(
            self.compact.len() == extents.compact_bytes,
            "trace compact output extent mismatch: {} != {}",
            self.compact.len(),
            extents.compact_bytes
        );
        let dir = root.join(trace_dir_name(ordinal, self));
        fs::create_dir_all(&dir)
            .with_context(|| format!("creating trace directory {}", dir.display()))?;
        let mut created: Vec<PathBuf> = Vec::new();
        if let Err(error) = self.write_files(&dir, &extents, ordinal, &mut created) {
            // Remove only files we created; never touch pre-existing data.
            for path in &created {
                let _ = fs::remove_file(path);
            }
            let _ = fs::remove_dir(&dir);
            return Err(error);
        }
        Ok(dir)
    }

    fn create_new(
        dir: &Path,
        name: &str,
        created: &mut Vec<PathBuf>,
    ) -> Result<File> {
        let path = dir.join(name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("creating trace file {}", path.display()))?;
        created.push(path);
        Ok(file)
    }

    fn write_files(
        &self,
        dir: &Path,
        extents: &ExpertTraceExtents,
        ordinal: u64,
        created: &mut Vec<PathBuf>,
    ) -> Result<()> {
        // Wire uploads and checkpoints are little-endian; route ids and FP32
        // weights are dumped as their native little-endian byte views.
        ensure!(
            cfg!(target_endian = "little"),
            "expert trace requires little-endian storage"
        );
        Self::create_new(dir, "input-payload.bin", created)?.write_all(self.hidden)?;
        let mut ids = Self::create_new(dir, "route-ids.bin", created)?;
        for id in self.route_ids {
            ids.write_all(&id.to_le_bytes())?;
        }
        let mut routing = Self::create_new(dir, "route-routing-fp32.bin", created)?;
        for weight in self.routing {
            routing.write_all(&weight.to_le_bytes())?;
        }
        Self::create_new(dir, "route-partials-fp32.bin", created)?
            .write_all(self.route_partials)?;
        Self::create_new(dir, "compact-output-bf16.bin", created)?.write_all(self.compact)?;
        let metadata = self.metadata(extents, ordinal);
        let json = serde_json::to_string_pretty(&metadata).context("encoding trace metadata")?;
        Self::create_new(dir, "metadata.json", created)?.write_all(json.as_bytes())?;
        Ok(())
    }

    fn metadata(&self, extents: &ExpertTraceExtents, ordinal: u64) -> serde_json::Value {
        let rows = self.rows;
        let shape_2d = |inner: usize| serde_json::json!([rows, inner]);
        serde_json::json!({
            "schema": TRACE_SCHEMA,
            "trace_ordinal": ordinal,
            "process_id": std::process::id(),
            "env_var": TRACE_ENV_VAR,
            "layer": self.layer,
            "executor_id": self.executor_id,
            "rank": self.rank,
            "wire": {
                "request_id": self.wire.request_id,
                "placement_version": self.wire.placement_version,
            },
            "rows": rows,
            "kernel_capacity_rows": self.kernel_capacity_rows,
            "topk": TRACE_TOPK,
            "hidden_dim": TRACE_HIDDEN,
            "input_dtype": input_dtype_name(self.input_dtype),
            "input_dtype_code": self.input_dtype,
            "input_row_bytes": self.input_row_bytes,
            "compact_destination": if self.registered_destination {
                "registered_response_slot"
            } else {
                "compact_output_allocation"
            },
            "selection": {
                "layer": TRACE_LAYER,
                "executor_id": TRACE_EXECUTOR_ID,
                "rows": TRACE_ROW_CHOICES,
                "policy": "first-successful-request-per-row-shape-per-process",
            },
            "files": {
                "input-payload.bin": {
                    "dtype": input_dtype_name(self.input_dtype),
                    "layout": "packed-native-row-bytes",
                    "shape": shape_2d(self.input_row_bytes),
                    "bytes": extents.hidden_bytes,
                },
                "route-ids.bin": {
                    "dtype": "int32",
                    "shape": shape_2d(TRACE_TOPK as usize),
                    "bytes": extents.route_ids_bytes,
                },
                "route-routing-fp32.bin": {
                    "dtype": "float32",
                    "shape": shape_2d(TRACE_TOPK as usize),
                    "bytes": extents.routing_bytes,
                },
                "route-partials-fp32.bin": {
                    "dtype": "float32",
                    "shape": [rows, TRACE_TOPK, TRACE_HIDDEN],
                    "bytes": extents.route_partials_bytes,
                    "order": "original-wire-route-order",
                },
                "compact-output-bf16.bin": {
                    "dtype": "bfloat16",
                    "shape": shape_2d(TRACE_HIDDEN as usize),
                    "bytes": extents.compact_bytes,
                },
            },
        })
    }
}

impl ExpertExecution<'_, '_> {
    /// Bounded opt-in capture of one completed wire request per selected row
    /// shape. Called after the wave's synchronize, while the completed route
    /// partials and compact output are still owned and before the response is
    /// emitted or storage reused. Never fails the request; capture errors are
    /// logged and the claim released for a later request.
    pub(super) fn trace_completed_request(
        &self,
        request: &ds41rt_transport::v41_expert::V41BackboneRequest<'_>,
        executor_id: u64,
        exchange: &HostExpertExchange,
        output: Ds41rtDeviceBuffer,
        registered_destination: bool,
    ) {
        let Some(selection) = ExpertTraceSelection::global() else {
            return;
        };
        let ExpertLayer::Backbone { layer, rank } = self._weights.layer else {
            return;
        };
        let rows = request.rows();
        if !selection.is_candidate(layer, executor_id, rows) {
            return;
        }
        let (kernel, _, _) = self.execution_state(rows);
        if kernel.accumulates_tokens() {
            // A token-accumulating kernel has no per-route planes; reject the
            // selected trace instead of fabricating six of them.
            tracing::warn!(target: "ds41rt::expert_trace",
                layer, executor_id, rows,
                "expert trace rejected: token-accumulating kernel exposes no route planes");
            return;
        }
        let Some(claim) = selection.try_claim(rows) else {
            return; // another lane already won this row shape
        };
        match self.capture_claim(
            &claim,
            selection,
            request,
            layer,
            rank,
            executor_id,
            exchange,
            output,
            registered_destination,
        ) {
            Ok(dir) => tracing::info!(target: "ds41rt::expert_trace",
                path = %dir.display(), rows,
                "expert route trace captured"),
            Err(error) => {
                claim.release();
                tracing::warn!(target: "ds41rt::expert_trace",
                    %error, rows,
                    "expert route trace capture failed; claim released");
            }
        }
    }

    fn capture_claim(
        &self,
        claim: &ExpertTraceClaim<'_>,
        selection: &ExpertTraceSelection,
        request: &ds41rt_transport::v41_expert::V41BackboneRequest<'_>,
        layer: usize,
        rank: usize,
        executor_id: u64,
        exchange: &HostExpertExchange,
        output: Ds41rtDeviceBuffer,
        registered_destination: bool,
    ) -> Result<PathBuf> {
        let rows = claim.rows();
        let (kernel, _, _) = self.execution_state(rows);
        let info = kernel.info();
        let extents = ExpertTraceExtents::for_rows(rows, info.input_row_bytes()?, TRACE_TOPK)?;
        extents.validate_capacity(info.capacity_rows, info.topk, info.input_row_bytes()?)?;
        let routes = rows as usize * TRACE_TOPK as usize;
        ensure!(
            exchange.ids.len() >= routes && exchange.routing.len() >= routes,
            "host exchange routing extents are too short"
        );
        // The actual packed wire bytes uploaded into the hidden allocation.
        let hidden = request.hidden();
        ensure!(
            hidden.len() == extents.hidden_bytes,
            "wire input payload is {} bytes, expected {}",
            hidden.len(),
            extents.hidden_bytes
        );
        // Read completed device buffers while exclusively owned. The stream
        // drained before this call, so plain synchronous copies observe the
        // kernel results; no work is enqueued and nothing is recomputed.
        let partials_buffer = self.route_partials(rows)?;
        ensure!(
            partials_buffer.bytes == extents.route_partials_bytes,
            "route partials allocation is {} bytes, expected {}",
            partials_buffer.bytes,
            extents.route_partials_bytes
        );
        let mut route_partials = vec![0u8; extents.route_partials_bytes];
        let partials_len = route_partials.len();
        self.library.copy_d2h(
            &mut route_partials,
            Ds41rtDeviceBuffer {
                bytes: partials_len,
                ..partials_buffer
            },
        )?;
        let mut compact = vec![0u8; extents.compact_bytes];
        let compact_len = compact.len();
        self.library.copy_d2h(
            &mut compact,
            Ds41rtDeviceBuffer {
                bytes: compact_len,
                ..output
            },
        )?;
        let capture = ExpertTraceCapture {
            wire: ExpertTraceWireIdentity {
                request_id: request.request_id(),
                placement_version: request.placement_version(),
            },
            layer,
            executor_id,
            rank,
            rows,
            kernel_capacity_rows: info.capacity_rows,
            input_dtype: info.input_dtype,
            input_row_bytes: extents.input_row_bytes,
            registered_destination,
            hidden,
            route_ids: &exchange.ids[..routes],
            routing: &exchange.routing[..routes],
            route_partials: &route_partials,
            compact: &compact,
        };
        capture.write(selection.dir(), claim.ordinal())
    }
}
