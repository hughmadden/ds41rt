//! Read-only diagnostic capture of the completed compressor producer's exact
//! operands and outputs.
//!
//! [`CompressorWave::trace_completed`](super::CompressorWave::trace_completed)
//! is the single explicit call site: the target pass invokes it once per
//! selected detail layer, after the layer's cache producers completed in both
//! the synchronous and cooperative paths, before any commit mutates pending
//! device state, and with no CUDA graph capture open on the wave's own stream.
//! Nothing here changes arithmetic, dispatch, or the default execution path:
//! unselected layers and untraced passes perform no extra synchronization, no
//! device-to-host copy, and no new allocation.
//!
//! The module is split so CPU tests cover the real production logic without a
//! device: completed-identity guards, chunk validation, extent planning,
//! per-row predecessor resolution and file writing are pure or
//! callback-driven; only the final `copy_d2h` binding in
//! [`capture_compressor_inputs`] needs a library.

use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use std::path::Path;

use super::CompressorLatentRow;

/// Hard bound on total captured bytes per invocation (weight tensors and host
/// descriptor bytes included). A capture that would exceed it fails explicitly
/// instead of dumping shared pools or truncating silently.
pub(crate) const TRACE_BYTE_BUDGET: usize = 64 << 20;

/// Fixed compressor geometry shared by the planner, the writer and the manifest.
pub(crate) const SOURCE_DIM: usize = 5120;
pub(crate) const LATENT_DIM: usize = 512;
/// Pending FP32 planes are `[slot_count, 512]` (kv and gate scores).
pub(crate) const PENDING_ROW_BYTES: usize = LATENT_DIM * 4;
/// Packed FP4 values per compressed row (`V41Kv::COMPRESSED_VALUE_BYTES`).
const KV_VALUE_BYTES: usize = 256;
/// E4M3 group scales per compressed row (`V41Kv::COMPRESSED_SCALE_BYTES`).
const KV_SCALE_BYTES: usize = 32;
/// Device maximum slot count (`CompressorState` bounds slots to 16).
pub(crate) const MAX_SLOTS: usize = 16;
/// Even-position rows carry no pooling predecessor; the producer uploads this
/// sentinel descriptor for them.
pub(crate) const DESCRIPTOR_SENTINEL: u64 = u64::MAX;

/// Wave-owned producer buffers for one completed invocation (plain metadata
/// only; every extent is revalidated by the planner before any copy).
#[derive(Clone, Copy)]
pub(crate) struct CompressorTraceBuffers {
    /// Packed BF16 input rows the projection read (the wave's D2D copy of the
    /// query hidden rows).
    pub input: Ds41rtDeviceBuffer,
    /// Projection output: FP32 rows at ratio two, BF16 rows at ratio one.
    pub projected: Ds41rtDeviceBuffer,
    /// Ratio-two gate scores (FP32), absent at ratio one.
    pub scores: Option<Ds41rtDeviceBuffer>,
    /// Final pooled/normalized BF16 rows exactly as `kv.pack` and the index
    /// projection read them. The trace reads this unchanged input buffer after
    /// the producer (including packing) has completed and before commit.
    pub output: Ds41rtDeviceBuffer,
    pub frequencies: Ds41rtDeviceBuffer,
    /// Device U64 positions upload: completed latents' first absolute token,
    /// zero padding elsewhere.
    pub positions: Ds41rtDeviceBuffer,
    /// Device U64 predecessor descriptors (ratio two only).
    pub descriptors: Option<Ds41rtDeviceBuffer>,
    pub kv_values: Ds41rtDeviceBuffer,
    pub kv_scales: Ds41rtDeviceBuffer,
    /// Whole pending-state FP32 planes `[slot_count, 512]` (ratio two only).
    pub pending_kv: Option<Ds41rtDeviceBuffer>,
    pub pending_scores: Option<Ds41rtDeviceBuffer>,
}

/// Checkpoint weight tensors needed for later component replay, resolved from
/// the already-loaded per-layer compressor weights (never a checkpoint reload).
#[derive(Clone)]
pub(crate) struct CompressorTraceWeights {
    /// `layers.N.attn.compressor.wkv.weight`, BF16 `[512, 5120]`.
    pub wkv: Ds41rtDeviceBuffer,
    /// `layers.N.attn.compressor.norm.weight`, BF16 `[512]`.
    pub norm: Ds41rtDeviceBuffer,
    /// `layers.N.attn.compressor.wgate.weight`, BF16 `[512, 5120]`, ratio two only.
    pub wgate: Option<Ds41rtDeviceBuffer>,
    /// Full checkpoint tensor names, for the manifest records.
    pub names: [String; 3],
}

/// Per-chunk host provenance of the completed proposal, resolved by the wave
/// from its validated `Prepared` metadata and the owning state.
#[derive(Clone)]
pub(crate) struct CompressorTraceChunk {
    /// Ordinal of this chunk inside the invocation, in native chunk order.
    pub index: usize,
    /// Actual bank request identity of the owning lease.
    pub request_id: u64,
    pub slot: usize,
    pub generation: u64,
    pub version: u64,
    /// Committed absolute position of the chunk's first token.
    pub position: u64,
    pub tokens: u32,
    /// Prepared offset: the chunk's first flattened wave row.
    pub offset: usize,
}

/// Validate the wave/state identity a trace may read. Pure so CPU tests
/// exercise the exact production guard: no pending query or commit, a
/// completed ready output owned by this state, and matching source layers.
pub(crate) fn ensure_completed_trace_ready(
    pending_query: bool,
    pending_commit: bool,
    ready_owner: Option<u64>,
    state_owner: u64,
    wave_layer: usize,
    state_layer: usize,
) -> Result<()> {
    ensure!(
        !pending_query,
        "compressor trace requires a settled query producer"
    );
    ensure!(!pending_commit, "compressor trace requires a settled commit");
    let owner = ready_owner.context("compressor output incomplete")?;
    ensure!(
        owner == state_owner,
        "compressor trace owner differs from state"
    );
    ensure!(
        wave_layer == state_layer,
        "compressor trace layer differs from state"
    );
    Ok(())
}

/// Where a ratio-two wave row's pooling predecessor actually lives, decoded
/// from the actual device descriptor value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Predecessor {
    /// Descriptor `< slot_count`: the row pools the pending state of that slot.
    PendingSlot(usize),
    /// Descriptor `slot_count + r` with `r < row`: the row pools earlier wave
    /// row `r` of this same invocation (an already-projected current row).
    EarlierWaveRow(usize),
    /// Even-position sentinel: the row completes no latent and pools nothing.
    Invalid,
    /// Any other value: not producible by a consistent producer; recorded as
    /// an explicit mismatch instead of being silently reinterpreted.
    Unresolved(u64),
}

/// Resolve one row's predecessor from the actual device descriptor. Pure and
/// total: unresolvable values are reported, never substituted.
pub(crate) fn resolve_predecessor(descriptor: u64, row: usize, slot_count: usize) -> Predecessor {
    if descriptor == DESCRIPTOR_SENTINEL {
        return Predecessor::Invalid;
    }
    if descriptor < slot_count as u64 {
        return Predecessor::PendingSlot(descriptor as usize);
    }
    let earlier = descriptor - slot_count as u64;
    if earlier < row as u64 {
        return Predecessor::EarlierWaveRow(earlier as usize);
    }
    Predecessor::Unresolved(descriptor)
}

/// One validated fixed-extent read. Pointer-free so planning and its tests
/// never touch a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlannedRead {
    pub name: &'static str,
    pub file: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub row_bytes: usize,
    pub bytes: usize,
}

/// Plan one fixed-extent read against its owning allocation.
fn push_read(
    reads: &mut Vec<PlannedRead>,
    layer: usize,
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    row_bytes: usize,
    allocation: Ds41rtDeviceBuffer,
) -> Result<()> {
    let rows = shape.first().copied().unwrap_or(0);
    let bytes = rows
        .checked_mul(row_bytes)
        .with_context(|| format!("compressor trace {name} extent overflow"))?;
    ensure!(
        !allocation.ptr.is_null() && bytes <= allocation.bytes,
        "compressor trace {name} needs {bytes} bytes, over its {} byte allocation",
        allocation.bytes
    );
    reads.push(PlannedRead {
        name,
        file: format!("layer{layer}-compressor-{name}.bin"),
        dtype,
        shape,
        row_bytes,
        bytes,
    });
    Ok(())
}

/// Plan every fixed-extent read for one invocation, including the checkpoint
/// weight tensors when they have not already been written for this trace root.
/// Pure: validates every extent against its owning allocation and the total
/// against the byte budget, before any copy is issued.
pub(crate) fn plan_compressor_reads(
    layer: usize,
    ratio: usize,
    rows: usize,
    slot_count: usize,
    buffers: &CompressorTraceBuffers,
    weights: Option<&CompressorTraceWeights>,
) -> Result<Vec<PlannedRead>> {
    ensure!(rows > 0, "compressor trace requires at least one live row");
    ensure!(
        (1..=MAX_SLOTS).contains(&slot_count),
        "compressor trace slot count {slot_count} exceeds the device bound {MAX_SLOTS}"
    );
    let mut reads: Vec<PlannedRead> = Vec::new();
    push_read(&mut reads, layer, "input", "bfloat16", vec![rows, SOURCE_DIM],
        SOURCE_DIM * 2, buffers.input)?;
    match ratio {
        2 => {
            push_read(&mut reads, layer, "projected", "float32", vec![rows, LATENT_DIM],
                LATENT_DIM * 4, buffers.projected)?;
            let scores = buffers
                .scores
                .as_ref()
                .context("compressor trace scores absent at ratio two")?;
            push_read(&mut reads, layer, "scores", "float32", vec![rows, LATENT_DIM],
                LATENT_DIM * 4, *scores)?;
            let descriptors = buffers
                .descriptors
                .as_ref()
                .context("compressor trace descriptors absent at ratio two")?;
            push_read(&mut reads, layer, "descriptors-device", "uint64", vec![rows], 8,
                *descriptors)?;
            // Whole bounded planes, exactly as the pool kernel addresses them.
            for (name, buffer) in [
                ("pending-kv", &buffers.pending_kv),
                ("pending-scores", &buffers.pending_scores),
            ] {
                let buffer = buffer
                    .as_ref()
                    .with_context(|| format!("compressor trace {name} absent at ratio two"))?;
                push_read(&mut reads, layer, name, "float32",
                    vec![slot_count, LATENT_DIM], PENDING_ROW_BYTES, *buffer)?;
            }
        }
        1 => {
            push_read(&mut reads, layer, "projected", "bfloat16", vec![rows, LATENT_DIM],
                LATENT_DIM * 2, buffers.projected)?;
            ensure!(
                buffers.scores.is_none()
                    && buffers.descriptors.is_none()
                    && buffers.pending_kv.is_none()
                    && buffers.pending_scores.is_none(),
                "compressor trace ratio-one wave owns ratio-two scratch"
            );
        }
        _ => anyhow::bail!("compressor trace ratio {ratio} is not a producer ratio"),
    }
    push_read(&mut reads, layer, "output", "bfloat16", vec![rows, LATENT_DIM],
        LATENT_DIM * 2, buffers.output)?;
    push_read(&mut reads, layer, "frequencies", "float32", vec![rows, 32, 2], 32 * 2 * 4,
        buffers.frequencies)?;
    push_read(&mut reads, layer, "positions", "uint64", vec![rows], 8, buffers.positions)?;
    push_read(&mut reads, layer, "kv-values", "fp4e2m1", vec![rows, KV_VALUE_BYTES * 2],
        KV_VALUE_BYTES, buffers.kv_values)?;
    push_read(&mut reads, layer, "kv-scales", "fp8e4m3", vec![rows, KV_SCALE_BYTES],
        KV_SCALE_BYTES, buffers.kv_scales)?;
    if let Some(weights) = weights {
        // Extent identity pins each tensor to its expected checkpoint layout.
        let expect = |tensor: Ds41rtDeviceBuffer, rows: usize, row_bytes: usize| -> Result<()> {
            let bytes = rows
                .checked_mul(row_bytes)
                .context("compressor trace weight extent overflow")?;
            ensure!(!tensor.ptr.is_null(), "compressor trace weight tensor is null");
            ensure!(
                tensor.bytes == bytes,
                "compressor trace weight tensor has {} bytes, expected the {rows}x{row_bytes} layout",
                tensor.bytes
            );
            Ok(())
        };
        expect(weights.wkv, LATENT_DIM, SOURCE_DIM * 2)?;
        expect(weights.norm, LATENT_DIM, 2)?;
        // Wgate exists only at ratio two; a ratio-one wave must not carry it.
        let wgate = if ratio == 2 {
            let wgate = weights
                .wgate
                .as_ref()
                .context("compressor trace wgate absent at ratio two")?;
            expect(*wgate, LATENT_DIM, SOURCE_DIM * 2)?;
            Some(*wgate)
        } else {
            ensure!(
                weights.wgate.is_none(),
                "compressor trace ratio-one weights own wgate"
            );
            None
        };
        push_read(&mut reads, layer, "wkv-weight", "bfloat16",
            vec![LATENT_DIM, SOURCE_DIM], SOURCE_DIM * 2, weights.wkv)?;
        push_read(&mut reads, layer, "norm-weight", "bfloat16", vec![LATENT_DIM], 2,
            weights.norm)?;
        if let Some(wgate) = wgate {
            push_read(&mut reads, layer, "wgate-weight", "bfloat16",
                vec![LATENT_DIM, SOURCE_DIM], SOURCE_DIM * 2, wgate)?;
        }
    }
    let total = total_bytes(&reads)?;
    ensure!(
        total <= TRACE_BYTE_BUDGET,
        "compressor input trace needs {total} bytes, over the {TRACE_BYTE_BUDGET} byte budget; \
         refusing to dump or truncate"
    );
    let unique: std::collections::HashSet<_> = reads.iter().map(|read| read.file.clone()).collect();
    ensure!(
        unique.len() == reads.len(),
        "compressor trace file names collide"
    );
    Ok(reads)
}

fn total_bytes(reads: &[PlannedRead]) -> Result<usize> {
    reads.iter().try_fold(0usize, |total, read| {
        total
            .checked_add(read.bytes)
            .context("compressor trace extent overflow")
    })
}

/// Checked slice of an allocation; every device read goes through it.
fn slice_buffer(
    buffer: Ds41rtDeviceBuffer,
    offset: usize,
    bytes: usize,
) -> Result<Ds41rtDeviceBuffer> {
    ensure!(
        !buffer.ptr.is_null(),
        "compressor trace read references a null buffer"
    );
    ensure!(
        offset <= buffer.bytes && bytes <= buffer.bytes - offset,
        "compressor trace read [{offset}, {bytes}+{offset}) exceeds {} byte allocation",
        buffer.bytes
    );
    Ok(Ds41rtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
        bytes,
        ..buffer
    })
}

/// Validate the resolved chunk provenance of a completed invocation: unique
/// non-empty chunks, contiguous prepared offsets, and slots inside the state
/// bound. Pure; returns the total live wave rows.
pub(crate) fn validate_trace_chunks(
    chunks: &[CompressorTraceChunk],
    slot_count: usize,
) -> Result<usize> {
    ensure!(!chunks.is_empty(), "compressor trace has no chunks");
    let mut seen = [false; MAX_SLOTS];
    let mut rows = 0usize;
    for chunk in chunks {
        ensure!(chunk.tokens > 0, "compressor trace chunk is empty");
        ensure!(
            chunk.slot < slot_count && slot_count <= MAX_SLOTS,
            "compressor trace chunk slot exceeds the state bound"
        );
        ensure!(
            !std::mem::replace(&mut seen[chunk.slot], true),
            "duplicate compressor trace chunk slot"
        );
        ensure!(
            chunk.offset == rows,
            "compressor trace chunk offsets are not contiguous"
        );
        rows = rows
            .checked_add(chunk.tokens as usize)
            .context("compressor trace row overflow")?;
    }
    Ok(rows)
}

/// Perform the capture and write every file plus the manifest. `copy` performs
/// device reads (production binds the library's synchronous `copy_d2h`; tests
/// bind a fake). `host_descriptors` are the staged host descriptor values of
/// this invocation (ratio two only); the device copy stays authoritative and
/// is compared against them. `weights` is `None` when the weight tensors were
/// already written for this trace root. Returns the total captured bytes.
pub(crate) fn write_compressor_inputs(
    copy: &mut dyn FnMut(Ds41rtDeviceBuffer, &mut [u8]) -> Result<()>,
    directory: &Path,
    layer: usize,
    ratio: usize,
    slot_count: usize,
    snapshot: u64,
    chunks: &[CompressorTraceChunk],
    completed: &[CompressorLatentRow],
    host_descriptors: Option<&[u64]>,
    buffers: &CompressorTraceBuffers,
    weights_directory: &Path,
    weights: Option<&CompressorTraceWeights>,
) -> Result<usize> {
    let rows = validate_trace_chunks(chunks, slot_count)?;
    if ratio == 2 {
        let host = host_descriptors
            .context("compressor trace host descriptors absent at ratio two")?;
        ensure!(
            host.len() == rows,
            "compressor trace host descriptor count {} differs from {rows} rows",
            host.len()
        );
    } else {
        ensure!(
            host_descriptors.is_none(),
            "compressor trace ratio one has no descriptors"
        );
    }
    for latent in completed {
        ensure!(
            (latent.source_row as usize) < rows,
            "compressor trace completed row exceeds the wave"
        );
    }
    // Every extent, weight identity and the cumulative budget (host
    // descriptor bytes included) is validated before the first copy is issued.
    let reads = plan_compressor_reads(layer, ratio, rows, slot_count, buffers, weights)?;
    let mut captured_bytes = total_bytes(&reads)?;
    if let Some(host) = host_descriptors {
        captured_bytes = captured_bytes
            .checked_add(host.len() * 8)
            .context("compressor trace extent overflow")?;
        ensure!(
            captured_bytes <= TRACE_BYTE_BUDGET,
            "compressor input trace host descriptors exceed the {TRACE_BYTE_BUDGET} byte budget"
        );
    }
    let allocation_of = |name: &str| -> Result<Ds41rtDeviceBuffer> {
        Ok(match name {
            "input" => buffers.input,
            "projected" => buffers.projected,
            "scores" => buffers.scores.context("scores read without buffer")?,
            "output" => buffers.output,
            "frequencies" => buffers.frequencies,
            "positions" => buffers.positions,
            "descriptors-device" => {
                buffers.descriptors.context("descriptor read without buffer")?
            }
            "kv-values" => buffers.kv_values,
            "kv-scales" => buffers.kv_scales,
            "pending-kv" => buffers.pending_kv.context("pending kv read without buffer")?,
            "pending-scores" => {
                buffers.pending_scores.context("pending scores read without buffer")?
            }
            "wkv-weight" | "norm-weight" | "wgate-weight" => {
                let weights = weights
                    .with_context(|| format!("compressor trace {name} without weights"))?;
                match name {
                    "wkv-weight" => weights.wkv,
                    "norm-weight" => weights.norm,
                    _ => weights.wgate.context("wgate read without buffer")?,
                }
            }
            other => anyhow::bail!("compressor trace read {other} has no allocation"),
        })
    };
    let mut captured: Vec<(PlannedRead, Vec<u8>)> = Vec::with_capacity(reads.len());
    for read in &reads {
        let mut bytes = vec![0u8; read.bytes];
        copy(slice_buffer(allocation_of(read.name)?, 0, read.bytes)?, &mut bytes)?;
        captured.push((read.clone(), bytes));
    }
    // Actual device descriptors stay authoritative: per-row predecessors and
    // the host comparison both resolve from the copied device bytes.
    let device_descriptors = match host_descriptors {
        Some(host) => {
            let (_, bytes) = captured
                .iter()
                .find(|(read, _)| read.name == "descriptors-device")
                .context("compressor trace descriptor capture absent")?;
            ensure!(
                bytes.len() == host.len() * 8,
                "compressor trace descriptor extent differs between device and host"
            );
            let device: Vec<u64> = bytes
                .chunks_exact(8)
                .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()))
                .collect();
            let matches = device.as_slice() == host;
            Some((device, matches))
        }
        None => None,
    };
    let manifest = manifest(
        layer, ratio, rows, slot_count, snapshot, chunks, completed, &captured,
        device_descriptors
            .as_ref()
            .map(|(device, matches)| (device.as_slice(), *matches)),
        weights_directory, weights, captured_bytes,
    )?;
    // Weight files land in the shared weights directory (create_dir_all, the
    // same once-per-root rule the projection trace uses); operands in the
    // batch directory. create_new refuses replacement in both.
    if weights.is_some() {
        std::fs::create_dir_all(weights_directory)
            .with_context(|| format!("creating compressor trace weights directory"))?;
    }
    for (read, bytes) in &captured {
        let target = match read.name {
            "wkv-weight" | "norm-weight" | "wgate-weight" => weights_directory,
            _ => directory,
        };
        write_new_file(target, &read.file, bytes)?;
    }
    if let Some(host) = host_descriptors {
        let staged: Vec<u8> = host.iter().flat_map(|value| value.to_ne_bytes()).collect();
        write_new_file(
            directory,
            &format!("layer{layer}-compressor-descriptors-host.bin"),
            &staged,
        )?;
    }
    write_new_file(
        directory,
        &format!("layer{layer}-compressor-inputs.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(captured_bytes)
}

/// Manifest record for one dumped buffer. Every field derives from the
/// validated live extent, so the JSON always matches the raw file on disk.
fn record(read: &PlannedRead) -> serde_json::Value {
    serde_json::json!({
        "name": read.name,
        "file": read.file,
        "dtype": read.dtype,
        "shape": read.shape,
        "bytes": read.bytes,
    })
}

fn predecessor_json(predecessor: Predecessor) -> serde_json::Value {
    match predecessor {
        Predecessor::PendingSlot(slot) => serde_json::json!({
            "kind": "pending_slot", "slot": slot,
        }),
        Predecessor::EarlierWaveRow(earlier) => serde_json::json!({
            "kind": "earlier_wave_row", "wave_row": earlier,
        }),
        Predecessor::Invalid => serde_json::json!({ "kind": "invalid_sentinel" }),
        Predecessor::Unresolved(value) => serde_json::json!({
            "kind": "unresolved", "value": value,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn manifest(
    layer: usize,
    ratio: usize,
    rows: usize,
    slot_count: usize,
    snapshot: u64,
    chunks: &[CompressorTraceChunk],
    completed: &[CompressorLatentRow],
    captured: &[(PlannedRead, Vec<u8>)],
    device_descriptors: Option<(&[u64], bool)>,
    weights_directory: &Path,
    weights: Option<&CompressorTraceWeights>,
    captured_bytes: usize,
) -> Result<serde_json::Value> {
    let find = |name: &str| {
        captured
            .iter()
            .find(|(read, _)| read.name == name)
            .map(|(read, _)| record(read))
            .with_context(|| format!("compressor trace {name} capture absent"))
    };
    let mut completed_by_row = std::collections::BTreeMap::new();
    for latent in completed {
        completed_by_row.insert(latent.source_row as usize, latent);
    }
    // Per-row resolution: the actual device descriptor first, absolute
    // position and completed-latent mapping second.
    let mut wave_rows = Vec::with_capacity(rows);
    for row in 0..rows {
        let chunk = chunks
            .iter()
            .find(|chunk| chunk.offset <= row && row < chunk.offset + chunk.tokens as usize)
            .context("compressor trace row belongs to no chunk")?;
        let position = chunk.position + (row - chunk.offset) as u64;
        let (descriptor, predecessor) = match device_descriptors {
            Some((device, _)) => {
                let value = device[row];
                (value, resolve_predecessor(value, row, slot_count))
            }
            None => (DESCRIPTOR_SENTINEL, Predecessor::Invalid),
        };
        let latent = completed_by_row.get(&row).map(|latent| {
            // 0-based logical compressed row completed by this wave row: the
            // latent's first token divided by the source ratio.
            serde_json::json!({
                "source_row": latent.source_row,
                "first_token": latent.position,
                "logical_compressed_row": latent.position / ratio as u64,
                "request_id": chunks
                    .iter()
                    .find(|chunk| chunk.slot == latent.lease.slot)
                    .map(|chunk| chunk.request_id),
            })
        });
        wave_rows.push(serde_json::json!({
            "row": row,
            "chunk": chunk.index,
            "request_id": chunk.request_id,
            "absolute_position": position,
            "device_descriptor": descriptor,
            "predecessor": predecessor_json(predecessor),
            "completed_latent": latent,
        }));
    }
    let descriptors_json = match device_descriptors {
        Some((device, matches)) => serde_json::json!({
            "device_file": format!("layer{layer}-compressor-descriptors-device.bin"),
            "host_file": format!("layer{layer}-compressor-descriptors-host.bin"),
            "dtype": "uint64",
            "rows": device.len(),
            "device_matches_host": matches,
            "note": "actual device bytes are authoritative; host bytes are the staged upload only",
        }),
        None => serde_json::Value::Null,
    };
    let pending_json = if ratio == 2 {
        serde_json::json!({
            "kv": find("pending-kv")?,
            "scores": find("pending-scores")?,
            "dtype": "float32",
            "shape": [slot_count, LATENT_DIM],
            "slots": (0..slot_count).map(|slot| {
                let chunk = chunks.iter().find(|chunk| chunk.slot == slot);
                serde_json::json!({
                    "slot": slot,
                    "state": if chunk.is_some() { "active" } else { "unscored" },
                    "request_id": chunk.map(|chunk| chunk.request_id),
                    "generation": chunk.map(|chunk| chunk.generation),
                })
            }).collect::<Vec<_>>(),
            "note": "whole bounded planes; inactive slots are UNSCORED padding",
        })
    } else {
        serde_json::Value::Null
    };
    let weights_json = match weights {
        Some(weights) => {
            let mut tensors = vec![serde_json::json!({
                "tensor": weights.names[0], "dtype": "bfloat16",
                "shape": [LATENT_DIM, SOURCE_DIM], "bytes": weights.wkv.bytes,
                "file": format!("layer{layer}-compressor-wkv-weight.bin"),
            })];
            if ratio == 2 {
                tensors.push(serde_json::json!({
                    "tensor": weights.names[2], "dtype": "bfloat16",
                    "shape": [LATENT_DIM, SOURCE_DIM],
                    "bytes": weights.wgate.as_ref().unwrap().bytes,
                    "file": format!("layer{layer}-compressor-wgate-weight.bin"),
                }));
            }
            tensors.push(serde_json::json!({
                "tensor": weights.names[1], "dtype": "bfloat16",
                "shape": [LATENT_DIM], "bytes": weights.norm.bytes,
                "file": format!("layer{layer}-compressor-norm-weight.bin"),
            }));
            serde_json::json!({
                "directory": weights_directory,
                "tensors": tensors,
                "note": "already-loaded per-layer buffers, copied read-only; the checkpoint is never reloaded",
            })
        }
        None => serde_json::json!({
            "directory": weights_directory,
            "already_written_for_root": true,
        }),
    };
    Ok(serde_json::json!({
        "schema": 1,
        "kind": "compressor-inputs",
        "layer": layer,
        "ratio": ratio,
        "rows": rows,
        "slot_count": slot_count,
        "proposal_snapshot": snapshot,
        "chunks": chunks.iter().map(|chunk| serde_json::json!({
            "index": chunk.index,
            "request_id": chunk.request_id,
            "lease": { "slot": chunk.slot, "generation": chunk.generation },
            "version": chunk.version,
            "position": chunk.position,
            "tokens": chunk.tokens,
            "prepared_offset": chunk.offset,
            "flattened_row_range": [chunk.offset, chunk.offset + chunk.tokens as usize],
            "absolute_positions": (chunk.position..chunk.position + u64::from(chunk.tokens))
                .collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "buffers": {
            "input": find("input")?,
            "projected": find("projected")?,
            "scores": if ratio == 2 { find("scores")? } else { serde_json::Value::Null },
            "output_before_kv_pack": find("output")?,
            "frequencies": find("frequencies")?,
            "positions": find("positions")?,
            "kv_values": find("kv-values")?,
            "kv_scales": find("kv-scales")?,
        },
        "descriptors": descriptors_json,
        "pending": pending_json,
        "wave_rows": wave_rows,
        "weights": weights_json,
        "budget_bytes": TRACE_BYTE_BUDGET,
        "captured_bytes": captured_bytes,
        "notes": [
            "wave rows 0..rows are producer rows of this invocation only; a logical compressed row (e.g. row 20 of a source layer) is derived solely through wave_rows[].completed_latent.logical_compressed_row",
            "positions carries each completed latent's first absolute token; uploaded zero padding marks rows that complete no latent",
            "pending planes are whole [slot_count,512] state; slots not in this wave are UNSCORED padding and must not be interpreted",
            "the input file is this wave's D2D copy of the query hidden rows; compare against layerN-query-hidden.bin to prove D2D identity",
            "descriptors and every operand are actual device bytes; host staging is recorded only for comparison",
        ],
    }))
}

fn write_new_file(directory: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(name))
        .with_context(|| format!("creating trace file {name}"))?;
    file.write_all(bytes)?;
    Ok(())
}

/// Production entry: bind the library's synchronous device-to-host copy. The
/// wave has already synchronized its own stream before calling.
pub(crate) fn capture_compressor_inputs(
    library: &NativeLibrary,
    directory: &Path,
    layer: usize,
    ratio: usize,
    slot_count: usize,
    snapshot: u64,
    chunks: &[CompressorTraceChunk],
    completed: &[CompressorLatentRow],
    host_descriptors: Option<&[u64]>,
    buffers: &CompressorTraceBuffers,
    weights_directory: &Path,
    weights: Option<&CompressorTraceWeights>,
) -> Result<usize> {
    write_compressor_inputs(
        &mut |buffer, destination| library.copy_d2h(destination, buffer),
        directory,
        layer,
        ratio,
        slot_count,
        snapshot,
        chunks,
        completed,
        host_descriptors,
        buffers,
        weights_directory,
        weights,
    )
}
