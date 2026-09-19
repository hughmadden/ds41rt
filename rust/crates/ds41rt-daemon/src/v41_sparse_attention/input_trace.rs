//! Read-only diagnostic capture of the exact native sparse-attention operands.
//!
//! The wave arms this capture at most once per staged execution (see
//! [`InputTraceTrigger`]). When armed, the wave synchronizes its own stream
//! after every operand upload is queued and before tail preparation, warmup,
//! graph capture or launch, then copies the actual device operands the native
//! kernel is about to read. Nothing here changes arithmetic, dispatch, or the
//! default execution path: with the trigger unset there is no extra
//! synchronization, no device-to-host copy, and no new allocation.
//!
//! The module is split so CPU tests cover the real production logic without a
//! device: extent planning, kernel-exact referenced-row resolution, and file
//! writing are pure or callback-driven; only the final `copy_d2h` binding in
//! [`capture_attention_inputs`] needs a library.
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41Kv, V41SparseBatch};
use std::collections::BTreeMap;
use std::path::Path;

use super::RequestLaunch;

/// Hard bound on total captured bytes per invocation. A capture that would
/// exceed it fails explicitly instead of dumping the shared pools or
/// truncating silently.
pub(crate) const TRACE_BYTE_BUDGET: usize = 64 << 20;

/// Ring rows per window slot; fixed by the native window cache layout.
const RING_ROWS: usize = 128;
const RING_VALUE_BYTES: usize = 512;
const RING_SCALE_BYTES: usize = 16;
const SOURCE_ROW_VALUES: usize = V41Kv::COMPRESSED_VALUE_BYTES;
const SOURCE_ROW_SCALES: usize = V41Kv::COMPRESSED_SCALE_BYTES;

/// One-shot diagnostic capture trigger owned by the wave. The target pass arms
/// it for at most the next staged execution and clears it every layer; the
/// wave consumes it on entry to the staged execution so an error or unwind can
/// never leak the trigger into later work. No global mutable state.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct InputTraceTrigger(Option<std::path::PathBuf>);
impl InputTraceTrigger {
    pub fn arm(&mut self, directory: &Path) {
        self.0 = Some(directory.to_path_buf());
    }
    pub fn disarm(&mut self) {
        self.0 = None;
    }
    pub fn take(&mut self) -> Option<std::path::PathBuf> {
        self.0.take()
    }
    pub fn armed(&self) -> bool {
        self.0.is_some()
    }
}

/// Which live allocation a planned read copies from. Pointer-free so planning
/// and its tests never touch a device; the executor resolves each tag to the
/// real (already validated) launch view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadSource {
    Query,
    Sink,
    Metadata,
    Selected,
    /// Wave-wide bounds upload (descriptor batch) or a request's own slice.
    ReplayBegins,
    Descriptors,
    RingValues,
    RingScales,
    WindowEnd,
    WindowProposalValues,
    WindowProposalScales,
    SourcePages,
    SourceEnd,
    SourceProposalValues,
    SourceProposalScales,
    SourceValues,
    SourceScales,
}

/// Content synthesized by the trace itself instead of read from a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SyntheticContent {
    /// The launch passes no bounds pointer, so the kernel reads an implicit
    /// zero lower bound; the buffer is not read to avoid unpinned bytes.
    ExplicitZero,
}

/// One validated fixed-extent read. `offset`/`bytes` describe the live window
/// inside the owning allocation, never a raw pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlannedRead {
    pub source: ReadSource,
    /// Request slot for per-request operands; `None` for wave-wide operands.
    pub request: Option<usize>,
    pub name: &'static str,
    pub file: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub bytes: usize,
    pub synthetic: Option<SyntheticContent>,
}

/// Wave-owned operand buffers for one staged execution (plain metadata only).
pub(crate) struct WaveOperandBuffers {
    pub query: Ds41rtDeviceBuffer,
    pub sink: Ds41rtDeviceBuffer,
    pub metadata: Ds41rtDeviceBuffer,
    pub selected: Option<Ds41rtDeviceBuffer>,
    pub replay_begins: Ds41rtDeviceBuffer,
    pub descriptors: Ds41rtDeviceBuffer,
}

/// Host-side request identity for the manifest: real bank request id and the
/// committed absolute positions of this request's flattened rows.
pub(crate) struct RequestProvenance {
    pub request_id: u64,
    pub positions: Vec<u64>,
}

/// Resolve a read to its full owning allocation. The same resolver backs
/// planning and execution, so every extent check applies to the exact buffer
/// the copy reads.
fn allocation_of(
    source: ReadSource,
    request: Option<usize>,
    launches: &[RequestLaunch],
    buffers: &WaveOperandBuffers,
) -> Result<(Ds41rtDeviceBuffer, &'static str)> {
    let launch = |request: Option<usize>| {
        launches
            .get(request.context("attention trace read is missing its request")?)
            .context("attention trace request launch absent")
    };
    let source_view = |request: Option<usize>| {
        launch(request)?
            .source
            .as_ref()
            .context("attention trace source absent")
    };
    Ok(match source {
        ReadSource::Query => (buffers.query, "query"),
        ReadSource::Sink => (buffers.sink, "sink"),
        ReadSource::Metadata => (buffers.metadata, "metadata"),
        ReadSource::Selected => (
            buffers
                .selected
                .context("attention trace selected buffer absent")?,
            "selected",
        ),
        ReadSource::ReplayBegins => match request {
            // Batched launches read the wave-wide bounds upload for every row.
            None => (buffers.replay_begins, "replay-begins"),
            Some(_) => (
                launch(request)?
                    .window
                    .replay_begins
                    .context("attention trace bounds were never uploaded")?,
                "replay-begins",
            ),
        },
        ReadSource::Descriptors => (buffers.descriptors, "descriptors"),
        ReadSource::RingValues => (launch(request)?.window.values, "ring-values"),
        ReadSource::RingScales => (launch(request)?.window.scales, "ring-scales"),
        ReadSource::WindowEnd => (launch(request)?.window.end, "window-end"),
        ReadSource::WindowProposalValues => {
            (launch(request)?.window.proposals, "window-proposal-values")
        }
        ReadSource::WindowProposalScales => (
            launch(request)?.window.proposal_scales,
            "window-proposal-scales",
        ),
        ReadSource::SourcePages => (source_view(request)?.pages, "source-pages"),
        ReadSource::SourceEnd => (source_view(request)?.end, "source-end"),
        ReadSource::SourceProposalValues => {
            (source_view(request)?.proposals, "source-proposal-values")
        }
        ReadSource::SourceProposalScales => (
            source_view(request)?.proposal_scales,
            "source-proposal-scales",
        ),
        ReadSource::SourceValues => (source_view(request)?.values, "source-values"),
        ReadSource::SourceScales => (source_view(request)?.scales, "source-scales"),
    })
}

fn trace_file(layer: usize, name: &str, request: Option<usize>) -> String {
    match request {
        None => format!("layer{layer}-attention-{name}.bin"),
        Some(slot) => format!("layer{layer}-attention-request{slot}-{name}.bin"),
    }
}

/// Checked slice of an allocation; every device read goes through it.
fn slice_buffer(
    buffer: Ds41rtDeviceBuffer,
    offset: usize,
    bytes: usize,
) -> Result<Ds41rtDeviceBuffer> {
    ensure!(
        !buffer.ptr.is_null(),
        "attention trace read references a null buffer"
    );
    ensure!(
        offset <= buffer.bytes && bytes <= buffer.bytes - offset,
        "attention trace read [{offset}, {bytes}+{offset}) exceeds {} byte allocation",
        buffer.bytes
    );
    Ok(Ds41rtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
        bytes,
        ..buffer
    })
}

/// Plan every fixed-extent operand read for one invocation. Pure: takes only
/// launch views and buffer extents, touches no device, and rejects any extent
/// past its allocation or past the capture byte budget.
pub(crate) fn plan_operand_reads(
    layer: usize,
    rows: usize,
    launches: &[RequestLaunch],
    buffers: &WaveOperandBuffers,
    batched: bool,
) -> Result<Vec<PlannedRead>> {
    ensure!(!launches.is_empty(), "attention trace has no launches");
    let mut reads: Vec<PlannedRead> = Vec::new();
    let mut push = |source: ReadSource,
                    request: Option<usize>,
                    name: &'static str,
                    dtype: &'static str,
                    shape: Vec<usize>,
                    bytes: usize,
                    synthetic: Option<SyntheticContent>|
     -> Result<()> {
        if synthetic.is_none() {
            let (allocation, label) = allocation_of(source, request, launches, buffers)?;
            ensure!(
                bytes <= allocation.bytes,
                "attention trace read {name} exceeds {label} allocation ({bytes} bytes)"
            );
        }
        reads.push(PlannedRead {
            source,
            request,
            name,
            file: trace_file(layer, name, request),
            dtype,
            shape,
            offset: 0,
            bytes,
            synthetic,
        });
        Ok(())
    };
    push(
        ReadSource::Query,
        None,
        "query",
        "bfloat16",
        vec![rows, 64, 512],
        rows.checked_mul(65536).context("query extent overflow")?,
        None,
    )?;
    push(
        ReadSource::Sink,
        None,
        "sink",
        "float32",
        vec![64],
        256,
        None,
    )?;
    push(
        ReadSource::Metadata,
        None,
        "metadata",
        "uint64",
        vec![rows, 10],
        rows.checked_mul(80).context("metadata extent overflow")?,
        None,
    )?;
    if buffers.selected.is_some() {
        push(
            ReadSource::Selected,
            None,
            "selected",
            "int32",
            vec![rows, 512],
            rows.checked_mul(2048).context("selected extent overflow")?,
            None,
        )?;
    }
    if batched {
        // The batch launch reads the bounds upload for every flattened row.
        push(
            ReadSource::ReplayBegins,
            None,
            "replay-begins",
            "uint64",
            vec![rows],
            rows.checked_mul(8).context("bounds extent overflow")?,
            None,
        )?;
        push(
            ReadSource::Descriptors,
            None,
            "descriptors-device",
            "bytes",
            vec![rows, 120],
            rows.checked_mul(120)
                .context("descriptor extent overflow")?,
            None,
        )?;
    }
    for (slot, launch) in launches.iter().enumerate() {
        let request = Some(slot);
        let window = &launch.window;
        push(
            ReadSource::RingValues,
            request,
            "ring-values",
            "fp8e4m3",
            vec![RING_ROWS, 512],
            RING_ROWS * RING_VALUE_BYTES,
            None,
        )?;
        push(
            ReadSource::RingScales,
            request,
            "ring-scales",
            "uint8-e8m0",
            vec![RING_ROWS, RING_SCALE_BYTES],
            RING_ROWS * RING_SCALE_BYTES,
            None,
        )?;
        push(
            ReadSource::WindowEnd,
            request,
            "window-end",
            "uint64",
            vec![1],
            8,
            None,
        )?;
        let capacity = window.proposal_capacity;
        push(
            ReadSource::WindowProposalValues,
            request,
            "window-proposal-values",
            "fp8e4m3",
            vec![capacity, 512],
            capacity
                .checked_mul(512)
                .context("proposal extent overflow")?,
            None,
        )?;
        push(
            ReadSource::WindowProposalScales,
            request,
            "window-proposal-scales",
            "uint8-e8m0",
            vec![capacity, RING_SCALE_BYTES],
            capacity
                .checked_mul(RING_SCALE_BYTES)
                .context("proposal extent overflow")?,
            None,
        )?;
        if !batched {
            let request_rows = launch.rows;
            let bounds = request_rows
                .checked_mul(8)
                .context("bounds extent overflow")?;
            if window.replay_begins.is_some() {
                push(
                    ReadSource::ReplayBegins,
                    request,
                    "replay-begins",
                    "uint64",
                    vec![request_rows],
                    bounds,
                    None,
                )?;
            } else {
                // No bounds pointer reaches the kernel: it reads an implicit
                // zero. Write explicit zeros instead of reading the
                // uninitialized buffer.
                push(
                    ReadSource::ReplayBegins,
                    request,
                    "replay-begins",
                    "uint64",
                    vec![request_rows],
                    bounds,
                    Some(SyntheticContent::ExplicitZero),
                )?;
            }
        }
        if let Some(source) = &launch.source {
            let stride = source.page_stride;
            push(
                ReadSource::SourcePages,
                request,
                "source-pages",
                "uint32",
                vec![stride],
                stride.checked_mul(4).context("pages extent overflow")?,
                None,
            )?;
            push(
                ReadSource::SourceEnd,
                request,
                "source-end",
                "uint64",
                vec![1],
                8,
                None,
            )?;
            let proposals = source.proposal_capacity;
            push(
                ReadSource::SourceProposalValues,
                request,
                "source-proposal-values",
                "fp4e2m1",
                vec![proposals, 512],
                proposals
                    .checked_mul(SOURCE_ROW_VALUES)
                    .context("source proposal extent overflow")?,
                None,
            )?;
            push(
                ReadSource::SourceProposalScales,
                request,
                "source-proposal-scales",
                "fp8e4m3",
                vec![proposals, SOURCE_ROW_SCALES],
                proposals
                    .checked_mul(SOURCE_ROW_SCALES)
                    .context("source proposal extent overflow")?,
                None,
            )?;
        }
    }
    ensure_bounded(total_bytes(&reads)?, 0)?;
    Ok(reads)
}

/// Referenced committed source rows for one request, derived exactly as the
/// native `locate` masks them.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ReferencedSourceRows {
    /// Unique committed physical rows the kernel actually reads, ascending.
    pub physical: Vec<u64>,
    /// Committed logical id -> physical row for every actually-read key.
    pub logical_to_physical: BTreeMap<u64, u64>,
    /// Selected slots dropped by each mask, so nothing silently disappears
    /// from the fixture: masked padding, ids at/past the causal source length,
    /// private-overlay ids (read via the source proposal buffers), ids at/past
    /// the page table, and physical rows at/past the pool capacity.
    pub masked: usize,
    pub past_causal_length: usize,
    pub private: usize,
    pub past_page_stride: usize,
    pub past_capacity: usize,
}

/// Kernel-exact referenced-row resolution. `selected` is the whole wave's
/// `[rows, 512]` selection and `metadata` the whole wave's `[rows, 10]` device
/// metadata; `row_offset` is the request's first flattened row. For every
/// selected slot: `id >= 0`, `id < m[5]`, `id < m[6]`, `id < page_stride*256`
/// with `physical = pages[id/256]*256 + id%256 < source_capacity` is a paged
/// source row actually read by the kernel; every other slot is counted by the
/// mask that drops it.
pub(crate) fn resolve_referenced_source_rows(
    request_rows: usize,
    row_offset: usize,
    selected: &[i32],
    metadata: &[u64],
    pages: &[u32],
    page_stride: usize,
    source_capacity: u64,
) -> Result<ReferencedSourceRows> {
    let total_rows = row_offset
        .checked_add(request_rows)
        .context("row extent overflow")?;
    ensure!(
        selected.len()
            >= total_rows
                .checked_mul(512)
                .context("selection extent overflow")?,
        "referenced-row resolution selection covers {} rows but needs {}",
        selected.len() / 512,
        total_rows
    );
    ensure!(
        metadata.len()
            >= total_rows
                .checked_mul(10)
                .context("metadata extent overflow")?,
        "referenced-row resolution metadata is shorter than its rows"
    );
    ensure!(
        pages.len() == page_stride,
        "captured page table has {} entries, expected the {} stride",
        pages.len(),
        page_stride
    );
    let stride_rows = page_stride
        .checked_mul(256)
        .and_then(|rows| u64::try_from(rows).ok())
        .context("page stride extent overflow")?;
    let mut out = ReferencedSourceRows::default();
    let mut physical: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for row in row_offset..total_rows {
        let m = &metadata[row * 10..row * 10 + 10];
        for id in &selected[row * 512..row * 512 + 512] {
            if *id < 0 {
                out.masked += 1;
                continue;
            }
            let id = u64::try_from(*id).expect("non-negative i32 fits u64");
            if id >= m[5] {
                out.past_causal_length += 1;
                continue;
            }
            if id >= m[6] {
                out.private += 1;
                continue;
            }
            if id >= stride_rows {
                out.past_page_stride += 1;
                continue;
            }
            let physical_row = u64::from(pages[(id / 256) as usize]) * 256 + id % 256;
            if physical_row >= source_capacity {
                out.past_capacity += 1;
                continue;
            }
            out.logical_to_physical.insert(id, physical_row);
            physical.insert(physical_row);
        }
    }
    out.physical = physical.into_iter().collect();
    Ok(out)
}

fn total_bytes(reads: &[PlannedRead]) -> Result<usize> {
    reads.iter().try_fold(0usize, |total, read| {
        total
            .checked_add(read.bytes)
            .context("attention trace extent overflow")
    })
}

fn ensure_bounded(total: usize, extra: usize) -> Result<usize> {
    let total = total
        .checked_add(extra)
        .context("attention trace extent overflow")?;
    ensure!(
        total <= TRACE_BYTE_BUDGET,
        "attention input trace needs {total} bytes, over the {TRACE_BYTE_BUDGET} byte budget; \
         refusing to dump or truncate"
    );
    Ok(total)
}

fn u64_at(bytes: &[u8]) -> Result<u64> {
    Ok(u64::from_ne_bytes(
        bytes.try_into().context("captured scalar is not 8 bytes")?,
    ))
}

fn u32_vec(bytes: &[u8]) -> Result<Vec<u32>> {
    ensure!(bytes.len() % 4 == 0, "captured page table is not u32 sized");
    bytes
        .chunks_exact(4)
        .map(|chunk| Ok(u32::from_ne_bytes(chunk.try_into().unwrap())))
        .collect()
}

fn i32_vec(bytes: &[u8]) -> Result<Vec<i32>> {
    ensure!(bytes.len() % 4 == 0, "captured selection is not i32 sized");
    bytes
        .chunks_exact(4)
        .map(|chunk| Ok(i32::from_ne_bytes(chunk.try_into().unwrap())))
        .collect()
}

fn u64_vec(bytes: &[u8]) -> Result<Vec<u64>> {
    ensure!(bytes.len() % 8 == 0, "captured bounds are not u64 sized");
    bytes
        .chunks_exact(8)
        .map(|chunk| Ok(u64::from_ne_bytes(chunk.try_into().unwrap())))
        .collect()
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

struct CapturedRead {
    read: PlannedRead,
    bytes: Vec<u8>,
}

impl CapturedRead {
    fn find<'a>(
        captured: &'a [CapturedRead],
        source: ReadSource,
        request: Option<usize>,
    ) -> Result<&'a CapturedRead> {
        captured
            .iter()
            .find(|entry| entry.read.source == source && entry.read.request == request)
            .with_context(|| format!("attention input trace {:?} capture absent", source))
    }
}

/// Manifest record for one dumped buffer. Every field derives from the
/// validated live extent, so the JSON always matches the raw file on disk.
fn record(read: &PlannedRead) -> serde_json::Value {
    let rows = read.shape.first().copied().unwrap_or(1);
    let row_bytes = if rows > 0 {
        read.bytes / rows
    } else {
        read.bytes
    };
    serde_json::json!({
        "name": read.name,
        "file": read.file,
        "dtype": read.dtype,
        "shape": read.shape,
        "row_bytes": row_bytes,
        "offset": read.offset,
        "bytes": read.bytes,
        "synthetic": read.synthetic.map(|content| match content {
            SyntheticContent::ExplicitZero => "explicit_zero",
        }),
    })
}

/// Referenced committed source rows of one request plus their captured bytes.
struct ReferencedCapture {
    slot: usize,
    resolution: ReferencedSourceRows,
    values: Vec<u8>,
    scales: Vec<u8>,
}

/// Perform the capture and write every file plus the manifest. `copy` performs
/// device reads (production binds the library's synchronous `copy_d2h`; tests
/// bind a fake). `batch_host_bytes` is the host staging slice that was just
/// uploaded for a descriptor-batch launch (`None` for per-request launches).
/// Returns the total captured bytes.
pub(crate) fn write_attention_inputs(
    copy: &mut dyn FnMut(Ds41rtDeviceBuffer, &mut [u8]) -> Result<()>,
    directory: &Path,
    layer: usize,
    rows: usize,
    provenance: &[RequestProvenance],
    launches: &[RequestLaunch],
    metadata: &[u64],
    buffers: &WaveOperandBuffers,
    batch_host_bytes: Option<&[u8]>,
) -> Result<usize> {
    ensure!(
        !launches.is_empty() && provenance.len() == launches.len(),
        "attention input trace request identity differs"
    );
    ensure!(
        metadata.len() == rows.checked_mul(10).context("metadata extent overflow")?,
        "attention input trace metadata rows differ"
    );
    let mut live_rows = 0usize;
    for (launch, identity) in launches.iter().zip(provenance) {
        ensure!(
            launch.rows == identity.positions.len(),
            "attention trace position count differs"
        );
        live_rows = live_rows
            .checked_add(launch.rows)
            .context("row extent overflow")?;
    }
    ensure!(
        live_rows == rows,
        "attention trace flattened row count differs"
    );
    let batched = batch_host_bytes.is_some();
    let planned = plan_operand_reads(layer, rows, launches, buffers, batched)?;
    let mut captured_bytes = ensure_bounded(
        total_bytes(&planned)?,
        batch_host_bytes.map_or(0, |b| b.len()),
    )?;
    let mut captured: Vec<CapturedRead> = Vec::with_capacity(planned.len());
    for read in planned {
        let bytes = match read.synthetic {
            Some(SyntheticContent::ExplicitZero) => vec![0u8; read.bytes],
            None => {
                let mut out = vec![0u8; read.bytes];
                let (allocation, _) = allocation_of(read.source, read.request, launches, buffers)?;
                copy(slice_buffer(allocation, read.offset, read.bytes)?, &mut out)?;
                out
            }
        };
        captured.push(CapturedRead { read, bytes });
    }
    // Referenced committed source rows per request, resolved from the bytes
    // just captured: the exact selection, metadata and page table the kernel
    // reads. Never the whole shared compressed pool.
    let mut referenced: Vec<ReferencedCapture> = Vec::new();
    let device_metadata =
        u64_vec(&CapturedRead::find(&captured, ReadSource::Metadata, None)?.bytes)?;
    let selected = match buffers.selected {
        Some(_) => Some(i32_vec(
            &CapturedRead::find(&captured, ReadSource::Selected, None)?.bytes,
        )?),
        None => None,
    };
    let mut row_offset = 0usize;
    for (slot, launch) in launches.iter().enumerate() {
        if let Some(source) = &launch.source {
            let selection = selected
                .as_ref()
                .context("source present without selection")?;
            let pages = u32_vec(
                &CapturedRead::find(&captured, ReadSource::SourcePages, Some(slot))?.bytes,
            )?;
            // Mask exactly like the kernel using the source view's capacity;
            // every physical read extent is still checked per row.
            let resolution = resolve_referenced_source_rows(
                launch.rows,
                row_offset,
                selection,
                &device_metadata,
                &pages,
                source.page_stride,
                source.capacity as u64,
            )?;
            let count = resolution.physical.len();
            let values_bytes = count
                .checked_mul(SOURCE_ROW_VALUES)
                .context("source values extent overflow")?;
            let scales_bytes = count
                .checked_mul(SOURCE_ROW_SCALES)
                .context("source scales extent overflow")?;
            captured_bytes = ensure_bounded(
                captured_bytes,
                values_bytes
                    .checked_add(scales_bytes)
                    .context("source extent overflow")?,
            )?;
            let mut values = vec![0u8; values_bytes];
            let mut scales = vec![0u8; scales_bytes];
            for (index, &physical) in resolution.physical.iter().enumerate() {
                let offset = usize::try_from(physical).context("physical row overflow")?;
                copy(
                    slice_buffer(
                        source.values,
                        offset
                            .checked_mul(SOURCE_ROW_VALUES)
                            .context("source values offset overflow")?,
                        SOURCE_ROW_VALUES,
                    )?,
                    &mut values[index * SOURCE_ROW_VALUES..(index + 1) * SOURCE_ROW_VALUES],
                )?;
                copy(
                    slice_buffer(
                        source.scales,
                        offset
                            .checked_mul(SOURCE_ROW_SCALES)
                            .context("source scales offset overflow")?,
                        SOURCE_ROW_SCALES,
                    )?,
                    &mut scales[index * SOURCE_ROW_SCALES..(index + 1) * SOURCE_ROW_SCALES],
                )?;
            }
            referenced.push(ReferencedCapture {
                slot,
                resolution,
                values,
                scales,
            });
        }
        row_offset += launch.rows;
    }
    // Batch descriptors: record the actual GPU bytes and the staged host bytes,
    // compared as bytes only (they embed raw device pointers).
    let descriptors = match batch_host_bytes {
        Some(host) => {
            let device = &CapturedRead::find(&captured, ReadSource::Descriptors, None)?.bytes;
            ensure!(
                device.len() == host.len(),
                "batch descriptor extent differs between device and host"
            );
            Some((device.clone(), host.to_vec(), device.as_slice() == host))
        }
        None => None,
    };
    for entry in &captured {
        write_new_file(directory, &entry.read.file, &entry.bytes)?;
    }
    for reference in &referenced {
        write_new_file(
            directory,
            &trace_file(layer, "source-referenced-values", Some(reference.slot)),
            &reference.values,
        )?;
        write_new_file(
            directory,
            &trace_file(layer, "source-referenced-scales", Some(reference.slot)),
            &reference.scales,
        )?;
    }
    if let Some((_, host, _)) = &descriptors {
        write_new_file(
            directory,
            &trace_file(layer, "descriptors-host", None),
            host,
        )?;
    }
    let mut manifest = input_manifest(
        layer,
        rows,
        provenance,
        launches,
        &captured,
        &referenced,
        descriptors
            .as_ref()
            .map(|(device, host, matches)| (device.as_slice(), host.as_slice(), *matches)),
        batched,
        captured_bytes,
    )?;
    manifest["metadata_device_matches_host"] = serde_json::json!(device_metadata == metadata);
    write_new_file(
        directory,
        &format!("layer{layer}-attention-inputs.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(captured_bytes)
}

fn manifest_request(
    layer: usize,
    slot: usize,
    row_offset: usize,
    launch: &RequestLaunch,
    provenance: &RequestProvenance,
    captured: &[CapturedRead],
    referenced: &[ReferencedCapture],
    batched: bool,
) -> Result<serde_json::Value> {
    let request = Some(slot);
    let window_end = u64_at(&CapturedRead::find(captured, ReadSource::WindowEnd, request)?.bytes)?;
    // Non-batched launches without an uploaded bounds slice pass no pointer,
    // so the kernel reads an implicit zero: that request's file holds explicit
    // zeros and is marked synthetic, never read from the uninitialized buffer.
    let replay_entry = if batched {
        CapturedRead::find(captured, ReadSource::ReplayBegins, None)?
    } else {
        CapturedRead::find(captured, ReadSource::ReplayBegins, request)?
    };
    let replay_values = u64_vec(&replay_entry.bytes)?;
    let replay_json = serde_json::json!({
        "file": replay_entry.read.file,
        "dtype": replay_entry.read.dtype,
        "shape": replay_entry.read.shape,
        "bytes": replay_entry.read.bytes,
        "uploaded": replay_entry.read.synthetic.is_none(),
        "synthetic": replay_entry.read.synthetic.map(|content| match content {
            SyntheticContent::ExplicitZero => "explicit_zero",
        }),
        "values": replay_values,
    });
    let mut source_json = serde_json::Value::Null;
    if launch.source.is_some() {
        let source_end =
            u64_at(&CapturedRead::find(captured, ReadSource::SourceEnd, request)?.bytes)?;
        let reference = referenced
            .iter()
            .find(|entry| entry.slot == slot)
            .context("source request without referenced rows")?;
        let count = reference.resolution.physical.len();
        let source = launch.source.as_ref().unwrap();
        source_json = serde_json::json!({
            "format": 2,
            "capacity_rows": source.capacity,
            "values_bytes": source.values.bytes,
            "row_values_bytes": SOURCE_ROW_VALUES,
            "row_scales_bytes": SOURCE_ROW_SCALES,
            "page_stride": source.page_stride,
            "pages": record(&CapturedRead::find(captured, ReadSource::SourcePages, request)?.read),
            "source_end": {
                "file": CapturedRead::find(captured, ReadSource::SourceEnd, request)?.read.file,
                "dtype": "uint64", "value": source_end,
            },
            "proposal_capacity": source.proposal_capacity,
            "proposal_values": record(&CapturedRead::find(captured, ReadSource::SourceProposalValues, request)?.read),
            "proposal_scales": record(&CapturedRead::find(captured, ReadSource::SourceProposalScales, request)?.read),
            "referenced": {
                "count": count,
                "physical_rows": reference.resolution.physical,
                "logical_to_physical": reference.resolution.logical_to_physical.iter()
                    .map(|(logical, physical)| serde_json::json!({
                        "logical": logical, "physical": physical }))
                    .collect::<Vec<_>>(),
                "values": {
                    "file": trace_file(layer, "source-referenced-values", request),
                    "dtype": "fp4e2m1", "shape": [count, 512],
                    "row_bytes": SOURCE_ROW_VALUES, "offset": 0, "bytes": count * SOURCE_ROW_VALUES,
                },
                "scales": {
                    "file": trace_file(layer, "source-referenced-scales", request),
                    "dtype": "fp8e4m3", "shape": [count, 32],
                    "row_bytes": SOURCE_ROW_SCALES, "offset": 0, "bytes": count * SOURCE_ROW_SCALES,
                },
                "skipped": {
                    "masked": reference.resolution.masked,
                    "past_causal_length": reference.resolution.past_causal_length,
                    "private": reference.resolution.private,
                    "past_page_stride": reference.resolution.past_page_stride,
                    "past_capacity": reference.resolution.past_capacity,
                },
            },
        });
    }
    Ok(serde_json::json!({
        "slot": slot,
        "request_id": provenance.request_id,
        "rows": launch.rows,
        "flattened_row_range": [row_offset, row_offset + launch.rows],
        "positions": provenance.positions,
        "width": launch.width,
        "split_parts": if !batched && launch.rows <= 16 {
            serde_json::json!(if launch.source.is_some() { 10 } else { 2 })
        } else { serde_json::Value::Null },
        "replay_begins": replay_json,
        "window": {
            "ring_rows": RING_ROWS,
            "ring_values": record(&CapturedRead::find(captured, ReadSource::RingValues, request)?.read),
            "ring_scales": record(&CapturedRead::find(captured, ReadSource::RingScales, request)?.read),
            "window_end": {
                "file": CapturedRead::find(captured, ReadSource::WindowEnd, request)?.read.file,
                "dtype": "uint64", "value": window_end,
            },
            "proposal_capacity": launch.window.proposal_capacity,
            "proposal_values": record(&CapturedRead::find(captured, ReadSource::WindowProposalValues, request)?.read),
            "proposal_scales": record(&CapturedRead::find(captured, ReadSource::WindowProposalScales, request)?.read),
        },
        "source": source_json,
    }))
}

fn input_manifest(
    layer: usize,
    rows: usize,
    provenance: &[RequestProvenance],
    launches: &[RequestLaunch],
    captured: &[CapturedRead],
    referenced: &[ReferencedCapture],
    descriptors: Option<(&[u8], &[u8], bool)>,
    batched: bool,
    captured_bytes: usize,
) -> Result<serde_json::Value> {
    let mut requests = Vec::with_capacity(launches.len());
    let mut row_offset = 0usize;
    for (slot, launch) in launches.iter().enumerate() {
        requests.push(manifest_request(
            layer,
            slot,
            row_offset,
            launch,
            &provenance[slot],
            captured,
            referenced,
            batched,
        )?);
        row_offset += launch.rows;
    }
    let wave = |source: ReadSource| -> Result<serde_json::Value> {
        Ok(CapturedRead::find(captured, source, None)
            .map(|entry| record(&entry.read))
            .unwrap_or(serde_json::Value::Null))
    };
    let parts = if batched {
        serde_json::json!(if launches_have_source(launches) {
            10
        } else {
            2
        })
    } else {
        serde_json::Value::Null
    };
    let descriptors_json = match descriptors {
        Some((device, host, matches)) => serde_json::json!({
            "device_file": trace_file(layer, "descriptors-device", None),
            "host_file": trace_file(layer, "descriptors-host", None),
            "bytes": device.len(),
            "device_bytes": device.len(),
            "host_bytes": host.len(),
            "device_matches_host": matches,
            "comparison": "bytes only; descriptors embed raw device pointers",
        }),
        None => serde_json::Value::Null,
    };
    Ok(serde_json::json!({
        "schema": 1,
        "kind": "sparse-attention-inputs",
        "layer": layer,
        "rows": rows,
        "launch_kind": if batched { "descriptor_batch" } else { "single_per_request" },
        "parts": parts,
        "wave": {
            "query": wave(ReadSource::Query)?,
            "sink": wave(ReadSource::Sink)?,
            "metadata": wave(ReadSource::Metadata)?,
            "selected": wave(ReadSource::Selected)?,
            "replay_begins": wave(ReadSource::ReplayBegins)?,
        },
        "descriptors": descriptors_json,
        "requests": requests,
        "budget_bytes": TRACE_BYTE_BUDGET,
        "captured_bytes": captured_bytes,
        "notes": [
            "all extents are pointer-independent: files record live rows, offsets and byte counts only",
            "only referenced committed source rows are captured; the shared compressed pool is never dumped whole",
            "masked/private selected slots stay in the full selected buffer and are represented by metadata and proposal buffers",
            "raw unused ring/proposal padding is unscored; use live metadata and referenced rows only",
            "descriptor bytes are opaque pointer-bearing payloads; compare as bytes, never as numbers",
        ],
    }))
}

/// Batch launches split by the same source-present rule as the dispatch.
fn launches_have_source(launches: &[RequestLaunch]) -> bool {
    launches.iter().any(|launch| launch.source.is_some())
}

/// Production entry: bind the library's synchronous device-to-host copy. The
/// wave has already synchronized its stream before calling.
pub(crate) fn capture_attention_inputs(
    library: &NativeLibrary,
    directory: &Path,
    layer: usize,
    rows: usize,
    provenance: &[RequestProvenance],
    launches: &[RequestLaunch],
    metadata: &[u64],
    buffers: &WaveOperandBuffers,
    batch: Option<&V41SparseBatch>,
) -> Result<usize> {
    write_attention_inputs(
        &mut |buffer, destination| library.copy_d2h(destination, buffer),
        directory,
        layer,
        rows,
        provenance,
        launches,
        metadata,
        buffers,
        batch.map(|batch| batch.bytes()),
    )
}
