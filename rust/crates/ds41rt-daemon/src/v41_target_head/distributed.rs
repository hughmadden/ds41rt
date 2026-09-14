//! Per-lane vocabulary projection on two GPUs, with compact GPU winner merging.
use super::*;
use crate::v41_memory::device::{Device, DeviceOwner, Stream, Allocation};
use crate::v41_tensors::VocabularyShard;

struct Rank<'w, 'a> {
    stream: LoadStream<'a>,
    projection: V41VocabularyProjection<'a>,
    _workspace: DeviceAllocation<'a>,
    input: DeviceAllocation<'a>,
    logits: DeviceAllocation<'a>,
    candidates: DeviceAllocation<'a>,
    weights: &'w VocabularyShard<'a>,
    capacity: usize,
    graphs: [[Option<*mut c_void>; 80]; 2],
}
fn slice(buffer: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Result<Ds41rtDeviceBuffer> {
    ensure!(offset <= buffer.bytes && bytes <= buffer.bytes - offset, "vocabulary slice exceeds allocation");
    Ok(Ds41rtDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer })
}
impl<'w, 'a> Rank<'w, 'a> {
    fn new(library: &'a NativeLibrary, weights: &'w VocabularyShard<'a>, capacity: usize) -> Result<Self> {
        let workspace = DeviceAllocation::new(library, V41VocabularyProjection::WORKSPACE_BYTES)?;
        let projection = unsafe { library.v41_vocabulary_shard(workspace.buffer, weights.tokens().len())? };
        let input = DeviceAllocation::new(library, capacity * 10240)?;
        ensure!(input.buffer.device_id == weights.weight().device_id, "vocabulary rank weight device differs");
        Ok(Self {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            projection, _workspace: workspace, input,
            logits: DeviceAllocation::new(library, capacity * weights.tokens().len() * 4)?,
            candidates: DeviceAllocation::new(library, capacity * 8)?,
            weights, capacity, graphs: [[None; 80]; 2],
        })
    }
    fn candidates(&self, rows: usize) -> Result<(Ds41rtDeviceBuffer, Ds41rtDeviceBuffer)> {
        Ok((slice(self.candidates.buffer, 0, rows * 4)?,
            slice(self.candidates.buffer, self.capacity * 4, rows * 4)?))
    }
    unsafe fn enqueue(&self, rows: usize, greedy: bool) -> Result<()> {
        let lib = self.stream.library;
        unsafe {
            self.projection.launch(self.input.buffer, self.weights.weight(), self.logits.buffer, rows, self.stream.raw)?;
            if greedy {
                let (ids, scores) = self.candidates(rows)?;
                lib.cuda_logits_argmax_checked_f32_async(self.logits.buffer, ids, scores,
                    rows, self.weights.tokens().len(), self.stream.raw)?;
            }
            Ok(())
        }
    }
    unsafe fn begin(&mut self, input: Ds41rtDeviceBuffer, rows: usize, greedy: bool) -> Result<()> {
        let mode = usize::from(greedy);
        unsafe {
            let lib = self.stream.library;
            if input.device_id == self.input.buffer.device_id {
                lib.copy_d2d_async(self.input.buffer, input, rows * 10240, self.stream.raw)?;
            } else {
                lib.copy_peer_async(self.input.buffer, input, rows * 10240, self.stream.raw)?;
            }
            if let Some(graph) = self.graphs[mode][rows - 1] { lib.cuda_graph_launch(graph, self.stream.raw) }
            else { self.enqueue(rows, greedy) }
        }
    }
    unsafe fn capture_ready(&mut self, rows: usize, greedy: bool) -> Result<()> {
        let mode = usize::from(greedy);
        if self.graphs[mode][rows - 1].is_none() {
            let lib = self.stream.library;
            unsafe { lib.cuda_graph_begin_capture(self.stream.raw)?; }
            let queued = unsafe { self.enqueue(rows, greedy) };
            let captured = unsafe { lib.cuda_graph_end_capture(self.stream.raw) };
            match (queued, captured) {
                (Ok(()), Ok(graph)) => self.graphs[mode][rows - 1] = Some(graph),
                (Err(error), Ok(graph)) => { unsafe { lib.cuda_graph_exec_destroy(graph)?; } return Err(error); }
                (Err(error), Err(_)) | (Ok(()), Err(error)) => return Err(error),
            }
            // Warmup already produced the result. The captured graph runs on
            // the next use of this shape, without repeating this projection.
        }
        Ok(())
    }
    async unsafe fn execute(&mut self, input: Ds41rtDeviceBuffer, rows: usize, greedy: bool) -> Result<()> {
        let queued = unsafe { self.begin(input, rows, greedy) };
        let drained = self.stream.wait().await;
        queued.and(drained)?;
        unsafe { self.capture_ready(rows, greedy) }
    }

}
impl Drop for Rank<'_, '_> {
    fn drop(&mut self) {
        let lib = self.stream.library;
        if let Err(error) = unsafe { lib.cuda_stream_synchronize(self.stream.raw) } {
            tracing::error!(%error, "draining vocabulary rank");
        }
        for graph in self.graphs.iter_mut().flatten().filter_map(Option::take) {
            if let Err(error) = unsafe { lib.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error, "destroying vocabulary rank graph");
            }
        }
    }
}

pub(crate) struct DistributedVocabularyWave<'w, 'a> {
    ranks: [DeviceOwner<'a, Rank<'w, 'a>>; 2],
    merge_stream: Stream<'a>,
    remote: Allocation<'a>,
    merged: Allocation<'a>,
    capacity: usize,
    split: usize,
    ready: Option<usize>,
    greedy_ready: bool,
    pending_logits: Option<(usize, [bool; 2])>,
    copy_pending: bool,
}
impl<'w, 'a> DistributedVocabularyWave<'w, 'a> {
    pub fn device_bytes(capacity: usize, split: usize) -> Result<[usize; 2]> {
        ensure!((1..=80).contains(&capacity) && (1..129280).contains(&split), "invalid distributed vocabulary geometry");
        Ok([V41VocabularyProjection::WORKSPACE_BYTES + capacity * (10240 + split * 4 + 8),
            V41VocabularyProjection::WORKSPACE_BYTES + capacity * (10240 + (129280 - split) * 4 + 24)])
    }
    pub fn new(devices: [Device<'a>; 2], weights: [&'w VocabularyShard<'a>; 2],
        capacity: usize, budgets: [usize; 2]) -> Result<Self> {
        let split = weights[0].tokens().end;
        ensure!(weights[0].tokens() == (0..split) && weights[1].tokens() == (split..129280),
            "vocabulary shards do not partition the full vocabulary");
        let bytes = Self::device_bytes(capacity, split)?;
        ensure!(bytes.iter().zip(budgets).all(|(bytes, budget)| *bytes <= budget), "distributed vocabulary exceeds budget");
        ensure!(devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
            "distributed vocabulary devices differ in identity or library");
        for rank in 0..2 {
            devices[rank].run(|| devices[rank].library.cuda_enable_peer(devices[1-rank].id))?;
        }
        Ok(Self {
            ranks: [devices[0].own(|| Rank::new(devices[0].library, weights[0], capacity))?,
                devices[1].own(|| Rank::new(devices[1].library, weights[1], capacity))?],
            merge_stream: Stream::new(devices[1])?,
            remote: Allocation::new(devices[1], capacity * 8)?,
            merged: Allocation::new(devices[1], capacity * 8)?,
            capacity, split, ready: None, greedy_ready: false, pending_logits: None, copy_pending: false,
        })
    }
    /// # Safety
    /// Normalized BF16 input [rows,5120] is complete on rank 1's GPU. Its owner
    /// remains alive and immutable until return/cancellation. All rank, copy,
    /// and merge work drains on errors/cancellation before buffers can be reused.
    pub async unsafe fn execute(&mut self, normalized: Ds41rtDeviceBuffer, rows: usize) -> Result<()> {
        ensure!(self.pending_logits.is_none() && !self.copy_pending, "distributed vocabulary work still pending");
        self.ready = None;
        self.greedy_ready = false;
        ensure!((1..=self.capacity).contains(&rows) && normalized.bytes >= rows * 10240
            && normalized.device_id == self.ranks[1].device.id, "invalid distributed vocabulary input");
        let [first, second] = &mut self.ranks;
        let first_device = first.device;
        let second_device = second.device;
        let stream = &self.merge_stream;
        let remote = self.remote.buffer;
        let capacity = self.capacity;
        let first_work = async {
            first_device.future(unsafe { first.get_mut().execute(normalized, rows, true) }).await?;
            let (ids, scores) = first.candidates(rows)?;
            let queued = stream.device.run(|| unsafe {
                let lib = stream.device.library;
                lib.copy_peer_async(slice(remote, 0, rows * 4)?, ids, rows * 4, stream.raw)?;
                lib.copy_peer_async(slice(remote, capacity * 4, rows * 4)?, scores, rows * 4, stream.raw)
            });
            let drained = stream.wait().await;
            queued.and(drained)
        };
        let second_work = second_device.future(unsafe { second.get_mut().execute(normalized, rows, true) });
        let (a, b) = tokio::join!(first_work, second_work);
        a.and(b)?;
        let local = self.ranks[1].candidates(rows)?;
        let remote = (slice(self.remote.buffer, 0, rows * 4)?, slice(self.remote.buffer, self.capacity * 4, rows * 4)?);
        let output = (slice(self.merged.buffer, 0, rows * 4)?, slice(self.merged.buffer, self.capacity * 4, rows * 4)?);
        let queued = stream.device.run(|| unsafe {
            stream.device.library.v41_vocabulary_merge_greedy([remote, local], output, rows, self.split, stream.raw)
        });
        let drained = stream.wait().await;
        queued.and(drained)?;
        self.ready = Some(rows);
        self.greedy_ready = true;
        Ok(())
    }
    /// Project both vocabulary shards without computing or transferring winners.
    ///
    /// # Safety
    /// The same input lifetime and completion requirements as `execute` apply.
    pub async unsafe fn execute_logits(&mut self, normalized: Ds41rtDeviceBuffer, rows: usize) -> Result<()> {
        unsafe { self.begin_logits(normalized, rows)?; }
        struct Cancel<'s, 'w, 'a> { wave: &'s mut DistributedVocabularyWave<'w, 'a>, armed: bool }
        impl Drop for Cancel<'_, '_, '_> {
            fn drop(&mut self) { if self.armed { self.wave.cancel_logits(); } }
        }
        let mut pending = Cancel { wave: self, armed: true };
        while !pending.wave.poll_logits()? { tokio::task::yield_now().await; }
        pending.armed = false;
        Ok(())
    }

    pub fn greedy(&self) -> Result<(Ds41rtDeviceBuffer, Ds41rtDeviceBuffer)> {
        ensure!(self.greedy_ready, "distributed vocabulary greedy output unpublished");
        let rows = self.ready.context("distributed vocabulary output unpublished")?;
        Ok((slice(self.merged.buffer, 0, rows * 4)?, slice(self.merged.buffer, self.capacity * 4, rows * 4)?))
    }
    /// Submit both projections without retaining a scheduler borrow.
    ///
    /// # Safety
    /// Same normalized-input contract as `execute_logits`. Input storage stays
    /// live and immutable until `poll_logits` completes or `cancel_logits` drains.
    pub unsafe fn begin_logits(&mut self, normalized: Ds41rtDeviceBuffer, rows: usize) -> Result<()> {
        ensure!(self.pending_logits.is_none() && !self.copy_pending, "distributed vocabulary work still pending");
        self.ready = None;
        self.greedy_ready = false;
        ensure!((1..=self.capacity).contains(&rows) && normalized.bytes >= rows * 10240
            && normalized.device_id == self.ranks[1].device.id, "invalid distributed vocabulary input");
        self.pending_logits = Some((rows, [false; 2]));
        let queued = (|| -> Result<()> {
            for rank in &mut self.ranks {
                let device = rank.device;
                device.run(|| unsafe { rank.get_mut().begin(normalized, rows, false) })?;
            }
            Ok(())
        })();
        if queued.is_err() { self.cancel_logits(); }
        queued
    }
    /// Queries only this wave's rank streams. Warm graph capture happens once
    /// each rank completes; a slow rank does not delay capture on the other GPU.
    pub fn poll_logits(&mut self) -> Result<bool> {
        let (rows, mut complete) = self.pending_logits.context("distributed vocabulary projection not pending")?;
        let result = (|| -> Result<bool> {
            for (index, rank) in self.ranks.iter_mut().enumerate() {
                if !complete[index] {
                    let device = rank.device;
                    complete[index] = device.run(|| unsafe {
                        if !rank.stream.library.cuda_stream_query(rank.stream.raw)? { return Ok(false); }
                        rank.get_mut().capture_ready(rows, false)?;
                        Ok(true)
                    })?;
                }
            }
            if complete == [true; 2] {
                self.pending_logits = None;
                self.ready = Some(rows);
                Ok(true)
            } else {
                self.pending_logits = Some((rows, complete));
                Ok(false)
            }
        })();
        if result.is_err() { self.cancel_logits(); }
        result
    }
    /// Error/cancellation cleanup only; ordinary progress uses stream queries.
    pub fn cancel_logits(&mut self) {
        if self.pending_logits.take().is_some() {
            for rank in &self.ranks {
                if let Err(error) = rank.device.run(|| unsafe { rank.stream.library.cuda_stream_synchronize(rank.stream.raw) }) {
                    tracing::error!(%error, "draining cancelled vocabulary projection");
                }
            }
        }
        self.ready = None;
        self.greedy_ready = false;
    }
    pub fn logits(&self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let rows = self.ready.context("distributed vocabulary output unpublished")?;
        Ok([slice(self.ranks[0].logits.buffer, 0, rows * self.split * 4)?,
            slice(self.ranks[1].logits.buffer, 0, rows * (129280 - self.split) * 4)?])
    }
    /// Assemble published shards in row-major vocabulary order on rank 1.
    ///
    /// # Safety
    /// Destination storage remains live and has no conflicting access through
    /// return or cancellation. It must not alias either shard. The lane's own
    /// copy stream drains before return/cancellation; no other lane is joined.
    pub async unsafe fn copy_logits_to(&mut self, destination: Ds41rtDeviceBuffer) -> Result<()> {
        unsafe { self.begin_copy_logits(destination)?; }
        struct Cancel<'s, 'w, 'a> { wave: &'s mut DistributedVocabularyWave<'w, 'a>, armed: bool }
        impl Drop for Cancel<'_, '_, '_> {
            fn drop(&mut self) { if self.armed { self.wave.cancel_copy_logits(); } }
        }
        let mut pending = Cancel { wave: self, armed: true };
        while !pending.wave.poll_copy_logits()? { tokio::task::yield_now().await; }
        pending.armed = false;
        Ok(())
    }
    /// # Safety
    /// Same destination contract as `copy_logits_to`, lasting until successful
    /// polling or cancellation. Source shards cannot be reused while pending.
    pub unsafe fn begin_copy_logits(&mut self, destination: Ds41rtDeviceBuffer) -> Result<()> {
        ensure!(!self.copy_pending && self.pending_logits.is_none(), "distributed vocabulary work still pending");
        let rows = self.ready.context("distributed vocabulary output unpublished")?;
        let bytes = rows * 129280 * 4;
        ensure!(destination.device_id == self.merge_stream.device.id && destination.bytes >= bytes,
            "invalid distributed logits destination");
        let sources = self.logits()?;
        self.copy_pending = true;
        let queued = self.merge_stream.device.run(|| unsafe {
            for rank in 0..2 {
                let offset = if rank == 0 { 0 } else { self.split * 4 };
                let width = if rank == 0 { self.split * 4 } else { (129280 - self.split) * 4 };
                self.merge_stream.device.library.copy_device_rows_async(
                    slice(destination, offset, bytes - offset)?, sources[rank],
                    width, rows, 129280 * 4, width, self.merge_stream.raw)?;
            }
            Ok(())
        });
        if queued.is_err() { self.cancel_copy_logits(); }
        queued
    }
    pub fn poll_copy_logits(&mut self) -> Result<bool> {
        ensure!(self.copy_pending, "distributed vocabulary copy not pending");
        let ready = self.merge_stream.device.run(|| unsafe {
            self.merge_stream.device.library.cuda_stream_query(self.merge_stream.raw)
        });
        match ready {
            Ok(true) => { self.copy_pending = false; Ok(true) }
            Ok(false) => Ok(false),
            Err(error) => { self.cancel_copy_logits(); Err(error) }
        }
    }
    pub fn cancel_copy_logits(&mut self) {
        if self.copy_pending {
            if let Err(error) = self.merge_stream.drain() {
                tracing::error!(%error, "draining cancelled vocabulary assembly");
            }
            self.copy_pending = false;
        }
    }

}

impl Drop for DistributedVocabularyWave<'_, '_> {
    fn drop(&mut self) {
        // Assembly reads rank buffers: drain before their field destructors run.
        self.cancel_copy_logits();
        self.cancel_logits();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn distributed_vocabulary_real_weights_match_full_and_cancel_safely() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        let a = devices[0].own(|| VocabularyShard::load(&lib, &catalog, 0..64640, 1 << 30, 16 << 20))?;
        let b = devices[1].own(|| VocabularyShard::load(&lib, &catalog, 64640..129280, 1 << 30, 16 << 20))?;
        let full = devices[1].own(|| VocabularyHead::load(&lib, &catalog, 2 << 30, 16 << 20))?;
        let workspace = Allocation::new(devices[1], V41VocabularyProjection::WORKSPACE_BYTES)?;
        let projection = devices[1].own(|| unsafe { lib.v41_vocabulary_head(workspace.buffer) })?;
        let full_stream = Stream::new(devices[1])?;
        let full_logits = Allocation::new(devices[1], 80 * 129280 * 4)?;
        let input = Allocation::new(devices[1], 80 * 10240)?;
        let values: Vec<u8> = (0..80 * 5120).flat_map(|i| {
            let value = ((i * 17 % 127) as f32 - 63.) / 128.;
            ((value.to_bits() >> 16) as u16).to_ne_bytes()
        }).collect();
        devices[1].run(|| lib.copy_h2d(input.buffer, &values))?;
        let other_input = Allocation::new(devices[1], 80 * 10240)?;
        let other_values: Vec<u8> = (0..80 * 5120).flat_map(|i| {
            let value = ((i * 19 % 127) as f32 - 61.) / 64.;
            ((value.to_bits() >> 16) as u16).to_ne_bytes()
        }).collect();
        devices[1].run(|| lib.copy_h2d(other_input.buffer, &other_values))?;
        let budgets = DistributedVocabularyWave::device_bytes(80, 64640)?;
        let mut first = DistributedVocabularyWave::new(devices, [&a, &b], 80, budgets)?;
        let mut second = DistributedVocabularyWave::new(devices, [&a, &b], 80, budgets)?;
        let assembled = [Allocation::new(devices[1], 80 * 129280 * 4)?,
            Allocation::new(devices[1], 80 * 129280 * 4)?];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for rows in [1, 3, 16, 40, 80, 3] {
            let mut references = Vec::new();
            for source in [input.buffer, other_input.buffer] {
                devices[1].run(|| unsafe {
                    projection.launch(source, full.weight()?, full_logits.buffer, rows, full_stream.raw)
                })?;
                full_stream.drain()?;
                let mut expected = vec![0u8; rows * 129280 * 4];
                devices[1].run(|| lib.copy_d2h(&mut expected, slice(full_logits.buffer, 0, rows * 129280 * 4)?))?;
                references.push(expected);
            }
            for greedy in [true, false, false, true] {
                runtime.block_on(async {
                    if greedy {
                        let (a, b) = tokio::join!(unsafe { first.execute(input.buffer, rows) }, unsafe { second.execute(other_input.buffer, rows) });
                        a.and(b)
                    } else {
                        let (a, b) = tokio::join!(unsafe { first.execute_logits(input.buffer, rows) }, unsafe { second.execute_logits(other_input.buffer, rows) });
                        a.and(b)
                    }
                })?;
                runtime.block_on(async {
                    let (a, b) = tokio::join!(unsafe { first.copy_logits_to(assembled[0].buffer) },
                        unsafe { second.copy_logits_to(assembled[1].buffer) });
                    a.and(b)
                })?;
                for ((wave, expected), output) in [&first, &second].into_iter().zip(&references).zip(&assembled) {
                    let mut combined = vec![0u8; rows * 129280 * 4];
                    devices[1].run(|| lib.copy_d2h(&mut combined, slice(output.buffer, 0, rows * 129280 * 4)?))?;
                    ensure!(combined == *expected, "GPU-assembled vocabulary differs from full head");
                    let logits = wave.logits()?;
                    let mut parts = [vec![0u8; rows * 64640 * 4], vec![0u8; rows * 64640 * 4]];
                    for rank in 0..2 { devices[rank].run(|| lib.copy_d2h(&mut parts[rank], logits[rank]))?; }
                    let mut max_error = 0f32;
                    for row in 0..rows {
                        for (rank, part) in parts.iter().enumerate() {
                            for token in 0..64640 {
                                let local = (row * 64640 + token) * 4;
                                let global = (row * 129280 + rank * 64640 + token) * 4;
                                let actual = f32::from_ne_bytes(part[local..local+4].try_into().unwrap());
                                let reference = f32::from_ne_bytes(expected[global..global+4].try_into().unwrap());
                                ensure!(actual.is_finite() && reference.is_finite(), "nonfinite vocabulary result");
                                max_error = max_error.max((actual - reference).abs());
                            }
                        }
                    }
                    ensure!(max_error <= 0.001, "shard/full real vocabulary error {max_error}");
                    if !greedy {
                        ensure!(wave.greedy().is_err(), "projection-only execution exposed stale winners");
                        eprintln!("PASS projection-only real vocabulary rows={rows}, all logits max_error={max_error}");
                        continue;
                    }
                    let (ids, scores) = wave.greedy()?;
                    let mut ids_bytes = vec![0u8; rows * 4];
                    let mut scores_bytes = vec![0u8; rows * 4];
                    devices[1].run(|| { lib.copy_d2h(&mut ids_bytes, ids)?; lib.copy_d2h(&mut scores_bytes, scores) })?;
                    for row in 0..rows {
                        let values: Vec<f32> = expected[row*129280*4..(row+1)*129280*4].chunks_exact(4)
                            .map(|v| f32::from_ne_bytes(v.try_into().unwrap())).collect();
                        let mut best = 0;
                        for i in 1..values.len() { if values[i] > values[best] { best = i; } }
                        assert_eq!(u32::from_ne_bytes(ids_bytes[row*4..row*4+4].try_into().unwrap()), best as u32);
                        let score = f32::from_ne_bytes(scores_bytes[row*4..row*4+4].try_into().unwrap());
                        ensure!((score - values[best]).abs() <= 0.001, "global greedy score differs");
                    }
                    eprintln!("PASS real vocabulary rows={rows}, all logits max_error={max_error}, global greedy exact");
                }
            }
        }
        // Cancel after submission at the first asynchronous wait. Reuse must
        // succeed and abandoned outputs must remain unpublished.
        use std::{future::Future, task::{Context, Poll, Waker}};
        unsafe { first.begin_logits(input.buffer, 80)?; second.begin_logits(other_input.buffer, 3)?; }
        assert!(unsafe { first.begin_logits(input.buffer, 3) }.is_err());
        assert!(runtime.block_on(unsafe { first.execute(input.buffer, 3) }).is_err());
        while !second.poll_logits()? { std::thread::yield_now(); }
        assert!(second.logits().is_ok());
        assert!(first.logits().is_err());
        assert!(first.pending_logits.is_some());
        first.cancel_logits();
        assert!(first.logits().is_err());
        unsafe { second.begin_copy_logits(assembled[1].buffer)?; }
        assert!(unsafe { second.begin_logits(other_input.buffer, 3) }.is_err());
        assert!(unsafe { second.begin_copy_logits(assembled[1].buffer) }.is_err());
        while !second.poll_copy_logits()? { std::thread::yield_now(); }
        assert!(second.poll_copy_logits().is_err());
        eprintln!("PASS vocabulary begin/poll: lane completes without polling its peer; pending projection/copy reuse rejected");
        let mut pending = Box::pin(unsafe { first.execute(input.buffer, 16) });
        assert!(matches!(pending.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
        drop(pending);
        assert!(first.greedy().is_err());
        runtime.block_on(unsafe { first.execute(input.buffer, 3) })?;
        assert!(first.greedy().is_ok());
        let mut pending = Box::pin(unsafe { first.execute_logits(input.buffer, 16) });
        assert!(matches!(pending.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
        drop(pending);
        assert!(first.greedy().is_err());
        assert!(first.logits().is_err());
        runtime.block_on(unsafe { first.execute_logits(input.buffer, 3) })?;
        assert!(first.logits().is_ok());
        assert!(first.greedy().is_err());
        assert!(runtime.block_on(unsafe { first.copy_logits_to(input.buffer) }).is_err());
        let foreign = Ds41rtDeviceBuffer { device_id: 0, ..assembled[0].buffer };
        assert!(runtime.block_on(unsafe { first.copy_logits_to(foreign) }).is_err());
        runtime.block_on(unsafe { first.execute_logits(input.buffer, 80) })?;
        let mut pending = Box::pin(unsafe { first.copy_logits_to(assembled[0].buffer) });
        assert!(matches!(pending.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
        drop(pending);
        runtime.block_on(unsafe { first.copy_logits_to(assembled[0].buffer) })?;
        assert!(first.logits().is_ok());
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!("PASS distributed vocabulary cancellation, reuse and device restoration");
        Ok(())
    }
}
