//! One lane's local routed-expert scratch; both lanes share immutable weights.
use super::{DeviceAllocation, ExpertLayer, ExpertWeights, LoadStream};
use crate::{v41_backbone_router::RouterOutput, v41_backbone_shared::SharedOutput};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs, V41LocalExpertReducer};
use std::{ffi::c_void, rc::Rc};

struct State<'a> {
    kernel: V41ExpertKernel<'a>,
    scratch: DeviceAllocation<'a>,
    slots: [*mut c_void; 44],
}
pub(crate) struct LocalExpertWave<'a> {
    // Drain before scratch, weights or output can be released on any exit.
    stream: LoadStream<'a>,
    states: Vec<State<'a>>,
    weights: Rc<Vec<ExpertWeights<'a>>>,
    output: DeviceAllocation<'a>,
    reducer: V41LocalExpertReducer<'a>,
    capacity: u32,
}
impl<'a> LocalExpertWave<'a> {
    fn capacities(capacity: u32) -> Result<Vec<u32>> {
        ensure!(matches!(capacity, 1 | 16 | 80 | 256 | 1024 | 4096), "invalid local expert capacity");
        let mut capacities: Vec<_> = [1, 16, 80].into_iter().filter(|&c| c < capacity).collect();
        capacities.push(capacity);
        Ok(capacities)
    }
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Self::capacities(capacity)?.into_iter().try_fold(capacity as usize * 10240, |sum, c| {
            sum.checked_add(usize::try_from(library.v41_local_expert_info(c)?.scratch_bytes)?)
                .context("local expert workspace size overflow")
        })
    }
    pub fn new(library: &'a NativeLibrary, weights: Rc<Vec<ExpertWeights<'a>>>,
        capacity: u32, budget: usize) -> Result<Self> {
        ensure!(!weights.is_empty() && weights.len() <= 40, "local experts require complete resident layers");
        for (layer, weight) in weights.iter().enumerate() {
            ensure!(weight.layer == ExpertLayer::BackboneFull { layer }
                && std::ptr::eq(weight.buffers[0].library, library), "local layer ownership differs");
        }
        ensure!(Self::device_bytes(library, capacity)? <= budget, "local expert workspace exceeds budget");
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        let mut states = Vec::new();
        for c in Self::capacities(capacity)? {
            let kernel = library.v41_local_expert_kernel(c)?;
            let scratch = DeviceAllocation::new(library, kernel.info().scratch_bytes as usize)?;
            let mut slots = [std::ptr::null_mut(); 44];
            unsafe { kernel.bind_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, &mut slots)?; }
            let initialized = unsafe { kernel.initialize_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, stream.raw) };
            let drained = unsafe { library.cuda_stream_synchronize(stream.raw) };
            initialized.and(drained)?;
            states.push(State { kernel, scratch, slots });
        }
        Ok(Self { stream, states, weights, output: DeviceAllocation::new(library, capacity as usize * 10240)?,
            reducer: library.v41_local_expert_reducer()?, capacity })
    }
    pub fn contains(&self, layer: usize) -> bool { layer < self.weights.len() }

    /// # Safety
    /// Router and shared outputs are complete and remain borrowed until this
    /// method drains, including on partial launch failure. No external mutation.
    pub unsafe fn execute(&mut self, routed: &RouterOutput<'_>, shared: &SharedOutput<'_>) -> Result<Ds41rtDeviceBuffer> {
        let rows = routed.rows;
        ensure!(self.contains(routed.layer) && rows > 0 && rows <= self.capacity,
            "local expert layer/rows are not resident or exceed capacity");
        ensure!(routed.binding()? == shared.binding()? && routed.layer == shared.layer && rows == shared.rows,
            "local router and shared expert binding differ");
        for (buffer, bytes) in [(routed.expert_input, rows as usize * 5280),
            (routed.ids, rows as usize * 24), (routed.routing, rows as usize * 24),
            (shared.values, rows as usize * 10240)] {
            ensure!(buffer.bytes == bytes && buffer.device_id == self.output.buffer.device_id,
                "local expert input extent or device differs");
        }
        let state = self.states.iter_mut().find(|s| s.kernel.info().capacity_rows >= rows)
            .context("local expert capacity state missing")?;
        self.weights[routed.layer].bind(&state.kernel, &mut state.slots)?;
        state.slots[0] = routed.expert_input.ptr;
        state.slots[1] = routed.ids.ptr;
        state.slots[2] = routed.routing.ptr;
        let info = state.kernel.info();
        let args = V41ExpertLaunchArgs { tensors: state.slots, num_tokens: rows as i32,
            max_rows: info.max_rows, scatter_rows: rows as i32 * 6, rows_padded: info.rows_padded,
            max_tasks: info.max_tasks, max_phys_tiles: info.max_phys_tiles,
            max_active_clusters: info.max_active_clusters, stream: self.stream.raw };
        // Own scratch through the drain, even if a pipeline fails after enqueue.
        let _scratch = &state.scratch;
        let launched = (|| -> Result<()> {
            unsafe { state.kernel.launch(&args)?; }
            unsafe { self.reducer.finish(state.slots[41].cast(), shared.values.ptr.cast(),
                self.output.buffer.ptr.cast(), rows, state.kernel.accumulates_tokens(), self.stream.raw) }
        })();
        let drained = unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) };
        launched.and(drained)?;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * 10240;
        Ok(output)
    }
}
