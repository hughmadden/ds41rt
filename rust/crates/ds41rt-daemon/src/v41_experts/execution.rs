//! Per-wave stable expert I/O, scratch, stream and graph ownership.
use super::{DeviceAllocation, ExpertWeights, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs, V41RouteReducer,
};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertExecutionBudget {
    pub scratch_bytes: usize,
    pub hidden_bytes: usize,
    pub routing_bytes: usize,
    pub output_and_shared_bytes: usize,
}
impl ExpertExecutionBudget {
    pub fn total(self) -> Result<usize> {
        [
            self.scratch_bytes,
            self.hidden_bytes,
            self.routing_bytes,
            self.output_and_shared_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |sum, n| {
            sum.checked_add(n)
                .context("expert execution budget overflow")
        })
    }
}

/// One exclusive wave's buffers and graph borrow immutable resident weights.
/// Upstream producers and transport must obey this owner's stream/lifetime contract.
pub(crate) struct ExpertExecution<'weights, 'library> {
    // Destroy stream before allocations; Drop drains and destroys the graph first.
    stream: LoadStream<'library>,
    _weights: &'weights ExpertWeights<'library>,
    library: &'library NativeLibrary,
    kernel: V41ExpertKernel<'library>,
    reducer: V41RouteReducer<'library>,
    scratch: DeviceAllocation<'library>,
    hidden: DeviceAllocation<'library>,
    ids: DeviceAllocation<'library>,
    routing: DeviceAllocation<'library>,
    output: Option<DeviceAllocation<'library>>,
    shared: Option<DeviceAllocation<'library>>,
    slots: [*mut c_void; 44],
    graph: Option<(*mut c_void, u32, bool)>,
    budget: ExpertExecutionBudget,
}
impl<'library> ExpertWeights<'library> {
    pub fn execution_budget(&self, capacity: u32) -> Result<ExpertExecutionBudget> {
        let library = self.buffers[0].library;
        let info = library.v41_expert_info(capacity)?;
        let hidden = (capacity as usize)
            .checked_mul(5120 * 2)
            .context("hidden buffer overflow")?;
        let routing = (capacity as usize)
            .checked_mul(info.topk as usize * 8)
            .context("routing buffer overflow")?;
        Ok(ExpertExecutionBudget {
            scratch_bytes: usize::try_from(info.scratch_bytes)?,
            hidden_bytes: hidden,
            routing_bytes: routing,
            output_and_shared_bytes: if info.role == 0 {
                hidden.checked_mul(2).context("output buffer overflow")?
            } else {
                0
            },
        })
    }
    pub fn execution(
        &self,
        capacity: u32,
        available_device_bytes: usize,
    ) -> Result<ExpertExecution<'_, 'library>> {
        let library = self.buffers[0].library;
        let budget = self.execution_budget(capacity)?;
        ensure!(
            budget.total()? <= available_device_bytes,
            "expert execution buffers exceed device budget"
        );
        let kernel = library.v41_expert_kernel(capacity)?;
        let reducer = library.v41_route_reducer()?;
        let scratch = DeviceAllocation::new(library, budget.scratch_bytes)?;
        let hidden = DeviceAllocation::new(library, budget.hidden_bytes)?;
        let ids = DeviceAllocation::new(library, budget.routing_bytes / 2)?;
        let routing = DeviceAllocation::new(library, budget.routing_bytes / 2)?;
        let output = if kernel.info().role == 0 {
            Some(DeviceAllocation::new(library, budget.hidden_bytes)?)
        } else {
            None
        };
        let shared = if kernel.info().role == 0 {
            Some(DeviceAllocation::new(library, budget.hidden_bytes)?)
        } else {
            None
        };
        let mut slots = [std::ptr::null_mut(); 44];
        unsafe {
            kernel.bind_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, &mut slots)?;
        }
        self.bind(&kernel, &mut slots)?;
        slots[0] = hidden.buffer.ptr;
        slots[1] = ids.buffer.ptr;
        slots[2] = routing.buffer.ptr;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        unsafe {
            kernel.initialize_scratch(
                scratch.buffer.ptr,
                scratch.buffer.bytes as u64,
                stream.raw,
            )?;
            library.cuda_stream_synchronize(stream.raw)?;
        }
        Ok(ExpertExecution {
            stream,
            _weights: self,
            library,
            kernel,
            reducer,
            scratch,
            hidden,
            ids,
            routing,
            output,
            shared,
            slots,
            graph: None,
            budget,
        })
    }
}
impl ExpertExecution<'_, '_> {
    pub fn budget(&self) -> ExpertExecutionBudget {
        self.budget
    }
    pub fn stream(&self) -> *mut c_void {
        self.stream.raw
    }
    /// Borrowed BF16 hidden, I32 route IDs and FP32 route weights; never free these views.
    /// Enqueue producers on stream() or provide event ordering before launch/replay.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 3] {
        [self.hidden.buffer, self.ids.buffer, self.routing.buffer]
    }
    pub fn shared(&self) -> Option<Ds41rtDeviceBuffer> {
        self.shared.as_ref().map(|b| b.buffer)
    }
    pub fn output(&self) -> Option<Ds41rtDeviceBuffer> {
        self.output.as_ref().map(|b| b.buffer)
    }
    pub fn route_partials(&self, rows: u32) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            rows > 0 && rows <= self.kernel.info().capacity_rows,
            "invalid expert output row count"
        );
        Ok(Ds41rtDeviceBuffer {
            ptr: self.slots[41],
            bytes: rows as usize * self.kernel.info().topk as usize * 5120 * 4,
            device_id: self.scratch.buffer.device_id,
            flags: 0,
        })
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }

    /// # Safety
    /// Initialize valid BF16 hidden, in-range I32 expert IDs and finite nonnegative
    /// FP32 routing weights for `rows` before this operation, with stream ordering.
    /// Initialize shared BF16 output too if requested. External readers/writers must
    /// finish before storage is reused; all operations belong to this GPU worker.
    pub unsafe fn launch(&mut self, rows: u32, include_shared: bool) -> Result<()> {
        ensure!(
            !include_shared || self.shared.is_some(),
            "shared output belongs on coordinator RTX"
        );
        let args = V41ExpertLaunchArgs::new(self.kernel.info(), self.slots, rows, self.stream.raw)?;
        unsafe {
            self.kernel.launch(&args)?;
        }
        if let Some(output) = &self.output {
            unsafe {
                self.reducer.launch(
                    [
                        self.slots[41].cast(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    ],
                    if include_shared {
                        self.shared.as_ref().unwrap().buffer.ptr.cast()
                    } else {
                        std::ptr::null()
                    },
                    output.buffer.ptr.cast(),
                    rows,
                    1,
                    3,
                    self.stream.raw,
                )?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// Same initialized-buffer and ordering contract as launch; captures fixed rows
    /// and shared-output participation. Input values may change between replays.
    pub unsafe fn capture(&mut self, rows: u32, include_shared: bool) -> Result<()> {
        ensure!(
            self.graph.is_none(),
            "expert execution already has a captured graph"
        );
        unsafe {
            self.launch(rows, include_shared)?;
        }
        self.synchronize()?;
        unsafe {
            self.library.cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launch = unsafe { self.launch(rows, include_shared) };
        // End capture on both paths so an error cannot leave the stream capturing.
        let captured = unsafe { self.library.cuda_graph_end_capture(self.stream.raw) };
        match (launch, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows, include_shared));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                unsafe {
                    self.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(error)
            }
            (Err(error), Err(_)) => Err(error),
            (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// The captured rows' inputs (and optional shared output) must satisfy launch's
    /// contract, with writes ordered before replay and reads ordered after it.
    pub unsafe fn replay(&mut self) -> Result<()> {
        let (graph, _, _) = self.graph.context("expert graph has not been captured")?;
        unsafe { self.library.cuda_graph_launch(graph, self.stream.raw) }
    }
}
impl Drop for ExpertExecution<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining V4.1 expert execution");
        }
        if let Some((graph, _, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying V4.1 expert execution graph");
            }
        }
    }
}

/// Reusable host exchange for the TCP fallback; RDMA can consume device route views.
pub(crate) struct HostExpertExchange {
    ids: Vec<i32>,
    routing: Vec<f32>,
    partials: Vec<u8>,
}
impl HostExpertExchange {
    pub fn new(capacity: u32) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "unsupported native host exchange capacity"
        );
        let routes = capacity as usize * 6;
        Ok(Self {
            ids: vec![0; routes],
            routing: vec![0.0; routes],
            partials: vec![0; routes * 5120 * 4],
        })
    }
}
impl ExpertExecution<'_, '_> {
    /// Host fallback from a validated wire request to ordered FP32 route response.
    /// The borrowed response prevents reuse of exchange storage until encoding/send ends.
    pub fn execute_host_request<'a>(
        &mut self,
        request: &ds41rt_transport::v41_expert::V41BackboneRequest<'_>,
        executor_id: u64,
        exchange: &'a mut HostExpertExchange,
    ) -> Result<ds41rt_transport::ExpertProtocolV2ResponseRef<'a>> {
        let super::ExpertLayer::Backbone { layer, .. } = self._weights.layer else {
            anyhow::bail!("backbone requests cannot execute on RTX dSpark weights");
        };
        ensure!(
            request.layer() as usize == layer,
            "request does not match resident expert layer"
        );
        ensure!(executor_id != 0, "native response needs executor identity");
        ensure!(
            request.rows() <= self.kernel.info().capacity_rows,
            "request exceeds native execution capacity"
        );
        let routes = request.rows() as usize * 6;
        let bytes = request.plane_bytes()?;
        ensure!(
            exchange.partials.len() >= bytes,
            "host exchange is too small"
        );
        request.copy_routes_into(&mut exchange.ids, &mut exchange.routing)?;
        self.synchronize()?;
        self.library
            .copy_h2d(self.hidden.buffer, request.hidden())?;
        // Supported hosts and CUDA use the checkpoint/wire little-endian byte order.
        ensure!(
            cfg!(target_endian = "little"),
            "native host exchange requires little-endian storage"
        );
        unsafe {
            self.library.copy_h2d(
                self.ids.buffer,
                std::slice::from_raw_parts(exchange.ids.as_ptr().cast::<u8>(), routes * 4),
            )?;
            self.library.copy_h2d(
                self.routing.buffer,
                std::slice::from_raw_parts(exchange.routing.as_ptr().cast::<u8>(), routes * 4),
            )?;
            self.launch(request.rows(), false)?;
        }
        self.synchronize()?;
        self.library.copy_d2h(
            &mut exchange.partials[..bytes],
            self.route_partials(request.rows())?,
        )?;
        request.response(executor_id, &exchange.partials[..bytes])
    }
}
