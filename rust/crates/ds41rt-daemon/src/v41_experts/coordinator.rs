//! One coordinator wave owns TP route planes through final native reduction.
use super::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41RouteReducer};
use ds41rt_transport::{
    v41_expert::{V41Tp4Pending, V41Tp4Tcp, V41_ROUTE_ROW_BYTES},
    ExpertProtocolV2Request,
};

pub(crate) struct NativeTp4Wave<'a> {
    transport: V41Tp4Tcp,
    // Drop drains the stream before fields release any GPU allocations.
    stream: LoadStream<'a>,
    planes: [DeviceAllocation<'a>; 4],
    shared: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    reducer: V41RouteReducer<'a>,
    ready_rows: Option<u32>,
}
impl<'a> NativeTp4Wave<'a> {
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP wave capacity"
        );
        (capacity as usize)
            .checked_mul(4 * V41_ROUTE_ROW_BYTES as usize + 2 * 5120 * 2)
            .context("native TP wave budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        transport: V41Tp4Tcp,
        available_bytes: usize,
    ) -> Result<Self> {
        let capacity = transport.capacity();
        ensure!(
            Self::device_bytes(capacity)? <= available_bytes,
            "native TP wave exceeds device budget"
        );
        let reducer = library.v41_route_reducer()?;
        let plane_bytes = capacity as usize * V41_ROUTE_ROW_BYTES as usize;
        let mut planes = Vec::with_capacity(4);
        for _ in 0..4 {
            planes.push(DeviceAllocation::new(library, plane_bytes)?);
        }
        let planes = planes.try_into().ok().expect("four native TP planes");
        let hidden_bytes = capacity as usize * 5120 * 2;
        Ok(Self {
            transport,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            planes,
            shared: DeviceAllocation::new(library, hidden_bytes)?,
            output: DeviceAllocation::new(library, hidden_bytes)?,
            library,
            reducer,
            ready_rows: None,
        })
    }
    /// TCP fallback with optional host BF16 shared-expert contribution.
    /// All GPU copies finish before frame storage can be reused, and reduction
    /// finishes before the borrowed output view is exposed. Cancellation leaves
    /// output unavailable until an entirely successful subsequent execution.
    pub async fn execute(
        &mut self,
        request: &ExpertProtocolV2Request,
        shared: Option<&[u8]>,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.ready_rows = None;
        self.synchronize()?;
        let rows = request.header.row_count;
        ensure!(
            rows > 0 && rows <= self.transport.capacity(),
            "native wave exceeds capacity"
        );
        let hidden_bytes = rows as usize * 5120 * 2;
        if let Some(shared) = shared {
            ensure!(
                shared.len() == hidden_bytes,
                "native shared output has wrong BF16 geometry"
            );
            self.library.copy_h2d(self.shared.buffer, shared)?;
        }
        self.execute_prepared(request, shared.is_some()).await
    }
    /// # Safety
    /// Shared device values are complete and immutable through the copy. The
    /// request and shared result must derive from the same actual block input.
    pub async unsafe fn execute_ffn<'w>(
        &'w mut self,
        request: &crate::v41_backbone_router::BoundExpertRequest,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    ) -> Result<NativeFfnOutput<'w>> {
        self.ready_rows = None;
        validate_shared(
            request,
            shared,
            self.shared.buffer,
            self.transport.capacity(),
        )?;
        unsafe { self.dispatch_ffn(request).await?.finish(shared).await }
    }
    /// Write all four expert requests before returning. The caller can then run
    /// shared FFN work on RTX while the Spark workers execute the routed experts.
    pub async fn dispatch_ffn<'w, 'r>(
        &'w mut self,
        request: &'r crate::v41_backbone_router::BoundExpertRequest,
    ) -> Result<NativePendingFfn<'w, 'a, 'r>> {
        self.ready_rows = None;
        self.synchronize()?;
        let header = &request.request().header;
        ensure!(
            header.layer_id as usize == request.binding().layer()
                && header.row_count > 0
                && header.row_count <= self.transport.capacity(),
            "native dispatched FFN identity or rows differ"
        );
        let capacity = self.transport.capacity();
        let pending = self.transport.dispatch(request.request()).await?;
        Ok(NativePendingFfn {
            pending,
            request,
            capacity,
            library: self.library,
            stream: &self.stream,
            planes: &self.planes,
            shared: self.shared.buffer,
            output: self.output.buffer,
            reducer: &self.reducer,
            ready_rows: &mut self.ready_rows,
        })
    }

    async fn execute_prepared(
        &mut self,
        request: &ExpertProtocolV2Request,
        has_shared: bool,
    ) -> Result<Ds41rtDeviceBuffer> {
        let rows = request.header.row_count;
        let library = self.library;
        let planes = &self.planes;
        self.transport
            .execute(request, |rank, first_row, bytes| {
                copy_chunk(library, planes, rank, first_row, bytes)
            })
            .await?;
        reduce_planes(
            self.library,
            &self.reducer,
            &self.stream,
            &self.planes,
            self.output.buffer,
            has_shared.then_some(self.shared.buffer),
            rows,
        )?;
        self.ready_rows = Some(rows);
        self.output()
    }
    /// Borrowed device view; never free it or retain it across wave reuse/drop.
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("native TP wave output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * 5120 * 2;
        Ok(output)
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }
}
impl Drop for NativeTp4Wave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native coordinator TP wave");
        }
    }
}

/// Complete ordered TP4 reduction plus the shared expert, borrowed until consumed.
pub(crate) struct NativeFfnOutput<'a> {
    pub values: Ds41rtDeviceBuffer,
    binding: crate::v41_attention_binding::QueryBinding,
    _owner: std::marker::PhantomData<&'a ()>,
}
impl NativeFfnOutput<'_> {
    pub fn binding(&self) -> crate::v41_attention_binding::QueryBinding {
        self.binding
    }
}

fn validate_shared(
    request: &crate::v41_backbone_router::BoundExpertRequest,
    shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    destination: Ds41rtDeviceBuffer,
    capacity: u32,
) -> Result<()> {
    let header = &request.request().header;
    ensure!(
        request.binding() == shared.binding()?
            && header.layer_id as usize == shared.layer
            && header.row_count == shared.rows
            && header.row_count > 0
            && header.row_count <= capacity
            && shared.values.bytes == header.row_count as usize * 10240
            && shared.values.device_id == destination.device_id,
        "native TP shared contribution differs from routed request"
    );
    Ok(())
}
fn copy_chunk(
    library: &NativeLibrary,
    planes: &[DeviceAllocation<'_>; 4],
    rank: usize,
    first_row: u32,
    bytes: &[u8],
) -> Result<()> {
    ensure!(rank < 4, "native route rank exceeds TP4");
    let offset = (first_row as usize)
        .checked_mul(V41_ROUTE_ROW_BYTES as usize)
        .context("native route chunk offset overflow")?;
    let end = offset
        .checked_add(bytes.len())
        .context("native route chunk extent overflow")?;
    ensure!(
        end <= planes[rank].buffer.bytes,
        "native route chunk exceeds destination"
    );
    let mut destination = planes[rank].buffer;
    destination.ptr = unsafe { destination.ptr.cast::<u8>().add(offset).cast() };
    destination.bytes = bytes.len();
    library.copy_h2d(destination, bytes)
}
fn reduce_planes(
    library: &NativeLibrary,
    reducer: &V41RouteReducer<'_>,
    stream: &LoadStream<'_>,
    planes: &[DeviceAllocation<'_>; 4],
    output: Ds41rtDeviceBuffer,
    shared: Option<Ds41rtDeviceBuffer>,
    rows: u32,
) -> Result<()> {
    let launched = unsafe {
        reducer.launch(
            std::array::from_fn(|rank| planes[rank].buffer.ptr.cast::<f32>().cast_const()),
            shared.map_or(std::ptr::null(), |b| b.ptr.cast()),
            output.ptr.cast(),
            rows,
            4,
            6,
            stream.raw,
        )
    };
    launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })
}

/// Borrows every mutable reduction buffer and owns all unread response sockets.
/// Dropping before completion leaves the wave unpublished and closes the sockets.
pub(crate) struct NativePendingFfn<'w, 'a, 'r> {
    pending: V41Tp4Pending<'w, 'r>,
    request: &'r crate::v41_backbone_router::BoundExpertRequest,
    capacity: u32,
    library: &'a NativeLibrary,
    stream: &'w LoadStream<'a>,
    planes: &'w [DeviceAllocation<'a>; 4],
    shared: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    reducer: &'w V41RouteReducer<'a>,
    ready_rows: &'w mut Option<u32>,
}
impl<'w> NativePendingFfn<'w, '_, '_> {
    /// # Safety
    /// Shared values hold the completed contribution for this exact request and
    /// remain immutable through the copy. Producers must be drained.
    pub async unsafe fn finish(
        self,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    ) -> Result<NativeFfnOutput<'w>> {
        validate_shared(self.request, shared, self.shared, self.capacity)?;
        let timing = std::time::Instant::now();
        self.library
            .copy_d2d(self.shared, shared.values, shared.values.bytes)?;
        let shared_copy_us = timing.elapsed().as_micros() as u64;
        let mut upload_us = 0u64;
        self.pending
            .receive(|rank, first_row, bytes| {
                let copy_start = std::time::Instant::now();
                let result = copy_chunk(self.library, self.planes, rank, first_row, bytes);
                upload_us += copy_start.elapsed().as_micros() as u64;
                result
            })
            .await?;
        let received_us = timing.elapsed().as_micros() as u64;
        let rows = self.request.request().header.row_count;
        reduce_planes(
            self.library,
            self.reducer,
            self.stream,
            self.planes,
            self.output,
            Some(self.shared),
            rows,
        )?;
        tracing::debug!(target: "ds41rt::timing", layer=self.request.request().header.layer_id, rows, shared_copy_us, upload_us, receive_us=received_us-shared_copy_us-upload_us, reduce_us=timing.elapsed().as_micros() as u64-received_us, "target collection");
        *self.ready_rows = Some(rows);
        let mut values = self.output;
        values.bytes = rows as usize * 10240;
        Ok(NativeFfnOutput {
            values,
            binding: self.request.binding(),
            _owner: std::marker::PhantomData,
        })
    }
}
