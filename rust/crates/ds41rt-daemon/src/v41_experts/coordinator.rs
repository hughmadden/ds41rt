//! One coordinator wave owns TP route planes through final native reduction.
use super::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41CompactReducer};
use ds41rt_transport::{
    v41_expert::{V41Tp4RocePending, V41Tp4Roce, V41_PARTIAL_ROW_BYTES},
    ExpertProtocolV2Request,
};

pub(crate) struct NativeTp4Wave<'a> {
    transport: V41Tp4Roce,
    // Drop drains the stream before fields release any GPU allocations.
    stream: LoadStream<'a>,
    planes: [DeviceAllocation<'a>; 4],
    upload_staging: HostAllocation<'a>,
    shared: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    reducer: V41CompactReducer<'a>,
    ready_rows: Option<u32>,
}
impl<'a> NativeTp4Wave<'a> {
    pub fn reset_connections(&mut self) {
        self.ready_rows = None;
        self.transport.reset_connections();
    }
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP wave capacity"
        );
        (capacity as usize)
            .checked_mul(4 * V41_PARTIAL_ROW_BYTES as usize + 2 * 5120 * 2)
            .context("native TP wave budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        transport: V41Tp4Roce,
        available_bytes: usize,
    ) -> Result<Self> {
        let capacity = transport.capacity();
        ensure!(
            Self::device_bytes(capacity)? <= available_bytes,
            "native TP wave exceeds device budget"
        );
        let reducer = library.v41_compact_reducer()?;
        let plane_bytes = capacity as usize * V41_PARTIAL_ROW_BYTES as usize;
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
            upload_staging: HostAllocation::new(library, capacity as usize * 4 * 10240)?,
            shared: DeviceAllocation::new(library, hidden_bytes)?,
            output: DeviceAllocation::new(library, hidden_bytes)?,
            library,
            reducer,
            ready_rows: None,
        })
    }
    /// RoCE execution with optional host BF16 shared-expert contribution.
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
    /// Enqueue all four expert requests before returning. The caller can then run
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
            upload_staging: &mut self.upload_staging,
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
    library.copy_h2d(chunk_destination(planes, rank, first_row, bytes.len())?, bytes)
}
fn chunk_destination(
    planes: &[DeviceAllocation<'_>; 4],
    rank: usize,
    first_row: u32,
    bytes: usize,
) -> Result<Ds41rtDeviceBuffer> {
    ensure!(rank < 4, "native route rank exceeds TP4");
    let offset = (first_row as usize)
        .checked_mul(V41_PARTIAL_ROW_BYTES as usize)
        .context("native route chunk offset overflow")?;
    let end = offset
        .checked_add(bytes)
        .context("native route chunk extent overflow")?;
    ensure!(
        end <= planes[rank].buffer.bytes,
        "native route chunk exceeds destination"
    );
    let mut destination = planes[rank].buffer;
    destination.ptr = unsafe { destination.ptr.cast::<u8>().add(offset).cast() };
    destination.bytes = bytes;
    Ok(destination)
}

/// A receive frame may be released after copying into this wave's pinned arena.
/// Each validated chunk owns a disjoint rank/row range until the final drain.
/// Cancellation and early errors must drain before the arena can be reused.
struct PlaneUploads<'s, 'a> {
    library: &'a NativeLibrary,
    stream: &'s LoadStream<'a>,
    planes: &'s [DeviceAllocation<'a>; 4],
    staging: &'s mut HostAllocation<'a>,
    rows: u32,
    pending: bool,
}
impl PlaneUploads<'_, '_> {
    fn copy(&mut self, rank: usize, first_row: u32, bytes: &[u8]) -> Result<()> {
        let destination = chunk_destination(self.planes, rank, first_row, bytes.len())?;
        let offset = first_row as usize * V41_PARTIAL_ROW_BYTES as usize;
        ensure!(offset + bytes.len() <= self.rows as usize * 10240,
            "native upload exceeds live rows");
        let plane_bytes = self.staging.buffer.bytes / 4;
        if self.rows as usize * 10240 > plane_bytes {
            return self.library.copy_h2d(destination, bytes);
        }
        let start = rank * plane_bytes + offset;
        let slice = &mut self.staging.bytes_mut()[start..start + bytes.len()];
        slice.copy_from_slice(bytes);
        self.pending = true;
        unsafe { self.library.copy_h2d_async(destination, slice, self.stream.raw) }
    }
}
impl Drop for PlaneUploads<'_, '_> {
    fn drop(&mut self) {
        if self.pending {
            if let Err(error) = unsafe { self.library.cuda_stream_synchronize(self.stream.raw) } {
                tracing::error!(%error, "draining interrupted native rank uploads");
            }
        }
    }
}
fn reduce_planes(
    library: &NativeLibrary,
    reducer: &V41CompactReducer<'_>,
    stream: &LoadStream<'_>,
    planes: &[DeviceAllocation<'_>; 4],
    output: Ds41rtDeviceBuffer,
    shared: Option<Ds41rtDeviceBuffer>,
    rows: u32,
) -> Result<()> {
    let launched = unsafe {
        reducer.reduce(
            std::array::from_fn(|rank| planes[rank].buffer.ptr.cast::<u16>().cast_const()),
            shared.map_or(std::ptr::null(), |b| b.ptr.cast()),
            output.ptr.cast(),
            rows,
            stream.raw,
        )
    };
    launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })
}

/// Borrows every mutable reduction buffer and owns all unread response sockets.
/// Dropping before completion leaves the wave unpublished and closes the sockets.
pub(crate) struct NativePendingFfn<'w, 'a, 'r> {
    pending: V41Tp4RocePending<'w, 'r>,
    request: &'r crate::v41_backbone_router::BoundExpertRequest,
    capacity: u32,
    library: &'a NativeLibrary,
    stream: &'w LoadStream<'a>,
    planes: &'w [DeviceAllocation<'a>; 4],
    upload_staging: &'w mut HostAllocation<'a>,
    shared: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    reducer: &'w V41CompactReducer<'a>,
    ready_rows: &'w mut Option<u32>,
}
impl<'w> NativePendingFfn<'w, '_, '_> {
    /// # Safety
    /// Shared values hold the completed contribution for this exact request and
    /// remain immutable until the final reduction drains. Producers must be drained.
    pub async unsafe fn finish(
        self,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    ) -> Result<NativeFfnOutput<'w>> {
        validate_shared(self.request, shared, self.shared, self.capacity)?;
        let timing = std::time::Instant::now();
        // The shared owner remains borrowed until reduction drains, so consume
        // its completed device output directly instead of copying it first.
        let shared_copy_us = 0u64;
        let mut uploads = PlaneUploads {
            library: self.library,
            stream: self.stream,
            planes: self.planes,
            staging: self.upload_staging,
            rows: self.request.request().header.row_count,
            pending: false,
        };
        let mut upload_us = 0u64;
        self.pending
            .receive(|rank, first_row, bytes| {
                let copy_start = std::time::Instant::now();
                let result = uploads.copy(rank, first_row, bytes);
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
            Some(shared.values),
            rows,
        )?;
        uploads.pending = false; // reduce_planes drained the same stream.
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

#[cfg(test)]
mod upload_tests {
    use super::*;

    #[test]
    fn pinned_rank_uploads_match_sync_and_drain_on_drop() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_PLANE_UPLOAD_LIBRARY") else {
            eprintln!("skip GPU rank upload test: DS41RT_PLANE_UPLOAD_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let planes = (0..4)
            .map(|_| DeviceAllocation::new(&library, 4096 * 10240))
            .collect::<Result<Vec<_>>>()?
            .try_into().ok().expect("four planes");
        let mut staging = HostAllocation::new(&library, 4096 * 4 * 10240)?;
        let shared = DeviceAllocation::new(&library, 4096 * 10240)?;
        let output = DeviceAllocation::new(&library, 4096 * 10240)?;
        let reducer = library.v41_compact_reducer()?;
        let staging_address = staging.buffer.ptr;
        for rows in [1u32, 6, 80, 81, 256, 1024, 4096, 6] {
            let bytes = rows as usize * 10240;
            let shared_bytes: Vec<u8> = (0..bytes / 2)
                .flat_map(|_| 0x3f00u16.to_ne_bytes()).collect();
            library.copy_h2d(shared.buffer, &shared_bytes)?;
            let payloads: Vec<Vec<u8>> = (0..4).map(|rank| {
                (0..bytes / 2).flat_map(|i| {
                    let value = rank as f32 + 1.0 + (i % 31) as f32 / 32.0;
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                }).collect()
            }).collect();
            for (rank, payload) in payloads.iter().enumerate() {
                copy_chunk(&library, &planes, rank, 0, payload)?;
            }
            reduce_planes(&library, &reducer, &stream, &planes,
                output.buffer, Some(shared.buffer), rows)?;
            let mut expected = vec![0u8; bytes];
            library.copy_d2h(&mut expected, Ds41rtDeviceBuffer { bytes, ..output.buffer })?;
            for plane in &planes { library.copy_h2d(plane.buffer, &vec![0; bytes])?; }
            {
                let mut uploads = PlaneUploads {
                    library: &library, stream: &stream, planes: &planes,
                    staging: &mut staging, rows, pending: false,
                };
                for first in (0..rows).step_by(3) {
                    let end = (first + 3).min(rows);
                    for rank in [3, 1, 0, 2] {
                        uploads.copy(rank, first,
                            &payloads[rank][first as usize * 10240..end as usize * 10240])?;
                    }
                }
                reduce_planes(&library, &reducer, &stream, &planes,
                    output.buffer, Some(shared.buffer), rows)?;
                uploads.pending = false;
            }
            let mut actual = vec![0u8; bytes];
            library.copy_d2h(&mut actual, Ds41rtDeviceBuffer { bytes, ..output.buffer })?;
            assert_eq!(actual, expected, "rows={rows}");
            assert_eq!(staging.buffer.ptr, staging_address);
            // Emulate dropping a receive future after its first upload, including
            // an early bounds error. The guard must drain before host reuse.
            for error in [false, true] {
                library.copy_h2d(planes[0].buffer, &vec![0; 10240])?;
                {
                    let mut uploads = PlaneUploads {
                        library: &library, stream: &stream, planes: &planes,
                        staging: &mut staging, rows: 1, pending: false,
                    };
                    uploads.copy(0, 0, &payloads[0][..10240])?;
                    if error { assert!(uploads.copy(4, 0, &payloads[0][..10240]).is_err()); }
                }
                // Overwriting immediately is safe only after the guard drains.
                staging.bytes_mut().fill(0);
                let mut actual = vec![0; 10240];
                library.copy_d2h(&mut actual, Ds41rtDeviceBuffer { bytes: 10240, ..planes[0].buffer })?;
                assert_eq!(actual, payloads[0][..10240]);
            }
            eprintln!("PASS rows={rows}: interleaved chunks/reduction exact, stable staging, cancellation/error drain");
        }
        Ok(())
    }
}
