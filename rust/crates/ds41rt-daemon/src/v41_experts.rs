//! Native V4.1 expert residency; one GPU worker owns each layer and its buffers.
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
mod execution;
pub(crate) mod service;
pub(crate) mod coordinator;
pub(crate) use execution::{ExpertExecution, ExpertExecutionBudget, HostExpertExchange};

use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    NativeLibrary, V41ExpertKernel, V41_EXPERT_POINTER_COUNT,
};
use ds41rt_loader::{OfficialV41Catalog, V41ExpertSelection};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertLayer {
    Backbone { layer: usize, rank: usize },
    Dspark { stage: usize },
}
impl ExpertLayer {
    fn expert(self, expert: usize) -> V41ExpertSelection {
        match self {
            Self::Backbone { layer, rank } => V41ExpertSelection::Backbone {
                layer,
                rank,
                expert,
            },
            Self::Dspark { stage } => V41ExpertSelection::Dspark { stage, expert },
        }
    }
    fn role(self) -> u32 {
        u32::from(matches!(self, Self::Backbone { .. }))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertLoadBudget {
    pub resident_bytes: usize,
    pub device_staging_bytes: usize,
    pub pinned_host_bytes: usize,
    pub read_scratch_bytes: usize,
}
impl ExpertLoadBudget {
    pub fn peak_device_bytes(self) -> Result<usize> {
        self.resident_bytes
            .checked_add(self.device_staging_bytes)
            .context("expert load budget overflow")
    }
}

/// Resident packed weights borrow the native library; no logical layer copy remains.
/// Captured graphs must be destroyed before this owner is dropped.
pub(crate) struct ExpertWeights<'a> {
    buffers: [DeviceAllocation<'a>; 4],
    layer: ExpertLayer,
    experts: usize,
    budget: ExpertLoadBudget,
}
impl<'a> ExpertWeights<'a> {
    fn layout(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<(ExpertLoadBudget, [usize; 4], u32, usize)> {
        let first = catalog.expert_staging(layer.expert(0))?;
        let info = library.v41_expert_info(16)?;
        ensure!(
            info.role == layer.role(),
            "native expert role does not match layer placement"
        );
        ensure!(
            info.logical_intermediate as usize == first.intermediate_size(),
            "native expert intermediate mismatch"
        );
        let experts = info.experts as usize;
        let packer = library.v41_expert_packer(info.logical_intermediate)?;
        let strides = packer.packed_bytes().map(usize::try_from);
        let mut sizes = [0usize; 4];
        for (size, stride) in sizes.iter_mut().zip(strides) {
            *size = stride?
                .checked_mul(experts)
                .context("resident expert allocation overflow")?;
        }
        let resident_bytes = sizes.iter().try_fold(0usize, |sum, size| {
            sum.checked_add(*size)
                .context("resident expert byte overflow")
        })?;
        let budget = ExpertLoadBudget {
            resident_bytes,
            device_staging_bytes: first.staging_bytes(),
            pinned_host_bytes: first.staging_bytes(),
            read_scratch_bytes: first
                .minimum_read_scratch_bytes()
                .checked_mul(64)
                .context("expert read scratch overflow")?,
        };
        Ok((budget, sizes, info.logical_intermediate, experts))
    }
    pub fn plan(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<ExpertLoadBudget> {
        Ok(Self::layout(library, catalog, layer)?.0)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
    ) -> Result<Self> {
        let (budget, sizes, intermediate, experts) = Self::layout(library, catalog, layer)?;
        let packer = library.v41_expert_packer(intermediate)?;
        ensure!(budget.peak_device_bytes()? <= available_device_bytes,
            "expert layer needs {} device bytes including staging, budget is {available_device_bytes}", budget.peak_device_bytes()?);
        // Fail role/device checks and allocation admission before opening payloads.
        let _kernel = library.v41_expert_kernel(16)?;
        let mut owned = Vec::with_capacity(4);
        for size in sizes {
            owned.push(DeviceAllocation::new(library, size)?);
        }
        let buffers: [DeviceAllocation<'a>; 4] =
            owned.try_into().ok().expect("four packed buffers");
        let device_staging = DeviceAllocation::new(library, budget.device_staging_bytes)?;
        let mut host = HostAllocation::new(library, budget.pinned_host_bytes)?;
        let mut read_scratch = vec![0; budget.read_scratch_bytes];
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        for expert in 0..experts {
            let plan = catalog.expert_staging(layer.expert(expert))?;
            plan.read_into(host.bytes_mut(), &mut read_scratch)?;
            unsafe {
                library.copy_host_buffer_h2d_async(
                    device_staging.buffer,
                    host.buffer,
                    plan.staging_bytes(),
                    stream.raw,
                )?;
                let sources = std::array::from_fn(|i| {
                    device_staging
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .add(plan.tensor_ranges()[i].start)
                        .cast_const()
                });
                let destinations = std::array::from_fn(|i| {
                    buffers[i]
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .add(expert * (sizes[i] / experts))
                });
                packer.pack(sources, destinations, stream.raw)?;
                // Reuse pinned and device staging only after this expert finishes.
                library.cuda_stream_synchronize(stream.raw)?;
            }
        }
        Ok(Self {
            buffers,
            layer,
            experts,
            budget,
        })
    }
    pub fn budget(&self) -> ExpertLoadBudget {
        self.budget
    }

    /// Bind prepared weight carriers after scratch binding and before argument validation.
    /// Returned raw slots borrow this owner and must not outlive it, including graph replay.
    pub fn bind(
        &self,
        kernel: &V41ExpertKernel<'_>,
        slots: &mut [*mut c_void; V41_EXPERT_POINTER_COUNT],
    ) -> Result<()> {
        ensure!(
            kernel.info().role == self.layer.role()
                && kernel.info().experts as usize == self.experts,
            "expert weights do not match kernel role"
        );
        ensure!(
            !slots[34].is_null() && !slots[37].is_null(),
            "bind initialized scratch before weights"
        );
        let [w13, s13, w2, s2] = std::array::from_fn(|i| self.buffers[i].buffer.ptr);
        for (slot, pointer) in [
            (22, w13),
            (23, s13),
            (24, w2),
            (25, s2),
            (26, s13),
            (27, s2),
            (28, slots[34]),
            (29, slots[34]),
            (30, w13),
            (31, s13),
            (32, w2),
            (33, s2),
            (38, slots[37]),
            (39, slots[37]),
        ] {
            slots[slot] = pointer;
        }
        Ok(())
    }
}
