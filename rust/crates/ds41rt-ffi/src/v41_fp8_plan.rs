//! Preloaded FP8 capacity plans with disjoint, caller-owned scratch slices.
use crate::{Ds41rtDeviceBuffer, NativeLibrary, V41Fp8Info, V41Fp8Kernel};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

#[derive(Clone, Copy, Debug)]
pub struct V41Fp8PlanInfo {
    pub capacity_rows: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub scratch_bytes: u64,
    pub packed_weight_scale_bytes: u64,
}

pub struct V41Fp8Plan<'a> {
    info: V41Fp8PlanInfo,
    kernels: Vec<(V41Fp8Kernel<'a>, usize)>,
}

fn capacities(capacity: u32) -> Result<Vec<u32>> {
    ensure!((1..=4096).contains(&capacity), "unsupported FP8 plan capacity");
    // The plan bounds live rows independently of the AOT kernel's maximum.
    // Kernel scratch remains fully sized; callers may size input/output storage
    // to live capacity because launches validate and predicate on actual rows.
    let compiled = [1, 16, 80, 256, 1024, 4096].into_iter()
        .find(|&n| n >= capacity).context("FP8 capacity has no compiled kernel")?;
    let mut result: Vec<_> = [1, 16, 80].into_iter().filter(|&n| n < compiled).collect();
    result.push(compiled);
    Ok(result)
}

fn layout(infos: &[V41Fp8Info]) -> Result<(V41Fp8PlanInfo, Vec<usize>)> {
    let largest = infos.last().context("empty FP8 plan")?;
    let mut bytes = 0u64;
    let mut offsets = Vec::with_capacity(infos.len());
    for info in infos {
        ensure!(
            info.input_dim == largest.input_dim
                && info.output_dim == largest.output_dim
                && info.packed_weight_scale_bytes == largest.packed_weight_scale_bytes,
            "FP8 capacity plans disagree on weight geometry"
        );
        offsets.push(usize::try_from(bytes)?);
        let aligned = info
            .scratch_bytes
            .checked_add(255)
            .context("FP8 scratch alignment overflow")?
            & !255;
        bytes = bytes
            .checked_add(aligned)
            .context("FP8 plan scratch overflow")?;
    }
    usize::try_from(bytes)?;
    Ok((
        V41Fp8PlanInfo {
            capacity_rows: largest.capacity_rows,
            input_dim: largest.input_dim,
            output_dim: largest.output_dim,
            scratch_bytes: bytes,
            packed_weight_scale_bytes: largest.packed_weight_scale_bytes,
        },
        offsets,
    ))
}

impl NativeLibrary {
    pub fn v41_fp8_matrix_plan_info(
        &self,
        capacity: u32,
        k: u32,
        n: u32,
    ) -> Result<V41Fp8PlanInfo> {
        let infos = capacities(capacity)?
            .into_iter()
            .map(|rows| self.v41_fp8_matrix_info(rows, k, n))
            .collect::<Result<Vec<_>>>()?;
        let mut info = layout(&infos)?.0;
        info.capacity_rows = capacity;
        Ok(info)
    }
    pub fn v41_fp8_matrix_plan(&self, capacity: u32, k: u32, n: u32) -> Result<V41Fp8Plan<'_>> {
        let kernels = capacities(capacity)?
            .into_iter()
            .map(|rows| self.v41_fp8_matrix_kernel(rows, k, n))
            .collect::<Result<Vec<_>>>()?;
        let infos: Vec<_> = kernels.iter().map(V41Fp8Kernel::info).collect();
        let (mut info, offsets) = layout(&infos)?;
        info.capacity_rows = capacity;
        Ok(V41Fp8Plan {
            info,
            kernels: kernels.into_iter().zip(offsets).collect(),
        })
    }
}

impl V41Fp8Plan<'_> {
    pub fn info(&self) -> V41Fp8PlanInfo {
        self.info
    }

    fn select(&self, rows: u32) -> Result<&(V41Fp8Kernel<'_>, usize)> {
        ensure!(
            rows > 0 && rows <= self.info.capacity_rows,
            "FP8 rows exceed planned capacity"
        );
        self.kernels
            .iter()
            .find(|(kernel, _)| rows <= kernel.info().capacity_rows)
            .context("FP8 capacity plan missing")
    }

    fn scratch_slice(
        &self,
        mut buffer: Ds41rtDeviceBuffer,
        kernel: &V41Fp8Kernel<'_>,
        offset: usize,
    ) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            !buffer.ptr.is_null() && buffer.bytes >= usize::try_from(self.info.scratch_bytes)?,
            "FP8 plan scratch is null or too small"
        );
        // Caller supplies a live allocation covering the complete planned arena.
        buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };
        buffer.bytes = usize::try_from(kernel.info().scratch_bytes)?;
        Ok(buffer)
    }

    /// # Safety
    /// Same current-device ownership as V41Fp8Kernel::initialize_scratch. Run
    /// once outside graph capture; the entire arena stays live through replay.
    pub unsafe fn initialize_scratch(
        &self,
        scratch: Ds41rtDeviceBuffer,
        alpha: Ds41rtDeviceBuffer,
        stream: *mut c_void,
    ) -> Result<()> {
        for (kernel, offset) in &self.kernels {
            unsafe {
                kernel.initialize_scratch(
                    self.scratch_slice(scratch, kernel, *offset)?,
                    alpha,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// # Safety
    /// Same buffer, stream and replay ownership contract as V41Fp8Kernel::launch.
    /// No module resolution, allocation or scratch initialization occurs here.
    pub unsafe fn launch(
        &self,
        source: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        scratch: Ds41rtDeviceBuffer,
        alpha: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let (kernel, offset) = self.select(rows)?;
        unsafe {
            kernel.launch(
                source,
                weight,
                scales,
                self.scratch_slice(scratch, kernel, *offset)?,
                alpha,
                output,
                rows,
                stream,
            )
        }
    }

    /// # Safety
    /// Same ownership contract as V41Fp8Kernel::launch_rope, including frequencies.
    pub unsafe fn launch_rope(
        &self,
        source: Ds41rtDeviceBuffer,
        frequencies: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        scratch: Ds41rtDeviceBuffer,
        alpha: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let (kernel, offset) = self.select(rows)?;
        unsafe {
            kernel.launch_rope(
                source,
                frequencies,
                weight,
                scales,
                self.scratch_slice(scratch, kernel, *offset)?,
                alpha,
                output,
                rows,
                stream,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scratch_slices_preserve_alignment_and_large_offsets() {
        let infos: Vec<_> = [1, 16, 80, 4096]
            .into_iter()
            .enumerate()
            .map(|(i, capacity_rows)| V41Fp8Info {
                capacity_rows,
                input_dim: 5120,
                output_dim: 1280,
                scratch_bytes: if i == 0 { (1u64 << 31) + 1 } else { 257 },
                packed_weight_scale_bytes: 204800,
                ..Default::default()
            })
            .collect();
        let (info, offsets) = layout(&infos).unwrap();
        assert_eq!(
            offsets,
            [
                0,
                (1usize << 31) + 256,
                (1usize << 31) + 768,
                (1usize << 31) + 1280
            ]
        );
        assert_eq!(info.scratch_bytes, (1u64 << 31) + 1792);
        assert_eq!(capacities(4096).unwrap(), [1, 16, 80, 4096]);
        assert_eq!(capacities(16).unwrap(), [1, 16]);
        assert_eq!(capacities(40).unwrap(), [1, 16, 80]);
        assert_eq!(capacities(1).unwrap(), [1]);
        assert!(capacities(0).is_err());
        assert_eq!(capacities(2048).unwrap(), [1, 16, 80, 4096]);
        assert_eq!(capacities(128).unwrap(), [1, 16, 80, 256]);
        assert_eq!(capacities(257).unwrap(), [1, 16, 80, 1024]);
        assert!(capacities(4097).is_err());
        let mut invalid = infos;
        invalid[0].scratch_bytes = u64::MAX;
        assert!(layout(&invalid).is_err());
        invalid[0].scratch_bytes = 256;
        invalid[0].input_dim = 256;
        assert!(layout(&invalid).is_err());
    }
}
