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
    diagnostic_wob_m1_cap16: bool,
}

const WOB_PROBE_ENV: &str = "DS41RT_DIAGNOSTIC_WOB_M1_CAP16";

fn diagnostic_wob_m1_cap16(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(value) => {
            ensure!(value == "0" || value == "1", "{WOB_PROBE_ENV} must be 0 or 1");
            Ok(value == "1")
        }
    }
}

fn select_index(info: V41Fp8PlanInfo, rows: u32, diagnostic: bool,
    mut capacities: impl Iterator<Item = u32>) -> Result<usize> {
    ensure!(rows > 0 && rows <= info.capacity_rows, "FP8 rows exceed planned capacity");
    if diagnostic && rows == 1 && info.input_dim == 8192 && info.output_dim == 5120 {
        // Select only an already-loaded exact capacity. Never widen live row
        // bounds or borrow a larger scratch layout under the diagnostic flag.
        capacities.position(|capacity| capacity == 16)
            .context("WO-B M1 diagnostic requires an existing capacity16 kernel")
    } else {
        capacities.position(|capacity| rows <= capacity).context("FP8 capacity plan missing")
    }
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
        // Capture the flag once, before graph capture. Normal serving never
        // reads the environment on launch or changes a captured graph's recipe.
        let diagnostic_wob_m1_cap16 = diagnostic_wob_m1_cap16(
            std::env::var_os(WOB_PROBE_ENV).as_deref())?;
        let planned = capacities(capacity)?;
        if diagnostic_wob_m1_cap16 && k == 8192 && n == 5120 {
            ensure!(planned.contains(&16),
                "WO-B M1 diagnostic requires an existing capacity16 kernel");
        }
        let kernels = planned
            .into_iter()
            .map(|rows| self.v41_fp8_matrix_kernel(rows, k, n))
            .collect::<Result<Vec<_>>>()?;
        let infos: Vec<_> = kernels.iter().map(V41Fp8Kernel::info).collect();
        let (mut info, offsets) = layout(&infos)?;
        info.capacity_rows = capacity;
        Ok(V41Fp8Plan {
            info,
            kernels: kernels.into_iter().zip(offsets).collect(),
            diagnostic_wob_m1_cap16,
        })
    }
}

impl V41Fp8Plan<'_> {
    pub fn info(&self) -> V41Fp8PlanInfo {
        self.info
    }

    fn select(&self, rows: u32) -> Result<&(V41Fp8Kernel<'_>, usize)> {
        let index = select_index(self.info, rows, self.diagnostic_wob_m1_cap16,
            self.kernels.iter().map(|(kernel, _)| kernel.info().capacity_rows))?;
        Ok(&self.kernels[index])
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

    fn selection_info(capacity_rows: u32, input_dim: u32, output_dim: u32) -> V41Fp8PlanInfo {
        V41Fp8PlanInfo { capacity_rows, input_dim, output_dim,
            scratch_bytes: 4096, packed_weight_scale_bytes: 1024 }
    }

    #[test]
    fn wob_probe_requires_explicit_flag() {
        assert!(!diagnostic_wob_m1_cap16(None).unwrap());
        assert!(!diagnostic_wob_m1_cap16(Some(std::ffi::OsStr::new("0"))).unwrap());
        assert!(diagnostic_wob_m1_cap16(Some(std::ffi::OsStr::new("1"))).unwrap());
        for value in ["", "true", "16", " 1"] {
            assert!(diagnostic_wob_m1_cap16(Some(std::ffi::OsStr::new(value))).is_err());
        }
    }

    #[test]
    fn wob_probe_changes_only_single_row_wob() {
        let caps = [1, 16, 80, 1024];
        let info = selection_info(1024, 8192, 5120);
        assert_eq!(select_index(info, 1, false, caps.into_iter()).unwrap(), 0);
        assert_eq!(select_index(info, 1, true, caps.into_iter()).unwrap(), 1);
        for rows in [2, 4, 16, 17, 80, 81, 1024] {
            assert_eq!(select_index(info, rows, false, caps.into_iter()).unwrap(),
                select_index(info, rows, true, caps.into_iter()).unwrap());
        }
        for (k, n) in [(8192, 512), (5120, 8192), (15360, 5120), (8192, 5121)] {
            assert_eq!(select_index(selection_info(1024, k, n), 1, true,
                caps.into_iter()).unwrap(), 0);
        }
    }

    #[test]
    fn wob_probe_preserves_live_bounds_and_requires_existing_exact_capacity() {
        let info = selection_info(4, 8192, 5120);
        for enabled in [false, true] {
            for rows in [0, 5, 16] {
                assert!(select_index(info, rows, enabled, [1, 16].into_iter()).is_err());
            }
        }
        assert_eq!(select_index(info, 1, true, [1, 16].into_iter()).unwrap(), 1);
        for caps in [vec![], vec![1], vec![1, 80], vec![1, 1024]] {
            assert!(select_index(info, 1, true, caps.into_iter()).is_err());
        }
        assert!(select_index(info, 4, false, [1].into_iter()).is_err());
        assert!(select_index(selection_info(1, 8192, 5120), 1, true,
            capacities(1).unwrap().into_iter()).is_err());
    }

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
