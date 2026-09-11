use super::*;
use crate::v41_memory::DeviceAllocation;

/// One admitted prompt's final encoder rows, retained across encoder chunk reuse.
/// Allocate at admission; capture copies only the final window's intersecting rows.
pub(crate) struct EncoderSuffix<'a> {
    residual: DeviceAllocation<'a>,
    pre: DeviceAllocation<'a>,
    tokens: Vec<u64>,
    next: u64,
    end: u64,
    binding: QueryBinding,
    invalid: bool,
}
impl<'a> EncoderSuffix<'a> {
    pub fn device_bytes(prompt_end: u64) -> Result<usize> {
        ensure!((1..=1048576).contains(&prompt_end), "invalid encoder suffix extent");
        Ok(prompt_end.min(128) as usize * (40960 + 16))
    }
    pub fn new(library: &'a NativeLibrary, prompt_end: u64, budget: usize) -> Result<Self> {
        ensure!(Self::device_bytes(prompt_end)? <= budget, "encoder suffix exceeds budget");
        let start = prompt_end.saturating_sub(128);
        let rows = (prompt_end - start) as usize;
        Ok(Self { residual: DeviceAllocation::new(library, rows * 40960)?,
            pre: DeviceAllocation::new(library, rows * 16)?, tokens: (start..prompt_end).collect(),
            next: start, end: prompt_end, binding: QueryBinding::new(19)?, invalid: false })
    }
    /// The producer is a completed encoder chunk from this admitted request.
    /// Incomplete/failed captures never expose an output; discard on request failure.
    pub fn capture(&mut self, output: &BlockOutput<'_>) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        let rows = output.tokens.len();
        ensure!(valid && output.layer == 19 && output.binding.layer() == 19
            && rows > 0 && output.tokens.windows(2).all(|p| p[0].checked_add(1) == Some(p[1]))
            && output.tokens[rows - 1] < self.end
            && output.residual.bytes == rows * 40960 && output.pre.bytes == rows * 16,
            "encoder suffix producer layer, positions or extents differ");
        let first = output.tokens.partition_point(|&position| position < self.tokens[0]);
        if first < rows {
            ensure!(output.tokens[first] == self.next, "encoder suffix has a gap or duplicate rows");
            let offset = (self.next - self.tokens[0]) as usize;
            for (source, destination, stride) in [(output.residual, self.residual.buffer, 40960),
                (output.pre, self.pre.buffer, 16)] {
                ensure!(source.device_id == destination.device_id, "encoder suffix device differs");
                let bytes = (rows - first) * stride;
                let src = Ds41rtDeviceBuffer { ptr: unsafe { source.ptr.cast::<u8>().add(first * stride).cast() }, bytes, ..source };
                let dst = Ds41rtDeviceBuffer { ptr: unsafe { destination.ptr.cast::<u8>().add(offset * stride).cast() }, bytes, ..destination };
                self.residual.library.copy_d2d(dst, src, bytes)?;
            }
            self.next = output.tokens[rows - 1] + 1;
        }
        self.invalid = false;
        Ok(())
    }
    pub fn output(&self) -> Result<BlockOutput<'_>> {
        ensure!(!self.invalid && self.next == self.end, "encoder suffix is incomplete or invalid");
        Ok(BlockOutput { binding: self.binding, residual: self.residual.buffer, pre: self.pre.buffer,
            layer: 19, tokens: &self.tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_encoder_suffix_crosses_chunk_boundaries() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_ENCODER_SUFFIX_LIBRARY") else {
            eprintln!("skip GPU suffix test: DS41RT_ENCODER_SUFFIX_LIBRARY unset"); return Ok(());
        };
        let lib = unsafe { NativeLibrary::load(path)? };
        let residual = DeviceAllocation::new(&lib, 80 * 40960)?;
        let pre = DeviceAllocation::new(&lib, 80 * 16)?;
        for end in [1, 127, 128, 129, 257, 2049] {
            let mut suffix = EncoderSuffix::new(&lib, end, EncoderSuffix::device_bytes(end)?)?;
            assert!(suffix.output().is_err());
            for start in (0..end).step_by(80) {
                let tokens: Vec<u64> = (start..end.min(start + 80)).collect();
                let rows = tokens.len();
                for (allocation, stride, salt) in [(&residual, 40960, 0), (&pre, 16, 79)] {
                    let values: Vec<u8> = tokens.iter().flat_map(|&p| std::iter::repeat_n(((p + salt) % 251) as u8, stride)).collect();
                    lib.copy_h2d(allocation.buffer, &values)?;
                }
                let output = BlockOutput { binding: QueryBinding::new(19)?, layer: 19, tokens: &tokens,
                    residual: Ds41rtDeviceBuffer { bytes: rows * 40960, ..residual.buffer },
                    pre: Ds41rtDeviceBuffer { bytes: rows * 16, ..pre.buffer } };
                suffix.capture(&output)?;
                if start + (rows as u64) < end { assert!(suffix.output().is_err()); }
            }
            let output = suffix.output()?;
            assert_eq!(output.tokens, &(end.saturating_sub(128)..end).collect::<Vec<_>>());
            for (buffer, stride, salt) in [(output.residual, 40960, 0), (output.pre, 16, 79)] {
                let mut actual = vec![0; buffer.bytes]; lib.copy_d2h(&mut actual, buffer)?;
                let expected: Vec<u8> = output.tokens.iter().flat_map(|&p| std::iter::repeat_n(((p + salt) % 251) as u8, stride)).collect();
                assert_eq!(actual, expected);
            }
        }
        let mut gap = EncoderSuffix::new(&lib, 257, EncoderSuffix::device_bytes(257)?)?;
        let output = BlockOutput { binding: QueryBinding::new(19)?, layer: 19, tokens: &[130],
            residual: Ds41rtDeviceBuffer { bytes: 40960, ..residual.buffer },
            pre: Ds41rtDeviceBuffer { bytes: 16, ..pre.buffer } };
        assert!(gap.capture(&output).is_err()); assert!(gap.output().is_err());
        assert!(EncoderSuffix::device_bytes(0).is_err());
        assert!(EncoderSuffix::device_bytes(1048577).is_err());
        eprintln!("PASS encoder suffix 1/127/128/129/257/2049 tokens: residual/pre bytes, chunk crossings, incomplete and gap guards");
        Ok(())
    }
}
