//! Direct checkpoint loading of contiguous vocabulary rows onto one GPU.
use super::*;
use std::ops::Range;

pub(crate) struct VocabularyShard<'a> {
    allocation: DeviceAllocation<'a>,
    tokens: Range<usize>,
}
impl<'a> VocabularyShard<'a> {
    const VOCAB: usize = 129280;
    const ROW_BYTES: usize = 5120 * 2;

    pub fn device_bytes(catalog: &OfficialV41Catalog, tokens: Range<usize>) -> Result<usize> {
        ensure!(tokens.start < tokens.end && tokens.end <= Self::VOCAB,
            "invalid vocabulary shard token range");
        ensure!(VocabularyHead::plan(catalog)? == Self::VOCAB * Self::ROW_BYTES,
            "unexpected vocabulary checkpoint geometry");
        Ok(tokens.len() * Self::ROW_BYTES)
    }

    /// The caller scopes construction/destruction to the owning GPU. Admission
    /// precedes allocation and payload reads. No full-head GPU staging is used.
    pub fn load(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
        tokens: Range<usize>, budget: usize, staging_bytes: usize) -> Result<Self> {
        let bytes = Self::device_bytes(catalog, tokens.clone())?;
        ensure!(bytes <= budget, "vocabulary shard exceeds device budget");
        ensure!((1..=64 * 1024 * 1024).contains(&staging_bytes),
            "vocabulary pinned staging must be 1 byte through 64 MiB");
        let reader = catalog.coordinator_tensor_reader("head.weight")?;
        let mut staging = HostAllocation::new(library, staging_bytes.min(bytes))?;
        let allocation = DeviceAllocation::new(library, bytes)?;
        let source_start = tokens.start * Self::ROW_BYTES;
        let mut offset = 0;
        while offset < bytes {
            let count = staging_bytes.min(bytes - offset);
            let source = &mut staging.bytes_mut()[..count];
            reader.read_into((source_start + offset) as u64, source)?;
            let destination = Ds41rtDeviceBuffer {
                ptr: unsafe { allocation.buffer.ptr.cast::<u8>().add(offset).cast() },
                bytes: count, ..allocation.buffer
            };
            library.copy_h2d(destination, source)?;
            offset += count;
        }
        Ok(Self { allocation, tokens })
    }
    pub fn weight(&self) -> Ds41rtDeviceBuffer { self.allocation.buffer }
    pub fn tokens(&self) -> Range<usize> { self.tokens.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_memory::device::Device;

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn dual_vocabulary_shards_match_complete_checkpoint_ranges() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        let reader = catalog.coordinator_tensor_reader("head.weight")?;
        lib.cuda_set_device(0)?;
        let mut shards = Vec::new();
        for (id, tokens) in [(0, 0..64640), (1, 64640..129280)] {
            let device = Device { library: &lib, id };
            let bytes = VocabularyShard::device_bytes(&catalog, tokens.clone())?;
            assert_eq!(bytes, 661913600);
            assert!(device.run(|| VocabularyShard::load(&lib, &catalog, tokens.clone(), bytes - 1, 7 << 20)).is_err());
            let shard = device.own(|| VocabularyShard::load(&lib, &catalog, tokens.clone(), bytes, 7 << 20))?;
            assert_eq!(shard.tokens(), tokens);
            assert_eq!(shard.weight().bytes, bytes);
            let mut expected = vec![0u8; 7 << 20];
            let mut actual = vec![0u8; expected.len()];
            for offset in (0..bytes).step_by(expected.len()) {
                let count = expected.len().min(bytes - offset);
                reader.read_into((tokens.start * VocabularyShard::ROW_BYTES + offset) as u64,
                    &mut expected[..count])?;
                let buffer = shard.weight();
                let source = Ds41rtDeviceBuffer {
                    ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes: count, ..buffer
                };
                device.run(|| lib.copy_d2h(&mut actual[..count], source))?;
                ensure!(actual[..count] == expected[..count], "vocabulary shard differs on GPU {id} at byte {offset}");
            }
            eprintln!("PASS GPU {id} vocabulary tokens {tokens:?}: all {bytes} bytes match checkpoint");
            shards.push(shard);
        }
        for tokens in [0..0, 129280..129281, 129281..129282] {
            assert!(VocabularyShard::device_bytes(&catalog, tokens).is_err());
        }
        drop(shards);
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }
}
