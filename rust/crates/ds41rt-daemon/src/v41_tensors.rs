//! Official coordinator tensors retain their native checkpoint representations.
use crate::v41_memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::OfficialV41Catalog;
use std::collections::{BTreeMap, BTreeSet};
#[path = "v41_tensors/vocabulary_shard.rs"]
mod vocabulary_shard;
pub(crate) use vocabulary_shard::VocabularyShard;

pub(crate) struct NativeRtxTensors<'a> {
    tensors: BTreeMap<String, DeviceAllocation<'a>>,
    resident_bytes: usize,
}
impl<'a> NativeRtxTensors<'a> {
    pub fn plan(catalog: &OfficialV41Catalog, names: &[String]) -> Result<usize> {
        ensure!(!names.is_empty(), "RTX tensor set is empty");
        let mut seen = BTreeSet::new();
        names.iter().try_fold(0usize, |total, name| {
            ensure!(seen.insert(name), "duplicate RTX tensor {name}");
            ensure!(
                !name.contains(".ffn.experts."),
                "routed expert weights require native expert packing"
            );
            let bytes = usize::try_from(catalog.device_tensor_bytes(name, None)?)?;
            total
                .checked_add(bytes)
                .context("RTX resident tensor budget overflow")
        })
    }
    /// Admission precedes allocation and payload reads; bounded pinned staging is
    /// released after synchronous uploads. Engram tables cannot enter this path.
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        names: &[String],
        device_budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        let resident_bytes = Self::plan(catalog, names)?;
        ensure!(
            resident_bytes <= device_budget,
            "RTX tensor set exceeds device budget"
        );
        ensure!(
            staging_bytes > 0 && staging_bytes <= 64 * 1024 * 1024,
            "RTX pinned staging must be 1 byte through 64 MiB"
        );
        let mut staging = HostAllocation::new(library, staging_bytes)?;
        let mut tensors = BTreeMap::new();
        for name in names {
            let reader = catalog.coordinator_tensor_reader(name)?;
            let bytes = usize::try_from(reader.bytes())?;
            let allocation = DeviceAllocation::new(library, bytes)?;
            let mut offset = 0;
            while offset < bytes {
                let count = staging_bytes.min(bytes - offset);
                let source = &mut staging.bytes_mut()[..count];
                reader.read_into(offset as u64, source)?;
                let mut destination = allocation.buffer;
                destination.ptr = unsafe { destination.ptr.cast::<u8>().add(offset).cast() };
                destination.bytes = count;
                library.copy_h2d(destination, source)?;
                offset += count;
            }
            tensors.insert(name.clone(), allocation);
        }
        Ok(Self {
            tensors,
            resident_bytes,
        })
    }
    /// Borrowed native representation; never free or retain after the owner drops.
    pub fn get(&self, name: &str) -> Result<Ds41rtDeviceBuffer> {
        Ok(self
            .tensors
            .get(name)
            .with_context(|| format!("RTX tensor is not resident: {name}"))?
            .buffer)
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

/// One coordinator copy of the official BF16 vocabulary weight, shared by the
/// backbone and dSpark; each execution owns its own handle and workspace.
pub(crate) struct VocabularyHead<'library> {
    tensors: NativeRtxTensors<'library>,
}
impl<'library> VocabularyHead<'library> {
    pub fn plan(catalog: &OfficialV41Catalog) -> Result<usize> {
        NativeRtxTensors::plan(catalog, &["head.weight".into()])
    }
    pub fn load(
        library: &'library NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        Ok(Self {
            tensors: NativeRtxTensors::load(
                library,
                catalog,
                &["head.weight".into()],
                budget,
                staging_bytes,
            )?,
        })
    }
    pub fn weight(&self) -> Result<Ds41rtDeviceBuffer> {
        self.tensors.get("head.weight")
    }
}
