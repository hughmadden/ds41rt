//! GPU projections follow Engram layers; CPU table gathering stays unchanged.
use super::layer::{EngramGate, EngramLayerWeights};
use super::*;
use crate::v41_backbone_cache::CachePlacement;
use crate::v41_backbone_lane::BackboneLane;
use crate::v41_memory::device::{Device, DeviceOwner};
use ds41rt_loader::OfficialV41Catalog;

pub(crate) struct PlacedEngramWeights<'a> {
    weights: [DeviceOwner<'a, EngramLayerWeights<'a>>; 2],
}
impl<'a> PlacedEngramWeights<'a> {
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        map: CachePlacement,
    ) -> Result<[usize; 2]> {
        let mut bytes = [0usize; 2];
        for (index, layer) in [1, 14].into_iter().enumerate() {
            let gpu = map.attention(layer)?;
            bytes[gpu] = bytes[gpu]
                .checked_add(EngramLayerWeights::device_bytes(library, catalog, index)?)
                .context("placed Engram weight budget overflow")?;
        }
        Ok(bytes)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        map: CachePlacement,
        budgets: [usize; 2],
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog, map)?
                .into_iter()
                .zip(budgets)
                .all(|(n, b)| n <= b),
            "Engram weights exceed a GPU budget"
        );
        let weights = [1, 14]
            .into_iter()
            .enumerate()
            .map(|(index, layer)| {
                let device = Device {
                    library,
                    id: map.attention(layer)? as i32,
                };
                device.own(|| {
                    EngramLayerWeights::load(
                        library,
                        catalog,
                        index,
                        EngramLayerWeights::device_bytes(library, catalog, index)?,
                        staging,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .ok()
            .expect("two Engram layers");
        Ok(Self { weights })
    }
}
/// Per-request-lane uploads are shared between gates on the same GPU only.
pub(crate) struct PlacedEngram<'w, 'a> {
    gates: [DeviceOwner<'a, EngramGate<'w, 'a>>; 2],
    uploads: [Option<DeviceOwner<'a, EngramDeviceRows<'a>>>; 2],
}
impl<'w, 'a> PlacedEngram<'w, 'a> {
    pub fn device_bytes(
        library: &NativeLibrary,
        map: CachePlacement,
        capacity: usize,
    ) -> Result<[usize; 2]> {
        let gate = EngramGate::device_bytes(library, capacity)?;
        let upload = EngramDeviceRows::device_bytes(capacity)?;
        let mut bytes = [0usize; 2];
        for layer in [1, 14] {
            let gpu = map.attention(layer)?;
            if bytes[gpu] == 0 {
                bytes[gpu] = upload;
            }
            bytes[gpu] = bytes[gpu]
                .checked_add(gate)
                .context("placed Engram workspace overflow")?;
        }
        Ok(bytes)
    }
    pub fn new(
        weights: &'w PlacedEngramWeights<'a>,
        capacity: usize,
        budgets: [usize; 2],
    ) -> Result<Self> {
        let library = weights.weights[0].device.library;
        let gate_bytes = EngramGate::device_bytes(library, capacity)?;
        let upload_bytes = EngramDeviceRows::device_bytes(capacity)?;
        let mut required = [0usize; 2];
        for weight in &weights.weights {
            let gpu = weight.device.id as usize;
            if required[gpu] == 0 {
                required[gpu] = upload_bytes;
            }
            required[gpu] = required[gpu]
                .checked_add(gate_bytes)
                .context("placed Engram workspace overflow")?;
        }
        ensure!(
            required.into_iter().zip(budgets).all(|(n, b)| n <= b),
            "Engram workspace exceeds a GPU budget"
        );
        let gates = weights
            .weights
            .iter()
            .map(|w| w.device.own(|| EngramGate::new(w, capacity, gate_bytes)))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .ok()
            .expect("two Engram gates");
        let mut uploads = [None, None];
        for gpu in 0..2 {
            if required[gpu] > 0 {
                uploads[gpu] = Some(
                    Device {
                        library,
                        id: gpu as i32,
                    }
                    .own(|| EngramDeviceRows::new(library, capacity, upload_bytes))?,
                );
            }
        }
        Ok(Self { gates, uploads })
    }
    /// # Safety
    /// The gathered CPU rows identify this prepared lane's pending Engram layer
    /// and exact request/token order. Retain the gather lease and lane through
    /// completion or drained cancellation. Each request lane owns its own gates.
    pub async unsafe fn apply(
        &mut self,
        lane: &mut DeviceOwner<'a, BackboneLane<'_, 'a>>,
        gathered: &EngramGatherView<'_>,
    ) -> Result<()> {
        let index = gathered.layer_index;
        ensure!(index < 2, "invalid gathered Engram layer");
        let gate = &mut self.gates[index];
        let device = gate.device;
        ensure!(
            device.id == lane.device.id && lane.pending_engram()?.0 == gate.layer(),
            "placed Engram lane/layer/GPU differs"
        );
        let upload = self.uploads[device.id as usize]
            .as_mut()
            .context("Engram upload GPU absent")?;
        device
            .future(async {
                let rows = upload.upload_cooperative(gathered).await?;
                unsafe {
                    lane.get_mut()
                        .apply_engram_cooperative(gate.get_mut(), &rows)
                        .await
                }
            })
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn placed_engram_budgets_share_uploads_only_within_gpu() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let same = CachePlacement::encoder_decoder();
        let split = CachePlacement::new(std::array::from_fn(|layer| usize::from(layer >= 14)))?;
        let same_bytes = PlacedEngram::device_bytes(&lib, same, 16)?;
        let split_bytes = PlacedEngram::device_bytes(&lib, split, 16)?;
        assert_eq!(same_bytes[1], 0);
        assert_eq!(
            split_bytes.iter().sum::<usize>(),
            same_bytes[0] + EngramDeviceRows::device_bytes(16)?
        );
        for map in [same, split] {
            let weight_bytes = PlacedEngramWeights::device_bytes(&lib, &catalog, map)?;
            let weights =
                PlacedEngramWeights::load(&lib, &catalog, map, weight_bytes, 1024 * 1024)?;
            let mut resident = [0usize; 2];
            for w in &weights.weights {
                resident[w.device.id as usize] += w.resident_bytes();
            }
            assert_eq!(resident, weight_bytes);
            let bytes = PlacedEngram::device_bytes(&lib, map, 16)?;
            assert!(PlacedEngram::new(&weights, 16, [bytes[0] - 1, bytes[1]]).is_err());
            let lane = PlacedEngram::new(&weights, 16, bytes)?;
            for (index, layer) in [1, 14].into_iter().enumerate() {
                let gpu = map.attention(layer)?;
                assert_eq!(lane.gates[index].device.id, gpu as i32);
                assert_eq!(lane.uploads[gpu].as_ref().unwrap().device.id, gpu as i32);
            }
            for gpu in 0..2 {
                assert_eq!(lane.uploads[gpu].is_some(), bytes[gpu] > 0);
            }
            assert_eq!(lib.cuda_get_device()?, 0);
            eprintln!("placed Engram weight bytes={weight_bytes:?}; lane C16 bytes={bytes:?}");
        }
        Ok(())
    }
}
