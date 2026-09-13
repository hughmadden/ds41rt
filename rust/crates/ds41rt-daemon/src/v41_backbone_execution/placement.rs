//! Load each cache producer beside its SWA/compressed storage.
use super::*;

/// One lane's per-layer producer workspaces, owned by the assigned GPUs.
pub(crate) struct PlacedProducerWaves<'w, 'a> {
    pub windows: Vec<DeviceOwner<'a, WindowWave<'w, 'a>>>,
    pub sources: Vec<DeviceOwner<'a, CompressorWave<'w, 'a>>>,
}
impl<'w, 'a> PlacedProducerWaves<'w, 'a> {
    pub fn device_bytes(
        library: &NativeLibrary,
        placement: CachePlacement,
        capacity: u32,
    ) -> Result<[usize; 2]> {
        let mut bytes = [0usize; 2];
        let window = WindowWave::device_bytes(library, capacity)?;
        for layer in 0..40 {
            let gpu = placement.attention(layer)?;
            bytes[gpu] = bytes[gpu]
                .checked_add(window)
                .context("placed SWA workspace overflow")?;
        }
        for layer in SOURCES {
            let gpu = placement.attention(layer)?;
            bytes[gpu] = bytes[gpu]
                .checked_add(CompressorWave::device_bytes(layer, capacity as usize)?)
                .context("placed compressor workspace overflow")?;
        }
        Ok(bytes)
    }
    pub fn new(
        weights: &'w CacheProducerWeights<'a>,
        capacity: u32,
        budgets: [usize; 2],
    ) -> Result<Self> {
        let placement = weights
            .placement
            .context("placed producer workspaces need a placement map")?;
        let bytes = Self::device_bytes(weights.library, placement, capacity)?;
        ensure!(
            bytes
                .into_iter()
                .zip(budgets)
                .all(|(need, budget)| need <= budget),
            "cache producer workspaces exceed a GPU budget"
        );
        let windows = weights
            .windows
            .iter()
            .map(|weights| {
                weights.device.own(|| {
                    weights.wave(
                        capacity,
                        WindowWave::device_bytes(weights.device.library, capacity)?,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = weights
            .sources
            .iter()
            .zip(SOURCES)
            .map(|(weights, layer)| {
                weights.device.own(|| {
                    weights.wave(
                        capacity as usize,
                        CompressorWave::device_bytes(layer, capacity as usize)?,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { windows, sources })
    }
}

impl<'a> CacheProducerWeights<'a> {
    pub fn distributed_device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        placement: CachePlacement,
    ) -> Result<[usize; 2]> {
        let mut bytes = [0usize; 2];
        for layer in 0..40 {
            let gpu = placement.attention(layer)?;
            let sink =
                NativeRtxTensors::plan(catalog, &[format!("layers.{layer}.attn.attn_sink")])?;
            ensure!(sink == 256, "unexpected attention sink size");
            bytes[gpu] = bytes[gpu]
                .checked_add(WindowWeights::device_bytes(library, catalog, layer)?)
                .and_then(|n| n.checked_add(sink))
                .context("placed window weight budget overflow")?;
        }
        for layer in SOURCES {
            let gpu = placement.attention(layer)?;
            bytes[gpu] = bytes[gpu]
                .checked_add(CompressorWeights::device_bytes(catalog, layer)?)
                .context("placed compressor weight budget overflow")?;
        }
        Ok(bytes)
    }
    pub fn load_distributed(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        placement: CachePlacement,
        budgets: [usize; 2],
        staging: usize,
    ) -> Result<Self> {
        let bytes = Self::distributed_device_bytes(library, catalog, placement)?;
        ensure!(
            bytes
                .into_iter()
                .zip(budgets)
                .all(|(need, budget)| need <= budget),
            "cache producer weights exceed a GPU budget"
        );
        Self::load_placed(library, catalog, staging, Some(placement))
    }
    pub(super) fn load_placed(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        staging: usize,
        placement: Option<CachePlacement>,
    ) -> Result<Self> {
        let original = library.cuda_get_device()?;
        let device_for = |layer| -> Result<Device<'a>> {
            Ok(Device {
                library,
                id: match placement {
                    Some(p) => p.attention(layer)? as i32,
                    None => original,
                },
            })
        };
        let windows = (0..40)
            .map(|layer| {
                device_for(layer)?.own(|| {
                    WindowWeights::load(
                        library,
                        catalog,
                        layer,
                        WindowWeights::device_bytes(library, catalog, layer)?,
                        staging,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = SOURCES
            .into_iter()
            .map(|layer| {
                device_for(layer)?.own(|| {
                    CompressorWeights::load(
                        library,
                        catalog,
                        layer,
                        CompressorWeights::device_bytes(catalog, layer)?,
                        staging,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let devices = if placement.is_some() {
            vec![0, 1]
        } else {
            vec![original]
        };
        let mut sinks = Vec::new();
        for id in devices {
            let names = (0..40)
                .filter_map(|layer| match device_for(layer) {
                    Ok(device) if device.id == id => {
                        Some(Ok(format!("layers.{layer}.attn.attn_sink")))
                    }
                    Ok(_) => None,
                    Err(error) => Some(Err(error)),
                })
                .collect::<Result<Vec<_>>>()?;
            if names.is_empty() {
                continue;
            }
            sinks.push(Device { library, id }.own(|| {
                NativeRtxTensors::load(
                    library,
                    catalog,
                    &names,
                    NativeRtxTensors::plan(catalog, &names)?,
                    staging,
                )
            })?);
        }
        let sink_views = (0..40)
            .map(|layer| {
                let id = device_for(layer)?.id;
                sinks
                    .iter()
                    .find(|group| group.device.id == id)
                    .context("attention sink GPU missing")?
                    .get()
                    .get(&format!("layers.{layer}.attn.attn_sink"))
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .ok()
            .expect("40 attention sinks");
        Ok(Self {
            library,
            windows,
            sources,
            _sinks: sinks,
            sink_views,
            placement,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_window::{WindowChunk, WindowState};
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT, and two CUDA GPUs"]
    fn real_placed_producers_and_window_graphs_match_across_devices() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let placement = CachePlacement::encoder_decoder();
        let budgets = CacheProducerWeights::distributed_device_bytes(&lib, &catalog, placement)?;
        assert_eq!(
            budgets.iter().sum::<usize>(),
            CacheProducerWeights::device_bytes(&lib, &catalog)?
        );
        let weights = CacheProducerWeights::load_distributed(
            &lib,
            &catalog,
            placement,
            budgets,
            16 * 1024 * 1024,
        )?;
        let devices = [
            Device {
                library: &lib,
                id: 0,
            },
            Device {
                library: &lib,
                id: 1,
            },
        ];
        let reference = devices[0].own(|| {
            WindowWeights::load(
                &lib,
                &catalog,
                20,
                WindowWeights::device_bytes(&lib, &catalog, 20)?,
                16 * 1024 * 1024,
            )
        })?;
        let reference_compressor = devices[0].own(|| {
            CompressorWeights::load(
                &lib,
                &catalog,
                20,
                CompressorWeights::device_bytes(&catalog, 20)?,
                16 * 1024 * 1024,
            )
        })?;
        let workspace = PlacedProducerWaves::device_bytes(&lib, placement, 16)?;
        assert_eq!(
            workspace.iter().sum::<usize>(),
            BackboneExecution::workspace_bytes(&lib, 16)?
        );
        let mut placed = PlacedProducerWaves::new(&weights, 16, workspace)?;
        for layer in 0..40 {
            assert_eq!(
                weights.windows[layer].device.id,
                placement.attention(layer)? as i32
            );
            assert_eq!(
                weights.sink_views[layer].device_id,
                placement.attention(layer)? as i32
            );
            assert_eq!(
                placed.windows[layer].input().device_id,
                placement.attention(layer)? as i32
            );
        }
        for (source, gpu) in weights.sources.iter().zip(placement.sources()) {
            assert_eq!(source.device.id, gpu as i32);
        }
        for (source, gpu) in placed.sources.iter().zip(placement.sources()) {
            assert_eq!(source.input().device_id, gpu as i32);
        }
        assert!(BackboneExecution::new(&weights, 16, usize::MAX).is_err());
        let mut states = [
            devices[0].own(|| WindowState::new(&lib, 20, 1, WindowState::device_bytes(20, 1)?))?,
            devices[1].own(|| WindowState::new(&lib, 20, 1, WindowState::device_bytes(20, 1)?))?,
        ];
        let leases = [
            states[0].begin_request(0, 1)?,
            states[1].begin_request(0, 1)?,
        ];
        let mut reference_wave =
            devices[0].own(|| reference.wave(16, WindowWave::device_bytes(&lib, 16)?))?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut previous = None;
        for changed in [false, true] {
            let host: Vec<u8> = (0..16 * 5120)
                .flat_map(|i| {
                    let value: f32 = if changed {
                        (i % 17) as f32 / 32.0 - 0.25
                    } else {
                        0.0
                    };
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                })
                .collect();
            let mut output = Vec::new();
            for rank in 0..2 {
                let wave = if rank == 0 {
                    reference_wave.get_mut()
                } else {
                    placed.windows[20].get_mut()
                };
                let state = states[rank].get();
                let chunks = [WindowChunk {
                    lease: leases[rank],
                    position: 0,
                    tokens: 16,
                }];
                let bytes = runtime.block_on(devices[rank].future(async {
                    lib.copy_h2d(wave.input(), &host)?;
                    tokio::task::yield_now().await;
                    if !changed {
                        unsafe {
                            wave.capture(state, &chunks)?;
                        }
                    }
                    unsafe {
                        wave.replay(state, &chunks)?;
                    }
                    let result = wave.output(state)?;
                    [
                        result.values,
                        result.scales,
                        result.projected,
                        result.normalized,
                        result.frequencies,
                    ]
                    .into_iter()
                    .map(|buffer| {
                        let mut bytes = vec![0; buffer.bytes];
                        lib.copy_d2h(&mut bytes, buffer)?;
                        Ok(bytes)
                    })
                    .collect::<Result<Vec<_>>>()
                }))?;
                output.push(bytes);
                assert_eq!(lib.cuda_get_device()?, 0);
            }
            assert!(
                output[0] == output[1],
                "placed SWA producer graph output differs between GPUs"
            );
            if let Some(old) = &previous {
                assert!(*old != output[0], "SWA graph ignored changed input");
            }
            previous = Some(output.remove(0));
        }
        let mut source_states = [
            devices[0]
                .own(|| crate::v41_compressor::CompressorState::new(&lib, 20, 1, 2, usize::MAX))?,
            devices[1]
                .own(|| crate::v41_compressor::CompressorState::new(&lib, 20, 1, 2, usize::MAX))?,
        ];
        let source_leases = [
            source_states[0].begin_request(0, 1)?,
            source_states[1].begin_request(0, 1)?,
        ];
        let mut reference_compressor_wave = devices[0]
            .own(|| reference_compressor.wave(16, CompressorWave::device_bytes(20, 16)?))?;
        let mut previous = None;
        for changed in [false, true] {
            let host: Vec<u8> = (0..16 * 5120)
                .flat_map(|i| {
                    let value: f32 = if changed {
                        (i % 17) as f32 / 32.0 - 0.25
                    } else {
                        0.0
                    };
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                })
                .collect();
            let mut output = Vec::new();
            for rank in 0..2 {
                let wave = if rank == 0 {
                    reference_compressor_wave.get_mut()
                } else {
                    placed.sources[3].get_mut()
                };
                let state = source_states[rank].get();
                let chunks = [crate::v41_compressor::CompressorChunk {
                    lease: source_leases[rank],
                    position: 0,
                    tokens: 16,
                }];
                output.push(runtime.block_on(devices[rank].future(async {
                    lib.copy_h2d(wave.input(), &host)?;
                    tokio::task::yield_now().await;
                    if !changed {
                        unsafe {
                            wave.capture(state, &chunks)?;
                        }
                    }
                    unsafe {
                        wave.replay(state, &chunks)?;
                    }
                    let result = wave.output(state)?;
                    [
                        result.buffer,
                        result.index_key,
                        result.index_packed,
                        result.index_scales,
                        result.kv_values,
                        result.kv_scales,
                    ]
                    .into_iter()
                    .map(|buffer| {
                        let mut bytes = vec![0; buffer.bytes];
                        lib.copy_d2h(&mut bytes, buffer)?;
                        Ok(bytes)
                    })
                    .collect::<Result<Vec<_>>>()
                }))?);
                assert_eq!(lib.cuda_get_device()?, 0);
            }
            assert!(
                output[0] == output[1],
                "placed source-20 graph output differs between GPUs"
            );
            if let Some(old) = &previous {
                assert!(*old != output[0], "compressor graph ignored changed input");
            }
            previous = Some(output.remove(0));
        }
        eprintln!(
            "placed cache-producer weight budgets: GPU0={} GPU1={}",
            budgets[0], budgets[1]
        );
        eprintln!(
            "placed C16 producer workspace budgets per lane: GPU0={} GPU1={}",
            workspace[0], workspace[1]
        );
        let single = CacheProducerWeights::load(
            &lib,
            &catalog,
            CacheProducerWeights::device_bytes(&lib, &catalog)?,
            16 * 1024 * 1024,
        )?;
        let single_execution =
            BackboneExecution::new(&single, 16, BackboneExecution::workspace_bytes(&lib, 16)?)?;
        assert!(single.sink_views.iter().all(|buffer| buffer.device_id == 0));
        assert!(
            single_execution
                .windows
                .iter()
                .all(|wave| wave.input().device_id == 0)
        );
        assert!(
            single_execution
                .sources
                .iter()
                .all(|wave| wave.input().device_id == 0)
        );
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }
}
