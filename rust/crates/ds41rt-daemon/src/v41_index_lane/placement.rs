//! Index query weights and retained selection workspaces follow their KV source.
use super::*;

impl<'a> IndexLaneWeights<'a> {
    pub fn distributed_device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        placement: CachePlacement,
    ) -> Result<[usize; 2]> {
        let mut bytes = [0usize; 2];
        for layer in LAYERS {
            let gpu = placement.attention(layer)?;
            bytes[gpu] = bytes[gpu]
                .checked_add(IndexQueryWeights::device_bytes(library, catalog, layer)?)
                .context("placed index weight budget overflow")?;
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
        ensure!(
            Self::distributed_device_bytes(library, catalog, placement)?
                .into_iter()
                .zip(budgets)
                .all(|(need, budget)| need <= budget),
            "index weights exceed a GPU budget"
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
        let weights = LAYERS
            .into_iter()
            .map(|layer| {
                let id = match placement {
                    Some(map) => map.attention(layer)? as i32,
                    None => original,
                };
                Device { library, id }.own(|| {
                    IndexQueryWeights::load(
                        library,
                        catalog,
                        layer,
                        IndexQueryWeights::device_bytes(library, catalog, layer)?,
                        staging,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            library,
            weights,
            placement,
        })
    }
}

impl<'w, 'a> IndexLane<'w, 'a> {
    pub fn placed_workspace_bytes(
        library: &NativeLibrary,
        placement: CachePlacement,
        capacity: u32,
        gpu: usize,
    ) -> Result<[usize; 3]> {
        ensure!(gpu < 2, "invalid index workspace GPU");
        let mut bytes = Self::workspace_bytes(library, capacity)?;
        bytes[2] = if placement.attention(20)? == gpu {
            crate::v41_index_selection::IndexSelectionWave::shared_device_bytes(capacity as usize)?
        } else { 0 };
        if !LAYERS
            .iter()
            .any(|&layer| placement.attention(layer).ok() == Some(gpu))
        {
            bytes = [0; 3];
        }
        Ok(bytes)
    }
    /// Each request lane has independent selection state on each participating
    /// GPU. Source 20 and all decoder reindex layers remain on the same GPU, so
    /// the retained source-20 candidates never need a peer copy.
    pub fn new_on_device(
        weights: &'w IndexLaneWeights<'a>,
        capacity: u32,
        budget: usize,
        gpu: usize,
    ) -> Result<DeviceOwner<'a, Self>> {
        ensure!(
            gpu < 2 && weights.placement.is_some(),
            "placed index lane requires GPU 0 or 1 and distributed weights"
        );
        let device = Device {
            library: weights.library,
            id: gpu as i32,
        };
        device.own(|| Self::new_inner(weights, capacity, budget, Some(device.id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_attention_query::{AttentionQueryWave, AttentionQueryWeights};
    use crate::v41_backbone_cache::{BackboneCache, CacheWork};
    use crate::v41_backbone_execution::{CacheProducerWeights, PlacedProducerWaves};
    use ds41rt_transport::ExpertV2SourceKind;

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and two CUDA GPUs"]
    fn placed_index_scorer_graph_replay_on_both_devices() -> Result<()> {
        use crate::v41_memory::{DeviceAllocation, LoadStream};
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        // Initialize both first, then revisit GPU0 to catch overwritten AOT symbols.
        for id in [0, 1] {
            Device { library: &lib, id }.run(|| {
                lib.v41_index_scores()?;
                Ok(())
            })?;
        }
        for id in [0, 1, 0] {
            Device { library: &lib, id }.run(|| {
                let fill = |bytes: &[u8]| -> Result<DeviceAllocation<'_>> {
                    let allocation = DeviceAllocation::new(&lib, bytes.len())?;
                    lib.copy_h2d(allocation.buffer, bytes)?;
                    Ok(allocation)
                };
                let u64s = |values: &[u64]| {
                    values
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>()
                };
                let q = fill(&vec![0x22; 2048])?;
                let qs = fill(&[127; 128])?;
                let weights = fill(&[0x80, 0x3f].repeat(32))?;
                let keys = fill(&vec![0x22; 256 * 64])?;
                let scales = fill(&vec![127; 256 * 4])?;
                let pages = fill(&0u32.to_ne_bytes())?;
                let lengths = fill(&u64s(&[256]))?;
                let metadata = fill(&u64s(&[0, 256, 256, 0, 0, 1]))?;
                let positions = fill(&u64s(&[0, 255, 256, u64::MAX]))?;
                let output = DeviceAllocation::new(&lib, 16)?;
                let stream = LoadStream {
                    library: &lib,
                    raw: lib.cuda_stream_create()?,
                };
                let score = lib.v41_index_scores()?;
                unsafe {
                    lib.cuda_graph_begin_capture(stream.raw)?;
                }
                let queued = unsafe {
                    score.execute_overlay(
                        q.buffer,
                        qs.buffer,
                        weights.buffer,
                        keys.buffer,
                        scales.buffer,
                        pages.buffer,
                        lengths.buffer,
                        metadata.buffer,
                        positions.buffer,
                        output.buffer,
                        keys.buffer,
                        scales.buffer,
                        256,
                        1,
                        4,
                        1,
                        1,
                        256,
                        stream.raw,
                    )
                };
                let captured = unsafe { lib.cuda_graph_end_capture(stream.raw) };
                let graph = captured?;
                let checked = (|| -> Result<()> {
                    queued?;
                    for (byte, expected) in [(0x22, 4096.0f32), (0, 0.0)] {
                        lib.copy_h2d(q.buffer, &vec![byte; 2048])?;
                        unsafe {
                            lib.cuda_graph_launch(graph, stream.raw)?;
                            lib.cuda_stream_synchronize(stream.raw)?;
                        }
                        let mut bytes = [0; 16];
                        lib.copy_d2h(&mut bytes, output.buffer)?;
                        let actual = bytes
                            .chunks_exact(4)
                            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                            .collect::<Vec<_>>();
                        assert_eq!(
                            actual,
                            [expected, expected, f32::NEG_INFINITY, f32::NEG_INFINITY]
                        );
                    }
                    Ok(())
                })();
                unsafe {
                    lib.cuda_graph_exec_destroy(graph)?;
                }
                checked
            })?;
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT, and two CUDA GPUs"]
    fn placed_index_queries_and_retained_decoder_candidates_match_direct() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let placement = CachePlacement::new(std::array::from_fn(|layer| usize::from(layer >= 14)))?;
        let budgets = IndexLaneWeights::distributed_device_bytes(&lib, &catalog, placement)?;
        assert_eq!(
            budgets.iter().sum::<usize>(),
            IndexLaneWeights::device_bytes(&lib, &catalog)?
        );
        assert!(
            IndexLaneWeights::load_distributed(
                &lib,
                &catalog,
                placement,
                [budgets[0] - 1, budgets[1]],
                1024 * 1024
            )
            .is_err()
        );
        let weights =
            IndexLaneWeights::load_distributed(&lib, &catalog, placement, budgets, 1024 * 1024)?;
        for (weight, layer) in weights.weights.iter().zip(LAYERS) {
            assert_eq!(weight.device.id, placement.attention(layer)? as i32);
        }
        let bytes = IndexLane::workspace_bytes(&lib, 16)?.iter().sum::<usize>();
        assert!(IndexLane::new(&weights, 16, bytes).is_err());
        let mut lanes = [
            IndexLane::new_on_device(&weights, 16, bytes, 0)?,
            IndexLane::new_on_device(&weights, 16, bytes, 1)?,
        ];
        assert!(lanes[0].reindex.is_none());
        assert!(lanes[1].reindex.is_some());
        let placed_bytes = [0, 1]
            .map(|gpu| {
                IndexLane::placed_workspace_bytes(&lib, placement, 16, gpu)
                    .map(|b| b.iter().sum::<usize>())
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        assert_eq!([lanes[0].next, lanes[1].next], [0, 2]);
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
        let references = devices
            .iter()
            .map(|device| {
                device.own(|| {
                    IndexLaneWeights::load(
                        &lib,
                        &catalog,
                        IndexLaneWeights::device_bytes(&lib, &catalog)?,
                        1024 * 1024,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut reference_lanes = references
            .iter()
            .map(|weights| weights.device.own(|| IndexLane::new(weights, 16, bytes)))
            .collect::<Result<Vec<_>>>()?;
        let producer_weights = CacheProducerWeights::load_distributed(
            &lib,
            &catalog,
            placement,
            CacheProducerWeights::distributed_device_bytes(&lib, &catalog, placement)?,
            1024 * 1024,
        )?;
        let mut producers = PlacedProducerWaves::new(
            &producer_weights,
            16,
            PlacedProducerWaves::device_bytes(&lib, placement, 16)?,
        )?;
        let pages = [2, 2, 2, 4];
        let mut bank = BackboneCache::new_distributed(
            &lib,
            placement,
            1,
            pages,
            BackboneCache::distributed_device_bytes(placement, 1, pages)?,
        )?;
        let lease = bank.begin_request(0, 1)?;
        let batch = bank.plan(&[CacheWork {
            lease,
            tokens: 16,
            kind: ExpertV2SourceKind::Prefill,
        }])?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // Reuse every index workspace and graph with changed hidden rows.
        for seed in [0, 9] {
            for gpu in 0..2 {
                devices[gpu].run(|| {
                    lanes[gpu].restart()?;
                    reference_lanes[gpu].restart()?;
                    Ok(())
                })?;
            }
            for (index, layer) in LAYERS.into_iter().enumerate() {
                let gpu = placement.attention(layer)?;
                let device = devices[gpu];
                let qw = device.own(|| {
                    AttentionQueryWeights::load(
                        &lib,
                        &catalog,
                        layer,
                        AttentionQueryWeights::device_bytes(&lib, &catalog, layer)?,
                        1024 * 1024,
                    )
                })?;
                let mut query =
                    device.own(|| qw.wave(16, AttentionQueryWave::device_bytes(&lib, 16)?))?;
                device.run(|| {
                    let host: Vec<u8> = (0..16 * 5120)
                        .flat_map(|i| {
                            let value = ((i + seed) % 31) as f32 / 32.0 - 0.5;
                            ((value.to_bits() >> 16) as u16).to_ne_bytes()
                        })
                        .collect();
                    lib.copy_h2d(query.input(), &host)?;
                    unsafe {
                        query.execute_tokens(&(0..16).collect::<Vec<_>>())?;
                    }
                    Ok(())
                })?;
                let output = query.output()?;
                // Cancellation owns index consumers and cache producers together.
                if index == if gpu == 0 { 0 } else { 2 } {
                    drop(unsafe {
                        producers.enqueue_production_and_index(
                            &bank,
                            &batch,
                            &output,
                            &mut lanes[gpu],
                        )?
                    });
                    device.run(|| lanes[gpu].restart())?;
                }
                let mut pending = unsafe {
                    producers.enqueue_production_and_index(
                        &bank,
                        &batch,
                        &output,
                        &mut lanes[gpu],
                    )?
                };
                runtime.block_on(async {
                    while !unsafe { pending.poll(&bank, &batch)? } {
                        assert_eq!(lib.cuda_get_device()?, 0);
                        tokio::task::yield_now().await;
                    }
                    Ok::<_, anyhow::Error>(())
                })?;
                drop(pending);
                let source = [2, 8, 14, 20].iter().rposition(|&l| l <= layer).unwrap();
                let cache = bank.attention(
                    &batch,
                    layer,
                    &producers.windows[layer],
                    Some(&producers.sources[source]),
                )?;
                assert_eq!(lib.cuda_get_device()?, 0);
                device.run(|| {
                    // A direct single-device lane can start at a source producer;
                    // subsequent decoder reindex operations must retain source 20.
                    if layer <= 20 {
                        reference_lanes[gpu].restart_at(index)?;
                    }
                    unsafe {
                        reference_lanes[gpu].select(&output, &cache)?;
                    }
                    let actual_query = lanes[gpu].query.output()?;
                    let expected_query = reference_lanes[gpu].query.output()?;
                    let actual = lanes[gpu].output(layer, &cache)?;
                    let expected = reference_lanes[gpu].output(layer, &cache)?;
                    for (a, b) in [
                        (actual_query.packed, expected_query.packed),
                        (actual_query.scales, expected_query.scales),
                        (actual_query.head_weights, expected_query.head_weights),
                        (actual.selected, expected.selected),
                    ] {
                        assert_eq!(a.device_id, gpu as i32);
                        let mut left = vec![0; a.bytes];
                        let mut right = vec![0; b.bytes];
                        lib.copy_d2h(&mut left, a)?;
                        lib.copy_d2h(&mut right, b)?;
                        assert_eq!(left, right, "placed index output differs at layer {layer}");
                    }
                    if layer >= 20 {
                        assert_eq!(lanes[gpu].source.output()?.selected.device_id, 1);
                    }
                    Ok(())
                })?;
            }
            assert_eq!([lanes[0].next, lanes[1].next], [8, 8]);
        }
        for gpu in 0..2 {
            devices[gpu].run(|| lanes[gpu].restart_decoder())?;
        }
        assert_eq!([lanes[0].next, lanes[1].next], [8, 3]);
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!(
            "placed index weight bytes={budgets:?}; per-device per-lane workspace={placed_bytes:?}"
        );
        Ok(())
    }
}
