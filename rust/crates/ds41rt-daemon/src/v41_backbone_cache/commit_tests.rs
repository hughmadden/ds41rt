use super::*;
use crate::v41_compressor::CompressorWeights;
use crate::v41_window::WindowWeights;
use ds41rt_ffi::Ds41rtDeviceBuffer;
fn read(lib: &NativeLibrary, buffer: Ds41rtDeviceBuffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.bytes];
    lib.copy_d2h(&mut bytes, buffer)?;
    Ok(bytes)
}
fn row(buffers: &[(&[u8], usize)], index: usize) -> Vec<u8> {
    buffers
        .iter()
        .flat_map(|(b, width)| b[index * width..(index + 1) * width].iter().copied())
        .collect()
}
fn produce(
    lib: &NativeLibrary,
    bank: &BackboneCache<'_>,
    batch: &CacheBatch,
    windows: &mut [WindowWave<'_, '_>],
    sources: &mut [CompressorWave<'_, '_>],
    seed: usize,
) -> Result<()> {
    let rows = batch.positions().len();
    let input = (0..rows * 5120)
        .flat_map(|i| {
            let x = (((i * 7 + seed * 31) % 127) as f32 - 63.0) / 128.0;
            ((x.to_bits() >> 16) as u16).to_ne_bytes()
        })
        .collect::<Vec<_>>();
    for (layer, wave) in windows.iter_mut().enumerate() {
        lib.copy_h2d(wave.input(), &input)?;
        unsafe {
            wave.execute(bank.window(batch, layer)?, &batch.window_chunks(layer)?)?;
        }
    }
    for (i, wave) in sources.iter_mut().enumerate() {
        lib.copy_h2d(wave.input(), &input)?;
        unsafe {
            wave.execute(
                bank.source(batch, SOURCES[i])?,
                &batch.source_chunks(SOURCES[i])?,
            )?;
        }
    }
    Ok(())
}
#[test]
fn real_all_cache_commits_preserve_prefixes_and_revoke_partial_failure() -> Result<()> {
    let Some(path) = std::env::var_os("DS41RT_CACHE_COMMIT_LIBRARY") else {
        eprintln!("skip cache commit GPU test: DS41RT_CACHE_COMMIT_LIBRARY unset");
        return Ok(());
    };
    let model = std::env::var_os("DS41RT_CACHE_COMMIT_MODEL")
        .context("DS41RT_CACHE_COMMIT_MODEL required")?;
    let lib = unsafe { NativeLibrary::load(path)? };
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        std::path::Path::new(&model),
    )?;
    let weights = (0..40)
        .map(|l| {
            WindowWeights::load(
                &lib,
                &catalog,
                l,
                WindowWeights::device_bytes(&lib, &catalog, l)?,
                1024 * 1024,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let compressed = SOURCES
        .into_iter()
        .map(|l| {
            CompressorWeights::load(
                &lib,
                &catalog,
                l,
                CompressorWeights::device_bytes(&catalog, l)?,
                1024 * 1024,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    // The AOT capacity is 4096; the live second batch contains 2064 rows.
    let capacity = 4096;
    let mut windows = weights
        .iter()
        .map(|w| w.wave(capacity, WindowWave::device_bytes(&lib, capacity)?))
        .collect::<Result<Vec<_>>>()?;
    let mut sources = compressed
        .iter()
        .zip(SOURCES)
        .map(|(w, l)| {
            w.wave(
                capacity as usize,
                CompressorWave::device_bytes(l, capacity as usize)?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut bank =
        BackboneCache::new(&lib, 16, [16; 4], BackboneCache::device_bytes(16, [16; 4])?)?;
    let leases = (0..16)
        .map(|s| bank.begin_request(s, 100 + s as u64))
        .collect::<Result<Vec<_>>>()?;
    let mut expected_windows = vec![vec![Vec::<Vec<u8>>::new(); 16]; 40];
    let mut expected_sources = vec![vec![Vec::<Vec<u8>>::new(); 16]; 4];
    let mut ends = [0u64; 16];
    for (cycle, tokens) in [9u32, 129, 3].into_iter().enumerate() {
        let work = leases
            .iter()
            .map(|&lease| CacheWork {
                lease,
                tokens,
                kind: ExpertV2SourceKind::Prefill,
            })
            .collect::<Vec<_>>();
        let batch = bank.plan(&work)?;
        let accepted = (0..16)
            .map(|i| match cycle {
                0 => (i % 10) as u32,
                1 => 129 - (i % 4) as u32,
                _ => 0,
            })
            .collect::<Vec<_>>();
        produce(&lib, &bank, &batch, &mut windows, &mut sources, cycle)?;
        for (layer, wave) in windows.iter().enumerate() {
            let output = wave.output(&bank.windows[layer])?;
            let values = read(&lib, output.values)?;
            let scales = read(&lib, output.scales)?;
            for (slot, &count) in accepted.iter().enumerate() {
                for i in 0..count as usize {
                    expected_windows[layer][slot].push(row(
                        &[(&values, 512), (&scales, 16)],
                        slot * tokens as usize + i,
                    ));
                }
            }
        }
        for (source, wave) in sources.iter().enumerate() {
            let output = wave.output(&bank.sources[source])?;
            let values = read(&lib, output.kv_values)?;
            let scales = read(&lib, output.kv_scales)?;
            let keys = read(&lib, output.index_packed)?;
            let key_scales = read(&lib, output.index_scales)?;
            let ratio = if source == 3 { 1 } else { 2 };
            for latent in output.completed {
                let slot = batch
                    .requests
                    .iter()
                    .position(|r| r.sources[source] == latent.lease)
                    .unwrap();
                if latent.position + ratio <= ends[slot] + u64::from(accepted[slot]) {
                    expected_sources[source][slot].push(row(
                        &[(&values, 512), (&scales, 16), (&keys, 64), (&key_scales, 4)],
                        latent.source_row as usize,
                    ));
                }
            }
        }
        bank.commit(&batch, &mut windows, &mut sources, &accepted)?;
        assert!(bank.validate_batch(&batch).is_err());
        assert!(bank
            .commit(&batch, &mut windows, &mut sources, &accepted)
            .is_err());
        for slot in 0..16 {
            ends[slot] += u64::from(accepted[slot]);
            assert_eq!(bank.committed_end(leases[slot])?, ends[slot]);
        }
        for (layer, state) in bank.windows.iter().enumerate() {
            for slot in 0..16 {
                let view = state.view(batch.requests[slot].windows[layer])?;
                assert_eq!(
                    u64::from_ne_bytes(read(&lib, view.device_end)?.try_into().unwrap()),
                    ends[slot]
                );
                let values = read(&lib, view.values)?;
                let scales = read(&lib, view.scales)?;
                for position in ends[slot].saturating_sub(128)..ends[slot] {
                    assert_eq!(
                        row(&[(&values, 512), (&scales, 16)], position as usize % 128),
                        expected_windows[layer][slot][position as usize],
                        "window layer={layer} slot={slot} cycle={cycle}"
                    );
                }
            }
        }
        for (source, state) in bank.sources.iter().enumerate() {
            let kv = state.kv_cache(batch.requests[0].sources[source])?;
            let idx = state.index_cache(batch.requests[0].sources[source])?;
            let values = read(&lib, kv.values)?;
            let scales = read(&lib, kv.scales)?;
            let keys = read(&lib, idx.packed)?;
            let key_scales = read(&lib, idx.scales)?;
            for slot in 0..16 {
                let view = state.kv_cache(batch.requests[slot].sources[source])?;
                let idx = state.index_cache(batch.requests[slot].sources[source])?;
                assert_eq!(view.rows, expected_sources[source][slot].len());
                assert_eq!(view.pages, idx.pages);
                assert_eq!(view.rows, idx.rows);
                assert_eq!(
                    u64::from_ne_bytes(read(&lib, view.device_rows)?.try_into().unwrap()),
                    view.rows as u64
                );
                let device_pages = read(&lib, view.device_pages)?;
                for (i, &page) in view.pages.iter().enumerate() {
                    assert_eq!(
                        u32::from_ne_bytes(device_pages[i * 4..i * 4 + 4].try_into().unwrap()),
                        page
                    );
                }
                for logical in 0..view.rows {
                    let physical = view.pages[logical / 256] as usize * 256 + logical % 256;
                    assert_eq!(
                        row(
                            &[(&values, 512), (&scales, 16), (&keys, 64), (&key_scales, 4)],
                            physical
                        ),
                        expected_sources[source][slot][logical],
                        "source={source} slot={slot} cycle={cycle}"
                    );
                }
            }
        }
        eprintln!("PASS cache prefix cycle={cycle} proposed={tokens} all 44 owners, 16 requests, packed bytes and device histories");
    }
    bank.release(&leases)?;
    // Source 20 exhausts after all 40 windows and the first three sources write.
    let pages = [16, 16, 16, 1];
    let mut limited = BackboneCache::new(&lib, 16, pages, BackboneCache::device_bytes(16, pages)?)?;
    let leases = (0..16)
        .map(|s| limited.begin_request(s, 200 + s as u64))
        .collect::<Result<Vec<_>>>()?;
    let batch = limited.plan(
        &leases
            .iter()
            .map(|&lease| CacheWork {
                lease,
                tokens: 2,
                kind: ExpertV2SourceKind::Prefill,
            })
            .collect::<Vec<_>>(),
    )?;
    produce(&lib, &limited, &batch, &mut windows, &mut sources, 4)?;
    let error = limited
        .commit(&batch, &mut windows, &mut sources, &[2; 16])
        .unwrap_err();
    assert!(error.to_string().contains("index cache pool exhausted"));
    eprintln!("expected late source exhaustion: {error}");
    for lease in leases {
        assert!(limited.request_id(lease).is_err());
    }
    let mut recovered = Vec::new();
    for slot in 0..16 {
        let lease = limited.begin_request(slot, 300 + slot as u64)?;
        assert_eq!(limited.committed_end(lease)?, 0);
        recovered.push(lease);
    }
    let retry = limited.plan(&[CacheWork {
        lease: recovered[0],
        tokens: 2,
        kind: ExpertV2SourceKind::Prefill,
    }])?;
    produce(&lib, &limited, &retry, &mut windows, &mut sources, 5)?;
    limited.commit(&retry, &mut windows, &mut sources, &[2])?;
    assert_eq!(limited.committed_end(recovered[0])?, 2);
    for &lease in &recovered[1..] {
        assert_eq!(limited.committed_end(lease)?, 0);
    }
    limited.release(&recovered)?;
    eprintln!("PASS late source failure revoked all 16 requests and all 44 component leases; admission and reclaimed-page commit recovered");
    Ok(())
}
