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
fn committed_bytes(lib: &NativeLibrary, bank: &BackboneCache<'_>, lease: CacheLease) -> Result<Vec<u8>> {
    let request = bank.request(lease)?;
    let mut result = Vec::new();
    for (state, &lease) in bank.windows.iter().zip(&request.windows) {
        let view = state.view(lease)?;
        let values = read(lib, view.values)?;
        let scales = read(lib, view.scales)?;
        for position in view.begin.max(view.end.saturating_sub(128))..view.end {
            result.extend(row(&[(&values, 512), (&scales, 16)], position as usize % 128));
        }
    }
    for (state, &lease) in bank.sources.iter().zip(&request.sources) {
        let view = state.kv_cache(lease)?;
        let index = state.index_cache(lease)?;
        let values = read(lib, view.values)?;
        let scales = read(lib, view.scales)?;
        let keys = read(lib, index.packed)?;
        let key_scales = read(lib, index.scales)?;
        for logical in 0..view.rows {
            let physical = view.pages[logical / 256] as usize * 256 + logical % 256;
            result.extend(row(&[(&values, ds41rt_ffi::V41Kv::COMPRESSED_VALUE_BYTES), (&scales, ds41rt_ffi::V41Kv::COMPRESSED_SCALE_BYTES), (&keys, 64), (&key_scales, 4)], physical));
        }
    }
    Ok(result)
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
    for layer in batch.stage.windows() {
        let wave = &mut windows[layer];
        lib.copy_h2d(wave.input(), &input)?;
        unsafe {
            wave.execute(bank.window(batch, layer)?, &batch.window_chunks(layer)?)?;
        }
    }
    for i in 0..batch.stage.source_count() {
        let wave = &mut sources[i];
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
fn qualify_encoder_prefix_replay(
    lib: &NativeLibrary,
    bank: &mut BackboneCache<'_>,
    windows: &mut [WindowWave<'_, '_>],
    sources: &mut [CompressorWave<'_, '_>],
) -> Result<()> {
    let original = bank.begin_request(0, 9100)?;
    let batch = bank.plan(&[CacheWork {
        lease: original,
        tokens: 385,
        kind: ExpertV2SourceKind::Prefill,
    }])?;
    produce(lib, bank, &batch, windows, sources, 81)?;
    bank.commit(&batch, windows, sources, &[385])?;
    let expected = committed_bytes(lib, bank, original)?;
    bank.queue_prefix(0, original, BackbonePrefix::device_bytes())?;
    while !bank.prefix_ready(0, original)? { std::thread::yield_now(); }
    let saved = bank.finish_prefix(0, original)?;
    bank.release(&[original])?;
    let resumed = (0..16)
        .map(|slot| bank.begin_request(slot, 9200 + slot as u64))
        .collect::<Result<Vec<_>>>()?;
    for &lease in &resumed {
        assert!(bank
            .restore_encoder_prefix(lease, &saved, 259, 300)
            .is_err());
        assert!(bank
            .restore_encoder_prefix(lease, &saved, 386, 400)
            .is_err());
        assert_eq!(bank.restore_encoder_prefix(lease, &saved, 258, 300)?, 130);
        assert_eq!(bank.committed_end(lease)?, 258);
        assert_eq!(bank.history_end(lease)?, 130);
        assert!(bank.begin_decoder_replay(lease).is_err());
    }
    let work = |tokens| {
        resumed
            .iter()
            .map(|&lease| CacheWork {
                lease,
                tokens,
                kind: ExpertV2SourceKind::Prefill,
            })
            .collect::<Vec<_>>()
    };
    assert!(bank.plan(&work(129)).is_err());
    let zero = bank.plan(&work(1))?;
    produce(lib, bank, &zero, windows, sources, 82)?;
    bank.commit(&zero, windows, sources, &[0; 16])?;
    assert!(bank.validate_batch(&zero).is_err());
    for rows in [63, 65] {
        let batch = bank.plan(&work(rows))?;
        assert_eq!(batch.stage(), CacheStage::EncoderReplay);
        assert!(batch.source_chunks(2).is_err());
        produce(lib, bank, &batch, windows, sources, 83)?;
        for layer in 0..20 {
            // Includes window-only layers 0/1 and all three encoder sources.
            bank.attention(&batch, layer, &windows[layer], None)?;
        }
        bank.commit(&batch, windows, sources, &[rows; 16])?;
        for &lease in &resumed {
            assert_eq!(bank.committed_end(lease)?, 258);
        }
    }
    for &lease in &resumed {
        assert_eq!(bank.history_end(lease)?, 258);
        assert_eq!(bank.stage(lease)?, CacheStage::Encoder);
        let request = bank.request(lease)?;
        for layer in 0..20 {
            let view = bank.windows[layer].view(request.windows[layer])?;
            assert_eq!((view.begin, view.end), (130, 258));
        }
        for layer in 20..40 {
            assert_eq!(bank.windows[layer].end(request.windows[layer])?, 0);
        }
    }
    assert!(bank.plan(&work(43)).is_err());
    let suffix = bank.plan(&work(42))?;
    produce(lib, bank, &suffix, windows, sources, 84)?;
    bank.commit(&suffix, windows, sources, &[42; 16])?;
    for &lease in &resumed {
        assert_eq!(bank.begin_decoder_replay(lease)?, 172);
    }
    let decoder = bank.plan_replay(&work(128))?;
    produce(lib, bank, &decoder, windows, sources, 85)?;
    bank.commit(&decoder, windows, sources, &[128; 16])?;
    for &lease in &resumed {
        assert_eq!(bank.committed_end(lease)?, 300);
        assert_eq!(bank.stage(lease)?, CacheStage::Full);
    }
    bank.release(&resumed)?;
    let original = bank.begin_request(0, 9300)?;
    bank.restore_prefix(original, &saved)?;
    assert_eq!(
        committed_bytes(lib, bank, original)?,
        expected,
        "encoder replay or divergent suffix changed retained future rows"
    );
    bank.release(&[original])?;
    eprintln!("PASS C16 partial encoder replay: shared complete-group sources, 128 bounded rows, zero acceptance, suffix append, decoder replay, immutable retained future");

    // Exact continuation preserves encoder rings and the odd compressor carry,
    // while avoiding all decoder writes until the final bounded replay.
    let full = bank.begin_request(0, 9400)?;
    let encoder = bank.begin_request(1, 9401)?;
    bank.restore_prefix(full, &saved)?;
    assert!(bank.restore_encoder_continuation(encoder, &saved, 512).is_err());
    assert!(bank.restore_encoder_continuation(encoder, &saved, 1048577).is_err());
    bank.restore_encoder_continuation(encoder, &saved, 514)?;
    assert_eq!(bank.history_end(encoder)?, 385);
    for layer in 20..40 {
        assert_eq!(bank.windows[layer].end(bank.request(encoder)?.windows[layer])?, 0);
    }
    let encoder_bytes = |bank: &BackboneCache<'_>, lease, full: bool| -> Result<Vec<u8>> {
        let mut bytes = committed_bytes(lib, bank, lease)?;
        if full { bytes.drain(20 * 128 * 528..40 * 128 * 528); }
        Ok(bytes)
    };
    assert_eq!(encoder_bytes(bank, encoder, false)?, encoder_bytes(bank, full, true)?);
    for rows in [128, 1] {
        for lease in [full, encoder] {
            let batch = bank.plan(&[CacheWork { lease, tokens: rows,
                kind: ExpertV2SourceKind::Prefill }])?;
            produce(lib, bank, &batch, windows, sources, 86 + rows as usize)?;
            bank.commit(&batch, windows, sources, &[rows])?;
        }
        assert_eq!(encoder_bytes(bank, encoder, false)?, encoder_bytes(bank, full, true)?,
            "encoder continuation differs from full continuation, including odd carry");
    }
    assert_eq!(bank.begin_decoder_replay(encoder)?, 386);
    let decoder = bank.plan_replay(&[CacheWork { lease: encoder, tokens: 128,
        kind: ExpertV2SourceKind::Prefill }])?;
    produce(lib, bank, &decoder, windows, sources, 90)?;
    bank.commit(&decoder, windows, sources, &[128])?;
    assert_eq!(bank.stage(encoder)?, CacheStage::Full);
    assert_eq!(bank.committed_end(encoder)?, 514);
    bank.release(&[full, encoder])?;
    let original = bank.begin_request(0, 9500)?;
    bank.restore_prefix(original, &saved)?;
    assert_eq!(committed_bytes(lib, bank, original)?, expected);
    bank.release(&[original])?;
    eprintln!("PASS exact encoder continuation: populated rings and odd carry match full continuation; decoder stays fresh until final 128 rows");
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
    // Two spare pages per source cover the private tails of both continuations
    // while their original source page remains retained by the prefix.
    let mut bank =
        BackboneCache::new(&lib, 16, [18; 4], BackboneCache::device_bytes(16, [18; 4])?)?;
    bank.install_prefix_pool(crate::v41_memory::SnapshotPool::new(
        &lib, BackbonePrefix::device_bytes(), 2)?)?;
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
                        &[(&values, ds41rt_ffi::V41Kv::COMPRESSED_VALUE_BYTES), (&scales, ds41rt_ffi::V41Kv::COMPRESSED_SCALE_BYTES), (&keys, 64), (&key_scales, 4)],
                        latent.source_row as usize,
                    ));
                }
            }
        }
        if cycle != 1 {
            unsafe { bank.enqueue_cache_commit(&batch, &mut windows, &mut sources, &accepted)?; }
            assert!(bank.release(&[leases[0]]).is_err());
            bank.validate_batch(&batch)?;
            while !windows.iter().map(|w| w.poll_commit()).chain(sources.iter().map(|s| s.poll_commit())).collect::<Result<Vec<_>>>()?
                .into_iter().all(|ready| ready) { std::thread::yield_now(); }
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
                            &[(&values, ds41rt_ffi::V41Kv::COMPRESSED_VALUE_BYTES), (&scales, ds41rt_ffi::V41Kv::COMPRESSED_SCALE_BYTES), (&keys, 64), (&key_scales, 4)],
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
    // Slot 1 ends at 129 tokens: an odd compressor group and a wrapped SWA.
    // Restore into a recycled slot, then compare the next real-weight commit.
    let expected = committed_bytes(&lib, &bank, leases[1])?;
    bank.queue_prefix(0, leases[1], BackbonePrefix::device_bytes())?;
    assert!(bank.release(&[leases[1]]).is_err());
    assert!(bank.retain_prefix(leases[1], BackbonePrefix::device_bytes()).is_err());
    assert!(bank.plan(&[CacheWork { lease: leases[1], tokens: 1, kind: ExpertV2SourceKind::Decode }]).is_err());
    assert!(bank.request_id(leases[1]).is_ok());
    bank.queue_prefix(1, leases[2], BackbonePrefix::device_bytes())?;
    assert!(bank.queue_prefix(1, leases[3], BackbonePrefix::device_bytes()).is_err());
    assert!(bank.prefix_ready(1, leases[1]).is_err());
    while !bank.prefix_ready(1, leases[2])? { std::thread::yield_now(); }
    drop(bank.finish_prefix(1, leases[2])?);
    // Aborting a second copy returns only its storage; lane zero remains reserved.
    bank.queue_prefix(1, leases[2], BackbonePrefix::device_bytes())?;
    bank.abort_prefix(1)?;
    assert!(bank.release(&[leases[1]]).is_err());
    let peer = bank.plan(&[CacheWork { lease: leases[2], tokens: 1, kind: ExpertV2SourceKind::Decode }])?;
    produce(&lib, &bank, &peer, &mut windows, &mut sources, 76)?;
    bank.commit(&peer, &mut windows, &mut sources, &[1])?;
    while !bank.prefix_ready(0, leases[1])? { std::thread::yield_now(); }
    let saved = bank.finish_prefix(0, leases[1])?;
    assert_eq!(saved.end(), 129);
    assert!(bank.retain_prefix(leases[1], BackbonePrefix::device_bytes() - 1).is_err());
    bank.release(&leases[..1])?;
    let restored = bank.begin_request(0, 900)?;
    let stale = bank.plan(&[CacheWork { lease: restored, tokens: 1, kind: ExpertV2SourceKind::Decode }])?;
    bank.restore_prefix(restored, &saved)?;
    assert!(bank.validate_batch(&stale).is_err());
    assert_eq!(committed_bytes(&lib, &bank, restored)?, expected);
    for lease in [leases[1], restored] {
        let batch = bank.plan(&[CacheWork { lease, tokens: 1, kind: ExpertV2SourceKind::Decode }])?;
        produce(&lib, &bank, &batch, &mut windows, &mut sources, 77)?;
        bank.commit(&batch, &mut windows, &mut sources, &[1])?;
    }
    assert_eq!(committed_bytes(&lib, &bank, restored)?, committed_bytes(&lib, &bank, leases[1])?);
    bank.release(&[restored])?;
    bank.release(&leases[1..])?;
    // Snapshot outlives every original admission and remains immutable.
    let restored = bank.begin_request(0, 901)?;
    bank.restore_prefix(restored, &saved)?;
    assert_eq!(committed_bytes(&lib, &bank, restored)?, expected);
    bank.release(&[restored])?;
    drop(saved);
    eprintln!("PASS retained full backbone prefix: all 44 owners, odd carry, slot reuse, exact next commit");
    qualify_encoder_prefix_replay(&lib, &mut bank, &mut windows, &mut sources)?;
    // CED publishes all global sources during encoder work, then only decoder
    // windows during replay. Reuse the same real-weight producers and bank.
    let ced = (0..16).map(|slot| bank.begin_request(slot, 1000 + slot as u64))
        .collect::<Result<Vec<_>>>()?;
    let stale = bank.plan(&[CacheWork { lease: ced[0], tokens: 1, kind: ExpertV2SourceKind::Prefill }])?;
    for &lease in &ced {
        bank.begin_encoder(lease, 129)?;
        assert!(bank.begin_encoder(lease, 129).is_err());
        assert!(bank.begin_decoder_replay(lease).is_err());
    }
    assert!(bank.validate_batch(&stale).is_err());
    // Keep one causal snapshot identity per request/source across appends. This
    // is the committed-view primitive needed by interleaved encoder chunks.
    let snapshots = (0..16).map(|_| (0..4)
        .map(|_| crate::v41_compressor::reserve_source_snapshot())
        .collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
    let mut prior_sources = Vec::new();
    for tokens in [64, 65] {
        let work = ced.iter().map(|&lease| CacheWork { lease, tokens, kind: ExpertV2SourceKind::Prefill }).collect::<Vec<_>>();
        assert!(bank.plan_replay(&work).is_err());
        let batch = bank.plan(&work)?;
        assert_eq!(batch.stage(), CacheStage::Encoder);
        assert!(bank.window(&batch, 20).is_err());
        produce(&lib, &bank, &batch, &mut windows, &mut sources, 10 + tokens as usize)?;
        let competing = bank.plan(&work)?;
        let old_end = bank.committed_end(ced[0])?;
        assert!(bank.publish_encoder_window(&batch, 20, &mut windows[20]).is_err());
        for layer in 0..20 {
            if tokens == 64 && layer % 2 != 0 { continue; }
            bank.publish_encoder_window(&batch, layer, &mut windows[layer])?;
            bank.validate_batch(&batch)?;
            assert!(bank.publish_encoder_window(&batch, layer, &mut windows[layer]).is_err());
            for &lease in &ced { assert_eq!(bank.committed_end(lease)?, old_end); }
        }
        assert!(bank.validate_batch(&competing).is_err());
        assert!(bank.plan(&work).is_err());
        for (i, &layer) in SOURCES.iter().enumerate() {
            bank.publish_encoder_source(&batch, layer, &mut sources[i])?;
            bank.validate_batch(&batch)?;
            assert!(bank.publish_encoder_source(&batch, layer, &mut sources[i]).is_err());
        }
        if tokens == 64 {
            let a = bank.attention(&batch, 3, &windows[3], None)?;
            let b = bank.attention(&batch, 5, &windows[5], None)?;
            assert!(bank.attention(&batch, 3, &windows[3], Some(&sources[0])).is_err());
            for (a, b) in a.sources.iter().zip(&b.sources) {
                assert_eq!(a.binding(), b.binding());
                assert_eq!(a.metadata(63)?, [0, 32, 32, 0, 0, 2]);
            }
        }
        assert!(bank.commit(&batch, &mut windows, &mut sources, &[0; 16]).is_err());
        bank.validate_batch(&batch)?;
        bank.commit(&batch, &mut windows, &mut sources, &vec![tokens; 16])?;
        assert!(bank.validate_batch(&batch).is_err());
        for (slot, &lease) in ced.iter().enumerate() {
            let r = bank.request(lease)?;
            for (i, &layer) in SOURCES.iter().enumerate() {
                let state = &bank.sources[i];
                let step = if layer == 20 { 1 } else { 2 };
                let view = state.committed_proposal(r.sources[i], 0..64, snapshots[slot][i])?;
                assert_eq!(view.source_layer, layer);
                assert_eq!(view.cache.rows as u64, r.end / step);
                for position in 0..64 {
                    assert_eq!(view.metadata(position)?,
                        [0, (position + 1) / step, r.end / step, 0, 0, step]);
                }
                if tokens == 64 { prior_sources.push(view.binding()); }
                else { assert_eq!(view.binding(), prior_sources[slot * 4 + i]); }
                let tail = state.committed_proposal(r.sources[i], r.end - 1..r.end, snapshots[slot][i])?;
                assert_eq!(tail.metadata(r.end - 1)?[1], r.end / step);
                assert!(tail.metadata(r.end).is_err());
                assert!(state.committed_proposal(r.sources[i], 0..r.end + 1, snapshots[slot][i]).is_err());
                assert!(state.committed_proposal(r.sources[i], 0..1, 0).is_err());
            }
        }
    }
    eprintln!("PASS committed encoder source views: all 16 requests/all 4 sources, causal ratios, odd carry, stable snapshots across append");
    let r = bank.request(ced[0])?;
    let preserved_windows = (0..20).map(|layer| {
        let view = bank.windows[layer].view(r.windows[layer])?;
        Ok((read(&lib, view.values)?, read(&lib, view.scales)?))
    }).collect::<Result<Vec<_>>>()?;
    let preserved_sources = (0..4).map(|i| {
        let view = bank.sources[i].kv_cache(r.sources[i])?;
        let idx = bank.sources[i].index_cache(r.sources[i])?;
        Ok((read(&lib, view.values)?, read(&lib, view.scales)?,
            read(&lib, idx.packed)?, read(&lib, idx.scales)?))
    }).collect::<Result<Vec<_>>>()?;
    for &lease in &ced { assert_eq!(bank.begin_decoder_replay(lease)?, 1); }
    let mut prior_binding = None;
    for tokens in [63, 65] {
        let work = ced.iter().map(|&lease| CacheWork { lease, tokens, kind: ExpertV2SourceKind::Prefill }).collect::<Vec<_>>();
        assert!(bank.plan(&work).is_err());
        let batch = bank.plan_replay(&work)?;
        assert_eq!(batch.stage(), CacheStage::Replay);
        assert!(bank.window(&batch, 19).is_err());
        produce(&lib, &bank, &batch, &mut windows, &mut sources, 100 + tokens as usize)?;
        let view = bank.attention(&batch, 20, &windows[20], None)?;
        let reused = bank.attention(&batch, 24, &windows[24], None)?;
        assert!(bank.attention(&batch, 20, &windows[20], Some(&sources[3])).is_err());
        assert_ne!(prior_binding, Some(view.sources[0].binding()));
        prior_binding = Some(view.sources[0].binding());
        for (i, source) in view.sources.iter().enumerate() {
            let start = batch.requests[i].position;
            assert_eq!(source.binding(), reused.sources[i].binding());
            assert_eq!(source.first_token(), start);
            assert_eq!(source.cache.rows, 129);
            for position in start..start + u64::from(tokens) {
                assert_eq!(source.metadata(position)?, [0, position + 1, 129, 0, 0, 1]);
            }
            assert!(source.metadata(start - 1).is_err());
            assert!(source.metadata(start + u64::from(tokens)).is_err());
            assert_eq!(view.windows[i].cache.begin, 1);
        }
        drop(reused); drop(view);
        bank.commit(&batch, &mut windows, &mut sources, &vec![tokens; 16])?;
        for &lease in &ced { assert_eq!(bank.committed_end(lease)?, 129); }
        assert!(bank.validate_batch(&batch).is_err());
    }
    let r = bank.request(ced[0])?;
    assert_eq!(r.phase, CachePhase::Full);
    for layer in 0..20 {
        let view = bank.windows[layer].view(r.windows[layer])?;
        assert_eq!((read(&lib, view.values)?, read(&lib, view.scales)?), preserved_windows[layer]);
    }
    for i in 0..4 {
        let view = bank.sources[i].kv_cache(r.sources[i])?;
        let idx = bank.sources[i].index_cache(r.sources[i])?;
        assert_eq!((read(&lib, view.values)?, read(&lib, view.scales)?,
            read(&lib, idx.packed)?, read(&lib, idx.scales)?), preserved_sources[i]);
    }
    let next = bank.plan(&[CacheWork { lease: ced[0], tokens: 1, kind: ExpertV2SourceKind::Decode }])?;
    assert_eq!(next.positions(), vec![129]);
    assert_eq!(next.stage(), CacheStage::Full);
    bank.release(&ced)?;
    eprintln!("PASS 16 CED transactions: encoder 64+65, replay 63+65, untouched global/encoder bytes, full decode readiness");
    // Reserve two chunks before any execution. Publish layer-major to model
    // later chunks reaching an early layer while earlier chunks await FFN.
    let queued = (0..16).map(|slot| bank.begin_request(slot, 4000 + slot as u64))
        .collect::<Result<Vec<_>>>()?;
    for &lease in &queued { bank.begin_encoder(lease, 129)?; }
    let work = |tokens| queued.iter().map(|&lease| CacheWork {
        lease, tokens, kind: ExpertV2SourceKind::Prefill,
    }).collect::<Vec<_>>();
    let first = bank.reserve_encoder(&work(64))?;
    let second = bank.reserve_encoder(&work(65))?;
    assert_eq!(first.requests[0].position, 0);
    assert_eq!(second.requests[0].position, 64);
    assert!(bank.reserve_encoder(&work(1)).is_err());
    assert!(bank.plan(&work(1)).is_err());
    assert!(bank.commit(&first, &mut windows, &mut sources, &[64; 16]).is_err());
    for layer in 0..20 {
        for batch in [&first, &second] {
            let input = vec![0u8; batch.positions().len() * 5120 * 2];
            lib.copy_h2d(windows[layer].input(), &input)?;
            unsafe { windows[layer].execute(bank.window(batch, layer)?, &batch.window_chunks(layer)?)?; }
            bank.publish_encoder_window(batch, layer, &mut windows[layer])?;
            for &lease in &queued { assert_eq!(bank.committed_end(lease)?, 0); }
        }
    }
    for (i, &layer) in SOURCES.iter().enumerate() {
        for batch in [&first, &second] {
            let input = vec![0u8; batch.positions().len() * 5120 * 2];
            lib.copy_h2d(sources[i].input(), &input)?;
            unsafe { sources[i].execute(bank.source(batch, layer)?, &batch.source_chunks(layer)?)?; }
            bank.publish_encoder_source(batch, layer, &mut sources[i])?;
        }
    }
    assert!(bank.commit(&second, &mut windows, &mut sources, &[65; 16]).is_err());
    bank.commit(&first, &mut windows, &mut sources, &[64; 16])?;
    bank.validate_batch(&second)?;
    assert!(bank.validate_batch(&first).is_err());
    for &lease in &queued { assert_eq!(bank.committed_end(lease)?, 64); }
    bank.commit(&second, &mut windows, &mut sources, &[65; 16])?;
    for &lease in &queued {
        assert_eq!(bank.committed_end(lease)?, 129);
        assert_eq!(bank.begin_decoder_replay(lease)?, 1);
    }
    bank.release(&queued)?;
    assert!(bank.validate_batch(&second).is_err());
    let queued = (0..16).map(|slot| bank.begin_request(slot, 5000 + slot as u64))
        .collect::<Result<Vec<_>>>()?;
    for &lease in &queued { bank.begin_encoder(lease, 32)?; }
    let work = queued.iter().map(|&lease| CacheWork { lease, tokens: 1,
        kind: ExpertV2SourceKind::Prefill }).collect::<Vec<_>>();
    for position in 0..16 {
        let batch = bank.reserve_encoder(&work)?;
        assert_eq!(batch.requests[0].position, position);
    }
    assert!(bank.reserve_encoder(&work).is_err());
    for &lease in &queued { assert_eq!(bank.request(lease)?.publication.len(), 16); }
    bank.release(&queued)?;
    eprintln!("PASS 16-request encoder reservations: layer-major 64+65, ordered completion, generation survival, capacity and release");
    // Source 20 exhausts after all 40 windows and the first three sources write.
    for early in [false, true] {
    let pages = [16, 16, 16, 1];
    let mut limited = BackboneCache::new(&lib, 16, pages, BackboneCache::device_bytes(16, pages)?)?;
    let leases = (0..16)
        .map(|s| limited.begin_request(s, 200 + s as u64))
        .collect::<Result<Vec<_>>>()?;
    if early { for &lease in &leases { limited.begin_encoder(lease, 2)?; } }
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
    let error = if early {
        for layer in 0..20 {
            limited.publish_encoder_window(&batch, layer, &mut windows[layer])?;
        }
        for i in 0..3 {
            limited.publish_encoder_source(&batch, SOURCES[i], &mut sources[i])?;
        }
        limited.publish_encoder_source(&batch, 20, &mut sources[3]).unwrap_err()
    } else {
        limited.commit(&batch, &mut windows, &mut sources, &[2; 16]).unwrap_err()
    };
    assert!(error.downcast_ref::<crate::v41_compressor::SourcePoolExhausted>().is_some());
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
    eprintln!("PASS late source failure early={early} revoked all 16 requests and all 44 component leases; admission and reclaimed-page commit recovered");
    }
    eprintln!("PASS early encoder publication: mixed/all windows, all sources, deferred logical end, stale/duplicate/partial guards");
    Ok(())
}
