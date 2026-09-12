//! Exercise physical addresses near the expanded limit with real CUDA consumers.
use super::*;
use crate::v41_memory::LoadStream;
use ds41rt_ffi::{V41Compressor, V41SparseSource, V41SparseWindow};

fn filled<'a>(lib: &'a NativeLibrary, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
    let allocation = DeviceAllocation::new(lib, bytes.len())?;
    lib.copy_h2d(allocation.buffer, bytes)?;
    Ok(allocation)
}
fn u64s(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[test]
#[ignore = "requires CUDA and 24 GiB free device memory"]
fn highest_physical_page_scatter_score_attention_and_fork() -> Result<()> {
    let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    let (free, total) = library.cuda_memory_info()?;
    ensure!(free > 24usize << 30, "high-page test requires 24 GiB free");
    eprintln!("high-page memory: free={free} total={total}");
    let stream = LoadStream {
        library: &library,
        raw: library.cuda_stream_create()?,
    };
    let mut cache = SourceCache::new(&library, 262144, 2)?;
    // Normal allocation starts at low pages. Place the highest page at the
    // next allocation position without fabricating initialized rows.
    let last = cache.pool.borrow().free.len() - 1;
    cache.pool.borrow_mut().free.swap(0, last);
    let plan = cache.reserve(&[(0, 0, 256)])?;
    let destinations: Vec<_> = (0..256)
        .map(|row| cache.destination(&plan, 0, row))
        .collect::<Result<_>>()?;
    assert_eq!(destinations[255], 67_108_863);
    let destination = filled(&library, &u64s(&destinations))?;
    let packed = filled(&library, &vec![0x22; 256 * 64])?; // E2M1 ones
    let scales = filled(&library, &vec![127; 256 * 4])?;
    let kv = filled(&library, &vec![0x22; 256 * KV_VALUES])?; // E2M1 ones
    let kv_scales = filled(&library, &vec![0x38; 256 * KV_SCALES])?;
    let workspace = DeviceAllocation::new(&library, V41Compressor::WORKSPACE_BYTES)?;
    let compressor = unsafe { library.v41_compressor(workspace.buffer)? };
    let encoding = library.v41_compressed_kv()?;
    unsafe {
        compressor.index_store(
            packed.buffer,
            scales.buffer,
            destination.buffer,
            cache.packed.buffer,
            cache.scales.buffer,
            256,
            cache.capacity,
            stream.raw,
        )?;
        encoding.store(
            kv.buffer,
            kv_scales.buffer,
            destination.buffer,
            cache.kv_values.buffer,
            cache.kv_scales.buffer,
            256,
            cache.capacity,
            stream.raw,
        )?;
        cache.upload(&plan, stream.raw)?;
        library.cuda_stream_synchronize(stream.raw)?;
    }
    cache.apply(plan);
    let view = cache.view(0, 256);
    assert_eq!(view.pages, &[262143]);
    for (allocation, width, expected) in [
        (&cache.packed, 64, 0x22),
        (&cache.scales, 4, 127),
        (&cache.kv_values, KV_VALUES, 0x22),
        (&cache.kv_scales, KV_SCALES, 0x38),
    ] {
        let mut actual = vec![0; width];
        library.copy_d2h(
            &mut actual,
            slice(allocation.buffer, 67_108_863 * width, width),
        )?;
        assert!(actual.iter().all(|&v| v == expected));
    }
    // Every index head has dot=128 and weight=1, yielding 32*128=4096.
    let query = filled(&library, &vec![0x22; 2048])?;
    let query_scales = filled(&library, &[127; 128])?;
    let weights = filled(&library, &[0x80u8, 0x3f].repeat(32))?; // BF16 ones
    let metadata = filled(&library, &u64s(&[0, 256]))?;
    let overlay_metadata = filled(&library, &u64s(&[0, 256, 256, 0, 0, 1]))?;
    let candidates = filled(&library, &u64s(&[255, 256, u64::MAX]))?;
    let scores = DeviceAllocation::new(&library, 12)?;
    let scoring = library.v41_index_scores()?;
    for overlay in [false, true] {
        unsafe {
            if overlay {
                scoring.execute_overlay(
                    query.buffer,
                    query_scales.buffer,
                    weights.buffer,
                    view.packed,
                    view.scales,
                    view.device_pages,
                    view.device_rows,
                    overlay_metadata.buffer,
                    candidates.buffer,
                    scores.buffer,
                    packed.buffer,
                    scales.buffer,
                    256,
                    1,
                    3,
                    1,
                    4096,
                    cache.capacity,
                    stream.raw,
                )?;
            } else {
                scoring.execute(
                    query.buffer,
                    query_scales.buffer,
                    weights.buffer,
                    view.packed,
                    view.scales,
                    view.device_pages,
                    view.device_rows,
                    metadata.buffer,
                    candidates.buffer,
                    scores.buffer,
                    1,
                    3,
                    1,
                    4096,
                    cache.capacity,
                    stream.raw,
                )?;
            }
            library.cuda_stream_synchronize(stream.raw)?;
        }
        let mut bytes = [0; 12];
        library.copy_d2h(&mut bytes, scores.buffer)?;
        let actual: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(actual, [4096.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
    }
    // One zero-valued window proposal, one high-page source value 1 and sink
    // weight 1, with zero queries, yield BF16(1/3) for every output component.
    let aq = filled(&library, &vec![0; 65536])?;
    let sink = filled(&library, &[0; 256])?;
    let meta = filled(&library, &u64s(&[0, 0, 1, 0, 0, 256, 256, 0, 0, 1]))?;
    let mut ids = vec![-1i32; 512];
    ids[0] = 255;
    let selected = filled(
        &library,
        &ids.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>(),
    )?;
    let zeros = filled(&library, &vec![0; 128 * 512])?;
    let zero_scales = filled(&library, &vec![127; 128 * 16])?;
    let end = filled(&library, &u64s(&[0]))?;
    let output = DeviceAllocation::new(&library, 65536)?;
    let window = V41SparseWindow {
        values: zeros.buffer,
        scales: zero_scales.buffer,
        proposals: zeros.buffer,
        proposal_scales: zero_scales.buffer,
        end: end.buffer,
        proposal_capacity: 128,
        replay_begins: None,
    };
    let source = V41SparseSource {
        values: cache.kv_values.buffer,
        scales: cache.kv_scales.buffer,
        proposals: kv.buffer,
        proposal_scales: kv_scales.buffer,
        pages: view.device_pages,
        end: view.device_rows,
        capacity: cache.capacity,
        proposal_capacity: 256,
        page_stride: 4096,
    };
    unsafe {
        library.v41_sparse_attention()?.launch(
            aq.buffer,
            sink.buffer,
            meta.buffer,
            Some(selected.buffer),
            &window,
            Some(&source),
            output.buffer,
            1,
            1,
            None,
            stream.raw,
        )?;
        library.cuda_stream_synchronize(stream.raw)?;
    }
    let mut actual = vec![0; 65536];
    library.copy_d2h(&mut actual, output.buffer)?;
    assert!(actual
        .chunks_exact(2)
        .all(|b| u16::from_ne_bytes(b.try_into().unwrap()) == 0x3eab));
    // Copy all four planes from a retained high partial page into a low page,
    // append privately, and verify the original highest row was not overwritten.
    let original = super::tests::read(&cache, 0, 256)?;
    let prefix = cache.retain_prefix(0, 255)?;
    cache.restore_prefix(1, &prefix)?;
    super::tests::append(&mut cache, 1, 255, 256, 0x11, stream.raw)?;
    assert_ne!(cache.pages[0][0], cache.pages[1][0]);
    assert_eq!(super::tests::read(&cache, 0, 256)?, original);
    let branch = super::tests::read(&cache, 1, 256)?;
    assert_eq!(&branch[..255 * SOURCE_ROW_BYTES], &original[..255 * SOURCE_ROW_BYTES]);
    assert!(branch[255 * SOURCE_ROW_BYTES..].iter().all(|&v| v == 0x11));
    cache.release(0)?;
    cache.release(1)?;
    drop(prefix);
    assert_eq!(cache.pool.borrow().free.len(), 262144);
    eprintln!("PASS highest row=67108863, KV byte offset=17179868928; scatter, both index scorers, attention, four-plane COW");
    Ok(())
}
