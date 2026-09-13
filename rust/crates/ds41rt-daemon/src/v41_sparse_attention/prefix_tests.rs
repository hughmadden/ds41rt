use super::*;

#[test]
#[ignore = "requires DS41RT_NATIVE_LIB and two CUDA GPUs"]
fn committed_prefix_survives_append_but_private_boundary_stays_exact() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    for gpu in 0..2 {
        crate::v41_memory::device::Device {
            library: &lib,
            id: gpu,
        }
        .run(|| {
            let kernel = lib.v41_sparse_attention()?;
            let allocate = |bytes, value| -> Result<DeviceAllocation<'_>> {
                let allocation = DeviceAllocation::new(&lib, bytes)?;
                lib.copy_h2d(allocation.buffer, &vec![value; bytes])?;
                Ok(allocation)
            };
            let query = allocate(65536, 0)?;
            let sink = allocate(256, 0)?;
            let metadata = allocate(80, 0)?;
            let selected = allocate(2048, 255)?;
            let mut ids = vec![-1i32; 512];
            ids[0] = 0;
            lib.copy_h2d(
                selected.buffer,
                &ids.iter().flat_map(|n| n.to_ne_bytes()).collect::<Vec<_>>(),
            )?;
            let window_values = allocate(128 * 512, 0x38)?;
            let window_scales = allocate(128 * 16, 0x38)?;
            let window_end = allocate(8, 0)?;
            let source_values = allocate(256 * V41Kv::COMPRESSED_VALUE_BYTES, 0x22)?;
            let source_scales = allocate(256 * V41Kv::COMPRESSED_SCALE_BYTES, 0x38)?;
            let source_end = allocate(8, 0)?;
            let pages = allocate(4, 0)?;
            let output = allocate(65536, 0)?;
            let scratch = allocate(V41SparseAttention::split_scratch_bytes(1, 10)?, 0)?;
            let window = V41SparseWindow {
                values: window_values.buffer,
                scales: window_scales.buffer,
                proposals: window_values.buffer,
                proposal_scales: window_scales.buffer,
                end: window_end.buffer,
                proposal_capacity: 128,
                replay_begins: None,
            };
            let source = V41SparseSource {
                values: source_values.buffer,
                scales: source_scales.buffer,
                proposals: source_values.buffer,
                proposal_scales: source_scales.buffer,
                pages: pages.buffer,
                end: source_end.buffer,
                capacity: 256,
                proposal_capacity: 1,
                page_stride: 1,
            };
            let stream = LoadStream {
                library: &lib,
                raw: unsafe { lib.cuda_stream_create()? },
            };
            for split in [None, Some((scratch.buffer, 10))] {
                let run = |end: u64, private: u64| -> Result<Vec<u8>> {
                    let values = [0u64, 0, 1, 0, 0, 1, 1, private, 0, 1];
                    lib.copy_h2d(
                        metadata.buffer,
                        &values
                            .iter()
                            .flat_map(|n| n.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    lib.copy_h2d(source_end.buffer, &end.to_ne_bytes())?;
                    unsafe {
                        kernel.launch(
                            query.buffer,
                            sink.buffer,
                            metadata.buffer,
                            Some(selected.buffer),
                            &window,
                            Some(&source),
                            output.buffer,
                            1,
                            0,
                            split,
                            stream.raw,
                        )?;
                        lib.cuda_stream_synchronize(stream.raw)?;
                    }
                    let mut bytes = vec![0; output.buffer.bytes];
                    lib.copy_d2h(&mut bytes, output.buffer)?;
                    Ok(bytes)
                };
                let prefix = run(1, 0)?;
                assert!(
                    prefix.iter().any(|&byte| byte != 0),
                    "valid prefix produced no attention"
                );
                assert_eq!(
                    run(2, 0)?,
                    prefix,
                    "append invalidated a committed causal prefix"
                );
                assert!(
                    run(0, 0)?.iter().all(|&byte| byte == 0),
                    "shortened backing accepted"
                );
                assert!(
                    run(1, 1)?.iter().any(|&byte| byte != 0),
                    "valid private boundary rejected"
                );
                assert!(
                    run(2, 1)?.iter().all(|&byte| byte == 0),
                    "stale private boundary accepted"
                );
            }
            Ok(())
        })?;
    }
    Ok(())
}
