use super::*;
use ds41rt_loader::V41_IMAGE_TOKEN_ID;

#[test]
#[ignore = "requires CUDA and official embedding weights"]
fn native_image_replacement_and_reuse() -> Result<()> {
    let lib = unsafe {
        NativeLibrary::load(std::env::var_os("DS41RT_VISION_LIBRARY").context("vision library")?)?
    };
    let model =
        std::path::PathBuf::from(std::env::var_os("DS41RT_VISION_MODEL").context("vision model")?);
    let catalog =
        ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &model)?;
    let names = vec!["embed.weight".to_owned()];
    let table = NativeRtxTensors::load(
        &lib,
        &catalog,
        &names,
        NativeRtxTensors::plan(&catalog, &names)?,
        16 << 20,
    )?;
    let mut wave =
        TargetEmbeddingWave::new(&lib, &table, 4096, TargetEmbeddingWave::device_bytes(4096)?)?;
    let peer = TargetEmbeddingWave::new(&lib, &table, 4096, TargetEmbeddingWave::device_bytes(4096)?)?;
    let stream = LoadStream { library: &lib, raw: lib.cuda_stream_create()? };
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let copy = |value: TargetEmbedding<'_>| -> Result<(Vec<u8>, Vec<u8>)> {
        let mut residual = vec![0; value.residual.bytes];
        let mut pre = vec![0; value.pre.bytes];
        lib.copy_d2h(&mut residual, value.residual)?;
        lib.copy_d2h(&mut pre, value.pre)?;
        Ok((residual, pre))
    };
    for rows in [1, 35, 129, 4096, 35] {
        let positions: Vec<u64> = (0..rows).map(|i| (i % 128) as u64).collect();
        let selected: Vec<usize> = (0..rows)
            .filter(|i| rows == 4096 || i % 7 == 0 || *i == rows - 1)
            .collect();
        let mut tokens = vec![42; rows];
        for &row in &selected {
            tokens[row] = V41_IMAGE_TOKEN_ID;
        }
        let (baseline, pre) = copy(wave.execute(&tokens, &positions)?)?;
        unsafe { wave.enqueue_into(&tokens, stream.raw, [peer.residual.buffer, peer.pre.buffer])?; }
        runtime.block_on(stream.wait())?;
        let mut residual = peer.residual.buffer; residual.bytes = rows * 40960;
        let mut incoming = peer.pre.buffer; incoming.bytes = rows * 16;
        let queued = copy(TargetEmbedding { residual, pre: incoming, token_ids: &tokens,
            positions: &positions, _owner: PhantomData })?;
        assert_eq!(queued, (baseline.clone(), pre.clone()));
        let features: Vec<Vec<u8>> = selected
            .iter()
            .map(|&row| {
                (0..5120)
                    .flat_map(|col| (0x3f00u16 + ((row * 17 + col) % 128) as u16).to_le_bytes())
                    .collect()
            })
            .collect();
        let bindings: Vec<_> = selected
            .iter()
            .copied()
            .zip(features.iter().map(Vec::as_slice))
            .collect();
        let (actual, image_pre) = copy(wave.execute_with_images(&tokens, &positions, &bindings)?)?;
        assert_eq!(pre, image_pre);
        for row in 0..rows {
            if let Ok(i) = selected.binary_search(&row) {
                for hc in 0..4 {
                    assert_eq!(
                        &actual[(row * 4 + hc) * 10240..(row * 4 + hc + 1) * 10240],
                        features[i]
                    );
                }
            } else {
                assert_eq!(
                    &actual[row * 40960..(row + 1) * 40960],
                    &baseline[row * 40960..(row + 1) * 40960]
                );
            }
        }
        assert_eq!(copy(wave.execute(&tokens, &positions)?)?, (baseline, pre));
        eprintln!("PASS image replacement {rows} rows / {} image rows, all HC lanes, text preservation, repeated positions and text reuse",selected.len());
    }
    let features = vec![0; 10240];
    for (tokens, indices) in [
        (vec![V41_IMAGE_TOKEN_ID; 2], vec![0, 0]),
        (vec![V41_IMAGE_TOKEN_ID; 2], vec![1, 0]),
        (vec![V41_IMAGE_TOKEN_ID; 2], vec![2]),
        (vec![42; 2], vec![0]),
    ] {
        let bindings: Vec<_> = indices
            .iter()
            .map(|&row| (row, features.as_slice()))
            .collect();
        assert!(wave
            .execute_with_images(&tokens, &[0, 1], &bindings)
            .is_err());
        assert!(wave.output().is_err());
    }
    assert!(wave
        .execute_with_images(&[V41_IMAGE_TOKEN_ID], &[0], &[(0, &features[..10239])])
        .is_err());
    assert!(wave.output().is_err());
    wave.execute(&[42], &[0])?;
    eprintln!("PASS image replacement invalid-index/type/extent rejection and recovery");
    Ok(())
}
