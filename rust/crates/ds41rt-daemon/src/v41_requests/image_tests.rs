use super::*;
use crate::v41_tensors::NativeRtxTensors;
use ds41rt_loader::{V41Image, V41ImageSpan, V41_IMAGE_TOKEN_ID};

#[test]
#[ignore = "requires CUDA, official checkpoint and mapped Engram tables"]
fn native_request_image_masks_features_and_history() -> Result<()> {
    let lib = unsafe {
        NativeLibrary::load(std::env::var_os("DS41RT_VISION_LIBRARY").context("vision library")?)?
    };
    let model =
        std::path::PathBuf::from(std::env::var_os("DS41RT_VISION_MODEL").context("vision model")?);
    let catalog =
        ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &model)?;
    let map = ds41rt_loader::EngramTokenMap::from_file(&model.join("tokenizer.json"))?;
    let pipeline = unsafe { EngramPipeline::new(&catalog, map, 4096, 2, 1 << 30)? };
    let mut requests = Requests::new(
        &lib,
        pipeline,
        16,
        [16; 4],
        BackboneCache::device_bytes(16, [16; 4])?,
    )?;
    let names = vec!["embed.weight".to_string()];
    let table = NativeRtxTensors::load(
        &lib,
        &catalog,
        &names,
        NativeRtxTensors::plan(&catalog, &names)?,
        16 << 20,
    )?;
    let mut embedding =
        TargetEmbeddingWave::new(&lib, &table, 4096, TargetEmbeddingWave::device_bytes(4096)?)?;
    let map = ds41rt_loader::EngramTokenMap::from_file(&model.join("tokenizer.json"))?;
    let mut all_tokens = Vec::new();
    let mut expected_masks = Vec::new();
    let mut expected_features = Vec::new();
    for slot in 0..16 {
        let start = slot % 3 + 1;
        let image = V41ImageSpan {
            start,
            image: V41Image::from_rgb(1, 1, &[slot as u8; 3])?,
        };
        assert_eq!(image.image.grid().tokens(), 184);
        let lease = requests.admit(slot, 100 + slot as u64)?;
        requests.attach_images(lease, RequestImages::new(&[image])?)?;
        let mut tokens = vec![42; 190];
        tokens[start..start + 184].fill(V41_IMAGE_TOKEN_ID);
        let mut mask = vec![false; 190];
        mask[start..start + 184].fill(true);
        let features: Vec<u8> = (0..184 * 5120)
            .flat_map(|i| (0x3f00u16 + ((slot * 17 + i) % 128) as u16).to_le_bytes())
            .collect();
        all_tokens.push(tokens);
        expected_masks.push(mask);
        expected_features.push(features);
    }
    assert_eq!(requests.image_requests, 16);
    let leases = requests
        .slots
        .iter()
        .flatten()
        .map(|r| r.lease)
        .collect::<Vec<_>>();
    let input = leases
        .iter()
        .zip(&all_tokens)
        .map(|(&lease, tokens)| RequestTokens {
            lease,
            tokens,
            image_mask: None,
            kind: ExpertV2SourceKind::Prefill,
        })
        .collect::<Vec<_>>();
    let mut batch = requests.prepare(&input)?;
    assert_eq!(
        batch.image_mask,
        expected_masks
            .iter()
            .flatten()
            .copied()
            .map(u8::from)
            .collect::<Vec<_>>()
    );
    assert!(requests.embedding_features(&batch).is_err());
    for (i, &lease) in leases.iter().enumerate() {
        requests.install_image_features(lease, 0, expected_features[i].clone())?;
    }
    let reversed_input = (0..16)
        .rev()
        .map(|i| RequestTokens {
            lease: leases[i],
            tokens: &all_tokens[i],
            image_mask: None,
            kind: ExpertV2SourceKind::Prefill,
        })
        .collect::<Vec<_>>();
    let mut reversed = requests.prepare(&reversed_input)?;
    assert_eq!(
        reversed.image_mask,
        expected_masks
            .iter()
            .rev()
            .flatten()
            .copied()
            .map(u8::from)
            .collect::<Vec<_>>()
    );
    for (row, data) in requests.embedding_features(&reversed)? {
        let slot = 15 - row / 190;
        let source_row = row % 190 - (slot % 3 + 1);
        assert_eq!(
            data,
            &expected_features[slot][source_row * 10240..(source_row + 1) * 10240]
        );
    }
    reversed.cancel();
    let positions = batch.cache.positions();
    let text = embedding.execute(&batch.tokens, &positions)?;
    let mut baseline = vec![0; text.residual.bytes];
    lib.copy_d2h(&mut baseline, text.residual)?;
    let mut baseline_pre = vec![0; text.pre.bytes];
    lib.copy_d2h(&mut baseline_pre, text.pre)?;
    let rows = requests.embedding_features(&batch)?;
    assert_eq!(rows.len(), 16 * 184);
    let image = embedding.execute_with_images(&batch.tokens, &positions, &rows)?;
    let mut actual = vec![0; image.residual.bytes];
    lib.copy_d2h(&mut actual, image.residual)?;
    let mut actual_pre = vec![0; image.pre.bytes];
    lib.copy_d2h(&mut actual_pre, image.pre)?;
    assert_eq!(baseline_pre, actual_pre);
    for slot in 0..16 {
        let start = slot % 3 + 1;
        for row in 0..190 {
            let flat = slot * 190 + row;
            if expected_masks[slot][row] {
                let data =
                    &expected_features[slot][(row - start) * 10240..(row - start + 1) * 10240];
                for hc in 0..4 {
                    assert_eq!(
                        &actual[(flat * 4 + hc) * 10240..(flat * 4 + hc + 1) * 10240],
                        data
                    );
                }
            } else {
                assert_eq!(
                    &actual[flat * 40960..(flat + 1) * 40960],
                    &baseline[flat * 40960..(flat + 1) * 40960]
                );
            }
        }
        for position in [
            start,
            start + 1,
            start + 2,
            start + 183,
            start + 184,
            start + 185,
            start + 186,
        ] {
            let mut full = requests.pipeline.new_history()?;
            let prefix = map.prepare_batch(
                &full,
                0,
                &all_tokens[slot][..position],
                Some(&expected_masks[slot][..position]),
                190,
            )?;
            full.commit(&prefix, position)?;
            let restored =
                requests.encoder_history_at(leases[slot], position, &all_tokens[slot])?;
            let expected = map.prepare_batch(
                &full,
                position as u64,
                &all_tokens[slot][position..],
                Some(&expected_masks[slot][position..]),
                190,
            )?;
            let resumed = map.prepare_batch(
                &restored,
                position as u64,
                &all_tokens[slot][position..],
                Some(&expected_masks[slot][position..]),
                190,
            )?;
            assert_eq!(expected.hashes(), resumed.hashes());
        }
    }
    eprintln!("PASS 16 request-owned image spans: masks, exact feature/HC/text rows, repeated positions and bounded Engram history at 112 image boundaries");
    let wrong = vec![false; 190];
    assert!(requests
        .prepare(&[RequestTokens {
            lease: leases[0],
            tokens: &all_tokens[0],
            image_mask: Some(&wrong),
            kind: ExpertV2SourceKind::Prefill
        }])
        .is_err());
    batch.cancel();
    for &lease in &leases {
        requests.begin_encoder(lease, 190)?;
    }
    let inputs = |start, end| {
        leases
            .iter()
            .enumerate()
            .map(|(i, &lease)| RequestTokens {
                lease,
                tokens: &all_tokens[i][start..end],
                image_mask: None,
                kind: ExpertV2SourceKind::Prefill,
            })
            .collect::<Vec<_>>()
    };
    let mut first = requests.reserve_encoder(&inputs(0, 128))?;
    let mut second = requests.reserve_encoder(&inputs(128, 190))?;
    assert_eq!(
        first.image_mask,
        expected_masks
            .iter()
            .flat_map(|m| m[..128].iter().copied().map(u8::from))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        second.image_mask,
        expected_masks
            .iter()
            .flat_map(|m| m[128..].iter().copied().map(u8::from))
            .collect::<Vec<_>>()
    );
    for (i, &lease) in leases.iter().enumerate() {
        let full = map.prepare_batch(
            &requests.request(lease)?.history,
            0,
            &all_tokens[i],
            Some(&expected_masks[i]),
            190,
        )?;
        assert_eq!(
            first.engram.as_ref().unwrap().batches()[i].hashes(),
            &full.hashes()[..128]
        );
        assert_eq!(
            second.engram.as_ref().unwrap().batches()[i].hashes(),
            &full.hashes()[128..]
        );
    }
    first.cancel();
    second.cancel();
    for &lease in &leases {
        requests.release(lease)?;
    }
    assert_eq!(requests.image_requests, 0);
    assert!(requests.embedding_features(&batch).is_err());
    for (slot, &old) in leases.iter().enumerate() {
        let lease = requests.admit(slot, 100 + slot as u64)?;
        assert_ne!(lease, old);
        assert!(requests.images(lease)?.is_empty());
        let mut batch = requests.prepare(&[RequestTokens {
            lease,
            tokens: &[42; 3],
            image_mask: None,
            kind: ExpertV2SourceKind::Prefill,
        }])?;
        assert_eq!(batch.image_mask, [0; 3]);
        batch.cancel();
        requests.release(lease)?;
    }
    eprintln!("PASS alternating encoder chunks preserve image masks and hashes; explicit-mask conflict rejected; cancellation and slot reuse clear image owners");
    Ok(())
}
