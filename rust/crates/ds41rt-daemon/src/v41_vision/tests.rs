use super::*;
use anyhow::Context;
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[test]
fn scratch_is_bounded_by_patch_capacity() -> Result<()> {
    let sizes = VisionRuntime::buffer_sizes(9216)?;
    assert_eq!(sizes.len(), 18);
    assert!(sizes.iter().sum::<usize>() < 768 << 20);
    assert_eq!(sizes[Slot::Scores as usize], 9216 * 16 * 128 * 4);
    assert!(VisionRuntime::buffer_sizes(0).is_err());
    assert!(VisionRuntime::buffer_sizes(9217).is_err());
    Ok(())
}

#[test]
#[ignore = "requires RTX and the official checkpoint; writes numerical qualification vectors"]
fn native_full_vision_vectors() -> Result<()> {
    let library =
        std::env::var_os("DS41RT_VISION_LIBRARY").context("DS41RT_VISION_LIBRARY required")?;
    let model = PathBuf::from(
        std::env::var_os("DS41RT_VISION_MODEL").context("DS41RT_VISION_MODEL required")?,
    );
    let output = PathBuf::from(
        std::env::var_os("DS41RT_VISION_OUTPUT").context("DS41RT_VISION_OUTPUT required")?,
    );
    ensure!(
        !output.exists(),
        "preserve qualification evidence: output already exists"
    );
    std::fs::create_dir_all(&output)?;
    let lib = unsafe { NativeLibrary::load(library)? };
    let catalog =
        ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &model)?;
    let bytes = VisionRuntime::device_bytes(&catalog, 9216)?;
    assert!(VisionRuntime::new(&lib, &catalog, 9216, bytes - 1).is_err());
    let started = Instant::now();
    let mut runtime = VisionRuntime::new(&lib, &catalog, 9216, bytes)?;
    runtime.fp32_attention = std::env::var_os("DS41RT_VISION_FP32_ATTENTION").is_some();
    let load_seconds = started.elapsed().as_secs_f64();
    let mut cases = Vec::new();
    let grids = if std::env::var_os("DS41RT_VISION_SMALL").is_some() {
        vec![(5, 7)]
    } else {
        vec![(5, 7), (39, 39), (3, 3063), (5, 7)]
    };
    let mut inputs = Vec::new();
    for (height, width) in grids {
        let grid = V41ImageGrid {
            pixel_height: height * 14,
            pixel_width: width * 14,
            vit_height: height,
            vit_width: width,
            llm_height: height.div_ceil(3),
            llm_width: width.div_ceil(3),
        };
        // BF16 values in [-1,1], exactly reproducible without host float math.
        let patches: Vec<u8> = (0..height * width * 588)
            .flat_map(|i| {
                let value = ((i * 17 + i / 97 * 29) % 257) as f32 / 128.0 - 1.0;
                let bits = value.to_bits();
                let bf = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
                bf.to_le_bytes()
            })
            .collect();
        inputs.push((format!("synthetic-{height}x{width}"), grid, patches));
    }
    if let Some(path) = std::env::var_os("DS41RT_VISION_IMAGES") {
        let images: Vec<PathBuf> = serde_json::from_slice(&std::fs::read(path)?)?;
        // Decode through the native loader, including real RGB image normalization.
        for path in images {
            let image = V41Image::decode(&std::fs::read(&path)?)?;
            inputs.push((
                path.display().to_string(),
                image.grid(),
                image.patches().to_vec(),
            ));
        }
    }
    for (case, (name, grid, patches)) in inputs.into_iter().enumerate() {
        let (height, width) = (grid.vit_height, grid.vit_width);
        let prefix = format!("case-{case}");
        std::fs::write(output.join(format!("{prefix}-patches.bin")), &patches)?;
        let mut traces = Vec::new();
        let mut observe = |name: &str, buffer: Buffer| -> Result<()> {
            if height * width > 35
                && name.starts_with("block-")
                && !["block-0", "block-15", "block-31"].contains(&name)
            {
                return Ok(());
            }
            let filename = format!("{prefix}-{name}.bin");
            let mut bytes = vec![0; buffer.bytes];
            lib.copy_d2h(&mut bytes, buffer)?;
            std::fs::write(output.join(&filename), bytes)?;
            traces.push(json!({"name":name,"file":filename,"bytes":buffer.bytes}));
            Ok(())
        };
        let started = Instant::now();
        let result = runtime.encode_patches(&patches, grid, Some(&mut observe))?;
        let traced_seconds = started.elapsed().as_secs_f64();
        let mut span = vec![0; result.bytes];
        lib.copy_d2h(&mut span, result)?;
        std::fs::write(output.join(format!("{prefix}-span.bin")), &span)?;
        // Repeat without observation to establish stable output and ordinary time.
        let started = Instant::now();
        let again = runtime.encode_patches(&patches, grid, None)?;
        let encode_seconds = started.elapsed().as_secs_f64();
        let mut repeated = vec![0; again.bytes];
        lib.copy_d2h(&mut repeated, again)?;
        assert_eq!(repeated, span, "repeat changed vision output");
        let mut short = patches.clone();
        short.pop();
        assert!(runtime.encode_patches(&short, grid, None).is_err());
        assert!(runtime.ready.is_none());
        cases.push(
            json!({"prefix":prefix,"name":name,"grid":grid,"traces":traces,"traced_seconds":traced_seconds,
            "encode_seconds":encode_seconds,"stable_repeat":true,"short_input_rejected":true}),
        );
        std::fs::write(
            output.join("cases.json"),
            serde_json::to_vec_pretty(&json!({"device_bytes":bytes,
            "weight_bytes":runtime.weights.resident_bytes(),"fp32_attention":runtime.fp32_attention,"load_seconds":load_seconds,"cases":cases}))?,
        )?;
        eprintln!(
            "PASS native vision {height}x{width}, {} span tokens, {:.4}s warm encode",
            grid.tokens(),
            encode_seconds
        );
    }
    Ok(())
}
