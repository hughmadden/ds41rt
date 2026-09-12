use super::*;

fn solid(value: u8) -> V41Image {
    V41Image::from_rgb(1, 1, &[value; 3]).unwrap()
}

#[test]
fn expanded_interleaved_images_have_exact_spans_and_context_accounting() {
    let a = solid(0);
    let b = solid(255);
    let n = a.grid().tokens();
    assert_eq!(n, 184);
    assert_ne!(a.identity(), b.identity());
    assert_eq!(a.identity(), solid(0).identity());
    let prompt = V41VisionPrompt::expand(
        &[12, V41_IMAGE_TOKEN_ID, 13, V41_IMAGE_TOKEN_ID, 14],
        vec![a, b],
        2 * n + 3,
    )
    .unwrap();
    assert_eq!(prompt.tokens.len(), 2 * n + 3);
    assert_eq!(
        prompt.images.iter().map(|s| s.start).collect::<Vec<_>>(),
        [1, n + 2]
    );
    assert_eq!(prompt.types[0], V41ImageTokenType::Text);
    for span in &prompt.images {
        let types = &prompt.types[span.start..span.start + n];
        assert_eq!(types.first(), Some(&V41ImageTokenType::Start));
        assert_eq!(types.last(), Some(&V41ImageTokenType::End));
        assert_eq!(
            types
                .iter()
                .filter(|&&t| t == V41ImageTokenType::Patch)
                .count(),
            169
        );
        assert_eq!(
            types
                .iter()
                .filter(|&&t| t == V41ImageTokenType::Newline)
                .count(),
            13
        );
        assert!(prompt.tokens[span.start..span.start + n]
            .iter()
            .all(|&t| t == V41_IMAGE_TOKEN_ID));
    }
    assert!(V41VisionPrompt::expand(&[V41_IMAGE_TOKEN_ID], vec![solid(0)], n - 1).is_err());
    assert!(V41VisionPrompt::expand(&[], vec![solid(0)], 1024).is_err());
    assert!(V41VisionPrompt::expand(&[V41_IMAGE_TOKEN_ID], vec![], 1024).is_err());
    let images = (0..16).map(solid).collect();
    assert_eq!(
        V41VisionPrompt::expand(&[V41_IMAGE_TOKEN_ID; 16], images, 32768)
            .unwrap()
            .images
            .len(),
        16
    );
    let images = (0..17).map(solid).collect();
    assert!(V41VisionPrompt::expand(&[V41_IMAGE_TOKEN_ID; 17], images, 32768).is_err());
}

#[test]
fn image_input_limits_reject_before_expansion() {
    assert!(V41ImageGrid::plan(0, 1).is_err());
    assert!(V41ImageGrid::plan(u32::MAX, u32::MAX).is_err());
    assert!(V41Image::decode(b"not an image").is_err());
    assert!(V41Image::decode(&[]).is_err());
    assert!(V41Image::from_rgb(2, 2, &[0; 3]).is_err());
}

#[test]
fn sixteen_bit_grayscale_is_clipped_before_rgb_resampling() {
    let gray =
        image::ImageBuffer::<image::Luma<u16>, _>::from_raw(3, 2, vec![0, 1, 254, 255, 256, 65535])
            .unwrap();
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma16(gray)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let expected: Vec<u8> = [0, 1, 254, 255, 255, 255]
        .into_iter()
        .flat_map(|v| [v; 3])
        .collect();
    assert_eq!(
        V41Image::decode(encoded.get_ref()).unwrap().patches(),
        V41Image::from_rgb(3, 2, &expected).unwrap().patches()
    );
}

#[test]
fn native_rgb_preparation_matches_pinned_reference_fixtures() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!("reference.json")).unwrap();
    for case in fixture["grids"].as_array().unwrap() {
        let w = case["width"].as_u64().unwrap() as u32;
        let h = case["height"].as_u64().unwrap() as u32;
        assert_eq!(
            serde_json::to_value(V41ImageGrid::plan(w, h).unwrap()).unwrap(),
            case["grid"],
            "{w}x{h}"
        );
    }
    for case in fixture["pixels"].as_array().unwrap() {
        let w = case["width"].as_u64().unwrap() as u32;
        let h = case["height"].as_u64().unwrap() as u32;
        let rgb: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    (0..3).map(move |c| ((x * 17 + y * 29 + c * 71 + (x * y) % 251) % 256) as u8)
                })
            })
            .collect();
        let image = V41Image::from_rgb(w, h, &rgb).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(image.patches())),
            case["patch_sha256"].as_str().unwrap(),
            "{w}x{h}"
        );
    }
}
