//! Separable Catmull-Rom convolution with the pinned Pillow RGB quantization
//! contract: 22-bit signed coefficients, byte rounding/clamping between axes,
//! and ties-to-even sizing/centering. See the vision preprocessing report.
use anyhow::{ensure, Result};

struct Weights {
    first: usize,
    values: Vec<i32>,
}
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    } else {
        0.0
    }
}
fn weights(input: usize, output: usize) -> Vec<Weights> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let inv = 1.0 / filter_scale;
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let first = ((center - 2.0 * filter_scale + 0.5) as isize).max(0) as usize;
            let end = ((center + 2.0 * filter_scale + 0.5) as usize).min(input);
            let values: Vec<f64> = (first..end)
                .map(|p| cubic((p as f64 - center + 0.5) * inv))
                .collect();
            let sum: f64 = values.iter().sum();
            Weights {
                first,
                values: values
                    .into_iter()
                    .map(|w| (w / sum * (1u32 << 22) as f64).round() as i32)
                    .collect(),
            }
        })
        .collect()
}
fn quantize(sum: i64) -> u8 {
    ((sum + (1 << 21)) >> 22).clamp(0, 255) as u8
}

pub(super) fn pad(
    rgb: &[u8],
    width: usize,
    height: usize,
    dst_w: usize,
    dst_h: usize,
) -> Result<Vec<u8>> {
    let ratio = width as f64 / height as f64;
    let destination_ratio = dst_w as f64 / dst_h as f64;
    let (w, h) = if ratio > destination_ratio {
        (
            dst_w,
            (height as f64 / width as f64 * dst_w as f64).round_ties_even() as usize,
        )
    } else if ratio < destination_ratio {
        (
            (width as f64 / height as f64 * dst_h as f64).round_ties_even() as usize,
            dst_h,
        )
    } else {
        (dst_w, dst_h)
    };
    ensure!(
        w > 0 && h > 0 && w <= dst_w && h <= dst_h,
        "image aspect ratio collapses under reference padding"
    );
    let horizontal = if w == width {
        rgb.to_vec()
    } else {
        let filters = weights(width, w);
        let mut out = vec![0; w * height * 3];
        for y in 0..height {
            for (x, filter) in filters.iter().enumerate() {
                for c in 0..3 {
                    let sum = filter
                        .values
                        .iter()
                        .enumerate()
                        .map(|(j, &k)| {
                            i64::from(k) * i64::from(rgb[(y * width + filter.first + j) * 3 + c])
                        })
                        .sum();
                    out[(y * w + x) * 3 + c] = quantize(sum);
                }
            }
        }
        out
    };
    let mut out = vec![127; dst_w * dst_h * 3];
    let x0 = ((dst_w - w) as f64 * 0.5).round_ties_even() as usize;
    let y0 = ((dst_h - h) as f64 * 0.5).round_ties_even() as usize;
    if h == height {
        for y in 0..h {
            out[((y + y0) * dst_w + x0) * 3..((y + y0) * dst_w + x0 + w) * 3]
                .copy_from_slice(&horizontal[y * w * 3..(y + 1) * w * 3]);
        }
    } else {
        let filters = weights(height, h);
        for (y, filter) in filters.iter().enumerate() {
            for x in 0..w {
                for c in 0..3 {
                    let sum = filter
                        .values
                        .iter()
                        .enumerate()
                        .map(|(j, &k)| {
                            i64::from(k)
                                * i64::from(horizontal[((filter.first + j) * w + x) * 3 + c])
                        })
                        .sum();
                    out[((y + y0) * dst_w + x0 + x) * 3 + c] = quantize(sum);
                }
            }
        }
    }
    Ok(out)
}
