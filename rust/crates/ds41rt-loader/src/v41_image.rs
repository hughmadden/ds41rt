//! Official V4.1 image preparation and expanded prompt spans, without CUDA or Python.
use anyhow::{ensure, Context, Result};
use image::{ImageReader, Limits};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::Cursor;

mod resize;
pub const V41_IMAGE_TOKEN_ID: u32 = 129264;
pub const V41_MAX_IMAGES: usize = 16;
const PATCH: usize = 14;
const DOWNSAMPLE: usize = 3;
const MAX_IMAGE_TOKENS: usize = 1024;
const MIN_PIXELS: usize = 544 * 544;
const MAX_PIXELS: u64 = 64 * 1024 * 1024;
const MAX_ENCODED_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct V41ImageGrid {
    pub pixel_height: usize,
    pub pixel_width: usize,
    pub vit_height: usize,
    pub vit_width: usize,
    pub llm_height: usize,
    pub llm_width: usize,
}
impl V41ImageGrid {
    /// Pinned model policy: 14px patches, 3x3 aligner, >=544² pixels before
    /// rounding, then aspect-preserving shrink to <=1024 complete span tokens.
    pub fn plan(width: u32, height: u32) -> Result<Self> {
        ensure!(
            width > 0 && height > 0 && u64::from(width) * u64::from(height) <= MAX_PIXELS,
            "image dimensions must be positive and at most 64 megapixels"
        );
        let (mut w, mut h) = (width as f64, height as f64);
        if w * h < MIN_PIXELS as f64 {
            let ratio = (MIN_PIXELS as f64 / (w * h)).sqrt();
            w = (w * ratio).trunc();
            h = (h * ratio).trunc();
        }
        let mut ph = (h / PATCH as f64).ceil() as usize * PATCH;
        let mut pw = (w / PATCH as f64).ceil() as usize * PATCH;
        let count = |h: usize, w: usize| {
            (h / PATCH).div_ceil(DOWNSAMPLE) * ((w / PATCH).div_ceil(DOWNSAMPLE) + 1) + 2
        };
        if count(ph, pw) > MAX_IMAGE_TOKENS {
            let ratio = h / w;
            let max_w = ((MAX_IMAGE_TOKENS - 2) as f64 / ratio + 0.25).sqrt() - 0.5;
            let max_h = max_w * ratio;
            let cell = PATCH * DOWNSAMPLE;
            if max_w < 1.0 {
                ph = (MAX_IMAGE_TOKENS - 2) / 2 * cell;
                pw = cell;
            } else if max_h < 1.0 {
                ph = cell;
                pw = (MAX_IMAGE_TOKENS - 3) * cell;
            } else {
                let beta = (max_w.floor() * cell as f64 / w).min(max_h.floor() * cell as f64 / h);
                ph = (h * beta / PATCH as f64).floor() as usize * PATCH;
                pw = (w * beta / PATCH as f64).floor() as usize * PATCH;
            }
        }
        ensure!(
            ph > 0 && pw > 0 && count(ph, pw) <= MAX_IMAGE_TOKENS,
            "invalid image resize grid"
        );
        Ok(Self {
            pixel_height: ph,
            pixel_width: pw,
            vit_height: ph / PATCH,
            vit_width: pw / PATCH,
            llm_height: (ph / PATCH).div_ceil(DOWNSAMPLE),
            llm_width: (pw / PATCH).div_ceil(DOWNSAMPLE),
        })
    }
    pub fn tokens(self) -> usize {
        self.llm_height * (self.llm_width + 1) + 2
    }
    pub fn token_types(self) -> Vec<V41ImageTokenType> {
        let mut types = Vec::with_capacity(self.tokens());
        types.push(V41ImageTokenType::Start);
        for _ in 0..self.llm_height {
            types.extend(std::iter::repeat_n(
                V41ImageTokenType::Patch,
                self.llm_width,
            ));
            types.push(V41ImageTokenType::Newline);
        }
        types.push(V41ImageTokenType::End);
        types
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(i8)]
pub enum V41ImageTokenType {
    Text = -1,
    Start = 0,
    Patch = 1,
    Newline = 2,
    End = 3,
}

pub struct V41Image {
    grid: V41ImageGrid,
    /// Contiguous little-endian BF16 [vit_h * vit_w, 3, 14, 14].
    patches: Vec<u8>,
    identity: [u8; 32],
}
impl V41Image {
    pub fn decode(encoded: &[u8]) -> Result<Self> {
        ensure!(
            !encoded.is_empty() && encoded.len() <= MAX_ENCODED_BYTES,
            "encoded image must be 1 byte through 32 MiB"
        );
        if encoded.starts_with(&[0xff, 0xd8]) {
            let mut decoder = turbojpeg::Decompressor::new()?;
            let header = decoder.read_header(encoded)?;
            let width = u32::try_from(header.width)?;
            let height = u32::try_from(header.height)?;
            let grid = V41ImageGrid::plan(width, height)?;
            let cmyk = matches!(
                header.colorspace,
                turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
            );
            let channels = if cmyk { 4 } else { 3 };
            let mut pixels = vec![0; header.width * header.height * channels];
            decoder.set_fast_upsample(false)?;
            decoder.decompress(
                encoded,
                turbojpeg::Image {
                    pixels: pixels.as_mut_slice(),
                    width: header.width,
                    height: header.height,
                    pitch: header.width * channels,
                    format: if cmyk {
                        turbojpeg::PixelFormat::CMYK
                    } else {
                        turbojpeg::PixelFormat::RGB
                    },
                },
            )?;
            // Pillow treats JPEG CMYK samples as inverted, then converts to RGB.
            let rgb = if cmyk {
                pixels
                    .chunks_exact(4)
                    .flat_map(|p| {
                        (0..3).map(move |i| ((u16::from(p[i]) * u16::from(p[3]) + 127) / 255) as u8)
                    })
                    .collect()
            } else {
                pixels
            };
            return Self::from_rgb_with_grid(width, height, &rgb, grid);
        }
        let mut reader = ImageReader::new(Cursor::new(encoded)).with_guessed_format()?;
        let mut limits = Limits::default();
        limits.max_alloc = Some(256 * 1024 * 1024);
        reader.limits(limits);
        let decoder = reader
            .into_decoder()
            .context("unsupported or invalid image")?;
        use image::ImageDecoder;
        let (width, height) = decoder.dimensions();
        let grid = V41ImageGrid::plan(width, height)?;
        // Match the reference's convert("RGB"): discard alpha, no EXIF transpose.
        let decoded = image::DynamicImage::from_decoder(decoder)?;
        let rgb = match decoded {
            // Pillow reads 16-bit grayscale PNG into integer mode and clamps
            // during RGB conversion; scaling 0..65535 to 0..255 changes content.
            image::DynamicImage::ImageLuma16(gray) => {
                image::RgbImage::from_fn(width, height, |x, y| {
                    image::Rgb([gray.get_pixel(x, y).0[0].min(255) as u8; 3])
                })
            }
            other => other.to_rgb8(),
        };
        Self::from_rgb_with_grid(width, height, rgb.as_raw(), grid)
    }
    pub fn from_rgb(width: u32, height: u32, rgb: &[u8]) -> Result<Self> {
        Self::from_rgb_with_grid(width, height, rgb, V41ImageGrid::plan(width, height)?)
    }
    fn from_rgb_with_grid(width: u32, height: u32, rgb: &[u8], grid: V41ImageGrid) -> Result<Self> {
        ensure!(
            rgb.len() == width as usize * height as usize * 3,
            "RGB byte extent differs from image dimensions"
        );
        let rgb = resize::pad(
            rgb,
            width as usize,
            height as usize,
            grid.pixel_width,
            grid.pixel_height,
        )?;
        let values: [u16; 256] = std::array::from_fn(|i| {
            let value = ((i as f32 / 255.0) - 0.5) / 0.5;
            let bits = value.to_bits();
            ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        });
        let mut patches =
            Vec::with_capacity(grid.vit_height * grid.vit_width * 3 * PATCH * PATCH * 2);
        for y in 0..grid.vit_height {
            for x in 0..grid.vit_width {
                for channel in 0..3 {
                    for py in 0..PATCH {
                        for px in 0..PATCH {
                            let i = ((y * PATCH + py) * grid.pixel_width + x * PATCH + px) * 3
                                + channel;
                            patches.extend_from_slice(&values[rgb[i] as usize].to_le_bytes());
                        }
                    }
                }
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"ds41rt-v41-image-patches-v1\0");
        hash.update((grid.vit_height as u64).to_le_bytes());
        hash.update((grid.vit_width as u64).to_le_bytes());
        hash.update(&patches);
        Ok(Self {
            grid,
            patches,
            identity: hash.finalize().into(),
        })
    }
    pub fn grid(&self) -> V41ImageGrid {
        self.grid
    }
    pub fn patches(&self) -> &[u8] {
        &self.patches
    }
    /// Content identity includes preprocessed pixels and two-dimensional geometry.
    /// URL/placeholder tokens alone are not safe image-prefix cache identities.
    pub fn identity(&self) -> &[u8; 32] {
        &self.identity
    }
}

pub struct V41ImageSpan {
    pub start: usize,
    pub image: V41Image,
}
pub struct V41VisionPrompt {
    pub tokens: Vec<u32>,
    pub types: Vec<V41ImageTokenType>,
    pub images: Vec<V41ImageSpan>,
}
impl V41VisionPrompt {
    /// Expand tokenized placeholders in conversation order before output/context
    /// admission. The caller must include images from every retained user turn.
    pub fn expand(tokens: &[u32], images: Vec<V41Image>, context_limit: usize) -> Result<Self> {
        ensure!(
            images.len() <= V41_MAX_IMAGES,
            "a prompt accepts at most 16 images"
        );
        ensure!(
            tokens.iter().filter(|&&t| t == V41_IMAGE_TOKEN_ID).count() == images.len(),
            "image placeholder count differs from supplied images"
        );
        ensure!(
            tokens.iter().all(|&t| t < 129280),
            "token outside official vocabulary"
        );
        let extra: usize = images.iter().map(|i| i.grid.tokens() - 1).sum();
        let count = tokens
            .len()
            .checked_add(extra)
            .context("expanded image prompt length overflow")?;
        ensure!(
            count <= context_limit.min(1_048_576),
            "expanded image prompt exceeds context limit"
        );
        let mut result = Self {
            tokens: Vec::with_capacity(count),
            types: Vec::with_capacity(count),
            images: Vec::with_capacity(images.len()),
        };
        let mut images = images.into_iter();
        for &token in tokens {
            if token == V41_IMAGE_TOKEN_ID {
                let image = images.next().expect("validated image count");
                result.images.push(V41ImageSpan {
                    start: result.tokens.len(),
                    image,
                });
                let grid = result.images.last().unwrap().image.grid;
                result
                    .tokens
                    .extend(std::iter::repeat_n(token, grid.tokens()));
                result.types.extend(grid.token_types());
            } else {
                result.tokens.push(token);
                result.types.push(V41ImageTokenType::Text);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
