# Native vision preprocessing

The loader now prepares official V4.1 image patches and expanded prompt spans
without a Python runtime. This is a component checkpoint: native vision serving
remains gated until the encoder, aligner, image routing and cache integration
are complete.

The pinned model uses a 32-layer, 1,024-wide vision encoder with 16 attention
heads, 14-pixel patches and a 3×3 spatial aligner. Image preprocessing first
applies the reference minimum-area and patch-rounding rules, then shrinks the
grid when its complete span would exceed 1,024 language-model tokens. The
span includes a start delimiter, one newline per aligned row and an end
delimiter. All span positions carry token ID 129264; separate types distinguish
patch features and learned delimiters from ordinary text.

The native implementation matches the reference's RGB conversion, centered
gray padding, bicubic resampling, BF16 rounding and channel-major patch order.
The RGB resampler follows Pillow's signed 22-bit coefficient and inter-pass
byte-rounding contract. JPEG uses statically linked libjpeg-turbo with SIMD and
accurate chroma upsampling. PNG, WebP and GIF use the Rust image codecs. Alpha is
discarded and EXIF orientation is not applied, matching the pinned reference.
Sixteen-bit grayscale is clipped during RGB conversion, rather than rescaled.

Prepared prompts accept up to sixteen images in conversation order, reject
missing or surplus placeholders, and account for expanded spans before checking
context capacity. Input decoding rejects empty data, encoded images above
32 MiB and dimensions above 64 megapixels; the Rust codecs also have a 256 MiB
decode allocation limit. The span representation preserves image boundaries
for chunked prefill and image masks for routing/Engram integration.

Each prepared image has a SHA-256 identity over its exact BF16 patches and
two-dimensional grid. Different image content must not share a prefix merely
because its placeholder and expanded token IDs match. The identity is available
to the upcoming cache integration; this component does not yet change cache
lookup or serving admission.

## Qualification

The reference image processor, vision tower and model source hashes match the
existing [reference lock](ds41-reference-lock.json). Four Rust tests cover 75
reference resize plans, six complete reference patch hashes, interleaved spans,
16-image acceptance/17-image rejection, exact expanded-context limits, content
identity, invalid inputs and grayscale conversion.

The independent format campaign compares all BF16 patch bytes, not only shape
or a selected sample. All 21 cases match exactly: eight RGB sizes/aspect ratios,
RGBA, grayscale, 16-bit grayscale, palette PNG, JPEG 4:4:4 and 4:2:0, progressive
JPEG, grayscale and CMYK JPEG, lossless/lossy WebP, GIF and JPEG EXIF metadata.
Reference Pillow is 12.3.0. Initial failures from grayscale scaling and the
preceding JPEG decoder are preserved; the final run uses the corrected decoder
path and keeps the exact-match criterion.

The release example and full daemon compile successfully. This work does not
change the live text-serving path or make a new throughput claim. The format
campaign is CPU-only; full vision GPU memory, performance, image understanding,
mixed-request cancellation and 16-image serving remain unqualified.

Reproduce the component check after installing NASM on an x86 build host
(the development Dockerfile now includes it):

```bash
cargo test --manifest-path rust/Cargo.toml -p ds41rt-loader v41_image
cargo build --release --manifest-path rust/Cargo.toml -p ds41rt-loader \
  --example v41-image-preprocess
.ds41rt-cache/reference-venv/bin/python scripts/qualify-ds41-image-preprocess.py \
  --reference-dir /tmp/ds41-reference --output-dir /tmp/new-vision-preprocess-check
```

[Evidence metadata](release-v1-vision-preprocess.json) records source, binary and
fixture hashes, complete comparison results and build/test logs. Decoder API
references are the [image crate](https://docs.rs/image/0.25.10/image/) and
[TurboJPEG decompressor](https://docs.rs/turbojpeg/1.5.1/turbojpeg/struct.Decompressor.html).
The model-specific preprocessing source remains the pinned reference above.
