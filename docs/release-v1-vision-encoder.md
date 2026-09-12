# Native vision encoder component

The native coordinator now has a complete V4.1 vision encoder and aligner:
32 transformer blocks, 2D rotary positions, 3×3 spatial merging, the language
projection and learned image delimiters. Its owner loads the 266 official BF16
tensors and reuses one bounded workspace between images. This report covers the
component's numerical checks; [native serving qualification](release-v1-vision-serving.md)
covers API admission, image-aware prefix reuse, routing and image answers.

## Execution and memory

Patch projection and transformer/aligner matrix products use cuBLAS with FP32
accumulation and BF16 storage. Biased projections add the bias before rounding
the result to BF16. Normalization and rotary arithmetic use FP32. Attention
processes at most 128 queries at a time, retaining FP32 scores and softmax but
storing probabilities in BF16 for the tensor-core value product. The FP32
probability/value-product path remains an internal numerical control; there is
no serving precision option. Probability scratch reuses the control's value
scratch, so the faster path requires no additional allocation.

The image owner accepts up to 9,216 patches and produces a complete span of at
most 1,024 language-model tokens. Spatial merging zero-pads the right/bottom
edges and uses channel-major 3×3 order. Every aligned row receives its learned
newline; the span also includes learned start and end embeddings. Inputs,
outputs and scratch have explicit extent, alignment, aliasing and geometry
checks. Failed executions drain outstanding work before reusing buffers.

Maximum component allocation is **1,587,976,256 bytes (1.48 GiB)**: 970,536,960
bytes of checkpoint weights, 613,244,992 bytes of tensor scratch and 4,194,304
bytes of cuBLAS workspace. The encoder workspace is shared between images;
serving must budget request image data and retained feature spans separately.

## Component measurements

These measurements use an RTX PRO 6000 Blackwell at an enforced **400 W power
limit and standard memory speed**, driver 595.91.07. The captured loaded memory
clock is 13,365 MHz; the reported maximum is 14,001 MHz. Runs are serial on GPU 0.
These are individual warm encoder measurements from the diagnostic owner, not
release throughput or API latency results. They exclude image decoding and
language-model prefill.

| Input / patch grid | BF16 attention | FP32 control |
| --- | ---: | ---: |
| Synthetic, 39×39 | 21.5 ms | 31.3 ms |
| Maximum-span synthetic, 3×3063 | 695.4 ms | 923.0 ms |
| 1920×1080 RGB pattern, 69×122 | 351.9 ms | 761.6 ms |
| Mountain photograph, 52×35 | 20.3 ms | 42.3 ms |
| Baidu logo, 22×71 | 16.5 ms | 33.1 ms |

## Correctness evidence and limits

Independent full-tensor operation checks pass 64 numerical/layout cases and
12 rejection checks. Coverage includes query counts 1, 35, 129 and 257, ordinary
and sharply peaked attention, both precision paths, biased/unbiased projection,
normalization, rotary positions, residual addition, SwiGLU, GELU, padded merging
and complete image spans. All operation checks also pass Compute Sanitizer
memcheck with zero errors. The complete 32-layer native encoder also passes
memcheck on small, normal and maximum-span grids, then a small grid again,
with zero errors. Operation checks additionally pass with the native CUDA 13.3
BLAS libraries preloaded instead of PyTorch's bundled libraries.

Both full encoder paths pass 13 stable-repeat cases: synthetic small/normal/
maximum-span grids and nine decoded images, including JPEG, RGBA, grayscale,
very wide input and two fixtures from the neighboring GLM release checkout.
Returning from a large image to a small one is stable, short patch buffers are
rejected, and the owner rejects insufficient device budgets. All compared
patch projections and learned delimiters match the pinned reference exactly.

The comparison uses the complete saved stage and output tensors. Final BF16
span relative L2 differences from the pinned reference range from 0.76% to
4.19%; the mountain photograph and Baidu logo are 1.74% and 1.71%. The FP32 control also
has accumulated rounding differences, reaching 4.04% on the very wide synthetic
pattern. Switching the reference itself from its 3D math attention to 4D fused
attention produces final-span differences up to 3.75%, without changing weights
or image inputs. That control is explicitly labeled as a modified backend;
the unmodified pinned implementation remains the principal reference.

The exploratory 3% relative-L2 / 0.9995 cosine bounds **do not pass at every
intermediate stage**, and two final BF16 spans exceed the relative-L2 bound.
Those failures are preserved. They are diagnostics for investigating numerical
drift, not a substitute for image-answer quality. The faster path is the
component default; accepting it for release still requires image understanding,
16-image serving, routing/Engram masks, chunked replay, correct image identity in
prefix reuse, cancellation/recovery and text-performance regression checks.

## Reproduction

Build the native CUDA library and daemon test binary, then run the ignored
`v41_vision::tests::native_full_vision_vectors` test with:

```bash
export CUDA_VISIBLE_DEVICES=0
export DS41RT_VISION_LIBRARY=/path/to/libds41rt_native.so
export DS41RT_VISION_MODEL=/path/to/official/checkpoint
export DS41RT_VISION_OUTPUT=/tmp/new-vision-vectors
# Optional: DS41RT_VISION_IMAGES=/path/to/json-array-of-image-paths
# Optional numerical control: DS41RT_VISION_FP32_ATTENTION=1
cargo test --manifest-path rust/Cargo.toml -p ds41rt-daemon --bin ds41rt \
  v41_vision::tests::native_full_vision_vectors -- --ignored --exact --nocapture

python scripts/qualify-ds41-vision-ops.py \
  --library "$DS41RT_VISION_LIBRARY" --report /tmp/new-vision-ops.json
python scripts/qualify-ds41-vision-encoder.py \
  --model "$DS41RT_VISION_MODEL" --vectors "$DS41RT_VISION_OUTPUT" \
  --report /tmp/new-vision-reference.json
```

Use an environment containing PyTorch, NumPy and safetensors for the independent
checks. The scripts verify the pinned reference source hash and refuse to
replace existing reports. The encoder comparator exits unsuccessfully when its
diagnostic bounds are exceeded. Its optional `--reference-attention flash` flag
selects the explicit fused-backend control. Reference spans can be saved and
compared across backends using `--reference-spans` and
`--compare-reference-spans`. Run the operation script and native full-vector test
under `compute-sanitizer --tool memcheck --error-exitcode 99` for device checks.

[Evidence metadata](release-v1-vision-encoder.json) records source, artifact and
input hashes with the complete comparison reports, logs and reproduction
settings. The native library for this checkpoint was built separately with
expert AOT and XGrammar disabled; the serving report records the separate full
AOT/XGrammar build used for live image requests.
