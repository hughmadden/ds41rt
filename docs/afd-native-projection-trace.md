# Completed projection diagnostics — 19 September 2026 AEST

This diagnostic branch extends the opt-in target activation trace. Set
`DS41RT_ACTIVATION_TRACE_DIR`, `DS41RT_ACTIVATION_TRACE_POSITION=10` and
`DS41RT_ACTIVATION_TRACE_DETAIL_LAYER=2` to capture the selected layer for batches
whose first native input position is 10. The existing owner-thread subscriber
activates only when a trace directory is supplied. Use a fresh directory for each
process; every binary/manifest uses `create_new` and refuses replacement.

Queued attention waits for the complete sparse/projection/FFN chain, then dumps
projection buffers before releasing the pending lane guard. It does not call
`AttentionOutputWave::output()`: the combined chain deliberately never marks that
standalone output ready. The synchronous path retains the same projection owner
through its FFN view and reads it only after completed execution. With tracing
disabled, the existing computation and wait paths are unchanged.

Each selected batch directory contains these additional files (`layer2` changes
with the selected layer):

| File | Exact native contents |
| --- | --- |
| `layer2-attention-values.bin` | BF16 `[live_rows,32768]`, projection buffer b0 |
| `layer2-wo-a-output.bin` | BF16 `[live_rows,8192]`, b1 and WO-B input |
| `layer2-wo-b-output.bin` | BF16 `[live_rows,5120]`, b2 |
| `layer2-attention-frequencies.bin` | FP32 `[live_rows,64]` inverse-RoPE frequencies |
| `layer2-projection-positions.bin` | Device u64 `[live_rows]` token positions |
| `layer2-projection-alpha.bin` | Actual FP32 alpha scalar |
| `layer2-wo-a-scratch.bin` | Entire scratch slice of the exact WO-A variant selected for this launch |
| `layer2-wo-b-scratch.bin` | Entire scratch slice of the exact WO-B variant selected for this launch |
| `layer2-projection.json` | Live geometry, filenames, both selected variants' native ABI metadata, and shared weight paths |

The scratch metadata includes `capacity_rows`, `input_dim`, `output_dim`,
`scratch_bytes`, `values_offset`, `row_scales_offset`, `mma_scales_offset`, and
`packed_weight_scale_bytes`. The accessor calls the same `V41Fp8Plan::select`
used by launch; it neither reconstructs quantization nor changes kernel selection.
Only live input rows are valid for the numerical oracle: the entire selected
scratch also includes padded/inactive and intermediate storage. All binary
values have the native x86-64 little-endian representation.

`<trace-root>/projection-weights` holds the actual loaded WO-B E4M3FN weight
`[5120,8192]`, original UE8M0 scales `[160,256]`, and native packed scales
`[1310720]`, in files named `layer2-wo-b-weight.bin`, `layer2-wo-b-scales.bin`, and
`layer2-wo-b-packed-scales.bin`. The immutable loaded weight owner copies these
once per explicit trace root across both lanes; no batch duplicates the 40 MiB
weight. The root is passed from the trace configuration, never guessed from the
batch path. The oracle should hash-pin these files and the manifest; no checkpoint
requantization is needed.

These copies synchronize diagnostic reads and are unsuitable for throughput
measurements. CPU checks validate live-row bounds and overflow rejection; the
parent campaign performs the GPU trace and oracle comparison. This instrumentation
does not by itself fix or establish numerical correctness.
