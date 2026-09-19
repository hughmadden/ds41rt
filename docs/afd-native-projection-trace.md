# Completed projection diagnostics — 19 September 2026 AEST

This diagnostic branch extends the opt-in target activation trace. Set
`DS41RT_ACTIVATION_TRACE_DIR`, `DS41RT_ACTIVATION_TRACE_POSITION=10` and
`DS41RT_ACTIVATION_TRACE_DETAIL_LAYER=2` to capture the selected layer for batches
containing any native member input at position 10. The existing owner-thread subscriber
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

## Grouped ABI amendment — 19 September 2026 AEST

On the grouped native ABI (52130044), batch selection no longer requires the
filter position to be the first flattened row. `DS41RT_ACTIVATION_TRACE_POSITION`
now matches the committed absolute position of **any** flattened row of **any**
group member, so one teacher-forced run can capture whichever step the filter
names (for example the committed position 42 of the diverging C4 output token 21)
without dumping every token. Setting the filter to an earlier committed position
still captures that earlier step. Batches with no member row at the filter
position are excluded entirely — no directory is created for them.

Each selected batch directory additionally contains `members.json` (schema 2)
with the real mapping: batch identity, stage, total rows, the owning lane when
the driver supplied one (`null` when it did not — never guessed), the filter
position, every matched flattened row, and per member the native concatenation
slot (group member index, not bank slot), bank request identity, its flattened row indices, their committed absolute
positions, and which of its rows matched. Binary dumps remain whole-live-row,
exactly as above; `positions.json` and `members.json` together let the oracle
address individual member rows without trusting assumed layouts. When the lane
is known, the directory is named `lane{lane}-batch{id}-{stage}-{rows}rows` so
both lane owners can share one trace root without `create_new` collisions;
the legacy name is kept when the lane is unknown.

Model, transport, kernel selection, and scratch geometry are untouched by these
diagnostics; the WO-B capacity-16 probe remains a separate branch and is not
part of this port.

## Completed FFN diagnostics — 19 September 2026 AEST

The same position and detail-layer selection now follows `PreparedLayer` into
its actual TP4 FFN dispatch and pending reduction owner. All constructors default
to no selection, and the transport wave retains no trace path across requests.
Only the selected layer copies completed live rows, before owners are released.
No kernel, wire format, reduction order or serving default changes.

For selected layer1, additional buffers are:

| Suffix after `layer1-` | Live row contents |
| --- | --- |
| `ffn-router-ids.bin` | Six uint32 expert IDs |
| `ffn-router-routing.bin` | Six FP32 routing weights |
| `ffn-router-expert-input.bin` | 5120 E4M3 values +160 UE8M0 scale bytes (5280 bytes) |
| `ffn-shared.bin` | Completed local shared contribution, 5120 BF16 |
| `ffn-plane-rank0.bin` through `ffn-plane-rank3.bin` | Actual received rank partials, each 5120 BF16 |
| `ffn-reduced-output.bin` | Completed shared+routed result before final mHC, 5120 BF16 |

`ffn-router.json`, `ffn-shared.json` and `ffn-reduction.json` record dtype,
shape, row stride and exact live bytes. The packed FP8 expert input's shape
counts logical values; its row stride additionally includes the scale bytes.
`ffn-routes.json` preserves the existing CPU request routes and each flattened
row's request ID, position and source kind. Join these identities to members.json;
never infer physical bank slots from the member ordinal. Partial row stride
comes from `V41_PARTIAL_ROW_BYTES`, not an assumed FP32 expert-slot layout.

Pure CPU checks reject zero rows, overflow and capacity overruns, and validate
selection and manifest extents. The existing native executor tests cover grouped
consumer/commit/cancel ownership. These diagnostics synchronize D2H reads: full
logit hashes must still match the untraced replay before using the captures, and
traced timings must not be used for performance comparisons.

## Sparse attention native operand capture — 19 September 2026 AEST

The same `DS41RT_ACTIVATION_TRACE_DIR` + `DS41RT_ACTIVATION_TRACE_POSITION` +
`DS41RT_ACTIVATION_TRACE_DETAIL_LAYER` selection now also captures the exact
native operands of that layer's sparse-attention invocation. The target pass
arms a one-shot trigger on the lane's sparse wave at the detail layer and
clears it at every layer; the wave consumes the trigger on entry to its staged
execution and, when armed, synchronizes its own stream after every query,
descriptor, metadata and bounds upload is queued — before tail preparation,
warmup, graph capture or launch, so the copies can never run inside a CUDA
graph capture. With the trigger unset there is no extra synchronization, no
device-to-host copy and no new allocation; cached and cold (deferred warmup)
paths are both covered because the capture sits before the graph-cache lookup.
A capture that would exceed 64 MiB fails explicitly instead of dumping pools
or truncating.

Each selected batch directory additionally contains (`layerN` is the detail
layer, `requestS` the member slot in native concatenation order; members.json
binds slots to real request IDs):

| File | Exact native contents |
| --- | --- |
| `layerN-attention-query.bin` | Live BF16 `[rows,64,512]` rotated queries |
| `layerN-attention-sink.bin` | FP32 `[64]` finite sink |
| `layerN-attention-metadata.bin` | Device U64 `[rows,10]` window+source metadata |
| `layerN-attention-selected.bin` | Device I32 `[rows,512]` selection, when a source is present |
| `layerN-attention-replay-begins.bin` | Batch bounds upload U64 `[rows]`, descriptor-batch launches only |
| `layerN-attention-descriptors-device.bin` / `-host.bin` | Actual GPU descriptor bytes and the staged host bytes, batch only, compared as bytes |
| `layerN-attention-requestS-ring-values.bin` / `-ring-scales.bin` | FP8 E4M3 values `[128,512]` and E8M0 group scales `[128,16]` of the ring |
| `layerN-attention-requestS-window-end.bin` | Actual device U64 window end scalar |
| `layerN-attention-requestS-window-proposal-values.bin` / `-scales.bin` | Private window proposal at its live capacity |
| `layerN-attention-requestS-source-pages.bin` | Device U32 page table `[page_stride]` |
| `layerN-attention-requestS-source-end.bin` | Actual device U64 source end scalar |
| `layerN-attention-requestS-source-proposal-values.bin` / `-scales.bin` | Private source proposal at its live capacity (FP4 values, E4M3 scales) |
| `layerN-attention-requestS-source-referenced-values.bin` / `-scales.bin` | Only the committed physical rows actually referenced: packed 256-byte FP4 rows and 32-byte scale rows in ascending physical order |
| `layerN-attention-inputs.json` | Manifest: launch kind, width, parts, per-request IDs/positions/flattened row ranges, capacities, dtypes, strides, byte offsets, actual window/source end and replay values, the logical→physical map, and per-mask skip counts |

Requests without an uploaded bounds slice get an all-zeros replay-begins file
marked `"synthetic": "explicit_zero"` (the launch passes no bounds pointer and
the buffer is never read). Referenced-row resolution duplicates the kernel's
`locate` masks exactly: only `0 <= id < m[5]`, `id < m[6]`, `id <
page_stride*256` with `physical = pages[id/256]*256 + id%256 <
source_capacity` rows are captured; masked, past-length, private-overlay,
past-stride and past-capacity slots stay in the full selected buffer and are
accounted in the manifest so nothing is silently dropped. The shared compressed
source pool is never dumped whole; raw unused ring/proposal padding is unscored.
All files use `create_new` and refuse replacement.
