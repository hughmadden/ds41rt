# Grouped FP8 attention output projection

Backbone and all three dSpark stages now retain the official FP8 `wo_a` weights and use grouped MXFP8 GEMM. The eight groups each project only their own 4096 input coordinates to 1024 output coordinates. Checkpoint FP8 bytes are unchanged; native UE8M0 32×32 weight scales are packed on the GPU at load time. This follows the optimization suggested beside the grouped einsum in the official reference, rather than dequantizing resident weights to BF16.

The runtime graph performs inverse RoPE, grouped K32 activation quantization, grouped GEMM with BF16 output, a group-to-token layout copy, and the existing FP8 `wo_b`. Activation quantization introduces numerical differences from the previous BF16 `wo_a`. It uses ceil power-of-two scales and a scale of one for all-zero blocks. The grouped quantizer does not use the dense projection's `1e-4` amax floor. The existing fused input-quantization GEMM was slower in the targeted probe, so it is not selected.

b12x exports runtime-row grouped GEMM and CuTeDSL quantization for capacities 1/16/80/256/1024/4096. The native launch owns no replay allocations, compilation or synchronization; callers retain preallocated scratch and initialized alpha. The grouped output copy uses 64-bit addressing. Shape `(K=32768,N=8192)` explicitly denotes the eight-group operation in the native/FFI interface; its actual weight and scale spans are checked rather than treating it as a dense matrix.

This removes approximately **1.39 GiB** of resident storage across forty backbone layers and three dSpark stages, including packed scale costs and the old retained dSpark FP8 source copies. It changes attention projection precision, not persistent KV format.

## Component evidence

The exported C ABI passed 48 changed-input graph cases across six capacities, with quantized values/scales and GEMM outputs exact against the existing grouped GPU path. Full native qualification then passed **174 cases**: all 43 official weight sets, normal/tiny/zero inputs, and live rows 1/6/capacity under every export for layer zero. Independent FP32 per-group GEMMs use explicitly quantized activations and decoded official weights. These are random-input component checks, not full-model numerical equivalence. Normal-input relative L2 error against BF16 `wo_a` is around 2.6–2.7% in the initial layer-zero probe.

Final native medians in microseconds, including activation quantization and the output layout copy:

| Rows | Previous BF16 | Grouped FP8 |
|---:|---:|---:|
| 1 | 51.20 | 30.72 |
| 16 | 49.01 | 30.72 |
| 80 | 55.30 | 47.12 |
| 256 | 100.35 | 75.78 |
| 1024 | 391.17 | 239.62 |
| 4096 | 706.56 | 870.40 |

Each candidate precedes/follows the other in alternating order, with a 256 MiB buffer zeroed before each graph replay; medians use twenty samples after four warmups. No clock/throttle admission was applied. **4096-row performance regresses and needs a different prefill schedule.** These measurements do not establish end-to-end prefill throughput.

Native/CUDA CTest both pass. Rust check and release build pass. The retained qualifier is `python/tools/qualify_v41_grouped_fp8.py`.

## Live development rollout and limits

The active coordinators are `ds41-grouped-fp8-target-api-dev` and `ds41-grouped-fp8-spec-api-dev`, ports 18041/18042. Both use `/tmp/ds41-grouped-fp8-artifacts/daemon` and its `cmake/libds41rt_native.so`. The four Spark workers and RoCE inference transport are unchanged; stopped packed-attention coordinators remain available as the baseline.

Three sequential counting streams give medians **31.01→31.65 target TPS** and **102.27→103.46 dSpark TPS**. The prompt is `Count from 1 to 20, separated by commas. Output only the numbers.` with temperature zero, thinking disabled and max_tokens 96. Rates exclude time through first content and include subsequent API overhead. This predictable short workload is not representative chat/coding qualification. Both APIs pass the streaming/usage, cancellation-recovery and unsupported-sampling smoke checks.

All eight quality cases produce identical text and usage between target and speculative execution. Seven preserve their previous output. The Unicode arithmetic case changes from `36` to correct working followed by `36`, violating its instruction to output only the number. **Only five of six strict objective checks pass; the quality script exits nonzero.** This is a recorded formatting regression, not a clean quality pass. Broader numerical, instruction-following and hosted-reference evaluation remain open.

[Commands, artifact hashes, native cases, API rates and quality outputs](ds41-grouped-fp8-output.json) retain the evidence. Raw exports/probes are under `/tmp/ds41-grouped-fp8-aot` and `/tmp/ds41-projection-fp8`; full native and rollout records are under `/tmp/ds41-grouped-fp8-artifacts` and `/tmp/ds41-grouped-fp8-rollout`. The 90/270/8000 TPS, C16, alternating-wave and release-quality gates remain open.
