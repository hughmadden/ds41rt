# Fused inverse RoPE and grouped FP8 quantization

Backbone and all dSpark output projections now consume sparse-attention BF16 output directly. The CuTeDSL grouped quantizer applies inverse RoPE from interleaved FP32 per-row frequencies, rounds rotated values to BF16, and writes the same grouped FP8 values and UE8M0 scales as the previous two-kernel path. It avoids writing and rereading the full rotated BF16 activation. Each output wave also drops that allocation: **65,536 bytes per capacity row**, or 256 MiB at capacity 4096.

The native FP8 launch gains an optional frequency input. Non-null frequencies are accepted only for the eight-group WO-A geometry, with extent/alignment/overlap validation. Null preserves the ordinary projection path. Both quantizers and GEMM are loaded before graph capture; all live row counts remain runtime arguments. No replay allocation or host synchronization is added. Frequency generation, grouped GEMM, output layout conversion and `wo_b` retain their prior behavior.

## Profiling and validation

The isolated 4096-row path initially measured approximately 311 µs for input quantization, 485 µs for grouped GEMM and 78 µs for output layout conversion, excluding inverse RoPE. A four-schedule BK128 comparison did not improve on the existing 128×64 GEMM tile. An experimental 128×128/BK64 path produced invalid global reads under compute-sanitizer and was rejected; it is not used by the native deployment. Its isolated diagnostic remains open.

Fusing the producer removed more work than those tile changes. The initial RoPE-plus-quantization probe fell from roughly 672 to 313 µs at 4096 rows, with exact FP8 values and scale bytes in all 32 replay cases.

The full native qualifier, `python/tools/qualify_v41_rope_fp8.py`, uses actual layer-zero checkpoint weights. Across rows 1/2/6/16/80/256/1024/4096, all 32 normal/changed/tiny/zero cases preserve both quantized operands and final BF16 grouped-projection output exactly. Changed replay also changes frequencies. Invalid row counts, undersized scratch and frequency/input aliasing reject before enqueue. Native/CUDA CTest and the daemon release build pass.

Full RoPE-through-WO-A native medians (µs):

| Rows | Separate RoPE + FP8 | Fused RoPE + FP8 |
|---:|---:|---:|
| 1 | 30.72 | 30.72 |
| 6 | 32.77 | 30.72 |
| 80 | 51.20 | 47.12 |
| 256 | 86.05 | 76.59 |
| 1024 | 288.77 | 242.70 |
| 4096 | 1234.94 | 868.35 |

These alternating CUDA-graph comparisons zero a 256 MiB buffer before each arm and take twenty samples after four warmups, without clock/throttle admission. They include grouped GEMM and output layout conversion but exclude `wo_b`, frequency generation and the rest of the model. They show about a 30% improvement in the 4096-row measured chain, not end-to-end prefill TPS. The grouped GEMM itself retains the previous schedule; fusion addresses producer memory traffic rather than claiming that standalone GEMM regression is fixed.

## Live development state

`ds41-rope-fp8-target-api-dev` and `ds41-rope-fp8-spec-api-dev` serve ports 18041/18042 using `/tmp/ds41-rope-fp8-artifacts/daemon` and `cmake/libds41rt_native.so`. The prior grouped-FP8 coordinators are stopped and retained. Spark workers and RoCE inference transport are unchanged.

Three short counting streams give medians 31.47→31.85 target TPS and 104.00→104.35 dSpark TPS. These small changes are not a convincing decode-speed improvement; the measured benefit is mainly at large rows. Streaming/usage, cancellation recovery and unsupported-sampling smoke checks pass.

All eight paired quality outputs and usage counts match the prior FP8 deployment exactly, and target/speculative results match each other. **The inherited Unicode arithmetic formatting regression remains: five of six strict objective cases pass and the quality script exits nonzero.** This fusion adds no observed change in that suite, but does not close broader quality gates.

[Commands, hashes, component samples, live results and diagnostic records](ds41-rope-fp8-fusion.json) retain the evidence. Raw probes are under `/tmp/ds41-grouped-prefill`, builds under `/tmp/ds41-rope-fp8-artifacts`, and API records under `/tmp/ds41-rope-fp8-rollout`. Full-context prefill, C16/alternating waves, numerical/instruction-following evaluation, and the 90/270/8000 TPS release targets remain open.
