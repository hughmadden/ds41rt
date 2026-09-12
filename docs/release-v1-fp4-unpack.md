# FP4 attention unpack optimization

SM120a attention now converts packed E2M1 directly to BF16 pairs before the
existing packed scale multiplication. This replaces four independent nibble
expansions. The portable path and FP8 window format are unchanged.

[Evidence](release-v1-fp4-unpack.json) and the
[raw archive](evidence/native-fp4-unpack.json.gz) preserve both the selected direct
conversion and the unselected byte-table prototype. Against the initial FP4
kernel, direct conversion reduces full-source attention time by 6.3–11.0% for
ordinary/grouped kernels and 9.3–11.9% for split kernels across 1, 2, 6, 16,
80, 128 and 256 query rows. These are CUDA graph microbenchmarks with identical
packed buffers and 100 repetitions, not serving throughput claims. Direct
conversion generally beats the table prototype on full-source timing.

Both prototypes pass 140 unaligned ordinary/split checks against independently
decoded BF16 source attention. The selected CUDA 13.2 full native build then
passes all 1,470 attention checks across the two GPUs and legacy/window paths.
All 96 native closed-form cases also pass Compute Sanitizer with zero errors.
An initial full-library test invocation lacked the CUTLASS runtime library path
and failed before loading the kernel; the completed rerun supplies that path.

The preceding initial FP4 serving build was also measured at longer contexts:

| Mode | Filler tokens | Median paired decode change vs FP8 |
| --- | ---: | ---: |
| Target | 4,096 | −0.41% |
| Target | 16,384 | −0.44% |
| Target | 24,576 | −0.43% |
| dSpark | 4,096 | +0.06% |
| dSpark | 16,384 | −0.14% |
| dSpark | 24,576 | −0.74% |

Each row uses four alternating paired measurements, 512 output tokens, thinking
disabled for controlled counting throughput, and sequential shared Spark workers.
Both arms produce identical text and have full prompt-cache hits throughout.
These prove prompt reuse, not completed-turn agentic reuse. Cold first requests
are retained in the archive but are single observations, not a qualified prefill
comparison. Hardware remains the 400 W RTX configuration with standard memory
speed documented in the [integration report](release-v1-compressed-serving.md).

The optimized serving comparison is pending. The full performance gate and the
subsequent needle/high-thinking tool qualification remain open.

For comparisons between FP4 implementations, use
`scripts/compare-ds41-sparse-attention.py --fp4-source --baseline-fp4-source`
with the baseline and candidate libraries. Omit `--baseline-fp4-source` when
comparing with the independently decoded BF16 reference reader.
