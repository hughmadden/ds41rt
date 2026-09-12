# Architectural compressed KV restoration

The release plan places this migration **after native XGrammar enforcement**.
Serving currently still uses FP8 compressed KV. The former interpretation of
“FP8 KV” as requiring replacement of DeepSeek's architectural compressed FP4
was incorrect. The intended layout is FP4 compressed KV, FP8 sliding-window KV,
and the separate existing FP4 index keys/queries.

The new standalone CUDA packing/scatter primitives and Rust FFI constructor are
preparation for that migration. They are not selected by the serving compressor.
A 512-coordinate compressed row occupies 256 E2M1 value bytes and 32 E4M3 scale
bytes (groups of 16), versus the current 512 + 16 bytes. Optional RoPE rotates
the final 64 coordinates and rounds to BF16 before quantization. Scales use
`E4M3(max(amax, 6 * 2^-9) / 6)`, with round-to-nearest E2M1 values. Inputs and
rotated coordinates must remain within the finite scale range (magnitude <=2688).

## Primitive qualification

[Machine-readable evidence](release-v1-compressed-kv.json) records 91 cases on
each RTX: 85 archived real compressor latent/frequency vectors, random batches
of 1/35/129/4096 rows, zeros, and a sweep of every finite BF16 value in the
supported magnitude range. The script verifies the pinned reference hashes and
uses the reference's **in-place** compressed quantizer, as reference serving does.
Decoded native FP4 values match its BF16 results exactly; scale bytes match the
reference formula independently evaluated in Torch. Reverse scatter, skipped
out-of-range destinations and untouched cache sentinels match exactly. Existing
window FP8 packing remains byte-identical to its pinned reference in all cases.
Compute Sanitizer memcheck passes all 91 RTX0 cases with zero errors.
`cargo check -p ds41rt-ffi` passes. With CUDA 13.3 targeting SM120,
the existing FP8 pack and scatter produce identical GPU instructions and encoded
instruction words before/after this refactor (208 and 48 instructions respectively,
ignoring disassembler whitespace and symbol names). This is a primitive check,
not full-serving performance qualification.

An initial test using the reference's out-of-place packed FP4 output failed at
`c0`: scale bytes agreed, but output bytes did not. This was not counted as a
pass or worked around by loosening tolerances. Qualification instead exercises
the reference's serving path and explicitly decodes native low/high nibbles for
exact value comparison. The packed-output discrepancy remains outside this
primitive's correctness claim. TileLang 0.1.8 uses the existing qualification
override `tir.disable_vectorize=true`.

Reproduce with an isolated build (do not overwrite a live server library):

```sh
nvcc -std=c++17 -O3 -arch=sm_120 -shared -Xcompiler -fPIC \
  -I native/include native/cuda/kernels/v41_kv.cu -o /tmp/libkv.so
.ds41rt-cache/reference-venv/bin/python scripts/qualify-ds41-compressed-kv.py \
  --reference-dir /tmp/ds41-reference --native-lib /tmp/libkv.so \
  --vectors-dir /tmp/ds41-kv-source-rtx0 --device 0 --output /tmp/fp4-kv.json
```

## Remaining migration and release gates

After XGrammar, wire compressed proposals and persistent pages to this format;
update mixed FP4/FP8 attention reads, cache copies/retention, buffer validation,
cache identity, and pool sizing together. Optimize unpacking and attention for
prefill, decode and dSpark, measuring against the existing FP8 baseline with
matched hardware settings. Smaller traffic is not proof of higher throughput.
Performance must match or improve before the requested needle and high-thinking
tool-call qualification. Validate cold, partial, exact and divergent cache reuse,
concurrency, eviction and cancellation as well. The final release suite, memory
report and three tool-eval runs must describe the corrected serving build.
