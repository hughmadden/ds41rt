# Device-descriptor sparse attention probe

RTX PRO 6000 Blackwell, **400 W power limit and standard memory speed**
(configured maximum 14,001 MHz; no memory overclock). One RTX was used. This
is an exploratory CUDA-graph microbenchmark with the standard service resident
and idle, without a locked-clock/throttling gate. It does not establish serving
TPS, prefill preservation, or adaptive-policy quality.

The existing small-verifier path launches split attention and merge separately
for each request. Its captured graph retains external cache pointers and request
row counts, so a new layout may require recapture. The isolated candidate reads
one 120-byte device descriptor per query row, then launches attention and merge
once for the entire wave. Cache bindings and request boundaries can change
without recapture. Each row retains its own causal width, the original key
partition count, BF16 probability rounding, and FP32 merge order.

No serving code or native public ABI was changed. The experimental entry point
assumes validated, live device descriptors and disjoint storage. It must not be
installed in serving without the equivalent host validation and lifetime rules.

| Cache source | Rows / requests | Existing graph µs | Batched graph µs |
| --- | ---: | ---: | ---: |
| SWA only | 1 / 1 | 16.43 | 16.44 |
| SWA only | 6 / 1 | 16.74 | 18.49 |
| SWA only | 24 / 8 | 123.45 | 18.49 |
| SWA only | 48 / 8 | 127.95 | 34.78 |
| FP4 compressed + FP8 SWA | 1 / 1 | 24.63 | 24.63 |
| FP4 compressed + FP8 SWA | 6 / 1 | 40.30 | 41.01 |
| FP4 compressed + FP8 SWA | 24 / 8 | 196.30 | 104.29 |
| FP4 compressed + FP8 SWA | 48 / 8 | 309.13 | 195.36 |

Times are medians of ten interleaved samples per arm, each containing 20 graph
replays after two warmup samples; arm order alternates. They include split and
merge kernels, exclude descriptor/metadata uploads and capture, and use a final
4K-position synthetic fixture with bounded local replay. SWA uses two partitions;
compressed source uses ten. The 24-row layout distributes 1–6 rows per request.
An earlier run measured FP4 24-row 196.00 → 103.91 µs and 48-row
309.51 → 195.12 µs. Keep the existing single-request path: batching has no C1
benefit here. Legacy FP8-source results are included in the JSON for compatibility.

## Numerical and memory checks

All **132 cases were byte-exact** against the existing library. CUDA memcheck
repeated those cases with **zero errors**. Coverage includes SWA-only, legacy FP8
source, and FP4 source; 1, 6, 24, and 48 total rows; private window and compressor
rows; paged committed source; negative selected IDs; zero, truncated, and invalid
replay bounds; positions around 64/128 tile boundaries and up to 131,072.
External cache allocations rotate on every replay, and the heterogeneous request
row counts rotate while total rows stay fixed. The candidate graph is captured
once per format/total-row case. Outputs are poisoned before each comparison.
These checks prove equivalence for the fixtures, not model-level quality.

Full samples, case coverage and artifact hashes are in
[the probe data](phase1-sparse-batch-probe.json).

## Reproduction

Generate a separate source file; leave the production kernel untouched:

```bash
mkdir -p /tmp/ds41-sparse-batch
python scripts/make-ds41-sparse-batch-probe.py --output /tmp/ds41-sparse-batch/candidate.cu
docker run --rm -v "$PWD:/src:ro" -v /tmp/ds41-sparse-batch:/audit \
  --entrypoint nvcc ds41rt-coordinator-dev:latest \
  -std=c++17 -O3 -arch=sm_120a --shared -Xcompiler=-fPIC \
  -I/src/native/include /audit/candidate.cu -o /audit/candidate.so
```

In the CUDA/PyTorch dev image, mount the repo, generated library and existing
native library, then run:

```bash
python /src/scripts/probe-ds41-sparse-batch.py \
  --baseline /baseline/libds41rt_native.so --candidate /audit/candidate.so \
  --output /audit/results.json
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python /src/scripts/probe-ds41-sparse-batch.py \
  --baseline /baseline/libds41rt_native.so --candidate /audit/candidate.so \
  --output /audit/memcheck.json --skip-timing
```

Select only the Phase 1 RTX with Docker `--gpus device=GPU-...`. Output files must
be new. The original audit directory was `/tmp/ds41-phase1-sparse-batch`.

## Integration follow-up

The subsequent implementation and serving measurements are tracked in
[batched sparse attention in serving](phase1-sparse-batch-serving.md). The following
requirements were identified by this isolated probe:

Validate and upload current per-row descriptors into wave-owned storage before
graph replay; retain all borrowed allocations until stream completion. Make the
small multi-request graph key depend on total rows and stable wave-owned launch
storage, with homogeneous source format and split count. Preserve the existing
single-request and large-prefill paths. Budget all scratch before allocation:
48 rows × 10 partitions × 64 heads × 514 FP32 elements requires 63,160,320 bytes
(60.23 MiB), versus 20.08 MiB for the current 16-row shared arena: **40.16 MiB
extra per wave**, plus at most 5,760 descriptor bytes. This is temporary attention
scratch, not token cache capacity.

Then measure complete verification and C16 throughput, including descriptor
uploads. Compare fixed and adaptive policies on the same optimized path, verify
cancellation/reuse lifetimes, and preserve prefill performance before promotion.
