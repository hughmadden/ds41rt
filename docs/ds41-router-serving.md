# Native router integration and upstream attention merge

The development APIs now use the native AOT router for backbone rows >=16 and
dSpark rows >=26. Smaller requests retain the original GEMV. This removes most
large-prefill router computation without changing its BF16 input, FP32 score
buffer or expert wire format. The thresholds come from b12x export metadata.

Both upstream commits `789bbb3c` and `3b862805` were merged in full into b12x
master, producing `e53fd5a1`. There were no merge conflicts. Source lock and all
coordinator AOT artifacts were rebuilt after the merge. This is not a partial
incorporation or a claim that upstream attention kernels are now on our hot path.

## End-to-end result

Sequential C1, 16,410 code / 16,411 repeated-text prompt tokens, greedy count 1–20,
59 output tokens, no prefix-cache hits, batch 2048, four unchanged Spark workers.
Second requests per kind/mode:

| Prompt | Mode | Previous prefill tok/s | Current prefill tok/s | TTFT | Decode tok/s |
|---|---|---:|---:|---:|---:|
| Code | Target | 1430 | 1554 | 10.558 s | 39.10 |
| Code | dSpark | 1447 | 1559 | 10.528 s | 113.58 |
| Repeated | Target | 1625 | 1776 | 9.243 s | 39.15 |
| Repeated | dSpark | 1638 | 1787 | 9.185 s | 116.63 |

Prefill improves roughly 8–9%. All eight benchmark responses, usage records and
prompt hashes match the previous version. First code requests reach 1406 target /
1426 dSpark tok/s. Short target decode is 39.18/40.32/40.39 tok/s; dSpark is
118.90/121.45/120.58 tok/s. Both API streaming, sampling rejection and cancellation
recovery checks pass. This does not qualify C16 or the release performance goals.

The eight paired quality cases retain all five previously passing objective
checks and the existing Unicode instruction-format failure. The Unicode response
adds a blank line. The dSpark explanation changes to a correct two-sentence
paraphrase while the target answer stays unchanged. Consequently paired exact
text/usage is 7/8 and the strict quality script exits 1. This is disclosed rather
than counted as an exact-equivalence pass; broad model quality remains open.

## Native validation

- 72 real-weight cases / 144 mutations across six representative gates, including
  15/16/17 and 25/26/27 crossover boundaries, 2048/4096 rows, and return to small rows.
  AOT outputs match the Python CuTe launch bit-for-bit; small-row outputs match the
  original native router bit-for-bit. The score/set/weight oracle checks pass for
  this corpus. Earlier all-43-gate precision discrepancies remain documented in
  [the initial experiment](ds41-router-gemm.md).
- Captured replay uses exact-sized inputs and guarded, poisoned outputs under
  frozen kernel resolution, with no Torch allocation during replay.
- Launch before initialization returns `cudaErrorNotReady`; repeated initialization,
  13 invalid argument/alias/span cases and exact fallback for 2-byte-aligned BF16
  input pass. Initialization occurs before capture in the Rust owner. Native
  modules belong to one serving device per process, matching the other AOT owners.
- Native CPU/CUDA CTest checks: 2 passed. Rust release daemon build passed.
- Upstream packed mixed-cache conversion and indexer tests: 21 passed, covering
  exhaustive scale/code combinations, graph mutation, tensor-core staged rounding,
  high-page cases and bounded replay/newest-source selection.

The existing dSpark component qualifier now initializes the router before capture
and runs multiple requested devices in separate processes, matching native AOT
ownership. Its numerical, graph and RNG checks pass on both RTX GPUs, including
the router ties/bias/guard checks and stochastic five-step graph comparisons.

## What the upstream patches contribute

| Change | Relationship to this runtime |
|---|---|
| Packed mixed FP8/FP4 attention conversion | Useful implementation to test in our FP8 KV staging; our native sparse-attention kernel does not call it yet. |
| Tensor-core MXFP4 paged indexer and bounded inactive work | Similar goal to our already-deployed native-overlay scorer; direct replacement needs our append-only proposal/causal contract. |
| BF16 prefill GEMM with compensated accumulation | Direct overlap with the router. On real layer-20 weights at M2048, ours takes 112.7 µs versus upstream 182.2 µs; at M4096, 214.5 versus 297.5 µs. Both pass FP64 score tolerance. Retain our measured faster implementation. |
| Lagged mHC prefill TF32 projection | Separate candidate; the current native mHC AOT path is unchanged. |
| PCIe DMA transport changes | Not our RoCE inference transport. |

Upstream `prefill_mg.py` reuses a gather across head groups of one query, not across
adjacent query tokens. Our native attention likewise maps each block to one query
and sixteen heads. Four adjacent full SWA windows have a union of 131 rows versus
512 independently processed rows, but L2 already captures some traffic reuse.
Explicit query tiling only wins if saved dequantization/staging exceeds additional
register pressure, shared-memory use, masking and synchronization. Profile those
costs before making it a priority; no speedup from query tiling is established.

## Artifacts and rollback

Current containers: `ds41-router2048-target-api-dev` and
`ds41-router2048-spec-api-dev`. Frozen daemon and native runtime are under
`/tmp/ds41-router-serving`; exact create commands are `api-create-commands.json`.
Raw benchmark, native, quality and upstream comparison results are in the same
directory. Hashes and detailed numbers are in [the companion JSON](ds41-router-serving.json).

The prior `ds41-receiveslots2048-{target,spec}-api-dev` containers remain stopped
with their original artifacts for rollback. The four Spark workers are unchanged.
