# Sparse attention shared layout and native Release builds

The sparse attention kernel now pads shared KV rows to 520 BF16 elements, accumulator scratch rows to 516 FP32 elements, and probability rows to 80 BF16 elements. Probabilities remain in explicit registers until accumulator rescaling finishes, then reuse the scratch buffer. Barriers separate each use. Dynamic shared storage falls from 101,056 to 100,288 bytes; static shared storage is unchanged at 1,024 bytes. The 64-key softmax tiles, arithmetic order, FP8 cache representation and public launch ABI are unchanged.

The KV descriptor is a const grid-constant kernel parameter and the locator takes an inlined const reference. This avoids copying the descriptor into each thread's local memory. Explicit probability carries avoid a dynamically indexed local array. The final full-library attention object uses 130 registers, zero stack bytes, and zero local-memory load/store instructions. The baseline standalone control used 130 registers and a 120-byte stack, with local load/store instructions. This is static compiled-object evidence, not a measurement of runtime DRAM traffic or bank-conflict counters.

## Selection and qualification

The first padded prototype improved diagnostic timings by about 10–13%, but increased register and stack usage. Wider KV padding and probability-only padding were less effective. Straight loop unrolling removed the added stack but lost performance. The selected explicit-register/grid-constant version retains about 8–10% improvement in the sampled cases while eliminating local memory instructions. Rejected variants remain only in `/tmp/ds41-attention-pad`; no tuning flags were added to serving.

An approximate-exponential experiment produced relative L2 differences up to about 6e-5 and no useful performance improvement. It was rejected. The selected version retains `expf` and passes bit-exact comparisons.

The actual selected full native library passes all 180 cases in `python/tools/qualify_v41_sparse_skip.py`: fifty aligned, fifty unaligned, fifty window-only/unaligned and thirty at rows 128/256/4096. They cover empty/late/gapped/full selections, stale metadata, private sources with strides one and two, zero/mixed scales and captured replay with changing inputs. This is exact comparison against the already-qualified native attention implementation, not a new full-model reference-logit qualification. Native/CUDA CTest both pass.

For full selections, the selected full library measured 0.628 ms versus 0.682 ms at 128 rows, 1.237 versus 1.337 ms at 256 rows, and 18.658 versus 20.093 ms at 4096 rows. These are diagnostic 20-replay event timings with baseline then candidate ordering, not a clock-controlled statistical benchmark. An unchanged-source standalone build matched the deployed baseline's performance, separating the layout experiment from standalone compiler flags.

## Build configuration

Single-config CMake builds now default to Release when no build type is selected. Explicit Debug/RelWithDebInfo choices and multi-config generators are preserved. Both WIP and release artifact scripts explicitly request Release, matching their release Rust build. Shell syntax checks pass, the actual native build cache records Release, and a separate explicit-Debug configure preserves Debug.

The prior deployed native cache had an empty build type. A Release-only target control retained the original attention kernel and measured 13.102 s to first content at 16,411 prompt tokens (1252.6 effective tokens/s), versus the preceding deployment's 13.066 s (1256.0). No standalone end-to-end Release speedup is claimed.

## Stage attribution and next work

The fresh post-index target trace has 17 prefill steps and 680 layer executions. Index production/selection fell to 0.210 s from 3.613 s before the CuTe scorer. Sparse attention remained 3.280 s. The expert stage accounted for 8.547 s, including routing/request preparation 1.314 s, dispatch 1.392 s, shared FFN 0.164 s and collection 5.675 s. Collection includes upload 1.012 s and wait/progress 4.519 s. These are nested totals; response wait includes remote compute and cannot be interpreted as isolated RoCE latency. Cold setup and instrumentation are included.

The next prefill work should assess larger coordinator/worker batches and expert/coordinator costs, using representative prompts as well as the existing repeated-filler status workload. Further attention work should consider a CuTe implementation with explicit accumulator layout and query reuse, allowing register rescaling without the shared accumulator round trip. The 2–3k checkpoint, original 8k prefill target, decode targets and C16 scheduling remain open.

## Deployment

Selected library SHA256: `07db52dec294d18ee98e44b553a7c442a43cdabb65ff8103690572fc4eeaa42b`. Release-only control SHA256: `a07847235fe213aaeb4394a2182fcb2f6b52fef48eae41b07ad9e2f557e1bafa`. The coordinator daemon remains `/tmp/ds41-index-aot/daemon`, SHA256 `6c49103accaaf91e3ca4096830e3d550baa4ef0356895ff2ea214c26c457bbba`; b12x remains pinned to `e5343cfb6874fecb9fe012a8c56e79ffdfca831f`.

Selected APIs are `ds41-attention-pad-{target,spec}-api-dev`; four `ds41-direct-worker` expert services retain their preceding artifacts and RoCE transport. Native artifacts, exact launch arrays, comparisons and raw API results are under `/tmp/ds41-attention-pad`. Release control artifacts are under `/tmp/ds41-attention-release-control`, and the fresh stage trace under `/tmp/ds41-post-index-profile`. Previous APIs and the temporary control remain stopped for rollback.

All eight paired API quality cases preserve baseline text and token usage. The inherited Unicode formatting failure and 7/8 cross-mode agreement remain; broad quality remains unqualified.

[Recorded measurements](ds41-attention-shared-layout.json).

## Live API status

The same sequential C1 repeated-`amber` filler/counting workload produced 59 completion tokens with zero prefix-cache hits. Effective prefill includes API/tokenization/first-output overhead. Decode uses 58 intervals between first content and finish.

| Prompt tokens | Mode | First content | Effective prefill | Decode |
|---:|---|---:|---:|---:|
| 3977 | target | 3.911 s | 1016.8 tokens/s | 26.20 tokens/s |
| 3977 | speculative | 3.872 s | 1027.0 tokens/s | 95.37 tokens/s |
| 16411 | target | 12.857 s | 1276.4 tokens/s | 25.92 tokens/s |
| 16411 | speculative | 12.764 s | 1285.7 tokens/s | 94.45 tokens/s |

The preceding 16k run measured 1256.0/1270.0 effective tokens/s. The observed API change is only about 1–2%; a single sequential run does not establish a precise speedup or confidence interval. Both context sizes/modes preserve baseline counting text and usage. Component timing and zero-local-storage evidence are stronger than the small endpoint delta.

Both API smoke suites pass streaming, cancellation/recovery and unsupported-sampling rejection.
