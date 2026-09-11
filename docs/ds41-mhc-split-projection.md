# Native split-K mHC projection, 2026-09-11

The selected small-row mHC path now splits each of its 24 FP32 projections
into eight contiguous K partitions. The CuTe AOT projector launches 192
single-warp CTAs per row; the existing CUDA finish kernel reduces the
partials, normalizes, and runs sigmoid/Sinkhorn. This replaces the previous
24-CTA projector, whose lanes each performed 640 sequential FP32 FMAs.
Both paths use two kernels. Rows above 16 retain the previous implementation.

The split changes FP32 summation order. It does not quantize the FP32
projection weights or BF16 activations. This is a qualified numerical
optimization, not a bit-exact replacement.

## Serving and build contracts

- b12x master commit: `907d5fd556fa729752e6c2c51c4bcf434196bbe0`, descending
  from the full upstream merge.
- One AOT specialization, with live rows passed as a runtime grid argument.
  The projector uses Int64 for all pool-scaled addressing.
- AOT loading occurs during mHC planning before capture. Replays perform no
  compilation or allocation.
- Scratch needs 1536 bytes per row. The daemon reuses the existing collapsed
  BF16 hidden allocation (10240 bytes per row); mixes finish consuming
  partials before pre-collapse overwrites it on the same stream.
- Native validation rejects short, misaligned, overflowing, or aliased
  scratch before any launch. Inputs/outputs retain their disjointness checks.
- The original mixes API remains available for component comparison.
  CUDA builds without coordinator AOT retain the original implementation.
- All 133 previous AOT objects are byte-identical. The new projector adds
  one object. The coordinator native library and daemon were rebuilt;
  Spark worker artifacts and RoCE transport are unchanged.

Native library SHA-256:
`cf202558ee672d1c34a273e9c62d09b1ca8f6bad1367f1cae2c63ceedc5f3560`.

Daemon SHA-256:
`3906379bf67cc1e3bc6db4eb6c05284a4ffafca27cdd400968c41c3619236c18`.

## Qualification and component timing

The production native library passed 32 synthetic cases, 774 real cases
across all 86 backbone/dSpark mHC weight sets, and 13 invalid-call checks.
Real cases cover rows 1/6/16 with random, tiny, and large residuals.
Maximum coefficient difference was **1.0729e-6**, within rtol=2e-5,
atol=2e-6. The unchanged larger-row path remained exact.

The durable qualifier poisons output capacity and scratch, verifies untouched
tails, changes residuals and checkpoint weights under captured graphs, and
uses fixed allocations across live counts 1/2/6/16/17/80/256/4096.
It also checks repeated initialization and failure before output writes.
The real-weight query integration fixture passed all 56 cases, including
rebindings, alternate output destinations, and recovery from a partially
enqueued producer. That fixture proves query ownership/order, not full-model
numerical equivalence of newly generated mHC coefficients.

Both native CTests passed. Warm graph component medians on RTX PRO 6000
Blackwell, in microseconds:

| Rows | Previous | Split-K |
|---|---:|---:|
| 1 | 16.41 | 6.43 |
| 2 | 16.41 | 8.22 |
| 6 | 16.41 | 8.27 |
| 16 | 16.70 | 12.32 |
| 17 | 34.88 | 34.87 |
| 80 | 40.79 | 40.80 |
| 256 | 68.00 | 68.00 |
| 4096 | 956.42 | 956.74 |

These are interleaved warm component measurements, not formally admitted
clock/throttle measurements or prefill throughput.

## Full-model behavior and measurements

Ran baseline, candidate, baseline, candidate cycles with the same four
Spark workers, sequentially exercising target-only and dSpark endpoints.
Each cycle includes three counting streams plus JSON, streaming,
cancellation/recovery, and unsupported-sampling checks. All smoke checks
passed.

| Mode | Baseline cycle medians TPS | Candidate cycle medians TPS | Pooled baseline → candidate |
|---|---|---|---|
| Target | 32.01, 31.32 | 32.96, 33.14 | 31.58 → 33.10 (+4.8%) |
| dSpark | 101.66, 103.69 | 108.82, 107.13 | 103.18 → 107.17 (+3.9%) |

The prompt is “Count from 1 to 20, separated by commas. Output only the
numbers.” This predictable single-client workload favors dSpark; these are
development measurements, not representative or release throughput. The
90/270 TPS goals remain open.

First-cycle instrumented medians also decrease: one-row query preparation
102→86 µs and FFN preparation 54→35 µs; six-row query preparation
110→83 µs and FFN preparation 56→33 µs. The candidate trace includes the
quality workload as well as smoke calls, so those stage medians are
diagnostic and are not a matched per-layer experiment. Medians are not additive.

Seven of eight paired quality cases preserve baseline text and usage in
both modes. In the Unicode arithmetic case, target-only output removes one
blank line before the final 36; dSpark preserves the baseline text. Usage
is unchanged. Three additional repetitions per mode on each deployment
confirm this difference is stable. Both calculate 36 correctly, and both
retain the prior failure to obey “output only the number.”

Consequently, strict objectives remain **5/6**, while target/dSpark text
agreement is **7/8**, down from 8/8. The quality script exits 1. This
optimization is selected with that explicit numerical/whitespace change;
broad quality and release equivalence are not established.

## Reproduction and artifacts

Raw builds, AOT export, component samples, API responses, repeated Unicode
responses, server traces, deployment command arrays, and GPU identity snapshot:
`/tmp/ds41-mhc-split`. Compact evidence:
[ds41-mhc-split-projection.json](ds41-mhc-split-projection.json).

Build with the pinned b12x source and the existing SM120 coordinator CMake
configuration. The component qualifier is:

```sh
python python/tools/qualify_v41_hc_split.py \
  --native /audit/cmake/libds41rt_native.so \
  --snapshot /hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
  --output /audit/native-results.json
```

It ran in `ds41rt-coordinator-dev:latest` with CUDA_VISIBLE_DEVICES=1,
the full HF cache mounted at /hf, and the speculative API stopped.
Build the Rust fixture with Python 3.12 selected for PyO3, then run
`v41_query_preparation --ignored --nocapture` with DS41RT_NATIVE_LIB and
DS41RT_V41_SNAPSHOT set to the same library and checkpoint.

The selected live containers are `ds41-mhc-split-target-api-dev` and
`ds41-mhc-split-spec-api-dev`, on ports 18041/18042. The previous
`ds41-query-chain-*` containers remain stopped for rollback.
