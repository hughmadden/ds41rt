# Coordinator-owned RoCE QPs, 2026-09-11

The V4.1 coordinator now posts requests and polls all four persistent QPs on
its inference thread. Previously it submitted each request to a separate
QP worker and received frames through cross-thread channels. Spark workers
already execute and poll on their GPU-owner thread and are unchanged here.

The change removes four coordinator QP worker threads. All four requests are
posted before the shared RTX FFN starts; response polling follows that work.
The existing native session owns registered request/response rings, and the
existing assembler validates frames and recycles pinned response storage.
Local channels still adapt that assembler, but no worker receives or polls
those channels. This is not a zero-copy or allocation-free transport claim.

## Ownership and failures

- One admitted pending wave exclusively borrows the transport owner.
- QP creation and capacity growth use the existing TCP bootstrap. Inference
  payloads and responses remain RoCE only.
- Polling rotates through all ranks and yields after a 250 µs quantum when
  responses remain pending. Each native CQ poll is nonblocking; the request
  retains a timeout deadline.
- Identity, row coverage, dtype, rank, duplicate/stale-frame, and final-frame
  validation remain in the existing assembler and V4.1 receiver.
- A partial dispatch failure resets all sessions. Dropping a pending wave or
  failing its receive sink also resets them before reuse. No partial request
  is automatically replayed.
- Payload sinks consume/copy each frame before the next poll. Coordinator
  upload guards retain their existing stream-drain and unpublished-output
  behavior on failure.

## Qualification

Eight V4.1 protocol tests pass. A new ignored live fixture exercises four
real Spark workers with FP8 wire activations and compact BF16 responses:

- Rows 1, 6, 16, 80, then 1 again, to cover capacity growth and reuse.
- Three successful waves for each shape, with finite, nonzero rank planes.
- Abandon an already-posted wave before receiving, then reconnect and replay.
- Inject a sink failure after a response arrives, then reconnect and replay.
- All successful replay planes remain byte-identical after both failures.

This is 15 completed waves, five abandoned dispatches, and five injected
sink failures. The real fixture passed in 2.14 seconds. Run it with idle
workers, the native library and RDMA devices available:

```sh
DS41RT_LIVE_ROCE_PEERS=10.55.0.1:19441,10.55.0.2:19441,10.55.0.3:19441,10.55.0.4:19441 \
  ds41rt_transport-22496a1a47e4e426 \
  local_qps_replay_cancel_and_recover_live --ignored --nocapture
```

Build that test executable using `cargo test --release -p ds41rt-transport
v41_expert --lib --no-run` from rust/, with Python 3.12 selected for PyO3.
The actual run used the coordinator image with host networking, infiniband,
unlimited memlock, CUDA_VISIBLE_DEVICES=1, and DS41RT_NATIVE_LIB pointing to
`/tmp/ds41-mhc-split/cmake/libds41rt_native.so` mounted at /audit/cmake.

Both API smoke suites pass, including streaming, cancellation/recovery, and
unsupported-sampling rejection. Repeating all eight quality cases with
coordinator timing/debug disabled also preserves text and usage. All eight
quality cases preserve text and
usage exactly against `d0b1afb`. The existing strict score remains 5/6 and
cross-mode text agreement remains 7/8 because of the previously documented
Unicode formatting/blank-line difference. This is not broad quality approval.

## Measurements

Sequential short counting runs on the same four workers:

| Coordinator logging | Mode | Previous TPS | Local QPs TPS |
|---|---|---:|---:|
| Development timing enabled | Target | 34.03 | 35.07, 35.12 |
| Development timing enabled | dSpark | 108.29 | 111.78, 111.77 |
| Coordinator timing/debug disabled | Target | 35.60 | 36.46 |
| Coordinator timing/debug disabled | dSpark | 110.29 | 113.40 |

Each entry is the median of three counting streams. The instrumented order
was candidate, baseline, candidate; the quiet order was baseline, candidate.
Quiet mode sets RUST_LOG=info and DS41RT_PROTOCOL_V2_TCP_TIMING=0 on the
coordinator only; worker instrumentation is unchanged. The legacy environment
variable name does not indicate TCP inference.

The first quiet comparison improves target throughput by 2.4% and dSpark by
2.8%. A final source rebuild (including the added ignored fixture) produced a
different daemon hash; its frozen artifact was separately deployed and passed
both smoke suites and all eight baseline-preserving quality comparisons.
Its quiet medians were **36.28 target / 113.75 dSpark TPS**, consistent with
the initial artifact. Both binary identities and samples are in the manifest.
This predictable C1 prompt favors speculation. Clock/throttle admission was
not enforced; these are development measurements, not representative
throughput or the 90/270 TPS release gates. No isolated link-latency or DRAM
bandwidth result follows from them.

Instrumented first-cycle medians show only a modest expert-stage change:
one-row expert phase 319→308 µs against the preceding mHC deployment trace,
with receive 199→198 µs. Those traces mix smoke and quality workloads and are
not a matched per-layer comparison. Removing QP threads also changes CPU
scheduling and logging contention; the whole throughput gain must not be
attributed to pure wire latency.

## Artifacts and deployment

The native library and all Spark binaries are unchanged. New daemon SHA-256:
`69c868eb530d14555d5fdddaab6fc66d7a9ec011112e078175b06eabd212062b`.
Native SHA-256:
`cf202558ee672d1c34a273e9c62d09b1ca8f6bad1367f1cae2c63ceedc5f3560`.

Raw test/build logs, API responses, traces, command arrays, and GPU identity
snapshot are in `/tmp/ds41-local-client`. Compact results:
[ds41-local-client-qp.json](ds41-local-client-qp.json).

The live APIs are `ds41-local-client-quiet-candidate-target-api-dev` and
`ds41-local-client-quiet-candidate-spec-api-dev` on ports 18041/18042. The
instrumented `ds41-local-client-{target,spec}-api-dev` pair is stopped and
available for profiling. Previous `ds41-mhc-split-*` containers remain stopped
for rollback. Actual candidate process inspection found no `verbs` QP threads.
