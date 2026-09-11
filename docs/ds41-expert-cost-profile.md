# Native expert cost profile

Optional worker instrumentation now separates expert GPU execution, GPU output compaction, host input staging and host output download. Reusable CUDA events bracket the kernel and compactor on the owning wave stream. Reading them uses the existing completion barrier; there are no additional GPU synchronization barriers or per-request event allocations. Events are created only for backbone workers when the `ds41rt::expert_timing` DEBUG target is enabled at wave construction.

The same record includes active-expert count, maximum routed rows for one expert, compact output bytes and the packed resident size of the unique experts touched. That last quantity excludes repeated reads and is **not measured DRAM traffic**. No peak-bandwidth utilization claim follows from these measurements.

Per-request ProtocolV2 receipt messages moved from INFO to DEBUG. Existing opt-in server boundary timings are retained. The new [summary tool](../scripts/summarize-ds41-expert-timing.py) reads both record types and reports counts, medians, nearest-rank p95 and maxima by source, batch size and timing kind.

## Workload and results

The workload is the full-target sixteen-request fixture (80-row prefill, two 16-row decode passes), followed by the eight paired C1 target/dSpark API quality cases. Both APIs share the workers and calls run sequentially. The per-rank samples below include 3,480 one-row, 1,160 six-row and 1,400 eighty-row operations. Ranges are the four ranks' individual medians, not confidence intervals.

| Cost | One-row decode | Six-row verification | Eighty-row prefill |
| --- | ---: | ---: | ---: |
| Expert GPU event interval | 414.6–417.2 µs | 658.1–664.3 µs | 3,145–3,182 µs |
| Compact-output GPU event interval | 3.90–3.94 µs | 3.90–3.94 µs | 23.39–24.38 µs |
| Host input staging | 26–31 µs | 33–47 µs | 62–146 µs |
| Host output download | 10–12 µs | 11–12 µs | 26–30 µs |
| Measured worker execution section | 463–473 µs | 720.5–732 µs | 3,272.5–3,384.5 µs |
| Server execution excluding socket-write time | 0.70–0.80 ms | 1.01–1.15 ms | 3.68–4.08 ms |
| Host socket-write handling | 0.02–0.03 ms | 0.03–0.04 ms | 0.14–0.36 ms |
| Active experts | 6 | 22 | 86 |

GPU event intervals include work between stream events; the expert interval covers the whole b12x routed kernel, including its device routing/packing phase, not just matrix instructions. Host input staging includes the preceding drain and CUDA upload calls. The measured worker section ends after D2H; it excludes earlier request validation/rebinding, response construction and socket handling. Server execution includes those broader costs and queueing; its socket-write metric measures host handling, not physical link transit. Differences between aggregate medians are not exact per-request overhead decompositions.

The persistent model split remains TP4. Each rank sees the same routes, so the active-expert counts agree. The eighty-row workload has substantial sharing: 480 routes but a median of 86 active experts. The observed sharing is workload-dependent; do not substitute the global `routes / 384` average for the active-expert histogram when analyzing reuse and padding.

The instrumentation preserves all three 8,273,920-byte full-target logit arrays byte-for-byte versus the compact-return baseline. All eight paired API text/usage checks and six objective checks pass. This establishes that these instrumentation changes preserve the tested outputs; it does not resolve the separate compact-return versus old-reduction [numerical gap](ds41-compact-return-rollout.md).

[Machine-readable timings, percentile spread and artifact hashes](ds41-expert-cost-profile.json). Raw logs are `/tmp/ds41-expert-timing-{ostrich,dodo,emu,kiwi}.log`; the complete summary, including other row sizes, is `/tmp/ds41-expert-timing-summary.json`. GPU fixture evidence is `/tmp/ds41-timed-target-gpu.log` and `/tmp/ds41-timed-target-output`; live quality results are `/tmp/ds41-timed-quality.json`.

## Using the instrumentation

Workers in this development run are named `ds41-timed-target-worker`, use frozen `/tmp/ds41-timed-artifacts/worker`, and retain the compact native library from `/tmp/ds41-compact-artifacts`. They started with:

```sh
RUST_LOG=warn,ds41rt::expert_timing=debug,ds41_real_tp4_worker::v41_experts::service=info
DS41RT_PROTOCOL_V2_TCP_TIMING=1
```

The fixture-specific service target controls load logs; the stable expert-timing target controls event creation and execution records. For uninstrumented runs, omit the expert-timing DEBUG target and the protocol timing variable before startup. These opt-in event/logging costs must be distinguished from an uninstrumented performance benchmark.

```sh
python3 scripts/summarize-ds41-expert-timing.py /tmp/ds41-expert-timing-*.log \
  --output /tmp/ds41-expert-timing-summary.json
```

## Implications for tuning

The output compactor is a small fraction of expert GPU time. Optimizing it alone will not meet the release targets. Decode has both a roughly 415 µs routed GPU kernel and substantial surrounding dispatch/server work. Forty serial expert intervals at that median alone would exceed the 11.1 ms budget for 90 target tokens/s, even before coordinator attention. This is a diagnostic budget comparison, not a claim that per-layer distributions or future scheduling are identical.

Next compare V4.1-aware decode plans, input sharing and direct routing with the existing fused grouped path, while preserving V4.1's activation/router-weight/BF16 boundaries. The generic b12x direct/decode candidate predicates do not establish support for `silu_v41`; do not blindly enable old tactics. Extend and qualify the CuTeDSL implementation and plan-time policy where required.

For prefill, increase planned capacity and measure expert occupancy, tile padding, actual DRAM/cache traffic and elapsed time under real routing. Keep coordinator-feed and server-boundary measurements beside kernel timings so improvements do not merely move the bottleneck. Track large-prefill and small-decode paths separately; their reuse and launch costs are different.

## Shared-input decode diagnostic

On ostrich, the existing `B12X_DYNAMIC_W4A8_SHARE_INPUT=1` path passed the FP32 oracle (relative L2 < 0.01, cosine > 0.9999), finite/nonzero output and changed-input/changed-routing CUDA graph replay with stable allocated bytes. This check used synthetic full Spark geometry: one row, 384 experts, top six, hidden 5120, local intermediate 576. The source matched pinned b12x master; no kernel source or live worker artifact changed.

Fifteen samples of twenty graph replays gave median 428.54 µs with sharing disabled and 438.96 µs enabled (candidate/baseline 1.0243). The sample distributions overlap. These sequential, warm-cache synthetic runs do **not** establish a regression, release throughput, or real-weight numerical equivalence; the graph also includes final route reduction. They provide no evidence to enable the switch for speed. Keep it disabled and prioritize structural decode task decomposition over this input-packing switch.

[Raw timings, source hashes, exact probe source and reproduction command](ds41-expert-shared-input-probe.json). No direct-routing candidate was tested in this experiment.

## Output-column split prototype

Source inspection identified a small-row parallelism limit: deterministic fused tasks own all intermediate slices and all output columns for one expert/M tile. A one-row top-six request therefore supplies six compute tasks to the 48-SM Spark. This is the task domain derived from source, not a profiler measurement of occupancy.

An isolated CuTeDSL prototype splits each task into five disjoint output-column ranges, expanding that request to thirty compute tasks. Each task repeats FC1 and the V4.1 activation/quantization boundary locally, then reads only its own FC2 columns. Intermediate activations remain in shared memory; FP32 accumulation across intermediate slices retains the original order and each output element has one owner. Queue metadata stays unchanged: the consumer expands and decodes the logical task domain.

| Synthetic Spark graph | Baseline median | Five splits | Ten splits |
| --- | ---: | ---: | ---: |
| One row, 384 experts, top six | 432.39 µs | 288.11 µs | 288.86 µs |
| Six rows, 384 experts, top six | 1,035.59 µs | 886.66 µs | Not tested |

The one-row five-split result repeated at 286.81 and 288.11 µs: approximately 1.50× baseline/candidate speed in this diagnostic. Both row counts passed the existing FP32 oracle thresholds, finite/nonzero checks and changed-input/changed-routing graph replay with stable allocated bytes. Five and ten splits produced exactly the same final changed-input one-row BF16 tensor as the baseline. This exact comparison does not cover FP32 route planes.

The six-row random-routing fixture has different sharing from the live API workload; its timings are not directly comparable to the live 658–664 µs kernel intervals. These graphs also include final route reduction, use synthetic weights and repeated warm-cache routing, and were measured in separate processes. No production speedup or end-to-end TPS change is established. The prototype is an isolated source overlay; live workers and b12x master remain unchanged.

[Raw samples, patch, complete probe source and commands](ds41-expert-output-split-probe.json) preserve the candidate for the next integration step. Before deployment, replace its fixed prototype constant with a validated b12x plan-time configuration, reject unsupported geometry/work sources, preserve runtime-count reuse and scratch ownership, and qualify native AOT execution against real-weight route outputs. Larger verification and prefill capacities need their own selection evidence because repeated FC1 work can outweigh additional parallelism.

## Direct routing combined with output splitting

A follow-up isolated prototype enables the existing nonstreaming direct-route producer for V4.1 Spark geometry with at most six routes, leaving the generic materialized W4A8 decode regime disabled. The planner selects direct routing through the existing candidate predicate. Direct routing gives each route an arithmetic physical tile and bypasses the grouped histogram/prefix work; output splitting independently expands the compute task domain.

One-row synthetic graph medians were 366.03 µs for direct routing alone, and 223.26 µs for direct routing plus five output splits. A second run with additional route-plane/inactive-route checks measured 218.39 µs; its pristine grouped baseline measured 418.24 µs. Thus both changes contribute, and the combined diagnostic is approximately 1.9× faster than its repeated baseline. These remain separate-process warm-cache synthetic measurements, not native worker or API throughput.

The combined candidate passed the FP32 oracle and changed-input graph checks. Setting every route ID to -1 produced zero output; restoring valid IDs recovered the expected output. The changed-input FP32 `[1,6,5120]` route planes and final BF16 `[1,5120]` output matched the pristine grouped baseline exactly. Native AOT, official resident weights, capacity transitions and broader route distributions remain unqualified. Live workers remain unchanged.

[Raw samples, direct-routing patch, probe source and exact comparisons](ds41-expert-direct-split-probe.json). The native exporter must consume the planner's route selection along with a validated output-split configuration; merely changing the public Python predicate does not update the native kernel export.
