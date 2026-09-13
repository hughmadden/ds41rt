# Independent lanes with the original adaptive cost policy

`--dspark --dspark-adaptive --independent-decode-lanes` now selects verification
prefixes using only the issuing lane's requests, accepted routing history and
draft time. The original confidence model, cost coefficients, bounded suffix
search and minimum prefixes are retained. A single-lane forecast makes the
3,168 µs cross-lane term zero. Neither selection nor the next verifier issue
requires the peer's proposals or commit. Admission and retirement still drain
both lanes to preserve the shared prefill and cache ownership boundaries.

One active lane uses a direct single-pass scheduler with the same draft,
selection, verification and commit sequence; it avoids shared-bank async borrows.
There is no paired decode fallback. The current source makes this policy the default when dSpark is enabled.
The joint decode implementation has been removed; `--dspark-fixed` retains a
fixed-length policy for comparisons, still using independent lanes. The former
`--dspark-adaptive` and `--independent-decode-lanes` flags remain accepted as hidden
compatibility spellings. The default-path confirmation passed;
this source change does not replace the published container.

## First matched serving comparison

One RTX PRO 6000 Blackwell at **400 W, standard memory speed**, four unchanged
Sparks, five complete local expert layers, 18 × 1,048,576-token KV capacity and
24 retained snapshots. Both arms use identical frozen Rust/native artifacts;
only `--independent-decode-lanes` differs. No builds overlap measurements.

| Workload | Paired | Independent | Change |
|---|---:|---:|---:|
| C1 code decode, tok/s | 128.38 | 127.73 | −0.5% |
| 32K prefill, tok/s | 7,797.10 | 7,594.24 | −2.6% |
| C2 mixed decode, tok/s | 96.79 | 101.32 | +4.7% |
| C4 mixed decode, tok/s | 109.04 | 109.74 | +0.6% |
| C8 mixed decode, tok/s | 153.89 | 160.87 | +4.5% |
| C16 mixed decode, tok/s | 165.56 | 179.19 | +8.2% |

C1 and prefill are three-sample medians, with one prefill warmup per arm.
Mixed traffic has one batch per concurrency, sequenced C2, C4, C8, C16;
aggregate rates include admission gaps. These are exploratory sequential arms,
not a balanced adoption qualification. The lower prefill measurement remains
unresolved; the unchanged prefill path alone does not establish no regression.

All three C1 outputs match exactly. Mixed output matches are 2/2, 4/4, 3/8,
and 6/16. C8/C16 prose responses and lengths differ, so those throughput changes
are not identical-output comparisons. Every applicable Python structure check
passes; prose quality is not assessed. This screen does not prove broad greedy
output equivalence or broad quality.

Both arms passed the 32,815-token middle needle, complete prompt reuse,
32,821-token retained-turn reuse, eight cancellations interleaved with eight
survivors, and subsequent recovery. The standard service was restored after
both arms. The release build passed before the experiment.

[Evidence](phase1-independent-adaptive.json) retains commands, artifact hashes,
individual C1/prefill samples, matched requests/responses for mixed traffic,
comparison checks and lifecycle summaries. Full lifecycle records remain under
`~/.cache/ds41rt-experiments/independent-adaptive`. The reused comparison script
calls its arms `fixed` and `adaptive`; here they mean **paired adaptive** and
**independent adaptive**, respectively. Neither arm is fixed-length.

The reverse-order confirmation and C16 trace below complete this scheduler
transition screen. Further host-completion work remains; these results do not
establish the full Phase 1 throughput targets.


## Focused C16 scheduling trace

A separate instrumented comparison confirms 206 matched independent verifier
issues/commits and 176 next-round issues before the peer's existing verifier
committed. Lane-local selection took 55 µs median / 158 µs maximum across 203
selections, compared with 240 / 729 µs across 93 paired selections. Counts and
batch sizes differ; this is host-work attribution, not a throughput comparison.

Across 83 independent rounds with eight requests in the lane, median preparation
was 6.471 ms and verification 166.177 ms. These include instrumentation and
first-use effects. Draft replay, head completion/download and commit still use
synchronous host calls on the common CUDA owner thread. They can delay polling
of the peer even without a lane-join barrier. These are candidates for measured
asynchronous completion work after confirming the default scheduler change.


## Default-path reverse-order confirmation

The rebuilt candidate removes the joint decode function and its cross-lane cost
term. A direct single-lane function preserves C1's synchronous numerical sequence;
two active lanes always use independent loops. `--dspark` alone enables adaptive
selection. CLI tests cover the default, fixed opt-out, alternate confidence policy,
legacy spellings and incompatible options; the release build passes.

This run starts the new default candidate first, then the frozen paired baseline.
The native library and workload sequence match the first comparison.

| Workload | Paired control | New default | Change |
|---|---:|---:|---:|
| C1 code decode, tok/s | 127.97 | 128.72 | +0.6% |
| 32K prefill, tok/s | 7,698.76 | 7,791.05 | +1.2% |
| C2 mixed decode, tok/s | 98.50 | 102.43 | +4.0% |
| C4 mixed decode, tok/s | 114.67 | 110.19 | −3.9% |
| C8 mixed decode, tok/s | 161.94 | 155.01 | −4.3% |
| C16 mixed decode, tok/s | 186.36 | 186.89 | +0.3% |

The reverse-order result does not show a consistent prefill loss. Mixed throughput
is variable: C2 improves in both comparisons, C4/C8 do not show a repeatable win,
and the initial C16 gain becomes parity in this control. Retaining the independent
default follows the requested architecture; it is not a claim of universal speedup.

Both arms pass needle, prompt/turn reuse, cancellation/survivor and recovery checks.
The new default also passes four simultaneous high-thinking requests: tool calls
and strict JSON, each as a normal response and SSE stream, with exact request-
specific values. Standard serving was restored after the comparison. A separate
target-only C1 check verifies the simplified direct execution path.


Target-only C1 also passes: three-sample median 43.95 tok/s paired control versus
44.25 tok/s with the direct single-lane function (+0.7%). All three paired outputs
match exactly and pass Python structure checks. Standard serving was restored.
