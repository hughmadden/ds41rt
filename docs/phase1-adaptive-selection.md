# Bounded adaptive prefix selection

The core selector now explores joint draft suffix removals for up to sixteen
requests, with at most five proposals each. Serving now exposes an experimental
`--dspark-adaptive` switch; see the [pilot status](phase1-adaptive-pilot.md).
The caller supplies calibrated conditional acceptance probabilities and a cost
function that sees all selected prefix lengths together.

For a prefix of length L, expected emitted tokens are
`1 + p1 + p1*p2 + ... + p1*...*pL`: one mandatory correction/bonus, plus accepted
proposals. Draft generation and both verifier lanes must be included in the
cost. Output-budget and grammar limits must bound the supplied prefixes first.

The selector starts at the full prefixes and removes one trailing token at a
time, evaluating every available request's next removal against the updated
joint batch. It retains the best predicted total-output/cost ratio encountered.
The exploration continues through temporary losses, allowing several removals
to cross a kernel capacity threshold. It returns full prefixes on a rate tie,
and never returns a predicted regression relative to the original shape.
This is a bounded greedy search, not an exhaustive global optimum. Its maximum
is 1,281 cost-function calls and it performs no GPU operations. Cost-model CPU
work still needs measurement when integrated.

Four core tests cover conditional probabilities and anchor retention, a joint
bucket crossing that initially loses rate, tie behavior and the work bound,
and rejection of invalid probabilities/costs.

## Corrected-path observations

The complete native V4.1 library was rebuilt from frozen revision `21174ac`,
using its architecture-specific expert/FP8 exports, RDMA, and XGrammar. The
optional legacy coordinator and W8A16 bridges were disabled, as in the existing
native development build. All five registered native tests pass. The current
Rust binary adds only opt-in route observations beyond that revision.

Nine identical code/fable/topic requests at draft limits one and five produce
**9/9 exact paired outputs** with logging disabled. Three-sample median rates:

| Case | One proposal | Five proposals |
|---|---:|---:|
| Code | 64.55 tok/s | 118.11 tok/s |
| Fable | 50.08 tok/s | 46.66 tok/s |
| Topic | 56.22 tok/s | 61.59 tok/s |

These compare fixed policies, not before/after optimization gains. All applicable
objective checks pass. Integrated source does not imply a newly published image;
the standard v1 coordinator was restored after the probes.

A separate instrumented eight-case sample and C1/C2/C4/C8/C16 counting sweep
recorded 54,800 layer-route observations covering 1,370 complete forty-layer
passes. Counting duplicates much of its route history across requests: median
unique experts stay near 22 at 6, 12, 24 and 48 rows per lane. The cost model must
therefore account for actual reuse and must not equate row count with unique
expert weight traffic. These homogeneous batches also do not cover heterogeneous
concurrent tools/code; calibration needs that distinction.

`RUST_LOG=warn,ds41rt::timing=debug,ds41rt::draft_policy=debug,ds41rt::route_policy=debug`
records owner/position pairs, six routes per row, unique experts, and complete
FFN wall time from existing host dispatch data. No worker restart or extra
route download is required. The confidence observation still downloads its
small confidence buffer. Summed FFN times across overlapping lanes are not
whole-pass latency, and unique experts are not measured DRAM traffic.

`scripts/summarize-ds41-route-policy.py TRACE --output JSON` validates route
geometry and unique counts, summarizes row groups, and assembles forty-layer
passes by their exact owner/position sequence. It rejects duplicate layers and
reports incomplete passes separately. [Integrated evidence](phase1-integrated-observation.json)
preserves artifact identities, fixed-policy results, and trace summaries.

## Remaining integration

The experimental path now fits costs, forecasts from accepted routing history,
and selects across both lanes. The first pilot identified graph recapture as a
major missing cost. Retain bounded compatible captures before repeating adaptive
performance checks. Broaden route/confidence validation to heterogeneous and
constrained traffic, measure transfer and Engram-overlap costs, and qualify
retained state, changing concurrency, cancellation and the full Phase 1 targets.
