# Five-layer RTX C1 profile

Measured commit `7c0dfcf` with complete routed-expert layers 0–4 on the RTX,
18 × 1,048,576-token KV capacity and 24 retained snapshots. One RTX PRO 6000
Blackwell at a 400 W power limit and standard memory speed; four Sparks retain
all 40 layers. Placement remains opt-in. No builds overlapped measurements.

Three no-thinking code requests per mode, with timing logging disabled:

| Mode | Individual decode tok/s | Median tok/s |
|---|---|---:|
| Target-only | 40.77, 43.89, 43.70 | 43.70 |
| Adaptive dSpark | 124.01, 123.38, 123.74 | 123.74 |

These are current measurements, without a matched remote-only control. They do
not establish a placement speedup. All six quiet requests and six separately
traced requests passed the Python structure checks; this is not a broader
semantic qualification. Prefill, concurrency and startup were not retested here.

Separate host timing traces yielded the following medians. Layer components
are first summed within each complete round, then their medians are computed.
Nested timings overlap and must not be added as independent costs.

| Host timing, ms per complete round | Target-only | dSpark verification |
|---|---:|---:|
| Complete verification | 22.62 | 38.37 |
| Complete scheduler round, including draft when enabled | 23.05 | 41.87 |
| Five local FFNs, including routing and shared experts | 1.08 | 2.13 |
| Local routing, nested within local FFNs | 0.17 | 0.18 |
| Remote receive waits across remaining 35 layers | 4.82 | 17.81 |
| Remote shared experts | 1.83 | 2.00 |
| Queued attention chain across all 40 layers | 5.56 | 5.48 |

The summary includes 636 of 639 target scheduler rounds and 114 of 125 dSpark
rounds: exactly one request, 40 consecutive layers, and one target row or six
verifier rows. It validates exactly five local FFN events and 35 remote FFN and
collection events per included round. Other shapes and partial rounds are
excluded. Host timings include instrumentation and synchronization; they do
not isolate GPU kernel time or prove that a wait can be removed.

Local chaining is a bounded opportunity: even eliminating the entire measured
local FFN region would leave most verification time. Source inspection shows
local routing, shared expert execution and local routed execution still publish
through separate completed stages. A larger queued region should preserve
explicit buffer lifetimes, adaptive route observation and error drains. Measure
its complete verification impact alongside work on the remaining remote waits
and common RTX stages; local kernel launch counts alone are insufficient.

The updated timing summarizer reproduces the previous remote-only trace counts
and timing values exactly. The standard service was restored after all four
runs. [Evidence](phase1-rtx-local-c1-profile.json) includes commands, artifact and
trace hashes, per-layer FFN timings and raw evidence locations.
