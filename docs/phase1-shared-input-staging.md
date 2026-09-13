# Shared-expert input staging: rejected

Queuing shared-expert input copies with their consumer did not establish a
serving improvement without concurrency regressions. Both candidates are
rejected; production retains the original completed D2D copy.

All runs used one RTX PRO 6000 Blackwell with a **400 W power limit and standard
memory speed** (14,001 MHz maximum, 13,365 MHz observed while loaded), four
unchanged Sparks, five complete local routed-expert layers, C16 admission,
18 × 1,048,576-token KV capacity and 24 retained snapshots. Code runs disabled
thinking. No builds overlapped benchmarks. The baseline is `b234e1f`, using the
unchanged runtime artifact from the five-layer C1 profile.

The first candidate queued the input copy on the shared-expert stream before
warmup or graph replay, preserving the final drain and draining on preparation
failure. It applied at all live row counts. ABBA results:

| Metric | Completed copy | Queued copy |
|---|---:|---:|
| Target-only code median tok/s, six requests each | 43.64 | 43.65 |
| Adaptive dSpark code median tok/s, six requests each | 120.89 | 120.97 |
| 32K prefill median tok/s, six measured requests each | 7,746.82 | 7,827.35 |
| C4 aggregate median tok/s, two batches each | 118.77 | 128.29 |
| C16 aggregate median tok/s, two batches each | 183.17 | 180.25 |

The C4 gain came with a C16 loss, while C1 was unchanged. A second candidate
restricted queued copies to at most 16 live rows and kept completed copies for
larger batches and prefill. The scheduler balances requests between two lanes;
four requests therefore fit within this bound at up to six verifier rows each.

Two separate ABBA comparisons of that restricted candidate followed. The cold
comparison began mixed traffic directly after startup. The warm comparison
used one code request and one 32K request before mixed traffic in every arm.
These scopes must not be pooled or compared as a cold-to-warm speedup.

| Restricted candidate, aggregate tok/s | Baseline C4 | Candidate C4 | Baseline C16 | Candidate C16 |
|---|---:|---:|---:|---:|
| Cold, median of two batches each | 114.07 | 114.52 | 174.89 | 186.49 |
| Matched warmup, median of two batches each | 124.11 | 122.61 | 189.71 | 177.79 |

Individual warm C4 candidate batches ranged from 116.62 to 128.59 tok/s. All
12 C4 batches across these experiments produced identical per-request response
hashes and lengths, so changed answers do not explain the variation. The
measurements include admission gaps and scheduler behavior; they do not isolate
kernel speed. The restricted candidate failed to establish a reliable gain
and also regressed warm C16. Neither a lower synchronization count nor the cold
C16 result is sufficient to adopt it.

Both candidates passed the real-weight GPU test: 32 exact final-output
comparisons against independently executed shared experts, with changed layers,
inputs, row counts 1/6/16/48/80/256, poisoned input/output storage and retained
small-shape graph handles. Each case also exercised a graph-preparation error
after queuing a copy, checked unpublished output and confirmed recovery.

The unrestricted serving experiment passed 24 code structure checks, all
prefill cache checks, and all four mixed/lifecycle runs. All 16 prefill responses
were `7`, with 32,768 new tokens and zero cache hits. Lifecycle checks covered
needle retrieval, prompt reuse, retained turns, cancellation, survivors and
recovery. Restricted follow-ups checked mixed response completion/nonemptiness;
the warm follow-up also passed its four code and four prefill warmup checks.
These are focused regression probes, not broad semantic qualification.

The standard service and original source were restored, and the restored release
build passed. The next investigation should measure the actual small-row dense
projection kernels from A3; this experiment gives no C1 justification for
repeating the same shared-copy change.

[Evidence](phase1-shared-input-staging.json) records all arms, commands, hashes,
scope and raw locations. The [unrestricted patch](phase1-shared-input-staging-candidate.patch)
and [restricted patch](phase1-shared-input-small-candidate.patch) preserve both
tested candidates without enabling them.
