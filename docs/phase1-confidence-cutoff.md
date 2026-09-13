# Independent confidence cutoff experiment

`--dspark --dspark-confidence-cutoff P` is an opt-in comparison policy, mutually
exclusive with `--dspark-adaptive`. The default policy is unchanged. P must be
finite and between zero and one. A value of zero retains the full available
draft; larger values require greater cumulative confidence.

Each request converts the existing draft confidence logits with sigmoid, then
retains the longest prefix whose product meets P. It stops at the first failing
prefix. The confidence head is treated as an acceptance estimate; this does not
establish calibration on every workload. One draft token is retained when one
is available, preserving the existing qualified minimum. Output budget and
fixed draft limit still bound proposals. Grammar-constrained requests retain
the existing grammar truncation without confidence selection.

Selection occurs immediately after that lane proposes, independently for each
request. It needs neither the other lane's proposals nor accepted route history.
The existing fixed-mode early Engram preparation is therefore also used here.
Confidence download remains required. Draft generation still produces its
existing proposals before truncation; this does not save draft-generation work.

This experiment does **not** remove the verifier round barrier: both execution
futures still complete before committing either lane and starting the next
round. Full independent progress also needs ownership changes because an active
verifier borrows the shared request bank while commit mutably borrows that bank.
It must preserve cache reservations, cancellation/error drains and shared draft
workspace lifetimes. A different cost equation alone cannot resolve that barrier.

Core tests cover cumulative stopping, low-confidence earlier positions, minimum
length, threshold boundaries, empty drafts and invalid probabilities. The daemon
checks and release build pass. CLI checks reject a missing `--dspark`, conflicting
cost policy, NaN and out-of-range thresholds. An initial host invocation lacked
the Python shared-library path; rerunning with the repository environment wrapper
passed these checks.

Serving comparison completed in `/tmp/ds41-confidence-policy`. It uses the same
frozen Rust binary and qualified narrow FP8 native library for all policies,
with cost-policy controls before and after cutoffs 0.25, 0.5 and 0.75. Each arm
runs three no-thinking code samples, one 32K prefill warmup and mixed C4/C16
completion checks. This is a policy screen, not full release qualification.
Hardware is one RTX PRO 6000 Blackwell at **400 W and standard memory speed**,
four Sparks, five complete local expert layers, 18 × 1,048,576-token KV capacity
and 24 retained snapshots. No builds overlap timed serving. The runner restores
standard serving on completion or failure.

## Initial screen results

| Policy | C1 code median tok/s | C4 mixed tok/s | C16 mixed tok/s |
|---|---:|---:|---:|
| Cost, before | 128.15 | 124.87 | 182.83 |
| Cumulative cutoff 0.25 | 119.02 | 119.67 | 196.35 |
| Cumulative cutoff 0.50 | 117.21 | 114.11 | 190.39 |
| Cumulative cutoff 0.75 | 103.07 | 110.48 | 183.34 |
| Cost, after | 127.86 | 124.57 | 179.38 |

All code structure, prefill-answer and mixed serving completion checks passed.
The mixed checks do not establish semantic quality or identical output tokens.
Each cutoff has only one mixed batch at each concurrency, so its apparent C16
advantage needs repetition before adoption. None dominates the cost policy:
larger cutoffs lose substantial C1/C4 throughput. Defaults remain unchanged.
The service was restored. [Evidence](phase1-confidence-cutoff.json) preserves
commands, artifact hashes, code samples and mixed rates. The next experiment
adjusts the threshold incrementally for predicted expert reuse within each lane.
