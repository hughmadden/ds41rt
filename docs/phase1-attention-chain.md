# Queued attention → projection → FFN preparation experiment

One RTX PRO 6000 Blackwell at **400 W, standard memory speed (no overclock)**,
four unchanged resident Sparks and official model weights. Baseline source:
`8c88ecb`, using the frozen `ds41-phase1-poll-scoped/artifacts` binary/native
library. The candidate changes only Rust execution ordering, without additional
GPU allocations or different kernel arithmetic.

## Ownership and execution

The existing path completes sparse attention, completes output projection, then
completes FFN preparation. The prototype queues all three on the sparse-attention
stream, then drains once before publishing FFN input. Cold projection graph setup
still drains its input copies before the existing private-stream warmup/capture.
Cached projection graphs replay on the chain stream. Existing cache descriptor,
query identity, layer, shape and device checks remain in effect on every call.

A distinct queued sparse view cannot be mistaken for completed sparse output.
The block enters `QueuedFfn` before enqueuing post-mixing and FFN normalization;
only successful chain completion publishes its `FfnInput`. The enclosing scope
drains after success or error, with an unwind guard. On a returned error the
caller resets block state only after the drain. Query and cache/selection views,
projection graph owners, pinned staging and both mHC owners stay alive and
exclusive throughout the synchronous scope. There is no suspension within it.

This removes two steady-state waits per layer. It does not yet combine all three
stages into a single CUDA graph, include query preparation, or remove every
steady-state capture from the full target pass. Those broader A1 criteria remain
open even if this smaller chain qualifies.

## Initial serving observations

The initial three-repeat code comparison measured speculative **117.64 → 120.08
tok/s**, target-only **42.81 → 43.57 tok/s**, with all six paired outputs exact.
These are exploratory sequential measurements.

A subsequent fixed/adaptive mixed serving pass completed all C1, C4/C16, 32K
needle, exact prefix reuse, retained turn, eight cancellations with eight
survivors, and recovery checks. Diagnostic compilation overlapped part of that
pass, so its throughput is **not a clean performance gate**. Fixed C1 code moved
120.33 → 117.04 while adaptive C1 code moved 118.93 → 122.41. Fixed C16 measured
167.83 → 168.93, adaptive C16 192.42 → 186.15. The conflicting results require a
quiet comparison before any decision to retain the production change.

The quiet A–B–B–A comparison completed with three code requests per restart in
both modes and no concurrent compilation. All four arms per mode have identical
requests and outputs. Pooled medians (six samples per version) are speculative
**118.68 → 119.74 tok/s (+0.9%)** and target-only **42.83 → 43.47 tok/s (+1.5%)**.
Per-restart medians in A–B–B–A order are 118.30 / 119.55 / 119.93 / 119.07 for
speculative, and 42.83 / 43.51 / 43.43 / 42.82 for target-only. This is a small
complete-step improvement, not evidence that the broader 90/270 targets are met.

A separate fault binary injected one error after all chain operations were
queued, before FFN publication. A diagnostic marker confirmed exactly one
injection; the streaming response terminated incompletely, and three subsequent
requests in the same process matched the control exactly. The first harness
attempt expected an SSE error object and was corrected to handle the existing
abrupt stream termination. The diagnostic binary is separate from serving.
The quiet prefill/C16 A–B–B–A gate also completed without parallel compilation.
Each fresh coordinator used adaptive dSpark, one warmup plus three measured
32,768-token prefill requests, then one C16 mixed batch. Corresponding prefill
prompt hashes match across all four arms, all 16 responses equal `7`, and every
prefill request has zero cache hits. C16 request payloads also match across arms.

| Arm | Median effective prefill tok/s | C16 aggregate tok/s |
| --- | ---: | ---: |
| A1 | 7,691.56 | 177.27 |
| B1 | 7,848.88 | 184.23 |
| B2 | 7,857.08 | 184.75 |
| A2 | 7,666.57 | 184.40 |

Pooled prefill medians are **7,679 → 7,854 tok/s (+2.3%)**. The median of the two
C16 batches per version is **180.83 → 184.49 tok/s (+2.0%)**. These focused gates
support retaining the chain. Prefill is an HTTP first-content proxy for one
32K cell, not a full matrix; C16 has only two batches per version and does not
assess prose quality. The broader Phase 1 targets remain open. The diagnostic hook is
kept in `phase1-attention-chain-fault.patch`, not in production source.

The release build and the existing FFN cancellation/failure state test pass.
No full release qualification suite has been rerun. Raw artifacts and runners
are under `/tmp/ds41-phase1-attention-chain`, the larger serving pass under
`/tmp/ds41-phase1-attention-chain-serving`, and the quiet comparison under
`/tmp/ds41-phase1-attention-chain-quiet`.

[Machine-readable evidence](phase1-attention-chain.json) retains individual quiet
samples, the compilation-overlapped observations with their limitation, paired
output checks, fault recovery, and artifact hashes. No diagnostic fault hook is
compiled into the serving candidate. Every runner restores standard serving.

## Complete verification attribution

A final candidate trace with the same three code requests and `ds41rt::timing`
enabled contains 123 complete six-row speculative rounds and 636 complete
one-row target rounds. Compared with the earlier current-baseline trace,
median complete verification is **39.33 → 39.06 ms** speculative and
**23.05 → 22.78 ms** target-only. Candidate attention/projection/FFN chain time
is 5.47 ms speculative and 5.56 ms target-only. These are instrumented host wall
timings from sequential traces, not an independent balanced throughput gate;
the quiet C1 comparisons above establish the observed serving improvement.
Raw candidate traces are under `/tmp/ds41-phase1-attention-chain-trace`.

The final clean release rebuild passed after removing the diagnostic hook.
Its ELF code, read-only data and writable data match the qualified frozen binary
exactly; only symbol/string tables and the build-ID note differ. Both whole-file
hashes and the differing section names are recorded in the JSON. The qualified
native library is unchanged.
