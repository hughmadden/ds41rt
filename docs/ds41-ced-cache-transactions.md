# CED cache transactions

The backbone cache bank now distinguishes full execution, encoder prefill and
bounded decoder replay. This closes the cache-commit prerequisite for omitting
full decoder work; the API still uses full execution until target-pass integration
is complete.

A fresh request can enter encoder mode with a fixed prompt extent. Encoder batches
start at the published encoder position, commit windows 0–19 and all four global
sources (including source 20), and leave decoder windows empty. A batch cannot
cross the prompt extent, mix phases, or use decode/verification work in a CED phase.

After encoder windows and global sources all reach the prompt end, replay
initialization creates empty decoder windows at `max(0, prompt_end - 128)`.
Replay planning is explicit and uses that decoder position. Replay commits only
windows 20–39; global sources and encoder windows remain unchanged. Published
encoder history stays at the prompt end throughout replay. Once the decoder suffix
is fully committed, ordinary decode/verification planning resumes at that end.

Admission/phase/commit transitions revoke stale batch snapshots through versions.
Window access is restricted to the phase's layer range, and replay cannot request
compressor producer chunks. A failure during decoder-window initialization revokes
the whole admission; component commit failure retains the existing all-request
cleanup path. This does not bypass the target executor's separate full-pass guard:
that executor still needs explicit encoder/source-only/replay entry points.

## Qualification

The Rust state test covers prompt lengths 1, 127, 128, 129, 2048, 16410 and 1048576,
including phase extents, partial replay progress and return to full execution.

The existing real-weight cache-commit test now also executes sixteen CED requests:

- Encoder chunks 64+65, exercising ratio-two compressor carry across an odd end.
- Decoder replay starts at position 1 and commits chunks 63+65.
- Encoder window payloads/scales and all global KV/index pool payloads/scales are
  byte-identical before and after replay. The global pool comparison covers all
  requests; explicit encoder-window byte comparison selects the first request.
- Every request validates its phase-specific device owner histories and reaches
  committed position 129; ordinary next-token planning resumes there.
- Wrong-phase planning, stale snapshots and cross-phase window access are rejected.

The same run retains the former full-pass prefix/packed-byte checks across sixteen
requests and late source-pool exhaustion, request revocation and reclaimed-page
recovery. It passes in 4.29 seconds on RTX GPU1 using the unchanged deployed native
library and official projection weights. The state test also passes. Rust daemon
check and test-binary compilation pass; existing warnings remain.

Raw logs are `/tmp/ds41-ced-bounds/cache-gpu.log`, `cache-state.log`,
`cache-check.log` and `cache-final-build.log`. The selected API/worker binaries are
unchanged. This evidence does not qualify full-model bounded replay or prefill
performance.

Remaining CED integration: construct causal views over already-committed source-20
KV/index state for decoder replay, retain the final encoder/mHC suffix, add target
pass entry points and progress guards, initialize dSpark from decoder suffix taps,
and switch API prefill to the split path with quality/performance qualification.
