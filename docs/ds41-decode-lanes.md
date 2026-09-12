# Decode lane ownership and cooperative progress

Decode and concurrent serving are the current priorities; prefill remains a
regression check. Requests stay in their assigned execution lane for the entire
token step or dSpark verification round. Only after accepted inputs are committed
may surviving requests change lanes. New admissions go to the lighter lane;
retirement can trigger rebalancing. Moving a request changes batch membership,
not its persistent KV or Engram identity. dSpark state must also remain attached
to the request, not the lane.

## Real-model execution qualification

The distributed target fixture now supports
`DS41RT_TARGET_PASS_DECODE_LANES=1`. It loads official coordinator weights and
uses the four resident Spark expert workers over RoCE. Two independent target
owners process batches of 1, 3 and 8 requests each: C2, C6 and C16.

Each case compares serial execution with cooperative overlapping execution for
an eight-token full-forward prompt and three decode steps. After the first decode
step, the larger cases retire two requests from one lane and migrate a survivor
from the other lane. At the following boundary the two memberships swap owners.
Every selected FP32 logit is byte-identical to the serial execution, and each
surviving request's committed position is checked after every step. All cases
passed both before and after the polling change below.

The fixture now also loads the serving dSpark runtime, admits the same request
identities, and commits each target batch into its three draft caches. It checks
all three draft-cache identities and committed positions after every step,
including after retirement and migration. Duplicate admission and capacity
overflow are rejected. Repeating the serial and overlapping runs reuses released
slots. This extended fixture passes in 10.91 seconds.

The serving draft runtime now maps request IDs to independent cache leases and
RNG state. Its batched commit resolves leases in target-batch order, independently
of execution-lane assignment. The single-request serving path uses this same
commit implementation. Shared execution workspaces and captured proposal graphs
remain separate from persistent request state.

A temporary release API using the request-indexed runtime passed the native API
lifecycle check, including cancellation recovery. All eight comparison prompts
preserved text and token usage against the selected speculative API. The strict
quality checker still exits 1 because both versions fail the inherited Unicode
format check; this is not full quality qualification. Artifacts are in
`/tmp/ds41-draft-requests/{component.log,api.json,quality.json}`. The temporary API
was stopped after validation; the normal selected APIs were left unchanged.

The draft runtime now generates proposals for up to 16 request identities in
caller order. Requests with short histories or one remaining output token use
their anchor alone; eligible requests form a compact draft batch. The packed
`[6, request_count]` output is unpacked back into caller order. Request RNGs follow
that same order. A single 16-request workspace retains graph executables lazily
by active count, so retirement does not require recapturing a previously used
shape or duplicating execution workspaces.

The extended fixture generates proposals after every committed step, including
mixed one-token budgets, retirement and membership swaps. Both target logits and
draft token lists match between serial and overlapping runs at C2/C6/C16. It
passes in 11.40 seconds; artifacts are in `/tmp/ds41-batch-draft/component.log`.
The release candidate also passes the native C1 API lifecycle check and preserves
all eight comparison texts and token usages against the selected speculative API.
The inherited Unicode format failure remains in both versions. Candidate API and
quality records are in the same directory; the temporary API was stopped afterward.

This fixture covers target execution, batched proposals and target/draft cache
ownership across retirement and migration. It does not qualify HTTP admission, mixed prefill scheduling,
concurrent cancellation or production C16 throughput. Production APIs remain on
their previously selected frozen binaries and still serialize whole requests.

## Cooperative transport progress

Previously, an unfinished FFN receive could busy-poll for 250 microseconds before
the other lane was polled. Short decode FFNs can finish within that interval,
preventing useful overlap despite two independent execution owners.

`receive_owned` now yields once after its first incomplete CQ poll, then resumes
the existing bounded polling quantum. An already completed response does not
yield. This yields execution opportunity without migrating or aborting the
waiting request. Response ownership, reduction and QP teardown rules are unchanged.
The live RoCE atomic-prefill recovery test passes through 4096 rows, including
abandoned dispatch and sink failure recovery (32.70 seconds).

Artifacts are under `/tmp/ds41-decode-lanes`: `command.json`, `component.log`
(before), `early-yield.log` (after), and `transport-live.log`. These are debug
correctness fixtures with serial cases preceding overlapping cases; their timings
are diagnostic, not a controlled speedup measurement. Release API latency and
throughput must be measured before selecting the transport change for deployment.

## Decode profile guiding the next work

The selected frozen runtime was profiled on two requests per mode with 16k code
context and a 599-token counting output. Last-request logs are retained at
`/tmp/ds41-decode-profile/{target,speculative}/server.log` and benchmark results
at the corresponding `api.json`. APIs were restored after profiling.

Target-only averages 26.35 ms per step (598 measured steps): 10.37 ms in the outer
expert stages, 6.68 ms attention and 3.37 ms query preparation. dSpark averages
50.79 ms per round (100 rounds), including 7.36 ms drafting. Five proposals yield
4.99 accepted drafts and 5.98 emitted tokens on average. Outer expert stages account
for 25.52 ms per verification round. Nested timings overlap and must not be summed
indiscriminately. The counting fixture already has almost maximal acceptance;
reducing round latency is the immediate opportunity for that workload.

Outstanding serving work: replace whole-request serialization with per-request
generation state, integrate batched proposals and verification, size head workspaces for
verification batches, overlap the two lanes and rebalance at committed boundaries.
Admission, retirement, cancellation and lane reuse then need API-level C1–C16
qualification and simultaneous aggregate/per-stream measurements.
