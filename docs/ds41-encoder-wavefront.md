# Independent encoder chunk progress

The two prefill chunks now advance through encoder layers independently. The
following chunk waits for its predecessor's KV publication at each layer, not
for the predecessor's expert result. Each chunk can start its next layer as soon
as its own FFN completes. This removes the previous per-layer join barrier while
retaining ordered source/window publication, suffix capture and history commits.
The pair still joins after all twenty encoder layers; wider chunk scheduling and
C16 admission are not implemented by this change.

Both futures run on the CUDA owner. Shared request/cache access uses synchronous
scoped borrows that end before awaiting expert work or Engram I/O. Per-layer
notifications record completed leading KV publication. Errors or cancellation
drop the other future and revoke both batches before the execution owners can
be reused. Decoder replay and the dSpark seed consume the completed suffix as
before.

## Qualification

The real-model fixture compares the new serving method with serial encoder
execution for 65+65 tokens (crossing the retained 128-token suffix boundary and
an odd compressor carry) and 79+1 tokens (uneven work). Both suffix residual and
pre-state arrays are byte-identical. Ordered commits and decoder replay readiness
pass. A separate first-poll cancellation drops an actively suspended pair, checks
both batches and their shared admission are revoked, then reuses the same owners for those comparisons.
The fixture passes in 6.39 s. Release and test builds pass.

Both API lifecycle qualifiers pass, including cancellation/recovery. All eight
quality cases retain each mode's previous text and usage. The inherited Unicode
format failure and differing open-ended explanation remain, so the strict quality
script still exits 1. These checks do not establish broad model equivalence or
close the release quality gate.

## Isolated 16k comparison

The selected serial CED binary and new binary each ran eight sequential C1 API
requests, with only one API pair resident at a time, no concurrent GPU work and
no compilation during measurements. Contexts contain 16,410 code or 16,411
repeated-text prompt tokens; each generates 59 counting tokens. All prompt hashes,
output text and usage match. No prefix cache hits occurred. The table uses the
second request of each kind/mode. Rates are tok/s, with decode shown serial/new.

| Workload / mode | Serial prefill | Independent prefill | Change | Short decode |
|---|---:|---:|---:|---:|
| code / target | 2966.01 | 3777.77 | +27.4% | 37.65 / 37.48 |
| code / speculative | 3041.21 | 3859.80 | +26.9% | 112.50 / 100.62 |
| repeated / target | 3484.77 | 4087.60 | +17.3% | 37.66 / 37.37 |
| repeated / speculative | 3506.86 | 4127.76 | +17.7% | 114.00 / 112.59 |

The roughly 27% code and 17–18% repeated-text prefill gains are observations from
this comparison, not a statistical guarantee. Short dSpark code throughput still
varies substantially. Earlier traces found a one-round acceptance difference,
but this comparison alone cannot attribute the remaining variation. Longer
counting output is checked separately to reduce the influence of one round.

An initial eight-request run reached warm code 3.68/3.75k and repeated-text
4.11/4.06k prefill tok/s. It also preserved all benchmark outputs and usage.

## Longer decode check

The same 16k code context with counting extended to 200 generates 599 tokens.
Serial and independent scheduling produce identical text, prompt hashes and
usage for both modes. Target decode is 38.00 versus 37.86 tok/s; dSpark decode is
120.40 versus 119.74 tok/s. This substantially reduces the influence of a single
verification round and shows no substantial sustained counting-decode regression
in this comparison. It remains one workload, not category-based decode evaluation.

## Artifacts and remaining work

Daemon SHA-256: `d4ab3445496e13137ef2250076590f370a2e84c9735ca4c3976602cf7576ccb3`.
The live development APIs now run this daemon on ports 18041/18042, in
`ds41-wavefront2048-live-target-api-dev` and
`ds41-wavefront2048-live-spec-api-dev`. Both answer a post-start prompt correctly.
The old serial CED containers are stopped and retained for rollback.
Raw logs, frozen daemon and exact launch commands are under `/tmp/ds41-wavefront`.
[Portable evidence](ds41-encoder-wavefront.json) includes the comparisons and
qualification summaries. `bench-ds41-prefill-api.py` now accepts `--count-to` and
`--max-output-tokens`, retaining the previous prompt and token limit by default.

Next prefill work: overlap across chunk-pair boundaries, worker queue/output
transfer scheduling, and the attention register-rescale prototype. Revisit expert
M grouping after the final schedule is measured. Full C16 serving and broader
category-based decode work remain open.
