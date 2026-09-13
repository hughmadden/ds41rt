# Cooperative shared draft completion

Two active decode lanes now yield the CUDA owner while a captured dSpark replay
and its token/confidence downloads finish. The peer can advance target execution
or commit its own requests during that wait. One lane still owns the shared draft
workspace at a time; this does not duplicate draft weights or scratch buffers.
The direct single-active-lane path keeps its synchronous replay.

Each replay holds reservations for the request slots it reads in all three draft
windows. Writes, restoration and recycling reject an overlapping reservation;
commits to disjoint slots remain allowed. Completion polling releases reservations
only after the stream completes. Launch/query errors drain before releasing them,
and chain destruction drains before releasing pending readers or storage.

The scheduler releases its request-bank and draft-runtime borrows before yielding.
A pending proposal is tied to its lane and exact request seeds. A cohort drain
does not cancel an in-flight proposal: both lane futures finish their owned work
before admission/retirement can reuse storage. Queue time waiting for the other
lane's draft workspace is excluded from the adaptive selector's draft cost.

Seed/sampling preparation, metadata upload, first-use graph capture and cache
commit still contain synchronous work. Request retirement/cancellation still
drains the cohort. These remain follow-up work; this change removes the shared
host wait around ordinary draft replay, not every synchronization in the loop.

## Verification

A host test checks overlapping readers, disjoint writable slots and release after
the last reader. A CUDA test seeds two real windows, reserves one request's slot,
rejects its append/recycle, commits the other slot, then releases the reservation
and successfully appends/recycles the first slot. The Rust check and release build
pass. Serving comparisons use one RTX PRO 6000 Blackwell at **400 W and standard
memory speed**, four unchanged Sparks, five RTX expert layers, KV capacity of
18 × 1,048,576 tokens and 24 retained snapshots. Both arms use the same native
library and GPU top-1; the control is `36e0ec6`. No builds overlap timed serving.

| Workload | Synchronous draft | Cooperative draft |
|---|---:|---:|
| C1 dSpark code, tok/s | 130.38 | 129.75 |
| 32K prefill, tok/s | 7,726.37 | 7,765.13 |
| C2 mixed, tok/s | 103.14 | 107.40 |
| C4 mixed, tok/s | 111.58 | 122.22 |
| C8 mixed, tok/s | 175.01 | 171.29 |
| C16 mixed, tok/s | 195.63 | 200.64 |

C1/prefill are three-sample medians. Mixed traffic is one cold batch per
concurrency, including admission gaps. C4 improves 9.5% in this sample, while C8
decreases 2.1%; these exploratory sequential arms do not establish a uniform gain.
Both arms pass needle retrieval, prompt/turn reuse, cancellation/survivor and
recovery checks. The candidate passes four concurrent high-thinking tool and
strict-JSON cases. Standard serving is restored.

[Evidence](phase1-async-draft-completion.json) retains artifact hashes, commands,
samples and checks. Raw results and frozen artifacts are under
`~/.cache/ds41rt-experiments/async-draft`.

The next draft investigation is separate eight-request workspaces per lane with
shared weights and request-owned cache state. The current projection/FFN storage
bucket is 80 rows for both eight and sixteen requests: splitting owners without
addressing that bucket duplicates oversized storage. Audit forty-row capacity and
fixed overhead before allocating two owners; keep each replay cooperative.
