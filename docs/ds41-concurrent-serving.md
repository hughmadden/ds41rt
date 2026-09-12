# Two-lane native API scheduling

The native serving loop now retains up to 16 independent requests instead of
generating each complete response before accepting the next. Requests have their
own tokenizer decoder, output budget, cache lease, draft state and lane assignment.
Each lane holds at most eight requests, so six-row dSpark verification requires
at most 48 selected head rows. Both target head workspaces reserve that capacity.

At each completed round, finished or disconnected requests retire. If the lane
counts differ by more than one, surviving requests move to the lighter lane.
New admissions also choose the lighter lane. No request migrates or retires
inside its target layer stack. Reassignment changes membership rather than
copying persistent KV, Engram or draft state.

The scheduler generates proposals, prepares two target batches and polls their
full executions together. Both executions drain before error cleanup. Each
request's greedy accepted prefix commits to target and draft caches before its
tokens are emitted. Output budgets and EOS apply individually. The target-only
path uses one input row per request through the same scheduler.

## Qualification

The temporary speculative API passes the native API lifecycle checks, including
cancellation recovery. All eight paired quality texts and token usages match the
previous selected speculative API; both retain the inherited Unicode-format
failure, so the strict quality script exits 1.

`scripts/qualify-ds41-concurrent-api.py` compares concurrent C2/C6/C16 counting
outputs and usage with serial runs, with output budgets of 32/64/128/256 tokens.
All pass. A further C16 run cancels three clients and admits four replacements
as requests finish; every surviving and replacement response matches its serial
result. Artifacts are in `/tmp/ds41-scheduler` (`api.json`, `quality.json`,
`concurrency.json` and logs).

The permanent qualifier uses four distinct counting ranges as well as differing
budgets. Target-only and speculative modes both pass this stronger isolation
check, including cancellation and replacement (`target-distinct.json` and
`spec-distinct.json`). Instrumented scheduler records confirm actual 8+8 execution
rounds and subsequent smaller balanced memberships, rather than merely accepting
16 HTTP connections. These records are summarized in
[the measurement artifact](ds41-concurrent-serving.json).

`scripts/bench-ds41-concurrent-api.py` measures repeated 599-token counting
responses. Aggregate throughput uses the interval from the earliest first content
to the last completion, including admission gaps; per-stream throughput excludes
each stream's first-content wait. This is a short-prompt counting measurement,
not the previous 16k context test or a broad category benchmark.

The second of two speculative runs per concurrency measured:

| Requests | Aggregate tokens/s | Mean per-stream tokens/s |
|---|---:|---:|
| 1 | 126.71 | 126.71 |
| 2 | 152.20 | 76.47 |
| 6 | 310.34 | 52.59 |
| 16 | 605.72 | 39.29 |

All responses preserve the C1 text and token usage. Raw observations are in
`/tmp/ds41-scheduler/bench.json`. This establishes scaling for counting with shared
prompt/routing patterns, not general workload scaling. Target-only API lifecycle
and C2/C6/C16 cancellation/replacement checks also pass.

## Remaining scheduling work

Prefill currently owns both lanes and pauses active decode while it runs. Both
target lanes also meet at a full-round barrier, and proposal generation precedes
target execution. Mixed prefill scheduling, independent lane progress and overlap
of drafting with another lane's FFN remain optimization opportunities.

Balancing currently uses request counts, not measured lane cost. Slow-reader
backpressure isolation, transport failure injection with concurrent survivors,
long-context concurrent quality and sustained serving remain unqualified. This
implementation establishes concurrent API execution, not the complete release
or the 90/270 TPS performance targets.

Temporary qualification APIs were stopped afterward. Normal ports 18041/18042
still use the preceding frozen runtime; this change has not been selected for
production deployment. The final error-path cleanup explicitly retains uncommitted
batches for lane reset even when execution has already invalidated their metadata.
