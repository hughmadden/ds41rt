# Paired encoder serving candidate

Native serving now constructs a second execution pass and RoCE wave, sharing
immutable weights. It executes encoder chunks in pairs, captures their final
suffix rows in order, publishes source 20, commits both request histories, then
uses the existing bounded decoder replay and dSpark seed. An odd leading chunk
runs serially before reserved pair mode. The second wave resets on request error.
Pair guards revoke participating requests on failure or cancellation.

The frozen candidate ran on ports 18043/18044 while the previous selected serial
CED APIs remained resident on 18041/18042. No simultaneous benchmark jobs ran.
Candidate containers are now stopped and retained; the serial deployment remains
selected. Candidate daemon, exact commands and raw results are under
`/tmp/ds41-paired-serving`; native kernels are unchanged from the CED deployment.
Build completed in 35.13 s. Both API lifecycle qualifiers pass, including
cancellation/recovery. All eight quality cases preserve each mode's previous text
and usage; the existing Unicode-format failure and seven-of-eight paired text
agreement remain. The strict quality script therefore still exits nonzero.

## 16k API results

Sequential C1, 2048-token chunks, 16,410 code or 16,411 repeated-text prompt tokens,
59 generated counting tokens. Prefill rate includes first-content API latency.
All eight prompt hashes, output texts and usage records match the serial baseline.

| Warm workload | Target prefill | dSpark prefill | Target decode | dSpark decode |
|---|---:|---:|---:|---:|
| Code | 3306.95 | 3318.71 | 37.40 | 101.28 |
| Repeated text | 3637.37 | 3734.79 | 37.16 | 114.27 |

Rates are tok/s. Previous serial CED warm prefill was 3027.27/3063.76 for code and
3487.99/3513.19 for repeated text, with approximately 39 target and 115 dSpark
decode tok/s. First candidate code prefill was 2183.23/2377.42 tok/s. The warm
prefill gain is modest and decode regressed, especially on code. These observations
do not isolate the reason: extra resident processes/connections, changed scheduling,
and speculative acceptance need separate measurement. No deployment improvement
is claimed from this candidate as a whole.

The paired component test previously established byte-identical final encoder
residual/pre-state for a small two-chunk fixture. These long counting prompts and
short quality cases are limited end-to-end evidence, not broad model equivalence.

Full results and artifact identity are in [the evidence JSON](ds41-paired-serving.json).

## Follow-up isolation and acceptance

Stopping both serial APIs did not remove the paired candidate's code decode
regression. Separate instrumented spec-only runs show the warm serial request
uses ten verification steps (49 accepted drafts / 50 proposals), while the warm
paired request uses eleven (48 / 55). Their summed verification intervals are
517,539 and 570,252 us respectively. The first round accepts five draft tokens
for warm serial but only one for paired. Cold serial also accepts just one and
uses eleven rounds. These observations explain much of the measured throughput
difference, but do not establish its cause or determinism across repetitions.

Warm host-timed prefill totals are 5.406 s serial versus 4.980 s paired. The paired
trace includes more time attributed to attention and collection, with a barrier
at each layer pair. These are overlapping host intervals, not additive GPU kernel
costs; in particular `finish_us` includes waiting for the other chunk. A GPU/NIC
timeline is still needed to quantify overlap and contention. Raw traces and the
per-step summaries are under `/tmp/ds41-paired-serving/comparison`.

The next scheduling experiment moves the following chunk's query preparation
inside the leading chunk's pending expert execution and uses deterministic
first-branch polling. See [query overlap](ds41-paired-query.md) for its measurements.
Serial CED remains the selected deployment.
