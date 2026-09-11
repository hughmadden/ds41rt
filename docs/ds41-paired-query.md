# Prefill query preparation overlap

The paired serving loop now polls the leading chunk's FFN first, then prepares
and executes the following chunk's query while that FFN is pending. Previously
both next-query/Engram preparations ran before the first expert dispatch, and
unbiased join polling could prepare the second attention before dispatching the
first. Independent lane, index and RoCE ownership and the pair cancellation guard
are retained. There is still a barrier at every layer pair; this is not a complete
multi-chunk wavefront scheduler or cross-request C16 serving.

## Isolated comparison

Both selected serial APIs were stopped for the comparison. The previous paired
binary and this candidate each ran eight sequential 16k API requests, with only
one candidate pair resident at a time and no concurrent compilation or GPU jobs.
Two code and two repeated-text requests per mode generated the same 59 counting
tokens. The table uses the second request of each kind and mode. Rates are tok/s;
decode columns are previous/current. No prefix-cache hits occurred.

| Workload / mode | Previous prefill | Query overlap prefill | Change | Decode |
|---|---:|---:|---:|---:|
| code / target | 3148.29 | 3268.40 | +3.82% | 37.43 / 37.48 |
| code / speculative | 3238.81 | 3301.92 | +1.95% | 111.57 / 110.97 |
| repeated / target | 3424.16 | 3483.19 | +1.72% | 37.04 / 37.39 |
| repeated / speculative | 3653.20 | 3678.23 | +0.69% | 113.74 / 113.78 |

This is a modest observed gain, not a statistically established speedup. Across
runs both prefill and first-round dSpark acceptance vary. In this isolated repeat,
the previous paired binary's warm code dSpark decode recovered to about 112 tok/s,
versus about 101 in its initial trial. The earlier serial/paired acceptance trace
therefore does not establish a deterministic regression caused by paired prefill.
A longer output and repeated acceptance traces are needed before attributing
remaining decode differences to kernel cost.

An earlier pass reached 3.12/3.34k code and 3.73/3.74k repeated prefill tok/s;
compilation overlapped its early requests, so it is retained as qualification
and context rather than the controlled performance comparison.

## Qualification and artifacts

Release and test builds pass. Both API lifecycle checks pass, including streaming,
cancellation/recovery and unsupported-sampling rejection. Eight quality cases
preserve each mode's previous text and usage, including the inherited Unicode
format failure and the differing open-ended explanation. The strict quality
script still exits 1; this does not close the quality gate. All eight benchmark
prompt hashes, outputs and usage records match between comparison binaries.

The real distributed encoder component test also checks ordered publication and
bit-identical final residual/pre-state against serial execution for its small
paired fixture. Full serving coverage comes from the API runs above.

Candidate daemon SHA-256: `f818a42252e9b923eafd1289cfedf6b5230b8d5929c128170c5947165f2e2749`.
The native library and Spark worker artifacts are unchanged. Reproduction commands,
raw results and build logs are under `/tmp/ds41-paired-query`; portable records are
in [the evidence JSON](ds41-paired-query.json).

Both candidate APIs are stopped; serial CED is restored on ports 18041/18042.
Keep prefill first: investigate the layer-pair barrier and worker queue/output
pipeline, qualify the attention register-rescale prototype, then revisit expert
M grouping on the resulting real schedule. Category-based dSpark decode tuning
follows this prefill work; counting is not a substitute for the user's supplied
code/math/structured/natural-language category measurements.
