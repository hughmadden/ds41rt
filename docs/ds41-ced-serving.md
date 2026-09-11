# Bounded CED prefill serving

The native target and speculative APIs now execute encoder layers 0–19 over the
entire prompt, produce global source 20 at the encoder boundary, and replay only
the final 128 encoder rows through decoder layers 20–39. Normal decode resumes
all forty layers. Decoder attention reads committed encoder global KV with causal
masking and a bounded local-window floor. Engram history advances during encoding
only; dSpark is seeded from the completed decoder suffix.

This is intentionally approximate bounded replay, not mathematical equivalence to
full-prompt decoder execution. Broader quality, exact replay, multi-turn prefix
reuse and C16 scheduling remain release gates.

The retained encoder residual and pre-state occupy at most 5,244,928 device bytes
per admitted prompt. Only chunks intersecting the final suffix are copied. The
current C1 API allocates this owner at admission; a concurrency pool is still
needed for the C16 scheduler. Phase-specific progress checks prevent committing
an incomplete encoder source or decoder pass. Minimum preallocated lane capacity
now covers the 128-row replay even with the default 80-row encoder chunks.

## Live measurements

Sequential C1 requests, 2048-token encoder chunks, four unchanged TP4 RoCE workers,
16,410 code or 16,411 repeated-text prompt tokens, and 59 completion tokens:

| Warm workload | Target prefill tok/s | Speculative prefill tok/s | Target decode tok/s | Speculative decode tok/s |
|---|---:|---:|---:|---:|
| Code | 3027.27 | 3063.76 | 39.15 | 114.97 |
| Repeated text | 3487.99 | 3513.19 | 39.16 | 115.48 |

Prefill is prompt tokens divided by first-content latency, including API overhead.
The first code requests reached 2418.48 and 2584.31 tok/s. Warm prefill improves
about 1.9× over the previous packed-KV deployment. All eight benchmark texts,
usage records and prompt hashes match that deployment. All eight quality cases
also preserve each mode's previous text and usage; the existing Unicode-format
failure persists (five of six objective checks pass), and target/speculative
text equality remains seven of eight. These checks do not establish broad quality.

Native CPU/CUDA self-tests pass (2/2), as do encoder-suffix GPU byte checks across
1/127/128/129/257/2049-token boundaries, phase-progress tests, capacity checks and
both API lifecycle qualifiers (streaming, cancellation/recovery, validation).

## Execution profile

Two instrumented target requests confirm nine encoder chunks per layer 0–19 and
one 128-row decoder replay per layer 20–39: 330,760 token-layer executions versus
656,400 for full-prompt forty-layer execution. Second-request totals:

| Host interval | Time |
|---|---:|
| Encoder steps | 5.323 s |
| Decoder replay | 0.132 s |
| Expert response wait/handling | 2.548 s |
| Sparse attention | 1.619 s |
| Expert dispatch/enqueue | 0.309 s |
| Router/request preparation | 0.125 s |
| Return reduction/final stream drain | 0.106 s |
| Engram between-layer work | 0.042 s |

These intervals are nested and may overlap remote work; they are not additive
kernel times or measurements of NIC latency. Encoder chunks still execute
serially. Restoring scheduling overlap is the next investigation, followed by
expert ordering/group-size retuning against the resulting workload.

## Deployment and evidence

Selected containers are `ds41-ced2048-target-api-dev` (GPU0, port 18041) and
`ds41-ced2048-spec-api-dev` (GPU1, port 18042). The temporary profile API is stopped;
both selected APIs were restored and returned HTTP 200 from `/v1/models`.
Previous `ds41-packedkv2048-{target,spec}-api-dev` containers remain stopped for
rollback. Worker containers and artifacts are unchanged.

Frozen daemon/native hashes, full benchmark and quality output, and both trace
aggregations are in [the evidence JSON](ds41-ced-serving.json). Exact container
commands, raw logs and binaries are under `/tmp/ds41-ced-serving`. The b12x pin
remains `e53fd5a1f4a8380377fef91a5b3e69414f192223`.
