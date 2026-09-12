# Native retained-frontier admission qualification

The native API now reuses complete retained frontiers: exact prompts and the
committed portion of a previous assistant turn. The token radix, target/dSpark
state restoration, completion publication and capacity-driven eviction run on
`serve-native`. Arbitrary matches inside an edge still require compression-boundary
reuse and bounded SWA replay; final release memory sizing also remains open.

## Correctness and memory pressure

- Both modes match their uncached reference for counting, complete prompt hits,
  and a two-turn access-code conversation. Cache usage fields agree with the
  restored frontier; the follow-up reuses committed assistant tokens as well as
  the original prompt. Raw outputs and every SSE event are retained.
- The initial candidate passes C2/C6/C16 counting, cancellation and replacement.
  The fixed candidate repeats target C2/C6/C16 at a 512-token context reservation
  and dSpark C2/C6/C16 at the normal development reservation, including cancellation
  and replacement. Outputs and token totals match the corresponding serial cases.
- A deliberately small source pool exposed an unnecessary copy reservation:
  three appending owners needed three copies when only two were necessary.
  One owner can now keep the original if every owner appends in that transaction.
  Copies finish before accepted writes. Snapshots and non-appending owners remain
  protected. The same live C16 pressure test failed twice before this fix and
  passes afterward without increasing its reservation.
- Four source ownership/device tests pass with CUDA memcheck reporting zero
  errors. They include divergent prefix lengths, retained snapshots, zero
  acceptance, page exhaustion and reclamation. Two radix/LRU tests pass.

## Controlled short-context performance

Each workload warms both arms, then runs four sequential AB/BA pairs with a
256-token output budget. The original target baseline and candidate share GPU0;
dSpark arms share GPU1. All inference workloads run sequentially across the same
four Spark workers. Both measured candidates use `RUST_LOG=info`, the same native
library and container image. Cache retention is enabled only in the candidate.
Reported decode clocks exclude time through first content; TTFT includes HTTP.
All paired text and token counts match. These are focused regression checks,
not the weighted release suite or long-context results.

| Mode / workload | Baseline tok/s | Candidate tok/s | Change | Baseline TTFT | Candidate TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| target / counting | 43.19 | 43.45 | +0.59% | 88.54 ms | 1.43 ms |
| target / code | 43.04 | 43.33 | +0.66% | 127.24 ms | 1.35 ms |
| spec / counting | 147.08 | 146.90 | -0.12% | 91.89 ms | 1.31 ms |
| spec / code | 126.04 | 125.39 | -0.51% | 128.77 ms | 1.60 ms |

The four-pair dSpark code result had greater candidate variability, so eight
additional AB/BA pairs were recorded: baseline 126.53
versus candidate 126.88 tok/s (+0.28%).
Both runs remain in the evidence; the follow-up does not replace the earlier
samples. This small comparison does not establish a universal no-regression
guarantee. The full release performance matrix remains required.

## Provenance and reproduction

[Machine-readable manifest](release-v1-native-admission.json) records source and
binary hashes, library/model/image identities and numerical summaries.
[Compressed raw evidence](evidence/native-prefix-admission.json.gz) contains
launch commands, requests, all saved SSE events, samples, logs and failures.
Decompress with `gzip -dc docs/evidence/native-prefix-admission.json.gz`.
The initial pressure failures and a startup memory exhaustion are preserved.
The latter came from leaving the temporary pressure coordinator alongside both
normal target arms; stopping it allowed the fixed candidate to start.

The comparison APIs remain on ports 18041/18042. Fixed candidates use
18043/18044. The pressure container used 18045 and is stopped after its check.
These temporary development artifacts do not replace final reproducible image
qualification or publication.

```bash
python3 scripts/qualify-ds41-prefix-api.py \
  --reference-url http://127.0.0.1:18042 \
  --base-url http://127.0.0.1:18044 --output /tmp/prefix.json
python3 scripts/qualify-ds41-concurrent-api.py \
  --base-url http://127.0.0.1:18044 --output /tmp/concurrency.json
python3 scripts/qualify-ds41-prefix-performance.py \
  --reference-url http://127.0.0.1:18042 \
  --candidate-url http://127.0.0.1:18044 --output /tmp/performance.json
```
