# Packed FP8 KV conversion

The native sparse-attention kernel now converts two FP8 values to BF16 and applies
their shared scale with packed instructions on SM120a. This adapts the useful
conversion idea from the merged upstream attention work to our FP8 cache format.
The generic architecture image retains the existing scalar implementation.

This changes staging instructions only. Query mapping, cache layout, selection,
softmax, split policy, scratch and transport are unchanged. Scale bytes 0 and 255
retain their previous decoding, including infinity for 255; the upstream mixed
cache helper's scale conversion is not substituted blindly.

## Measurements

Same frozen router daemon, four unchanged RoCE workers, batch 2048, sequential C1,
16,410 code or 16,411 repeated-text prompt tokens, greedy counting with 59 output
tokens and no prefix-cache hits. Second requests per kind/mode:

| Prompt | Mode | Previous prefill tok/s | Packed prefill tok/s | TTFT | Decode tok/s |
|---|---|---:|---:|---:|---:|
| Code | Target | 1554 | 1576 | 10.409 s | 39.18 |
| Code | dSpark | 1559 | 1601 | 10.253 s | 114.40 |
| Repeated | Target | 1776 | 1825 | 8.991 s | 39.11 |
| Repeated | dSpark | 1787 | 1836 | 8.939 s | 115.71 |

This is roughly a 1–3% end-to-end prefill improvement. No decode speedup is claimed.
All eight benchmark texts, usage records and prompt hashes match the previous
router deployment. All eight quality-case texts and usage records match per mode.
The existing Unicode instruction-format failure remains, and the explanation
still differs between target and dSpark: strict paired equality is 7/8 and the
quality script exits 1. Both API lifecycle qualifiers pass.

The full-selection synthetic kernel fixture uses 128 window plus 512 compressed
keys. With six alternating baseline/candidate timings, median microseconds are:

| Query rows | Scalar staging | Packed staging |
|---:|---:|---:|
| 80 | 416.7 | 384.4 |
| 256 | 1225.3 | 1129.3 |
| 1024 | 4592.8 | 4248.5 |
| 4096 | 18506.3 | 17109.1 |

These highly shared cache-resident fixtures demonstrate lower conversion cost,
not a measured benefit from sharing tiles across queries. Both architecture
images retain the same register/shared-memory resources and zero stack/local
storage. The native library grows by 122,896 bytes.

## Validation

- Exhaustive comparison of all 65,536 FP8 pairs and all 256 scale bytes:
  16,777,216 pairs, zero bit mismatches, including NaN payloads.
- 100 final native comparisons across ten cache/selection patterns: 40 sequential
  prefill cases and 60 split cases, including unaligned values and window-only
  attention. Outputs are bit-exact against the prior native library.
- Seven independent pinned TileLang/Torch cases pass existing numerical bounds,
  changed graph replay and metadata/alias/span guards. Three analytic 8 GiB pool
  cases pass, including high and invalid physical page indices.
- Native CPU/CUDA CTest: two passed. Python syntax and Git whitespace checks pass.

The exhaustive probe includes the actual production source. The split comparison
qualifier can now run both libraries with the same split count, require bit
identity and alternate timing order. No input-size specialization is introduced.

## Adjacent-query reuse

Neither our native kernel nor the inspected upstream prefill kernel explicitly
shares KV staging across adjacent queries. Four full adjacent 128-entry windows
have a union of 131 rows versus 512 independent row references. L2 can already
serve overlapping reads, however, and QK/PV arithmetic remains query-specific.
Explicit sharing adds storage, synchronization and register pressure. It remains
an unproven candidate; no multi-query rewrite or speedup is claimed here.

## Deployment and evidence

Raw results and frozen native runtime are under `/tmp/ds41-attention-packed`;
[the companion JSON](ds41-attention-packed.json) preserves component cases,
conversion hashes, API measurements and equivalence checks. The daemon is reused
from `/tmp/ds41-router-serving`; the Spark workers are unchanged.

The prior `ds41-router2048-{target,spec}-api-dev` containers and their native
artifacts remain stopped for rollback.

A same-artifact 4096-token chunk trial preserved all eight benchmark responses and
all eight quality cases per mode. Warm code reached 1591/1615 tok/s (target/dSpark),
but repeated text fell to 1744/1735 tok/s versus 1825/1836 at 2048. First code
requests also slowed to 1316/1252 tok/s. Decode remained about 39/114–117 tok/s.
The small code gain does not justify the repeated-text regression; retain 2048.
The selected containers are `ds41-packedkv2048-{target,spec}-api-dev`; the
4096 trial containers remain stopped. This bounded comparison does not identify
the cause of the chunk-size sensitivity.
