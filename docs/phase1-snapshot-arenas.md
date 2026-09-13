# Reusable snapshot arenas

Serving now allocates bounded backbone and draft snapshot arenas before API
readiness. Retention takes a free slot; eviction returns it without CUDA
allocation or free calls. Outstanding storage keeps its arena alive, and a slot
cannot be reused until its storage owner is dropped. Existing synchronous copy
paths drain before dropping storage, including enqueue failures.

Both 24-entry retention banks are covered, with two additional slots reserved for
one pending snapshot per lane in the forthcoming cooperative copy path. The
default dSpark arenas occupy **139.4 MiB**: 50 aligned backbone tails and three
sets of 50 packed draft rings. Unused arenas are omitted when retention is disabled.
Standalone cache fixtures retain the existing demand-allocation fallback.

Snapshot allocations are included in occupied memory before KV and local RTX
placement. The default pool trades whole page groups for the arena bytes. At C16
this changes global KV from 16,827,351,040 to **16,681,077,760 bytes**, providing
**18,710,016 tokens plus private-tail headroom**, while preserving five local RTX
layers and both 24-entry retention limits. Explicit KV-size overrides remain exact
(rounded to page groups); total-memory reservations already account for the live
arenas. Minimum admission and copy-on-write capacity remains enforced.

An initial attempt kept the old default pool and failed the startup budget check
for five RTX layers. That failure and its logs are preserved. The adjusted default
uses the authorized KV tradeoff instead of reducing local layers or runtime headroom.

## Validation

- The GPU arena check verifies exhaustion without aliasing live slots, reuse of
  only released storage, peer contents surviving overwrite, and outstanding
  storage surviving destruction of the allocator handle.
- The packed draft prefix test now uses a one-slot arena through short, wrapped,
  and long frontiers, preserving exact bytes across request-slot reuse.
- The all-cache fixture now uses pooled backbone tails and passes retained prefix,
  partial/zero replay, immutable future, failure revocation and recovery checks
  across all 44 owners.
- Five memory-policy tests pass, including default byte tradeoffs, explicit
  override preservation and minimum capacity for short contexts. Test compilation
  and release build pass.

Snapshot copies still wait on the host. The next step is queued copy ownership and
cooperative completion; this allocation change alone does not complete lane
asynchrony.

## Serving observations

One RTX PRO 6000 Blackwell at **400 W and standard memory speed**, four unchanged
Sparks, five RTX resident expert layers, adaptive independent dSpark, and ordered
first-content admission for mixed batches. Candidate ran first. No builds overlap
timed inference; the prefill matrix and full release suite were not rerun.

| Workload | Dynamic snapshots (`8eb1d68`) | Startup arenas |
|---|---:|---:|
| C1 code median, tok/s (three samples) | 129.48 | 130.70 |
| C2 mixed, tok/s | 92.18 | 92.42 |
| C4 mixed, tok/s | 112.34 | 114.86 |
| C8 mixed, tok/s | 157.00 | 153.03 |
| C16 mixed, tok/s | 176.56 | 177.79 |
| Startup to API ready, seconds | 5.323 | 5.317 |
| Peak observed GPU memory, MiB | 96,962 | 96,928 |

These single mixed batches do not establish a throughput gain; C8 is 2.5% lower
while the other samples are slightly higher. Startup remains approximately 5.3
seconds. Both arms pass needle retrieval, prompt and retained-turn reuse, eight
cancellations with eight survivors, recovery, and four concurrent high-thinking
constraint checks.

The control's explicit global pool was calculated using an overestimate of draft
snapshot storage: it is 16,671,508,480 bytes, **9.126 MiB smaller** than the
candidate's actual default. This is not an exactly matched pool comparison. Raw
commands and observed capacities are preserved; the measurements are a functional
and regression pilot, not a precise attribution of small timing differences.

The [evidence record](phase1-snapshot-arenas.json) contains both the initial failed
fit and successful serving comparison. Standard serving was restored after both
attempts. Stopped experiment containers were removed after saving their logs.
