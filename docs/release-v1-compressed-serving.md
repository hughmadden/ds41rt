# Compressed FP4 serving integration

The candidate uses E2M1 compressed KV with E4M3 scales in groups of 16,
including proposal buffers, accepted pages, copy-on-write retention and mixed
attention. SWA stays FP8 with E8M0 scales in groups of 32; the independent
index format is unchanged. This completes implementation of the migration,
not its full performance or release qualification.

Compressed rows shrink from 528 to 288 bytes. Including the unchanged 68-byte
index row, the default global source pool shrinks from 37,545,902,080 to
22,426,746,880 bytes, saving 15,119,155,200 bytes. The C16 + eight spare context
policy and 24 completed-turn / 24 prompt snapshot limits are unchanged. This
is source-pool storage, not total GPU memory.

The attention descriptor explicitly distinguishes legacy FP8 and architectural
FP4 sources. Window slots always retain their original layout. CUDA graphs
capture this format alongside their pointers; retained pages belong to the
same immutable pool owner. Source reads unpack four FP4 values at a time and
use packed BF16 scale multiplication on SM120a. Separate kernel specializations
remove the format branch from legacy attention. All eight legacy kernel
variants across SM120 and SM120a retain identical encoded GPU instructions
against the preserved FP8 native build.

## Completed checks

[Evidence manifest](release-v1-compressed-serving.json) identifies the immutable
candidate artifacts and [archived raw results](evidence/native-compressed-serving.json.gz).

- 1,470 attention checks on the final native library: 1,260 FP4 checks across
  both RTX GPUs and 210 legacy/SWA checks. FP4 outputs match attention over
  independently decoded BF16 source values exactly. Coverage includes grouped,
  split, bounded replay, private strided proposals, stale pages, unaligned
  pointers, zero and mixed scales, and changing buffers during graph replay.
- 96 native closed-form cases independently check attention values and invalid
  replay bounds. Compute Sanitizer reports zero errors for all 96 cases and
  for a separate 40-case Python graph/unaligned/bounded matrix.
- Five source-cache GPU tests pass on each RTX, including copy-on-write,
  eviction, exhaustion, truncation and the highest physical row 67,108,863.
  Three CPU pool-sizing tests pass. Native and Rust release builds pass.

The first runtime-branch prototype showed a legacy attention cost and was
replaced by the separate specializations above. An early sanitizer attempt was
interrupted; only the completed final checks are counted. BF16-reader reference
kernel timing is not an FP8 serving performance comparison.

## Initial serving performance

Four alternating AB/BA pairs per workload use equal 256-token output budgets,
sequential shared Spark workers and the same GPU for each mode's two arms.
Both RTX PRO 6000 Blackwell GPUs have 400 W limits and standard memory speed;
these controlled ordinary-text workloads disable thinking. Tool evaluation
will use high thinking. Development contexts are 32,768 tokens, with C16 target
and C2 dSpark admission limits; measurements here are C1.

| Mode | Workload | FP8 tokens/s | FP4 tokens/s | Median paired change |
| --- | --- | ---: | ---: | ---: |
| Target | Counting | 43.15 | 43.03 | −0.32% |
| Target | Code | 42.95 | 42.79 | −0.36% |
| dSpark | Counting | 145.73 | 145.59 | −0.10% |
| dSpark | Code | 126.49 | 128.90 | +1.95% |

These short measurements do not establish a general speedup or close the
performance gate. Precision migration permits text differences while requiring
equal output-token counts; raw outputs and parity flags remain in the archive.
The manifest's completed flags denote completed measurements and assertions,
not satisfaction of a throughput threshold.

Next, qualify and optimize cold prefill and retained-context decode, then
broaden contexts and concurrency. Only after performance matches or improves
should the requested needle checks through 1M and high-thinking tool-call
qualification follow, including agentic reuse, cancellation and recovery.
Final release measurements and the three final tool-eval runs must use the
corrected serving build.
