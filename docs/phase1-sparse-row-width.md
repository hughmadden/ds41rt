# Query-local sparse attention width

Short-context verification previously used the last proposal's position to set
one window width for every query in the request. Compressed keys follow this
padding, so trimming a draft moved an earlier query's keys between 64-key online
softmax tiles. The kernel rounds unnormalized probabilities to BF16 per tile;
changing the tile layout can therefore change the greedy output even when the
query's causal history is identical.

The automatic-width path now computes `min(query_position + 1, 128)` separately
for each query. Positive explicit widths retain their existing meaning. Graph
arguments remain stable, no launches or buffers are added, and short queries do
not acquire extra padding. Positions at or beyond 127 use the same width as
before. This fixes one source of shape dependence; it does not assert bitwise
invariance across all kernels, split counts, or concurrency shapes.

## Evidence

The first fable verification pass, starting at position 49, initially matched
through layers 0 and 1 and first differed at layer 2, the first compressed-source
layer. Its layer-39 residual RMS differences were 0.522 and 0.401 for the two
shared rows. An isolated fixed-128-width control made all 40 layers exact; the
query-local implementation also makes all 40 layers exact without that padding.
The code, fable, and topic requests improve from **0/3 to 3/3 byte-identical
paired outputs** between draft limits one and five. Applicable objective checks
pass in both arms.

The focused qualifier passes **63 cases**: window-only, FP8 source and FP4 source
storage; 1/2/6 query rows; positions 49, 60, 63, 124, 127, 128, and 2048; and
changed-position CUDA graph replay. Its reference is the published kernel run
separately for each query with that query's explicit causal width. The published
kernel fails the same test with automatic width at position 49 and six rows,
confirming that the check detects the original defect.

All **60 long-context regression cases** remain byte-exact against the published
kernel, covering ten selection/masking patterns at M1/M2/M6/M16/M80/M256. Existing
graph timings show roughly unchanged M6 attention (**47.37 → 47.19 µs**) and M80
attention (**92.36 → 92.22 µs**). These are medians across synthetic patterns,
not end-to-end serving throughput or a claimed speedup.

[Machine-readable evidence](phase1-sparse-row-width.json) records comparisons,
artifact hashes, checks, and timing summaries. The native probe was compiled
with CUDA 13.3 as a sparse-attention shared-library overlay linked to the
published v1 native library; all other native operations and the four resident
Spark workers were unchanged. A final integrated build and uninstrumented
serving performance checks remain necessary. The standard v1 coordinator was
restored after the disposable probes.

## Reproduction tools

- `scripts/qualify-ds41-sparse-row-width.py --baseline BASE.so --candidate NEW.so --output JSON`
  checks automatic widths against explicit per-query baseline launches.
- `scripts/compare-ds41-sparse-attention.py --baseline BASE.so --candidate NEW.so --fp4-source --baseline-fp4-source --rows 1 2 6 16 80 256 --output JSON`
  checks the existing long-context routing/masking cases and graph timings.
- [Activation tracing instructions](phase1-confidence-observation.md#layer-boundary-activation-tracing)
  describe the opt-in layer dumps and comparison tool.

The fixed-128 control was an isolated diagnostic and is absent from the final
source. Adaptive verification is still pending; fresh calibration must use the
corrected attention path rather than assume earlier fixed-length trajectories
remain unchanged.
