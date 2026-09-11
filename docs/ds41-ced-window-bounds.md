# Decoder replay window boundary

The sparse-attention native ABI and Rust wrapper now accept explicit device-side
SWA lower bounds for bounded decoder replay. This is a CED implementation
prerequisite, not a serving rollout: the current daemon still binds no replay
boundary and executes the full backbone.

`ds41rt_v41_sparse_attention_bounded` accepts a U64 lower bound per query row.
The window resolver masks positions below it before dereferencing values or scales.
Compressed/global source keys retain their existing causal selection. The boundary
must not exceed the committed window end; an invalid boundary zeros the query.
A zero boundary retains full-history behavior. Bounds can change between graph
replays without recapture. The new pointer must have a valid aligned span and be
disjoint from output and split scratch. Both sequential and split execution use
the same boundary semantics. Existing exported entry points and descriptor layout
are unchanged.

Rust's `V41SparseWindow::replay_begins` carries the optional device buffer; the
wrapper checks its device and size and calls the bounded entry point only when
present. Loading an older native library remains possible for ordinary execution;
requesting bounded execution against it fails explicitly. The native daemon still
sets this field to `None` until request/cache progression is integrated.

The independent sparse-attention qualifier now supports `--bounded-replay`.
It changes the bound from zero to the proposal start in a captured graph, compares
against independently gathered FP32 attention and the pinned official TileLang
attention, restores valid bounds after testing invalid ones, and rejects null,
misaligned and output/scratch-aliasing pointers. The corpus includes window-only,
compressed, strided source proposals, exact 128-row replay and a 129-row boundary
case. Existing numerical bounds remain unchanged. This validates the attention
primitive, not full-model bounded-replay quality.

Ten independent bounded-reference cases pass; thirty ordinary-entry comparisons
are bit-exact against the deployed library. [Detailed evidence](ds41-ced-window-bounds.json)
preserves the results and source hashes.

Raw builds, qualification output and Rust check log are under
`/tmp/ds41-ced-bounds`. Both SM120 and SM120a images compile. The daemon Rust check
passes with 479 existing warnings. Full native artifact integration and Rust GPU
owner qualification remain part of the upcoming CED rollout.

The remaining implementation must separate encoder/global-source advancement
from decoder window progress; retain the final encoder suffix; publish layer-20
compressed KV/index state for all encoder rows; and rebuild decoder windows and
dSpark prefix state on only the retained suffix. The current all-layer cache
commit and pass-progress guards deliberately reject incomplete passes and must be
replaced with explicit CED transactions rather than bypassed.

After CED, attention integration and wavefront scheduling, revisit expert grouping
and GLM's ordering using measured live routing. Combining ready same-layer work
can increase explicit reuse; merely alternating layers may instead reduce cache
locality. Keep the current measured M16 choice until larger groups and ordering
show an end-to-end win on the new schedule. Earlier M32/M64 trials reject those
implementations under the former workload, not all larger-group approaches.
