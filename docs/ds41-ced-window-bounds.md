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
requesting bounded execution against it fails explicitly. The attention wave now derives this field from the window owner’s initialized
interval, using a preallocated device-bound array and its existing staging buffer.
Ordinary zero-origin windows avoid the extra upload. Request admission still uses
ordinary windows until CED request/cache progression is integrated.

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

## Window owner integration

`WindowState::begin_replay` initializes a fresh decoder lease as an empty interval
`[position, position)`, advances its proposal version, and publishes the device
end. It rejects encoder layers, reused leases and positions beyond model capacity.
No ring bytes are declared valid before the new lower bound. A failed device update
revokes the lease; release and reacquisition reset the lower bound with a new
lease generation. Subsequent committed rows retain the replay floor.

The GPU owner test `decoder_replay_window_lease_boundaries` passes across all
sixteen slots, covering starts 0/1/127/128/16384/1048576, invalid/foreign/stale leases,
version changes, device-end readback and slot reuse. It uses the frozen packed
native library; this is window ownership qualification, not a complete bounded
Rust attention pass. The first host test invocation could not load the CuTe runtime;
the same compiled test passes in the CUDA development image (0.17 seconds).

Attention now reserves eight device bytes per planned row for replay bounds. Total
lane workspace budgets become 2,301,989 bytes at capacity 1, 80,607,196 at capacity
80, and 2,985,263,116 at capacity 4096. Existing host staging is reused. Graph
fingerprints include bound-buffer identity and entry-point choice; boundary values
are uploaded without becoming graph keys. The Rust daemon check passes.

Raw owner test and final compilation logs are under `/tmp/ds41-ced-bounds` as
`window-test-container.log` and `window-final-check.log`. Live API artifacts are
unchanged. Separate encoder/global-source and decoder commit progress is still
required before bounded prefill can run through the API.
