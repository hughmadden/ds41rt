# Native prefix cache implementation

The release API uses `serve-native`, entering `v41_native_serve.rs` and its
`scheduler.rs`. The older `commands/real_full` radix/cache is not on that path.
The native scheduler now restores retained request state through a token radix
and reports the reused token count through the API. The original development
APIs remain available for uncached comparisons; separately launched candidates
exercise this integration.

## Compressed source ownership

The native source pool now supports retaining and restoring initialized prefixes
without copying their GPU data. Every active request table and retained prefix
owns references to physical pages. Dropping a retained prefix releases its
references; a page becomes free only when its final owner disappears. Prefix
handles cannot be restored into another pool. The pool tracks committed row
counts and rejects retaining an uninitialized portion of a partially filled page.

KV values/scales and index keys/scales share one page table. When an append
touches a shared partial page, the transaction reserves a private replacement,
copies all four planes on the commit stream, writes accepted rows, then publishes
device metadata and host ownership. Stream draining and the existing request
invalidation path protect failed transactions. Appending after a completed page
uses a new page without copying the completed page.

No GPU data copy is added to appends on exclusive pages. Prefix restoration
uploads the page table and initialized row count. A shared partial tail copies
256 × (64 + 4 + 512 + 16) = 152,576 bytes, independent of total prefix length.
Host reference accounting occurs at page allocation/publication/release; this
is not an end-to-end performance result.

When every owner of a shared partial page appends in the same transaction, one
owner keeps the original page. All other copies finish before any accepted rows
are written. A retained snapshot or non-appending owner prevents reuse by a
writer. This avoids reserving an unnecessary extra page at full concurrency;
divergent prefix lengths and zero acceptance are covered by device tests.

## Admission and eviction

`--prefix-cache-entries` defaults to 16 and accepts 0 through 16; zero disables
retention. A compressed-edge token radix stores complete retained frontiers and
selects an exact ancestor or a compression-aligned partial match. Each value owns target state,
optional dSpark state and the already-computed next greedy token. Exact prompt
hits can therefore emit the first token without rerunning a model pass.

The scheduler captures successful prompt prefill and normal completion at drained
transaction boundaries. It excludes failed and cancelled completions, and stores
only the token frontier actually committed by greedy verification. Resuming a
turn restores the forty target rings, compressor carry, Engram history, shared
source pages and three draft rings for short continuations. With at least 128
new prompt tokens, [encoder continuation](release-v1-exact-suffix.md) restores
only encoder rings and sources, then rebuilds the final decoder/draft window.

Least-recently-used eviction drops a radix value and its state together. Duplicate
frontiers are removed before recapture; full-cache eviction precedes allocating
another tail, so retained tail residency stays within the configured limit.
Before admission and each decode round, source capacity is checked for the
upcoming writes, evicting retained entries until the work fits. Active references
continue to protect shared pages. Invalidated request leases can be cleaned up
without stopping the scheduler.

[Compression-boundary reuse](release-v1-partial-prefix.md) now reconstructs
encoder state with at most 128 replay tokens when a prompt matches inside a radix
edge. Exact retained ancestors remain preferable when they skip more computation.
Final release pool sizing and long-context qualification remain required.

## Qualification and remaining integration

### Retained request state

Native request owners now expose retention/restoration of all forty SWA rings,
four compressed-source prefixes and the live odd-token compressor carry. The
bounded tail uses one GPU allocation and a dedicated stream; capture and restore
drain before publishing state, including error paths. Only initialized ring
spans and live pending rows are copied. Source pages stay shared through the
previously qualified copy-on-write owner.

The backbone tail reserves 2,720,064 bytes per saved request. Three saved dSpark
rings add at most 202,752 bytes. Sixteen complete tails therefore need at most
46,765,056 GPU bytes, excluding shared source pages and small host metadata.
These are allocation sizes, not measured total server memory.

Engram history forks preserve absolute position and the three-token lookback,
including image barriers, while obtaining a fresh identity. Old prepared batches
cannot be committed into a resumed request. The native request owner restores
that history with its backbone state; the draft owner separately verifies all
three saved draft rings end at the same target position before reuse. New request
RNG identities are preserved.

The real-weight cache-producer qualification compares every initialized byte
across all 44 backbone owners after restoring a 129-token prefix into a recycled
slot. The following commit, which completes an odd compressor group, is exact
against continuation of the original request. The snapshot also survives release
of every original request. The same qualification retains its 16-request CED,
ordered encoder publication and late-source-failure recovery coverage. This is
component execution with official producer weights, not full-model/API quality.

The restored and original continuations both need private partial pages while
the original prefix stays retained. The qualification bank includes two spare
pages per source. The release pool sizing and eviction policy must provision
these live copy-on-write needs in addition to retained capacity.

See [retained-state evidence](release-v1-retained-state.json) for focused GPU,
Engram identity and real-weight results, source hashes and memcheck output.

The component qualification uses the selected native library whose SHA256 is
`ccd82d862473f71d01622bcf5decd9d5789444210db1be4cc245dbdcb44b8a07`.
Tests cover exact and divergent retained prefixes, all four stored planes,
copy-on-write isolation, completed page sharing, restoration after original
request release, eviction, pool exhaustion and recovery when an exclusive tail
can be appended without free pages. See the accompanying JSON evidence.

See [native admission qualification](release-v1-native-admission.md) for live
complete-prompt and retained-turn parity, concurrency, memory-pressure recovery
and controlled decode measurements. These focused checks do not replace the
release's long-context, agentic, vision or full performance gates. Partial replay
has separate focused qualification; release pool sizing remains open. The original comparison
containers have not been replaced by this change.

Reproduce component tests from the repository root:

```bash
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/cu13/lib" \
DS41RT_NATIVE_LIB=/tmp/ds41-draft-release-default/selected-native/libds41rt_native.so \
DS41RT_PYTHON=.ds41rt-cache/reference-venv/bin/python \
scripts/run-with-python-env.sh cargo test --manifest-path rust/Cargo.toml \
  -p ds41rt-daemon --bin ds41rt source_cache -- --include-ignored --test-threads=1
```

The native library path is a development artifact; substitute a matching built
library when reproducing after a reboot. Run the resulting test executable under
`compute-sanitizer --tool memcheck --error-exitcode 99` with the same environment
and test arguments to qualify device bounds.
