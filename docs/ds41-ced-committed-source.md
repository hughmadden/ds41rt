# Committed global source views for decoder replay

Decoder replay can now bind source-20 global KV/index data already committed by
the encoder. It no longer requires a live compressor proposal for the decoder
suffix. This is integrated at the cache-attention binding; the target execution
loop and API still require the CED split-pass change.

`CompressorState::committed_proposal` validates the source layer, request lease
and query range. Its metadata uses each query's causal length while retaining the
full committed cache length and zero private-source count. Read-only cache buffers
provide valid backing for the unused private overlay; no private row is selectable.
The existing sparse/index address rules mask future global keys even when they
are already physically resident.

Each replay batch reserves its source snapshot from the same monotonic namespace
as ordinary compressor proposals. All decoder consumers in that batch reuse it;
a new replay batch gets a different snapshot. This avoids both stale candidate
reuse and accidental identifier collisions between independent counters. Normal
encoder/decode source proposal behavior is unchanged.

The backbone attention binder obtains committed views for replay and rejects a
supplied compressor wave. Range origins match the decoder window proposal while
global history remains at the encoder prompt end. Existing Rust references retain
cache ownership through the consumer lifetime.

## Qualification

The sixteen-request real-weight cache test passes in 4.18 seconds. It verifies
source snapshot equality across decoder layers 20 and 24, inequality across replay
batches, exact causal metadata at every replay position, rejected positions outside
the range, and decoder window lower bounds. Prior byte-preservation and failed
commit/recovery checks remain passing. This checks Rust owners/bindings, not a
complete Rust decoder forward pass.

The independent native attention qualifier adds `--committed-source`. With
`--bounded-replay`, the global cache includes every query position while local
window history is truncated. Ten cases pass unchanged FP32/pinned TileLang bounds,
including 128/129-row replay boundaries, strided source representation, changed
captured replay and metadata/alias guards. The selected invalid indices include
future positions, including positions physically present in the committed cache.
Rust daemon check and test compilation pass; existing warnings remain.

[Detailed results and source hashes](ds41-ced-committed-source.json) are retained.
Raw artifacts are under `/tmp/ds41-ced-bounds`: `committed-source.json/log`,
`source-owner-gpu.log`, `source-check.log` and `source-build.log`. Native attention
uses the previously qualified bounded-kernel artifact. Live API/worker artifacts
are unchanged; no prefill speedup is claimed until the split execution is deployed.
