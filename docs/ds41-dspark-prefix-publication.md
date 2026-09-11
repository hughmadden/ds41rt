# Request-bound dSpark context publication

`DsparkMainContext::execute_rows` now returns an exclusively borrowed proposal that retains the target batch identity, request IDs, absolute positions and flattened source offsets. Publication validates all three window lease sets and accepted counts before any GPU writes. Each request publishes only its accepted prefix into the three persistent FP8 rings. Zero acceptance consumes the proposal without advancing that request. Dropping the proposal invalidates producer readiness.

Validation errors preserve the proposal for a corrected retry. A started GPU commit consumes it; the existing window commit path revokes participating leases on a native failure. Callers must still coordinate this operation with target-cache and engram commits and revoke requests if a subsequent operation fails. The later [combined commit path](ds41-dspark-combined-commit.md) coordinates those owners with admission revocation after partial failure. Draft generation remains disabled in the API.

The real-checkpoint fixture passed on RTX GPU 1 with all four Spark workers. Sixteen requests ran an 80-row prefill and two 16-row decode passes; full publication advanced every stage's committed end to 5, 6 and 7. A separate partial publication accepted `[0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3]` rows. Every accepted packed row matched the corresponding full-publication bytes in all three stages. All rejected tails retained the initial `0x5a` pattern.

The fixture also checks overacceptance, foreign batches, incorrect request leases, stale lease generations, stale committed positions even with zero acceptance, duplicate commits, and discard. The CPU grouping test and production release build passed. All fifteen exported target-logit, main-context and stage-KV files remain byte-identical to the prior handoff run. This establishes publication behavior and regression equality, not an independent numerical reference for the KV producers.

[Recorded checks, output hashes and test logs](ds41-dspark-prefix-publication.json).
