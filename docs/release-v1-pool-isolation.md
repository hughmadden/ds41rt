# Configured-pool failure isolation

Native serving now reports compressed-source capacity pressure as a typed error
that identifies the ordered append participant. Before starting a decode round,
the scheduler maps that participant back to its active request, terminates only
that owner, releases its pages at the completed boundary, and retries the other
requests. Invalid bindings and other global failures remain fatal to the round.
Retention eviction is attempted only for actual pool pressure.

The live qualification uses C2, no retained prefixes, and an exact 1,822,720-byte
global pool: four page groups, just enough for two 932-token prompts. Concurrent
512-token completions force another page. The preserved pre-change target and
dSpark binaries terminate both requests; the candidate terminates one and lets
the peer complete all 512 tokens. Separate 1,132-token admissions reject one
request without affecting the admitted peer. JSON, SSE, cancellation, and a
fresh arithmetic recovery request pass in both modes. Late SSE errors terminate
the HTTP body without a success marker, while JSON preserves the pool message.

Four real GPU source-cache tests cover page boundaries, copy-on-write, shared
writers, exhaustion metadata, invalid-binding classification, and retained
future rows. The full real-backbone transaction test covers all 44 cache owners
and sixteen requests, including early and late failure rollback and recovery.
The release build and test compilation also pass.

[Machine-readable summary](release-v1-pool-isolation.json) records artifact and
raw-result hashes. The full raw responses and logs remain in
`/tmp/ds41-native-pool-isolation` for the release evidence bundle.
