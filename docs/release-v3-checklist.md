# DS41RT v3 release checklist

Status: release candidate qualified; final publication remains.

- [x] Implement forced and automatic one/two-RTX serving without changing the
  one-RTX execution path.
- [x] Keep the two target/draft lanes independent through decode, with owned
  workspaces and cooperative GPU completion.
- [x] Host encoder routed experts 0–19 and all shared experts TP2 on the RTX
  pair; keep decoder routed experts 20–39 on the four Spark ranks.
- [x] Keep attention beside its compressed cache source, split vocabulary rows,
  place vision/embeddings on RTX0, and place dSpark on RTX1.
- [x] Default dual mode to a 13.094 GB FP4 source pool representing 14,680,064
  logical tokens plus 32,768 tail tokens, with 24 retained turns and prompt
  snapshots. Preserve explicit pool, reservation, and concurrency controls.
- [x] Select placement-aware adaptive K5 for one RTX and retain explicit K7.
- [x] Build clean release containers, verify both expert ABIs, launch on port
  8000, and exercise replacement accounting and cleanup.
- [x] Run matched three-sample one/two-RTX target, dSpark, prefill, retained
  context, counting/code/topic concurrency, and mixed-traffic measurements.
- [x] Pass the focused dual lifecycle check for needle retrieval, exact reuse,
  retained continuation, cancellation, survivors, and recovery.
- [x] Replicate every performance-report table in the README, state the 400 W
  limit and stock memory speed, show cache bytes with token capacity, and replace
  the obsolete single-RTX diagram.
- [x] Build the final source revision and verify a clean standard launch from
  the exact resulting images.
- [x] Retire the unrelated pre-existing coordinator and Spark `v3` package
  versions, then publish the candidate as `v3` and `latest` and verify manifest,
  platform, label, and digest identity.
- [ ] Assemble binary and evidence assets with checksums, publish the GitHub v3
  release notes/assets/tag, and record the final publication identities.
- [ ] Update `main` and `release/v3`, leave the standard service healthy, and
  push every final commit.

See the [v3 release notes](release-v3-notes.md), [performance
report](phase2-release-performance.md), [machine-readable
results](phase2-release-performance.json), and [implementation/evidence
log](phase2-dual-rtx.md).
