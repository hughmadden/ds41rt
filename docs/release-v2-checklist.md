# V2 release qualification

Status: scoped qualification and v2 registry publication complete; GitHub release pending.

The release uses adaptive dSpark with independent execution lanes. Admission and
prefill remain coordinated. Normal decode GPU completion and receive waits yield;
error/cancellation cleanup still drains owned work before releasing storage.

## Required evidence

- [x] Finish the combined attention graph comparison with decode receive spinning
  removed. Preserve prior graph and counting results, including losses.
- [x] Commit and push the selected implementation. Build from a clean isolated
  checkout with `build.sh`, then exercise the standard `run.sh` deployment using
  the newly built coordinator and four matching Spark images. The clean build
  finished in 304.85 seconds after the missing local-expert AOT export was caught
  and fixed; the final standard service remains running on port 8000.
- [x] Verify automatic bottom-up RTX placement, actual loaded layer count, memory
  peaks, Spark budget/residency, and startup readiness. Record GPU UUID, 400 W
  cap, standard memory clocks and artifact/source identities. The standard plan
  loads RTX layers 0–4, keeps all 40 layers on each Spark under a 100 GiB budget,
  and reached readiness in 57.76 seconds.
- [x] Run three samples per case for the eight-type target and dSpark tables,
  including warm exact counting. Prioritize weighted and individual real types
  when assessing performance; counting is a headline and diagnostic workload.
  The [v2 performance report](release-v2-performance.md) records 44.29 target-only
  and 79.80 dSpark weighted tok/s.
- [x] Run retained-context dSpark at 0, 32K, 64K, 128K and 256K, three samples per
  type with verified reuse. Run warm counting concurrency C1/C2/C4/C8/C16,
  three samples each. Update matching README and performance-report tables. All
  reuse checks passed and C16 reached 934.05 aggregate tok/s.
- [x] Preserve the existing prefill matrix and official API reference explicitly as
  prior measurements. Do not rerun the full qualification suite.
- [x] Run three high-concurrency tool-eval campaigns, thinking enabled/high effort,
  preserving failures and raw traces. Distinguish fresh v2 coverage from inherited
  v1 vision, needle and other qualification evidence. The [tool report](release-v2-tool-eval.md)
  records 155, 153, and 158 of 176 points at C16 without retries.
- [x] Add RTX resident layers, actual Spark memory budget, KV size with represented
  token capacity, memory use and startup to the release headlines.
- [x] Publish matching v2 serving images and verify `v2`/`latest` manifest
  identity, platforms, labels, and digests. The [clean build and registry report](release-v2-build-run.md)
  records the exact qualified pair.
- [ ] Publish a GitHub v2 release with high-level optimization notes, measured
  impacts, source/artifact identities and assets. Verify the release and assets,
  leave the qualified standard service running, and push each final commit so
  the branch can be monitored remotely.

Development evidence includes [weighted decode](phase1-async-weighted.md),
[counting diagnosis](phase1-async-counting.md), and
[receive polling](phase1-async-poll-yield.md). These are not substitutes for
qualification of the clean release artifacts.
