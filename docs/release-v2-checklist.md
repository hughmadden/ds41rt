# V2 release qualification

Status: candidate comparison in progress; no v2 release or image published yet.

The release uses adaptive dSpark with independent execution lanes. Admission and
prefill remain coordinated. Normal decode GPU completion and receive waits yield;
error/cancellation cleanup still drains owned work before releasing storage.

## Required evidence

- Finish the combined attention graph comparison with decode receive spinning
  removed. Preserve prior graph and counting results, including losses.
- Commit and push the selected implementation. Build from a clean isolated
  checkout with `build.sh`, then exercise the standard `run.sh` deployment using
  the newly built coordinator and four matching Spark images.
- Verify automatic bottom-up RTX placement, actual loaded layer count, memory
  peaks, Spark budget/residency, and startup readiness. Record GPU UUID, 400 W
  cap, standard memory clocks and artifact/source identities.
- Run three samples per case for the eight-type target and dSpark tables,
  including warm exact counting. Prioritize weighted and individual real types
  when assessing performance; counting is a headline and diagnostic workload.
- Run retained-context dSpark at 0, 32K, 64K, 128K and 256K, three samples per
  type with verified reuse. Run warm counting concurrency C1/C2/C4/C8/C16,
  three samples each. Update matching README and performance-report tables.
- Preserve the existing prefill matrix and official API reference explicitly as
  prior measurements. Do not rerun the full qualification suite.
- Run three high-concurrency tool-eval campaigns, thinking enabled/high effort,
  preserving failures and raw traces. Distinguish fresh v2 coverage from inherited
  v1 vision, needle and other qualification evidence.
- Add RTX resident layers, actual Spark memory budget, KV size with represented
  token capacity, memory use and startup to the release headlines.
- Publish matching v2 serving images and a GitHub v2 release with high-level
  optimization notes, measured impacts, source/artifact identities and assets.
  Verify published tags/digests/assets and leave the qualified standard service
  running. Push each final commit so the branch can be monitored remotely.

Development evidence includes [weighted decode](phase1-async-weighted.md),
[counting diagnosis](phase1-async-counting.md), and
[receive polling](phase1-async-poll-yield.md). These are not substitutes for
qualification of the clean release artifacts.
