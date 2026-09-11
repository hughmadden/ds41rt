# Expert tiling and down-projection staging revisit

Neither wider M tiles nor double-buffering only the down projection establishes a useful skewed-prefill win. Keep the deployed M16/N192 kernel. These are component diagnostics, not new API throughput measurements.

Both experiments use official layer 39/rank 0/all 384 expert weights on ostrich (GB10), GPU packing, synthetic activations, and the skewed routing distribution described in `ds41-expert-skew-counters.md`. Every case uses planned capacity 4096, including six rows; the latter does not represent serving's small decode state. Each arm passes the existing rtol 2e-6 / atol 2e-5 comparison, poisoned output/tail checks, and graph replay without allocation. Timing uses eight interleaved rotations/reversals with five graph replays per sample. Clocks are uncontrolled.

The recovered wider-M experiment runs in `/tmp/ds41-wide-revisit/b12x`, based on b12x `907d5fd5` plus the archived wide prototype. Its M16 control matches the deployed native atomic path closely. Median complete GPU graph durations in milliseconds:

| Workload | Native | M16/N192 | M32/N192 | M64/N192 |
|---|---:|---:|---:|---:|
| 1024 skewed | 7.642 | 7.626 | 7.860 | 8.391 |
| 4096 skewed | 15.832 | 15.744 | 16.733 | 16.288 |
| 4096 uniform mixed | 12.287 | 12.301 | 14.927 | 12.586 |
| 6 shared, full capacity | 0.198 | 0.199 | 0.263 | 0.309 |

The structural follow-up overlaps the next down-projection weight tile's asynchronous copy with current-tile compute. Two slots reuse the gate/up shared weight allocation; N192 needs only 512 additional shared-memory bytes for the second scale slot. It preserves MMA and token-atomic order within each CTA and adds no global intermediate. Median graph durations:

| Workload | Native | M16 control | M16 with prefetch |
|---|---:|---:|---:|
| 1024 skewed | 7.667 | 7.587 | 7.549 |
| 4096 skewed | 15.960 | 15.762 | 15.810 |
| 4096 uniform mixed | 12.252 | 12.177 | 11.724 |
| 6 shared, full capacity | 0.181 | 0.191 | 0.171 |

The relevant comparison isolates prefetch against its M16 control: approximately 0.5% faster at 1024 skewed and 0.3% slower at 4096 skewed. That does not establish a prefill improvement. Uniform routing benefits more, illustrating why it cannot select this optimization alone. No rollout is justified by these measurements.

Raw graph samples, complete expert histograms, source hashes and baseline identity are in `ds41-expert-staging-revisit.json`. Temporary source and launchers remain under `/tmp/ds41-wide-revisit`; rejected alternatives are not added to the production kernel. The next structural investigation should cover the gate/up staging loop and compiled resource use rather than another wider-M sweep.
