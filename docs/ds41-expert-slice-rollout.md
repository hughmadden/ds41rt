# Four-worker expert-slice development rollout

The mixed-capacity native library is live on all four Sparks, with the existing target and dSpark APIs on RTX. Each worker uses width 64 at capacity one and width 192 at its wider capacity eighty. Coordinator/worker executables, input FP8 wire rows and compact BF16 rank returns are unchanged. This is a development rollout, not completion of the release or model-quality gates.

The mounted library SHA256 is `872591810ab2d9f3ed1715c13d8207b3d715aabd7bf466bd33a661b64a5c29ad` on all four running containers. Worker containers are now `ds41-slice-target-worker`; prior `ds41-fp8-target-worker` containers remain stopped and intact for rollback. All four loaded all forty layers and reported ready. APIs remain `ds41-native-api-dev` on port 18041 and `ds41-native-spec-api-dev` on 18042.

## Quality and lifecycle

Seven of eight prompts preserve text and usage exactly before/after, in both API modes. All six objective cases pass. All eight target/dSpark pairs agree within each deployment. Streaming counting output/usage, cancellation recovery and unsupported-sampling rejection pass in both modes.

**Exact before/after output preservation failed for the open-ended spoon explanation.** Both deployments give the same first sentence: “Metal conducts heat away from your hand much faster than wood does.” The baseline second sentence is “So your skin loses heat quickly to the metal, making it feel colder.” The candidate uses “This rapid heat loss makes the metal spoon feel colder even though both are at room temperature.” Completion usage changes from 29 to 32 tokens. Both answers are factually appropriate and satisfy the two-sentence request, but semantic acceptability is a manual observation, not numerical equivalence. The candidate remains a development deployment with this divergence explicitly open; broader logit/reference checks are required. This also does not close the older compact-return quality gate.

The first candidate API test was launched before the APIs finished startup and received connection-refused errors. That attempt is excluded; the recorded final tests started after both APIs reported ready. Candidate workers were also recreated once to restore the baseline RUST_LOG/timing setting omitted from their first launch configuration.

## End-to-end measurements

Same short counting prompt, one client, greedy generation, three streaming runs per mode, sequential target then speculative execution. Reported decode TPS excludes the first completion token and prompt processing. The first run warms; all three samples are retained. No statistical-significance or release-throughput claim is made.

| Mode | Before TPS | Candidate TPS | Median change |
|---|---|---|---:|
| Target | 11.02, 10.51, 11.15 | 12.02, 11.02, 11.64 | +5.6% |
| dSpark | 47.45, 46.41, 51.61 | 49.02, 49.85, 48.26 | +3.3% |

Ranges overlap. The 90/270 TPS targets remain far away; these are bounded measurements rather than a broad performance characterization.

Median expert GPU kernel times across the four workers during qualification:

| Input rows | Before | Candidate |
|---|---:|---:|
| 1 | 239–241 µs | 169–173 µs |
| 6 | 635–645 µs | 506–509 µs |
| 80 | 3.129–3.154 ms | 2.091–2.116 ms |

These windows include whole qualification traffic, with differing generated text on the explanation; they are not matched individual-layer pairs. They have no per-arm clock/throttle admission or actual DRAM counters. The component improvements are nevertheless consistent on all four workers.

Coordinator phase medians explain the smaller API gain. Target one-row expert wait/execution remains about 1,308 µs per layer after the change, versus about 169–173 µs inside each Spark kernel; attention remains about 579 µs. Six-row expert time is about 1,783 µs versus roughly 506–509 µs inside a Spark, with attention about 599 µs. These independently aggregated medians cannot be subtracted as an exact critical-path decomposition, but they identify coordinator/transport overhead and attention as the next profiling priorities.

## Evidence and rollback

[Raw API results, comparisons, timing summaries, mounted hashes and exact worker creation arguments](ds41-expert-slice-rollout.json) are retained. Full worker/API phase logs remain under `/tmp/ds41-slice-rollout` on the coordinator. The library, export and native official checks are documented in [the complete build](ds41-expert-mixed-native-build.md) and [official component comparison](ds41-expert-native-official.md).

To roll back, stop both RTX APIs, then on each of ostrich/dodo/emu/kiwi stop `ds41-slice-target-worker` and start `ds41-fp8-target-worker`. Wait for all forty layers/ready on each worker, then start both APIs and verify their readiness before sending prompts. Candidate artifacts are separately mounted at `/tmp/ds41-slice-artifacts/cmake`; old libraries were not overwritten. Repeat the same paired tests after any rollback.
