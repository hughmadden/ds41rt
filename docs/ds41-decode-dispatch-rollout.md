# Prewarmed one-row expert dispatch rollout

Each backbone execution wave now owns a dedicated capacity-one kernel/scratch state alongside its grouped state. One-row requests select that state for both execution and compact output. Startup memory accounting includes the extra arena (736,440 bytes with this export). Input allocations and resident weights are shared; row selection performs no allocation or compilation. Layer rebinding updates both states after draining execution, and captured graphs retain the existing prohibition on layer rebinding.

## Official-weight ownership qualification

On ostrich, a Rust fixture loaded all 384 experts for official layers zero and one, TP rank zero. It visited layers `[0, 1, 0]`, used two captured real input/routing cases and launched row counts `[1, 16, 1, 80, 1]` per case. It verified insufficient-budget rejection, distinct and stable one-row/grouped output pointers, layer rebinding, changed-input one-row graph replay and rejection of rebinding under a captured graph.

The same new owner code was run against the previous native library and the candidate. All 36 saved FP32 route and compact BF16 output files were byte-exact (77,475,840 bytes); route planes were finite and nonzero. Thus the baseline here uses the old capacity-one kernel, not the old live worker's capacity-eighty dispatch. Median one-row host launch-and-drain time was 381.136 versus 211.663 µs, measured as ten queued launches followed by synchronization and divided by ten; this is not an isolated GPU event interval.

Root release build and fixture builds passed. The full candidate native library re-exported all six expert capacities (1, 16, 80, 256, 1024, 4096), rebuilt the expert wrapper and relinked the unchanged nonexpert native objects. This was not a clean rebuild of every native component.

## Four-Spark rollout and live checks

`ds41-decode-target-worker` is running on ostrich/dodo/emu/kiwi, ranks zero through three, with capacity eighty and all forty backbone layers. Each host's frozen worker and native library hashes match the accompanying evidence. The stopped `ds41-timed-target-worker` containers remain available for rollback; stop the new worker before restarting the old one because they share port 19441. Coordinator API artifacts were unchanged.

The qualification workload recorded 15,480 dispatch observations per host: every one-row observation selected capacity one, and every larger request selected capacity eighty. Each host contributed 10,560 one-row GPU timing samples:

| Spark | Kernel median µs | Compact median µs | Worker total median µs |
|---|---:|---:|---:|
| ostrich | 237.312 | 3.936 | 283 |
| dodo | 240.384 | 3.904 | 292 |
| emu | 240.352 | 3.936 | 288 |
| kiwi | 236.640 | 3.936 | 281 |

Three short counting streams measured target decode at 11.431, 11.162 and 11.652 TPS and speculative decode at 47.074, 47.825 and 47.599 TPS. JSON response, streaming, cancellation recovery and unsupported-sampling checks passed on both APIs. All eight paired quality cases (six objective, two open-ended) passed; before/after rollout text and usage matched exactly for both target and speculative modes.

These instrumented short workloads do not establish sustained-load performance, C16 scheduling, long-prefill throughput or broad model quality. The pre-existing compact-return logit discrepancy remains unresolved; exact output for this kernel replacement does not close that separate gate. Neither 90 target TPS nor the speculative/prefill targets have been reached.

[Evidence, fixture source, artifact hashes, API results (without duplicate SSE event arrays) and timing summaries](ds41-decode-dispatch-rollout.json). Qualification entry points are `scripts/qualify-ds41-native-api.py`, `scripts/qualify-ds41-speculative-quality.py` and `scripts/summarize-ds41-expert-timing.py`. Original logs and owner outputs use `/tmp/ds41-decode-*`; these external fixtures are not production entry points.
