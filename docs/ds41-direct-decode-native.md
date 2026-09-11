# Direct V4.1 decode: planned kernel and native export

b12x master `f9ce62c` integrates the direct-routing/output-split candidate. The heuristic selects it for SM121 with capacity one, 384 experts, hidden 5120, local intermediate 576 and top six. Explicit policy validation and the private compile boundary reject unsupported direct V4.1 geometry. The generic materialized W4A8 decode predicate remains unchanged. Five output tasks per route retain ordered accumulation across all intermediate slices and keep their intermediates in shared memory. Consumer task expansion does not expand producer metadata allocation. The existing direct-routing flag identifies the compiled specialization; no live request count was added to a compile key.

The native exporter now consumes the resolved b12x routing mode and M tile instead of unconditionally exporting grouped execution. Its manifest records routing mode and output splits. A target-native export of capacities one and sixteen produced direct/five-split and grouped/unsplit variants respectively, and both linked through the actual native C ABI wrapper.

Validation on ostrich GB10:

- Eight policy tests pass, including rejection of other capacities/geometries.
- The integrated Python path passes the FP32 oracle, changed-input/changed-route graph replay, stable allocated-byte check, inactive-route zero output and recovery. Its synthetic graph diagnostic remains approximately 221 µs.
- Native C ABI/AOT execution passes the FP32 oracle and changed-input graph checks for one and sixteen rows. The one-row case additionally verifies direct policy selection, inactive routes and recovery.
- The native component qualifier retains b12x-managed input, resident packing and scratch and substitutes the exported callable at the compiled-kernel boundary. It does not qualify Rust scratch ownership, official resident weights or full serving performance.

[Artifact hashes, export metadata, complete qualifier and reproduction commands](ds41-direct-decode-native.json). Local logs are `/tmp/ds41-direct-{policy-tests,final-check,native-export,aot-qualification}.log` (the Python diagnostic is `/tmp/ds41-final-direct-check.log`). Remote source and artifacts are isolated under `/tmp/ds41-direct-native-source` and `/tmp/ds41-direct-native-export`.

## Serving integration

The dedicated one-row kernel and scratch state are now preinitialized alongside each backbone wave, included in memory accounting, and rebound with the same resident layer weights. One-row launches and compaction select that state; larger requests retain grouped execution. No allocation or compilation is added to request dispatch.

The [four-Spark rollout qualification](ds41-decode-dispatch-rollout.md) covers actual Rust ownership, official resident weights, row/layer transitions, changed-input graph replay and live API checks. It supersedes the component-only serving limitation above; the component experiment itself remains scoped as originally measured.
