# Combined target, engram and dSpark publication

`TargetPass::commit_with_dspark` publishes one accepted-count vector through all three dSpark FP8 windows and the existing forty target windows, four compressed sources and engram histories. It validates the completed target identity, request histories, acceptance counts and all dSpark leases before publication. Invalid preflight input preserves the completed pass and proposal for correction.

After publication starts, an error revokes every participating target/engram admission and all three dSpark leases, including requests accepting zero rows. The failed pass requires discard before reuse. Device writes are not undone; revocation prevents their partially advanced histories from being reused. Publication runs synchronously under exclusive ownership, without exposing intermediate cache states to another consumer.

Two real-checkpoint fixture runs passed on RTX GPU 1 with the four resident Spark workers:

- Normal publication: sixteen requests, 80-row prefill and two 16-row decode passes, with every dSpark and target end reaching 5, 6 and 7. An excessive accepted count preserved the pass, request batch and producer for a successful retry.
- Injected late failure: after two successful passes, the fixture reset target execution progress before the final combined commit. dSpark published alternating zero/one accepted rows, then target commit rejected the incomplete producer. All target and dSpark leases became unusable; discard and fresh target/engram/dSpark admissions succeeded, starting with empty histories. Old leases stayed invalid after readmission.

The production release build passes. This test covers an error after dSpark publication, not every possible CUDA/transport failure or failure after individual target writes. The API still uses target-only generation. Draft generation, verification, sampling/RNG policy, concurrent scheduling and full cancellation qualification remain open.

[Run logs and failure parameters](ds41-dspark-combined-commit.json).
