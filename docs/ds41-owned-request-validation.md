# Validate coordinator expert requests without temporary wire frames

Router preparation and RoCE dispatch previously each serialized the entire owned request into a temporary vector and parsed it back solely to validate the native contract. Both now validate the owned header, row descriptors, routes and payload extent directly. A shared canonical validator serves the worker wire parser and the coordinator, preserving six distinct expert routes, finite nonnegative weights, layer bounds, dtype/stride constraints and row capacity. The receiver derives its identity from the validated header and preserves frame-budget, rank, chunk and cancellation checks.

This removes two temporary full-frame allocations/copies per layer dispatch. For 16,411 prompt tokens and forty layers, the removed activation-payload copies total approximately 6.46 GiB, excluding metadata. That is source-derived logical copy volume, not measured DRAM traffic. Request payload clones already share `Bytes`; they were not additional vector copies. RoCE registered-buffer copies, router staging, route metadata ownership, worker execution and return uploads remain. The wire representation and all GPU arithmetic are unchanged.

## Qualification

Ten V4.1 transport tests pass. Added owned-request coverage checks row capacities 1/80/1024/4096, insufficient frame budget, malformed payload extent, invalid executor identity and out-of-order rank arrival. Existing malformed-route and legacy-mode cases now also exercise direct validation. All canonical errors are checked before indexing owned route/row vectors.

The final live atomic-output RoCE fixture passes in 13.66 seconds through four workers at capacities through 4096, with 27 completed waves, nine abandoned dispatches and nine injected sink failures. Existing byte-exact small-row and bounded atomic large-row gates are retained. The production daemon builds successfully; 481 pre-existing warnings remain.

All eight paired quality cases preserve preceding text and usage. The inherited strict Unicode format failure and 7/8 cross-mode agreement remain; this is not broad model quality qualification.

## Deployment

Candidate coordinator daemon SHA256: `20304f3f9f7e2fb0e3d82cab85da28f8d45aaad46e9c77c643296156029f8286`. Native library remains `07db52dec294d18ee98e44b553a7c442a43cdabb65ff8103690572fc4eeaa42b` (b12x e5343cfb). APIs are `ds41-owned-{target,spec}-api-dev`, ports 18041/18042, prefill batch 1024; the previous attention-pad APIs remain stopped for rollback. All four `ds41-prefill4096-worker` services retain their preceding artifacts and configuration. Frozen daemon, exact launch arrays and raw evidence are under `/tmp/ds41-owned-validation`.

## API measurements

Sequential C1, greedy counting after the same frozen context as the 4096 trial, no prefix-cache hits. Effective prefill divides actual prompt tokens by time to first content. Each entry is a single status measurement; control runs precede candidate runs and results are not clock-controlled.

| Context | Mode | Control / candidate prefill tok/s | Control / candidate TTFT s | Candidate decode tok/s |
|---|---|---|---|---|
| repeated | target | 1260.1 / 1326.4 | 13.024 / 12.373 | 25.19 |
| repeated | speculative | 1236.1 / 1338.7 | 13.277 / 12.259 | 93.14 |
| code | target | 1051.5 / 1066.3 | 15.607 / 15.390 | 25.30 |
| code | speculative | 1032.4 / 1102.2 | 15.895 / 14.888 | 83.35 |

All four benchmark texts, usage counts and prompt hashes match the control. Repeated-text prefill improves in this comparison; code-context changes are smaller and overlap previous run variation. Retain the removal of redundant allocations without claiming a statistically established speedup for every workload. The 2–3k checkpoint and original throughput/concurrency goals remain open.

Both API smoke suites pass JSON, three counting streams, cancellation/recovery and unsupported-temperature rejection. Short counting runs measure 35.63–35.81 target and 104.68–105.46 dSpark tok/s. In the 16k comparison, control/candidate decode is 25.83/25.19 target and 95.30/93.14 dSpark on repeated text, and 25.83/25.30 target and 84.48/83.35 dSpark on code. Decode is slightly slower in this cycle; no decode speedup or statistically established regression is claimed.
