# Phase 2: two RTX coordinators and four Sparks

Status: implementation planning and ownership audit; dual-GPU serving is not yet implemented.

## Required outcome

- Automatically choose two suitable RTX GPUs when their available memory can
  support the planned deployment. Provide an explicit one/two GPU override.
  Validate feasibility before stopping a running service; account for memory
  owned by a deployment being replaced without counting unrelated processes.
- Preserve the single-RTX serving path and qualify its performance against v2.
- Keep independent adaptive dSpark lanes. GPU transfers and reductions may wait
  on their own dependencies, but must not join the two request lanes. Allocate
  execution and transfer storage before the decode loop.
- Host all twenty encoder routed-expert layers on the RTX pair using TP2.
  Partition shared experts with TP2 and vocabulary rows across both GPUs,
  merging greedy winners deterministically with global token IDs.
- Place attention and its compressed KV/index consumers together. Allocate
  vision, embedding, and dSpark to balance the remaining per-device budget.
- Reduce Spark residency for layers fully hosted on RTX, preserve decoder
  execution and bounded replay, and measure the actual remote calls during
  prefill and verification.
- Preserve FP4 compressed KV, FP8 SWA, exact reuse, retained snapshots,
  constraints, tool calling, cancellation, and rapid startup.

## Initial evidence and placement constraint

The v2 baseline is tag `v2`, commit
`bff7d2ee4044ab7a267585ef83d1a90d5f7f114a`. Both RTX PRO 6000 Blackwell cards
report 97,887 MiB total and a 400 W power limit. At phase-2 entry GPU0 runs
the qualified v2 service (96,946 MiB used); GPU1 reports 12 MiB used.

Compressed sources originate at layers 2, 8, 14, and 20. The existing pool
uses page counts `[g, g, g, 2g]`: placing sources 2/8/14 on one device and
source 20 on the other divides compressed source and index bytes 60/40.
Compute the shared page-group count from the tighter device budget after
weights, workspaces, snapshots, staging, and runtime headroom. Do not treat
free memory on the other device as additional capacity for the limiting source.

The user's referenced detailed layer map was not included in the received
request. The 3/1 source placement above is a candidate pending that clarification.
Consumer placement must follow the actual CED/index dependency map, including
bounded decoder replay, rather than treating encoder residency as proof that
every prefill operation is local.

## Implementation sequence

1. Derive tensor, cache, and workspace byte accounting for both devices;
   enumerate source consumers and transfer boundaries. Validate memory and
   peer-copy capability on the available hardware.
2. Add explicit topology/placement objects and asynchronous device-transfer
   ownership. Preserve a direct single-device implementation.
3. Implement TP2 routed/shared experts and vocabulary partitioning with
   independent numerical checks, changed-input graph replay, and cancellation.
4. Integrate per-source cache and attention placement, auxiliary components,
   encoder residency, and reduced Spark loading into complete serving.
5. Wire automatic GPU discovery and forced one/two mode into the standard
   launcher only when both runtime paths are usable.
6. Qualify complete two-GPU serving, asynchronous lane progress, startup,
   memory, cache reuse, constrained/tool output, and weighted real workloads.
   Compare one-GPU performance against v2 under matched settings and preserve
   all results, including losses. Count remote encoder calls explicitly.

Commit and push each completed development increment on `dev`. Keep `main`,
`release/v2`, and the published v2 images as the qualified rollback baseline.
