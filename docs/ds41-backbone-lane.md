# Owned backbone execution lane

`v41_backbone_lane.rs` assembles one mHC block, query wave, output-projection wave, shared-FFN wave and sparse-attention wave. It borrows a single immutable owner containing all 40 layers' corresponding weights. Creating a second lane reuses those weights and allocates independent mutable workspaces. Window/compressor/index/router weights, caches, candidate selection, engram, vision, dSpark and transport remain outside this owner.

The external index query can consume `query_output`, while an external selection owner keeps its selected rows across later backbone layers. This avoids tying candidate reuse or request-cache lifetimes to a mutable lane-wide borrow. Sparse attention validates those supplied views against the current query and layer before projection and mHC FFN preparation.

## Execution boundary

| Operation | Resulting phase | Work |
| --- | --- | --- |
| `restart` | Idle | Reset block publication and rebind projection/shared/mHC owners to layer zero |
| `begin_embedded` | Query | Copy text embeddings, run attention mHC and produce the bound query |
| `apply_engram` | Prepared | Apply the externally associated gather to a prepared layer |
| `begin_prepared` | Query | Require completed engram where applicable, then run attention mHC/query |
| `attention_ffn` | FFN | Execute sparse attention, output projection and FFN mHC; return stable normalized input plus access to shared execution |
| `LaneFfn::execute_shared` | Shared ready | Compute the shared contribution from that exact normalized input |
| `finish_ffn` | Complete | Require shared completion and the exact reduction binding; copy the combined result and finish FFN mHC |
| `advance` | Prepared | Rebind all three projection owners and advance mHC to the adjacent layer using the existing allocations |

The scheduler can build and dispatch routed work from `LaneFfn::input` before calling `execute_shared`. Its work handle retains the lane's normalized rows while those consumers run. The lane does not itself dispatch transport. The [split native TP4 API](ds41-tp4-dispatch-qualification.md) now provides `dispatch_ffn` and pending `finish` for this ordering; scheduler wiring and actual RTX shared-compute overlap remain open. `finish_ffn` requires the caller's completed shared-plus-routed result, including correct request order; its unsafe contract is not a replacement for scheduler association.

Mutating steps enter an invalid phase before touching components and publish the next phase only on success. This includes shared execution through its borrowed work handle. A phase mismatch, a failed rebind, an invalid reduction or a GPU error requires `restart` before further execution. Read-only attempts to inspect a not-yet-prepared engram input do not discard progress. Output borrows prevent safe callers from restarting or advancing during consumption. dSpark tap reads must finish before beginning the prepared query.

## Budget evidence

The official checkpoint headers and SM120 AOT metadata give a conservative device budget of **8,037,937,600 bytes** for the four weight groups across all 40 layers. Each loader's transient peak is included in its contribution. This excludes the other weight families listed above; it is not the model's total RTX weight footprint.

| Capacity | Five workspace groups, bytes per lane |
| --- | ---: |
| 1 | 5,047,628 |
| 80 | 61,750,284 |
| 4,096 | 2,949,251,084 |

The retained `official_lane_budget_rejects_before_device_allocation` test verifies these values through production planning methods. It rejects a weight budget one byte below the aggregate, a lane budget one byte below each capacity's requirement, missing layers and invalid capacities. The test passed in a container with no GPU device access: only the native library and static CUDA driver library dependency were mounted for metadata reads. This proves the tested rejection paths run before device allocation. The test is explicitly skipped unless `DS41RT_LANE_PLAN_LIBRARY` and the accompanying model-path setting are supplied; the recorded run supplied them.

The production daemon and exact test module build successfully. **The assembled lane has not executed on GPU.** The RTX driver/userspace mismatch still prevents its real-weight qualification. The graph-cache and mHC-reuse GPU qualifications remain separate evidence for their components. The next GPU checks must cover complete lane arithmetic, invalid-phase recovery, engram ordering, bound FFN reduction, adjacent transitions and independent lanes.

Remaining runtime work includes owned window/compressor/index/router composition, asynchronous TP4 dispatch and reduction, CED replay, request histories, dSpark verification and rollback, vision, concurrency-16 scheduling and API integration. This owner does not establish full-model readiness, memory fit or throughput. Exact source hashes and logs are in [the evidence record](ds41-backbone-lane.json).
