# Owned backbone execution lane

`v41_backbone_lane.rs` assembles one mHC block, query wave, output-projection wave, shared-FFN wave, sparse-attention wave and router wave. It borrows a single immutable owner containing all 40 layers' corresponding weights. Creating a second lane reuses those weights and allocates independent mutable workspaces. Window/compressor/index weights, caches, candidate selection, engram, vision, dSpark and transport remain outside this owner.

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
| `LaneFfn::execute_tp4` | Shared ready | Route the preserved input, dispatch all four requests, compute shared FFN, then collect and reduce the exact bound result |
| `finish_ffn` | Complete | Require shared completion and the exact reduction binding; copy the combined result and finish FFN mHC |
| `advance` | Prepared | Rebind all three projection owners and advance mHC to the adjacent layer using the existing allocations |

`LaneFfn::execute_tp4` now supplies the routing/dispatch/shared/collection sequence using the [split native TP4 API](ds41-tp4-dispatch-qualification.md). Its work handle retains normalized rows throughout. The scheduler supplies placement, request metadata and modality mask in their actual input order. The lower-level input and shared-execution entry remain available for explicit composition. Actual RTX overlap and full scheduler integration remain open. `finish_ffn` requires the completed shared-plus-routed result with its exact binding; the unsafe metadata contract is not a replacement for scheduler-owned association. The new method and router reuse have a separate [qualification record](ds41-lane-ffn-qualification.md).

Mutating steps enter an invalid phase before touching components and publish the next phase only on success. This includes shared execution through its borrowed work handle. A phase mismatch, a failed rebind, an invalid reduction or a GPU error requires `restart` before further execution. Read-only attempts to inspect a not-yet-prepared engram input do not discard progress. Output borrows prevent safe callers from restarting or advancing during consumption. dSpark tap reads must finish before beginning the prepared query.

## Budget evidence

The official checkpoint headers and SM120 AOT metadata give a conservative device budget of **8,195,346,880 bytes** for the five weight groups across all 40 layers, including routing. Each loader's transient peak is included in its contribution. This excludes the other weight families listed above; it is not the model's total RTX weight footprint.

| Capacity | Six workspace groups, bytes per lane |
| --- | ---: |
| 1 | 5,059,453 |
| 80 | 62,696,284 |
| 4,096 | 2,997,686,284 |

The retained `official_lane_budget_rejects_before_device_allocation` test verifies these values through production planning methods. It rejects a weight budget one byte below the aggregate, a lane budget one byte below each capacity's requirement, missing layers and invalid capacities. The test passed in a container with no GPU device access: only the native library and static CUDA driver library dependency were mounted for metadata reads. This proves the tested rejection paths run before device allocation. The test is explicitly skipped unless `DS41RT_LANE_PLAN_LIBRARY` and the accompanying model-path setting are supplied; the recorded run supplied them.

The production daemon and exact test module build successfully. **The assembled lane has not executed on GPU.** The RTX driver/userspace mismatch still prevents its real-weight qualification. The graph-cache and mHC-reuse GPU qualifications remain separate evidence for their components. The next GPU checks must cover complete lane arithmetic, invalid-phase recovery, engram ordering, bound FFN reduction, adjacent transitions and independent lanes.

Remaining runtime work includes owned window/compressor/index composition, scheduler use of asynchronous TP4 dispatch/reduction, CED replay, request histories, dSpark verification and rollback, vision, concurrency-16 scheduling and API integration. This owner does not establish full-model readiness, memory fit or throughput. The [original pre-router evidence](ds41-backbone-lane.json) retains its four-weight/five-workspace scope; [current FFN evidence](ds41-lane-ffn-qualification.json) includes routing and the updated allocation checks.
