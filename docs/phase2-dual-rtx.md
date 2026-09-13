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

The user's explicit placement map assigns vision, embeddings, encoder attention
and SWA (0–19), and sources 2/8/14 to RTX0. RTX1 owns decoder attention and SWA
(20–39), source 20, and dSpark. Both cards hold half the vocabulary rows, TP2
encoder routed experts, and TP2 shared experts for all layers. CPU Engram stays
unchanged; Sparks retain decoder routed experts only in two-GPU mode.
The user subsequently authorized rebalancing non-expert attention/dense and
auxiliary placement instead of requiring the 19/20 boundary. Preserve expert
TP2 and vocabulary partitioning, keep cache consumers colocated where practical,
and account for extra transfer costs when moving layers. A reduced KV pool is
acceptable for initial integration before selecting the final balanced budget.
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

## First implementation checkpoint

`v41_native_serve/memory/distributed.rs` now computes page-group capacity from
per-device residual budgets and an explicit source-owner map. It preserves
whole-group rounding, checks minimum admission and maximum physical capacity,
and reports unused memory independently for each card. This is startup accounting
only; it is not wired into allocation or serving yet. Four standalone Rust tests
pass, covering either GPU as bottleneck, exact-size rounding, limits, invalid
owners, and arithmetic overflow. The full daemon passes offline `cargo check`.

The host reports a PCIe `NODE` connection between the two RTX cards, with peer
read and write support in both directions (`nvidia-smi topo -p2p r/w`). This
establishes advertised capability, not measured transfer bandwidth or asynchronous
CUDA ownership correctness. Those require a real transfer fixture.

Current cache binding selects the latest source at or before the attention
layer: source 2 serves layers 2–7, source 8 serves 8–13, source 14 serves 14–19,
and source 20 serves 20–39. Layers 0–1 use only their sliding windows. Therefore
the requested 3/1 split naturally groups encoder attention on one device and
decoder attention on the other; TP2 FFNs still require transfers within each
layer. Existing local expert execution only accepts full-width weights, so TP2
requires new packing/kernel ownership rather than relabeling existing buffers.

## Device-transfer checkpoint

Native and Rust FFI now expose explicit current-device selection, idempotent
direct peer enablement, and asynchronous peer copies on a destination stream.
The existing single-device copy path is unchanged. Source readiness can use the
existing cross-device stream/event dependency; no device-wide wait is introduced.
These low-level APIs still require caller-owned buffer lifetimes and are not yet
wired into model execution.

`ds41rt_cuda_peer_selftest` passes on the RTX pair: three changed byte patterns
are copied and checked in each direction, with an unrelated stream deliberately
held pending until the copy completes. It also checks invalid extents, null
streams, wrong current devices, buffer device IDs, and repeated peer enablement.
The full daemon passes offline `cargo check`. This proves the transfer primitive,
not graph replay, cancellation ownership, bandwidth, or full serving performance.

The first fixture launch could not create a CUDA context beside the v2 server
(only 305 MiB free on RTX0). The v2 coordinator container was stopped to free
the pair for phase-2 development; its image and container remain available.

## Device ownership and weight accounting

`v41_memory/device.rs` provides explicit allocation/stream/event owners and a
preallocated peer-transfer direction per lane. Device selection is scoped to
synchronous enqueue/query calls and restored before cooperative yields. The
transfer future borrows source, destination, and producer; its drain guard runs
on errors or cancellation. Legacy single-device owners are unchanged.

The ignored CUDA owner test passes on both cards: simultaneous opposite-direction
copies preserve three changed input patterns; device selection restores after an
error, concurrent completion, and destruction. The test needs the local Python
3.12 library directory in `LD_LIBRARY_PATH`. Cancellation drain behavior still
needs an explicit pending-transfer test; model integration remains pending.

Initial exact checkpoint-header sums (decimal bytes, **not runtime occupancy**):

| Tensor group | Bytes |
|---|---:|
| Encoder routed experts, all 20 layers | 144,388,915,200 |
| Encoder routed experts per TP2 half | 72,194,457,600 |
| Shared experts, all 40 layers | 1,416,960,000 |
| Encoder non-expert, excluding Engram namespace and shared experts | 2,741,203,936 |
| Decoder non-expert, excluding shared experts | 2,725,876,192 |
| dSpark (`mtp`) | 7,932,874,632 |
| Embeddings | 1,323,827,200 |
| Vocabulary head | 1,323,827,200 |
| Vision, aligner, image marker vectors | 970,536,960 |

Ordinary non-expert layers contribute 134,631,128 checkpoint bytes each.
These sums come from all 48 safetensors headers in official snapshot
`dba1be0a40aa45a94ad051997016db3960a90277`. Runtime planning must additionally
account for transformed packing/scales, GPU Engram projections (separate from
the host table), duplicate/aliased tensors, all lane and prefill workspaces,
SWA/index/KV pools, snapshots, transfer buffers, and CUDA headroom. In particular,
moving a few ordinary attention layers saves far less than the dSpark allocation;
the final split must be chosen using the complete ledger.
