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

## TP2 expert staging

The loader now supports `BackboneTp2` with 1,152 logical intermediate channels.
It reads W1/W3 row halves and W2 column halves directly from the official payload,
including matching scale slices, using bounded reusable scratch and prefetch.
Each expert/rank needs 9,400,320 staging bytes. TP4, full-width, and dSpark reads
retain their existing contracts. Synthetic byte-level tests cover both TP2 ranks,
weights and scales, partial final read batches, invalid rank/layer/expert IDs,
undersized staging/scratch, and untouched trailing storage. All four catalog tests
and the full daemon offline check pass. This is loader support; TP2 packing,
kernel exports, reductions, and serving integration remain unfinished.

## TP2 packed layout and initial exports

The native packer accepts 1,152-channel halves. Packed sizes per expert are
`[5,898,240, 368,640, 2,949,120, 184,320]` bytes: 9,400,320 total, with no
intermediate padding overhead. Thus all twenty encoder layers at 384 experts
cost exactly 72,194,457,600 packed weight bytes per GPU before other allocations.

`qualify_v41_expert_packing.py` compares all four native packed arrays against
SparkInfer's independent tensor converters for widths 576, 1152, and 2304 on
both GPUs, including trailing canaries. All six cases pass byte-for-byte. The
scale test spans bytes 0–247; SparkInfer clamps larger values, whereas the native
packer preserves raw checkpoint bytes, so that intentionally different domain
is excluded from exact equality. Existing native representations are unchanged.

The slice exporter supports role `rtx_tp2` (ABI role 3, 384 experts, top-6,
FP8 K32 input, 1,152 intermediate channels). C1 and C16 SM120 kernels export
successfully at width 192. They have not yet been executed or performance-qualified.
The native expert handle table currently binds each variant to its first GPU;
TP2 needs independent per-device handles before integration. Other capacities,
shared-expert TP2, reductions, and complete serving remain pending.

## TP2 native handle ownership

A separate `ds41rt_v41_tp2_expert_*` interface now provides independent module
tables for the two coordinator-visible devices (0/1). Each capacity has distinct
GPU0/GPU1 handles; wrong-device scratch binding, scratch initialization, and
launch are rejected. The two selected physical GPUs must be exposed as devices
0/1 by the eventual launcher. Full-width and dSpark interfaces remain separate.

The C1/C16 TP2 library builds against the exported objects. Its handle fixture
passes on both cards, checking distinct handles, stable repeated initialization,
owner-local scratch initialization, and rejection of crossed handles without
slot mutation. C1 scratch is 877,600 bytes; C16 scratch is 13,925,776 bytes per
execution workspace. CMake now offers `DS41RT_ENABLE_V41_TP2_EXPERT_AOT` alongside
the coordinator expert build with capacities 1/16/80/256/1024/4096. The full
enabled CMake build, remaining capacities, Rust bindings, and numerical expert
execution have not yet been qualified.

## Initial TP2 numerical execution

Rust FFI now selects and validates the TP2 role and geometry, and the daemon's
expert placement enum connects TP2 staging to its packer, metadata, and kernels.
The full daemon passes offline `cargo check`; dual-device owner integration into
the main serving loop remains pending.

`qualify_v41_tp2_numerics.py` executes both native halves at C1 and C16 using
synthetic full-width expert weights sliced consistently for W1/W3/W2 and scales.
Summed rank output is compared with the unsplit FP32 oracle using the official
BF16/FP8 projection sequence. All four initial/changed cases pass: relative L2
is about 0.17%, cosine exceeds 0.999997. Changed-input graph replay updates
input values, expert IDs, and routing weights without allocating during replay.
Each GPU uses an explicit device-local capture stream; relying on the Python
graph helper's default stream caused a wrong-device rejection on the second GPU.

This validates C1/C16 synthetic expert numerics, not real checkpoint quality,
parallel lane progress, device-side final reduction, prefill capacities, shared
experts, or throughput. The fixture sums rank outputs on the host for comparison;
serving must use an asynchronous device reduction.

## Device-side TP2 reduction

`ds41rt_v41_reduce_tp2_experts_async` and its Rust owner now reduce two FP32
rank contributions on the destination GPU, supporting six route planes or one
pre-accumulated token plane. Rank pairs and routes accumulate in FP32 before
one BF16 output conversion. Both inputs must already be resident and ordered
on the destination stream; serving still needs to connect the peer-copy events.
The existing single-GPU local-expert reduction is unchanged.

The reduction fixture passes 12 combinations across both GPUs, rows 1/16/4096,
and both layouts. It checks exact BF16 results, changed-data graph replay,
trailing canaries, input/output overlap rejection, and invalid row counts.
The C1/C16 expert fixture now invokes native GPU reduction rather than host
summation and still passes all four comparisons against the unsplit oracle:
relative L2 is about 0.23% after final BF16 rounding. Its cross-device staging
currently uses PyTorch for the fixture; it does not qualify the serving transfer
loop or asynchronous lane progress. Native build and daemon offline check pass.

## Chained peer reduction owner

`PeerReduction` now connects the native peer-copy and FP32 reduction through one
preallocated destination stream. A remote producer event orders the copy; a
local producer event orders reduction after both inputs. The owner polls only
after the whole chain is queued, with no host completion barrier between copy
and reduction. Its callback and input/output storage remain retained through
completion or cancellation drain. Separate owners can target either GPU.

Worst-case destination storage is `capacity * 5120 * 26` bytes for six FP32
peer routes plus BF16 output, excluding stream/event overhead. A rank that
already emits token sums copies only its live FP32 token bytes. This owner is
not yet connected to the serving layer loop; full lane progress and a deliberate
pending-transfer cancellation test remain necessary.

The Rust CUDA fixture passes with two opposite-direction reduction owners joined
cooperatively on one host thread. It checks changed values, C1/C16, both route
and token-sum layouts, correct BF16 results on both devices, and restoration of
the caller's device. It does not deliberately stall either producer. The daemon
offline check also passes.

## Rust TP2 expert execution owner

`RankWeights` loads encoder TP2 layers inside explicit device scopes and frees
them on the same devices. Immutable rank weights can be shared by both lanes.
`RankWave` owns independent streams, capacity variants, scratch, and FP32 output
storage. `ExpertWave` enqueues both ranks, then chains peer transfer/reduction
onto either requested output GPU. Error/cancellation guards drain partially
enqueued ranks before input borrows can be released. Its execute path performs
no allocation. This backend is not yet selected by the serving layer loop.

The real-checkpoint Rust fixture passes with layer 0 loaded as two TP2 halves,
two concurrent lane owners sharing those weights, and opposite output devices.
C16 zero input produces zero; changed nonzero input produces finite nonzero
outputs identical on either destination GPU. This verifies loading, execution,
ownership and reduction plumbing; it is not a full-model quality or throughput
qualification. Current rank workspaces copy FP32 kernel scratch output into a
stable preallocated buffer before peer reduction; account for those copies and
buffers when measuring and planning memory.

A deliberate stalled-producer CUDA test also passes: the peer chain returns
Pending, another stream remains ready, and dropping the future drains both copy
and its queued follow-up before returning. The test releases the producer after
50 ms and checks final data and device restoration. Cancellation can block for
cleanup; normal progress still polls cooperatively with no cross-lane join.
