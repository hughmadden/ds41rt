# Phase 2: two RTX coordinators and four Sparks

Status: TP2 expert execution and peer-transfer foundations verified; dual-GPU serving is not yet implemented.

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
The clarified deployment baseline keeps the 19/20 attention boundary: layers
14–19 and compressed/index source 14 stay together on RTX0. Layers 15–19 are
not an independent balancing knob; do not introduce remote cache reads or
cache replication to move them. Keeping three sources on RTX0 leaves a smaller
compressed pool on RTX1 to accommodate the planned unsplit dSpark model there.
Vision placement remains flexible. Preserve expert TP2 and vocabulary
partitioning. A reduced KV pool is acceptable for initial integration before
selecting the final budget, which must include dSpark and all runtime storage.

The layer-14 boundary used in the integration checkpoints below exercises
source/cache ownership on GPU1. References to that map as "rebalanced" describe
the test arrangement, not an established memory improvement or the deployment
baseline. It moves source 14 onto the device already carrying dSpark and changes
compressed/index pool ownership from 60/40 to 40/60, worsening that imbalance.
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

## Shared-expert TP2 kernels

The FP8 exporter and native shape checks now support `[1152,5120]` up/gate
weights and `[5120,1152]` down weights. A separate 1,152-wide SwiGLU entry
preserves the full-width entry. FP8 matrix and mHC module ownership now allows
two device-specific module tables; the first device may retain any CUDA ordinal.
Module loading remains at initialization, not in graph replay.

C1/C16 shared TP2 projections export and build. The official layer-0 shared
weight fixture runs both halves on their own devices, including changed-input
graph replay, and checks distinct matrix handles. Summed BF16 half outputs
match the full-width quantized reference within about 0.23% relative L2, with
cosine above 0.999992. The daemon offline check passes. This fixture stages
weights and sums halves in Python; Rust shared-weight slicing, shared execution
ownership, peer reduction, other capacities, and serving integration remain
pending. No single-GPU throughput claim follows from these numerical checks.

## Rust shared-expert ownership and AOT library fix

The catalog supports bounded row/column TP2 reads for coordinator matrices,
with tests covering both ranks and axes, offsets above 2 GiB, trailing canaries,
and invalid extents/placement. Existing TP4 and full expert tests continue to pass.
The shared rank loader keeps native FP8 weights and expanded packed scales,
discarding temporary source scales after packing. Resident weights require
18,247,680 bytes per layer per rank, or 729,907,200 bytes for all forty layers
per GPU. Peak loading additionally reserves 5,760 device bytes for source scales;
pinned staging, CPU read scratch, execution workspaces and CUDA state are separate.

The Rust shared `RankWave` now preallocates FP8 capacity plans, activation buffers,
scratch and producer events. Two waves can share immutable weights on each GPU.
The official layer-0 C16 fixture passes with both cards loaded before execution,
zero/nonzero inputs, two independent waves per card, finite outputs, same-rank
lane agreement, and caller-device restoration. Shared cross-rank reduction and
serving selection remain pending.

This fixture exposed a limitation missed by the earlier Python ordering:
generated AOT launch symbols are process-global. Initializing a second CUDA
library overwrote the first GPU's kernel symbols and made its FP8 GEMM reject
launches. The final implementation uses **one CUDA library per exported variant,
configured on each GPU**, with separate device-bound handles. FP8, mHC and TP2
routed expert initialization now follow that model. Separate module libraries
per GPU described in earlier checkpoints are superseded. Both real shared and
routed Rust layer fixtures pass after this fix, as does the daemon offline check.

## Complete shared TP2 operation

The Rust shared `Wave` now owns both rank workspaces plus preallocated peer
staging and output on either GPU. It queues both halves, copies the remote
partial, waits for the local partial on the reduction stream, and adds on GPU.
Normal completion polls cooperatively; each lane owns its own streams, events,
and buffers. Error/cancellation drains retain rank inputs through queued work.
Reduction staging and output add 20,480 bytes per capacity row per GPU per lane,
separate from the rank execution workspaces. No loop allocation is introduced.

The combined official layer-0 fixture passes C1/C16, zero and changed nonzero
inputs, and two concurrent lane owners returning on opposite GPUs. Destination
results agree byte-for-byte, and each value matches a CPU round-to-nearest-even
sum of the BF16 rank outputs. The first run caught that the existing residual
addition truncates BF16; a dedicated TP2 addition entry now rounds to nearest,
without changing that existing single-device kernel. Native build and the Rust
fixture pass. These checks do not establish full-model quality, serving lane
overlap, or throughput; integration and larger prefill capacities remain pending.

## Prefill capacities and combined native build

TP2 routed and shared exports now have numerical evidence for capacities
1, 16, 80, 256, 1024, and 4096. The qualification tools accept `--rows` to select
exported capacities; routed reduction follows the export's route/token output
layout, including atomic token accumulation at 256 rows and above. Changed-input
graph replay passes at every capacity. Against the unsplit references, maximum
relative L2 is 0.239% for routed experts and 0.234% for shared experts, with
minimum cosine above 0.999997. Metrics accumulate in FP64 to avoid inaccurate
FP32 reductions over large prefill vectors.

The real-checkpoint Rust lane fixtures accept `DS41RT_TP2_TEST_CAPACITY` (default
16). At capacity 4096 they alternate one-row and full-capacity executions with
zero/nonzero inputs and opposite output destinations. Shared results match
exactly. Routed deterministic variants match exactly; independent atomic
executions show about 8e-9 relative L2 variation, bounded to one BF16 encoding
step per element and less than 1e-6 relative L2 by the fixture. Zero input still
produces exact zero after a nonzero execution.

A fresh CMake build with CUDA, coordinator expert AOT, full-width RTX expert
AOT, TP2 expert AOT, and FP8 AOT enabled succeeds. Both 4096-capacity Rust lane
fixtures also pass against that combined library, rather than only the isolated
test libraries. This establishes build coexistence and expert workspace reuse;
complete serving integration, distributed attention/cache placement, vocabulary
partitioning, automatic launcher selection, and full-model qualification remain.

## Serving FFN connection and copy-engine independence

`v41_experts/tp2_ffn.rs` now combines routed and shared TP2 execution for one
encoder FFN. It borrows the completed normalized/router buffers on the owning
GPU, copies only the peer inputs, runs both contributions concurrently, and
returns their rounded BF16 sum to the input GPU. Each request lane owns its
streams, peer buffers, expert workspaces, and output. Immutable weights remain
shared between lanes. Execution performs no allocation.

`NativeTp4Wave::install_tp2` and the serving lane's local FFN branch now select
this operation when installed, validating query binding, layer, row count and
token identity. Startup does not install it yet: complete placement, distributed
attention/cache ownership, decoder shared TP2, and the remaining phase-2 serving
components must still be integrated before enabling two-GPU mode.

The official layer-0 fixture passes C1/C16 changed inputs with distinct values,
expert IDs, and routing weights in each lane. Each result exactly matches a
rounded sum of independently executed routed and shared contributions. It also
checks cancellation and subsequent reuse of the same workspace.

A deliberately held upload exposed copy-engine head-of-line blocking: queuing
copies behind the held dependency prevented unrelated peer copies and the other
lane from finishing until the 500 ms gate released, despite separate streams.
The TP2 path now polls its own upload/remote producer cooperatively before
submitting a not-yet-ready copy. The gated test then finishes the other lane in
about 0.23 ms, with the gate still held; independent peer copies also complete.
This is an isolation test, not a throughput measurement. No cross-request lane
join or blocking synchronization was added to normal execution. Cancellation
still drains queued work before releasing borrowed storage.

Per-GPU, per-lane workspace accounting now includes exported scratch sizes,
stable routed outputs, peer reductions, shared projections, peer inputs, and the
final FFN output. It excludes immutable weights, CUDA modules/streams, allocator
rounding, and runtime headroom:

| Row capacity | Explicit workspace bytes per GPU per lane |
| --- | ---: |
| 1 | 1,231,360 |
| 16 | 19,200,672 |
| 80 | 95,860,384 |
| 256 | 153,787,808 |
| 1024 | 404,235,680 |
| 4096 | 1,460,135,984 |

The transfer boundary adds 25,808 bytes per row per GPU per lane beyond the
shared/routed backends. At capacity 4096, two complete FFN lanes require
2,920,271,968 explicit workspace bytes on each GPU. Full placement must also
account for attention, cache/snapshots, auxiliary components, loading peaks,
weights, and CUDA overhead; these FFN numbers alone are not a serving budget.

## Distributed cache storage and scoped device execution

`BackboneCache::new_distributed` now allocates each layer's FP8 SWA and each
compressed source's FP4 KV, index data, page tables, lengths, and pending carry
on its assigned GPU. Per-device cache byte accounting checks both budgets before
allocation. The existing constructor retains its single-device placement.
Shared request leases, versions, and publication metadata remain one CPU bank.

`CachePlacement` derives source owners from the attention map and validates
colocation across each source's consumer group: 2–7, 8–13, 14–19, and 20–39.
The original 0–19/20–39 split and a rebalanced map moving source 14 with its six
consumers both pass allocation and request lease/reuse checks on the actual GPUs.
Moving only part of a source's consumer group is rejected by this colocated path
and is outside the deployment plan: source 14 and layers 14–19 remain colocated.

Device owners now construct and destroy cache components in the proper context.
`Device::future` scopes every future poll and cancellation cleanup, restoring the
caller's device before yielding. A two-GPU fixture verifies alternating polls,
owned stream cleanup on cancellation, and owner destruction without leaving the
thread on another device. This allows existing attention components to be placed
without holding a thread-local CUDA device selection across an await.

Non-empty prefix fixtures also pass with cache storage on GPU1 and retained
snapshot storage/streams on GPU0: wrapped SWA spans survive slot reuse, odd
compressor carry rows survive, and complete-group compressed-prefix restoration
preserves its bounds. The initial distributed constructor therefore retains
snapshot-copy coordination on GPU0, while retained compressed pages stay on
their source GPU. Snapshot arena bytes must be charged to GPU0 separately.

The existing single-device request-bank lifecycle/isolation check and daemon
compile check pass. Running that previously environment-gated lifecycle test
exposed a stale assertion rejecting page counts above 65,536; it now checks the
existing 262,144-page limit and accepts 65,537, matching the current source cache.
The allocation and snapshot evidence does not yet cover full distributed
attention execution, end-to-end cache reuse, startup selection, or performance.

## Placed cache-producer weights and workspaces

Cache producer loading now follows `CachePlacement`: each SWA projection,
compressed-source weight set, and attention sink loads on its owning GPU.
Per-device weight budgets are validated first, and device owners preserve the
correct context during loading and destruction. Attention sink pointers are
resolved once at startup instead of formatting a tensor name and looking it up
at every layer.

`PlacedProducerWaves` allocates all forty SWA producer workspaces and four
compression workspaces on those same devices, independently for each lane.
Its budget splits the existing workspace calculation without counting another
GPU's free space. For the original encoder/decoder placement:

| Component budget, bytes | GPU0 | GPU1 |
| --- | ---: | ---: |
| Cache-producer weights | 85,998,336 | 59,519,232 |
| Producer workspaces per lane, C16 | 22,990,736 | 14,024,592 |

These figures exclude attention query/output projections, FFNs, KV/SWA storage,
snapshots, auxiliary components, and CUDA overhead. They are component budgets,
not the complete serving memory requirement.

The official-weight fixture verifies all producer/sink/input-buffer device
assignments. Layer-20 SWA and source-20 compression graph replay on GPU1 match
the same weights executing on GPU0 byte-for-byte, including changed inputs and
FP4 KV/index packing. Device-scoped futures restore the calling context after
each execution. The existing single-device loader and execution constructor also
initialize with every producer and sink on GPU0 after the dual-device checks.

The placed workspace owner is not yet selected by the complete layer loop.
The old execution constructor rejects distributed weights to prevent silently
allocating their workspaces on the caller's GPU. Integrating placed producer
polling/commit with attention, index, and layer-state ownership remains next;
no full-model quality or throughput claim follows from these component checks.

## Placed cache commit and retained-prefix recovery

The existing cache transaction validator and publication logic now accept both
ordinary producer waves and device-owned waves through static dispatch. Ordinary
waves retain their direct calls; placed waves enqueue, query, publish, and abort
inside their GPU scope. Validation also rejects a producer/cache GPU mismatch
before submitting accepted writes.

Placed producer lanes expose separate enqueue, poll, finish, and abort methods.
Only enqueue borrows the bank immutably; polling borrows the lane alone, and
publication/abort take a short mutable bank borrow. No mutable request-bank
borrow spans an async wait. A fixed-size acceptance record binds completion to
the original batch/counts and introduces no allocation for that tracking state.
The underlying existing commit operations retain their existing staging and
reservation behavior.

The real-weight two-lane fixture uses the rebalanced map with source 14 and its
consumers on GPU1, covering both compression ratios there. Accepted bytes across
all forty SWA windows and all four FP4 KV/index sources exactly match direct
single-GPU execution. It verifies distinct requests/acceptance counts, deferred
logical publication, rejection of changed acceptance, and continuation across an
odd compressed frontier. After retaining a prefix, an aborted copy-on-write
append revokes only the affected request; the peer remains byte-identical.
Restoring the retained prefix and accepting the next row matches direct
execution, including the saved carry state.

The existing larger single-GPU commit fixture also passes: 16-request CED
transactions, bounded replay, encoder reservations/publication, late source-pool
exhaustion, lease revocation, and recovery. These checks cover cache transactions;
the complete distributed layer loop, attention/index placement, startup mode
selection, and serving performance qualification still remain.

## Placed asynchronous query production

Placed producer workspaces now accept completed attention-query outputs and
return a pending owner that borrows only the current lane. Enqueue, stream
queries, graph capture/replay, and cancellation cleanup run on the assigned GPU;
each call restores the caller's CUDA device. The shared cache bank is supplied
only during enqueue/poll and is not retained across waits. Device and batch
checks reject mismatched inputs. Dropping unfinished production drains that
layer's producers before their storage can be reused.

Ordinary layers enqueue SWA and their compressed source when applicable. At the
encoder boundary, layer 20 enqueues only source compression, preserving the CED
separation from decoder SWA execution. Replayed phases keep their existing
compressed sources. This owner adds no new cross-lane barrier; underlying
producer preparation retains its existing staging behavior.

The official-weight CUDA fixture passes for layers 2 on GPU0 and 14 on GPU1,
and for source-only layer 20 on GPU1. Queued outputs match direct production
byte-for-byte for SWA values/scales and compressed FP4 KV/index values/scales,
including changed-input graph replay. Cancellation followed by immediate reuse,
wrong-batch rejection, shared-bank access while pending, and caller-device
restoration also pass. This is not an end-to-end serving or throughput result.
The full execution loop still requires attention/index/layer-state placement
and selection of these placed producers.

## Placed learned index and producer overlap

Index query weights now load beside their attention/cache consumer groups. Each
request lane owns separate index workspaces on the participating GPUs, and each
GPU advances only through its assigned index layers. Source 20 and layers
24/28/32/36 retain their candidates on the same GPU. Decoder restart skips GPUs
without decoder index work. The ordinary constructor retains all eight layers
on its original device.

The placed producer owner now also accepts a placed index lane: it enqueues
index projection alongside SWA/compression, polls the three producers separately,
and starts selection after their required outputs complete. It retains no bank
borrow across a wait. Cancellation drains index consumers before cache producers.
The producer and index weights have independent borrow lifetimes.

The first hardware check exposed a single-device restriction in the native index
scorer. Its initializer now configures one canonical AOT CUDA library on two
devices, preserving the generated global launch symbols and already initialized
devices if a later initialization fails. Normal initialized calls avoid the
initialization mutex. Direct scorer graph tests initialize both devices, then
execute on GPU0/GPU1/GPU0 with changed queries; exact expected finite scores and
masked negative infinities pass.

For the rebalanced boundary at layer 14, component budgets are:

| Index component, bytes | GPU0 | GPU1 |
| --- | ---: | ---: |
| Query weights | 11,479,040 | 34,437,120 |
| Workspace per lane, C16 | 11,259,012 | 17,911,940 |
| Workspace per lane, 4096-row capacity | 1,809,562,628 | 3,512,712,196 |

GPU0 omits the unused decoder reindex workspace, saving 1,703,149,568 bytes
(1.59 GiB) per 4096-row lane compared with allocating both selection workspaces
there. These budgets exclude attention, producers, FFNs, and persistent caches.

The official-weight combined producer/index fixture passes all eight index
layers across both GPUs, with changed hidden inputs, cancellation/restart,
source-20 candidate retention through decoder reindex, and exact packed-query,
head-weight, and selected-ID parity against direct execution. Its short context
checks binding and placement; the separate scorer graph fixture explicitly runs
the scoring kernel. Existing index budget checks also pass. Full distributed
attention/layer-state execution and serving performance qualification remain.

## Placed backbone state and adjacent-layer handoff

Backbone mHC, attention query/output, and router weights now load on the GPU
assigned to each attention layer. A placed lane allocates its reusable block,
query, projection, sparse-attention, and router workspaces on that GPU. Full
shared-expert weights and their workspace are absent from placed lanes: the TP2
rank owners provide them separately. Ordinary loading still includes all forty
full shared experts and their existing workspace.

For the rebalanced layer-14 boundary, backbone weight budgets excluding shared
experts are 1,902,329,296 bytes on GPU0 and 3,532,897,264 bytes on GPU1. Omitting
the duplicate full shared weights avoids 1,461,196,800 additional bytes across
the two GPUs; this does not remove the separately required TP2 shared weights.
Placed backbone workspace at C16 is 32,742,680 bytes per GPU per request lane.
This remains component accounting, not a complete startup memory ledger.

Each directed layer handoff uses its own lane's transfer stream and copies
completed residual/pre state directly into the destination block's existing
input buffers. There is no intermediate device allocation or host staging.
Stream polling scopes the destination CUDA device and restores it before yield;
cancellation drains before borrowed buffers can be reused. Destination state is
published only after both copies finish. Token storage is reserved with the
block capacity at construction. Layer-14 inputs remain gated on Engram work.
The reusable backbone lane now exposes this handoff and rebinds its attention
query/output and router to the imported layer. Ordinary `advance` rejects a
cross-device successor so it cannot accidentally execute the wrong placement.

The official-weight handoff fixture checks 16/1/16 changed rows against direct
same-device initialization, through both the block and placed backbone lane.
Normalized/projected/rotated attention-query bytes match exactly. A deliberately
held stream proves pending cancellation drains, leaves no published prepared
state, and permits immediate reuse with unchanged buffer addresses and token
capacity. The Engram gate at the rebalanced boundary is also preserved.

Router execution exposed additional single-device native ownership in the
router-score and expert-input quantizer exports. They now configure canonical
CUDA libraries on both devices; quantizer handles retain their device identity.
Router graph fixtures for layers 0, 14, and 20 compare scores, expert IDs, routing
weights, and quantized inputs byte-for-byte against direct execution, including
changed 16/1/16 rows. Both-GPU checks and ordinary loader/lane initialization and
restart pass. The lane allocation guard test now uses the linked AOT library's
scratch extents instead of stale totals from an older export build.

The complete serving layer loop still needs to select these placed lanes and
producers, use the peer handoff at placement boundaries, and wire decoder shared
TP2 before distributed decoder FFNs can execute. The existing FFN finish path
also still copies next-layer inputs locally; a boundary-specific finish can
omit that redundant local copy when integrating the handoff. End-to-end serving,
loading speed, and one-/two-GPU throughput remain unqualified.

## Decoder shared TP2 with dispatched Spark work

The existing TP2 FFN lane now exposes a shared-only operation for decoder layers.
It reuses its shared rank workspaces and normalized-input peer buffer, so this
adds no GPU scratch allocation. Only normalized BF16 rows cross to the other
rank; router IDs, routing weights, and expert-wire inputs stay out of this copy.
Both shared ranks reduce onto the input GPU. The upload waits cooperatively for
its own stream before submitting peer DMA, preserving the independent-lane rule.

`NativePendingFfn` now retains access to the lane's TP2 workspace after Spark
routed-work dispatch. Its TP2 completion path validates the block/request/device,
executes the shared contribution, then uses the existing ordered Spark-plane
reduction with that completed shared buffer. The placed backbone lane selects
this path when shared TP2 weights cover the decoder layer. The ordinary full
shared-expert path remains available. The coordinator must be allocated on the
same GPU as the decoder input/output; startup selection still needs to enforce
that placement.

The real-weight fixture now loads all forty shared rank weights and checks
layers 20 and 39 with changed 1/16/1-row inputs from either GPU. Shared-only peer
broadcast and reduction match the independently invoked shared TP2 operation
byte-for-byte. A held upload in one lane leaves the other lane's decoder shared
work free to complete; pending cancellation drains and subsequent reuse passes.
The existing encoder FFN parity and independent-lane checks pass in the same
fixture. These tests establish the local computation and ownership behavior;
the new coordinator completion branch still needs a live Spark-backed decoder
run after the distributed serving loop and startup are connected. No end-to-end
throughput or tool-quality claim follows from this checkpoint.

The existing rank-upload/reduction regression also passes through 4096 rows,
including interleaved chunks, bounds rejection, cooperative cancellation, and
reuse. The daemon compile check passes.

## Connected distributed layer execution

`DistributedExecution` now owns one lane's placed cache producers and pass
progress. It connects asynchronous producer/index work to the GPU-owned
attention lane, returns prepared FFN work without retaining a cache-bank borrow,
and completes the layer on its owning device. Batch/layer progress prevents a
partial pass from publishing accepted cache writes. Encoder boundary source-only
production has an explicit completion transition. Reserved encoder batches are
rejected until their early-publication flow is connected, rather than silently
changing publication semantics.

Prepared attention has device-scoped destruction as well as device-scoped future
polling: cancelling before the FFN future's first poll must still drain queued
GPU1 attention on GPU1. Ordinary non-indexed attention now shares the same queued
consumer implementation as indexed attention. At a placement boundary, mHC
completion omits the local next-input copy; the directed peer transfer consumes
its output and fills the next GPU's existing block storage directly. Same-GPU
completion retains its local copy path.

A real-weight fixture executes layer zero entirely on GPU1 through embedding,
cache production, sparse attention, query/output projections, routed/shared TP2,
and final mHC. Changed 1/16/1-row residual/pre outputs match the ordinary GPU0
execution path using the same TP2 arithmetic byte-for-byte. It then transfers
the output to GPU0 layer one and verifies the required Engram gate. Dropping an
unpolled prepared-layer execution and restarting also passes, with caller-device
restoration; partial-pass cache commit is rejected. The fixture supplies distinct dummy transport
endpoints and executes the local encoder branch without using a Spark result.

This is a connected single-layer check, not a complete forty-layer serving run.
The target-pass loop still needs placed Engram/taps/head ownership, all layer
transitions, reserved encoder publication, and startup mode selection. Indexed
attention and live Spark-backed decoder execution need qualification through the
connected execution owner. Full-model performance and release tables remain
pending.

## Placed Engram projection and upload

Engram gate weights now follow attention layers 1 and 14. `PlacedEngram` owns
two device-scoped gates and one gathered-row upload workspace per participating
GPU, per request lane. Gates on the same GPU reuse that lane's upload workspace;
split gates have separate uploads. CPU table gathering and its request/history
validation remain unchanged. The caller retains the gather lease while upload
and residual gating run under a device-scoped future.

Application validates the gathered layer index, the lane's pending Engram layer,
and the GPU assignment before upload. It invokes the existing cooperative
upload and gate operations, preserving cancellation cleanup and the prepared
state transition. No cross-lane owner or wait is introduced.

Actual weight allocations and exact-budget workspace construction pass for the
original split and the rebalanced layer-14 boundary:

| Engram component, bytes | Original GPU0 | Original GPU1 | Rebalanced GPU0 | Rebalanced GPU1 |
| --- | ---: | ---: | ---: | ---: |
| Gate weights | 324,874,240 | 0 | 162,437,120 | 162,437,120 |
| Per-lane workspace, C16 | 5,266,488 | 0 | 2,782,244 | 2,782,244 |

The original placement saves one 298,000-byte C16 upload workspace by sharing
it between its two gates. A budget one byte below the required GPU0 allocation
is rejected before gate/workspace allocation.

The existing real-weight block/lane handoff fixture now continues through the
GPU1 layer-14 Engram gate. With synthetic gathered FP8 rows and mixed text masks,
changed 16/1/16-row residual/pre outputs exactly match the ordinary upload/gate
path and become available for subsequent attention. This verifies placed GPU
projection/application; it is not a new CPU table-lookup or full-model quality
evaluation. The forty-layer target-pass loop still needs to invoke this owner
along with placed taps/head and reserved encoder publication.

## Full distributed pass integration

`DistributedTargetPass` now sequences all forty layers through placed embedding,
Engram, cache/index production, attention, TP2 FFNs, peer handoffs, decoder taps,
and the target head. Each request lane owns its pass and transport; request-bank
borrows end before asynchronous waits. The full vocabulary head is temporarily
on the decoder GPU. Reserved encoder early publication and normal serving
startup integration remain pending.

The real-model `distributed_target_prefill_decode_commit_smoke` fixture passes
with two RTX cards and all four live Spark endpoints. It loads all twenty encoder
expert layers and all forty shared experts as TP2, places the attention boundary
at layer 14, runs a four-token prefill and one-token decode, and publishes both
cache/history commits. It also discards a subsequent completed proposal without
advancing committed history, then successfully executes and commits another
decode using the same owners. Discard invalidates each GPU's local lane rather
than attempting to restart GPU1 at GPU0's layer zero. The fixture uses a small
cache and synthetic token IDs: it establishes execution and lifecycle behavior,
not semantic quality, throughput, or concurrent-lane performance. Its native
build requires `DS41RT_ENABLE_RDMA=ON` for the live decoder path.

## Queued encoder cache publication

Early encoder publication now accepts device-owned window and compressor waves.
It can enqueue writes on their assigned GPU, return the request-bank borrow,
and publish after cooperative polling. Publication rejects unfinished writes
before changing publication masks or request completion. Direct single-GPU
callers retain their synchronous path; accepted prompt counts use a fixed
sixteen-request array.

The placed cache transaction fixture verifies two reserved chunks of three and
four tokens against direct publication, including ratio-two carry across the odd
boundary. All twenty encoder windows and four compressed sources match exactly
with source 14 and source 20 on GPU1. Publication leaves request completion
unchanged until the enclosing commit. Duplicate publication is rejected, and
aborting queued GPU1 window/source writes preserves a disjoint peer request.
The distributed target loop still rejects reserved chunks until this primitive
is connected to its chunk scheduling and cancellation guards.

## Reserved chunks through the distributed target loop

The distributed loop now accepts reserved encoder chunks. Compressed sources
publish cooperatively before index selection, so selection and subsequent
attention consumers share the committed causal snapshot. Window publication
follows the completed layer. Each poll releases the request-bank borrow; the
reserved-pass cancellation guard drains pending writes before revoking admission
and invalidating both GPU-local lanes. Source 20 publishes at the encoder
boundary, and final chunk commit advances cache and Engram history together.

The full real-model fixture passes reserved chunks of three and four tokens,
then replays the seven retained encoder rows through the live Spark decoder.
Its greedy token matches a full seven-token prefill, with the winning score
within 0.25 logits. Cancelling at a cooperative suspension revokes the reserved
request; the same GPU owners then successfully execute and commit a fresh full
pass. Ordinary full prefill/decode, commit, and discard/reuse checks also pass.
These are short integration checks, not release quality or throughput results.
Interleaving encoder chunks across two request lanes and connecting the normal
serving scheduler remain pending.

## Interleaved distributed encoder chunks

The two distributed lanes can now reserve, execute and commit alternating chunks
of one prompt. A chunk waits for its predecessor's per-layer cache publication;
suffix capture, source-20 publication and history completion remain ordered for
that prompt. Each lane reserves its next chunk after its own commit, without a
pair barrier. Request-bank borrows end before waits. Suffix capture uses a stream
owned by that lane on the encoder-output GPU, with cooperative completion and
drained cancellation. Stream failure revokes the prompt after both active chunk
guards drain their work.

Interleaving exposed a sparse-attention validation bug: a committed source's
device length can advance after an earlier causal view is queued. Requiring
exact length equality caused valid prefix readers to return zero attention.
Committed-only views now accept an appended source; private overlays still
require an exact boundary. A focused native regression on both GPUs checks
ordinary and split attention, equal output after append, rejection of shortened
backing, and rejection of stale private overlays.

The real-model integration fixture compares sequential and interleaved execution
for identical 3+4 and 1+2+1+3 chunk partitions. Winning token and score match
exactly after live Spark decoder replay. Disconnecting during active encoder
work revokes both chunks, and subsequent execution reuses the owners. These
short fixtures do not establish release quality or throughput. Publication still
follows each layer's FFN completion; overlapping a follower's attention with that
FFN and connecting the normal serving scheduler remain further work.

### Follow-up: interleaving remains under correctness investigation

A subsequent untraced integration run failed the four-chunk comparison: both
paths selected token 200, but the winning scores were 17.553558 and 21.930965.
The earlier passing runs therefore do not establish reliable interleaved
execution. Layer tracing, first synchronous and then lane-local asynchronous,
passed without reproducing the failure. Diagnostic readback changes timing and
is not qualification evidence. `DS41RT_TRACE_ENCODER` enables the temporary
test-only trace; ordinary runs leave it disabled. The failure log is
`~/.cache/ds41rt-experiments/phase2-planner/distributed-serving-output.log`.
Resolve this intermittent difference before treating the stream as qualified
or connecting it to production serving.

The investigation also exposed an independent host validation error: an older
chunk's committed-prefix view was rejected while a follower held an append
reservation. `committed_proposal` now permits that immutable published prefix;
future extents and mutation/release during a pending append remain rejected.
The native ratio-two/ratio-one compressor commit regression passes, including
those guards and existing carry, direct-value, and abort comparisons. A later
16-case untraced integration run passed, but a prior run with this prefix fix
still failed. The numeric interleaving issue therefore remains unresolved;
no throughput improvement is claimed for this correctness change.

Removing the temporary native invalid-attention logging reproduced failures
with both placements. The deployment boundary at 20 failed the 3+4 partition
(token 200 in both paths, scores 15.589296 versus 16.83005); boundary 14 failed
the four-chunk partition with different winning tokens (74 versus 200).
Logs are `interleave20-clean.log` and `interleave14-clean.log` in the same
experiment directory. The fixture now defaults to boundary 20 and accepts
`DS41RT_ATTENTION_BOUNDARY=14` for alternate ownership coverage. Neither
placement is qualified for interleaved execution yet.

Deferred test tracing now queues pinned readback on each lane's mHC producer
stream without a per-layer host wait, and collects after execution. Selecting
only layer 19 reproduced differing encoder residuals for every chunk; selecting
layer 14 likewise reproduced differing residuals. A layer-0-only capture
reproduced the final mismatch with identical layer-0 residuals. This places the
observed divergence after initialization and before completion of the encoder;
decoder replay and the vocabulary head are not its first cause. Full-layer
tracing still masks reproduction. Logs are `interleave20-deferred19.log`,
`interleave20-deferred14.log`, and `interleave20-deferred0.log`.

Single-layer captures at 1/2/3/4 matched exactly in runs that still failed at
the head. Layer 5 captured a differing first-chunk residual, but capturing 4
and 5 together changed the schedule and both matched despite a later failure.
Thus the first affected layer depends on overlap timing; the evidence does not
prove a fixed layer-5 defect. A temporary invalid-view trap in sparse attention
did not fire in another reproducing run (`interleave20-invalid-trap.log`),
ruling out that kernel's descriptor rejection for that failure. The trap was
removed after the experiment. Continue with data ownership and overlap checks.

An independent-request comparison now gives each lane a separate cache lease
and compares retained encoder residuals against sequential execution. Four
concurrent pairs passed exactly for equal three-token requests and again for
unequal three/seven-token requests (`independent-encoder.log` and
`independent-encoder-unequal.log`). The current fixture enables the unequal case
with `DS41RT_INDEPENDENT_ENCODER_CHECK=1`. These short checks point toward
same-request cache overlap; they do not establish general concurrency quality.

The distributed pass also exposes device-scoped greedy/logit downloads and a
statically dispatched target-cache interface for queued dSpark commits. The
fixture checks finite full-vocabulary downloads, exact agreement with GPU
argmax, compact-row bounds, and restoration of the caller's GPU. Ordinary
serving still selects its existing target pass; installing the distributed
pass and placing the actual draft runtime remain outstanding.

Further isolation: separate-cache encoder requests also match exactly when the
second starts only after the first completes five layers
(`independent-encoder-staggered.log`). The native prefix test now queues 64
attention reads while another stream stores different FP4 values beyond the
prefix and updates its length; all outputs match on both GPUs, with ordinary
and split attention. This checks the primitive, not the complete cache
publication lifecycle. Capturing mHC pre output at layers 1 and 4 also matched
in runs that later failed (`interleave-pre1.log`, `interleave-pre4.log`). The
next useful boundary is attention/projection versus FFN within an affected
encoder layer; a residual-only end-of-layer trace cannot distinguish them.

Paired tracing now retains captures in GPU memory and downloads after the run,
avoiding diagnostic PCIe reads between layers. `DS41RT_TRACE_FFN=1` captures
normalized FFN input plus each layer's residual/pre output. This reproduced the
failure with complete traces: `interleave-vram-paired.log` first differs at
layer 7's FFN input; `interleave-vram-full-state.log` first differs at layer 8's
FFN input, with layer 7 residual and FP32 pre bytes both matching exactly.
The latter localizes that run's first difference before expert execution,
within attention/query/projection or its input preparation. It does not prove
a fixed offending layer or establish the root cause. Earlier trace logs
represented paired output stages as layer+40; the current formatter labels
stages and reports BF16 and FP32 differences separately.

Query tracing further narrows one failing execution to the second query
projection (QB). `DS41RT_TRACE_QUERY=1` captures query components before and
after cache production; `DS41RT_QUERY_COMPONENT=qb_pair` retains both projected
output and normalized input. In `interleave-qb-pair.log`, layer 13 has identical
three-row normalized inputs but differing projected outputs. All 274 differing
BF16 elements lie in columns 24832–24959, one 128-column tile. Both captures
already contain the difference before cache production. This supersedes the
earlier cache-overlap hypothesis for this particular failure; the underlying
cause is not yet established.

`DS41RT_TRACE_DUMP_DIR` optionally saves that first equal-input mismatch as
`qb.json`, `input.bf16`, `expected.bf16`, and `actual.bf16`, without overwriting
an existing capture. The local `qb-pair-capture` artifact reproduces the
sequential result exactly in an isolated native QB test, including paired
streams and graph replay. The isolated race check reports no hazards. Full
fixture memory checking (`interleave-memcheck.log`) reports only 20 repeated
`cudaErrorPeerAccessAlreadyEnabled` API results, with no reported invalid
memory access; the instrumented fixture passes, so this is not proof that the
concurrent mismatch is resolved. Full dense-kernel race checking with tensor
operation checks is the next investigation. An initial filter written as
`kns=regex:.*dense_gemm.*` matched no kernels and provides no race coverage;
the corrected filter is `kns=dense_gemm`.

The paired QB diagnostic also retains its complete quantization scratch before
cache production and reports byte differences, without interpreting FP8/scale
bytes as BF16. New equal-input mismatch dumps include `actual-scratch.bin` and
`expected-scratch.bin`. Scratch includes padding and possibly inactive rows;
a difference alone is not evidence of corrupted live inputs. Identical scratch
would further narrow an equal-input/different-output capture to the GEMM or
its weight/output lifetime. This additional diagnostic is test-only and has
compiled successfully; reproduction with it remains pending while the full
race check uses the preceding binary.

Vocabulary projection now has a native/FFI contiguous-shard constructor. Each
rank owns its cuBLAS handle and 4 MiB workspace; the existing full-head and
Markov constructors retain their dimensions and accumulation policy. The
standalone `native/tests/v41_vocabulary_shard_selftest.py` compared two
64,640-token shards on separate GPUs/streams against the full 129,280-token
projection using synthetic BF16 inputs/weights. Rows 1/3/16/80 and three graph
replays matched full logits exactly (maximum error zero); a separate FP32
reference checked columns on both sides of the boundary. CPU merging of local
winners reproduced global greedy IDs. Invalid shard sizes, row overflow, and
wrong-device launch were rejected. This is a projection primitive check, not
real-model quality or speed qualification. Distributed weight loading, GPU
winner merging, sampling/constrained outputs, and target/draft integration
remain required. Native compilation and `cargo check -p ds41rt-ffi` passed.
