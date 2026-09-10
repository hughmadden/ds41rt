# Query, output and shared-FFN workspace reuse

The three backbone projection wave owners now provide `rebind` methods. Query and output projection retain their existing scratch, activations, positions, rotary frequencies, stream and kernel handles while switching to another layer's immutable weights. The shared FFN also validates all three weight and packed-scale extents/devices before installing its binding. Rebinding invalidates published results and drains the owning stream. No device allocation or capture occurs in `rebind`.

Each wave uses `LayerGraphs`, a fixed 40-slot cache with one captured shape per layer. A graph is reusable only for the exact borrowed weight owner; matching a layer number alone is insufficient. The cache retains the complete owner, including packed scales, until graph destruction. A changed row count replaces only that layer's graph. A different owner at the same layer cannot replay the old graph and replaces it on capture. Explicit `clear_graph` evicts the current layer, while wave destruction drains its stream and clears all layers. Output borrows prevent rebinding during an active consumer borrow.

The backbone shared wrapper retains graphs and their full weight owners; the shared inner FFN's new unsafe rebinding method requires that retention and completed consumers. dSpark continues using its existing fixed binding. The arithmetic kernels, quantization and persistent cache formats are unchanged.

## Evidence and limits

The production daemon builds with all three rebinding paths. The retained `cuda_layer_graphs_reuse_storage_and_retain_exact_weight_bindings` test executes actual CUDA graph capture and replay on ostrich's GB10. It uses independent streams and output allocations for two lanes and distinct borrowed device source allocations for all 40 layers. At row counts 1, 80 and 4,096, two changed-payload passes produce 240 captures and 480 replays. The second pass reuses the first pass's handles, all outputs match exactly, and the banks never exceed 40 entries. An additional owner-replacement replay, invalid layer/row/null-handle guards, whole-bank clear and repeated clear pass.

The test uses CUDA copies to isolate graph ownership. It does **not** execute the rebound FP8 query, grouped/FP8 output or shared-FFN projections. Those paths require real-weight comparisons against fresh owners on RTX, including changed shapes, layer revisits, bound query/FFN identities and independent lanes. RTX still reports a loaded driver/userspace mismatch. Earlier projection numerical qualifications do not prove the newly rebound paths.

The same retained test can run within the daemon test target when `DS41RT_LAYER_GRAPH_TEST_LIBRARY` names a CUDA-enabled native library. Without that environment variable it explicitly skips GPU execution. The recorded Spark run sets the variable and executes the test, using an external small Cargo harness containing unchanged copies of the production test module and allocation owner.

## Allocation accounting

At capacity 4,096, query, output and shared-FFN buffers total 1,488,388,108 bytes per lane. Reusing each across 40 layers avoids 58,047,136,212 bytes (54.06 GiB) compared with a wave per layer. This is additional to the [33.55-GiB mHC saving](ds41-reusable-block-qualification.md).

Sparse attention was already layer-independent. One instance each of mHC, query, output, shared FFN and sparse attention would total 2,949,251,084 bytes (2.75 GiB) per 4,096-row lane, or twice that for two such lanes. These are production allocation formulas and SM120 AOT scratch metadata, not measured peak memory. They exclude weights, window/compressor/index/router workspaces, all persistent caches, dSpark, vision, graph/runtime allocations and scheduling/transport buffers. Complete lane assembly and full memory accounting remain open.

Exact formulas at six capacities, test scope, source hashes and artifact locations are in [the machine-readable record](ds41-layer-workspace-reuse.json).
