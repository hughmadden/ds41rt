# Native V4.1 expert AOT exports

`python/tools/export_b12x_v41_experts_aot.py` exports the qualified `silu_v41` dynamic expert kernel as C headers/objects and serializes b12x's scratch tensor offsets, initialization modes, capacities, and object hashes.

Exports run on their actual target architecture: SM120 for the coordinator's 128 full-width dSpark experts and SM121 for each Spark's 384 backbone expert shards.

Logical Spark shards are 576 columns wide, while b12x's prepared execution representation pads them to 640; the exporter obtains that width from b12x and records both widths rather than applying an engine-side packing heuristic.

## Build integration

`DS41RT_ENABLE_V41_EXPERT_AOT=ON` exports capacities 1, 16, 80, 256, 1024, and 4096 and links their objects into `libds41rt_native.so`.

Release and WIP artifact builds enable this option, package `V41_EXPERT_AOT.json`, and retain it through Docker image export, dist checksums, WIP checksums, and WIP slot finalization.

The existing legacy exports remain until their callers are replaced; enabling V4.1 exports alone does not migrate those callers.

## Qualification

Both the local RTX and ostrich complete native CMake/CUDA library builds with all six capacities; these targeted builds disable XGrammar and the old optional AOT families and do not replace full release-image qualification.

M16 numerical and mutated CUDA graph checks pass through the exported C entry points in the linked native libraries for both coordinator (128 experts, width 2304, top-3) and Spark (384 experts, logical width 576, top-6).

The qualification fixture uses the existing GPU test's prepared tensors, redirects the dynamic launch to the exported C function, and retains its original relative-L2/cosine gates and replay-allocation checks; local output reduction remains the existing GPU reduction launch.

The Spark C-entry check initially caught the 576-versus-640 export mismatch before launch, and the corrected exporter passes the check.

The 33 WIP-process/namespace regression checks and shell syntax checks pass; source, object, manifest, and linked-library identities are recorded in `ds41-expert-aot-qualification.json`.

## Remaining execution work

The Rust executor still needs the native argument binding, weight preparation/loading, scratch initialization, graph ownership, and FP32 route-plane response path.

The exported core kernel does not include final expert reduction or weight preparation; those operations still need native serving entry points.

Only M16 C-entry execution is qualified here; the larger capacities are compiled and linked, with integrated prefill execution, concurrency-16 memory accounting, and throughput still open.
