# Reusable backbone mHC workspaces

`BackboneBlockWave::advance` reuses a lane's attention and FFN mHC allocations across adjacent layers. It validates both weight bindings, drains the boundary streams, copies the completed residual and next-pre coefficients, then installs the bindings together. Positions and predecessor identity survive the transition. Layers 1 and 14 still withhold the prepared input until engram is applied. `restart` returns a completed or explicitly reset lane to layer zero and requires new input initialization.

The mHC owner retains up to 40 distinct weight owners, keeping their storage alive across rebinding. Backbone boundaries currently run on their own synchronized streams. Rebinding an externally captured boundary requires draining its consumers and recapturing for the new weights; this change does not introduce graph reuse across different weight pointers.

## Memory accounting

At capacity 4,096, the two mHC boundaries allocate 923,664,384 bytes per lane. Reusing them across 40 layers saves 36,022,910,976 bytes (33.55 GiB) compared with allocating a pair for every layer. At 80 rows, one lane uses 18,040,320 bytes and saves 703,572,480 bytes.

The initial audit counted mHC, query, attention-output, shared-FFN and sparse-attention workspaces using production formulas and the built SM120 FP8 AOT scratch metadata. Replicating these five groups across 40 layers at 4,096 rows would consume 117,970,043,360 bytes. Sharing only mHC reduces that accounting to 81,947,132,384 bytes. This excludes weights, indexing, windows, compressors, dSpark and persistent caches. Query/output/FFN/attention and index workspace sharing, complete memory budgeting and alternating-lane assembly remain necessary; this is not evidence that the assembled model fits.

## Qualification and copy-ordering fix

On ostrich's GB10, an external fixture loaded all 40 official checkpoint mHC parameter sets. For capacities 1, 80 and 4,096, it ran two changed-input passes through all 40 layers, comparing the reused owner with a fresh same-layer owner fed identical residual/pre inputs. All 240 layer comparisons were byte-exact after the fix below. The lane's input addresses remained stable, copied transitions matched every byte, positions and identities survived, engram-required views stayed withheld, and invalid advance/restart calls invalidated publication and allowed reset recovery.

This executes actual mHC kernels and weights with normalized identity stand-ins for attention and FFN. It does not execute the full transformer, engram arithmetic, attention/FFN projections or scheduler. Earlier component reference qualifications remain distinct from this storage-reuse check.

The first run passed one 40-layer pass, then failed on the second. Diagnostic replay isolated a mismatch at layer 30. The native `ds41rt_copy_d2d` helper used `cudaMemcpy` without waiting for completion. CUDA's synchronous-named device-to-device copy returns without host synchronization, and the runtime's nonblocking consumer streams could race it. The helper now waits on the default stream before returning; callers must still complete producers first. Explicitly stream-ordered code can use the unchanged asynchronous helper. See [NVIDIA's synchronization contract](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

The retained native regression `test_cuda_copy_d2d_completes_before_nonblocking_consumer` checks default-stream completion immediately after the helper, then verifies a copy consumed on a nonblocking stream. It uses eight changed 16-MiB payloads. An isolated invocation of this exact test fails against the prior library with `device not ready` and passes against the corrected library. The complete 240-layer fixture also passes with the corrected library. This prevents later default-stream reads from masking the missing wait.

The daemon and SM120/SM121 native libraries build successfully. RTX execution remains unqualified because the loaded NVIDIA kernel driver and userspace libraries differ. Results, exact byte accounting, source/artifact hashes and external fixture locations are in [the machine-readable record](ds41-reusable-block-qualification.json). Full-model and performance qualification remain open.
