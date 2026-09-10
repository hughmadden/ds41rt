# Native V4.1 expert arithmetic

SparkInfer `be1b4b90edb44c528ea7554bd9959569b316631b` introduces the `silu_v41` semantic contract for native FP4/K32 weights and BF16 inputs/outputs.

The official expert rounds its gate and up projections to BF16, clamps gate at 10 and up to [-10, 10], multiplies the activated intermediate by the routing weight, then rounds and quantizes that intermediate before FC2.

The inherited W4A8 path instead retained FP32 FC1 values and applied routing weights after FC2; using it unchanged produced approximately 3.96% relative error in the initial official-reference comparison.

The new fused path preserves these projection boundaries, uses caller-owned FP32 per-route scratch for all FC2 intermediate slices, and rounds each completed expert to BF16 before a fixed-order FP32 top-k sum.

The planner owns this reduction policy and selects supported fused M16/M32 tiles; unsupported swapped, materialized-intermediate, and nondeterministic direct kernel configurations reject the V4.1 contract.

## Evidence

The six-case native regression suite passes on both local RTX GPUs and on ostrich and dodo, covering hidden size 5120, intermediate widths 576/2304, M=1/16/80, top-6/top-3, full 384/128-expert inventories at M=16, mutated input/routing CUDA graph replay, and unchanged replay allocation.

All cases retain relative L2 below 0.01 and cosine above 0.9999; the earlier BF16 partial accumulation failed one replay at relative L2 0.01033, which was fixed without changing either gate.

Separately, six eight-expert fixtures compared against the actual pinned official GPU `Expert` give relative L2 0.001627–0.001689 after the fix.

`ds41-expert-qualification.json` preserves per-device test output, numerical measurements, reference identity, and source-formatting chronology; the suite lives at `third_party/sparkinfer/tests/moe/test_v41_expert_numerics.py`.

## Remaining integration

These checks qualify isolated expert execution, including full-width dSpark geometry, but do not qualify AFD transport, the shared-expert addition, tensor-parallel global rounding order, complete dSpark proposal/verification behavior, or throughput.

Backbone integration must preserve per-expert accumulation across TP ranks before applying the reference's final BF16 rounding; simply summing already rounded per-rank top-k outputs does not establish that contract.

The production runtime still needs native loading/binding, persistent workspace capacity accounting, and graph ownership for all serving shapes; all dSpark tensors remain assigned to the coordinator RTX.
