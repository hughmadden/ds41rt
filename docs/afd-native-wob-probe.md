# Diagnostic WO-B single-row dispatch

Set `DS41RT_DIAGNOSTIC_WOB_M1_CAP16=1` before constructing the native target to
select the already-loaded capacity16 FP8 kernel for live M1, K8192, N5120 only.
Unset or `0` preserves the original dispatch. Other values fail initialization.
The flag is captured once per plan, before graph capture; changing it later does
not change an existing plan or captured graph.

This is a numerical falsification probe, not a production arithmetic default.
The retained M1 WO-B AOT recipe uses two FP32 K-partials; capacity16 uses one
accumulation. Both perform final BF16 rounding. Existing singleton goldens must
remain recorded separately from this diagnostic's results.

The probe adds no kernel, allocation, scratch requirement, or row capacity.
It requires an exact capacity16 entry already present in the plan and fails
closed if absent (including plans reserved for only one row). Live rows still
must be within the original requested capacity. Other geometries and live
M2+ retain their original selection. The native CUDA library is unchanged.

CPU validation:

```sh
cargo test --offline --locked -p ds41rt-ffi v41_fp8_plan::tests --lib
```

The focused tests cover explicit opt-in, invalid settings, exact geometry,
unchanged ordinary dispatch, live bounds, and missing capacity16. They do not
qualify numerical agreement or performance. Compare frozen actual WO-B inputs,
quantized operand bytes, output bits, and full-model results before selecting a
production recipe.
