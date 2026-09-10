# Native engram FP8 AOT build

The coordinator exporter now builds K32 BF16-to-FP8 activation quantization and native 32x32-scale FP8 GEMM for engram `wkv` (`N=25600`, `K=6144`) at capacities 1, 16, 80, 256, 1024 and 4096. It consumes the pinned SparkInfer AOT helpers with K128 scale reuse disabled; every selected standalone plan compiled without split-K. Plans requiring unsupported split-K remain rejected rather than silently changing accumulation.

`DS41RT_ENABLE_V41_FP8_AOT=ON` enables the coordinator-only CMake export and links all twelve objects into the native library. It requires native SM120 compilation, verifies the SparkInfer source pin and validates generated pointer/scalar/stream argument order before writing its completion manifest. The manifest includes activation scratch offsets, artifact hashes, GPU geometry and the dependency revision, and is removed before a new export so a failed build cannot leave a stale success marker.

The CMake build and shared-library link completed on local RTX GPU 0 using `ds41rt-coordinator-dev:latest`; exact image, source and artifact identities are recorded in `ds41-native-fp8-aot-build.json`. Activation scratch budgets are 31,744 / 125,952 / 531,456 / 1,671,168 / 6,684,672 / 26,738,688 bytes for the six capacities respectively, excluding output and resident weight storage.

This is build evidence only. Native module initialization/launch bindings, native 32x32-to-MMA scale packing, runtime quantizer launch geometry, full engram invocation and release/WIP enablement remain open. No kernel numerical checks, graph replay checks, throughput benchmarks or model loads were run, and compilation does not qualify variable-M behavior.
