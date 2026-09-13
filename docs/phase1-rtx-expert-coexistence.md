# dSpark and full RTX experts in one native library

The optional `DS41RT_ENABLE_V41_LOCAL_EXPERT_AOT=ON` CMake build adds a second
expert table alongside the ordinary SM120 dSpark export. Namespaced local
symbols reuse the existing validated C++ launch implementation while owning
separate modules, handles and scratch layouts. Input quantization and packing
remain shared. Rust selects local symbols explicitly for complete backbone
weights; the serving scheduler does not select or dispatch local layers yet.

A clean CMake configure and complete native build passed. That exact library
passed nine full-width native numerical/replay cases through 80 live rows,
including high expert IDs through 383. At capacities 1, 16, 80, 256, 1024 and
4096, both tables initialized, returned distinct handles, and rejected crossed
scratch bindings without changing pointer slots. The daemon passes release
checking and the FFI ABI layout test passes. These checks do not establish
unchanged dSpark serving performance or full-model quality.

Local capacities through 80 use ordered route output. Capacities 256 and above
use the existing direct token accumulation recipe at full intermediate width;
those larger variants have only been compiled and initialized so far. Their
numerical, prefill and concurrency qualification remains required. Local scratch
is 128,580,496 bytes at capacity 80 and 123,706,912 at capacity 4096, excluding
weights and other execution storage. No speedup or startup claim is made.

Run `python/tools/qualify_v41_expert_handles.py --native-lib LIB` for handle
isolation and `python/tools/qualify_v41_slice_native.py --native-lib LIB --width
192 --capacities 1,16,80 --full-backbone --local-entrypoints` for ordered local
execution. [Evidence](phase1-rtx-expert-coexistence.json) records checks and
artifact hashes; raw files are in `/tmp/ds41-rtx-full-experts`.

Next work is per-lane local execution and reduction, bottom-up memory planning,
official-weight checks, and startup/serving comparisons. Keep the existing
service and ordinary build defaults until that integration is qualified.
