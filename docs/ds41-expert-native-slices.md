# Native slice pipeline through the existing expert ABI

b12x `c1b2b8f` includes `V41SlicePipeline`: publish the runtime row count to persistent device scratch, group routes, execute fused expert slices, then reduce FP32 slices into original route order. The combined callable exports to a native object. There is one extra row-publication launch compared with the earlier Python-planned timing fixture; no new native performance claim is made here.

`python/tools/export_b12x_v41_slices_aot.py` generates an ABI-checked bridge for the existing 44-pointer expert interface. The ordinary `native/src/v41_experts.cc` binds and initializes caller-owned scratch, loads the module and dispatches it. Existing RP weight slots and FP32 route-output slot 41 retain their meanings. The wire-scale pointer comes directly from byte offset 5120 of the 5280-byte input row. Public ABI version remains 2. The exporter requires an explicit width and records generated header/object hashes.

Native width-128 libraries were exported and linked on RTX SM120 and Spark SM121. `python/tools/qualify_v41_slice_native.py` passes six routing cases on each: one row, shared two rows, mixed six rows, shared sixteen rows, shared eighty rows and return to one row. All native route outputs are FP32-exact against the independently launched same-width grouped kernel using CPU metadata. That grouped fixture also checks its synthetic numerical oracle. Captured native graphs reuse scratch and input allocations; invalid tail routes produce zero. This qualifies ABI composition and ownership on synthetic weights, not all-rank official or whole-model quality.

Scratch sizes for width 128 are 754,720 bytes at capacity 1, 11,959,696 at capacity 16, and 59,767,696 at capacity 80. Final per-slice FP32 output still occupies VRAM; intermediate FFN activations stay in shared memory. Scratch reduction and large-prefill efficiency remain open.

The explicit experimental CMake setting is `-DDS41RT_V41_EXPERT_SLICE_WIDTH=128`, alongside CUDA, SM121 and V4.1 expert AOT. Widths 64/192 are also accepted but are not yet qualified through this native bridge. Default builds retain the deployed backend. CMake configure/generation succeeds for the experimental Spark setting; this turn's native libraries were linked directly with nvcc, so the complete CMake build across all six capacity exports remains unverified.

Reproduction inside the architecture-matched development image:

```sh
python python/tools/export_b12x_v41_slices_aot.py --output-dir /output --rows 1,16,80 --width 128
nvcc -shared -Xcompiler -fPIC -std=c++17 -arch=sm_121 \
  -I native/include -I /output native/src/v41_experts.cc \
  native/cuda/kernels/v41_route_reduce.cu native/cuda/kernels/v41_expert_pack.cu \
  /output/*.o -L /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/cu13/lib \
  -lcute_dsl_runtime -o /output/libds41rt_native.so
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/cu13/lib \
  python python/tools/qualify_v41_slice_native.py --native-lib /output/libds41rt_native.so --width 128
```

Use a fresh output directory, and `sm_120` for the RTX qualification. [Native evidence](ds41-expert-native-slices.json) retains test output and manifests. Next: official native comparison, useful 64/192 variants, all-rank binding/quality/performance, then serving dispatch. No live worker binary changed.
