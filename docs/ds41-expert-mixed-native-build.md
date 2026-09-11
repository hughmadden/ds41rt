# Complete native build with explicit expert capacity tiling

The experimental exporter and CMake build now accept an explicit capacity-to-width map. The qualified candidate uses width 64 at capacity 1 and width 192 at capacities 16/80/256/1024/4096. This is an explicit experimental configuration, not a new hidden device heuristic or a change to default serving policy. The worker already dispatches one-row work through capacity 1 and wider work through its configured capacity; no Rust dispatch change is required.

On ostrich, the complete CMake native library built successfully with CUDA, SM121, RDMA and V4.1 expert AOT enabled. All six mapped expert capacities and the wire input quantizer were exported. Both CTest tests (`ds41rt_native_selftest`, `ds41rt_cuda_selftest`) passed. Native synthetic qualification passes six cases against the matching same-width independent grouped kernel: two one-row cases through width 64, and shared M2/mixed six/shared sixteen/shared eighty through width 192. Graph replay keeps allocation ownership stable and checks exact FP32 route output. Capacities above 80 were compiled, not runtime-qualified.

The candidate library is `/tmp/ds41-mixed-native/build/libds41rt_native.so` on ostrich, SHA256 `872591810ab2d9f3ed1715c13d8207b3d715aabd7bf466bd33a661b64a5c29ad`. It is a complete native library, unlike the smaller direct-linked qualification libraries. [Build/test output and the export manifest](ds41-expert-mixed-native-build.json) are retained.

| Capacity | Width | Scratch bytes |
|---|---:|---:|
| 1 | 64 | 1,246,240 |
| 16 | 192 | 8,027,536 |
| 80 | 192 | 40,106,896 |
| 256 | 192 | 128,325,136 |
| 1024 | 192 | 513,277,456 |
| 4096 | 192 | 2,053,086,736 |

The larger scratch costs remain a prefill optimization target. The choice is supported by the [native official-weight comparison](ds41-expert-native-official.md), but a paired distributed live comparison is still required. Worker/API binaries and their current library mounts remain unchanged.

Build inside the Spark development image, with the repository at `/workspace/ds41rt` and a fresh writable `/output`:

```sh
cmake -S native -B /output/build -G Ninja \
  -DDS41RT_ENABLE_CUDA=ON -DDS41RT_CUDA_ARCHITECTURES=121 \
  -DDS41RT_ENABLE_V41_EXPERT_AOT=ON -DDS41RT_ENABLE_RDMA=ON \
  -DDS41RT_ENABLE_XGRAMMAR=OFF \
  -DDS41RT_V41_EXPERT_SLICE_WIDTH=1:64,16:192,80:192,256:192,1024:192,4096:192
cmake --build /output/build -j 8
ctest --test-dir /output/build --output-on-failure
python python/tools/qualify_v41_slice_native.py --native-lib /output/build/libds41rt_native.so --width 64 --capacities 1
python python/tools/qualify_v41_slice_native.py --native-lib /output/build/libds41rt_native.so --width 192 --capacities 16,80
```

The standalone exporter's `--width` accepts the same explicit map and rejects missing/duplicate capacities or unsupported widths. The qualifier's capacity filter selects the relevant native cases; its independent grouped oracle still exercises its full fixture. A final CLI guard now rejects unsupported/empty capacity selections and requires every requested capacity to have been checked; that guard was added after the retained run and does not change kernel execution.
