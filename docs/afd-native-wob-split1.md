# Explicit WO-B M1 split1 candidate

The default export remains unchanged. `--wob-m1-split1` changes only the native
WO-B K8192/N5120 capacity1 GEMM from two FP32 partials to one K accumulation.
It preserves its tile and activation quantizer. Other projections and capacities
keep their original compiler arguments. This is a candidate requiring GPU
numerical and performance qualification, not a claim of improved accuracy.

The flag requires both `o_b` and capacity1 in the export. The compiler must
return split1 or export fails without publishing its manifest. The manifest
records `wob_m1_split1` and each variant's `force_split_k_one`, actual split count,
output type, scratch bytes, and artifact hashes. The native launch ABI is unchanged.

Pinned b12x fork: `2683868f8f839f27b58c094308e57ca42c361abe`, based on retained
`3882b935ede761d6c73a5d6fd68e690f1e3f5380`. Its explicit AOT-only
`force_split_k_one=True` changes no ordinary tensor API defaults. Updating the
submodule, repository URL and source-tree lock together preserves source checks;
do not bypass the verifier or patch the deployed dependency.

## Retained build inventory, 19 September 2026 AEST

Read-only Romeo inventory found the exact retained native library and manifest
in both `/srv/dsv41-flash-tp4/ds41rt-v3port/.ds41rt-release-image/` and
`/srv/dsv41-flash-tp4/ds41rt-v3build/.ds41rt-release-image/`:

- `libds41rt_native.so`: SHA256 `b86872ebea0dc8f6422dd3e3da1a1b8fcd842a38a12499b77c54e0771c17385d`.
- `V41_FP8_AOT.json`: SHA256 `7e27d324fc62c3fa61a1bf89249620cbc91f0773bbcc7b0cace3726c830fd596`;
  physical SMs170, b12x3882b935, WO-B M1 split2 and M16/80/256/1024/4096 split1.
- Retained `ds41rt-coordinator-dev:latest` image ID:
  `sha256:d62e7d302bded8b667464003e0210a11da5675c1da42ba368cea5bad7fcffb45`.

No coordinator `v41_fp8` AOT objects/headers or native CMake cache were found
under the retained `ds41rt*`, `afd-build*` and `build-chain` trees. The unrelated
`build-overlay1/src/build` cache exists. The release build script removes its
temporary native build. Therefore a full-library relink cannot yet reuse the
retained coordinator objects. The release image contains the library and
manifests, not those relocatable objects.

## Bounded build sequence

Coordinate the GPU window before running export. Use the immutable retained
development image above, networking disabled, committed source mounts read-only,
and a new writable output directory. Reuse installed compiler dependencies;
do not build/download a new image or run package installs.

First export an isolated component candidate, with the fork mounted `/b12x` and
this clean DS41RT checkout mounted `/source`:

The committed `scripts/build-v41-wob-component.sh /build/component` performs
source verification, both CPU test suites, exact SM170 preflight, subset export,
component-only linking and hashed provenance. Supply the clean checkout's full
SHA as `DS41RT_SOURCE_REVISION`. The output directory must not already exist.
Its internal export command is:

```sh
DS41RT_SPARKINFER_SOURCE_DIR=/b12x \
DS41RT_SPARKINFER_LOCK_FILE=/source/third_party/sparkinfer.lock.json \
python3 /source/python/tools/export_b12x_v41_fp8_aot.py \
  --output-dir /build/wob-split1 --projections o_b \
  --rows 1,16,80,256,1024,4096 --wob-m1-split1
```

This produces only WO-B quant/GEMM variants plus the exporter's required HC
projection object. Link `native/src/v41_fp8.cc`,
`native/cuda/kernels/v41_fp8.cu`, and those generated objects against CUDA runtime,
CUDA driver and `cute_dsl_runtime`, with native/include and the generated header
directory on the include path. The runtime directory is obtained with the
retained `python3 -m cutlass.cute.export.aot_config --libdir` command.
The resulting **component-only** library supports the existing FP8 ctypes
probe API; it cannot replace the full serving library.

After component qualification, a complete native build uses the retained
coordinator's existing options plus:

```text
-DDS41RT_ENABLE_V41_FP8_AOT=ON
-DDS41RT_V41_WOB_M1_SPLIT1=ON
-DDS41RT_SPARKINFER_SOURCE_DIR=/b12x
-DDS41RT_SPARKINFER_LOCK_FILE=/source/third_party/sparkinfer.lock.json
```

The split1 option fails configuration if FP8 AOT is disabled. A fresh full
native build must retain every other serving feature/ABI; do not silently replace
it with the component library. Keep the new build tree and all generated objects
for subsequent targeted relinks. No export or GPU build was run for this source packet.

## Oracle and tests

Replay frozen actual WO-A output through old capacity1, old capacity16 and new
capacity1 WO-B. Inspect quantized activation values and row-scale bytes using
the native FP8 info offsets; they must match before attributing output differences
to accumulation. Use the exact checkpoint FP8 values and 32x32 UE8M0 scales.
Compute a chunked FP64 dot product of those same dequantized operands, then round
once to BF16. Report both reference error and cross-shape equality, separately
from original full-model goldens. Check graph replay, poisoned tails, stable
scratch and repeated M1/2/4/16/80/256/1024 boundaries before measuring performance.

CPU-only tests execute the real constructor/exporter with stub compiler/device:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /b12x/tests/gemm/test_native_aot_split1.py
PYTHONDONTWRITEBYTECODE=1 python3 python/tests/test_v41_fp8_aot_options.py
```

These prove option plumbing, defaults, scope, metadata and failure behavior.
They do not establish numerical correctness or a performance improvement.
