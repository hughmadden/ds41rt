# Clean build, launcher, and container qualification

The corrected DS41RT v1 pair was built with the unmodified standard `./build.sh` from revision `9ea5c96468da690fe7dd01471d4fa2fb8555a606`. It contains model revision `dba1be0a40aa45a94ad051997016db3960a90277`, SparkInfer `7299b3b92e70d539b2c0a63aaadce36932ceef4d`, and XGrammar `557becfb64c503ae9c04344b0047661f43f44320`. Both images carry the DS41RT v1 source, role, architecture, dependency, and license metadata.

## Corrected standard build

The build started with the service stopped and the `v1` image references absent on the coordinator and four Sparks. It completed in 268.82 seconds, rebuilt the native artifacts from the current source, verified their manifests and licenses, and distributed one identical 24.47 GB logical Spark image from ostrich to the remaining workers at 8.97–9.71 Gb/s.

| Role | Local image identity | Executable SHA-256 | Native library SHA-256 |
|---|---|---|---|
| amd64 coordinator | `sha256:67f2954e18f69b39f8fbb68164f7d9e2b8f4c4b9e3242281ecfcb8afec8552e9` | `e5aef25f998a6c5a07467d645f0e7b3d899857993e4cdc99857097ebf89e4e5c` | `ea9a845f2518b279d6f4ecf2a7d38b01acafecc3613ff44712591623d6894503` |
| arm64 Spark expert | `sha256:579245a5a01ad942cc1d86e0ebab0ecca224e93f16e1280fe789ff25bbad10f0` | `ec92649bbb56c4c441a16ce0a5e91c35af266a9f8d32febb7a99d034f007a690` | `95364759db0e0d49ac588f403eaf7276501c04704ff054490ee51338a3fc23e7` |

All four Spark hosts report the same ARM64 identity. The Spark manifest exposes the qualified fused-slice variants: width 64 at capacity 1, width 192 at capacities 16, 80, 256, 1,024, and 4,096, and direct token accumulation at capacities 256 and above. Seventeen comparisons against the preserved optimized libraries pass with maximum absolute difference `2.288818359375e-05`.

## Standard launcher and restored performance

The exact images reached port 8000 in 53.88 seconds target-only and 56.77 seconds with the default dSpark path. The final standard service reports FP4 compressed source, FP8 SWA, an independent FP4 index, C16, 24 prompt and completed-turn entries, 1,048,576-token context, 393,216-token output, a 2,048-token prefill batch, and dSpark enabled.

The complete performance rerun uses three measured samples and passes every local serving and cache check. The headline prefill cell reaches 7,743.47 tok/s median, warm counting reaches 150.51 tok/s, and C16 aggregate counting reaches 742.91 tok/s. Request lifecycle checks pass at C2, C6, and C16, including cancellation and replacement. See the [performance report](release-v1-performance.md) for every table and raw sample.

## Registry publication

`./push-containers.sh v1` published `v1` and `latest` in 37.15 seconds. Raw-manifest comparison proves that both tag names are identical for each role.

| Role | Published digest | Platform manifest/config |
|---|---|---|
| coordinator | `sha256:67f2954e18f69b39f8fbb68164f7d9e2b8f4c4b9e3242281ecfcb8afec8552e9` | amd64 manifest `sha256:2f6061eae3aa073da3e31e13e6c717f151d7f7a638be5d73d5d86989e48d6d7e`; config `sha256:a8208b2509d4b268de51fb835a1c7a390ee552a8af13e5447b100ecdb33f843e` |
| Spark expert | `sha256:1f1bff295a1d112c8a2fb80918b5abcafcf10eec4936717e234d3d727635a0be` | arm64 config `sha256:579245a5a01ad942cc1d86e0ebab0ecca224e93f16e1280fe789ff25bbad10f0` |

The coordinator is an OCI index containing its amd64 image and BuildKit attestation. Pull and label inspection verifies the `v1` version, source revision, V4.1 Flash description, role, and CUDA architecture. The GHCR packages remain private for the repository owner's final manual visibility change.

## Empty-state coverage

An earlier five-host gate removed DS41RT containers, images, and build products before invoking the same standard scripts. That build completed in 374.44 seconds and launched in 55–56 seconds. It caught a legacy DS4 AOT target, a removed model alias in readiness validation, and an inherited V4 Pro image description. The corrected `v1` pass then rebuilt the release-tagged artifacts and caught the separate missing Spark-kernel default through the full performance matrix.

All measurements use RTX driver 595.91.07, an enforced 400 W power limit, and the standard 14,001 MHz maximum memory clock. The four GB10 workers use driver 580.159.03.

The original empty-state evidence remains in [`evidence/native-clean-build-run.tar.gz`](evidence/native-clean-build-run.tar.gz). The corrected v1 build, launches, performance canary, image inspection, manifests, numerical checks, lifecycle checks, and publication logs are preserved in [`evidence/native-release-v1-build-run.tar.gz`](evidence/native-release-v1-build-run.tar.gz).
