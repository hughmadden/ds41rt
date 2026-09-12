# Clean build, launcher, and container qualification

The v3 release pair was built from source revision `23a6670c6b3695df2e81b67b8ef08d28343f8dae` with model revision `dba1be0a40aa45a94ad051997016db3960a90277`, SparkInfer `7299b3b92e70d539b2c0a63aaadce36932ceef4d`, and XGrammar `557becfb64c503ae9c04344b0047661f43f44320`. Both images carry the correct V4.1 Flash description, v3 version, source, role, architecture, and dependency labels.

## Final v3 build

Before the final build, the running five-host service was stopped and the v3 release and development image references were removed from the coordinator and all four Sparks. Unmodified `./build.sh` completed in 286.081 seconds. Every exported artifact, dependency tree, provenance file, license, and checksum passed. The 24.47 GB logical Spark image was then distributed from ostrich to dodo, emu, and kiwi through the two-rail RDMA path.

| Role | Local image identity | Executable SHA-256 | Native library SHA-256 |
|---|---|---|---|
| amd64 coordinator | `sha256:0d8a29160924dc62694d65f46e5101bf39071fb28e7611344489dde416bfe950` | `e6629684c95533917c2ecacb1031f71d0fa87a5cf652a25aaa9a459b8d906b76` | `ee89310d1f0ddab9ea56c52bcacb7711af7850cc68ae013bea901e8bd01984f3` |
| arm64 Spark expert | `sha256:d9feacd00d79baa451537555b98b73834397e007c7fd509ce53dfbb1b64b59c2` | `4d183e0d800278afe6544cf7668f983e24a6c9f074fcbae34d523c2f32e7ccaf` | `473ee4d93a7c84521a61f35241fd33a810ad38eb41cee1b5063353ed7a5aa977` |

All four Spark hosts report the same ARM64 image identity. The release commit differs from the performance image source only in documentation, release scripts/tests, benchmark tooling, and embedded revision metadata; no Rust or CUDA inference implementation changed.

## Standard launcher and performance equivalence

Unmodified `./run.sh` brought the exact final pair up on port 8000 in 56.758 seconds. It verified the configured model through `/v1/models` before declaring readiness and reported the standard FP4 compressed source, FP8 SWA, FP4 index, C16, 24 retained turns, 1,048,576-token context, 393,216-token output maximum, 2,048-token prefill batch, and dSpark.

A request with omitted thinking controls returned `36` for 17+19 with separate reasoning content and the `ds41rt-native-fp4-kv-dspark` fingerprint. The protocol-matched concurrency canary then ran one warmup plus three timed repetitions at C1, C2, C4, C8, and C16. Every timed request was a complete prompt hit and returned the identical 599-token sequence.

| Concurrency | Final v3 median tok/s | Release-suite median tok/s | Change |
|---:|---:|---:|---:|
| 1 | 126.96 | 124.81 | +1.72% |
| 2 | 179.37 | 176.58 | +1.58% |
| 4 | 289.45 | 288.89 | +0.19% |
| 8 | 416.94 | 416.34 | +0.15% |
| 16 | 694.90 | 683.70 | +1.64% |

Two shorter jump-to-C16 diagnostics are also retained. They show first-use/setup samples as low as 600.57 tok/s, followed by 668.76–690.69 tok/s. The matched progression above is the comparison to the release protocol; no sample is omitted from the evidence.

The focused release configuration, launcher, corpus, and provenance suite passes 29/29 tests. Shell parsing passes for build, run, stop, publish, and shared release helpers.

## Registry publication

`./push-containers.sh v3` published the final v3 and `latest` tags in 37.534 seconds. Both tag names resolve to identical raw manifests for each role.

| Role | Published digest | Platform manifest/config |
|---|---|---|
| coordinator | `sha256:0d8a29160924dc62694d65f46e5101bf39071fb28e7611344489dde416bfe950` | amd64 manifest `sha256:45b4f5a351dbcc8ef891291c4fbccb8d664429b76830cf56ae7136fe416c9857` |
| Spark expert | `sha256:672f82a1a99872cdc8014811b99c0967e0955c8c3e1e29b91bd48e7b06d3566d` | arm64 config `sha256:d9feacd00d79baa451537555b98b73834397e007c7fd509ce53dfbb1b64b59c2` |

The coordinator top-level object is an OCI index containing its amd64 image manifest plus a BuildKit attestation manifest. Pull-by-digest on the coordinator and ostrich verified the published architecture, role, CUDA architecture, source revision, v3 version, and V4.1 Flash description. The GHCR packages remain private for the repository owner’s requested final manual visibility change.

## Earlier empty-state gate and issues caught

The earlier release gate removed all DS41RT containers, images, and build products, then ran the same standard scripts from an empty image state. That build completed in 374.439 seconds, distributed one identical worker image to all four Sparks at 8.95–9.34 Gb/s, and launched in 55–56 seconds. It also exercised a real C2/2 GiB/three-turn/65,536-token target-only override before restoring defaults. This proves the scripts do not depend on an inherited image; the final v3 pass above proves the release metadata and commit.

The first empty-state attempt exposed and fixed a legacy DS4 AOT target that no longer matched the pinned SparkInfer ABI. During v3 packaging, the stricter readiness check initially reused an old helper that required a removed `-full` model alias; the native helper now validates the one configured model. Registry inspection then caught an inherited “DeepSeek V4 Pro” OCI description. Both superseded private tags were overwritten, and the exact corrected images were rebuilt, relaunched, remeasured, republished, and pulled by digest.

All measurements use the RTX driver 595.91.07, enforced 400 W limit, and standard 14,001 MHz maximum memory clock. The four GB10 workers use driver 580.159.03.

The original empty-state evidence is in [`evidence/native-clean-build-run.tar.gz`](evidence/native-clean-build-run.tar.gz). Final v3 build, launch, canary, local/remote image inspection, registry manifests, pull-by-digest output, package visibility, tests, hardware state, and superseded diagnostic attempts are preserved in [`evidence/native-release-v3-build-run.tar.gz`](evidence/native-release-v3-build-run.tar.gz). The public bundle replaces local paths, private RDMA addresses, and GPU UUIDs with explicit placeholders; performance data and artifact identities are unchanged.
