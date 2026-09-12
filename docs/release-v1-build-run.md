# Clean release build and launcher qualification

The candidate at `d5fb015d00aa7b302e3ce179b097482da678db76` was
qualified from a clean DS41RT container/image state on the RTX coordinator and
all four Spark hosts. The build used the pinned DeepSeek-V4.1-Flash model
snapshot `dba1be0a40aa45a94ad051997016db3960a90277`, SparkInfer revision
`7299b3b92e70d539b2c0a63aaadce36932ceef4d`, and XGrammar revision
`557becfb64c503ae9c04344b0047661f43f44320`.

## Clean build

The first clean build exposed a release-only defect: the release artifact build
still enabled the legacy DS4 Flash/Pro AOT bridge, whose generated wrapper no
longer matched the pinned SparkInfer ABI. That build stopped in
`ds4_flash_aot.cu` after 182.778 seconds. Release images serve the native V4.1
path, so the fix excludes that legacy bridge from release artifacts while
leaving it enabled for development builds. The fix also makes the V4.1 router
qualification import the pinned SparkInfer bootstrap. The focused release
launcher, namespace, and provenance suite passes 29/29 cases inside the freshly
built coordinator image.

After removing the partial images and build products again, unmodified
`./build.sh` completed in 374.439 seconds. It built both architecture-specific
images, verified every exported artifact, and distributed the 24,729,185,792-byte
Spark image to the other three workers at 8.95–9.34 Gb/s over two RDMA channels.

| Role | Qualified local image ID | Native executable SHA-256 | Native library SHA-256 |
| --- | --- | --- | --- |
| amd64 coordinator | `sha256:a13258e92dd25ddb889bd31bb77c8813c7881868b103ceec0c140e30e893c213` | `8bf754f1f11f2026f395e46f9e04b26be4f0b9fd9b667687224046416b7dc344` | `bfed8269593ed3ffffb13e0cec4841ff312cf672a9642b9bfa4ab421afdb08b2` |
| arm64 Spark expert | `sha256:2f5d328a14f1a52a0d3b2356b041415d744635f3f3557e41403246429984093b` | `0865699b9e355b5eb456398e13c79b14c70886ee863e9175f8c9dc899e9b050f` | `903a4a4a9e92033f4489280677b33541a22333ee3520bf6a65514c651e96332b` |

All four Spark hosts report the same arm64 image ID and the expected ranks
0–3. Both images carry the exact engine and SparkInfer revisions above. These
are the locally qualified image IDs; registry digests will be recorded after
publication and verified against this pair.

## Standard launcher

Both a default and an option-override dry run passed. Unmodified `./run.sh`
then brought the five-host service up on port 8000 in 56.481 seconds with the
release defaults: concurrency 16, 24 retained turns, 1,048,576 context tokens,
393,216 output tokens, a 2,048-token prefill batch, and dSpark enabled. The API
returned the `ds41rt-native-fp4-kv-dspark` fingerprint and included
`reasoning_content` when the request omitted thinking controls, proving the
default high-thinking path was active. The coordinator used about 65.5 GiB of
GPU memory; its target backbone/index/embedding weight phase took 1.768 seconds.

A real override launch then passed with concurrency 2, a 2 GiB KV pool, three
retained turns, 65,536 context tokens, 8,192 output tokens, a 1,024-token
prefill batch, and dSpark disabled. It became ready in 53.628 seconds, exposed
the requested model limits and returned the `ds41rt-native-fp4-kv`
fingerprint. Coordinator GPU use fell to about 20.8 GiB. A separate dry run
also accepted concurrency 7, an 80 GiB memory reservation, five retained turns,
and explicit context/output/prefill values.

The standard launch was restored afterward in 55.152 seconds. Its final API
response again reported the FP4 compressed source, FP8 SWA and FP4 index dSpark
fingerprint, the official model limits, and default reasoning. The coordinator
and four rank-specific expert containers remain running on the exact qualified
image pair.

The RTX campaign used driver 595.91.07, a 400 W enforced power limit, and the
standard 14,001 MHz maximum memory clock. The four GB10 workers used driver
580.159.03; their platform does not expose power and clock limits through the
queried NVIDIA interface. Phase-level Spark storage/transform and graph-capture
accounting remains part of the startup performance report even though the
observed end-to-end launch is already inside the requested 60–90 second range.

[Machine-readable summary](release-v1-build-run.json) records the commands,
settings, timings, revisions, hashes, and raw-evidence archive.
