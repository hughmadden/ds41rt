# V2 clean build, launch, and container publication

The final DS41RT v2 pair was built with the standard `./build.sh` from a clean,
standalone checkout at revision
`9477b6e39f4bbe431c4cd6d48c8b303045f9238f`. It contains model revision
`dba1be0a40aa45a94ad051997016db3960a90277`, SparkInfer
`bae6e5cf08fc7e51e8ea40f287dcfa95440036fc`, and XGrammar
`557becfb64c503ae9c04344b0047661f43f44320`.

## Clean build and packages

The final build completed in 304.85 seconds with no serving containers running.
An earlier clean-launch attempt exposed a missing local RTX expert AOT export in
the coordinator image; the release build now compiles that AOT, verifies its
exported ABI, and passed the standard launch. The final packages include their
dependency licenses, provenance records, AOT manifests, and per-file SHA-256
manifests.

| Role | Local image identity | Executable SHA-256 | Native library SHA-256 |
|---|---|---|---|
| amd64 coordinator | `sha256:3ffe75377cb901119b0ead470d4e047ff5b623c8c0eac6b98024e51814a222a8` | `448cab6b3ff2267b58e400e382144a3cf3624be1b6e1f9c973f051506ae7c742` | `c74da89370e8cc0af183f6bc39c3b65fd4c504edc5decc42a81483843e7fd946` |
| arm64 Spark expert | `sha256:62ceef2aaf502c03d977f7639ac8fd011aa43fe112ef75f228684bb226dc550b` | `83f81f5793ae09c81268ab813d96a0fad5a86ac612afa92d6c15cd326b99273a` | `4df74a88e5268a6476ffab0b4f4fb534caa57b3b0ac99b87360845f7dbe56825` |

The release bundles are `ds41rt-v2-coordinator-linux-amd64.tar.gz`
(`63576c0dd347c675f7f62bb55c6d03c9cc15d0e4f56a23181f283b84317178e5`),
`ds41rt-v2-spark-expert-linux-arm64.tar.gz`
(`41ccd1a316b623652fa8a631bff5a1a247a1448895a26d72580d1471a14cbbb2`),
and `ds41rt-v2-qualification-evidence.tar.gz`
(`e462a1db3ab8ab9df9a0a5edaa60b89c5172da820d5afe366db3db50cee6a9`).
Both binary bundles pass their embedded manifests; the evidence archive passes
its complete per-file manifest.

## Standard launch

The exact images reached port 8000 in 59.50 seconds target-only and 55.12–57.76
seconds with dSpark. The final standard service uses C16, dSpark, 24 prompt
snapshots, 24 completed-turn snapshots, and a 16.681 GB global FP4 pool for
18,710,016 logical tokens plus 32,768 private-tail tokens. Automatic placement
loads routed-expert layers 0–4 on the RTX, while every Spark retains all 40 TP
expert layers under its 100 GiB device budget. Peak observed coordinator GPU
memory use was 96,950 MiB.

The [performance report](release-v2-performance.md) records the scoped clean-image
qualification. The prefill matrix and official API comparison are preserved v1
measurements; the full feature suite was intentionally not rerun.

## Registry publication

`./push-containers.sh v2` published `v2` and `latest` for both roles. Raw manifest
comparison proves that the two tag names are identical for each image.

| Role | Published digest | Platform manifest/config |
|---|---|---|
| coordinator | `sha256:3ffe75377cb901119b0ead470d4e047ff5b623c8c0eac6b98024e51814a222a8` | amd64 manifest `sha256:5da3ae2053e0c624d2c5cf8a66a10f9f7e58290dd77f222f41280104a9360ee5`; config `sha256:239529111f876dcc75f1949eb9bda67fca092d75555508d5e288cc1bc45d9477` |
| Spark expert | `sha256:d9b4bc411eb3480a956d3a993597c5b102bbd438adb9c73e57d6ba2e5cf21e39` | arm64 config `sha256:62ceef2aaf502c03d977f7639ac8fd011aa43fe112ef75f228684bb226dc550b` |

Both images carry version `v2`, the measured source revision, role, architecture,
dependency, license, and model-description labels. The source branch contains
only launcher logging and release documentation after the measured image
revision; no inference implementation changed after the clean build.
Both GHCR packages are public.
