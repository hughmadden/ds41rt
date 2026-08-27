# DS4RT containers

DS4RT publishes two inference images because the release roles have different
host architectures and CUDA targets:

| Image | Platform | CUDA target | Purpose |
| --- | --- | --- | --- |
| `ghcr.io/tpurtell/ds4rt-coordinator:v2` | `linux/amd64` | `sm_120` | API, scheduler, attention, cache, sampling |
| `ghcr.io/tpurtell/ds4rt-spark-expert:v2` | `linux/arm64` | `sm_121` | TP4 routed-expert worker |

Weights are mounted read-only from the host Hugging Face cache. They are never
embedded in an image and the release image runs offline.

## Native release build

Run `./build.sh` from the repository root. It:

1. validates configuration, submodule locks, Docker, SSH, disk, and source
   identity;
2. builds the amd64 development image and release artifacts locally;
3. assembles the coordinator inference image;
4. synchronizes the identical source snapshot to the first configured Spark;
5. builds the arm64 development and inference images natively on that Spark;
6. verifies artifact provenance and distributes the expert image to the other
   three Sparks; and
7. exports binaries and hashes under ignored `dist/`.

The release build refuses an unrecorded dirty tree. Development iterations use
`./wip.sh`; an intentional frozen dirty snapshot must provide the source
manifest described by `./build.sh --help`.

## Image contract

Both images record the same concrete engine revision and SparkInfer revision.
They also include OCI source, description, and MIT license labels. `run.sh`
rejects missing or mismatched labels.

Release images contain the `ds4rt` binary, native library, Python capture
modules required at startup, pinned SparkInfer source, XGrammar/SparkInfer
license and provenance records, and the runtime entrypoint. Development
toolchains and quantization utilities are not part of the serving contract.

## Publish v2

After a clean build and five-host runtime qualification:

```bash
./push-containers.sh v2
```

The script verifies that the local coordinator and first-Spark expert images
carry the same engine revision, pushes `v2`, and updates `latest` for both
packages. It never builds an image; publication therefore cannot bypass the
normal build and runtime gates.

Authenticate the local host and the first Spark to GHCR before publishing.
The publisher needs `write:packages`. The images carry
`org.opencontainers.image.source=https://github.com/tpurtell/ds4rt-pro-rtx-4spark`,
which records the intended source association. After the first push, use each
package's settings to connect this repository and set visibility to public,
then verify both `v2` manifests from a Docker client with no registry login.

## Quantization images

`Dockerfile.quantization` and the `quant-coordinator`/`quant-expert` bake
targets preserve the reproducible conversion environment used to create the
published checkpoint. They are developer tools, not required for v2 serving.
The public quick-deploy path starts from the already quantized Pro artifact.
