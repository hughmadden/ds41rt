# DS41RT v3 clean build, launch, and container publication

The final DS41RT v3 image pair was built with the standard `./build.sh` from
clean revision `c38746aeac9204eff12b7d2f0433013ed24526c4`. It contains model
revision `dba1be0a40aa45a94ad051997016db3960a90277`, SparkInfer revision
`3882b935ede761d6c73a5d6fd68e690f1e3f5380`, and XGrammar revision
`557becfb64c503ae9c04344b0047661f43f44320`.

## Clean build and packages

The five-host build generated coordinator binaries locally, generated arm64
expert binaries natively on `ostrich`, verified dependency provenance and both
full-width/TP2 RTX expert interfaces, and distributed the Spark image to all
four workers over RDMA. The final run used empty RTX cards; a preliminary run
correctly failed AOT allocation while the previous service still occupied both
cards and produced no release image.

| Role | Local image identity | Executable SHA-256 | Native library SHA-256 |
|---|---|---|---|
| amd64 coordinator | `sha256:354561f06863b5f5ce9632edddee4a2d4e008a90d2d61d2ab8c309a3dc371f43` | `d536d9069f1755d4440a3ba0b0d18b5e761f52540a14e23c48f35eeb26e956a1` | `51f991cfe4c0cd84ac3931f82ff4d1fdb92c303e1dd1e60ed38be852fba1bcd1` |
| arm64 Spark expert | `sha256:622398bf2562f3d570fa8821715b501bfe69be718a00dc0dcf7304420fe4287b` | `e47465017ee0cbc8be269727014ab8e89a159b231fbbbd81fe1d6cb3cd464caa` | `87472d6105cd323c681afed9d0b0692595fd3c99d0c02cba7ccf5e5b74164ea6` |

The release bundles are `ds41rt-v3-coordinator-linux-amd64.tar.gz`
(`14e1fc097184eb3a8f375cc6ed16c82f551a208b7cd8a10c3e3e3fdc56a30aa1`),
`ds41rt-v3-spark-expert-linux-arm64.tar.gz`
(`9ba75ff744262a639beed7cb8a5197c42cc984c4379e41793f86ab90c7e934de`),
and `ds41rt-v3-qualification-evidence.tar.gz`
(`4d724a2ef7c78a93ff63f5bf1814c89c7a380338d61bb1c19b7bbbfa7dd03295`).
All three archives pass gzip validation and their embedded per-file checksum
manifests. The external `SHA256SUMS` is 321 bytes and has SHA-256
`35ba1d5f9f6dd148f715e31980bfa74a068ffb930c7f2069abd1e0e55fb66e17`.

## Standard launch

`./run.sh --dry-run` selected both peer-connected RTX cards and Spark routed
layer 20. The exact image pair then reached the OpenAI-compatible API on port
8000 with automatic two-RTX selection, C16, dSpark, 24 prompt snapshots, 24
completed-turn snapshots, and the 13.094 GB global FP4 pool representing
14,680,064 logical tokens plus 32,768 private-tail tokens.

An API smoke request returned exactly `V3 READY`. Loaded memory was 95,338 MiB
on RTX0 and 95,578 MiB on RTX1, leaving 1,913 MiB and 1,670 MiB free. Both cards
retained the qualified 400 W power limit and standard 14,001 MHz maximum memory
clock. All four Spark containers remained healthy on the matching v3 image.

## Registry publication

The unrelated older package versions carrying `v3` were superseded by pushing
the new manifests. Their tag lists are now empty; the historical `v1` and `v2`
versions were untouched. `./push-containers.sh v3` published the following
public images:

| Role | Published digest | Platform manifest/config |
|---|---|---|
| coordinator | `sha256:354561f06863b5f5ce9632edddee4a2d4e008a90d2d61d2ab8c309a3dc371f43` | amd64 manifest `sha256:36f2197dadbb4f561c0b339cf8f1e6e389c68b6294419816b285a273c95bea85`; config `sha256:b088cf290844c33ec77270f799b6ab7b63a596eabd4c9fbd447dbe4fd9f9f090` |
| Spark expert | `sha256:78819846cd2514db80ff6196709ca778fdce245d9967dfbdccef928ac032c09e` | arm64 config `sha256:622398bf2562f3d570fa8821715b501bfe69be718a00dc0dcf7304420fe4287b` |

Raw manifest comparison proves that `v3` and `latest` are identical for each
role. Both images identify source revision `c38746a`, version `v3`, the correct
role and architecture, and one/two-RTX support. The performance qualification
was recorded from source behavior at `f652dc8`; every later source commit adds
reports, the dual-layout SVG, release identity, or OCI description metadata and
does not change inference execution.
