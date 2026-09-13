# DS41RT v2 publication record

DS41RT v2 is published as [DS41RT v2](https://github.com/tpurtell/ds41rt/releases/tag/v2).
It is a final release rather than a draft or prerelease. GitHub release ID
`387948994` contains the concise high-level overview, measured optimization
impact, exact container references, and links to the detailed reports.

Published release assets:

| Asset | Bytes | SHA-256 | GitHub asset ID |
|---|---:|---|---:|
| `ds41rt-v2-coordinator-linux-amd64.tar.gz` | 17,028,492 | `63576c0dd347c675f7f62bb55c6d03c9cc15d0e4f56a23181f283b84317178e5` | `561480165` |
| `ds41rt-v2-spark-expert-linux-arm64.tar.gz` | 15,435,229 | `41ccd1a316b623652fa8a631bff5a1a247a1448895a26d72580d1471a14cbbb2` | `561480168` |
| `ds41rt-v2-qualification-evidence.tar.gz` | 13,151,519 | `e462a1db3ab8ab9df9a0a5edaa60b89c5172da820d5afe366db3db50cee6a9` | `561480160` |
| `SHA256SUMS` | 321 | `8724d8ef0404ca4c92a2ab99742c309f4d6f7c5c5399b467e54eab891ebf3b56` | `561480167` |

GitHub's asset metadata reports the same size and digest for every uploaded
file. Both binary archives pass their embedded per-file checksum manifests, and
the evidence archive passes its complete manifest after extraction.

Published containers:

- `ghcr.io/tpurtell/ds41rt-coordinator:v2` and `latest`:
  `sha256:3ffe75377cb901119b0ead470d4e047ff5b623c8c0eac6b98024e51814a222a8`
- `ghcr.io/tpurtell/ds41rt-spark-expert:v2` and `latest`:
  `sha256:d9b4bc411eb3480a956d3a993597c5b102bbd438adb9c73e57d6ba2e5cf21e39`

Raw-manifest comparison confirms `v2` and `latest` are identical for each role.
The public coordinator package contains a linux/amd64 platform manifest and its
BuildKit attestation; the public Spark package contains the linux/arm64 image.
Both identify measured source revision
`9477b6e39f4bbe431c4cd6d48c8b303045f9238f`. Later source commits add launcher
logging and release documentation only; no inference implementation changed
after the clean image build.

The scoped v2 qualification records 79.80 weighted dSpark tok/s, 934.05
aggregate tok/s at C16, verified retained-prefix decode through 256K, and three
C16 high-thinking tool campaigns scoring 155, 153, and 158 of 176. The prefill
matrix and official API comparison are explicitly preserved v1 measurements;
the full v1 feature suite was not rerun.

`dev`, `main`, `release/v2`, and annotated tag `v2` point to the publication
record. The qualified standard v2 dSpark service remains healthy on port 8000,
using the published coordinator image and the matching Spark image on all four
workers.
