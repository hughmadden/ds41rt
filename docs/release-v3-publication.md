# DS41RT v3 publication record

DS41RT v3 is published as [DS41RT
v3](https://github.com/tpurtell/ds41rt/releases/tag/v3). It is a final release,
not a draft or prerelease. GitHub release ID `388252464` contains the high-level
dual-RTX overview, measured performance impact, exact container references, and
links to the detailed reports.

Published release assets:

| Asset | Bytes | SHA-256 | GitHub asset ID |
|---|---:|---|---:|
| `ds41rt-v3-coordinator-linux-amd64.tar.gz` | 17,772,772 | `14e1fc097184eb3a8f375cc6ed16c82f551a208b7cd8a10c3e3e3fdc56a30aa1` | `562979524` |
| `ds41rt-v3-spark-expert-linux-arm64.tar.gz` | 15,889,235 | `9ba75ff744262a639beed7cb8a5197c42cc984c4379e41793f86ab90c7e934de` | `562979519` |
| `ds41rt-v3-qualification-evidence.tar.gz` | 22,918,901 | `4d724a2ef7c78a93ff63f5bf1814c89c7a380338d61bb1c19b7bbbfa7dd03295` | `562979514` |
| `SHA256SUMS` | 321 | `35ba1d5f9f6dd148f715e31980bfa74a068ffb930c7f2069abd1e0e55fb66e17` | `562979518` |

GitHub's asset metadata reports the same size and digest for every uploaded
file. Both binary archives pass their embedded per-file checksum manifests; the
evidence archive preserves every fresh one/two-RTX sample, the compact report,
machine-readable summary, final image identity, and a complete checksum
manifest.

Published containers:

- `ghcr.io/tpurtell/ds41rt-coordinator:v3` and `latest`:
  `sha256:354561f06863b5f5ce9632edddee4a2d4e008a90d2d61d2ab8c309a3dc371f43`
- `ghcr.io/tpurtell/ds41rt-spark-expert:v3` and `latest`:
  `sha256:78819846cd2514db80ff6196709ca778fdce245d9967dfbdccef928ac032c09e`

Raw-manifest comparison confirms `v3` and `latest` are identical for each role.
The prior unrelated package versions that carried `v3` are now untagged; v1 and
v2 were untouched. Both public images identify measured source revision
`c38746aeac9204eff12b7d2f0433013ed24526c4`, version `v3`, their role and
architecture, SparkInfer revision, and one/two-RTX serving support.

The scoped v3 qualification records 8,454 tok/s best median prefill, 79.33
weighted dual-RTX dSpark tok/s, 1,181.49 aggregate code tok/s at C16, 309.06
tok/s at C16 mixed traffic, and exact reuse in all 240 retained-context
requests. It also passed a focused 32K needle, continuation, cancellation,
survivor, and recovery lifecycle check. The full feature suite was intentionally
not repeated; unchanged vision, tool, constrained-output, and agentic interfaces
retain their prior detailed evidence.

`dev`, `main`, `release/v3`, and annotated tag `v3` point to this publication
record. The qualified automatic two-RTX dSpark service remains healthy on port
8000 using the published coordinator image and matching Spark image on all four
workers.
