# Completed checkpoint inventory

The coordinator raptor and ostrich, dodo, emu and kiwi each have snapshot `dba1be0a40aa45a94ad051997016db3960a90277` under `/home/tj/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/`.
All 48 shards are present, totaling 510,296,708,312 bytes, and all five hosts have identical index, shard-header and audited asset hashes.
The older `2bc89ac599031fa673cab993f1df02fc4a98c673` cache entries lack shards 47 and 48 and must not be selected for bring-up.

The production Rust catalog reader accepts the completed local checkpoint's 96,085 tensors and reports these source-storage amounts:

| Placement | Checkpoint bytes |
| --- | ---: |
| Coordinator excluding dSpark | 18,750,160,200 |
| dSpark on RTX | 7,932,874,632 |
| Each Spark's TP4 share | 72,194,457,600 |
| Host-mapped engram tables and scales | 202,758,032,400 |

These are checkpoint bytes, not runtime GPU allocation or measured I/O: packed storage, staging, caches and execution workspaces require separate admission.
At a sustained 5 GB/s, each Spark's useful weight share alone corresponds to about 14.4 seconds of read time; this is a planning estimate, and physical bytes read, filesystem caching, packing and end-to-end load time still need measurement.
Graph setup has a separate 75-second target ceiling under the unified release plan.

All audited assets present in `docs/ds41-reference-lock.json` match that lock, including model/kernel code, configuration, tokenizer vocabulary, conversion, engram and image processing.
`tokenizer_config.json` was not in that lock; its completed-snapshot hash agrees across all hosts and its bytes match the older local snapshot.
The reference revision stays `df42c109f1defefcbfcedbe7d905718a12266e40`, distinct from the completed checkpoint snapshot revision.

The reusable `scripts/audit-ds41-checkpoint.py SNAPSHOT` reads headers and assets, checks index ownership and contiguous tensor extents, and verifies declared extents against actual file sizes.
It does not scan or hash the 510 GB of weight payloads, prove artifact authenticity, load the model, or establish numerical correctness.
Production local catalog validation adds strict architecture, shape and dtype checks; identical remote header hashes establish the same metadata, while selected checkpoint-backed execution and full-model qualification remain open.
