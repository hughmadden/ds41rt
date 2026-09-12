# Final native prefix qualification

The clean-built corrected-FP4 candidate passes the complete native prefix contract: cold admission, compression-boundary reuse with bounded SWA replay, exact hits, divergent and shorter branches, multi-chunk continuation, C16 isolation, cancellation/replacement, and 24-turn LRU eviction. The final service uses FP4 E2M1 compressed KV with group-16 E4M3 scales, FP8 SWA, and the independent FP4 index.

This campaign used coordinator image `sha256:a13258e92dd25ddb889bd31bb77c8813c7881868b103ceec0c140e30e893c213`, Spark image `sha256:2f5d328a14f1a52a0d3b2356b041415d744635f3f3557e41403246429984093b`, model revision `dba1be0a40aa45a94ad051997016db3960a90277`, the enforced 400 W RTX limit, and standard memory speed. The candidate ran dSpark at C16 on port 8000. An uncached target-only instance of the same image ran on the second identical RTX with retention disabled and a 2 GiB test pool; requests were issued serially across the shared Spark workers.

## Replay and branch cases

| Case | Reuse | Prompt | Hit | Miss | Candidate TTFT | Uncached TTFT | Answer |
|---|---|---:|---:|---:|---:|---:|---|
| Parent seed | cold | 3,887 | 0 | 3,887 | 3.157 s | 3.335 s | `OK` |
| Divergent suffix | partial | 3,894 | 3,752 | 142 | 0.350 s | 3.748 s | `quartz731` |
| Exact repeat | full | 3,894 | 3,894 | 0 | 0.005 s | 1.521 s | `quartz731` |
| Shorter branch | partial | 1,974 | 1,830 | 144 | 0.334 s | 1.135 s | `quartz731` |
| Original parent | full | 3,887 | 3,887 | 0 | 0.005 s | 1.519 s | `OK` |
| Multi-chunk suffix | partial | 6,198 | 3,750 | 2,448 | 1.304 s | 3.333 s | `quartz731` |

Every answer and total prompt-token count matches the uncached execution. Partial hits exclude the replayed window from reported cache hits, leave at least 128 tokens for bounded reconstruction, and preserve the longer parent after copy-on-write branching. Exact hits reuse all prompt tokens and retain the computed first token.

## Concurrency and eviction

Serial references were followed by C2, C6, and C16 requests. All 24 concurrent results match their corresponding serial output and token counts. A C16 cancellation run cancelled three streams after content, admitted four replacements, and completed every non-cancelled request correctly.

The eviction pass creates 24 distinct completed turns, proves all 24 complete prompts hit, and resumes each completed assistant frontier exactly. Inserting turn 25 evicts the least-recently-used oldest turn: its followup reports zero cached tokens. The untouched second-oldest turn still restores 34 committed tokens. Prompt snapshots and completed turns therefore retain independent 24-entry banks rather than splitting one limit.

The radix matches tokens within compressed edges, aligns source reuse to the two-token common compressor boundary, and reconstructs no more than the last 128 encoder-window tokens. Retained source pages remain shared; appending to a shared partial page uses copy-on-write, while eviction drops the radix value, target/dSpark tails, and page references together. Source admission preflight can evict retained values before active work and never evicts active owners.

Raw requests, SSE records, scheduler admissions, cancellation results, and restored standard-launch logs are preserved in [`evidence/native-release-prefix-final.tar.gz`](evidence/native-release-prefix-final.tar.gz). Structured results and exact source hashes are in [`release-v1-prefix-final.json`](release-v1-prefix-final.json).
