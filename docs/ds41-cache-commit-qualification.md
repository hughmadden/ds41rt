# RTX backbone cache commit qualification

The production `BackboneCache::commit` passed an actual RTX test across all **40 window and four compressed-source owners**, using the official checkpoint weights and generated finite BF16 attention inputs. The retained test is `v41_backbone_cache::commit_tests::real_all_cache_commits_preserve_prefixes_and_revoke_partial_failure`.

Sixteen requests ran three proposal/acceptance cycles:

| Cycle | Proposed tokens per request | Accepted tokens |
| --- | ---: | --- |
| Partial prefix | 9 | request index modulo 10 |
| Window wrap | 129 | 129 minus request index modulo 4 |
| Reject all | 3 | 0 for every request |

For every accepted token, the test retained the exact produced FP8 window bytes and scales and compared the logical last 128 positions against committed ring destinations. It checked all host ends and device ends. For each compressed source, it retained only completed latents within accepted prefixes, then compared committed FP8 KV values/scales and native FP4 index keys/scales against those produced rows. It also checked paired KV/index page lists, device page tables, host row counts and device row counts. Mixed odd/even prefixes exercise ratio-two pending-group ownership across calls. Old batches and repeated commits reject after every cycle, including zero acceptance.

The initial test bank provides 16 physical pages per source. A separate bank provides 16/16/16/1 pages to sources 2/8/14/20. Accepting two tokens for each of 16 requests deliberately exhausts source 20 after the preceding 40 windows and three sources have committed. The expected `index cache pool exhausted` error occurs; every root request lease is revoked. All 16 slots readmit with zero history. A subsequent one-request accepted commit succeeds through all 44 owners, proving that pages consumed by the earlier sources were reclaimed; the other 15 histories remain zero.

The final test passed in **3.71 seconds** on RTX GPU 0 with the [isolated matching driver libraries](ds41-rtx-testing-restored.md). Workspace capacity is the supported **4096-row AOT bucket**, with 144, 2064 and 48 live rows for the three cycles. The initial fixture mistakenly requested a 2064-row AOT capacity and correctly failed during metadata lookup; the corrected test uses the supported capacity while retaining the intended 2064 live rows.

This qualifies accepted-prefix publication, committed storage bytes, ring wrapping, paired source publication and recovery from late page-pool exhaustion for this workload. It does not inject a CUDA hardware fault or transport cancellation, qualify two concurrent passes, establish independent reference arithmetic, or execute the query-derived cache/attention/FFN handoffs. The producers run their real uncaptured `execute` methods; the complete model driver and engram/dSpark transaction remain open.

Log: `/tmp/ds41-cache-commit-rtx-test.log`. External build harness: `/tmp/ds41-lane-ffn`. The test is retained in the production tree, and [the evidence record](ds41-cache-commit-qualification.json) records source/library hashes.
