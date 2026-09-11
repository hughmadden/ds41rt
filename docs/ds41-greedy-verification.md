# Real greedy draft verification

`verify_dspark_greedy` consumes an already emitted anchor plus up to five proposed tokens and the corresponding target next-token predictions. It accepts matching draft prefixes, emits the first target correction or the final bonus token, and returns the input-row count to commit. A correction or bonus remains unevaluated as input until the next pass. EOS and remaining output length stop emission and prevent the rest of the proposed tail from being accepted.

The real-model fixture runs all six target verification rows, applies this decision to the combined target/engram/dSpark transaction, and resumes from a correction or bonus where applicable. On RTX GPU 1 with four Spark expert workers:

| Case | Accepted input rows, including anchor | Result |
| --- | ---: | --- |
| Count from 1 to 20 | 6 | All five drafts match; bonus and resumed token exactly match sequential target decoding. |
| Same prompt, second draft deliberately changed | 2 | Only anchor and first draft commit; correction and resumed token exactly match sequential target decoding. |
| Arithmetic answer `4` | 2 | First draft is EOS; only EOS is newly emitted and no resume occurs. |

The unmodified counting run produces token IDs `[19,14,223,20,14,223,21,14]` including the anchor and resumed prediction, exactly matching eight sequential target-only outputs. The forced-rejection run matches the first four of those IDs. Both runs validate target and all three dSpark committed ends after verification and resumption; engram history consistency is checked during resumed request preparation. The arithmetic run emits `[22,1]` including its anchor, matching the earlier target-only result.

CPU tests cover each mismatch position, full acceptance plus bonus, length truncation, matched and unmatched EOS, completed requests and malformed inputs. The production release build passes. These checks establish the greedy decision and a few real single-request commit/resumption cases. They do not establish broad quality, stochastic verification, multi-request acceptance, a speculative API path, or performance targets.

[Reports, token comparisons and test logs](ds41-greedy-verification.json).
