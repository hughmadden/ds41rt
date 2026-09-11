# Paired live speculative quality checks

Eight prompts were run sequentially through the target-only API on RTX GPU 0 and the greedy dSpark API on RTX GPU 1, sharing the same four resident Spark expert workers. Both used build `4f0c056`, the official checkpoint and temperature zero. Text and complete token-usage objects matched exactly for every pair.

Six objective cases passed in both modes: integer arithmetic, signed sorting with duplicates, JSON extraction, ordering constraints, Chinese arithmetic and retrieval from 100 records. Two open-ended cases were compared without an automated quality score. Inspection found the Traditional Chinese translation faithful and the two-sentence explanation of heat conduction correct. This small set is a regression check, not a standardized benchmark or evidence of broad model quality.

| Case | Prompt tokens | Completion tokens including EOS | Target finish seconds | dSpark finish seconds |
| --- | ---: | ---: | ---: | ---: |
| arithmetic | 20 | 2 | 0.386 | 0.405 |
| signed_sort | 37 | 19 | 2.143 | 0.990 |
| extraction | 42 | 17 | 1.932 | 0.976 |
| logic | 37 | 13 | 1.534 | 0.656 |
| unicode | 25 | 2 | 0.396 | 0.436 |
| translation | 21 | 8 | 0.946 | 0.758 |
| explanation | 26 | 29 | 3.102 | 2.067 |
| long_retrieval | 1363 | 5 | 9.224 | 8.694 |

Finish times run from HTTP request start to the finish event. These are single observations per case, including prefill and transport overhead; graph warmth varies across cases. dSpark helps several longer answers but slightly increases time on the two-token arithmetic answers. This does not establish a universal speedup.

The 1,363-token retrieval case exercises multiple 80-row prefill chunks and more than one wrap of the 128-row dSpark windows. Both modes return `quartz-731`. First content arrives at 8.597 seconds target-only and 8.478 seconds with dSpark, showing that prefill dominates this case. Those times are not a kernel-only prefill benchmark, and this prompt is below the requested 8k+ prefill workload.

Run `python3 scripts/qualify-ds41-speculative-quality.py --output /tmp/quality.json` against the development endpoints; optional `--target-url` and `--speculative-url` select other instances. The script records each completed pair immediately, preserves errors, checks exact text/usage equality, and exits unsuccessfully on objective failures or differences. Open-ended answers still require inspection.

These checks do not compare against hosted DeepSeek or independently establish mathematical equivalence. Longer contexts, diverse difficult prompts, stochastic sampling, concurrent acceptance and sustained performance remain open.

[Prompts, full responses, timings and checks](ds41-speculative-quality.json).
