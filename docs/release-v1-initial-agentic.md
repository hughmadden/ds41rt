# Initial native agentic qualification

The first tool-eval-bench run and a real dsh coding task completed on the native
dSpark candidate with retained-frontier reuse enabled. This is an initial
correctness check, not one of the five final release benchmark runs. Source and
binary identities are in the [admission manifest](release-v1-native-admission.json).

## Tool-eval-bench

Version `2.6.1.dev45+gcf54b4bfe` ran 88 scenarios, including hard mode, at C16,
temperature zero, thinking disabled, a 600-second request timeout and twelve
maximum turns. All scenarios completed without an inference transport error.

| Suite | Points |
| --- | ---: |
| Basic | 123 / 138 |
| Hard | 32 / 38 |
| Total | 155 / 176 |

All six structured-output scenarios passed (12/12 points). This is observed
schema compliance; it does not prove that arbitrary schemas are enforced by a
constrained decoder. The general tools/structured-output release gate stays open.

The three zero-score scenarios were async polling (TC-61), ambiguous recipient
lookup (TC-71), and answer-only number construction across follow-ups (TC-88).
TC-61 declined to execute a named analysis function with the offered tools.
TC-71 asked the user for details without first looking up contacts. TC-88 included
unrequested explanation instead of only a twenty-digit answer.

A focused C3 comparison used the original uncached dSpark API and the candidate.
Both failed TC-61/88 and passed TC-71. Text was not identical, and this smaller
comparison does not replace the original C16 score or establish a cause for its
variation. Preserve these cases for the final numerical/concurrency campaign.

The benchmark's `vllm` backend field is its supported OpenAI-compatible adapter
label. Both endpoints are **ds41rt `serve-native`**. An attempt to use a `ds41rt`
label was rejected before inference and is retained in the setup evidence.

## Real dsh coding task

dsh `0.1.5-rc.1` used an isolated home and workspace, a local API endpoint, a
workspace-write sandbox, and a 4,096-token per-call output allowance. It read a
Unicode transaction fixture, wrote a Python implementation and tests, ran them,
inspected the generated JSON and reported the result. No external model API was
used. Six main model steps and eight local tool calls completed in 20.84 seconds,
including harness overhead and a separate title request.

The generated four-test suite passed independently. Additional verification
checked literal UTF-8 output, NFC keys and a separate combining-character/emoji
case. The actual output was `{"café": 11, "台北": 2}`. This is useful Unicode
coverage within a tool workflow, not comprehensive tokenizer/streaming proof.

| Model step | Reused prompt tokens | Uncached prompt tokens | Output tokens |
| --- | ---: | ---: | ---: |
| 1 | 0 | 7,249 | 123 |
| 2 | 7,372 | 238 | 869 |
| 3 | 8,479 | 92 | 104 |
| 4 | 8,675 | 131 | 89 |
| 5 | 8,894 | 169 | 219 |
| 6 | 9,282 | 183 | 435 |

Every continuation reused the preceding prompt plus its committed generated
tokens. A one-token difference in step five is consistent with an emitted token
that had not been committed. This exercises full retained-turn resume through
real file and shell tools, beyond the earlier short access-code conversation.

An initial dsh overlay had inconsistent sandbox/approval preset definitions and
failed before inference. The corrected isolated overlay retains the workspace
sandbox and explicitly defines its noninteractive preset. Both attempts are saved.

## Evidence and remaining work

[Machine-readable results](release-v1-initial-agentic.json) contain point splits,
follow-up verdicts, dsh usage and provenance references.
[Compressed raw evidence](evidence/native-initial-agentic.json.gz) preserves
benchmark results and traces, the dsh session, exact task and overlay, generated
files, independent test output and setup failures. The dsh session includes stream
timing, but this task is not a controlled performance comparison.

Next cache work is compression-boundary reuse inside radix edges with bounded
SWA reconstruction, followed by final pool sizing. Full vision, comprehensive
Unicode/API qualification and the complete release performance campaign remain
open. The final Frogger task is separate from this small coding check.
