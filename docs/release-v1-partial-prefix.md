# Compression-boundary prefix reuse

Status: compression-boundary reuse is enabled in native admission and has passed
focused GPU, API and C16 qualification. Long-context release qualification and
the final pool sizing policy remain open.

## Reference contract

The pinned [official V4.1 report, section 3.2.2](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/DeepSeek_V41_Tech_Report.pdf)
describes replaying the final window of a cached encoder prefix, then processing
the new suffix. Replay reconstructs local SWA while reading existing global KV;
only new tokens generate global KV. The decoder then processes the final window
of encoder outputs. These reconstructed states are approximate and depend on the
hit position; numerical identity with a full forward pass is not promised.

The native sources use ratios 2/2/2/1. A common complete-group token boundary is
therefore a multiple of two. The 256-row physical page boundary is separate:
shortened prefixes may share a partially initialized physical tail, with writes
isolated through copy-on-write.

## Qualified building blocks

Retained source handles can now yield shorter initialized frontiers after their
original active request is gone. They retain only the required pages, reject
extension beyond initialized rows, and preserve later rows owned by a longer
snapshot. The compressor restores complete groups without reading a saved odd
carry; ratio-two odd boundaries are rejected, while ratio-one boundaries remain
valid. Read-only source views reject queries beyond the restored end.

Engram history can now be rebuilt under a fresh owner from just the immediately
preceding three normalized tokens, including image barriers. It preserves the
absolute position without hashing or fetching the older prefix. A loader helper
normalizes that bounded suffix through the official token map.

[Component evidence](release-v1-partial-prefix-components.json) records exact
hash agreement with sequential Engram histories, identity isolation, device
source truncation after release, divergent writes across all four stored planes,
compression alignment and CUDA memcheck with zero errors. Native admission now
uses these helpers for partial matches.

## Native admission and replay

The radix can choose a retained descendant when a prompt matches only part of
an edge. It compares the computation saved by that partial match with an exact
retained ancestor, accounting for the replay window. Only the selected value's
LRU clock advances. Admission truncates global sources at the aligned match and
reconstructs encoder windows over at most 128 cached tokens with global writes
disabled. New suffix tokens then advance the encoder and compressed sources.
The encoder-output suffix spans that transition, allowing the final-window
decoder path to seed decoder and dSpark state. Engram history advances through
replay while the committed global frontier stays fixed.

API cache-hit accounting excludes the replayed tokens: a match at token 3,884
with 128 replay tokens reports 3,756 reused tokens. Source-capacity preflight
reserves only new global rows, since replay consumes existing source pages.
The original longer snapshot stays valid after shorter or divergent requests.

The real-weight cache test restores sixteen requests from a longer snapshot,
replays 63 then 65 encoder tokens, appends a divergent suffix and completes the
128-token decoder window. It checks all twenty encoder attention bindings,
causal bounds, zero acceptance, ownership isolation and preservation of every
initialized byte in the original snapshot. CUDA memcheck reports zero errors.

Live target-only and dSpark checks cover a divergent question, an exact repeat,
a shorter branch, the original parent and a suffix spanning multiple prefill
chunks. Objective answers match the uncached reference in every case. Sixteen
simultaneous divergent branches pass in both modes, as do the existing C2/C6/C16
and cancellation/replacement checks.

Four warm AB/BA pairs per workload compare ordinary 256-token decode with the
original uncached APIs. Target-only counting/code medians change by +0.49%/+1.09%;
dSpark changes by -0.24%/+0.24%. Paired output text and token counts agree. This
small spread provides no clear regression signal in these short-context checks;
it does not replace retained-context or release throughput measurements.

The [admission manifest](release-v1-partial-prefix.json) records binary/source
identities and links the raw requests, SSE events, test output and measurements.

## Agentic rerun

With partial replay enabled, the same initial C16 tool-eval-bench configuration
completes all 88 scenarios: **125/138 basic + 32/38 hard = 157/176 points**,
with 72 passes, 13 partials and three failures. All six structured-output cases
pass again (12/12). Zero-score cases are TC-61 (analysis script not attempted),
TC-74 (calendar event not created) and TC-88 (answer-only number formatting).
The earlier retained-frontier run remains 155/176; this single rerun is not a
controlled attribution of score changes or one of the three final release runs.

A fresh isolated dsh coding task completes in 20.67 seconds with five main model
steps and eight actual local tool calls. Its three generated tests pass
independently. Actual Unicode totals, literal UTF-8 output and an additional
NFC/emoji case also pass. Continuation cache counts are 7,362, 8,390, 8,632 and
8,975 tokens, each exactly the preceding prompt plus committed output. The task,
configuration, generated files and complete session are archived. This check
preserves full-turn reuse while the new partial path is enabled; it is separate
from the final Frogger task and comprehensive Unicode qualification.

## Remaining release work

[Large exact-resume suffixes](release-v1-exact-suffix.md) now restore encoder
state and use bounded decoder prefill. Short continuations keep the complete
retained-turn fast path.

Qualify long-context quality and measure replay cost and retained-context
prefill across the full release matrix. Final pool sizing and memory reservation
options remain separate open release gates.
