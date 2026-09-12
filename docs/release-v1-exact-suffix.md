# Encoder continuation after a retained turn

An exact retained prefix with at least 128 new prompt tokens restores the twenty
encoder SWA rings, all four global sources, odd compressor carry and Engram
history. Decoder and dSpark rings remain fresh. New tokens run through the
encoder; the decoder processes only the final 128 encoder outputs and seeds the
draft state. Every final-window encoder output comes from the new suffix, so
this path needs no encoder replay or older encoder-output buffer.

Shorter suffixes restore all forty target rings and the draft rings, then run the
full backbone over only the new tokens. Complete hits still use the saved next
greedy token. Partial matches keep their compression-aligned bounded replay.
Selection happens during admission, avoiding copies of rings that would be
discarded immediately before decoder replay.

## Qualification

The real-weight cache test compares restored encoder and global state with full
continuation from the same 385-token prefix. It appends 128 then one token,
covering the saved odd carry and the following complete compression group.
Every initialized encoder and global byte agrees. Decoder rings stay empty until
the final-window replay, after which the request returns to ordinary decode.
The retained original stays byte-for-byte unchanged. The same test retains its
sixteen-request partial replay and all-owner failure recovery coverage. CUDA
memcheck reports zero errors.

The API comparison uses the previous cached implementation on the same GPU and
shared expert workers. Each measured request follows a completed parent turn of
about 1,960 tokens, then appends 160 through 14,376 tokens. Unique session tags
prevent complete-prompt hits. AB/BA order alternates, and raw evidence includes
the initial warmup and every sample. The access-code answer and complete usage
accounting must agree within each pair.

This is a focused short-context qualification, not the release's retained-context
prefill matrix. Bounded decoder reconstruction is the model's approximate CED
policy; objective answer checks do not prove numerical identity with a full
forty-layer forward pass. First-use and warm timings must be distinguished.

After priming each suffix shape, median first-content latency from two AB/BA
pairs is:

| New tokens | Target previous / new (s) | dSpark previous / new (s) |
| ---: | ---: | ---: |
| 160 | 0.2021 / 0.1887 | 0.1985 / 0.1863 |
| 880 | 0.4946 / 0.3336 | 0.4902 / 0.3333 |
| 1,744 | 0.8726 / 0.5299 | 0.8664 / 0.5197 |
| 3,472 | 1.7116 / 0.6011 | 1.7065 / 0.5941 |
| 6,928 | 3.0606 / 1.1456 | 3.0391 / 1.1017 |
| 14,376 | 6.0888 / 2.0725 | 6.1104 / 1.9655 |

The initial campaign is retained separately. Its first target sample with a
3,472-token suffix took 2.0701 seconds versus 1.7454 for the previous path;
the next candidate sample took 0.5925 seconds. Warm results do not erase that
first-use observation.

Four paired ordinary-decode samples per workload preserve identical text and
token counts. Counting/code median changes are +0.46%/-0.21% for target-only and
-0.19%/+0.18% for dSpark. These small differences show no clear regression in this
focused comparison. Exact hits, partial hits, C2/C6/C16 and cancellation with
replacement pass in both modes.

A fresh dsh task exercises both large- and short-suffix continuations over seven
model steps and eight local tool calls. It finishes in 26.07 seconds; all five
generated tests and independent UTF-8/NFC/emoji checks pass. Every continuation
reuses the preceding prompt plus committed output. This cache diagnostic uses
explicitly disabled thinking to match the prior task; tool-eval qualification
and final agentic reporting use thinking enabled at high effort.

The [manifest and measurements](release-v1-exact-suffix.json) record source and
binary identities, individual timing summaries and the compressed raw evidence.
Final pool sizing, memory options and long-context release qualification remain
open.
