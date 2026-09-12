# Compression-boundary prefix reuse

Status: component prerequisites qualified; partial reuse is not yet enabled in
native admission. Exact retained-frontier reuse remains the qualified live path.

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
compression alignment and CUDA memcheck with zero errors. These helpers are not
yet called by partial-hit admission, so the live serving artifacts are unchanged.

## Remaining integration

The radix must choose a retained descendant when a prompt matches only part of
an edge. Admission will truncate global sources at the aligned match, reconstruct
encoder windows over at most 128 cached tokens with global writes disabled, then
continue encoder prefill through the uncached suffix. The retained encoder-output
suffix must span that transition so the existing final-window decoder path can
seed decoder and dSpark state. Engram history advances through the replay segment;
global source ownership remains fixed until new tokens begin.

Keep exact retained-turn restoration as the short-continuation fast path. Large
uncached suffixes should use encoder continuation plus bounded decoder prefill
instead of unnecessarily executing all forty layers over every new token.

Before enabling this path, qualify causal source bounds, replay truncation,
failure invalidation, odd/even boundaries, zero/partial acceptance, divergent
branches, eviction, C16 isolation and long-context quality. Measure replay cost
and retained-context prefill against the existing path. Final pool sizing and
memory reservation options remain separate open release gates.
