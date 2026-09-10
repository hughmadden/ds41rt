# dSpark FP8 projection qualification

The native bridge now supports the main-hidden projection and each stage’s query A, query B, KV and output B matrices, with thirteen Rust-owned tensor bindings and resident packed scales. These are raw FP8 linear operations; normalization, RoPE, grouped BF16 output A, attention/sink computation, private draft KV and full-stage sequencing remain unfinished.

The exporter retains the b12x planner’s choices. Main and output B at capacity one select two split-K slices, stored as FP32 planes and reduced natively before one BF16 rounding; all other exported variants use one slice. Scratch admission includes the partial planes. The existing default b12x export API remains compatible and the opt-in metadata export passes both regression cases.

All eight geometries at capacities 1, 16, 80, 256, 1024 and 4096 passed on both RTX PRO 6000 GPUs, using live row counts 1, 16, 80, 255, 1023 and 4095. Checks cover numerical projection, changed-input graph replay, exact activation quantization and scale packing, zero/tiny inputs, and invalid buffer/overlap guards. Projection comparisons use 0.008 relative and 0.002 absolute tolerance; BF16 outputs are not generally bitwise identical. Existing shared-FFN composition checks also passed using their recorded rounding/quantization propagation bounds.

A sparse synthetic checkpoint fixture loaded the production Rust owners and exercised all thirteen bindings at capacities one and sixteen on both GPUs. Stage-specific weight values 1, 1.5 and 2 and scale bytes 120, 119 and 118 independently distinguish weight and scale selection. Constant inputs +1 and -0.5 produce analytically checked BF16 results. Capture/replay, unpublished warmup output, invalid stage/row/budget rejection and recovery all passed. This fixture contains no real model payloads.

Projection packed scales add 11,182,080 resident bytes. Thirteen projection owners reserve 7,888,948 bytes per capacity-16 wave or 38,235,188 per capacity-80 wave. The current partial two-wave capacity-80 dSpark total is 8,521,855,064 bytes, excluding remaining attention/norm/RoPE/grouped-BF16 scratch, shared vocabulary residency, driver/graph allocations and additional larger prefill owners.

The native library and Rust daemon builds passed. Exact source/binary hashes, selected split variants and both GPUs’ results are recorded in `ds41-dspark-projection-qualification.json`; temporary fixtures and logs are listed in `TO_DELETE_SCAFFOLDING.md`. This is component correctness evidence, with no full-model quality or performance claim.
