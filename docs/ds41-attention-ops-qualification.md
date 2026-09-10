# dSpark normalization and rotary operations

Normalization evidence in this historical record is superseded by [the official epsilon correction](ds41-normalization-correction.md), which fixes the inherited `1e-6` RMS epsilon to `1e-20` and reruns primitive, stage and complete-draft checks.

The native attention operations now provide RMS normalization for dimensions 512, 1280 and 5120, fused KV normalization/rotation at dimension 512, and forward/inverse rotation for one or sixty-four 512-wide heads. Rust bindings validate buffer lengths and geometry; the C ABI rejects malformed pointers, overflow and output overlap. Operations allocate no storage, launch asynchronously and support graph capture.

Normalization promotes BF16 input and weight to FP32, originally applied epsilon 1e-6 (now corrected to the official 1e-20), and rounds to BF16 before any rotation. Fused KV normalization retains that intermediate rounding boundary and rounds the rotated pair again. RoPE rotates adjacent pairs in the final 64 dimensions and copies the preceding dimensions unchanged. Frequencies are supplied as per-row FP32 complex pairs, allowing mixed request positions without requiring a shared batch offset.

Qualification extracts the unmodified RMSNorm, apply_rotary_emb and precompute_freqs_cis definitions from the hash-checked reference. dSpark uses the pure sliding-window branch, so tests disable YaRN and use base 10000 with independently selected positions below 262144. Frequency construction in the production runtime remains unfinished.

Both RTX PRO 6000 GPUs passed 48 cases each at rows 1, 16, 80, 255, 1023 and 4095. Standalone forward/inverse rotations are bitwise equal to the reference, including changed-input/frequency graph replay. Normalization and fused normalization/rotation pass 0.008 relative and 0.002 absolute tolerance, with exact zero outputs and tiny-input checks; reduction order can affect BF16 rounding. Invalid geometry and overlapping output checks passed.

Initial qualification exposed a missing reference decorator import in the harness and then complex-multiply contraction differences. The final kernel explicitly matches the CUDA reference multiplication order, using fma(x,c,-round(y*s)) and fma(y,c,round(x*s)) before BF16 rounding. No tolerance was relaxed to resolve the rotation mismatch. Final per-case errors and source/binary hashes are recorded in `ds41-attention-ops-qualification.json`.

The native and daemon builds pass. These are attention primitives with callable Rust bindings; frequency ownership, Rust-owned projection/norm/cache composition, attention softmax/sink, grouped BF16 output projection, stage sequencing and serving integration remain open. There is no full-model or performance claim.
