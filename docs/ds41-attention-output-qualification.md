# Real backbone attention output

`v41_attention_output.rs` owns the official backbone output weights and captured inverse-rotary/grouped-projection path. The graph generates layer-specific frequencies, conjugate-rotates the last 64 coordinates of each of 64 attention heads, applies eight independent BF16 `wo_a` groups (4096→1024 each), and applies K32 FP8 `wo_b` (8192→5120). It uses the existing native rotary, grouped cuBLAS and exported FP8 operations.

Loading dequantizes official FP8/E8M0 `wo_a` into the BF16 representation used by the reference. The temporary source allocation is released only after dequantization drains, before loading `wo_b`. Immutable weights and packed scales are shared between waves. Final resident storage is 110,403,584 bytes per layer; the earlier dequantization phase peaks at 100,696,064 bytes. Admission checks the greater of transient and final allocation before payload reads.

Each wave owns BF16 attention input, inverse-rotated input, grouped output, final projection, absolute positions and frequencies, plus its own grouped handle/workspace, FP8 scratch and aligned scalar. Device allocation is 4 MiB plus the exported FP8 scratch requirement, four scalar bytes and 157,960 bytes per capacity row. Capacity80 uses 17,539,716 bytes; capacity4096 uses 686,850,052 bytes. Supported capacities are 1, 16, 80, 256, 1024 and 4096.

Capture fixes the live row count. Changed inputs/positions replay with shared weights and separate wave workspaces. Outputs borrow their wave; preparation clears publication, failed launches drain, and graph destruction precedes workspace/storage release. Input/request correspondence remains the caller's explicit unsafe contract pending backbone composition.

## Qualification

Both RTX PRO 6000 Blackwell GPUs pass **60 owner cases**: every backbone layer at capacity80, plus the other five capacities on layers 0, 1, 20 and 39. Each case uses two independent captured waves sharing real checkpoint weights and compares their final outputs exactly before and after changed input/position replay. Coverage includes exact/one-byte-short budgets, unpublished output, replay before capture, duplicate capture, mismatched live rows, recovery and graph clearing. Inputs are finite fixtures, not outputs of a fully composed backbone attention layer.

`scripts/qualify-ds41-attention-output.py` checks **120 output sets per GPU**. Frequencies and inverse rotary match the actual pinned reference exactly. Grouped projection is compared with the reference BF16 weight conversion and `einsum` expression. Final FP8 projection uses the actual pinned quantizer and GEMM, taking the native grouped BF16 intermediate to isolate stage arithmetic. Every coordinate passes `rtol=0.008, atol=0.002`; maximum absolute differences are 0.015625 for grouped output and 0.03125 for final projection on both GPUs. This is not a claim of byte-exact end-to-end projection or full-model logits.

The qualifier pins reference source hashes and records checkpoint/vector payload hashes. It retains the documented TileLang 0.1.8 quantizer compiler workaround `tir.disable_vectorize=True`, without changing model source or arithmetic. Daemon and external owner-fixture builds pass. Adjacent JSON records source, fixture, build and result hashes. Reusable numerical qualification is tracked; the small Rust driver and vectors remain outside Git.

Complete query/index/attention/output composition, backbone mHC/CED execution, scheduler-wide transactions, dSpark FP8 storage, vision and full-model correctness/performance remain open.
