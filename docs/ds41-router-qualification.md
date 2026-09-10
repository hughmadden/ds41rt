# Native V4.1 routing

The native router implements the pinned official Gate at hidden width 5120: FP32 accumulation of BF16 hidden/weights, sqrt(softplus(logits)), correction-biased selection, and unbiased selected-score normalization with denominator epsilon 1e-20 and route scale 1.5. It supports 128/top-3 dSpark experts and 384/top-6 backbone experts, including per-row image-mask selection of the VL correction bias. Exact corrected-score ties choose ascending expert IDs; PyTorch does not guarantee identical tie ordering. Very close scores can change selection with FP32 reduction order.

Two allocation-free kernels produce scores and sorted IDs/weights in caller-owned storage. The native entry validates geometry, alignment, extents and output overlap. It requires finite logits and correction biases. This is a correctness implementation; throughput and fusion with expert dispatch remain unqualified.

Each DsparkRouter borrows its exact stage's checkpoint owner and owns FP32 scores scratch. ExpertExecution::route_draft rejects a router from different resident stage weights, writes directly into stable expert input buffers, and drains the wave stream even if launch returns an error. The router's mutable borrow excludes simultaneous reuse of scratch. This synchronized fallback is not a combined routing/expert CUDA graph.

At capacity 80, scratch is 40,960 bytes per stage, 122,880 bytes per wave, and 245,760 bytes for two waves. The combined partial dSpark admission budget is now 8,401,515,352 bytes; shared vocabulary-head residency, attention caches, dense packing/scratch and driver/graph overhead remain separate.

CUDA native and Rust daemon builds passed. The reusable scripts/qualify-ds41-dspark.py passed on both RTX PRO 6000 GPUs using the pinned reference's formulas: both expert geometries at rows 1/16/80, changed-input graphs, text/VL correction, selection-only bias, exact ties, extreme finite logits and invalid-buffer guards. Selected IDs matched the reference in tested non-tie cases; maximum score error was 4.7683716e-7 and routing-weight error 1.1920929e-7. The prior mHC, head, Markov, confidence and sampling cases also passed in this run.

Machine-readable evidence, library and qualifier hashes are in ds41-router-qualification.json. These checks exercise native calls with synthetic data, not the Rust-owned router/expert composition or full model execution. Full dSpark attention/shared-FFN execution, verification, rollback, serving and alternating-wave scheduling remain open.
