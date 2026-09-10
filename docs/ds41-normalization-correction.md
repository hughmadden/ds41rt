# Official normalization epsilon correction

The pinned official `inference/config.json` sets `norm_eps=1e-20` and `hc_eps=1e-6`. `RMSNorm`'s standalone Python constructor defaults to `1e-6`, but model construction explicitly passes `args.norm_eps`. The previous native attention norms, mHC coefficient RMS statistic and dSpark attention/FFN/final-head norm calls incorrectly used that constructor default. This can substantially change low-magnitude activations and was a production numerical bug.

Native attention normalization (including fused KV normalization/RoPE/FP8), the flattened mHC RMS statistic and Rust sublayer/final-head normalization now use `1e-20`. The mHC pre-mix offset and Sinkhorn denominators remain `1e-6`, as required. Engram normalization already used `1e-20`; its independent `1e-6` gate clamp also remains unchanged. Historical V4 execution defaults are outside this correction and still require removal during runtime migration.

Both committed qualifiers now hash-check and read the official inference configuration, separately identifying the two epsilon values. The attention qualifier explicitly constructs the extracted reference RMSNorm with the configured epsilon. Previously it instantiated the default constructor, so its numerical agreement did not detect the same epsilon mistake in the native kernel. The mHC qualifier adds changed-input graph checks at magnitudes `1e-8`, `1e-10` and `1e-12`, spanning both sides of the configured RMS epsilon scale.

Qualification passed on both RTX PRO 6000 GPUs:

- 54 native attention-operation cases per GPU covering normalization dimensions 512/1280/5120, zero/tiny inputs, norm-plus-RoPE rounding, fused KV quantization and graph replay, across row counts 1/16/80/255/1023/4095. Existing BF16 comparison tolerances remain unchanged; pure RoPE and fused-versus-separated KV checks remain exact.
- mHC coefficient generation at tiny magnitudes with maximum absolute errors below `4.8e-7`; coefficient tolerances remain `2e-5` absolute/relative. The existing mHC pre/post, router, vocabulary/Markov/confidence, RNG and sampling checks also pass.
- 18 complete-stage cases per GPU match the corrected structured oracle bitwise; the FFN oracle uses `1e-20` while keeping mixing epsilon `1e-6`.
- Six complete draft graph cases per GPU match separately synchronized stages and terminal execution bitwise, including proposal tokens, corrected logits and confidence. Cache rings remain unchanged and request/capture/publication guards pass.

The vocabulary projection check exposed a separate oracle precision issue. One near-cancelling output was native `0.026185380294919014`, FP32 reference `0.026164770126342773`, and FP64 reference `0.026170535045821453`. The pair differed by `2.061e-5`, though each was within `2e-5` of the FP64 dot. Projection-only checks now use FP64 accumulation rounded to FP32 and retain the original `2e-5` absolute/relative tolerance. Combined normalization/projection checks retain their existing tolerance and FP32 reference; production GEMM code was unchanged.

Native and daemon builds pass. The adjacent JSON records current source hashes, corrected evidence and fixture/build logs. The earlier attention-normalization and mHC-normalization evidence is superseded by this record; earlier stage/draft composition evidence alone did not prove the correct epsilon. These remain synthetic component and composition tests, not real-checkpoint logits, generation, performance or complete runtime qualification.
