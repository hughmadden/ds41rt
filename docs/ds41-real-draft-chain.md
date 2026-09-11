# Complete dSpark chain on real target context

The distributed target fixture now drives the complete existing dSpark chain after combined target/engram/dSpark publication. It uses real target-derived FP8 context windows, shared embedding and vocabulary weights, all three RTX transformer stages, Markov correction, confidence and request-owned sampling ranges. Seeds are the target's next greedy tokens. Draft outputs stay private; this fixture does not feed them into target verification or commit them as history.

The real checkpoint passed on RTX GPU 1 with sixteen requests at committed ends 5, 6 and 7. Each attempt produced an anchor plus five in-range draft tokens, finite corrected logits and finite raw confidence. At temperature zero, every proposed token attained its corrected logit's maximum. Graph replay matched eager token, logit and confidence bytes exactly, including after seed and committed-position changes. The three target logit arrays remained byte-identical to the earlier target run.

A separate single-request run used the official chat encoding of “What is 2 + 2? Answer with just the number.” Target generation remained `4` followed by EOS. With `4` as the anchor, the first proposed token was EOS, matching the next target token. The fixture records all five raw proposals, including positions after EOS; those positions are not user-visible generated text and must be truncated by serving logic.

This requalifies the complete chain against the current packed caches and vocabulary head on RTX with real context. Eager/replay equality and greedy self-consistency do not establish independent numerical correctness of the full chain or broad draft quality. Recorded qualification times include host copies, assertions and repeated execution; they are not inference throughput measurements. Target verification, acceptance policy, EOS handling, cancellation and speculative API integration remain open.

[Run logs, token IDs and output hashes](ds41-real-draft-chain.json).
