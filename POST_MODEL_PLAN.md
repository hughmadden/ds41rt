# Full-checkpoint bring-up

Goal 2 begins after Goal 1 qualification, when the checkpoint is complete and all four Sparks are available.

- [ ] Delete the temporary artifacts listed in TO_DELETE_SCAFFOLDING.md and retain reusable tests and qualification tools.
- [ ] Verify the completed official checkpoint revision, shard inventory, tensor representations, tokenizer, and image-processing assets.
- [ ] Build and deploy matching ds41 images and dependency pins to the coordinator and ostrich, dodo, emu, and kiwi.
- [ ] Load the native checkpoint with memory-mapped engram tables and measure resident memory, page faults, and startup time.
- [ ] Validate full-checkpoint text logits and generation against the official reference before enabling speculative execution.
- [ ] Validate native vision with single-image, multiple-image, interleaved text, and multi-turn requests.
- [ ] Measure RTX-resident versus remote dSpark with full-checkpoint acceptance, throughput, memory headroom, and coordinator contention.
- [ ] Validate dSpark proposal distributions, confidence policy, acceptance, rollback, and drafted versus full-generation correctness.
- [ ] Validate exact and bounded CED replay behavior across long prompts, prefix reuse, and multi-turn conversations.
- [ ] Exercise all serving features including streaming, tools, constraints, cancellation, admission, errors, and restart readiness.
- [ ] Qualify concurrency 1 through 16 with mixed prefill, decode, verification, and vision traffic.
- [ ] Measure alternating-wave scheduling against alternatives on the complete four-Spark AFD topology.
- [ ] Tune fusions, batch shapes, expert reduction, engram prefetch, and memory budgets using measured full-model bottlenecks.
- [ ] Run sustained full-stack load and recovery tests and publish reproducible correctness and performance evidence.
- [ ] Finalize release documentation, images, commits, and dependency pins after all checkpoint-dependent gates pass.
