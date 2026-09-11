# Router stream chain

The router now stages hidden inputs and masks on its own CUDA stream and captures the three pinned request downloads with the router graph. Publishing a request requires one completed stream drain instead of a separate download drain. Rows above the 80-row pinned request arena retain the direct download fallback.

The official-weight reuse fixture passes 332 comparisons across rows 1, 6, 16, 80, 81 and 4096, including changed inputs/masks, poisoned input and staging buffers, weight rebinding, graph reuse and invalid-input recovery. Both API smoke suites pass. All eight paired quality cases preserve target and speculative text and usage against the coordinator-owned QP deployment; the inherited Unicode formatting failure and 7/8 cross-mode agreement remain.

One short counting cycle measured 37.73 target tokens/s and 114.82 speculative tokens/s. This is not a matched performance comparison and establishes no speedup. Raw qualification artifacts are in `/tmp/ds41-router-chain`; deployed daemon SHA256 is `3b5bf1dae007c773329368b036b26216def3a9f211553959a0000b2c9fe4fbc7`. Native library and Spark workers are unchanged.
