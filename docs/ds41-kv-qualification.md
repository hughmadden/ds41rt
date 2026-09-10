# Fixed FP8 source KV and paired publication

Serving KV uses one fixed representation: E4M3 values with one E8M0 scale per 32 coordinates, across the full 512-wide vector including its rotary tail. Each row occupies 512 value bytes plus 16 scale bytes. `ds41rt_v41_kv_pack` optionally rotates the last 64 coordinates, rounds them to BF16, then applies the official K32 FP8 activation quantization arithmetic. The scale is the next power of two at or above `max(amax,1e-4)/448`. Values/scales are separate contiguous arrays suitable for paged attention consumers. There are no serving-format profiles.

**Reference distinction:** this matches the reference window-KV quantization policy. The pinned reference compresses source KV with FP4, groups of 16 and E4M3 scales. The user's FP8-only serving policy replaces that compressed-KV quantization with K32 FP8; it is not claimed byte-equivalent to reference compressed FP4. Index keys and learned index queries retain their independent architectural FP4/E8M0 encoding. Full-model logit and generation qualification remains required for the serving policy.

The compressor graph produces index keys from the normalized, unrotated latent, then packs compressed serving KV from that same latent using its first-token frequencies. It does not overwrite the unrotated latent. Proposal values/scales remain private to the wave, including rows that must be discarded because a ratio-two group is incomplete or its suffix is rejected.

## Ownership and commit

`SourceCache` replaces the index-only cache owner. Index and FP8 KV share physical page IDs, one page allocator and one committed-length publication. A page has 256 rows, totaling 152,576 bytes of index and KV payload. `CompressorState::kv_cache` and the companion KV fields in `IndexProposal` expose borrowed values/scales plus the same page table and length as the index view.

Admission reserves the paired payload before allocation. At 64 pages and 16 request slots, source state uses 9,834,624 bytes for ratio two and 9,769,088 bytes for ratio one, including pending half-group state and device metadata. Each proposal wave adds 528 bytes per capacity row: capacity 80 uses 5,534,144 bytes for ratio two and 5,287,744 for ratio one; capacity 4096 uses 72,794,112 and 60,178,432 respectively. Weight allocation is unchanged.

Accepted-prefix commit reserves all required pages before writes. Both index and KV scatter use the same accepted-complete-row destinations; incomplete and rejected rows use the skip sentinel. Both writes and pending half-group copies precede page-table/length upload on one stream. Host lengths/versions advance only after that stream drains. Reservation failure changes neither cache nor metadata. Device write/drain failure revokes the participating leases and clears their published lengths through the existing failure path; it does not promise rollback of unpublished physical bytes. Release/reuse shares the same generation and page-reclamation rules. This is a paired source-cache transaction, not yet the outer transaction across every layer/window in a complete model step.

## Qualification

`scripts/qualify-ds41-kv.py` passes ten packing cases and four scatter cases on each RTX PRO 6000 Blackwell. Packing covers row counts 1/3/16/80/255/4096, every finite BF16 bit pattern, optional rotary, changed input/frequency graph replay, signed zero and output sentinel bytes. Values and scales match the actual pinned reference K32 quantizer exactly. The known TileLang 0.1.8 quantizer vectorization workaround is explicit, as in the learned-query qualification: only `tir.disable_vectorize=True` is changed, preserving source arithmetic.

Scatter checks cover 1/80/4096 source rows, skipped/invalid destinations, changed graph replay and physical capacity 16,777,216 rows. The largest case allocates 8 GiB of values and 256 MiB of scales and writes the final rows, exercising addressing beyond 4 GiB. Small pools are compared in full; the largest case compares four explicitly initialized watched rows. Native null, span, overlap, alignment and shape guards are checked.

The real-weight compressor fixture passes its 55 existing cases per GPU, with **85 paired-cache transactions** and **425 borrowed proposal/scoring checks**. Every transaction compares the full physical KV and index pools against exactly the accepted completed rows. It checks identical KV/index page IDs and device publication pointers/counts. Coverage includes zero acceptance, rejection, stale competing waves, partial ratio-two groups, 16 requests, pool exhaustion with unchanged KV bytes, release/reuse, non-monotonic physical page order, maximum page-table stride and 4096-token prefill/replacement tails.

`scripts/qualify-ds41-source-kv.py` independently validates all 85 dumped real-source proposal sets per GPU. It applies the actual pinned rotary function and K32 FP8 quantizer to the exact BF16 latent/frequency inputs from the owned graph and compares every value/scale byte. This qualifies the selected FP8 serving policy, not the reference compressed-FP4 policy.

The owned selection fixture also passes again on both GPUs with paired storage: 15 source executions, 24 follower cases with two executions each, four 80→1→80 graph-rebinding cases and two full 4096-row prefill executions. Native library, daemon and external owner/selection fixture builds pass. Adjacent JSON records source, build, fixture and qualification hashes. Temporary drivers/data remain outside Git; reusable qualifiers are tracked.

Persistent window KV ownership, attention consumption of these FP8 pages/proposals, dSpark cache storage integration, full-model execution and throughput remain open. These checks establish no full-model readiness or performance claim.
