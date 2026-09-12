# Next-session handoff

The user explicitly closed the goal after completing the RTX draft expert
optimization. This does **not** mean the broader V1 release or original
90 target / 270 dSpark TPS goals were achieved. The remaining contract is in
[TO_SHIP_V1.md](../TO_SHIP_V1.md). Start future work from the user's new scope.

## Repository and running stack

- Repository `/home/tj/Developer/ds41rt`, branch **main**.
- `third_party/sparkinfer` uses **master**, pinned by `sparkinfer.lock.json`.
  Final b12x revision: `7299b3b92e70d539b2c0a63aaadce36932ceef4d`.
- Preserve the user's unrelated dirty `.gitignore` and `run-agent.sh`.
- Target-only API: `ds41-draft-default-live-target-api-dev`, RTX GPU0,
  `http://127.0.0.1:18041`.
- dSpark API: `ds41-draft-default-live-spec-api-dev`, RTX GPU1,
  `http://127.0.0.1:18042`.
- Both use `/tmp/ds41-draft-release-default/daemon`, SHA256
  `cde19ddc4619000b66d8184bd45392b3a83adc914c5a6364dd5fae7a67464756`.
- Both use `/tmp/ds41-draft-release-default/selected-native/libds41rt_native.so`,
  SHA256 `ccd82d862473f71d01622bcf5decd9d5789444210db1be4cc245dbdcb44b8a07`.
- Exact launch arrays: `/tmp/ds41-draft-release-default/live-create-commands.json`.
  `promote.py` includes rollback logic. Previous
  `ds41-draft-slices-live-{target,spec}-api-dev` containers are stopped and retained.
- The four Spark workers remain `ds41-mapped-worker` on ostrich/dodo/emu/kiwi,
  TP ranks 0/1/2/3, port19441, addresses `10.55.0.1` through `.4`.
  Worker daemon `/tmp/ds41-mapped/worker`, native
  `/tmp/ds41-expert-direct/cmake/libds41rt_native.so`; those binaries were not
  changed during the draft work. Reinspect remote state before redeploying.
- Serving configuration: up to16 requests, two alternating lanes, at most8 per
  lane; head48, max context32768, prefill capacity4096/live chunk2048,
  draft request capacity16 / proposal rows80. Both GPUs are identical RTX PRO
  6000 Blackwell96GB. Each serving instance uses one RTX plus all four Sparks.

## Last completed optimization

[Release recipe and budget correction](ds41-draft-release-recipe.md) contains the
final evidence. Local dSpark experts now use the fused N192 slice pipeline,
including BF16-to-FP8 quantization, GPU route planning and ordered FP32 slice
reduction. The native ABI and resident GPU-packed official FP4 weights are
preserved. The ordinary coordinator exporter now selects b12x's recipe without
an explicit CMake width override. Spark defaults still need a separate audit
against their optimized running binaries.

N64 and N192 were compared through the API in sequential N64/N192/N192/N64
order on the same GPU, with a priming and warm sweep per process. They are
within measurement variation. Retain N192: it uses roughly one-third of the
slice scratch and wins larger dispersed component cases. N64 was not promoted.

N64 also exposed a budget bug: the prefill capacity4096 was used to estimate
expert workspaces, although the draft chain uses at most80 proposal rows.
`DsparkWeights::load_serving` now uses the request-derived expert capacity and
separately reserves the complete large-context owner. The 32GiB admission limit
remains. This fixes an overestimate, not a huge live allocation that was freed.

Default-library route and BF16 outputs remain exact against the previously
selected N192 library in changed-input graph tests. API, C2/C6/C16 recovery,
eight paired prompt texts/usages and both post-rollout API checks pass. The
known Unicode objective failure remains in both quality arms. N64 native
memcheck reports zero errors. Earlier N192 native memcheck and full-model lane
qualification also pass.

## Performance context

Latest controlled warm counting (599 output tokens/request) with N192:
approximately **144 C1 TPS, 363 C6 aggregate, 679–680 C16 aggregate**.
The preceding same-process comparison reached146/371/691; measurement protocol
and warming differ. Do not treat either as representative code/prose speed.
Before the expert improvement, the comparable sampler-era warm result was
134/330/634. The component improvement is much larger than the API improvement.

Target-only long-code decode was last about42 TPS. Long-code dSpark was about
124–125 before the latest expert improvement and has not been remeasured.
Warm16k prefill was about6.9–7.2k code TPS and approximately8k repeated-text TPS;
it was not remeasured during the latest decode work.

## Suggested next work if requested

1. Reprofile the current decode path. Local draft experts were previously56%
   of draft GPU time; that profile is now stale. Draft attention had only20CTAs
   at C1, and the vocabulary head was another substantial cost. Target-side
   attention/indexing and launch overhead still need attention.
2. Mixed prefill/decode scheduling: prefill still pauses decode. Slow-client
   isolation and admission during generation remain open. Rebalance only at
   complete token/committed draft-round boundaries.
3. Broader quality and feature audit: known Unicode failure, vision,
   tools/structured constraints, prefix reuse and restart behavior. CED prefill
   is an approximation, and reduced expert returns change numerical results.
   Eight paired outputs do not establish broad quality.
4. Clean release reproduction on all five hosts, including Spark optimized
   backend selection; remove obsolete DS4 code, options and notes. Existing
   build/run infrastructure still contains legacy paths.
5. Rebaseline representative code/prose, long-context, concurrency and prefill
   workloads before claiming the original targets.

## Engram and useful tools

Engram uses mmap with early background gather and **four staging slots** for
two lanes × two layers. io_uring is **not selected**. If selected later, the user
explicitly requires `run.sh` to apply `docker/seccomp-io-uring.json` automatically.
The cache-drop helper is installed and authorized for cold-cache experiments;
never flush caches during API performance measurements. See
`docs/ds41-engram-io-comparison.md` and `docs/ds41-engram-staging-slots.md`.

Official checkpoint:
`/home/tj/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277`.
Use `.ds41rt-cache/reference-venv/bin/python` for Torch/tokenizer tools.
The user authorized official DeepSeek API comparisons using the local
`/home/tj/.local/bin/ds4p-claude` credential; never print or expose its key.

Qualification entry points: `scripts/qualify-ds41-native-api.py`,
`qualify-ds41-concurrent-api.py`, `qualify-ds41-speculative-quality.py`, and
`python/tools/qualify_v41_draft_expert_slices.py`. Detailed commands, raw logs and
frozen candidates are under `/tmp/ds41-draft-native-slices/`,
`/tmp/ds41-draft-native64/` and `/tmp/ds41-draft-release-default/`.
`/tmp` is tmpfs: those binaries disappear on reboot; committed source and
manifests are the durable record. Do not mutate selected artifact directories.
