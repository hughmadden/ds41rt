# DS41RT upstream test-coverage inventory

Sources (blobless shallow clones, 2026-09-14 AEST):
- vllm-project/vllm @ HEAD
- sgl-project/sglang @ HEAD
- ggml-org/llama.cpp @ HEAD

Method: 25 parallel census agents read every test file (file-level rows; rolled-up
rows for homogeneous GPU/hardware dirs after representative spot-reads). Each row is
one verdict: PORT-CPU (relevant to ds41rt + CPU-runnable on pg), DEFER-GPU (relevant,
needs GPU/fleet), or SKIP-* (not applicable). Target = ds41rt component: C1 OpenAI API,
C2 constrained output, C3 sampler, C4 tokenizer/chat-template, C5 KV cache/prefix/
persistence, C6 spec decode (dSpark), C7 MoE routing, C8 scheduler/admission, C9
transport/RPC, C10 weights/quant formats, C11 vision, C12 tool-call parsing.

## Verdict summary

| verdict | rows |
|---|---|
| PORT-CPU | 793 |
| DEFER-GPU | 615 |
| SKIP | 496 |
| PORT-CPU as contract patterns | 3 |
| PORT-CPU as API-contract patterns | 2 |
| DEFER-GPU; PORT-CPU the pure-logic subset | 1 |
| DEFER-GPU; PORT-CPU the `test_cpu_*`/`test_zen_cpu_*` + routing reference paths | 1 |
| DEFER-GPU; PORT-CPU small pure-torch norm/activation reference subsets | 1 |
| PORT-CPU** | 1 |
| MIXED: 1 file PORT-CPU, rest DEFER-GPU | 1 |
| data artifacts | 1 |
| MIXED: ctx_shift/slot_save/kv_keep → PORT-CPU | 1 |

Total rows: 1916.


<!-- ===== 01-vllm-v1-kv-connector.md ===== -->

# Inventory 01 — vllm/tests/v1/kv_connector + simple_kv_offload

Census of 83 py files (76 test files; 7 helpers: `__init__.py` ×5, `conftest.py` ×2, `utils.py` ×2, plus `predictable_llama.py`, `nixl_side_channel_probe.py`, `toy_proxy_server.py`). `nixl_push_integration/` contains only `config_sweep_accuracy_test.sh` (no py tests). Non-test helpers excluded from rows below.

Verdicts: PORT-CPU / DEFER-GPU / SKIP-TRAINER / SKIP-OTHERMODEL / SKIP-INFRA / SKIP-DUP.
Targets: C1 OpenAI API · C2 constrained output · C3 sampler · C4 tokenizer/chat · C5 KV cache/prefix/radix/disk offload · C6 spec decode · C7 MoE routing · C8 scheduler/admission · C9 transport/RPC · C10 weights/quant · C11 vision · C12 tool-call.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py | 18 | native offloading connector | yes | canonical page mapping across TP/DCP/MLA layouts | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_config.py | 22 | native offloading connector | yes | KV-cache-spec → offload config translation (mamba/hybrid/MLA) | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_events.py | 19 | native offloading connector | yes | BlockStored/BlockRemoved event tracking per group | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_metrics.py | 22 | native offloading connector | yes | offload prometheus metric defs, aggregation, stats | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py | 102 | native offloading connector | yes | scheduler-side store/load policy: grouping, mamba, events | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py | 4 | native offloading connector | yes | worker metadata aggregation across jobs/workers | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_worker.py | 8 | native offloading connector | mixed | worker offload ops, load/store spec selection per backend | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py | 28 | NixlConnector P/D | yes | P pulls KV from D; remote_block_ids lifecycle, partial coverage | PORT-CPU | C5, C9 |
| vllm/tests/v1/kv_connector/unit/test_cache_pollution_prevention.py | 1 | scheduler + prefix cache | yes | failed sync-load evicts invalid blocks from prefix cache | PORT-CPU | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_config.py | 7 | KV transfer config | yes | KVTransferConfig → connector factory translation | PORT-CPU | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_decode_bench_connector.py | 15 | decode bench connector | yes | dummy KV fill connector for decode benchmarking | SKIP-INFRA | - |
| vllm/tests/v1/kv_connector/unit/test_error_propagation.py | 2 | scheduler + connector | yes | connector load failure propagates to request abort | PORT-CPU | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_example_connector.py | 1 | example connector | no | multimodal example connector E2E with vision model | DEFER-GPU | C5, C11 |
| vllm/tests/v1/kv_connector/unit/test_flexkv_connector.py | 16 | FlexKV connector | yes | delegation + ImportError when flexkv missing (mocked) | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/test_handshake_pp_aggregation.py | 3 | connector handshake | yes | handshake metadata aggregation across PP ranks | PORT-CPU | C9 |
| vllm/tests/v1/kv_connector/unit/test_hf3fs_client.py | 6 | HF3FS connector | yes | 3FS client ctor-failure cleanup, idempotent close (mocked) | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_hf3fs_connector.py | 12 | HF3FS connector | yes | mock-client I/O correctness, stats serialization | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_hf3fs_metadata_server.py | 13 | HF3FS connector | yes | metadata server page allocation, cache-hit semantics | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_hidden_states_connector.py | 10 | hidden-states connector | yes | hidden-state KV-cache-group resolution (hybrid specs) | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/test_hisparse_connector.py | 4 | HiSparse connector | yes | HiSparse scheduler/worker wiring with example connector | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_hisparse_stats.py | 5 | HiSparse connector | yes | hot-buffer hit/miss stats record/reduce | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_hma_auto_config.py | 2 | HMA + connectors | yes | HMA auto-disable when KV transfer connector configured | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py | 3 | scheduler + prefix cache | yes | invalid-block recompute/free semantics, no caching after fail | PORT-CPU | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py | 1 | kv transfer group | yes | init/shutdown lifecycle of KVTransferState group | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py | 9 | scheduler + connector | yes | recovery paths when async KV load fails (recompute/resume) | PORT-CPU | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_lmcache_connector.py | 23 | LMCache connector | yes | LMCache engine event → BlockStored translation (mocked) | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_lmcache_integration.py | 8 | LMCache integration | mixed | vLLM-interface assumption stability for LMCache adapter | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_connector.py | 21 | Mooncake connector | mixed | Mooncake scheduler/worker transfer state machine (zmq) | SKIP-OTHERMODEL | C5, C9 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_connector_hma.py | 8 | Mooncake connector | yes | HMA sliding-window clip, multi-group send trimming | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py | 9 | Mooncake connector | yes | GDN/MambaSpec hybrid transfer path | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_stats.py | 15 | Mooncake connector | yes | transfer stats record/reduce/threading | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_store_connector.py | 37 | Mooncake store (shared storage) | mixed | store connector lookup/store lifecycle, event pub | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py | 36 | Mooncake store (shared storage) | yes | external block pool, chunk hashes, coordinated allocation | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py | 9 | Mooncake store (shared storage) | yes | save→lookup E2E on hybrid SWA+Full config, mock store | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_store_layout.py | 5 | Mooncake store (shared storage) | yes | LBHNC/LBNHC/rank-local payload layouts | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py | 42 | Mooncake store (shared storage) | yes | request tracker, load spec build, metadata flow | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_mooncake_store_worker.py | 144 | Mooncake store (shared storage) | mixed | store worker I/O, transfer completion, eviction, TTL | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_moriio_connector.py | 17 | MoRIIO connector | mixed | MoRIIO connector scheduler/worker config + zmq wiring | SKIP-OTHERMODEL | C5, C9 |
| vllm/tests/v1/kv_connector/unit/test_moriio_kv_layout.py | 17 | MoRIIO connector | mixed | MoRIIO KV layout, block mapping across TP | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_moriio_proxy_routing.py | 5 | MoRIIO proxy | yes | flat_interleaved_dp_route fairness over (instance, dp) slots | SKIP-OTHERMODEL | C8 |
| vllm/tests/v1/kv_connector/unit/test_moriio_routing_fairness.py | 6 | MoRIIO connector | yes | decode-side read-source routing across P/D TP configs | SKIP-OTHERMODEL | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_moriio_routing.py | 3 | MoRIIO connector | yes | router-authoritative DP-rank routing via request_finished | SKIP-OTHERMODEL | C5, C9 |
| vllm/tests/v1/kv_connector/unit/test_moriio_toy_proxy_server.py | 1 | MoRIIO proxy | no | toy proxy server runs (ROCm-only, quart) | SKIP-OTHERMODEL | - |
| vllm/tests/v1/kv_connector/unit/test_moriio_tp_ack.py | 17 | MoRIIO connector | yes | heterogeneous-TP ack counting, head validation, ack resolve | SKIP-OTHERMODEL | C5, C9 |
| vllm/tests/v1/kv_connector/unit/test_moriio_unmap.py | 4 | MoRIIO connector | yes | request-id map/unmap with transfer_id fallback | SKIP-OTHERMODEL | C8 |
| vllm/tests/v1/kv_connector/unit/test_multi_connector.py | 23 | MultiConnector | mixed | N-connector delegation: first-wins load, store-to-all, E2E LLM | DEFER-GPU | C5 |
| vllm/tests/v1/kv_connector/unit/test_nixl_connector.py | 53 | NixlConnector | mixed | NIXL agent lifecycle, metadata serde, send/recv state machine | SKIP-OTHERMODEL | C5, C9 |
| vllm/tests/v1/kv_connector/unit/test_nixl_connector_hma.py | 39 | NixlConnector HMA | yes | NIXL HMA scheduler: sliding-window, mamba N-1 prefill | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_nixl_desc_geometry.py | 14 | NixlConnector | yes | transfer byte-range invariants under P/D block geometry | PORT-CPU | C9 |
| vllm/tests/v1/kv_connector/unit/test_nixl_heartbeat.py | 7 | NixlConnector | yes | scheduler-driven heartbeat / KV lease renewal semantics | PORT-CPU | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_nixl_push_connector.py | 51 | NixlPushConnector | yes | push design: staged registrations, eviction enqueue, notify | SKIP-OTHERMODEL | C5 |
| vllm/tests/v1/kv_connector/unit/test_nixl_rocm_gpu_mem_diag.py | 2 | NixlConnector | no | GPU memory released after NIXL shutdown on ROCm | SKIP-INFRA | - |
| vllm/tests/v1/kv_connector/unit/test_nixl_simple_cpu_offload.py | 3 | MultiConnector | yes | Nixl + SimpleCPUOffload delegation, HMA detection, metadata agg | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/test_offloading_connector.py | 5 | native offloading connector | no | E2E LLM offload with zmq events, block-size multiple constraint | DEFER-GPU | C5 |
| vllm/tests/v1/kv_connector/unit/test_output_aggregator.py | 2 | connector output agg | yes | merge finished_sending/recving/invalid/failed across outputs | PORT-CPU | C5, C9 |
| vllm/tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py | 4 | scheduler + connector | yes | remote-decode request lifecycle via example connector | PORT-CPU | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py | 9 | scheduler + connector | yes | remote-prefill request lifecycle, finish/retry paths | PORT-CPU | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_scheduler_kv_connector_override.py | 3 | scheduler + connector | yes | plugin/factory override of scheduler connector instance | PORT-CPU | C5, C8 |
| vllm/tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py | 2 | SimpleCPUOffload | yes | connector-level wrapper behavior over offload scheduler | PORT-CPU | C5 |
| vllm/tests/v1/kv_connector/unit/test_tp_mapping.py | 10 | NixlConnector | yes | TP mapping: source ranks, split handles, desc IDs, no GPU | PORT-CPU | C5, C9 |
| vllm/tests/v1/kv_connector/unit/test_transfer_topology_sharded.py | 4 | transfer topology | yes | sharded TransferTopology registration of engine info | PORT-CPU | C9 |
| vllm/tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py | 3 | hidden-states extraction | no | multi-GPU LLM hidden-states extraction to file | DEFER-GPU | C5 |
| vllm/tests/v1/kv_connector/mooncake_integration/test_accuracy.py | 1 | Mooncake PD | no | gsm8k accuracy vs live Mooncake P/D server (lm_eval) | SKIP-INFRA | C5 |
| vllm/tests/v1/kv_connector/mooncake_integration/test_mooncake_imports.py | 1 | Mooncake wheel canary | yes | mooncake-transfer-engine CUDA wheel import canary | SKIP-INFRA | - |
| vllm/tests/v1/kv_connector/nixl_integration/test_accuracy.py | 2 | NIXL PD | no | gsm8k accuracy vs live NIXL P/D server | SKIP-INFRA | C5 |
| vllm/tests/v1/kv_connector/nixl_integration/test_disagg_accuracy.py | 0 | NIXL PD | no | disagg prefill/decode accuracy harness (script-style) | SKIP-INFRA | C5 |
| vllm/tests/v1/kv_connector/nixl_integration/test_edge_cases.py | 1 | NIXL PD | no | long/short prompt edge cases through P/D proxy | SKIP-INFRA | C5 |
| vllm/tests/v1/kv_connector/nixl_integration/test_mamba_prefix_cache.py | 1 | NIXL PD mamba | no | D-side prefix cache hits reduce transfer bytes | SKIP-INFRA | C5 |
| vllm/tests/v1/kv_connector/nixl_integration/test_multi_connector_edge_cases.py | 9 | NIXL + offload | no | block-boundary correctness + decode metrics, cold/warm/partial | SKIP-INFRA | C5 |
| vllm/tests/v1/kv_connector/nixl_integration/test_nixl_imports.py | 1 | NIXL wheel canary | mixed | NIXL/UCX import + CUDA wheel selection canary | SKIP-INFRA | - |
| vllm/tests/v1/kv_connector/nixl_integration/test_spec_decode_acceptance.py | 1 | NIXL PD + spec decode | no | EAGLE3/MTP acceptance-length parity vs standalone baseline | SKIP-INFRA | C5, C6 |
| vllm/tests/v1/simple_kv_offload/test_hip_mem_ops.py | 5 | simple KV offload mem ops | yes | HIP runtime version gating, memcpy attrs (monkeypatched) | PORT-CPU | C5 |
| vllm/tests/v1/simple_kv_offload/test_integration.py | 4 | SimpleCPUOffload E2E | no | real-model E2E offload correctness (CUDA/ROCm-gated) | DEFER-GPU | C5 |
| vllm/tests/v1/simple_kv_offload/test_kv_events.py | 13 | SimpleCPUOffload | yes | BlockStored/Removed medium + per-group metadata emission | PORT-CPU | C5 |
| vllm/tests/v1/simple_kv_offload/test_scheduler.py | 41 | SimpleCPUOffload | yes | CPU offload scheduler: lookup, alloc, free, pin accounting | PORT-CPU | C5 |
| vllm/tests/v1/simple_kv_offload/test_worker.py | 12 | SimpleCPUOffload | mixed | GPU→CPU store cross-stream sync ordering (no stale reads) | PORT-CPU | C5 |

## Notable PORT-CPU candidates

- **unit/offloading_connector/test_scheduler.py (n=102)** — the largest pure-policy suite in the batch: block selection, grouping, mamba/hybrid handling, event emission for the native disk-offload connector; near-1:1 analogue of ds41rt hostcache store/load scheduling (C5).
- **unit/offloading_connector/test_canonical_mapping.py (n=18)** — canonical page mapping across TP/DCP/MLA layouts; exactly the layout-derivation logic a disk KV persistence layer must get right (C5, FP4 KV layout adjacent).
- **simple_kv_offload/test_scheduler.py (n=41)** — CPU offload scheduler lookup/alloc/free/pin accounting; the cleanest small reference for hostcache radix/lookup policy (C5).
- **simple_kv_offload/test_kv_events.py (n=13)** — pins BlockStored/BlockRemoved medium semantics (MEDIUM_CPU vs MEDIUM_STORAGE) and per-group metadata; directly transferable to hostcache eventing (C5).
- **unit/offloading_connector/test_config.py (n=22)** — KV-cache-spec → offload config translation incl. Mamba/MLA/SlidingWindow; port to ds41rt hostcache config validation (C5).
- **unit/test_nixl_heartbeat.py (n=7)** — lease/heartbeat renewal semantics; ds41rt disk-persistence leases have the same expiry/renewal invariants (C5/C8).
- **unit/test_cache_pollution_prevention.py + test_invalid_blocks_correctness.py + test_kv_load_failure_recovery.py (n=13 combined)** — prefix-cache integrity under failed/partial KV loads: eviction of invalid blocks, no caching after failure, recovery paths; core correctness for any reuse+disk tier (C5/C8).
- **unit/test_remote_prefill_lifecycle.py + test_remote_decode_lifecycle.py (n=13 combined)** — request lifecycle with remote KV participation (finish/retry/resume); scheduler-level invariants portable to ds41rt daemon admission (C5/C8).
- **unit/test_tp_mapping.py + test_transfer_topology_sharded.py + test_output_aggregator.py (n=16 combined)** — pure-CPU transfer planning and output-aggregation invariants; reusable for ds41rt transport protocol_v2 planning (C9).
- **unit/offloading_connector/test_metrics.py (n=22)** — offload metric definitions and cross-worker aggregation; template for hostcache observability (C5).


<!-- ===== 02-vllm-v1-kv-offload-ec.md ===== -->

# Inventory 02 — vllm/tests/v1/kv_offload + ec_connector

Census of test files, DS41RT component map: C1 API surface; C2 structured output; C3 sampler; C4 tokenizer/chat templates; C5 KV cache/prefix reuse/radix/block mgmt/FP4 KV/disk offload (ds41rt-hostcache + v41_kv); C6 speculative; C7 MoE; C8 scheduler/admission; C9 transport/RPC + failure injection; C10 weights/quant; C11 vision; C12 tool-call.

Note on counts: brief said 26 + 24 py files; actual tree = kv_offload 24 py (19 test files + 4 `__init__.py` + `tiering/p2p/p2p_connector_proxy.py` helper) and ec_connector 24 py (22 test files + `__init__.py` + `unit/utils.py` helper). Rows below cover all 41 test files. `n` = `grep -c "def test_"`.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| kv_offload/test_factory.py | 29 | offload spec registry/factory (CPU+Tiering spec) | yes | pure config/registry logic, mocked ctors | PORT-CPU | C5 |
| kv_offload/test_file_mapper.py | 14 | disk file naming/namespace for offloaded blocks | yes | pure path/namespace logic, no I/O | PORT-CPU | C5 |
| kv_offload/cpu/policies/test_factory.py | 8 | cache-policy factory (LRU/ARC registration) | yes | registry resolution only | PORT-CPU | C5 |
| kv_offload/cpu/test_canonical_layout.py | 6 | canonical NHD page layout / copy-plan | mixed | 4 pure-numpy CPU tests; 2 CUDA skipif | PORT-CPU | C5 |
| kv_offload/cpu/test_gpu_worker.py | 8 | GPU<->CPU offload worker, shutdown lifecycle | mixed | 5 pure-mock lifecycle tests; 3 need CUDA | PORT-CPU | C5 |
| kv_offload/cpu/test_manager.py | 28 | CPUOffloadingManager: LRU/ARC eviction, admission | yes | zero GPU touch; pure policy state machine | PORT-CPU | C5 |
| kv_offload/cpu/test_shared_offload_region.py | 43 | mmap SharedOffloadRegion (/dev/shm) | yes | Linux /dev/shm; threads+multiproc, no GPU | PORT-CPU | C5 |
| kv_offload/cpu/test_swap_blocks_batch.py | 2 | cuMemcpyBatchAsync C++ op | no | CUDA-only kernel op | DEFER-GPU | C5 |
| kv_offload/cpu/test_swap_blocks_triton.py | 1 | Triton swap_blocks_batch kernel | no | CUDA-only Triton kernel | DEFER-GPU | C5 |
| kv_offload/tiering/test_async_lookup.py | 17 | AsyncLookupManager state machine | yes | in-memory backend, threads only | PORT-CPU | C5 |
| kv_offload/tiering/test_factory.py | 11 | SecondaryTierFactory registration/loading | yes | MagicMock args only | PORT-CPU | C5 |
| kv_offload/tiering/test_fs_tier.py | 29 | FileSystemTierManager: real disk I/O tier | yes | real disk I/O; C-ext/O_DIRECT optional fallback | PORT-CPU | C5 |
| kv_offload/tiering/test_kvcr_tier.py | 12 | KVCR (NVIDIA KV-cache-router) tier adapter | yes | importorskip kvcr bindings; router-specific | SKIP-OTHERMODEL | - |
| kv_offload/tiering/test_metrics.py | 6 | TieringMetricsTracker gauges/histograms | yes | pure tracker logic | PORT-CPU | C5 |
| kv_offload/tiering/test_obj_tier.py | 35 | ObjectStore (S3/NIXL OBJ) tier, mocked agent | yes | nixl module import; agent fully mocked | PORT-CPU | C5 |
| kv_offload/tiering/test_tiering_offloading.py | 42 | TieringOffloadingManager cascade/promotion | yes | mock mmap region, CPU tensors | PORT-CPU | C5 |
| kv_offload/tiering/p2p/test_data_transport.py | 25 | NixlTransport base/fingerprint/poll/cancel | yes | nixl agent mocked or patched to None | PORT-CPU | C5, C9 |
| kv_offload/tiering/p2p/test_manager.py | 65 | P2PSecondaryTierManager session/job routing | yes | fake transports/sessions, no real net | PORT-CPU | C5, C9 |
| kv_offload/tiering/p2p/test_sessions.py | 101 | P2PSession bidirectional xfer protocol | yes | fake control connections only | PORT-CPU | C5, C9 |
| kv_offload/tiering/p2p/test_zmq_transport.py | 16 | ZmqTransport bind/poll loopback | yes | needs pyzmq; 127.0.0.1 loopback | PORT-CPU | C9 |
| ec_connector/integration/test_epd_correctness.py | 0 | EPD disagg correctness vs baseline (script) | no | argparse main(); live services, Qwen2.5-VL | DEFER-GPU | C11, C9 |
| ec_connector/integration/test_nixl_failure.py | 1 | NIXL failure injection, encoder-cache xfer | no | 2 CUDA GPUs + nixl pkg | DEFER-GPU | C9 |
| ec_connector/unit/cpu/scheduler/test_embedding_cache.py | 34 | EmbeddingCache block bookkeeping | yes | pure ints, no mmap/torch | PORT-CPU | C11 |
| ec_connector/unit/cpu/scheduler/test_scheduler.py | 21 | ECCPUScheduler save/load scheduling | yes | real ECSharedRegion mmap, CPU tensors | PORT-CPU | C11, C8 |
| ec_connector/unit/cpu/test_connector.py | 5 | ECCPUConnector role wiring | yes | monkeypatched make_scheduler/worker | PORT-CPU | C11 |
| ec_connector/unit/cpu/test_ec_shared_region.py | 13 | ECSharedRegion mmap substrate | yes | Linux mmap; /dev/shm residency checks | PORT-CPU | C11 |
| ec_connector/unit/cpu/worker/test_worker.py | 22 | ECCPUWorker byte-level save/load | mixed | 6 lifecycle CPU; 16 need CUDA/XPU accel | DEFER-GPU | C11 |
| ec_connector/unit/test_control.py | 20 | ControlConnection ABC + ZmqClientConnection | yes | mocked dealer sockets | PORT-CPU | C9 |
| ec_connector/unit/test_data.py | 14 | DataTransport ABC + NixlDataTransport | yes | NixlWrapper fully mocked | PORT-CPU | C9 |
| ec_connector/unit/test_ec_cpu_connector.py | 1 | E2E EC CPU cache accuracy+latency | no | needs GPU + Qwen2-VL-2B download | DEFER-GPU | C11 |
| ec_connector/unit/test_ec_example_connector.py | 27 | ECExampleConnector producer/consumer | mixed | bulk CPU mocks; 1 test cuda-gated | PORT-CPU | C11 |
| ec_connector/unit/test_ec_output_aggregator.py | 4 | ECOutputAggregator rank folding | yes | cpu_test mark, pure aggregation | PORT-CPU | C11, C8 |
| ec_connector/unit/test_ec_transfer_params.py | 5 | ec_transfer_params API plumbing | yes | importorskip flash-attn-built tests.v1.core.utils | PORT-CPU | C1 |
| ec_connector/unit/test_epd_proxy_retry.py | 5 | EPD proxy decode-retry body handling | yes | loads real examples/ proxy; aiohttp+httpx | PORT-CPU | C9, C11 |
| ec_connector/unit/test_epd_proxy_round_robin.py | 8 | EPD proxy encoder fan-out round-robin | yes | loads real examples/ proxy | PORT-CPU | C9, C11 |
| ec_connector/unit/test_metadata.py | 7 | ECCPUConnectorMetadata shapes | yes | trivial constructors | PORT-CPU | C11 |
| ec_connector/unit/test_protocol.py | 3 | XferReq/XferAck msgspec roundtrip | yes | msgspec only | PORT-CPU | C9 |
| ec_connector/unit/test_scheduler_nixl_consumer.py | 12 | NIXL-path consumer scheduling (gate-off) | yes | CPU tensors; monkeypatched nixl fields | PORT-CPU | C11, C9 |
| ec_connector/unit/test_scheduler_nixl_ctor.py | 5 | NIXL enable/config gating | yes | skipif NixlWrapper absent | PORT-CPU | C11, C9 |
| ec_connector/unit/test_scheduler_nixl_producer.py | 4 | NIXL-path producer scheduling | yes | CPU tensors; monkeypatched nixl fields | PORT-CPU | C11, C9 |
| ec_connector/unit/test_session.py | 36 | Producer/Consumer Session/Xfer state machines | yes | MagicMock data transport | PORT-CPU | C9 |
| ec_connector/unit/test_utils.py | 4 | mem-descriptor serialize/deserialize (msgspec) | yes | msgspec only | PORT-CPU | C9 |
| ec_connector/unit/test_worker_ec_connector.py | 4 | V2 GPU runner EC connector wrapper | yes | cpu_test mark, patched get_ec_transfer | PORT-CPU | C11 |

## Notable PORT-CPU candidates

- `kv_offload/cpu/test_manager.py` (28) — pure eviction/admission policy (LRU+ARC, store_threshold, ref-counting); direct analog for ds41rt hostcache admission logic, zero GPU touch.
- `kv_offload/cpu/test_shared_offload_region.py` (43) — mmap shared-region lifecycle: creator/joiner races, barrier+unlink leak immunity, madvise population fallback; hostcache substrate patterns (page-alignment, populate, cleanup).
- `kv_offload/tiering/test_fs_tier.py` (29) — real-disk KV tier round-trip, corrupt-file verdict correction, batch partial-failure; the closest upstream model for disk KV persistence semantics.
- `kv_offload/tiering/test_tiering_offloading.py` (42) — primary/secondary cascade + promotion orchestration incl. in-flight-write deferral and failed-store finalization; exactly the "stores but never serves" class of bug ds41rt is hunting.
- `kv_offload/tiering/test_async_lookup.py` (17) — async tier-lookup state machine incl. mark_miss livelock regression (#49176); portable lookup-contract spec for hostcache.
- `kv_offload/test_file_mapper.py` (14) — storage namespace identity (rank/parallel-agnostic/replicated/canonical-layout); reusable for ds41rt on-disk layout naming.
- `kv_offload/tiering/p2p/test_sessions.py` (101) + `test_manager.py` (65) — fully-mocked P2P transfer session/job state machines; C9 failure/lifecycle logic without a NIC.
- `kv_offload/cpu/policies/test_factory.py` (8) + `kv_offload/tiering/test_factory.py` (11) — policy/tier plugin registration + out-of-tree module loading; cheap CI sentinels for hostcache plugin registry.
- `kv_offload/cpu/test_canonical_layout.py` (4 of 6 CPU) — canonical NHD copy-plan construction (numpy); pure layout/serialization logic reusable for FP4 KV canonical layout.
- `ec_connector/unit/test_session.py` (36) + `test_control.py` (20) + `test_data.py` (14) — fully-mocked producer/consumer transfer protocol over abstract control/data transports; template for v41_kv connector unit tests with failure injection.


<!-- ===== 03-vllm-v1-engine-e2e.md ===== -->

# vLLM tests/v1 inventory — e2e, engine, metrics, tracing, fault_tolerance (slice 03)

Census of 80 py files. `n` = `grep -c "def test_"`. Verdicts: PORT-CPU / DEFER-GPU / SKIP-TRAINER / SKIP-OTHERMODEL / SKIP-INFRA / SKIP-DUP. Component map: C1 API surface; C2 structured output; C3 sampler; C4 tokenizer; C5 KV cache/prefix; C6 speculative decoding; C7 MoE; C8 scheduler/admission/lifecycle; C9 transport/RPC; C10 weights/quant; C11 vision; C12 tool-call parsing.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| tests/v1/e2e/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/general/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/general/test_async_scheduling.py | 3 | async scheduling | no | VllmRunner; large-GPU 16GB | DEFER-GPU | C8 |
| tests/v1/e2e/general/test_attention_backend_per_kind.py | 2 | attention backend per KV group | no | gemma-3-1b; CUDA-only | DEFER-GPU | C5 |
| tests/v1/e2e/general/test_cascade_attention.py | 1 | cascade attention | no | new process per test; LLM | DEFER-GPU | C5 |
| tests/v1/e2e/general/test_context_length.py | 2 | context-length validation | no | VllmRunner; ValueError semantics | DEFER-GPU | C8 |
| tests/v1/e2e/general/test_correctness_sliding_window.py | 2 | sliding-window correctness | no | vllm_runner | DEFER-GPU | C5 |
| tests/v1/e2e/general/test_hisparse.py | 1 | HiSparse KV transfer | no | forked procs; CUDA | DEFER-GPU | C5, C9 |
| tests/v1/e2e/general/test_kv_sharing_fast_prefill.py | 2 | KV sharing fast prefill | no | LLM; forked; seed 42 | DEFER-GPU | C5 |
| tests/v1/e2e/general/test_mamba_prefix_cache.py | 4 | mamba prefix cache | no | Qwen3-Next-80B; datasets | DEFER-GPU | C5 |
| tests/v1/e2e/general/test_min_tokens.py | 7 | min_tokens / stop handling | no | opt-125m small | DEFER-GPU | C3, C8 |
| tests/v1/e2e/general/test_pooling_chunked_prefill.py | 2 | pooling chunked prefill | no | vllm_runner; CUDA-only | DEFER-GPU | C5, C8 |
| tests/v1/e2e/general/test_rocm_aiter_custom_ar.py | 1 | ROCm aiter custom allreduce | no | ROCm multi-GPU only | SKIP-OTHERMODEL | C9 |
| tests/v1/e2e/general/test_sharded_sampling.py | 1 | sharded vs replicated sampling | no | multi-GPU; tolerance-based | DEFER-GPU | C3 |
| tests/v1/e2e/general/test_streaming_input.py | 14 | AsyncLLM streaming input | no | opt-125m | DEFER-GPU | C1, C8 |
| tests/v1/e2e/spec_decode/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/conftest.py | 0 | support | - | fixtures: sampling config, model | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/utils.py | 0 | support | - | shared spec-decode helpers | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/acceptance_rates/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/acceptance_rates/utils.py | 0 | support | - | gsm8k acceptance helpers | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/acceptance_rates/dflash/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py | 2 | dflash acceptance rate | no | gsm8k eval; single-GPU | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/acceptance_rates/dspark/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/acceptance_rates/dspark/test_dspark.py | 1 | dspark acceptance rate | no | gsm8k eval; CUDA | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/acceptance_rates/mtp_other/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/acceptance_rates/mtp_other/test_medusa.py | 1 | medusa acceptance rate | no | vicuna-7b medusa ckpt | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/acceptance_rates/mtp_other/test_mtp.py | 1 | gemma MTP acceptance lengths | no | vllm_runner; mrv2 param | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/acceptance_rates/mtp_other/test_synthetic.py | 1 | synthetic rejection sampling | no | single-GPU; acceptance 1.875 | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/draft_model/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/draft_model/test_async.py | 1 | sync-free spec decode async | no | eagle3 llama 1B | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/draft_model/test_draft_model.py | 10 | draft-model spec decode | no | Qwen3-0.6B; 1-2 GPU | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/draft_model/test_lora.py | 1 | LoRA + spec decode batch | no | vllm_runner | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/eagle/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/eagle/utils.py | 0 | support | - | eagle correctness helpers | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/eagle/test_eagle3_pp.py | 1 | eagle3 pipeline parallel | no | multi-GPU(2); llama 1B | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/eagle/test_eagle_correctness.py | 2 | eagle correctness | no | single-GPU; gsm8k | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/mtp/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/mtp/test_mtp.py | 2 | MTP spec decode | no | single-GPU; gsm8k | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/ngram_suffix/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/ngram_suffix/test_ngram_suffix.py | 3 | ngram suffix spec decode | no | single-GPU | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/speculators/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/e2e/spec_decode/speculators/test_speculators.py | 1 | speculator draft models | no | single-GPU | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/test_mtp_parallel_load.py | 2 | MTP parallel weight load | no | multi-GPU marks | DEFER-GPU | C6 |
| tests/v1/e2e/spec_decode/test_sharded_sampling.py | 1 | sharded sampling under spec | no | LLM; multi-GPU | DEFER-GPU | C3, C6 |
| tests/v1/e2e/test_cpu_linear_attn_chunked_prefix.py | 2 | CPU linear-attn chunked prefix | yes | cpu_model mark; KV space 1 | PORT-CPU | C5 |
| tests/v1/e2e/test_cpu_spec_decode.py | 2 | CPU spec decode e2e | yes | cpu_model; Triton gate | PORT-CPU | C6 |
| tests/v1/e2e/test_hybrid_chunked_prefill.py | 2 | hybrid chunked prefill | no | Qwen3.5-4B; 30-80GB | DEFER-GPU | C5, C8 |
| tests/v1/e2e/test_replayssm_decode.py | 5 | ReplaySSM decode parity | no | Nemotron-3 4B; 40GB | DEFER-GPU | C5 |
| tests/v1/engine/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/engine/conftest.py | 0 | support | - | dummy vectors, tokenizer fixtures | SKIP-INFRA | - |
| tests/v1/engine/utils.py | 0 | support | - | dummy output-processor vectors | SKIP-INFRA | - |
| tests/v1/engine/test_abort_final_step.py | 1 | abort at final step | no | CUDA-only module skip | DEFER-GPU | C8 |
| tests/v1/engine/test_admission_control.py | 31 | admission control unit tests | yes | pure mocks; 503 mapping | PORT-CPU | C8, C1 |
| tests/v1/engine/test_async_llm.py | 18 | AsyncLLM lifecycle | no | CUDA-only module skip | DEFER-GPU | C1, C8 |
| tests/v1/engine/test_core_engine_actor_manager.py | 3 | core engine actor manager | yes | mocked sockets/paths | PORT-CPU | C8, C9 |
| tests/v1/engine/test_dp_placement_node_allowlist.py | 5 | DP placement allowlist | yes | env-var unit tests | PORT-CPU | C8 |
| tests/v1/engine/test_engine_args.py | 4 | engine arg validation | yes | CLI parsing unit tests | PORT-CPU | C8 |
| tests/v1/engine/test_engine_core_client.py | 20 | core client + DPLB routing | mixed | DPLB pure-mock; fork procs | PORT-CPU | C9, C8 |
| tests/v1/engine/test_engine_core.py | 10 | EngineCore lifecycle | mixed | in-proc dummy cores; TP 2-GPU | DEFER-GPU | C8 |
| tests/v1/engine/test_fast_incdec_prefix_err.py | 1 | detok invalid-UTF8 edge | yes | AutoTokenizer only | PORT-CPU | C4 |
| tests/v1/engine/test_init_error_messaging.py | 2 | init error messaging | yes | mocked config | PORT-CPU | C5, C8 |
| tests/v1/engine/test_input_processor_trace_replay.py | 5 | input processor trace replay | yes | mocked VllmConfig | PORT-CPU | C5, C8 |
| tests/v1/engine/test_iteration_logging.py | 4 | iteration logging | yes | FakeEngineCore | PORT-CPU | C8 |
| tests/v1/engine/test_llm_engine_finalizer_is_weak.py | 3 | finalizer weakref | yes | gc only | PORT-CPU | C8 |
| tests/v1/engine/test_llm_engine.py | 4 | LLM engine metrics/parallel | no | opt-125m runner boot | DEFER-GPU | C8, C3 |
| tests/v1/engine/test_logprobs_processor.py | 2 | logprobs truncation invariant | yes | numpy unit | PORT-CPU | C3, C4 |
| tests/v1/engine/test_output_processor.py | 11 | output processor/detok/stop | yes | dummy test vectors | PORT-CPU | C4, C8 |
| tests/v1/engine/test_parallel_sampling.py | 3 | parallel sampling parent req | yes | pure unit | PORT-CPU | C3 |
| tests/v1/engine/test_preprocess_error_handling.py | 1 | preprocess error handling | no | engine boot; fork | DEFER-GPU | C8 |
| tests/v1/engine/test_startup_watch_processes.py | 3 | startup watch processes | yes | mocked zmq | PORT-CPU | C8, C9 |
| tests/v1/metrics/test_engine_logger_apis.py | 2 | engine logger APIs | no | distilgpt2 engine boot | DEFER-GPU | C8 |
| tests/v1/metrics/test_histogram_buckets.py | 6 | histogram bucket snapshots | yes | hard-coded literals | PORT-CPU | C8 |
| tests/v1/metrics/test_metrics_reader.py | 5 | metrics reader | yes | cpu_test mark | PORT-CPU | C8 |
| tests/v1/metrics/test_perf_metrics.py | 51 | flops estimators | yes | HF configs only | PORT-CPU | C10 |
| tests/v1/metrics/test_ray_metrics.py | 10 | ray prometheus wrappers | no | ray.init; num_gpus=1 | DEFER-GPU | C9 |
| tests/v1/metrics/test_stats.py | 12 | iteration stats accounting | yes | pure unit | PORT-CPU | C5, C8 |
| tests/v1/tracing/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/tracing/test_tracing.py | 1 | OTel tracing spans | no | opt-125m engine boot | DEFER-GPU | C1 |
| tests/v1/fault_tolerance/__init__.py | 0 | support | - | package marker | SKIP-INFRA | - |
| tests/v1/fault_tolerance/test_fault_tolerance_e2e.py | 2 | elastic fault tolerance | no | nixl_ep FT hardware gate | DEFER-GPU | C8, C9 |

Totals: 80 files — 15 PORT-CPU, 27 DEFER-GPU, 1 SKIP-OTHERMODEL, 15 SKIP-INFRA support files. ~228 test functions (68 in the PORT-CPU set).

## Notable PORT-CPU candidates

- `tests/v1/engine/test_admission_control.py` (31 tests) — admission control, queue overflow, GracefulHTTPError→503 mapping; pure mocks, no live model; directly maps to rust-daemon C8 admission semantics.
- `tests/v1/engine/test_engine_args.py` (4) — CLI/engine-arg validation (prefix caching, hashing); pure parsing, prime engine-param validation port.
- `tests/v1/engine/test_engine_core_client.py` (20) — DPLB load-balancing, sticky routing, backpressure are pure-mock unit tests; only a minority fork processes; strong C9/C8 lifecycle fit.
- `tests/v1/engine/test_output_processor.py` (11) — detokenization, stop token/string, incremental output, abort handling on dummy vectors; C4/C8 semantics without a model.
- `tests/v1/metrics/test_stats.py` (12) — scheduler iteration stats, prompt-token cache-hit accounting (local/external transfer); locks the exact metrics shape C8 should emit.
- `tests/v1/metrics/test_histogram_buckets.py` (6) — hard-coded Prometheus bucket snapshots; trivially portable metric-shape contract.
- `tests/v1/metrics/test_metrics_reader.py` (5) — cpu_test-marked reader API over prometheus_client; no engine boot.
- `tests/v1/engine/test_parallel_sampling.py` (3) — parent/child request fan-out and output stream merging; pure unit, C3/C8.
- `tests/v1/engine/test_fast_incdec_prefix_err.py` + `test_logprobs_processor.py` (3 total) — incremental-detok invalid-UTF8 edge and logprobs sentinel truncation; tokenizer/sampler invariants, CPU-only deps.
- `tests/v1/e2e/test_cpu_spec_decode.py` + `test_cpu_linear_attn_chunked_prefix.py` (4 total) — already cpu_model-marked e2e correctness for spec decode and linear-attention chunked prefix; the only upstream e2e files proven CPU-runnable (C6/C5).


<!-- ===== 04-vllm-v1-core.md ===== -->

# 04 — vLLM tests/v1/core + determinism + streaming_input + shutdown

Census of 45 py files (24 core incl. prefix_cache subdir, 9 determinism, 5 streaming_input, 7 shutdown). `n` = `grep -c 'def test_'`. Verdicts: PORT-CPU / DEFER-GPU / SKIP-*. Target = DS41RT component map (C1–C12).

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| tests/v1/core/test_async_scheduler.py | 12 | async scheduler output/builder, pp_size, spec tokens | yes | mocks, pp 1/3, num_spec 0–3 | PORT-CPU | C8 |
| tests/v1/core/test_contiguous_kv_packing.py | 19 | contiguous KV cache packing across group layouts | yes | allocate_kv_cache on torch cpu device, LBNHC/BLHNC | PORT-CPU | C5 |
| tests/v1/core/test_deferred_block_free.py | 13 | deferred block free under async scheduling | yes | pytestmark cpu_test, opt-125m, kv-connector mocks | PORT-CPU | C8 |
| tests/v1/core/test_encoder_cache_manager.py | 20 | encoder (multimodal) cache manager unit tests | yes | pure mocks, no GPU | PORT-CPU | C8 |
| tests/v1/core/test_kv_cache_metrics.py | 16 | KV cache usage/queue metrics | yes | pure unit, patch | PORT-CPU | C5 |
| tests/v1/core/test_kv_cache_utils.py | 98 | block hashing, spec unify, unified page size, hisparse | yes | some subprocess/mmap, HiSparseConfig, LoRA | PORT-CPU | C5 |
| tests/v1/core/test_kv_sharing.py | 3 | kv-sharing layers added to cache groups | yes | torch cpu tensors | PORT-CPU | C5 |
| tests/v1/core/test_mamba_align_chunk_split.py | 9 | mamba align prefill chunk splitting, MTP poisoning reg | yes | hybrid full+mamba specs, partial hits | PORT-CPU | C5 |
| tests/v1/core/test_output.py | 6 | SchedulerOutput NewRequestData mm stripping | yes | pure cpu | PORT-CPU | C8 |
| tests/v1/core/test_prefix_caching.py | 109 | BlockPool, BlockHashToBlockMap, radix reuse, connectors | yes | kv-events, offloading/hisparse connector mocks | PORT-CPU | C5 |
| tests/v1/core/test_priority_preemption_bug.py | 1 | priority-scheduler preemption regression | yes | scheduler-level, no GPU | PORT-CPU | C8 |
| tests/v1/core/test_priority_scheduler_random.py | 1 | randomized priority-scheduler fuzz vs model | yes | seeded random, spec tokens param | PORT-CPU | C8 |
| tests/v1/core/test_repetition_detection.py | 19 | check_stop / sequence repetition detection | yes | pure unit + integration w/ mocks | PORT-CPU | C8 |
| tests/v1/core/test_reset_prefix_cache_e2e.py | 1 | reset_prefix_cache via LLMEngine e2e | no | full engine, model weights | DEFER-GPU | C5 |
| tests/v1/core/test_scheduler_e2e.py | 2 | scheduler behavior through LLM generate | no | full engine, model weights | DEFER-GPU | C8 |
| tests/v1/core/test_scheduler.py | 113 | Scheduler: admission, preemption, chunked prefill, scheduling policy | yes | mocks + MockKVConfig, some hisparse/DCP cases | PORT-CPU | C8 |
| tests/v1/core/test_single_type_kv_cache_manager.py | 17 | SingleTypeKVCacheManager + BlockPool accounting | yes | block_size 3584/4608, in-flight chunks | PORT-CPU | C5 |
| tests/v1/core/test_swa_inflight_window_free.py | 7 | sliding-window out-of-window free vs in-flight steps | yes | ChunkedLocalAttentionSpec, SlidingWindowSpec | PORT-CPU | C5 |
| tests/v1/core/test_worker_slot_overflow.py | 3 | MRv2 "No free indices" slot overflow regression | yes | scheduler-level, mocks | PORT-CPU | C8 |
| tests/v1/core/utils.py | 0 | create_scheduler/create_requests/mock_kv harness | yes | shared fixture code | SKIP-support | — |
| tests/v1/core/__init__.py | 0 | package marker | — | — | SKIP-support | — |
| tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py | 6 | mamba align checkpoints at EAGLE resume position | yes | eagle_group, num_prefill_lookahead params | PORT-CPU | C5 |
| tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py | 46 | partial prefix-cache hits for hybrid full+mamba | yes | dcp_world_size 1/2/4, HMA connector mock | PORT-CPU | C5 |
| tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py | 10 | partial-hit primitives: block pool, event wiring | yes | dcp_world_size 1/2/4, kv-events | PORT-CPU | C5 |
| tests/v1/determinism/test_batch_invariance.py | 5 | decode logprob batch invariance via LLM | no | CUDA, FLASH_ATTN, flaky reruns, 1000s timeout | DEFER-GPU | C3 |
| tests/v1/determinism/test_batch_invariance_vlm.py | 2 | VLM batch-invariance (image+video) | no | CUDA, PIL, FLASH_ATTN/TORCH_SDPA | DEFER-GPU | C11 |
| tests/v1/determinism/test_cutlass_batch_invariance.py | 3 | CUTLASS MoE/linear batch invariance | no | CUDA, cutlass, fp8/nvfp4, MoE kernels | DEFER-GPU | C7 |
| tests/v1/determinism/test_matmul_batch_invariant.py | 3 | batch-invariant matmul kernel vs reference | no | CUDA platform-gated | DEFER-GPU | C3 |
| tests/v1/determinism/test_nvfp4_batch_invariant.py | 1 | NVFP4 GEMM batch invariance via LLM | no | CUDA, NVFP4 | DEFER-GPU | C10 |
| tests/v1/determinism/test_online_batch_invariance.py | 1 | online serving batch invariance via OpenAI client | no | RemoteOpenAIServer + model | DEFER-GPU | C3 |
| tests/v1/determinism/test_rms_norm_batch_invariant.py | 11 | batch-invariant RMSNorm kernels vs torch ref | no | CUDA, dtype/eps/seed sweeps | DEFER-GPU | C3 |
| tests/v1/determinism/conftest.py | 0 | local conftest | — | — | SKIP-support | — |
| tests/v1/determinism/utils.py | 0 | shared determinism helpers (skip_if_not_cuda etc.) | — | — | SKIP-support | — |
| tests/v1/streaming_input/test_async_llm_streaming.py | 2 | AsyncLLM streaming-input API surface | yes | AsyncMock/MagicMock, no GPU | PORT-CPU | C8 |
| tests/v1/streaming_input/test_gpu_model_runner_streaming.py | 2 | GPUModelRunner._update_streaming_request | no | pinned (UVA) memory needs CUDA device | DEFER-GPU | C8 |
| tests/v1/streaming_input/test_gpu_model_runner_v2_streaming.py | 2 | MRv2 GPUModelRunner.add_requests streaming | no | pinned (UVA) memory needs CUDA device | DEFER-GPU | C8 |
| tests/v1/streaming_input/test_scheduler_streaming.py | 8 | Scheduler streaming-update lifecycle | yes | scheduler-level, torch tensors, mocks | PORT-CPU | C8 |
| tests/v1/streaming_input/__init__.py | 0 | package marker | — | — | SKIP-support | — |
| tests/v1/shutdown/test_delete.py | 3 | engine delete/shutdown w/ TP1/TP2 | no | real model, TP2, wait_for_gpu_memory_to_clear | DEFER-GPU | C8 |
| tests/v1/shutdown/test_forward_error.py | 2 | model-forward error → EngineDeadError shutdown | no | real Llama model, TP1/TP2 | DEFER-GPU | C8 |
| tests/v1/shutdown/test_processor_error.py | 1 | Processor error isolation in AsyncLLM | no | real model via AsyncEngineArgs | DEFER-GPU | C8 |
| tests/v1/shutdown/test_startup_error.py | 2 | startup forward/load_weights failure shutdown | no | real Llama model, TP1/TP2 | DEFER-GPU | C8 |
| tests/v1/shutdown/conftest.py | 0 | local conftest | — | — | SKIP-support | — |
| tests/v1/shutdown/utils.py | 0 | SHUTDOWN_TEST_TIMEOUT_SEC, MODELS | — | — | SKIP-support | — |

Counts: 23 PORT-CPU files (~488 test fns), 16 DEFER-GPU files, 6 SKIP-support.

## Notable PORT-CPU candidates

- `test_scheduler.py` (113 fns) — the single richest source: admission, preemption, chunked-prefill, scheduling-policy tests against a mocked scheduler; near-1:1 mapping to ds41rt daemon (C8) request lifecycle.
- `test_prefix_caching.py` (109 fns) — BlockPool/BlockHashToBlockMap accounting, radix reuse, eviction, hit counting; core reference suite for ds41rt-hostcache (C5).
- `test_kv_cache_utils.py` (98 fns) — block hashing (sha256/xxhash, CBOR), spec unification, page-size logic; the hashing semantics hostcache must match (C5).
- `test_single_type_kv_cache_manager.py` (17 fns) — per-request allocation/refcount/free accounting on BlockPool; directly portable block-pool policy tests (C5).
- `prefix_cache/test_partial_prefix_cache_hits.py` (46 fns) — fine-grained partial-hit matching for hybrid (full+mamba) caches, including DCP world sizes; the hardest prefix-cache matching logic to get right (C5).
- `test_deferred_block_free.py` (13 fns) — async-scheduling deferred-free protocol; block-pool lifecycle under pipelined steps, exactly the daemon/hostcache free-list contract (C8/C5).
- `test_contiguous_kv_packing.py` (19 fns) — contiguous allocation/packing invariants across cache-group layouts; allocation-policy reference for hostcache (C5).
- `test_mamba_align_chunk_split.py` + `prefix_cache/test_mamba_eagle_resume_checkpoint.py` (15 fns) — mamba-align chunk boundaries + EAGLE resume checkpoints; boundary-condition tests for any hybrid-cache port (C5).
- `test_priority_scheduler_random.py` + `test_priority_scheduler_random.py` companion (`test_priority_preemption_bug.py`, `test_worker_slot_overflow.py`) — priority/preemption/slot-accounting edge cases the daemon's admission path needs (C8).
- `test_async_scheduler.py` (12 fns) + `test_scheduler_streaming.py` (8 fns) — async-scheduler output assembly and streaming-update lifecycle; relevant if ds41rt daemon adopts async/pipelined scheduling (C8).


<!-- ===== 05-vllm-v1-worker.md ===== -->

# vLLM v1 worker / executor / cudagraph / distributed — test inventory (slice 05)

Census of `tests/v1/worker` (49 py), `tests/v1/executor` (6), `tests/v1/cudagraph` (6), `tests/v1/distributed` (9) from the upstream vLLM checkout. `n` = `grep -c "def test_"`. Verdicts: PORT-CPU (relevant + CPU-runnable), DEFER-GPU (needs GPU/multi-node), SKIP-* (not relevant / empty / other platform). "subset" in pins = file is mixed; only some tests are CPU.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| worker/__init__.py | 0 | — | — | empty | SKIP-EMPTY | — |
| worker/test_attn_utils.py | 12 | KV-cache views/layouts, padded-page strides, block-copy, hisparse alloc | yes | none (no markers; CPU tensors) | PORT-CPU | C5 |
| worker/test_cp_utils.py | 2 | DCP context-attention skip gate, rank invariance | yes | none | PORT-CPU | C8 |
| worker/test_dsv4_packed_zeroer_geometry.py | 2 | Packed DSV4 fp8 MLA zeroer geometry (cross-layer pages) | yes | pytestmark cpu_test | PORT-CPU | C5 |
| worker/test_eagle3_aux_hidden_states_pp.py | 2 | EAGLE3 aux-hidden-state layer sorting, PP support check | yes | none | PORT-CPU | C6 |
| worker/test_encoder_runner.py | 12 | V2 EncoderRunner gather_mm_embeddings, draft +1 look-ahead | yes | pytestmark cpu_test | PORT-CPU | C11 |
| worker/test_gpu_autoregressive_speculator.py | 11 | MTP/EAGLE drafter config, mm-support, draft metadata | mostly | no skipif; mock-based; graph-replay tests may need CUDA | PORT-CPU (subset) | C6 |
| worker/test_gpu_bad_words.py | 4 | BadWords sampler state kernel | no | importorskip triton + module-level CUDA skip | DEFER-GPU | C3 |
| worker/test_gpu_batch_ordering.py | 10 | Decode/prefill classification, batch sort order (runner V1+V2) | yes | none; CPU tensors | PORT-CPU | C8 |
| worker/test_gpu_batch_shard.py | 8 | Batch-sharded sampling routing math / all-to-all layout | no | DEVICE=cuda; per-test requires_cuda | DEFER-GPU | C3 |
| worker/test_gpu_block_table.py | 6 | BlockTables fused multi-group writes | no | pytestmark skipif !is_cuda | DEFER-GPU | C5 |
| worker/test_gpu_extract_hidden_states_speculator.py | 4 | Extract-hidden-states speculator init/dispatch validation | yes | none; cpu device in tests | PORT-CPU | C6 |
| worker/test_gpu_gumbel_sample.py | 10 | Gumbel-max sampling kernel accuracy (heavy tail) | no | importorskip triton + CUDA skip | DEFER-GPU | C3 |
| worker/test_gpu_input_batch.py | 5 | InputBatch state, sampling metadata, pooling lens | yes | parametrized over current_platform device; CPU-capable | PORT-CPU | C8 |
| worker/test_gpu_input_batch_v2.py | 4 | V2 InputBatch dummy-request token distribution, DCP | yes | none; CPU | PORT-CPU | C8 |
| worker/test_gpu_kv_connector.py | 2 | ActiveKVConnector lifecycle event ordering | yes | Mock backend; no markers | PORT-CPU | C5 |
| worker/test_gpu_logit_bias.py | 4 | Logit-bias/min-tokens sampler kernel | no | importorskip triton + CUDA skip | DEFER-GPU | C3 |
| worker/test_gpu_model_runner_mm_gather.py | 4 | V1 _gather_mm_embeddings draft look-ahead (mirrors encoder_runner) | yes | pytestmark cpu_test | PORT-CPU | C11 |
| worker/test_gpu_model_runner.py | 40 | Runner init, KV cache config/grouping, lora/reasoning plumbing | mostly | only 2 CUDA-only tests (FlashInfer, mamba views); gloo init | PORT-CPU (subset) | C8 |
| worker/test_gpu_model_runner_v2_cudagraph_profiling.py | 13 | V2 cudagraph memory-profiling orchestration | yes | runner built via __new__, GPU helpers faked | PORT-CPU | C8 |
| worker/test_gpu_model_runner_v2_eplb.py | 4 | EPLB state machine, from_mapping kwargs, update flow | yes | Fake memory profiler/EPLB state; no GPU | PORT-CPU | C7 |
| worker/test_gpu_model_runner_v2.py | 6 | V2 slot mapping (QSA circular), KV group specs, BlockTables wiring | yes | runner via __new__; CPU | PORT-CPU | C5 |
| worker/test_gpu_pcp_manager.py | 15 | PCP prefill partitioning, row order, graph padding policy | mostly | 1/15 skipif CUDA (GPU-kernel test); rest CPU math | PORT-CPU (subset) | C8 |
| worker/test_gpu_profiler.py | 51 | ProfilerConfig validation, WorkerProfiler wrapper, URI paths | mostly | proton tests skipif CUDA; config tests CPU | PORT-CPU (subset) | C8 |
| worker/test_gpu_rejection_sampler_chunking.py | 2 | Rejection-sampler request chunking + chunked-vs-full scores | partial | 1/2 skipif CUDA | PORT-CPU (subset) | C6 |
| worker/test_gpu_rejection_sampler_i64.py | 1 | int64 indexing in rejection-sample kernels (GLM-scale vocab) | no | ~5 GiB GPU per case; CUDA tensors | DEFER-GPU | C6 |
| worker/test_gpu_sampler_flags.py | 3 | Sampler flag kernels (reasoning tokens etc.) | no | importorskip triton + CUDA skip | DEFER-GPU | C3 |
| worker/test_gpu_thinking_budget.py | 13 | Thinking-budget state kernel | no | importorskip triton + CUDA skip | DEFER-GPU | C3 |
| worker/test_gpu_trace_replay.py | 7 | Trace-replay forced-token sampler state | no | importorskip triton + CUDA skip | DEFER-GPU | C3 |
| worker/test_gpu_ubatch_slicing.py | 22 | DBO microbatch slicing: V2 vs V1 metadata equivalence | partial | ~8 tests skipif CUDA (DBO exec, triton, graph pool); slicing pins CPU | PORT-CPU (subset) | C8 |
| worker/test_gpu_warmup_blocks.py | 6 | Warmup must reserve lookahead KV blocks like scheduler | no | pytestmark skipif !is_cuda (accelerator sync) | DEFER-GPU | C5 |
| worker/test_gpu_worker.py | 7 | Startup-plan persistence (apply/save, hash) | yes | SimpleNamespace worker; no GPU | PORT-CPU | C8 |
| worker/test_gpu_worker_weight_transfer.py | 11 | Weight-transfer engine delegation, update-session guard | yes | recording-engine mocks; CPU LoRA layers | PORT-CPU | C10 |
| worker/test_jit_warmup_migration.py | 1 | Registry warmup kernel vs runtime dispatch parity | no | skipif !is_cuda_alike | DEFER-GPU | C8 |
| worker/test_kv_block_zeroer.py | 9 | KVBlockZeroer kernel + segment tables across specs | no | all tests skipif CUDA | DEFER-GPU | C5 |
| worker/test_kv_cache_allocation_scope.py | 3 | KV-cache allocation scope guards on Worker/runner | yes | mock scope/context managers | PORT-CPU | C5 |
| worker/test_late_interaction_runner.py | 4 | Late-interaction (ColBERT-style) pooling runner | yes | none; CPU | SKIP-IRRELEVANT (pooling, not in DS41RT map) | — |
| worker/test_mamba_hybrid_model_state.py | 4 | Mamba hybrid model state, RecoverSSM metadata | partial | 2/4 skipif CUDA | SKIP-IRRELEVANT (mamba-specific) | — |
| worker/test_mamba_utils.py | 24 | Mamba state copy specs, conv/temporal copy, hybrid state | mostly | 1 CUDA-marked test; rest CPU mocks | SKIP-IRRELEVANT (mamba-specific) | — |
| worker/test_mixed_warmup_gate.py | 2 | Mixed prefill+decode warmup max_num_reqs gate | yes | callback-must-not-run asserts; CPU | PORT-CPU | C8 |
| worker/test_mrope_prompt_embeds.py | 3 | M-RoPE position init for prompt_embeds-only inputs (GHSA fix) | yes | fake model; CPU | PORT-CPU | C11 |
| worker/test_pp_utils.py | 4 | PP sampled-token broadcast row selection | yes | numpy + Mock batch; CPU | PORT-CPU | C9 |
| worker/test_prompt_embeds_state.py | 4 | V2 prompt-embeds overlay kernel (chunked prefill, masks) | no | importorskip triton + CUDA skip | DEFER-GPU | C4 |
| worker/test_sleep_mode_backend.py | 8 | Sleep-mode backend registry/capability flags (cumem) | yes | explicitly CPU; e2e elsewhere | SKIP-IRRELEVANT (cumem sleep-mode, not a ds41rt concern) | — |
| worker/test_spec_decode_embed_sharing_pp.py | 5 | Spec-decode target-embed sharing across PP ranks | yes | fake PP groups; CPU | PORT-CPU | C6 |
| worker/test_utils.py | 39 | KV block copy specs, hisparse mirror flatten/select, misc worker utils | yes | mocks; one monkeypatched is_cuda_alike | PORT-CPU | C5 |
| worker/test_worker_memory_snapshot.py | 1 | Memory-snapshot op ordering across TP worker procs | no | spawns real TP=2 Workers, dummy weights | DEFER-GPU | C9 |
| worker/test_workspace.py | 5 | Workspace lane count from spec config (use_dspark) | yes | stub config; CPU | PORT-CPU | C6 |
| worker/test_xpu_model_runner.py | 1 | XPU torch.cuda shim vs Torch Dynamo | no | skipif no torch.xpu.current_stream; forked | SKIP-IRRELEVANT (XPU-only) | — |
| executor/__init__.py | 0 | — | — | empty | SKIP-EMPTY | — |
| executor/test_executor.py | 8 | Executor async-scheduling support, output aggregators, engine ctor | partial | base/agg tests CPU; AsyncLLM/LLMEngine cases need GPU/model | DEFER-GPU (subset CPU) | C9 |
| executor/test_multiproc_executor.py | 2 | WorkerProc RPC payload lifetime (weakref GC) | yes | fake queue; CPU | PORT-CPU | C9 |
| executor/test_multiproc_executor_timeout.py | 5 | Stale multiproc RPC deadline / FutureWrapper math | yes | pure Python futures + monotonic clock | PORT-CPU | C9 |
| executor/test_ray_utils.py | 2 | Zero-copy detach of ModelRunnerOutput numpy views | yes | numpy only; CPU | PORT-CPU | C9 |
| executor/test_vllm_net_devices.py | 11 | PCI address normalization for vLLM net devices | yes | pure parsing | PORT-CPU | C9 |
| cudagraph/__init__.py | 0 | — | — | empty | SKIP-EMPTY | — |
| cudagraph/test_breakable_cudagraph.py | 19 | Breakable-cudagraph primitives (piecewise capture, eager breaks) | partial | ~5 pure-CPU state tests; most need cuda_capture_stream | DEFER-GPU (subset CPU) | C8 |
| cudagraph/test_cudagraph_dispatch.py | 8 | CudagraphDispatcher dispatch decisions, wrapper modes | partial | several tests skipif CUDA; spawn per test | DEFER-GPU (subset CPU) | C8 |
| cudagraph/test_cudagraph_manager.py | 9 | Graph manager config/orchestration | yes | pytestmark cpu_test | PORT-CPU | C8 |
| cudagraph/test_cudagraph_mode.py | 2 | Backend x cudagraph-mode support matrix | no | LLM() with FA3/FA2/FlashInfer backends | DEFER-GPU | C8 |
| cudagraph/test_encoder_cudagraph.py | 44 | EncoderCudaGraphManager: budget selection, stats, modality routing | partial | 3 no-GPU classes; capture/replay classes CUDA/ROCm | DEFER-GPU (subset CPU) | C11 |
| distributed/__init__.py | 0 | — | — | empty | SKIP-EMPTY | — |
| distributed/test_async_llm_dp.py | 12 | AsyncLLM data-parallel engine behavior | no | spawns DP engines; DP_SIZE=2 default | DEFER-GPU | C8 |
| distributed/test_dbo.py | 1 | DBO + DP + EP GSM8K eval (DeepSeek-V2-Lite) | no | RemoteOpenAIServer, deep_ep, 2+ GPUs | DEFER-GPU | C7 |
| distributed/test_dense_dp_world_size.py | 1 | Dense DP world-size server config | no | RemoteOpenAIServer, DPxTP GPUs | DEFER-GPU | C8 |
| distributed/test_eagle_dp.py | 1 | EAGLE spec decode under DP, batch invariance | no | DP engines + attn backends; SM90 gate | DEFER-GPU | C6 |
| distributed/test_external_lb_dp.py | 3 | External load balancer across DP API servers | no | multiple server procs + openai client | DEFER-GPU | C9 |
| distributed/test_hybrid_lb_dp.py | 3 | Hybrid (per-node) load balancer, 2 nodes | no | 4 DP ranks across 2 nodes | DEFER-GPU | C9 |
| distributed/test_internal_lb_dp.py | 5 | Internal LB multi-node balancing checks | no | 2-node server managers | DEFER-GPU | C9 |
| distributed/test_pp_dp_v2.py | 4 | V2 runner PP+DP interaction, sampled-token broadcast/all-reduce | no | requires 4 GPUs (DP2 PP2) | DEFER-GPU | C9 |

Totals: 70 rows (66 test files + 4 `__init__.py`). 31 PORT-CPU (several partial-subset), 24 DEFER-GPU, 15 SKIP (4 empty, 8 irrelevant, plus XPU/pooling/mamba/cumem folded into SKIP-IRRELEVANT).

## Notable PORT-CPU candidates

- **worker/test_gpu_model_runner_v2_eplb.py** (4) — EPLB expert-placement state machine driven entirely by fakes; the closest upstream analog to ds41rt C7 MoE routing/expert-placement policy, runnable CPU-only today.
- **worker/test_dsv4_packed_zeroer_geometry.py** (2) — `cpu_test`-marked regression for packed cross-layer MLA zeroer geometry; pure layout arithmetic over block/page strides (C5).
- **worker/test_attn_utils.py** (12) — KV-cache layout strides, padded-page views, block-copy semantics on CPU tensors; directly transferable to ds41rt host-cache layout checks (C5).
- **worker/test_utils.py** (39) — hisparse connector row-mirror flatten/select, slot-mapping staging, KV block copy specs; large CPU mock surface for C5-adjacent logic.
- **worker/test_gpu_pcp_manager.py** (15, 14 CPU) — prefill-partition/row-order/padding policy math; scheduler-side request-lifecycle policy in the C8 wheelhouse.
- **worker/test_gpu_batch_ordering.py** (10) — decode/prefill classification and batch sort invariants, pure CPU tensors; cheap C8 ports.
- **worker/test_gpu_worker.py** (7) + **worker/test_mixed_warmup_gate.py** (2) — startup-plan persistence hash/apply and warmup gate policy; config-validation style, no GPU.
- **worker/test_cp_utils.py** (2) — DCP skip-gate rank-invariance; small but sharp collective-deadlock regression, CPU.
- **worker/test_workspace.py** (5) — workspace lane derivation from `use_dspark()`; touches dSpark config plumbing (C6) with stub configs.
- **executor/test_multiproc_executor_timeout.py** (5) + **test_multiproc_executor.py** (2) + **test_vllm_net_devices.py** (11) — RPC deadline/future math and PCI normalization; pure-CPU transport-layer (C9) tests ds41rt can adopt verbatim.


<!-- ===== 06-vllm-v1-attn-sample.md ===== -->

# Inventory 06 — vLLM tests/v1/attention + sample + logits_processors

Census of 56 test files (43 attention, 10 sample, 3 logits_processors) from
`vllm/tests/v1/{attention,sample,logits_processors}`. `n` = `grep -c 'def test_'`.
Targets use the DS41RT component map (C1–C12).

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| attention/test_attention_backends.py | 14 | FA backend correctness vs reference impl | no | device_type | DEFER-GPU | C5 |
| attention/test_attention_backends_selection.py | 2 | backend selection logic (env/config) | yes | — | PORT-CPU | C8 |
| attention/test_attention_splitting.py | 22 | attention workload splitting into branches | yes | cpu-device | PORT-CPU | C8 |
| attention/test_b12x.py | 12 | B12X backend correctness (reuses test_attention_backends) | no | device-cap check | DEFER-GPU | C5 |
| attention/test_backend_per_kind.py | 8 | backend-per-layer-kind resolution, buffer-size clamps | yes | — | PORT-CPU | C8 |
| attention/test_batch_reordering.py | 1 | batch reorder metadata | yes | — | PORT-CPU | C8 |
| attention/test_chunked_local_attention.py | 1 | chunked local attention correctness | yes | device_type | PORT-CPU | C5 |
| attention/test_cpu_mla_backend.py | 5 | CPU MLA backend | yes | cpu-only | PORT-CPU | C5 |
| attention/test_cuda_backend_probe_errors.py | 5 | CUDA backend probe error paths | no | cuda | DEFER-GPU | C8 |
| attention/test_dcp_a2a_pack_mask.py | 3 | DCP all-to-all pack mask | no | cuda-alike | DEFER-GPU | C9 |
| attention/test_deepseek_v4_swa_visible.py | 9 | DeepSeek-V4 SWA visible-window attention | no | cuda | DEFER-GPU | C5 |
| attention/test_dspark_noncausal_sparse_mla.py | 3 | dSpark noncausal sparse MLA | no | cuda | DEFER-GPU | C5 |
| attention/test_flashinfer_dcp_spec_reorder.py | 1 | FlashInfer DCP spec-decode reorder | no | cuda | DEFER-GPU | C6 |
| attention/test_flashinfer_mla_dcp.py | 4 | FlashInfer MLA DCP | no | cuda+flashinfer-mla | DEFER-GPU | C5 |
| attention/test_flashinfer_mla_sparse_sm90.py | 6 | FlashInfer sparse MLA (SM90) | no | cuda sm90 | DEFER-GPU | C5 |
| attention/test_flashinfer_sparse_mla_sm120_api.py | 5 | FlashInfer sparse MLA SM120 API | no | cuda sm120 | DEFER-GPU | C5 |
| attention/test_gdn_metadata_builder.py | 3 | GDN metadata builder | yes | cpu-device | PORT-CPU | C8 |
| attention/test_group_head_counts.py | 3 | grouped head counts | yes | cpu-only | PORT-CPU | C5 |
| attention/test_group_sliding_window.py | 1 | sliding-window group config | yes | — | PORT-CPU | C5 |
| attention/test_indexer_dcp_localize.py | 17 | DCP indexer localisation | no | cuda+cutedsl | DEFER-GPU | C5 |
| attention/test_indexer_deepseek_v4_slot_mapping.py | 8 | DeepSeek-V4 indexer slot mapping | no | cuda | DEFER-GPU | C5 |
| attention/test_indexer_native_next_n.py | 4 | native next-N indexer dispatch (mocked platform) | yes | — | PORT-CPU | C5 |
| attention/test_kpool_tail_slot_mapping.py | 9 | kpool tail slot mapping | yes | — | PORT-CPU | C5 |
| attention/test_kv_head_stride_canonicalization.py | 10 | KV head stride canonicalisation | yes | — | PORT-CPU | C5 |
| attention/test_linear_attention_metadata_builder.py | 4 | linear-attention metadata builder cudagraph support | yes | cpu-device | PORT-CPU | C8 |
| attention/test_mamba_update_block_table.py | 7 | mamba update_block_table metadata reuse regression | yes | cpu-device | PORT-CPU | C8 |
| attention/test_mla_backends.py | 15 | MLA backend correctness | no | device_type | DEFER-GPU | C5 |
| attention/test_mla_context_chunks.py | 9 | MLA context chunking | no | cuda | DEFER-GPU | C5 |
| attention/test_mla_noncausal.py | 6 | MLA noncausal block / cudagraph padding | yes | cpu-device | PORT-CPU | C5 |
| attention/test_mla_prefill_quant_output.py | 7 | MLA prefill quant output selection (mocked platform) | yes | — | PORT-CPU | C5 |
| attention/test_mla_prefill_registry.py | 9 | MLA prefill registry | yes | — | PORT-CPU | C5 |
| attention/test_mla_prefill_selector.py | 21 | MLA prefill selector (mocked platform) | yes | — | PORT-CPU | C5 |
| attention/test_mm_prefix.py | 11 | multimodal prefix attention | no | cuda+fa4 | DEFER-GPU | C11 |
| attention/test_replayssm_metadata_builder.py | 3 | replay-SSM metadata builder | yes | cpu-device | PORT-CPU | C8 |
| attention/test_rocm_aiter_mla_fp8_decode_routing.py | 11 | ROCm aiter MLA fp8 decode routing | no | rocm | DEFER-GPU | C5 |
| attention/test_rocm_aiter_mla_mtp_split.py | 16 | ROCm aiter MLA MTP split | no | rocm | DEFER-GPU | C6 |
| attention/test_rocm_attention_backends_selection.py | 14 | ROCm attention backend selection | no | rocm | DEFER-GPU | C8 |
| attention/test_rocm_glm5next_sparse.py | 9 | ROCm GLM5-next sparse aiter triton | no | rocm+triton | DEFER-GPU | C5 |
| attention/test_sparse_indexer_decode_seq_lens.py | 7 | sparse indexer decode seq lens (platform-agnostic math) | yes | — | PORT-CPU | C5 |
| attention/test_sparse_mla_backends.py | 59 | sparse MLA backend correctness | no | cuda | DEFER-GPU | C5 |
| attention/test_sparse_mla_mask.py | 3 | sparse MLA mask | no | cuda | DEFER-GPU | C5 |
| attention/test_trtllm_attention_integration.py | 2 | TRT-LLM attention integration | no | sm100-cap | DEFER-GPU | C5 |
| sample/test_batched_count_greater_than.py | 2 | logprobs gather op; no dynamo 0/1-specialization recompile | yes | device_type | PORT-CPU | C3 |
| sample/test_head_dtype.py | 8 | lm_head fp32 head_dtype (logits path) | partial | 1 cuda + 1 e2e LLM (core_model) | PORT-CPU | C3 |
| sample/test_logprobs_e2e.py | 2 | prompt_logprobs e2e vs lm_eval arc-easy | no | LLM+RemoteOpenAIServer | DEFER-GPU | C3 |
| sample/test_logprobs.py | 26 | sample/prompt logprobs bookkeeping vs HF ref | no | large_gpu, HfRunner/VllmRunner | DEFER-GPU | C3 |
| sample/test_rejection_sampler.py | 20 | rejection sampler unit tests (mocked sampler) | yes | device_type | PORT-CPU | C3, C6 |
| sample/test_sampler.py | 5 | Sampler semantics (penalties, allowed-token masks) | yes | device_type | PORT-CPU | C3 |
| sample/test_sampling_params_e2e.py | 8 | SamplingParams validation e2e (n>1, penalties, min_p, bad-words) | no | LLM tiny-random | DEFER-GPU | C3 |
| sample/test_thinking_budget_state.py | 2 | ThinkingBudgetStateHolder batch-index moves | yes | cpu-only | PORT-CPU | C3 |
| sample/test_topk_topp_sampler.py | 26 | top-k/top-p sampler (pytorch path + flashinfer) | partial | large_gpu subset | PORT-CPU | C3 |
| sample/test_trace_replay_params.py | 11 | SamplingParams trace_decode_token_ids validation | yes | — | PORT-CPU | C3 |
| logits_processors/test_correctness.py | 30 | v1 LogitsProcessors correctness (bias/min-p/min-tokens/budget) | yes | device_type | PORT-CPU | C3, C2 |
| logits_processors/test_custom_offline.py | 4 | custom (entrypoint) logits processors offline | no | LLM | DEFER-GPU | C3 |
| logits_processors/test_custom_online.py | 2 | custom logits processors via OpenAI server | no | RemoteOpenAIServerCustom | DEFER-GPU | C3, C1 |

## Notable PORT-CPU candidates

- `sample/test_trace_replay_params.py` (11 tests) — pure SamplingParams validation; zero GPU, zero model; directly ports to coordinator-side sampling-param validation.
- `sample/test_topk_topp_sampler.py` (26) — temperature/top-k/top-p unit tests; the `apply_top_k_top_p_pytorch` / `random_sample` subset runs on CPU; prime C3 material.
- `sample/test_rejection_sampler.py` (20) — rejection sampler with fully mocked Sampler; CPU-pure; covers the spec-decode acceptance path the coordinator sampling must match.
- `sample/test_sampler.py` (5) — Sampler penalties + allowed-token-id masks on synthetic logits; CPU-pure; core sampler semantics.
- `logits_processors/test_correctness.py` (30) — the whole v1 logits-processor suite (logit bias, min-p, min-tokens, thinking budget) on fake tensors; largest CPU-pure sampling-logic block here.
- `sample/test_thinking_budget_state.py` (2) — thinking-budget state-holder batch moves on CPU; small but free.
- `sample/test_batched_count_greater_than.py` (2) — logprobs gather op incl. a dynamo no-recompile guard; runs on CPU device_type.
- `sample/test_head_dtype.py` — 6 of 8 tests are CPU-pure head_dtype/logits-processor unit tests; only the mm-fast-path and e2e tests need GPU.
- `sample/test_logprobs.py` (26) — the reference-implementation bookkeeping is worth porting, but as-written it needs GPU (large_gpu mark, HfRunner/VllmRunner) — port its helpers, defer the file.
- `attention/test_cpu_mla_backend.py` + `attention/test_mla_prefill_selector.py`/`test_mla_prefill_registry.py` (35 CPU tests combined) — attention-side but CPU-runnable; useful when validating the MLA prefill/registry logic on the fork.


<!-- ===== 07-vllm-spec-structured.md ===== -->

# 07 — vLLM v1 spec_decode + structured_output test inventory

Batch: `tests/v1/spec_decode/` (30 py) + `tests/v1/structured_output/` (10 py). Census of test files, not a review. ds41rt targets: **C6** (spec decode / dSpark) and **C2** (xgrammar constrained output) are both native to ds41rt, so this batch is high-value.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| spec_decode/__init__.py | 0 | — | — | — | SKIP-empty | — |
| spec_decode/test_acceptance_estimator.py | 6 | Online acceptance-rate estimator (logistic fit, damping, cudagraph padding rows) | no | hardcodes `torch.device("cuda")`; pure tensor math | DEFER-GPU (trivially portable — swap device) | C6 |
| spec_decode/test_acceptance_length.py | 1 | EAGLE3 acceptance-length metric via live generate() | no | VllmRunner, `Llama-3.2-1B` + eagle3, metrics API | DEFER-GPU | C6 |
| spec_decode/test_adaptive_verification.py | 8 | Adaptive spec-decode budget: cudagraph mode resolve, cost curves, marginal-draft cutoff, grammar-row bookkeeping | yes | monkeypatched fakes; imports gpu worker modules | PORT-CPU | C6, C8 |
| spec_decode/test_backup_token_async_spec.py | 7 | Backup-token bookkeeping for async spec decode (inflation bug regression) | yes | pure fake batch/request classes | PORT-CPU | C6 |
| spec_decode/test_dflash2.py | 5 | dFlash2 speculator: grouped-conv vs reference, selector edges/leaves, decoder-layer class, fp32 proposal logits | yes | meta-device model init, monkeypatch | PORT-CPU | C6 |
| spec_decode/test_dflash_causality.py | 5 | dFlash layer-causality config flags + eagle aux-layer ids | yes | config-level only | PORT-CPU | C6 |
| spec_decode/test_dflash_lookahead.py | 3 | dFlash prefill lookahead KV block reservation + window sizing | yes | Scheduler/KV-cache fakes, StructuredOutputManager | PORT-CPU | C5, C6 |
| spec_decode/test_dflash_prepare_inputs.py | 3 | prepare_dflash_inputs: rejected-context suffix exclusion, DCP, null-block safety | no | module-level `skipif(not cuda)` | DEFER-GPU (tensor-prep logic, portable via device swap) | C6 |
| spec_decode/test_draft_attention_backend_override.py | 3 | Draft-model attention backend override does not mutate target config | yes | monkeypatched `load_eagle_model` | PORT-CPU | C6 |
| spec_decode/test_draft_moe_backend_override.py | 3 | Draft MoE backend override/inherit semantics | yes | monkeypatched `load_eagle_model` | PORT-CPU | C6, C7 |
| spec_decode/test_dspark_topk.py | 5 | DSpark Markov-head gathered-bias math: dense-vs-gathered equivalence, w4a16 dequant, nvfp4 group-size guard | partial | 3 CPU-monkeypatch tests; 2 tests `skipif(not cuda)` incl. cudagraph capture | PORT-CPU (3/5; 2 kernel tests DEFER-GPU) | C6, C10 |
| spec_decode/test_dynamic_sd_cug.py | 6 | Dynamic-SD cudagraph shape coverage (uniform decode, clamped query len, piecewise fallback) | yes | `pytestmark = cpu_test`, monkeypatch | PORT-CPU | C6, C8 |
| spec_decode/test_dynamic_sd.py | 17 | Dynamic spec-decode schedule: config validation, k clamping, scheduler integration, DP disable | yes | Scheduler fakes, caplog | PORT-CPU | C6, C8 |
| spec_decode/test_eagle_draft_attn_metadata.py | 4 | Draft attn metadata: CPU seq-len upper bound, padded tail zeroing, max_model_len clamp, DCP recompute | yes | asserts bound `.device.type == "cpu"` | PORT-CPU | C6 |
| spec_decode/test_eagle.py | 10 | Eagle/dflash/draft-model proposer: prepare_inputs, next-token bookkeeping, metadata padding | partial | `DEVICE_TYPE = current_platform.device_type`; test_load_model + 2 propose tests need Llama weights/kernels | PORT-CPU (8/10; load/propose DEFER-GPU) | C6 |
| spec_decode/test_eagle_step_kernel.py | 3 | Fused EAGLE slot-mapping Triton kernel vs Python reference | no | module-level skip unless CUDA/XPU; `importorskip triton` | DEFER-GPU | C6 |
| spec_decode/test_extract_hidden_states.py | 9 | ExtractHiddenStatesProposer init, VLM text-config regression, per-layer hidden-state extraction | yes | mocked model, `DEVICE_TYPE` from platform | PORT-CPU | C6 |
| spec_decode/test_llm_base_proposer.py | 3 | Kernel block-size vs KV-cache-spec fallback; deterministic draft-layer iteration | yes | monkeypatch only | PORT-CPU | C6 |
| spec_decode/test_llm_base_proposer_sampling.py | 2 | fp64 exponential-race sampling in proposer; MTP tuple-return detection | yes | `current_platform`, `set_random_seed` | PORT-CPU | C3, C6 |
| spec_decode/test_max_len.py | 3 | ngram/eagle max_model_len clamping end-to-end; MTP spec config max len | no | `vllm_runner` fixture, real models | DEFER-GPU | C6 |
| spec_decode/test_mtp.py | 2 | MTP load-model wiring + propose token flow (mocked model) | yes | mocked `get_model`/layers/pp_group | PORT-CPU | C6 |
| spec_decode/test_mtp_structured_output.py | 13 | Spec-decode × structured output: draft bitmask padding, grammar termination mid-window, reasoning-end advance, xgrammar validate/accept round-trip | yes | parametrize backend xgrammar/guidance; StructuredOutputManager fakes | PORT-CPU | C2, C6 |
| spec_decode/test_ngram.py | 2 | ngram longest-match proposer: token proposal + full proposer flow | yes | pure | PORT-CPU | C6 |
| spec_decode/test_rejection_sampler_utils.py | 18 | Rejection-sampling core: greedy/stochastic/gumbel correctness, unbiasedness, block verification, placeholder rejection, chunked-vs-full | no | module-level `skipif(not cuda)` + `importorskip triton` | DEFER-GPU (highest-value math in batch; needs kernel shim or reference impl) | C3, C6 |
| spec_decode/test_request_acceptance.py | 11 | Per-request spec-decode metrics: dense histogram, observe/to_dict, engine-output round-trip | yes | pure data classes | PORT-CPU | C6 |
| spec_decode/test_speculators_correctness.py | 2 | Speculator configs correctness vs baseline via live LLM.generate | no | `LLM(...)`, `cleanup_dist_env_and_memory` | DEFER-GPU | C6 |
| spec_decode/test_speculators_eagle3.py | 1 | eagle3 speculator model output through runner | no | `vllm_runner`, `skipif` non-cuda-alike | DEFER-GPU | C6 |
| spec_decode/test_synthetic_rejection_sampler_utils.py | 5 | unconditional→conditional acceptance rates, acceptance-length→rates, min-variance schedule | yes | pure | PORT-CPU | C3, C6 |
| spec_decode/test_vocab_mapping.py | 3 | `_detect_space_prefix` against real HF tokenizers | yes | HF tokenizer downloads (meta-llama/Qwen) | PORT-CPU | C4, C6 |
| structured_output/__init__.py | 0 | — | — | — | SKIP-empty | — |
| structured_output/test_backend_guidance.py | 4 | Guidance backend: rollback-on-terminated, grammar bitmask under spec decode, async grammar init, mistral tokenizer compile | yes | guidance + tokenizer deps | PORT-CPU | C2 |
| structured_output/test_backend_xgrammar_stop_tokens.py | 1 | Request stop-tokens gated to grammar terminal vocab | yes | xgrammar | PORT-CPU | C2 |
| structured_output/test_guidance_negative_draft_tokens.py | 3 | GuidanceGrammar.validate_tokens with trailing padding / negative draft tokens | yes | pure (fake tokenizer/matcher) | PORT-CPU | C2, C6 |
| structured_output/test_outlines_cache.py | 10 | OutlinesDiskCache: store/retrieve, persistence, version invalidation | yes | `pytestmark = cpu_test` | PORT-CPU | C2 |
| structured_output/test_reasoning_structured_output.py | 19 | Reasoning-aware structured output: bitmask fill/advance, reasoning-end index, engine-adapter delta scanning | yes | mock reasoner, StructuredOutputManager | PORT-CPU | C2 |
| structured_output/test_regex_compilation_timeout.py | 5 | compile_regex_with_timeout: success, timeout→ValueError, zero=disabled, error surfacing | yes | pure | PORT-CPU | C2 |
| structured_output/test_scheduler_speculative_padding.py | 4 | strip_speculative_padding utility | yes | pure | PORT-CPU | C2, C6 |
| structured_output/test_utils.py | 4 | xgrammar JSON-schema feature support detection via grammar accept/reject | yes | `pytestmark = cpu_test`, xgrammar | PORT-CPU | C2 |
| structured_output/test_validation.py | 6 | Request-time structured-output validation: bad schema/regex, diffusion-model gate, error types | yes | `pytestmark = cpu_test` | PORT-CPU | C2 |

## Notable PORT-CPU candidates

- **test_rejection_sampler_utils.py** (18 tests, currently CUDA-gated) — the acceptance-sampling math core (greedy/stochastic/gumbel unbiasedness, block verification, placeholder rejection); ds41rt dSpark needs exactly this; port by swapping device or providing a CPU reference path.
- **test_synthetic_rejection_sampler_utils.py** (5) — unconditional→conditional acceptance-rate conversion is already pure CPU and directly mirrors dSpark acceptance-length bookkeeping; cheapest immediate port.
- **test_mtp_structured_output.py** (13) — draft-token bitmask bookkeeping under grammar constraints; the exact intersection of ds41rt's two native features (C2 × C6); already CPU-runnable.
- **test_dynamic_sd.py** (17) — dynamic-k schedule validation + scheduler clamping; large pure-CPU suite for the C6/C8 boundary.
- **test_eagle.py** (10, 8 CPU) — prepare_inputs / next-token-id bookkeeping for EagleProposer, DFlashProposer, DraftModelProposer; core draft-token plumbing, device-agnostic.
- **test_adaptive_verification.py** (8) — marginal-draft budget cutoff + grammar-row-per-draft bookkeeping; pure fakes, high-value adaptive-spec logic.
- **test_backend_guidance.py + test_reasoning_structured_output.py + test_utils.py + test_validation.py** — grammar-compile, reasoning-aware bitmask advance, JSON-schema validation; direct coverage of ds41rt's xgrammar constrained-output path (C2).
- **test_backup_token_async_spec.py** (7) — regression suite for async-spec backup-token off-by-one; pure fakes, exactly the draft-token bookkeeping class of bug ds41rt dSpark must not reintroduce.
- **test_dspark_topk.py** (3 of 5 CPU) — DSpark Markov-head gathered-bias equivalence math; the CUDA-only pair is kernel-vs-reference and can follow later.
- **test_ngram.py** (2) — ngram longest-match proposal logic; smallest, fully pure spec-decode proposer test and a template for CPU proposer tests.


<!-- ===== 08-vllm-openai-entrypoints.md ===== -->

# 08 — vLLM `tests/entrypoints/openai/` test inventory

Source: `/home/turq/dev/upstream-test-inventory/vllm/tests/entrypoints/openai/` (69 py files).
Method: one row per file; `n` = `grep -c "def test_"`. Verdicts: **PORT-CPU** (relevant + CPU-runnable), **DEFER-GPU** (needs a served model), **SKIP-OTHERMODEL** (external correctness harness), **SKIP-INFRA** (package init / fixtures / helpers, no tests).
`cpu` = whether the file's tests run without a GPU/model (`partial` = mixed). `target` = DS41RT component map (C1 OpenAI API surface, C2 constrained output, C3 sampler/logprobs, C4 tokenizer, C7 MoE, C10 weights/quant, C11 vision, C12 tool-call parsing).

Totals: 29 PORT-CPU · 30 DEFER-GPU · 1 SKIP-OTHERMODEL · 9 SKIP-INFRA.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| `__init__.py` | 0 | package init | – | – | SKIP-INFRA | – |
| `utils.py` | 0 | SSE stream-accumulate / response-verify helpers used by many tests | – | helper only | SKIP-INFRA | – |
| `test_watermarking.py` | 3 | `watermarking` flag → SamplingParams for chat/completion/responses requests | yes | protocol models only | PORT-CPU | C1 |
| `test_session_id.py` | 5 | session-id precedence: body field vs `session-id` header vs correlation header vs `vllm-xargs` fallback | yes | starlette Request, protocol models | PORT-CPU | C1 |
| `test_reasoning_enable_thinking.py` | 10 | reasoning_effort / reasoning.effort → `enable_thinking` chat-template kwarg injection (chat + responses) | yes | protocol models only | PORT-CPU | C1 |
| `test_stop_token_ids.py` | 9 | default_sampling_params stop_token_ids merging into request SamplingParams (gpt-oss regression) | yes | protocol models only | PORT-CPU | C1 |
| `test_tool_calls_serialization.py` | 5 | tool_calls Iterable materialisation after model_dump_json (Pydantic lazy-iterator regression) | yes | protocol models only | PORT-CPU | C1 (C12) |
| `test_tool_choice_content_none.py` | 6 | named tool_choice with `content=None`; empty tool_calls omitted from response AND stream-chunk payloads | yes | dummy DelegatingParser | PORT-CPU | C12 |
| `test_render_parity.py` | 8 | cross-API parity: chat-completions vs responses prep paths pass identical args to renderer (templates, content-format, tool kwargs) | yes | mocked OnlineRenderer / serving | PORT-CPU | C1 (C4) |
| `test_render_token_offsets.py` | 4 | `return_token_offsets` flag wiring: request → TokenizeParams → GenerateRequest serialization | yes | Mock ModelConfig | PORT-CPU | C1 |
| `test_chunked_prompt.py` | 2 | chat with chunked prefill enabled | no | Qwen3-0.6B server, `--enable-chunked-prefill` | DEFER-GPU | C1 (C8) |
| `test_openai_schema.py` | 1 | schemathesis OpenAPI-schema conformance fuzzing against live server | no | SmolVLM-256M-Instruct server (vision) | DEFER-GPU | C1 (C11) |
| `test_return_routed_experts.py` | 1 | base64 routed-experts array in response (MoE top-k per token/layer) | no | tiny-mixtral server (8 experts, top-2) | DEFER-GPU | C7 |
| `test_return_token_ids.py` | 4 | `return_token_ids` response field incl. tool-use and logprob interplay | no | Qwen2.5-1.5B + hermes tool parser | DEFER-GPU | C1 (C3) |
| `test_return_tokens_as_ids.py` | 3 | `return_tokens_as_token_ids` (completion + chat; 1 protocol unit test) | partial | Qwen3-0.6B + LoRA server | DEFER-GPU | C1 |
| `chat_completion/__init__.py` | 0 | package init | – | – | SKIP-INFRA | – |
| `chat_completion/test_chat.py` | 35 | main chat e2e: streaming, logprobs, guided decoding, tokenizer/template params | no | zephyr-7b-beta server | DEFER-GPU | C1 (C2/C3) |
| `chat_completion/test_chat_completion.py` | 6 | guided-decoding 400s (json_schema/regex/grammar) + KV-transfer prompt-token-ids round trip | no | Qwen2.5-1.5B server | DEFER-GPU | C1 (C2) |
| `chat_completion/test_batched_chat_completions.py` | 7 | batched chat endpoint: json_schema, logprobs placeholders, tokens-as-ids, echo prefix | no | Qwen2.5-1.5B server | DEFER-GPU | C1 (C3) |
| `chat_completion/test_chat_completion_with_prompt_embeds.py` | 7 | `prompt_embeds` content parts e2e (single/multi part, streaming, base64 errors) | no | opt-125m + chatml.jinja template, torch | DEFER-GPU | C1 |
| `chat_completion/test_chat_echo.py` | 3 | echo mode + prompt_logprobs/top_logprobs on chat | no | Qwen2-1.5B server | DEFER-GPU | C1 (C3) |
| `chat_completion/test_chat_error.py` | 15 | serving-chat error paths with mocked engine: validation errors, generation errors, error-response shape | yes | MagicMock/AsyncMock engine | PORT-CPU | C1 |
| `chat_completion/test_chat_logit_bias_validation.py` | 5 | logit_bias 400 validation against real vocab size | no | Qwen2.5-1.5B server + ModelConfig | DEFER-GPU | C1 (C3) |
| `chat_completion/test_completion_with_function_calling.py` | 5 | function-calling chat e2e with jsonschema validation of arguments | no | Qwen3-0.6B server | DEFER-GPU | C12 |
| `chat_completion/test_enable_force_include_usage.py` | 1 | `--enable-force-include-usage`: usage chunk on every stream chunk | no | Qwen3-0.6B server, fixed port | DEFER-GPU | C1 |
| `chat_completion/test_extra_content_fields.py` | 2 | extra fields on content parts survive parsing to template (text + image branches) | no | translategemma-4b-it server | DEFER-GPU | C11 |
| `chat_completion/test_include_reasoning.py` | 6 | `include_reasoning` suppress/keep reasoning content, reasoning tokens in usage, stream + non-stream | no | Qwen3-0.6B + qwen3 reasoning parser | DEFER-GPU | C1 |
| `chat_completion/test_logprob_token_ids.py` | 7 | `logprob_token_ids` pinned-vocab logprobs: 4 unit validation + 3 server e2e | partial | Qwen2.5-1.5B server (3 tests) | PORT-CPU | C1 (C3) |
| `chat_completion/test_non_object_body_validation.py` | 2 | non-object JSON body → clean 4xx not AttributeError 500 | yes | pydantic ValidationError only | PORT-CPU | C1 |
| `chat_completion/test_root_path.py` | 1 | `--root-path` routing + API key auth | no | Qwen2-1.5B server | DEFER-GPU | C1 |
| `chat_completion/test_serving_chat.py` | 34 | serving-chat unit tests (streaming accumulation, aborts, mm handling, metrics, echo) w/ mocked engine; 2 gpt-oss harmony server tests | partial | mocks; gpt-oss server for 2 tests | PORT-CPU | C1 (C11) |
| `chat_completion/test_thinking_token_budget.py` | 4 | thinking_token_budget e2e with reasoning models | no | Qwen3-0.6B; Qwen3.5-35B-FP8 MTP multi-GPU variant | DEFER-GPU | C1 |
| `chat_completion/test_thinking_token_budget_validation.py` | 6 | thinking_token_budget validation (reject negative/float, accept valid) | yes | protocol models only | PORT-CPU | C1 |
| `completion/__init__.py` | 0 | package init | – | – | SKIP-INFRA | – |
| `completion/test_completion.py` | 21 | main completions e2e: streaming, logprobs, echo, tokenizer params | no | opt-125m server | DEFER-GPU | C1 (C3) |
| `completion/test_completion_error.py` | 28 | serving-completion errors/metrics with mocked engine: per-request metrics, spec-decode metrics, prompt limits, structured-output 400s | yes | MagicMock engine, AsyncLLM spec | PORT-CPU | C1 (C6) |
| `completion/test_completion_with_prompt_embeds.py` | 5 | prompt_embeds completions: logprobs interplay, prompt_logprobs error, LoRA | no | opt-125m + LoRA server, torch | DEFER-GPU | C1 |
| `completion/test_lora_resolvers.py` | 4 | LoRA resolver registry serving paths (multi-LoRA, resolution failures) | yes | mocks | PORT-CPU | C1 (C10) |
| `completion/test_prompt_validation.py` | 4 | empty-prompt 400, OOV token ids, prompt_embeds limits (2 server + 2 unit) | partial | gpt2 server (2 tests); torch embeds | PORT-CPU | C1 |
| `completion/test_tensorizer_entrypoint.py` | 1 | tensorized (serialized) model served via OAI endpoint | no | llama-3.2-1b + tensorizer | DEFER-GPU | C10 |
| `completion/test_token_in_token_out.py` | 1 | tokens-in/tokens-out endpoint (tokenizer-only weight download) | no | Qwen3-0.6B tokenizer-only | DEFER-GPU | C1 (C4) |
| `correctness/__init__.py` | 0 | package init | – | – | SKIP-INFRA | – |
| `correctness/test_lmeval.py` | 1 | lm-eval gsm8k accuracy over OAI API (500 concurrent) | no | Qwen2-1.5B + lm_eval harness | SKIP-OTHERMODEL | C1 |
| `models/__init__.py` | 0 | package init | – | – | SKIP-INFRA | – |
| `models/test_models.py` | 1 | /v1/models listing incl. LoRA modules | no | Qwen3-0.6B + LoRA server | DEFER-GPU | C1 |
| `parser/__init__.py` | 0 | package init | – | – | SKIP-INFRA | – |
| `parser/test_harmony_render_parity.py` | 10 | chat↔responses input paths converge to identical Harmony messages + identical rendered token sequences | yes | pure unit, openai types | PORT-CPU | C1 (C4) |
| `parser/test_harmony_utils.py` | 66 | harmony utils: message parsing, tool definitions, recipient extraction, analysis-drop, system message | yes | pure unit | PORT-CPU | C1 (C12) |
| `responses/__init__.py` | 0 | package init | – | – | SKIP-INFRA | – |
| `responses/conftest.py` | 0 | shared fixtures: `server_with_store` (Qwen3-1.7B, store enabled), event-type pairs, streaming validators | – | helper only | SKIP-INFRA | – |
| `responses/test_basic.py` | 6 | responses create/stream basics: reasoning + message output ordering | no | conftest client (Qwen3-1.7B server) | DEFER-GPU | C1 |
| `responses/test_errors.py` | 3 | `_raise_if_error` error mapping in base serving | yes | MagicMock engine | PORT-CPU | C1 |
| `responses/test_function_call_parsing.py` | 14 | function_call input items parsed into ResponseFunctionToolCall objects | yes | protocol only | PORT-CPU | C12 |
| `responses/test_function_call.py` | 6 | tool use e2e: auto/named tool_choice, streaming argument assembly, multi-turn | no | conftest client | DEFER-GPU | C12 |
| `responses/test_harmony.py` | 32 | Harmony responses integration: structured output, store/background, streaming types, web search | no | gpt-oss harmony server | DEFER-GPU | C1 (C12) |
| `responses/test_harmony_utils.py` | 16 | harmony_to_response_output / previous-response-input conversion | yes | pure unit | PORT-CPU | C1 |
| `responses/test_mcp_tools.py` | 9 | MCP tool server: description post-processing, tool sessions, namespace tools; 1 server test | partial | mock MCP server; 1 RemoteOpenAIServer | PORT-CPU | C12 |
| `responses/test_namespace_tool_separator.py` | 2 | `namespace__tool` flat-name round trip, stream + non-stream | no | conftest client | DEFER-GPU | C12 |
| `responses/test_parsable_context.py` | 5 | ParsableContext e2e tool-call extraction over served model | no | harmony server env | DEFER-GPU | C12 |
| `responses/test_parsable_context_unit.py` | 10 | ParsableContext: parser deltas → response output items | yes | mocks | PORT-CPU | C12 |
| `responses/test_protocol.py` | 2 | serialize_message(s) helpers | yes | pure unit | PORT-CPU | C1 |
| `responses/test_response_input_to_harmony.py` | 17 | every responses input-item type → correct Harmony Message (role/channel/recipient/content) | yes | pure unit | PORT-CPU | C1 |
| `responses/test_responses_utils.py` | 33 | output-item/message construction + input message construction utils | yes | pure unit | PORT-CPU | C1 |
| `responses/test_sampling_params.py` | 8 | ResponsesRequest → SamplingParams mapping incl. structured-outputs params | yes | unit (torch import only) | PORT-CPU | C1 (C2/C3) |
| `responses/test_serving_responses.py` | 35 | serving-responses unit: renderer handling of harmony history, streaming event emission (preamble, reasoning deltas, multi-tool), tool sessions, input validation, token counting | yes | mocks / monkeypatch | PORT-CPU | C1 (C12) |
| `responses/test_simple.py` | 10 | e2e simple responses incl. streaming event-stack validation | no | Qwen3-8B server | DEFER-GPU | C1 |
| `responses/test_stateful.py` | 7 | store/retrieve/cancel response lifecycle | no | conftest client (store enabled) | DEFER-GPU | C1 |
| `responses/test_streaming_events.py` | 7 | Responses SSE event state machine: event types, delta splitting, lifecycle ordering | yes | pure unit | PORT-CPU | C1 |
| `responses/test_structured_output.py` | 2 | json_schema text format e2e incl. `.parse()` | no | conftest client | DEFER-GPU | C2 |

## Notable PORT-CPU candidates

- `responses/test_streaming_events.py` (7) — pure-unit SSE event sequence/state machine for the Responses API; the streaming chunk conformance surface in its most portable form.
- `chat_completion/test_serving_chat.py` (34, mostly) — mocked-engine serving tests: streaming accumulation, aborts, error responses; the 2 gpt-oss harmony tests defer.
- `completion/test_completion_error.py` (28) — mocked-engine error paths, per-request metrics, prompt-limit validation; large protocol-error corpus with zero GPU.
- `responses/test_serving_responses.py` (35) — rendering rules, streaming event emission (preamble/reasoning/multi-tool), input validation; all mock-based.
- `parser/test_harmony_utils.py` (66) + `responses/test_response_input_to_harmony.py` (17) — exhaustive request-input → Harmony message conversion; defines cross-API input semantics.
- `chat_completion/test_chat_error.py` (15) — serving-chat validation/generation error mapping to OpenAI error response shapes.
- `responses/test_function_call_parsing.py` (14) + `responses/test_parsable_context_unit.py` (10) — tool-call item parsing and delta→tool-call assembly; direct C12 port material.
- `responses/test_sampling_params.py` (8) + `chat_completion/test_stop_token_ids.py` (9) + `test_watermarking.py` (3) — request→SamplingParams mapping rules across all three APIs (chat/completion/responses).
- `test_tool_choice_content_none.py` (6) — exact response/stream payload rules for empty tool_calls and content=None tool choice.
- `test_session_id.py` (5) + `chat_completion/test_non_object_body_validation.py` (2) — small but precise HTTP-semantics tests (header precedence, malformed-body 4xx) that port verbatim.


<!-- ===== 09-vllm-serve-scaleout.md ===== -->

# 09 — vLLM entrypoints: serve / llm / launchers / scale_out / unit_tests

Census of `/home/turq/dev/upstream-test-inventory/vllm/tests/entrypoints/{serve,llm,launchers,scale_out,unit_tests}`.
`n` = `grep -c "def test_"`. Verdicts: PORT-CPU / DEFER-GPU / SKIP-INFRA.
Targets per DS41RT component map: C1 OpenAI API, C2 constrained output, C3 sampler, C4 tokenizer, C5 KV cache, C6 spec decode, C7 MoE, C8 scheduler/admission (config + engine args), C9 transport/RPC, C10 weights/quant, C11 vision, C12 tool parsing.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| serve/dev/test_sleep.py | 1 | engine sleep/wake via API | no | RemoteOpenAIServer, Llama-3.2-1B, sleep-mode | DEFER-GPU | C5 |
| serve/dev/rpc/test_collective_rpc.py | 3 | worker-extension RPC over HTTP | no | RemoteOpenAIServer, Qwen3-0.6B | DEFER-GPU | C9 |
| serve/dev/rlhf/conftest.py | 0 | RLHF pause/resume fixtures | no | RemoteOpenAIServer helpers | DEFER-GPU | C8 |
| serve/dev/rlhf/state_transitions/test_pause_resume.py | 4 | RLHF pause/resume lifecycle | no | RemoteOpenAIServer, conftest above | DEFER-GPU | C8 |
| serve/exception_handling/test_error_sanitization.py | 11 | error-message sanitization | yes | pure function (sanitize_message), CVE-2026-22778 | PORT-CPU | C1 |
| serve/exception_handling/test_http_status_metrics.py | 11 | http status→Prometheus metric mapping | yes | FastAPI build_app, mocked engine | PORT-CPU | C1 |
| serve/exception_handling/test_validation_exception_handler.py | 11 | 4xx param field from pydantic loc | yes | fake request, no server | PORT-CPU | C1 |
| serve/instrumentator/test_basic.py | 7 | /metrics + /health + version endpoints | no | RemoteOpenAIServer, Qwen3-0.6B | DEFER-GPU | C1 |
| serve/instrumentator/test_metrics.py | 4 | Prometheus metric content | no | RemoteOpenAIServer, TinyLlama/SmolVLM, subprocess | DEFER-GPU | C1 |
| serve/instrumentator/test_orca_metrics.py | 3 | ORCA format metrics | no | RemoteOpenAIServer, Qwen3-0.6B | DEFER-GPU | C1 |
| serve/instrumentator/test_uds.py | 1 | UDS socket serving | no | RemoteOpenAIServer, --uds | DEFER-GPU | C9 |
| serve/lora/test_lora_adapters.py | 10 | LoRA load/unload API + bad-request cases | no | RemoteOpenAIServer, Qwen3-0.6B | DEFER-GPU | C1 |
| serve/lora/test_serving_models.py | 9 | OpenAIServingModels / pooling models logic | yes | MagicMock engine/client | PORT-CPU | C1 |
| serve/middleware/test_authentication_middleware.py | 2 | api-key middleware route coverage | yes | FastAPI TestClient, no server | PORT-CPU | C1 |
| serve/middleware/test_optional_middleware.py | 9 | --api-key / request-id-headers behaviour | no | RemoteOpenAIServer, e5-small | DEFER-GPU | C1 |
| serve/sagemaker/conftest.py | 0 | SageMaker fixtures | no | RemoteOpenAIServer, HF downloads | SKIP-INFRA | — |
| serve/sagemaker/test_sagemaker_handler_overrides.py | 7 | SM handler override precedence | yes | in-process TestClient, but SM machinery | SKIP-INFRA | — |
| serve/sagemaker/test_sagemaker_lora_adapters.py | 7 | SM /adapters endpoint | no | RemoteOpenAIServer + LoRA | SKIP-INFRA | — |
| serve/sagemaker/test_sagemaker_middleware_integration.py | 3 | SM custom middleware loading | no | RemoteOpenAIServer | SKIP-INFRA | — |
| serve/sagemaker/test_sagemaker_stateful_sessions.py | 4 | SM session headers/lifecycle | no | RemoteOpenAIServer | SKIP-INFRA | — |
| serve/tokenize/test_serving_tokenization.py | 8 | /tokenize serving logic | yes | TestClient + AsyncMock renderer | PORT-CPU | C4 |
| serve/tokenize/test_tokenization.py | 9 | /tokenize, /detokenize HTTP | no | RemoteOpenAIServer, SmolLM2-135M | DEFER-GPU | C4 |
| serve/tokenize/test_tokenization_vlm.py | 1 | /tokenize image-placeholder expansion | no | RemoteOpenAIServer, Qwen2.5-VL-3B | DEFER-GPU | C4/C11 |
| serve/tokenize/test_tokenize_then_chat_vlm.py | 1 | /tokenize must not pollute mm caches | no | RemoteOpenAIServer, Qwen2.5-VL-3B | DEFER-GPU | C4/C11 |
| serve/utils/test_api_utils.py | 12 | stream-options/max-tokens helpers, arg redaction | yes | pure functions, Namespace args | PORT-CPU | C1 |
| serve/utils/test_fingerprint.py | 4 | system_fingerprint construction | yes | SimpleNamespace config | PORT-CPU | C1 |
| serve/utils/test_request_logger.py | 9 | RequestLogger output formatting | yes | MagicMock logger | PORT-CPU | C1 |
| serve/utils/test_sse_keep_alive.py | 11 | SSE keep-alive wrapper semantics | yes | pure async generator | PORT-CPU | C1 |
| llm/test_accuracy.py | 2 | server accuracy via lm-eval gsm8k | no | lm_eval, Qwen3-1.7B, NUM_CONCURRENT=500 | DEFER-GPU | C1 |
| llm/test_chat.py | 5 | LLM.chat API, tokenize=, bad-request cases | no | vllm_runner Llama-3.2-1B | DEFER-GPU | C1/C4 |
| llm/test_collective_rpc.py | 1 | LLM.collective_rpc serialization | no | vllm_runner, torch.accelerator, tp1/2 | DEFER-GPU | C9 |
| llm/test_generate.py | 5 | LLM.generate prompts/token-ids validation | no | vllm_runner distilgpt2 | DEFER-GPU | C1 |
| llm/test_gpu_utilization.py | 1 | gpu_memory_utilization is per-instance | no | 3× vllm_runner opt-125m | DEFER-GPU | C8 |
| llm/test_prompt_validation.py | 3 | empty prompt / OOV token / mm-embeds errors | no | vllm_runner gpt2/llava | DEFER-GPU | C1 |
| llm/test_struct_output_generate.py | 6 | guided structured output via LLM API | no | vllm_runner + jsonschema/regex/xgrammar | DEFER-GPU | C2 |
| llm/offline_mode/test_offline_mode.py | 2 | HF_HUB_OFFLINE=1 LLM behaviour | no | vllm_runner opt-125m/Qwen3/Mistral | DEFER-GPU | C1 |
| launchers/test_cli_args.py | 30 | serve CLI parser + validate_parsed_serve_args | yes | pure argparse, no engine | PORT-CPU | C8 |
| launchers/test_dp_supervisor.py | 20 | DPSupervisor lifecycle + arg building | yes | fake aiohttp children, no GPU per docstring | PORT-CPU | C8 |
| launchers/test_grpc_health.py | 10 | gRPC health servicer states | yes | importorskip grpc, MagicMock LLM | SKIP-INFRA | C9 |
| launchers/test_launch_cli.py | 11 | `vllm launch`/`launch render` parsing | yes | pure subcommand parser | PORT-CPU | C8 |
| launchers/test_run_batch.py | 25 | run_batch request handling, upload/download | yes | MagicMock connection, subprocess bits | PORT-CPU | C8 |
| launchers/test_shutdown.py | 7 | signal/shutdown/timeout behaviour | no | RemoteOpenAIServer, psutil | DEFER-GPU | C8 |
| launchers/test_ssl_cert_refresher.py | 1 | SSL cert hot-reload | yes | MockSSLContext, tmpfiles | SKIP-INFRA | C9 |
| launchers/api_server/test_api_server_process_manager.py | 7 | APIServerProcessManager, zmq addr plumbing | yes | mock workers, zmq sockets | SKIP-INFRA | C9 |
| launchers/api_server/test_multi_api_servers.py | 2 | multi api-server request balancing | no | RemoteOpenAIServer, --api-server-count 4 | DEFER-GPU | C1 |
| scale_out/test_factories.py | 5 | router registration for scale-out paths | yes | FastAPI app, Namespace args | PORT-CPU | C9/C1 |
| scale_out/derender/test_derender.py | 35 | /derender HTTP endpoints | yes | RemoteLaunchRenderServer (GPU-less) | PORT-CPU | C9/C4 |
| scale_out/derender/test_derender_stream.py | 28 | streaming derender delta correctness | yes | tokenizer-only unit layer + render server | PORT-CPU | C9/C4 |
| scale_out/derender/test_derender_parity.py | 5 | coupled vs disaggregated parse parity | no | RemoteOpenAIServer full GPU serve | DEFER-GPU | C9 |
| scale_out/ec_integration/test_scale_out_ec_e2e.py | 0 | EC connector disagg E2E (script-driven) | no | bash-scripted multi-instance, Qwen3-VL | SKIP-INFRA | C9 |
| scale_out/render/test_render.py | 32 | /render request validation + serving | yes | mocked ServingRender + GPU-less server | PORT-CPU | C9 |
| scale_out/render/test_launch_render.py | 8 | render via `vllm launch` (GPU-less) | yes | RemoteLaunchRenderServer | PORT-CPU | C9 |
| scale_out/render/test_render_multimodal.py | 2 | /render with images/video | no | RemoteOpenAIServer, Qwen3-VL-2B | DEFER-GPU | C11/C9 |
| scale_out/token_in_token_out/test_generate_stream.py | 12 | ServingTokens stream generation | yes | AsyncMock AsyncLLM, no server | PORT-CPU | C9 |
| scale_out/token_in_token_out/test_mm_serde.py | 13 | multimodal kwargs serde roundtrip | yes | CPU torch tensors, pydantic | PORT-CPU | C9/C11 |
| scale_out/token_in_token_out/test_protocol.py | 8 | GenerateRequest pydantic validators | yes | direct protocol objects | PORT-CPU | C9 |
| scale_out/token_in_token_out/test_return_routed_experts.py | 1 | return_routed_experts via /generate | no | RemoteOpenAIServer, tiny-mixtral | DEFER-GPU | C7/C9 |
| scale_out/token_in_token_out/test_serving_multimodal_tokens.py | 4 | mm through /render→/generate→/detokenize | no | RemoteOpenAIServer, Qwen3-VL-2B | DEFER-GPU | C9/C11 |
| scale_out/token_in_token_out/test_serving_tokens.py | 9 | /inference/v1/generate end-to-end | no | RemoteOpenAIServer, Qwen3-0.6B | DEFER-GPU | C9 |
| scale_out/token_in_token_out/test_tokens_logprobs.py | 2 | tokens+logprobs response shaping | yes | pure ServingTokens static method | PORT-CPU | C9 |
| unit_tests/test_chat_utils.py | 81 | chat-message parsing, mm tracker, templates | yes | HF tokenizers, image/audio/video assets | PORT-CPU | C1/C4/C12 |
| unit_tests/test_context.py | 22 | Harmony responses context assembly | yes | openai_harmony, mocked outputs | PORT-CPU | C1 |
| unit_tests/test_non_object_body_validation.py | 6 | non-object JSON body → clean 4xx | yes | direct pydantic validators, 8 protocols | PORT-CPU | C1 |
| unit_tests/test_offline_utils.py | 4 | OfflineInferenceMixin param validation | yes | pure mixin, SamplingParams | PORT-CPU | C1/C8 |
| unit_tests/test_remote_vllm_server.py | 3 | CLI-arg redaction for spawned servers | yes | pure arg-list transform | PORT-CPU | C8 |

Totals: 64 files (62 test files + 2 conftest), 566 test functions. PORT-CPU 28, DEFER-GPU 28, SKIP-INFRA 8 (incl. 2 conftest).

## Notable PORT-CPU candidates

- `launchers/test_cli_args.py` (30 tests) — `make_arg_parser` + `validate_parsed_serve_args`: the canonical corpus for serve-arg precedence, mutual-exclusion and error-message wording; directly adoptable by ds41rt's daemon/config layer (C8).
- `launchers/test_launch_cli.py` (11) — `vllm launch`/`launch render` subcommand parsing semantics, CPU-only (C8).
- `launchers/test_dp_supervisor.py` (20) — `_build_vllm_dp_server_args` rank/port derivation plus lifecycle, runs GPU-free with fake children (C8).
- `launchers/test_run_batch.py` (25) — batch-request validation, `VLLMValidationError` wording, upload/download helpers; mostly mocked (C8).
- `unit_tests/test_non_object_body_validation.py` (6) — validator guard semantics: non-object bodies must yield clean 4xx across 8 request protocols; exact error-message pinning ds41rt's API layer needs (C1).
- `unit_tests/test_chat_utils.py` (81) — largest CPU suite: chat-template validation, message parsing, multimodal item tracking, `_load_embeds_dict`; carries C12 tool-parsing and C4 tokenizer semantics.
- `serve/utils/test_api_utils.py` (12) — `get_max_tokens` / `should_include_usage` / `redact_sensitive_args` precedence rules, pure functions over `Namespace` args (C1/C8).
- `scale_out/token_in_token_out/test_protocol.py` (8) — `GenerateRequest` pydantic validator semantics (max_tokens defaults, mm features); spec-level contract for the token-in/token-out RPC boundary (C9).
- `serve/exception_handling/test_validation_exception_handler.py` (11) — `param`-field derivation from pydantic `loc` and 4xx shape; verbatim error-contract tests (C1).
- `serve/middleware/test_authentication_middleware.py` (2, with `scale_out/test_factories.py` 5) — route-registration coverage vs `--api-key` guarding and scale-out path gating; config-flag→route semantics, no server needed (C1/C9).


<!-- ===== 10-vllm-pooling-mm-stt.md ===== -->

# Inventory 10 — vLLM entrypoints: pooling, multimodal, speech_to_text, cohere

Scope: `vllm/tests/entrypoints/{pooling,multimodal,speech_to_text,cohere}`.
`unit_tests/` excluded (owned by another agent). ds41rt does not serve pooling or
speech endpoints; vision is V4.1-native. CPU = runs without GPU/model
(monkeypatch/pydantic/mocks). n = `grep -c "def test_"`.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| pooling/basic/test_encode.py | 2 | pooling embed offline | no | vllm_runner | SKIP-OTHERMODEL | C1 |
| pooling/basic/test_truncation.py | 3 | pooling truncation server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/basic/test_tiling_engine.py | 4 | pooling input tiling | yes | none | PORT-CPU | C1 |
| pooling/classify/test_online.py | 14 | pooling classify server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/classify/test_offline.py | 6 | pooling classify offline | no | vllm_runner | SKIP-OTHERMODEL | C1 |
| pooling/classify/test_online_vision.py | 4 | pooling classify vision | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| pooling/embed/test_online.py | 17 | embeddings server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/embed/test_online_dimensions.py | 1 | embeddings server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/embed/test_online_long_text.py | 5 | embeddings server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/embed/test_online_vision.py | 5 | embeddings vision | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| pooling/embed/test_offline.py | 5 | embeddings offline | no | vllm_runner | SKIP-OTHERMODEL | C1 |
| pooling/embed/test_protocol.py | 19 | embed protocol schemas | yes | none | PORT-CPU | C1 |
| pooling/embed/test_io_processor.py | 34 | embed io processor | yes | none | PORT-CPU | C1 |
| pooling/embed/test_cohere_online.py | 15 | cohere-embed server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/embed/test_cohere_online_vision.py | 5 | cohere-embed vision | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| pooling/embed/test_cohere_openai_parity.py | 3 | cohere/embed parity | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/embed/test_correctness_mteb.py | 1 | embed accuracy | no | vllm_runner+MTEB | SKIP-OTHERMODEL | C1 |
| pooling/reward/test_token_reward_online.py | 11 | reward server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/reward/test_token_reward_offline.py | 2 | reward offline | no | vllm_runner | SKIP-OTHERMODEL | C1 |
| pooling/scoring/test_bi_encoder_online.py | 14 | scoring server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/scoring/test_bi_encoder_offline.py | 4 | scoring offline | no | vllm_runner | SKIP-OTHERMODEL | C1 |
| pooling/scoring/test_cross_encoder_online.py | 18 | scoring server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/scoring/test_cross_encoder_offline.py | 6 | scoring offline | no | vllm_runner | SKIP-OTHERMODEL | C1 |
| pooling/scoring/test_cross_encoder_online_vision.py | 11 | scoring vision | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| pooling/scoring/test_cross_encoder_correctness_mteb.py | 2 | scoring accuracy | no | vllm_runner+MTEB | SKIP-OTHERMODEL | C1 |
| pooling/scoring/test_late_interaction_online.py | 7 | scoring server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/scoring/test_late_interaction_offline.py | 4 | scoring offline | no | vllm_runner | SKIP-OTHERMODEL | C1 |
| pooling/scoring/test_late_interaction_online_vision.py | 6 | scoring vision | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| pooling/scoring/test_late_interaction_offline_vision.py | 3 | scoring vision | no | vllm_runner | SKIP-OTHERMODEL | C11 |
| pooling/scoring/test_late_interaction_serving.py | 1 | scoring serving logic | yes | none | PORT-CPU | C1 |
| pooling/scoring/test_io_processor_unit.py | 1 | scoring io processor | yes | none | PORT-CPU | C1 |
| pooling/scoring/test_jina_ranking_io_processor_unit.py | 3 | ranking io processor | yes | none | PORT-CPU | C1 |
| pooling/scoring/test_utils.py | 1 | scoring input validation | yes | none | PORT-CPU | C1 |
| pooling/token_classify/test_online.py | 2 | token classify server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/token_classify/test_offline.py | 4 | token classify offline | no | vllm_runner | SKIP-OTHERMODEL | C1 |
| pooling/token_embed/test_online.py | 2 | token embed server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| pooling/token_embed/test_offline.py | 3 | token embed offline | no | vllm_runner | SKIP-OTHERMODEL | C1 |
| pooling/test_factories.py | 3 | pooling task factories | yes | monkeypatch | PORT-CPU | C1 |
| pooling/test_io_processor.py | 7 | pooling io processor | yes | monkeypatch | PORT-CPU | C1 |
| pooling/test_utils.py | 3 | pooling output utils | yes | none | PORT-CPU | C1 |
| multimodal/llm/test_mm_processor_kwargs.py | 17 | mm processor kwargs | yes | monkeypatch | PORT-CPU | C11 |
| multimodal/llm/test_mm_cache_stats.py | 1 | mm processor cache | no | vllm_runner | DEFER-GPU | C11 |
| multimodal/llm/test_mm_cache_external_injection.py | 2 | mm cache injection | no | vllm_runner | DEFER-GPU | C11 |
| multimodal/llm/test_mm_embeds_only.py | 3 | mm embeds-only gen | no | vllm_runner | DEFER-GPU | C11 |
| multimodal/llm/test_chat.py | 1 | vision LLM offline | no | vllm_runner | DEFER-GPU | C11 |
| multimodal/openai/chat_completion/test_vision.py | 11 | vision chat API | no | RemoteOpenAIServer | DEFER-GPU | C11 |
| multimodal/openai/chat_completion/test_vision_embeds.py | 2 | vision embeds API | no | RemoteOpenAIServer | DEFER-GPU | C11 |
| multimodal/openai/chat_completion/test_video.py | 10 | video chat API | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| multimodal/openai/chat_completion/test_audio.py | 7 | audio chat API | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| multimodal/openai/chat_completion/test_audio_in_video.py | 3 | audio-in-video API | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| multimodal/openai/chat_completion/test_chat_completion_with_image_embeds.py | 1 | image-embeds API | no | RemoteOpenAIServer | DEFER-GPU | C11 |
| multimodal/openai/chat_completion/test_chat_completion_with_mixed_image_embeds.py | 2 | mixed image embeds | no | RemoteOpenAIServer | DEFER-GPU | C11 |
| multimodal/openai/chat_completion/test_chat_completion_with_mixed_audio_embeds.py | 2 | mixed audio embeds | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| multimodal/openai/chat_completion/test_default_mm_loras.py | 1 | mm LoRA serving | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C11 |
| multimodal/openai/responses/test_image.py | 3 | responses-API image | no | RemoteOpenAIServer | DEFER-GPU | C11 |
| speech_to_text/test_upload_size_limit.py | 7 | upload size guard | yes | none | PORT-CPU | C1 |
| speech_to_text/test_speech_to_text_cancellation.py | 3 | request cancellation | yes | none | PORT-CPU | C1 |
| speech_to_text/transcription/test_transcription_validation.py | 3 | STT validation server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| speech_to_text/transcription/test_transcription_validation_whisper.py | 16 | whisper validation | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| speech_to_text/transcription/test_transcription_inter_chunk_spacing.py | 14 | inter-chunk timing | yes | none | PORT-CPU | C1 |
| speech_to_text/transcription/test_chunk_timestamp_offset.py | 1 | chunk timestamp math | yes | none | PORT-CPU | C1 |
| speech_to_text/transcription/test_enable_force_include_usage.py | 1 | usage-flag serving | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| speech_to_text/transcription/test_qwen3_asr_sanitize_prompt.py | 3 | qwen3-asr sanitize | yes | none | SKIP-OTHERMODEL | C1 |
| speech_to_text/realtime/test_realtime_validation.py | 4 | websocket realtime | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C9 |
| speech_to_text/translation/test_translation_validation.py | 8 | translation validation | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| speech_to_text/correctness/test_transcription_api_correctness.py | 2 | STT correctness | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |
| cohere/test_protocol.py | 11 | cohere protocol schemas | yes | none | PORT-CPU | C1 |
| cohere/test_serving_conversion.py | 100 | cohere→openai conversion | yes | none | PORT-CPU | C1 |
| cohere/test_serving_streaming.py | 38 | cohere streaming logic | yes | none | PORT-CPU | C1 |
| cohere/test_cohere_chat_message.py | 9 | cohere chat message | yes | none | PORT-CPU | C1 |
| cohere/test_api_router.py | 28 | cohere API router gating | yes | monkeypatch | PORT-CPU | C1 |
| cohere/test_registry_and_args.py | 7 | cohere CLI/registry | yes | none | PORT-CPU | C1 |
| cohere/test_chat_v2.py | 9 | cohere chat v2 server | no | RemoteOpenAIServer | SKIP-OTHERMODEL | C1 |

Totals: 95 test files inventoried; 780 tests counted (pooling 49 files incl.
`__init__.py`/helpers — 37 test files, 267 tests; multimodal 21 incl. helpers —
15 test files, 66 tests; speech_to_text 17 incl. conftest/helpers — 12 test
files, 63 tests; cohere 8 — 7 test files, 230 tests).

## Notable PORT-CPU candidates

- **cohere/test_serving_conversion.py** (100 tests) + **test_serving_streaming.py** (38) — the largest pure-CPU protocol-validation trove in this slice; generic request-conversion/streaming-chunk patterns directly reusable for any alternate-API-surface validation on ds41rt (C1).
- **cohere/test_api_router.py** (28) — env-flag/SDK-presence gating and router registration logic; a template for testing feature-gated API surfaces without a server (C1).
- **pooling/embed/test_io_processor.py** (34) — exhaustive CPU coverage of input-processing edge cases (parametrized); the pattern transfers to ds41rt's chat-input preprocessing (C1).
- **pooling/embed/test_protocol.py** (19) + **multimodal/llm/test_mm_processor_kwargs.py** (17) — pydantic schema and mm-kwarg validation without a model; mm_processor_kwargs is the strongest C11 candidate here (input-side only, no vision encoder needed).
- **speech_to_text/transcription/test_transcription_inter_chunk_spacing.py** (14) — serverless serving-layer timing/validation logic; generic request-lifecycle patterns applicable to ds41rt's OpenAI serving (C1).
- **pooling/basic/test_tiling_engine.py** (4) + **speech_to_text/test_upload_size_limit.py** (7) — small, fully isolated CPU units (input chunking; upload-size guard) that port as-is.


<!-- ===== 11-vllm-tool-parsers.md ===== -->

# Inventory 11 — vLLM tool parsers + tool_use

Scope: `/home/turq/dev/upstream-test-inventory/vllm/tests/tool_parsers/` (45 py files) and `/home/turq/dev/upstream-test-inventory/vllm/tests/tool_use/` (13 py files incl. mistral/). `n` = `grep -c "def test_"`. DS41RT component map: C1 OpenAI API, C2 constrained, C12 tool-call parsing (primary). All tool_parsers tests run the parser objects directly (no engine) — the only CPU question is tokenizer provenance; all tool_use tests that touch a server spin up `RemoteOpenAIServer` (full model load → GPU).

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| tool_parsers/common_tests.py | 12 | Shared ToolParserTests mixin: no-tool/single/parallel/empty-args/surrounding-text/escaped/malformed/streaming-reconstruction matrix with xfail hooks | yes | gpt2 default tok via conftest | PORT-CPU | C12 |
| tool_parsers/conftest.py | 0 | `default_tokenizer` fixture = `openai-community/gpt2` AutoTokenizer | yes | gpt2 tok (HF download/cache) | PORT-CPU | C12 |
| tool_parsers/__init__.py | 0 | package marker | yes | — | PORT-CPU | C12 |
| tool_parsers/utils.py | 0 | StreamingToolReconstructor + token-delta split/run helpers used by all parser tests | yes | gpt2 tok | PORT-CPU | C12 |
| tool_parsers/test_utils.py | 128 | `vllm.tool_parsers.utils`: schema-type coercion, anyOf/oneOf/enum, NaN/Inf fallback, bracket/escape/string-recovery, reserved kwargs, AST repair — fully generic pure-Python | yes | none (pure py) | PORT-CPU | C12 |
| tool_parsers/test_structural_tag_registry.py | 29 | Structural-tag (xgrammar builtin) registry: named/required choice, format constraints, hermes/DSv4/kimi-k3 registration, invalid-rejection | yes | xgrammar lib; mock/gpt2 toks | PORT-CPU | C12, C2 |
| tool_parsers/test_hermes_tool_parser.py | 13 | Hermes XML streaming at stream_interval 1–8 (bug 19056 failure case), content+tool single chunk, non-streaming invalid JSON | yes | gpt2 tok | PORT-CPU | C12 |
| tool_parsers/test_llama3_json_tool_parser.py | 15 | Llama3 JSON prefix extraction incl. multiple JSON blobs w/ whitespace/surrounding text, deep nesting, quotes/brackets in strings, regex timeout | yes | HF llama-3 tok | PORT-CPU | C12 |
| tool_parsers/test_pythonic_tool_parser.py | 4 | Pythonic (signature-style) calls, large-step streaming, regex timeout | yes | gpt2 tok | PORT-CPU | C12 |
| tool_parsers/test_llama4_pythonic_tool_parser.py | 4 | Same pythonic matrix for llama4 variant | yes | gpt2 tok | PORT-CPU | C12 |
| tool_parsers/test_olmo3_tool_parser.py | 4 | Same pythonic matrix for olmo3 variant | yes | gpt2 tok | PORT-CPU | C12 |
| tool_parsers/test_jamba_tool_parser.py | 4 | Jamba XML-style calls, streaming incl. all-args-in-single-delta | yes | HF jamba tok | PORT-CPU | C12 |
| tool_parsers/test_internlm2_tool_parser.py | 2 | common-tests mixin + streaming args-in-single-delta | yes | gpt2 tok | PORT-CPU | C12 |
| tool_parsers/test_granite_tool_parser.py | 4 | Granite prefix formats, streaming parallel calls with batched deltas at chunk_size 2–5 | yes | gpt2 tok | PORT-CPU | C12 |
| tool_parsers/test_granite4_tool_parser.py | 1 | Complex chunked streaming across min/max chunk splits | yes | HF granite-4 tok | PORT-CPU | C12 |
| tool_parsers/test_granite_20b_fc_tool_parser.py | 1 | config-only common-tests wrapper | yes | gpt2 tok | SKIP-OTHERMODEL | C12 |
| tool_parsers/test_xlam_tool_parser.py | 7 | xlam list-structure extraction, streaming list structure, incremental streaming, non-ASCII | yes | HF xlam tok | PORT-CPU | C12 |
| tool_parsers/test_dots_tool_parser.py | 10 | XML invoke blocks: partial-marker buffering, malformed-block-doesn't-block-later-call, stray-end-marker, EOF flush, complete-JSON-before-end | yes | mock tok | PORT-CPU | C12 |
| tool_parsers/test_apertus_tool_parser.py | 19 | Mock-tok parser: incomplete call, missing suffix, MTP massive-chunk/burst/skip-catch-up, char-by-char, empty deltas | yes | mock tok | PORT-CPU | C12 |
| tool_parsers/test_kimi_k2_tool_parser.py | 26 | kimi-k2 marker parser: invalid-JSON-still-extracted, invalid id skip, truncated-no-end-marker, streaming==nonstreaming, marker suppression, stream-interval sweep | yes | HF kimi-k2 tok (trust_remote_code) | PORT-CPU | C12 |
| tool_parsers/test_kimi_k3_named_tool_choice.py | 2 | named tool choice allowed/rejected w/ vs w/o structural tag | yes | none obvious (registry-level) | PORT-CPU | C12, C2 |
| tool_parsers/test_deepseekv3_tool_parser.py | 1 | config-only common-tests wrapper | yes | gpt2 tok | SKIP-OTHERMODEL | C12 |
| tool_parsers/test_deepseekv31_tool_parser.py | 2 | DSv3.1 extract single/multiple tools | yes | gpt2/mock tok | PORT-CPU | C12 |
| tool_parsers/test_deepseekv32_tool_parser.py | 48 | DSv3.2 engine parser (DSML), mock tok, multi-char chunk realism, structural-tag registry interaction, skip_global_cleanup | yes | mock tok; xgrammar structural tags | PORT-CPU | C12, C2 |
| tool_parsers/test_deepseekv4_tool_parser.py | 11 | DSv4 engine parser: missing-wrapper recovery, incremental arg chunks, no-DSML-leak in streamed args, angle-bracket content, composed schema, structural tag | yes | mock tok; xgrammar structural tags | PORT-CPU | C12, C2 |
| tool_parsers/test_qwen3coder_tool_parser.py | 23 | qwen3coder XML parser: missing closing/opening tags (stream+non-stream), malformed XML no-gt, anyOf type conversion stream/non-stream, next-call-starts-in-close-delta, no-double-serialization, structural tag | yes | HF qwen3 tok; xgrammar | PORT-CPU | C12, C2 |
| tool_parsers/test_step3p5_tool_parser.py | 17 | step3.5 XML: missing-tag recovery stream/non-stream, mixed content + multiple calls (3 variants), multi-token chunk boundary, MTP variable chunks | yes | HF step3.5 tok | PORT-CPU | C12 |
| tool_parsers/test_step3_tool_parser.py | 1 | config-only common-tests wrapper | yes | gpt2 tok | SKIP-OTHERMODEL | C12 |
| tool_parsers/test_minicpm5xml_tool_parser.py | 36 | minicpm5 XML: CDATA multiline, collapsed function/param tags, tokenizer-space marker, streaming partial chunks/incremental args, unknown-tool preserved, duplicate-param invalid | yes | gpt2 tok | PORT-CPU | C12 |
| tool_parsers/test_minimax_m2_tool_parser.py | 21 | minimax_m2: incremental chunks, two-invokes single delta, anyOf nullable, none/nil preservation, whitespace preserved | yes | fake tok stub | PORT-CPU | C12 |
| tool_parsers/test_minimax_m3_tool_parser.py | 6 | minimax_m3: nested calls, multiple calls, streaming variants | yes | fake tok stub | PORT-CPU | C12 |
| tool_parsers/test_glm4_moe_tool_parser.py | 6 | GLM-4.5 shared glm47 parser: newline format extract/multiple, schema coercion, streaming | yes | mock tok stub | PORT-CPU | C12 |
| tool_parsers/test_glm47_moe_tool_parser.py | 13 | glm47 MoE: zero-arg inline/newline, args w/ newlines, whitespace preserved, multiple, namespace round-trip to responses output | yes | HF glm47 tok | PORT-CPU | C12 |
| tool_parsers/test_hy_v3_tool_parser.py | 14 | hy_v3: zero-arg/args streaming+nonstreaming, all-in-one-delta | yes | mock tok | PORT-CPU | C12 |
| tool_parsers/test_hunyuan_a13b_tool_parser.py | 3 | hunyuan extract, char-split streaming, non-ASCII | yes | mock tok | PORT-CPU | C12 |
| tool_parsers/test_gigachat3_tool_parser.py | 3 | gigachat3 streaming/non-streaming, large-step streaming | yes | gpt2 tok | PORT-CPU | C12 |
| tool_parsers/test_ernie45_moe_tool_parser.py | 3 | ernie4.5: no-tools, extract, streaming incremental | yes | HF ernie4.5 tok (trust_remote_code) | PORT-CPU | C12 |
| tool_parsers/test_functiongemma_tool_parser.py | 15 | functiongemma XML: text-before, empty args, buffering flush, skip_special_tokens toggling | yes | mock tok | PORT-CPU | C12 |
| tool_parsers/test_gemma4_tool_parser.py | 63 | gemma4 engine parser: partial-value withholding (empty value, trailing-dot float, malformed array), stray brackets, streaming multi-arg/no-extra-brace, namespaced names | yes | mock tok | PORT-CPU | C12 |
| tool_parsers/test_lfm2_tool_parser.py | 13 | lfm2: full-block-in-single-delta (leading/trailing), echo-body not leaked, char-by-char multi dict/list, dotted names, skip_special_tokens, regex timeout | yes | HF LFM2.5 tok | PORT-CPU | C12 |
| tool_parsers/test_longcat_tool_parser.py | 1 | config-only common-tests wrapper | yes | gpt2 tok | SKIP-OTHERMODEL | C12 |
| tool_parsers/test_phi4mini_tool_parser.py | 1 | config-only common-tests wrapper | yes | gpt2 tok | SKIP-OTHERMODEL | C12 |
| tool_parsers/test_poolside_v1_tool_parser.py | 13 | poolside v1: required/named skip structured outputs, whitespace preservation, newline-after-name stream/non-stream, responses logprobs empty delta | yes | stub tok | PORT-CPU | C12 |
| tool_parsers/test_k2_horizon_tool_parser.py | 11 | k2-horizon: request-level format override, fail-closed mismatched formats, malformed→content, char-split streaming, EOF flush as content | yes | mock tok | PORT-CPU | C12 |
| tool_parsers/test_rust_tool_parser.py | 8 | Rust-tool-parser adapter (importorskip `vllm._rust_tool_parser`): complete/streaming multi-call, empty-delta ignore, model-id preference/fallback, adjust_request opacity | yes | rust ext build; mock tok | PORT-CPU | C12 |
| tool_use/conftest.py | 0 | session fixtures: `--models/--extended` opts, HF snapshot_download, RemoteOpenAIServer per CONFIGS | no | server + HF full model | DEFER-GPU | C1 |
| tool_use/utils.py | 0 | server ARGS/CONFIGS table, system-prompt patch helpers | no | server | DEFER-GPU | C1 |
| tool_use/__init__.py | 0 | package marker | yes | — | SKIP-OTHERMODEL | C1 |
| tool_use/test_chat_completions.py | 3 | end-to-end chat w/ and w/o tools vs OpenAI client | no | server+model | DEFER-GPU | C1 |
| tool_use/test_tool_calls.py | 2 | e2e tool_call_and_choice (stream compare vs base client), tool_call_with_results | no | server+model | DEFER-GPU | C1 |
| tool_use/test_parallel_tool_calls.py | 3 | e2e parallel tool calls, with results, parallel_tool_calls=false | no | server+model | DEFER-GPU | C1 |
| tool_use/test_tool_choice_required.py | 8 | required tool_choice validation vs server | no | server+model | DEFER-GPU | C1 |
| tool_use/test_chat_completion_request_validations.py | 8 | request-validation error paths (no server) | yes | none | PORT-CPU | C1 |
| tool_use/test_responses_request_validations.py | 17 | responses-API request validation (no server) | yes | none | PORT-CPU | C1 |
| tool_use/test_gemma4_responses_adjust_request.py | 8 | adjust_request skip_special_tokens/required/named structured-output behavior, stub tok | yes | stub tok | PORT-CPU | C12, C1 |
| tool_use/test_kimi_k3_tool_parser.py | 25 | pure kimi_k3 parser+reasoning delegation: delta-level extract, typed args, tool calls preserved after reasoning, adjust_request | yes | none | PORT-CPU | C12 |
| tool_use/test_muse_glimmer.py | 21 | muse_glimmer parser (mock tok): parallel calls across EOM boundaries, reasoning→toolcall handoff, charwise/bigchunk streaming, truncated CoT, namespace handling | yes | mock tok | PORT-CPU | C12 |
| tool_use/test_muse_glimmer_parse_delta.py | 5 | parse_delta driven token-by-token with real MuseGlimmer tok: reasoning-then-answer, truncated CoT, reasoning suppression | yes | HF MuseGlimmer ckpt (trust_remote_code; skips if absent) | PORT-CPU | C12 |
| tool_use/mistral/conftest.py | 0 | mistral server fixtures (RemoteOpenAIServer per config) | no | server+model | DEFER-GPU | C1 |
| tool_use/mistral/utils.py | 0 | mistral server ARGS/CONFIGS | no | server | DEFER-GPU | C1 |
| tool_use/mistral/__init__.py | 0 | package marker | yes | — | SKIP-OTHERMODEL | C1 |
| tool_use/mistral/test_mistral_tool_calls.py | 6 | e2e mistral tool_choice auto/required/none, with results, parallel | no | server+mistral model | DEFER-GPU | C1 |

Totals: tool_parsers 45 files (41 test/support + package files), 723 `def test_` across them; tool_use 13 files, 106 tests, of which 6 files/84 tests need a live server (GPU).

## Notable PORT-CPU candidates

- `tool_parsers/test_hermes_tool_parser.py` — the deepest streaming-delta sweep (stream_interval 1–8, content+tool in one chunk, historical bug 19056 regression); hermes XML is the closest vLLM analog to ds41rt's xgrammar_v41_tools surface.
- `tool_parsers/test_deepseekv4_tool_parser.py` — DeepSeek-lineage DSML parser: incremental argument chunks, missing-`tool_calls`-wrapper recovery, no-tag-leak in streamed args; directly maps to ds41rt's native format.
- `tool_parsers/test_deepseekv32_tool_parser.py` — 48 DSv3.2 cases on mock tokenizer with realistic multi-char chunks plus structural-tag registry interaction (C12+C2 overlap).
- `tool_parsers/test_structural_tag_registry.py` — xgrammar structural-tag registry: named/required choice, format constraints, registration of hermes/DSv4/kimi-k3 builtins; the C2↔C12 seam ds41rt lives on.
- `tool_parsers/test_utils.py` — 128 pure-Python tests of shared arg-coercion/AST-repair utilities (anyOf, NaN fallback, bracket/escape recovery); model-agnostic, port wholesale.
- `tool_parsers/test_granite_tool_parser.py` — streaming **parallel** calls with batched deltas at chunk sizes 2–5; compact parallel-call delta-assembly reference.
- `tool_parsers/test_dots_tool_parser.py` — malformed-block-doesn't-block-later-valid-call, stray-end-marker, EOF-flush recovery patterns in one small file.
- `tool_parsers/test_kimi_k2_tool_parser.py` — truncated-tool-call-no-end-marker, invalid-JSON-still-extracted, streaming==nonstreaming equivalence, marker suppression.
- `tool_use/test_muse_glimmer.py` + `test_muse_glimmer_parse_delta.py` — parallel calls across EOM boundaries and token-by-token `parse_delta` assembly with a real tokenizer; the closest upstream replica of serving-time delta feeding.
- `tool_parsers/test_qwen3coder_tool_parser.py` — systematic missing-opening/missing-closing-tag recovery in both streaming and non-streaming modes, plus next-call-starts-in-close-delta sequencing.


<!-- ===== 12-vllm-reasoning-parser.md ===== -->

# 12 — vLLM reasoning / parser / renderers test inventory

Census of `vllm/tests/reasoning/` (21 test files), `vllm/tests/parser/` (27 test files incl. `cohere/`, `engine/`, `mistral/` subdirs), `vllm/tests/renderers/` (15 test files incl. `inputs/`). Counts via `grep -c "def test_"`. All files are pure parsing / template rendering — no GPU, no platform skips, no `torch.cuda` anywhere. "HF tok dl" = downloads a small tokenizer/config from HF Hub at runtime (CPU, network needed).

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| reasoning/test_base_thinking_reasoning_parser.py | 24 | BaseThinkingReasoningParser base class: content/delta extraction, split reasoning | yes (HF tok dl) | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B tok | PORT-CPU | C1 |
| reasoning/test_deepseekr1_reasoning_parser.py | 1 | DeepSeekR1ReasoningParser registered & `<think>` parse | yes (HF tok dl) | DeepSeek-R1-Distill-Qwen-1.5B tok | PORT-CPU | C1 |
| reasoning/test_deepseekv3_reasoning_parser.py | 3 | DeepSeek-V3 parser streaming + ChatCompletionRequest | yes (HF tok dl) | deepseek-ai/DeepSeek-V3.1 tok | PORT-CPU | C1 |
| reasoning/test_ernie45_reasoning_parser.py | 1 | Ernie45 reasoning parser | yes (HF tok dl) | baidu/ERNIE-4.5-21B-A3B-Thinking tok | PORT-CPU | C1 |
| reasoning/test_gemma4_reasoning_parser.py | 3 | Gemma4 parser incl. builtin tool turns | yes | none | PORT-CPU | C1 |
| reasoning/test_glm4_moe_reasoning_parser.py | 2 | GLM-4.7 reasoning parser (incl. VLM path) | yes (HF tok dl) | zai-org/GLM-4.7 tok | PORT-CPU | C1 |
| reasoning/test_gptoss_reasoning_parser.py | 1 | GptOssReasoningParser | yes | none | PORT-CPU | C1 |
| reasoning/test_granite_reasoning_parser.py | 2 | Granite reasoning parser | yes (HF tok dl) | facebook/opt-125m tok | PORT-CPU | C1 |
| reasoning/test_holo2_reasoning_parser.py | 4 | Holo2 parser; cross-checks DeepSeekR1 delimiter reuse | yes (HF tok dl) | HCompany/Holo2-4B tok | PORT-CPU | C1 |
| reasoning/test_hunyuan_reasoning_parser.py | 1 | Hunyuan reasoning parser | yes (HF tok dl) | tencent/Hunyuan-A13B-Instruct tok (trust_remote_code) | PORT-CPU | C1 |
| reasoning/test_hy_v3_reasoning_parser.py | 4 | Hunyuan-V3 (hy_v3) parser | yes | none | PORT-CPU | C1 |
| reasoning/test_k2_horizon_reasoning_parser.py | 8 | K2-Horizon parser via ParserManager | yes | none | PORT-CPU | C1 |
| reasoning/test_kimi_k2_reasoning_parser.py | 6 | Kimi-K2 parser, DeltaMessage streaming | yes | none | PORT-CPU | C1 |
| reasoning/test_kimi_k3_reasoning_parser.py | 18 | Kimi-K3 parser: boundary states, stream deltas | yes | none | PORT-CPU | C1 |
| reasoning/test_minimax_m2_append_reasoning_parser.py | 1 | MiniMax-M2 append-mode parser | yes (HF tok dl) | MiniMaxAI/MiniMax-M2 tok | PORT-CPU | C1 |
| reasoning/test_minimax_m2_reasoning_parser.py | 1 | MiniMax-M2 parser | yes (HF tok dl) | MiniMaxAI/MiniMax-M2 tok | PORT-CPU | C1 |
| reasoning/test_minimax_m3_reasoning_parser.py | 22 | MiniMax-M3 parser: ChatCompletionRequest + stream | yes | none | PORT-CPU | C1 |
| reasoning/test_nemotron_v3_reasoning_parser.py | 8 | Nemotron-V3 via DelegatingParser | yes | none | PORT-CPU | C1 |
| reasoning/test_olmo3_reasoning_parser.py | 1 | OLMo-3 thinking parser | yes (HF tok dl) | allenai/Olmo-3-7B-Think tok | PORT-CPU | C1 |
| reasoning/test_qwen3_reasoning_parser.py | 3 | Qwen3 parser, token-boundary cases | yes (HF tok dl) | Qwen/Qwen3-0.6B, Qwen3-4B-Thinking-2507 (…, tok params) | PORT-CPU | C1 |
| reasoning/test_step3p5_reasoning_parser.py | 1 | Step-3.5-Flash parser | yes (HF tok dl) | stepfun-ai/Step-3.5-Flash tok | PORT-CPU | C1 |
| parser/test_harmony.py | 35 | OpenAI harmony format: structured outputs / function call parsing | yes | none | PORT-CPU | C1 |
| parser/test_include_reasoning.py | 16 | include_reasoning flag → reasoning_content deltas | yes | none | PORT-CPU | C1 |
| parser/test_parse.py | 17 | tool-call parse from ChatCompletion/Responses requests | yes | none | PORT-CPU | C1 |
| parser/test_streaming.py | 20 | streaming DeltaMessage tool-call accumulation | yes | none | PORT-CPU | C1 |
| parser/cohere/test_citations.py | 5 | Cohere Command parser: citations | yes | none | PORT-CPU | C1 |
| parser/cohere/test_reasoning.py | 7 | Cohere reasoning parser via ParserManager | yes | none | PORT-CPU | C1 |
| parser/cohere/test_structural_tags.py | 17 | Cohere structural tags (tools-as-tags) parsing | yes | none | PORT-CPU | C1 |
| parser/cohere/test_tool_calls.py | 8 | Cohere tool-call emission | yes | none | PORT-CPU | C12 |
| parser/engine/test_deepseek_v32.py | 17 | DeepSeek-V3.2 parser engine adapter | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_deepseek_v41.py | 4 | DeepSeek-V4.1 parser registration/config | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_deepseek_v4.py | 68 | DeepSeek-V4 parser: events, tool calls, streaming | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_delegating_replay.py | 2 | DelegatingParser replay over recorded traces | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_engine.py | 44 | parser-engine internals: incremental lexer, events | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_gemma4_streaming_reasoning.py | 58 | Gemma4 streaming reasoning via engine | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_inkling.py | 64 | Inkling structured-output parser (constrained fmt) | yes | none (mock tokenizer) | PORT-CPU | C1/C2 |
| parser/engine/test_ling3.py | 6 | Ling3 parser; reasoning + tool parser registration | yes | none (mock tokenizer) | PORT-CPU | C1/C12 |
| parser/engine/test_minimax_m2.py | 14 | MiniMax-M2 engine parser | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_nemotron_v3.py | 14 | Nemotron-V3 engine parser | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_parser_engine.py | 97 | ParserEngine core: state machine, semantic events | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_qwen3.py | 51 | Qwen3 engine parser: tags, tool calls | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_qwen3_reasoning.py | 61 | Qwen3 reasoning state machine via DelegatingParser | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_replay.py | 11 | replay harness against registered adapters | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_seed_oss.py | 11 | SeedOss parser registered adapters | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_token_id_scanner.py | 30 | token-id scanner / streaming engine | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/engine/test_ufffd_reasoning_transition.py | 3 | U+FFFD boundary reasoning transition | yes | none (mock tokenizer) | PORT-CPU | C1 |
| parser/mistral/test_reasoning.py | 7 | Mistral reasoning parser (mistral tokenizer mode) | yes (HF tok dl) | mistral tokenizer | PORT-CPU | C1 |
| parser/mistral/test_tool_calls.py | 42 | Mistral tool-call parsing, streaming | yes | none | PORT-CPU | C12 |
| renderers/test_chat_utils_prompt_embeds.py | 14 | chat_utils resolve inputs, prompt-embeds requests | yes (HF tok dl) | AutoTokenizer from request.params | PORT-CPU | C4 |
| renderers/test_cohere.py | 91 | Cohere renderer: chat params, tokenize, errors | yes | none | PORT-CPU | C4 |
| renderers/test_completions.py | 28 | completions protocol → renderer params | yes | openai-community/gpt2 tok | PORT-CPU | C4 |
| renderers/test_gemma4_chat_template.py | 16 | gemma4 tool chat template jinja rendering | yes | none (jinja only, no vllm import) | PORT-CPU | C4 |
| renderers/test_hf.py | 37 | HF renderer: template resolution, tool calling | yes (HF tok dl) | Qwen/Qwen2-VL-2B-Instruct, NousResearch/Hermes-3-Llama-3.1-8B tok | PORT-CPU | C4 |
| renderers/test_inkling.py | 26 | Inkling renderer + encoding | yes | none | PORT-CPU | C4/C2 |
| renderers/test_kimi_k3.py | 16 | Kimi-K3 renderer param handling, validation | yes | none | PORT-CPU | C4 |
| renderers/test_mistral.py | 2 | Mistral renderer + safe_apply_chat_template | yes (HF tok dl) | mistralai/Mistral-7B-Instruct-v0.3 tok | PORT-CPU | C4 |
| renderers/test_multimodal_hashes.py | 1 | multimodal hash caching across preprocess | yes (HF tok dl + asset dl) | llava-hf/llava-onevision-qwen2-0.5b-ov-hf tok | PORT-CPU | C11 |
| renderers/test_process_multi_modal_uuids.py | 6 | multi-modal UUID processing in renderer inputs | yes (HF tok dl + asset dl) | llava-hf/llava-onevision-qwen2-0.5b-ov-hf tok | PORT-CPU | C11 |
| renderers/test_sparse_tensor_concurrent_race.py | 11 | sparse embedding media thread-safety | yes (HF tok dl) | facebook/opt-125m tok | PORT-CPU | C11 |
| renderers/test_sparse_tensor_validation.py | 34 | sparse tensor embedding validation, errors | yes | none | PORT-CPU | C11 |
| renderers/test_token_offsets.py | 12 | TokenizeParams → token offset mapping | yes (HF tok dl) | openai-community/gpt2 tok | PORT-CPU | C4 |
| renderers/test_warmup.py | 24 | renderer warmup round-trip | yes | none | PORT-CPU | C4 |
| renderers/inputs/test_preprocess.py | 9 | renderer input preprocessing | yes | none | PORT-CPU | C4 |

Non-test helpers (not rows): `reasoning/utils.py`, `parser/engine/{conftest,replay_harness,trace_builder,streaming_helpers}.py`, `parser/cohere/{conftest,utils}.py`, `__init__.py` files. Engine tests are hermetic — `conftest.make_mock_tokenizer()` mocks encode/decode, no HF downloads.

## Notable PORT-CPU candidates

- `parser/engine/test_parser_engine.py` (97) — parser state-machine core; fully hermetic mock tokenizer, biggest single CPU win.
- `parser/engine/test_deepseek_v4.py` (68) — DeepSeek-V4 delimiter/tool-call semantics; direct ds41rt lineage, no downloads.
- `parser/engine/test_qwen3_reasoning.py` (61) + `test_gemma4_streaming_reasoning.py` (58) — streaming reasoning state machines, hermetic.
- `parser/engine/test_inkling.py` (64) — constrained-format structured output parser (C1/C2 crossover).
- `reasoning/test_base_thinking_reasoning_parser.py` (24) — base class every reasoning parser (incl. ds41rt's) inherits.
- `reasoning/test_deepseekr1_reasoning_parser.py` (1) + `reasoning/test_minimax_m3_reasoning_parser.py` (22) + `reasoning/test_kimi_k3_reasoning_parser.py` (18) — `<think>`-delimiter boundary suites; direct V4.1-thinking analogues, no HF pins.
- `parser/test_streaming.py` (20) + `parser/test_parse.py` (17) — OpenAI protocol delta/tool-call accumulation, zero pins.
- `parser/mistral/test_tool_calls.py` (42) — largest tool-parsing suite (C12).
- `renderers/test_gemma4_chat_template.py` (16) — pure-jinja template test, no vllm import at all; cheapest possible port.
- `renderers/test_hf.py` (37) — chat-template resolution/rendering incl. tool calling; small public tokenizer pins only.


<!-- ===== 13-vllm-kernels-attn-mm.md ===== -->

# Inventory 13 — vLLM kernels: ATTENTION + MULTIMODAL/MISC-OP kernel tests

Scope: `vllm/tests/kernels/attention/`, attention-adjacent loose files in `kernels/`, vision (mm) tests, and all other kernels subdirs except quantisation/moe/mamba/activation (covered by another agent).

Notes on scope:
- `kernels/mm/` **does not exist** in this tree. Multimodal preprocessing tests live at `tests/multimodal/` (outside kernels/, not inventoried here); vision-encoder kernels are `kernels/core/test_vit_*.py` (→ C11).
- **Skipped (other agent owns):** `kernels/quantization/`, `kernels/moe/`, `kernels/mamba/`, plus activation-scoped files `kernels/core/test_activation.py` (10 tests), `kernels/core/test_cpu_activation.py` (3), `kernels/ir/test_activation.py` (10), `kernels/test_fused_quant_activation.py` (5).
- Helper files (no tests, not rows): `kernels/allclose_default.py`, `conftest.py`, `utils.py`, `quant_utils.py`, `attention/conftest.py`, `helion/{helpers,utils}.py`, `helion/test_utils.py` (2 tests, counted in helion row), `turboquant/__init__.py`, `scripts/check_no_tilelang_hijack.py`.
- `n` = `grep -c "def test_"` summed over the row.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| `attention/` — selector & backend choice: test_attention_selector.py, test_rocm_attention_selector.py, test_use_trtllm_attention.py, test_amx_mla.py | 53 | attention backend registry/selection; AMX CPU MLA | partial (amx_mla is CPU-only+AMX; selector tests are mock/platform-based) | CUDA/ROCm mocks; AMX tile CPU | DEFER-GPU (amx file is CPU-farm, SKIP on GPU rig) | C5 |
| `attention/` — Flash/FlashMLA/TRT-LLM/cutlass family: test_flash_attn.py, test_flashinfer.py, test_flashinfer_mla_decode.py, test_flashinfer_trtllm_attention.py, test_flashmla.py, test_flashmla_sparse.py, test_cascade_flash_attn.py, test_cutlass_mla_decode.py, test_deepgemm_attention.py, test_prefix_prefill.py, test_mha_attn.py, test_trtllm_kvfp8_dequant.py, test_mixed_causal_attn.py, test_attention.py | 59 | vendor attention kernels (FA, FlashMLA, TRT-LLM, DeepGEMM) | no | CUDA, SM90/100, flashinfer, flashmla, trtllm | DEFER-GPU | C5 |
| `attention/` — Triton attention: test_triton_decode_attention.py, test_triton_prefill_attention.py, test_triton_unified_attention.py, test_triton_unified_attention_diffkv.py, test_merge_attn_states.py, test_lightning_attn.py, test_pack_unpack_triton.py | 27 | triton prefill/decode/unified attention, merge states | no | Triton, CUDA/ROCm | DEFER-GPU | C5 |
| `attention/` — KV cache ops: test_cache.py | 18 | KV cache copy/reshape, MLA cache, fp8 cache | partial (cpu↔cuda copy cases, platform-gated) | CUDA primary, triton | DEFER-GPU | C5 |
| `attention/` — CPU/MLA/model-specific: test_cpu_attn.py, test_mla_decode_cpu.py, test_mla_cross_layer_kernel_equivalence.py, test_kimi_k3_mla_fused_epilogue.py, test_kimi_k3_mla_key_concat_kv_cache.py, test_minimax_m3.py, test_minimax_m3_msa_cutlass_sparse_decode.py, test_xpu_mla_sparse.py | 77 | CPU attn backend, MLA decode (CPU ref), Minimax/Kimi MLA kernels | partial (test_cpu_attn 15 + test_mla_decode_cpu 1 are CPU-only/pure-torch ref) | CPU for 2 files; CUDA/ROCm/XPU for rest | PORT-CPU (test_cpu_attn, test_mla_decode_cpu); DEFER-GPU rest | C5 |
| `attention/` — ROCm aiter: test_rocm_aiter_candidate_mask.py, test_rocm_aiter_fa.py, test_rocm_aiter_mla_causal_verify_mask.py, test_rocm_aiter_mla_decode_metadata.py, test_rocm_aiter_mla_decode.py, test_rocm_aiter_mla_fp8_prefill.py, test_rocm_aiter_mla_fp8_support.py, test_rocm_aiter_mla_head_padding.py, test_rocm_aiter_mla_op_registration.py, test_rocm_aiter_mla_sink.py, test_rocm_aiter_mla_sparse_metadata_sync.py, test_rocm_aiter_unified_attn.py | 68 | AMD aiter MLA/FA attention | no | ROCm, aiter | SKIP-OTHERMODEL (AMD-only) | C5 |
| `kernels/core/` (excl. activation-owned): test_apply_rotary_emb.py, test_batched_weight_rms_norm.py, test_fused_allreduce_gemma_rms_norm.py, test_fused_embed_norm.py, test_fused_qk_norm_rope.py, test_fused_q_kv_rmsnorm.py, test_fused_quant_layernorm.py, test_fused_rms_norm_gated.py, test_fused_silu_mul_block_quant.py, test_layernorm.py, test_minimax_reduce_rms.py, test_mrope.py, test_opcheck.py, test_pos_encoding.py, test_rocm_aiter_ops.py, test_rocm_misc_ops.py, test_rotary_embedding_mla_cache_fused.py, test_rotary_embedding.py, test_uva.py, test_vit_bilinear_pos_embed.py, test_vit_fp8_attn.py, test_vit_fp8_quant.py, test_vit_fp8_scaling.py, test_vocab_parallel_embedding.py | 81 | core fused ops: rope/norm/pos-enc, UVA, ViT (C11) | partial (vit_bilinear_pos_embed compares triton vs pure-torch native, but triton side needs GPU) | CUDA/triton; ROCm files; UVA pinned mem | DEFER-GPU (vit tests → C11, GPU-bound) | C5/C11 |
| `kernels/helion/` | 128 | Helion kernel availability/registry/config + helion norm/quant kernels | partial (~75 pure-python config/registry/case-key tests CPU-portable; ~53 kernel tests GPU) | helion pkg, CUDA | PORT-CPU (test_case_key, test_config_manager, test_register, test_pattern_matching, test_utils, test_autotune, test_benchmark_script); DEFER-GPU (kernel tests) | C5 |
| `kernels/ir/` (excl. test_activation.py): test_ir_ops.py, test_layernorm.py | 14 | IR op infra meta-test + IR layernorm | partial (test_ir_ops is pure-python meta-test) | kernel backend | PORT-CPU (test_ir_ops); DEFER-GPU (test_layernorm) | C5 |
| `kernels/turboquant/`: test_flydsl_turboquant_decode.py | 1 | TurboQuant decode kernel | no | flydsl | DEFER-GPU (quant-adjacent, C10 overlap) | C10 |
| `kernels/test_apply_repetition_penalties.py` | 2 | sampler repetition-penalty kernel | no | CUDA | DEFER-GPU | C3 |
| `kernels/test_awq_int4_to_int8.py` | 6 | AWQ int4→int8 dequant | partial (is_cpu skip → CPU variant exists) | CUDA + CPU oneDNN paths | DEFER-GPU (quant-adjacent, C10 overlap) | C10 |
| `kernels/test_bf16_skinny_gemm.py` | 45 | bf16 skinny GEMM | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_bf16x3_router_gemm_cutedsl.py` | 1 | router GEMM (cutedsl) | no | CUDA SM100, cutedsl | DEFER-GPU | C5 |
| `kernels/test_cache_kernels.py` | 1 | KV cache kernel | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_compressor_kv_cache.py` | 18 | compressed KV cache | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_concat_mla_q.py` | 6 | MLA q concat | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_cp_gather_fp8.py` | 6 | context-parallel fp8 gather | no | CUDA | DEFER-GPU | C9 |
| `kernels/test_deepseek_v4_cpu_kernels.py` | 16 | DSv4 CPU port parity (sgl-kernels CPU ops vs eager torch ref) | yes (CPU-only, self-contained refs) | CPU | PORT-CPU | C5 |
| `kernels/test_engram.py` | 15 | Engram n-gram cache kernels | no | CUDA (one CPU-side mask util) | DEFER-GPU | C6-adj (spec-decode ngram; C5/C6) |
| `kernels/test_fla_layernorm_guard.py` | 9 | FLA (linear-attn) layernorm guard | no | CUDA triton | DEFER-GPU (mamba-adjacent) | C5 |
| `kernels/test_flashinfer_bf16_gemm.py` | 1 | flashinfer bf16 GEMM | no | CUDA, flashinfer | DEFER-GPU | C5 |
| `kernels/test_flex_attention.py` | 10 | FlexAttention backend integration | no | torch ≥2.7, CUDA | DEFER-GPU | C5 |
| `kernels/test_fp32_router_gemm.py` | 6 | fp32 router GEMM | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_fused_deepseek_v32_norm_rope.py` | 13 | DSv3.2 fused norm+rope | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | 8 | DSv4 fused qnorm/rope/kv-insert (GPU counterpart of CPU parity tests) | no (GPU file; CPU ops exercised in test_deepseek_v4_cpu_kernels) | CUDA | DEFER-GPU | C5 |
| `kernels/test_fused_gdn_post_conv.py` | 4 | gated conv post-op | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_fused_indexer_q_rope_quant.py` | 5 | indexer q-rope quant | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_fused_inv_rope_fp8_quant.py` | 12 | inverse rope + fp8 quant | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py` | 4 | Minimax M3 fused qknorm/rope/kv-insert | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_fused_qk_norm_rope_gate.py` | 1 | qk-norm rope gate | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_fused_recurrent_packed_decode.py` | 3 | fused recurrent packed decode | no | CUDA | DEFER-GPU (mamba-adjacent) | C5 |
| `kernels/test_fused_sigmoid_gating_delta_rule.py` | 2 | sigmoid gating delta-rule | no | CUDA | DEFER-GPU (mamba-adjacent) | C5 |
| `kernels/test_gate_linear_rocm_dispatch.py` | 10 | ROCm gate-linear dispatch | no | ROCm | SKIP-OTHERMODEL | C5 |
| `kernels/test_kimi_k3_gemm_rs_ar.py` | 1 | Kimi K3 GEMM reduce-scatter/all-reduce | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_kpool_decode_update_batched.py` | 6 | kpool decode update | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_ll_bf16_gemm.py` | 35 | lightllm bf16 GEMM | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_mhc_jit_warmup.py` | 5 | MHC (multi-head-compression) JIT warmup | no | CUDA, tilelang | DEFER-GPU | C5 |
| `kernels/test_mhc_kernels.py` | 27 | MHC fused pre/post ops | no | CUDA, tilelang/triton | DEFER-GPU | C5 |
| `kernels/test_mhc_tilelang_jit.py` | 2 | MHC tilelang JIT | no | CUDA, tilelang | DEFER-GPU | C5 |
| `kernels/test_minimax_m3_amd_ops.py` | 13 | Minimax M3 AMD ops | no | ROCm | SKIP-OTHERMODEL | C5 |
| `kernels/test_minimax_m3_sparse_attn_fp8_scale.py` | 2 | Minimax M3 sparse attn fp8 scale | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_onednn.py` | 2 | oneDNN CPU quant ops | yes (CPU-only module skip) | CPU, oneDNN | PORT-CPU | C10 |
| `kernels/test_relu2_fp8_quant.py` | 7 | relu² + fp8 quant | no | CUDA | DEFER-GPU (quant-adjacent, C10 overlap) | C10 |
| `kernels/test_rocm_fp32_router_gemm.py` | 7 | ROCm fp32 router GEMM | no | ROCm | SKIP-OTHERMODEL | C5 |
| `kernels/test_shuffle_rows.py` | 10 | row permutation kernel | no | CUDA | DEFER-GPU | C5 |
| `kernels/test_top_k_per_row.py` | 23 | radix/DeepSelect top-k | no | CUDA SM100/SM103 | DEFER-GPU | C3-adj (sampler top-k) |

## Notable PORT-CPU candidates

- **`kernels/test_deepseek_v4_cpu_kernels.py` (16 tests)** — strongest candidate: CPU-only module testing DSv4 CPU-port sgl-kernels ops against self-contained eager-torch references; directly relevant to ds41rt CPU lanes.
- **`attention/test_cpu_attn.py` (15 tests)** — CPU-only attention backend tests (guarded `is_cpu()`), incl. AMX path; pure CPU-farm work.
- **`attention/test_mla_decode_cpu.py` (1 test)** — CPU-only, pure-torch `ref_mla` reference vs CPU op; small and portable.
- **`kernels/test_onednn.py` (2 tests)** — CPU-only oneDNN quant ops; useful for C10 CPU coverage.
- **`kernels/helion/` pure-python layer (~75 tests: test_register 37, test_config_manager 22, test_case_key 7, test_pattern_matching 2, test_autotune 1, test_benchmark_script 4, test_utils 2)** — no GPU kernels involved in these files; portable if the helion package installs on CPU.
- **`kernels/ir/test_ir_ops.py` (1 test)** — pure-python registry meta-test; trivially portable.

Everything else is GPU/ROCm/SM100-pinned (DEFER-GPU) or AMD-only (SKIP-OTHERMODEL). `kernels/mm/` does not exist; the C11 vision-encoder kernel tests are `kernels/core/test_vit_*.py` (14 tests, triton-vs-native, GPU-bound) and multimodal preprocessing tests live outside kernels/ at `tests/multimodal/` (not inventoried here).


<!-- ===== 14-vllm-kernels-quant-moe.md ===== -->

# vLLM kernels — QUANT / MoE / MAMBA / ACTIVATION / NORM inventory (slice 14)

Scope: `tests/kernels/quantization/`, `quant_utils.py`, `moe/`, `mamba/`, `turboquant/`, `core/` (activation/norm/rotary/embed/ViT-fp8), `ir/`, `helion/`, and matching loose files in `kernels/` root. Attention and mm dirs excluded (owned by another agent). Counts are `grep -c 'def test_'` summed per row. Layout note: there is no `quant_utils/` subdir — the dir is `quantization/` plus a helper module `quant_utils.py`.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| `kernels/quantization/` (test_aiter_hipb_mm_linear_kernel, test_allspark_gemm, test_awq, test_awq_triton, test_block_fp8, test_cpu_fp8_scaled_mm, test_cutlass_scaled_mm, test_cutlass_w4a8, test_cutlass_w4a8_moe, test_flashinfer_mxfp8_trtllm, test_flashinfer_nvfp4_scaled_mm, test_flashinfer_scaled_mm, test_fp8_min_max_helper, test_fp8_quant, test_fp8_quant_group, test_gptq, test_hadacore, test_int4_emulation_moe, test_int8_kernel, test_int8_quant, test_machete_mm, test_marlin_gemm, test_marlin_tile_padding, test_mxfp4_kernel_selection, test_mxfp4_qutlass, test_mxfp4_triton_ep, test_mxfp6_kernel_selection, test_nvfp4_emulation, test_nvfp4_kernel_selection, test_nvfp4_quant, test_nvfp4_qutlass, test_nvfp4_scaled_mm, test_per_token_group_quant, test_quantized_embedding, test_quant_op_schema, test_rdna3_compile_guards, test_rdna3_moe_w4a16, test_rdna3_w4a16, test_rdna3_w4a16_selection, test_rdna_hybrid_w4a16, test_rocm_aiter_grouped_quant, test_rocm_compressed_tensors_w4a16, test_rocm_fp8, test_rocm_mxfp4, test_rocm_mxfp8_linear, test_rocm_skinny_gemms, test_scaled_mm_kernel_selection, test_silu_mul_nvfp4_quant, test_triton_scaled_mm, test_triton_w4a16, test_w4a16_kernel_selection; helpers `nvfp4_utils.py`, `quant_utils.py`) | 258 | quant GEMM/scaled-mm kernels (FP8/FP4/INT4/INT8/Marlin/Machete/cutlass/triton/rocm), per-token-group quant, nvfp4/mxfp4 quant+emulation, kernel-selection logic, RDNA3 w4a16 | 1 explicit (`test_cpu_fp8_scaled_mm`); plus pure-logic files: fp8_min_max_helper (mocked platform), mxfp4/mxfp6/nvfp4/w4a16/scaled_mm kernel-selection (docstring "CPU-only"), quant_op_schema (cuda tensors but pure schema) | 102 skipif | DEFER-GPU; PORT-CPU the pure-logic subset (kernel-selection, fp8_min_max, schema) | C10 |
| `kernels/moe/` (conftest, test_b12x, test_batched_deepgemm, test_batched_moe, test_block_fp8, test_block_int8, test_count_expert_num_tokens, test_cpu_fused_moe, test_cpu_int4_moe, test_cpu_quant_fused_moe, test_cutedsl_moe, test_cutlass_moe, test_deepep_deepgemm_moe, test_deepep_moe, test_deepep_v2_async_finalize, test_deepep_v2_moe, test_deepgemm, test_flashinfer_b12x_moe, test_flashinfer_cutedsl_layout, test_flashinfer_cutedsl_nvfp4_moe, test_flashinfer_moe, test_flashinfer, test_flydsl_moe, test_fused_topk, test_gemma4router, test_gpt_oss_triton_kernels, test_grouped_topk, test_marlin_vs_trtllm_mxint4, test_modular_kernel_combinations, test_modular_oai_triton_moe, test_moe_align_block_size, test_moe_fused_mul_sum, test_moe_kernel_oracle, test_moe_layer, test_moe_permute_unpermute, test_moe, test_moe_weight_loading_padded, test_mxfp4_moe, test_mxfp8_aiter_backend_selection, test_nvfp4_moe, test_ocp_mx_moe, test_profile_modular_kernel, test_rocm_aiter_moe, test_rocm_aiter_topk, test_routed_experts_capture_monolithic, test_routing, test_routing_simulator, test_shared_fused_moe_routed_transform, test_silu_mul_fp8_quant_deep_gemm, test_silu_mul_per_token_group_quant_fp8_colmajor, test_situ_mul_fp8_quant, test_topk_softplus_sqrt, test_triton_moe_no_act_mul, test_triton_moe_ptpc_fp8, test_trtllm_bf16_moe, test_trtllm_nvfp4_moe, test_unquantized_backend_selection, test_zen_cpu_fused_moe, test_zen_cpu_int8_moe, test_zero_expert_moe; `modular_kernel_tools/`, `parallel_utils.py`) | 315 | fused-MoE expert GEMMs all backends (triton/cutlass/deepgemm/deepep/flashinfer/trtllm/rocm-aiter), routing (fused_topk/grouped_topk/topk_softplus_sqrt), permute/align, silu-mul+fp8 quant fusion, nvfp4/mxfp4 MoE, moe_kernel_oracle | 5 explicit cpu files (cpu_fused_moe, cpu_int4_moe [ARM-gated], cpu_quant_fused_moe, zen_cpu_fused_moe, zen_cpu_int8_moe); test_routing largely torch-CPU-composable | 100 skipif | DEFER-GPU; PORT-CPU the `test_cpu_*`/`test_zen_cpu_*` + routing reference paths | C7 |
| `kernels/mamba/` (test_causal_conv1d, test_cpu_short_conv, test_gdn_forward_core_split, test_gdn_fused_mtp, test_gdn_prefill_cutedsl, test_gdn_prefill_flashinfer, test_mamba_mixer2, test_mamba_ssm_configs, test_mamba_ssm, test_mamba_ssm_ssd, test_memcpy_u64_tiled, test_precopy_mamba_align, test_replayssm_prefill_decode_equivalence_mamba2, test_replayssm_standard_decode_mamba2, test_ssu_dispatch; `cpu/`, `utils.py`) | 51 | mamba2 SSM/SSD, causal-conv1d, gated-delta-net (GDN) prefill/decode, replay-SSM equivalence, ssu dispatch | 3 cpu-gated (cpu_short_conv module-level `is_cpu` skip; causal_conv1d/mamba_ssm have cpu paths) | 22 skipif | DEFER-GPU | C2 |
| `kernels/turboquant/` (test_flydsl_turboquant_decode; `__init__.py`) | 1 | TurboQuant decode kernel (flydsl) | 0 | 1 skipif | DEFER-GPU | C10 |
| `kernels/core/` (test_activation, test_apply_rotary_emb, test_batched_weight_rms_norm, test_cpu_activation, test_fused_allreduce_gemma_rms_norm, test_fused_embed_norm, test_fused_qk_norm_rope, test_fused_q_kv_rmsnorm, test_fused_quant_layernorm, test_fused_rms_norm_gated, test_fused_silu_mul_block_quant, test_layernorm, test_minimax_reduce_rms, test_mrope, test_opcheck, test_pos_encoding, test_rocm_aiter_ops, test_rocm_misc_ops, test_rotary_embedding, test_rotary_embedding_mla_cache_fused, test_uva, test_vit_bilinear_pos_embed, test_vit_fp8_attn, test_vit_fp8_quant, test_vit_fp8_scaling, test_vocab_parallel_embedding) | 94 | activation (silu_mul etc.), layernorm/RMSNorm plain+fused, rotary embedding (+MLA cache fused), pos-encoding/mrope, fused quant-norm, ViT fp8 quant/scaling | 1 (test_cpu_activation) | 35 skipif | DEFER-GPU; PORT-CPU small pure-torch norm/activation reference subsets | C5/C10 |
| `kernels/ir/` (test_activation, test_ir_ops, test_layernorm) | 24 | activation/layernorm/IR-op tests on the torch-IR backend | 0 | 17 skipif | DEFER-GPU | C2 |
| `kernels/helion/` (helpers, test_autotune, test_benchmark_script, test_case_key, test_config_manager, test_dynamic_per_token_scaled_fp8_quant, test_fused_qk_norm_rope, test_helion_available, test_pattern_matching, test_per_token_group_fp8_quant, test_register, test_rms_norm_dynamic_per_token_quant, test_rms_norm_per_block_quant, test_silu_and_mul_per_block_quant, test_silu_mul_fp8, test_utils; `utils.py`) | 128 | helion-compiled fp8 quant kernels (per-token/per-token-group/per-block scale), rms-norm+quant fusion, silu-mul fp8, helion autotuner/registry infra | 0 (helion_available gating) | 6 skipif | DEFER-GPU | C10 |
| `kernels/quant_utils.py` (helper module, 0 tests) | 0 | reference impls: ref_dynamic_per_token_quant, group_broadcast, get_fp8_min_max | — | — | PORT-CPU (pure torch reference code, imported by many GPU tests) | C10 |
| `kernels/test_cache_kernels.py` | 1 | reshape_and_cache / cache ops | 0 | 1 | DEFER-GPU | C5 |
| `kernels/test_compressor_kv_cache.py` | 18 | compressed KV cache kernels | 0 | 10 | DEFER-GPU | C5/C10 |
| `kernels/test_awq_int4_to_int8.py` | 6 | AWQ INT4→INT8 W4A8 CPU GEMM: weight packing (`pack_cols`, packed scale/zp), `int4_scaled_mm_cpu` vs float ref, create_weights shapes | 1 (CPU-only, numpy+torch CPU) | 0 | **PORT-CPU** | C10 |
| `kernels/test_cp_gather_fp8.py` | 6 | context-parallel gather with fp8 quant | 0 | 0 | DEFER-GPU | C9/C10 |
| `kernels/test_shuffle_rows.py` | 10 | expert-token shuffle rows util | 0 | 0 | DEFER-GPU | C7 |
| `kernels/test_top_k_per_row.py` | 23 | batched top-k per row kernel | 0 | 26 | DEFER-GPU | C7 |
| `kernels/test_relu2_fp8_quant.py` | 7 | relu² activation + fp8 quant fused kernel | 0 | 1 | DEFER-GPU | C10 |
| `kernels/test_fused_quant_activation.py` | 5 | fused activation+quant kernel | 0 | 0 | DEFER-GPU | C10 |
| `kernels/test_deepseek_v4_cpu_kernels.py` | 16 | DSv4 CPU port sgl-kernels parity: compress_norm_rope_store(_indexer)_cpu, qnorm/rope/kv-insert vs eager reference | 1 (CPU-only by design) | 1 | **PORT-CPU** (already CPU; self-contained reference math) | C5/C10 |
| `kernels/test_fla_layernorm_guard.py` | 9 | flash-linear-attention layernorm guard kernel | 0 | 0 | DEFER-GPU | C2 |
| `kernels/test_onednn.py` | 2 | Intel oneDNN backend ops | 1 | 0 | SKIP-OTHERMODEL (x86/oneDNN-specific) | C2 |
| `kernels/test_engram.py` | 15 | Engram n-gram table kernels (DSv4.1-relevant) | 0 | 11 | DEFER-GPU | C2 |
| `kernels/test_concat_mla_q.py` | 6 | MLA q concat kernel | 0 | 0 | DEFER-GPU | C5 |
| `kernels/test_fused_gdn_post_conv.py` | 4 | gated-delta-net post-conv fusion | 0 | 0 | DEFER-GPU | C2 |
| `kernels/test_fused_recurrent_packed_decode.py` | 3 | packed recurrent decode kernel (mamba-family) | 0 | 1 | DEFER-GPU | C2 |
| `kernels/test_fused_sigmoid_gating_delta_rule.py` | 2 | delta-rule sigmoid gating kernel | 0 | 0 | DEFER-GPU | C2 |
| `kernels/test_kpool_decode_update_batched.py` | 6 | kpool decode update kernel | 0 | 2 | DEFER-GPU | C2 |
| `kernels/test_fused_indexer_q_rope_quant.py` | 5 | indexer q-rope + quant fused (engram-path) | 0 | 6 | DEFER-GPU | C2/C10 |
| `kernels/test_fused_inv_rope_fp8_quant.py` | 12 | inverse rope + fp8 quant fused | 0 | 0 | DEFER-GPU | C10 |
| `kernels/test_fused_qk_norm_rope_gate.py` | 1 | qk-norm+rope+gate fusion | 0 | 1 | DEFER-GPU | C5 |
| `kernels/test_fused_deepseek_v32_norm_rope.py` | 13 | DSv3.2 norm-rope fused kernel | 0 | 2 | DEFER-GPU | C5 |
| `kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | 8 | DSv4 qnorm-rope-kv-insert fused (has shared fixtures used by CPU parity test) | 0 | 3 | DEFER-GPU | C5/C10 |
| `kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py` | 4 | MiniMax-M3 qknorm-rope-kv-insert fused | 0 | 2 | SKIP-OTHERMODEL (vendor-specific) | C5 |

Excluded root files (attention/mm, other agent): `test_flex_attention.py`, `test_mhc_*.py`, `test_bf16_skinny_gemm.py`, `test_bf16x3_router_gemm_cutedsl.py`, `test_flashinfer_bf16_gemm.py`, `test_fp32_router_gemm.py`, `test_rocm_fp32_router_gemm.py`, `test_ll_bf16_gemm.py`, `test_kimi_k3_gemm_rs_ar.py`, `test_gate_linear_rocm_dispatch.py`, `test_minimax_m3_sparse_attn_fp8_scale.py`, `test_minimax_m3_amd_ops.py`.

Spot-reads: `quantization/test_per_token_group_quant.py` (CUDA-only, tail-group FP8 value+scale contract, group 64/128), `quantization/test_fp8_min_max_helper.py` (pure-python, mocked platform, e4m3fn vs fnuz), `quantization/test_nvfp4_quant.py` (sm100-gated), `quantization/test_nvfp4_emulation.py` (embeds `ref_nvfp4_quant_dequant` python reference), `quantization/test_mxfp4_kernel_selection.py` (CPU-only dispatch logic), `quantization/test_quant_op_schema.py` (cuda tensors, schema/contract + rms_norm determinism), `moe/test_cpu_int4_moe.py` (ARM-CPU-gated W4A8 INT4 fused MoE with `_make_int4_moe_weights` reference), `moe/test_routing.py` (router factory/top-k reference paths, torch-CPU composable), `mamba/test_cpu_short_conv.py` (module-level `is_cpu` skip, mocked dist), `test_awq_int4_to_int8.py` (numpy/torch-CPU pack + CPU GEMM vs float ref), `test_deepseek_v4_cpu_kernels.py` (CPU parity, own reference reimplementations).

## Notable PORT-CPU candidates

- `kernels/quantization/test_fp8_min_max_helper.py` — 3 pure-python tests (mocked `current_platform`): e4m3fn 448 vs fnuz 224 min/max logic. Directly mirrors the kind of dtype-boundary validator DS41RT needs for FP8 E4M3 scale windows.
- `kernels/test_awq_int4_to_int8.py` — 6 CPU-only tests: INT4 sub-byte weight packing (`pack_cols`, packed scale/zp roundtrip) + `int4_scaled_mm_cpu` vs float reference. The packing/unpacking roundtrip pattern is exactly the bit-level contract class DS41RT's python validators should own (adapted to FP4 E2M1 + group-16 E4M3 scales).
- `kernels/quantization/test_mxfp4_kernel_selection.py`, `test_mxfp6_kernel_selection.py`, `test_nvfp4_kernel_selection.py`, `test_w4a16_kernel_selection.py`, `test_scaled_mm_kernel_selection.py` — docstring-marked "CPU-only" dispatch-logic tests (unittest.mock, no device math). Good templates for testing kernel-selection contracts on ds41rt CPU CI.
- `kernels/quantization/test_per_token_group_quant.py` — CUDA-only, but tests the tail-group scale/value preservation contract for group-quant (group 64/128); the same group-boundary + E4M3 scale semantics DS41RT uses at group-16 — port the reference-math half (drop the custom-op call) for CPU artifact-contract tests.
- `kernels/moe/test_cpu_int4_moe.py` + `test_cpu_quant_fused_moe.py` + `test_cpu_fused_moe.py` (and `test_zen_cpu_*`) — CPU fused-MoE vs torch-reference comparisons incl. group-wise INT4/INT8 quant weight construction; reusable reference scaffolding for DS41RT MoE expert-weight artifact validation on CPU.
- `kernels/test_deepseek_v4_cpu_kernels.py` — 16 CPU parity tests whose references are self-contained reimplementations of the kernel math (no production-code dependency); the pattern (kernel-under-test vs eager reference, ULP-distance asserts) is the model for ds41rt CPU contract tests.
- `kernels/quantization/test_quant_op_schema.py` — quant op schema/contract tests (per-tensor/per-token/group-fp8); needs cuda tensors for op calls but the schema assertions translate directly to CPU-side op-signature contract checks.
- `kernels/quantization/test_nvfp4_emulation.py` — carries `ref_nvfp4_quant_dequant` / `dequantize_to_dtype` python references for NVFP4 (FP4 values + E4M3 block scales — closest upstream analog to DS41RT's FP4 E2M1 + E4M3-scale format); the reference functions are portable even though the test itself runs the Triton emulation on GPU.


<!-- ===== 15-vllm-models.md ===== -->

# 15 — vllm/tests/models/ inventory

Source: `/home/turq/dev/upstream-test-inventory/vllm/tests/models/` (~230 py files). Census only; spot-read 2–3 representatives per family. DS41RT is a single-checkpoint (DeepSeek V4.1 Flash MoE) native-stack server, not vllm-model-plugin based, so nearly all per-model correctness/generation tests are SKIP-OTHERMODEL or DEFER-GPU. Valuable extractions are model-agnostic harness patterns (registry consistency, config validation, weight-mapper unit tests, input-processor patterns).

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| registry.py | 1 | HF example-model registry (HfExamplesInfo, version gating) | yes | HF transformers ver | PORT-CPU | C10 |
| utils.py | 1 | shared test utils (output equality, logprob checks, input helpers) | yes | none | PORT-CPU | C1 |
| test_registry.py | 1 | ModelRegistry consistency: imports, text-gen/pooling/MM/speculative arch sets, HF_EXAMPLE_MODELS coherence | yes | none | PORT-CPU | C10 |
| test_utils.py | 1 | WeightsMapper / AutoWeightsLoader unit tests (rename/stack/skip patterns, no GPU) | yes | torch CPU | PORT-CPU | C10 |
| test_adapters.py | 1 | pooling/seq-cls adapter wrapper tests, tiny synthetic models | yes | torch CPU | PORT-CPU | C10 |
| test_language_model_cache_is_weak.py | 1 | weakref cache contract (get_language_model must not pin weights) | yes | none | PORT-CPU | C1 |
| test_initialization.py | 1 | minimal-arch init smoke via LLM(); KV-cache layout resolution | partial | GPU for init | DEFER-GPU | — |
| test_qwen3_5_mtp_config.py | 1 | MTP config wiring for Qwen3.5 | partial | model-specific | SKIP-OTHERMODEL | — |
| test_vision.py | 1 | vision tower correctness | no | GPU + vision ckpts | SKIP-OTHERMODEL | — |
| test_terratorch.py | 1 | geospatial model correctness | no | GPU | SKIP-OTHERMODEL | — |
| test_deepseek_v4_*.py (3 files: fi_moe_ep, mega_moe, vl_rocm, rocm_compressor_gemm_fusion) | 4 | DeepSeek V4 ROCm/EP kernels | no | ROCm GPU | SKIP-OTHERMODEL | — |
| test_dspark_mla.py | 1 | dSpark MLA kernel | no | GPU | SKIP-OTHERMODEL | — |
| test_hyv4_rocm.py | 1 | HunyuanVideo ROCm | no | ROCm | SKIP-OTHERMODEL | — |
| deepseek_v32/* (1: test_sequence_parallel.py) | 1 | sequence-parallel sharding | no | GPU | SKIP-OTHERMODEL | — |
| glm5next/* (2: test_kda_recurrent, __init__) | 2 | KDA recurrent kernel | no | GPU | SKIP-OTHERMODEL | — |
| inkling/* (13: contract_validation, fa4_*, moe_weight_layout, mtp_input_fusion, qkvr_prep, sconv_*, mm_towers, rocm/*) | 13 | Inkling arch kernels/layout, ROCm variants | no | GPU/ROCm | SKIP-OTHERMODEL | — |
| kimi_k3/* (14: kda, attn_res, latent_moe, mla_prefill, eagle3, weight_loading, amd_*, sequence_parallel) | 14 | Kimi K3 arch kernels + weight loading | partial | GPU | SKIP-OTHERMODEL | — |
| language/generation/* (8: test_common + gemma/granite/hybrid/mistral/phimoe/gdn_sleep_wake) | 8 | per-model generation correctness, shared test_common harness | no | GPU + ckpts | SKIP-OTHERMODEL | — |
| language/generation_ppl_test/* (5: ppl_utils + gemma/gpt/qwen) | 5 | perplexity harness + per-model PPL | partial | ckpts | SKIP-OTHERMODEL | — |
| language/pooling/* (23: conftest, embed_utils, 21 pooling tests) | 23 | embedding/reranker correctness; some config-level tests (pooler_config_init_behaviour, truncation_control) | partial | GPU for runner tests | DEFER-GPU | — |
| language/pooling_mteb_test/* (17: mteb utils + 13 model tests) | 17 | MTEB benchmark harness + per-model scores | no | ckpts + MTEB | SKIP-OTHERMODEL | — |
| multimodal/conftest.py + root tests (test_mapping, test_conformer_encoder, test_cohere_asr, test_mimo_v2_omni, test_nano_nemotron_vl, test_openpangu_vl) | 7 | mm root harness; test_mapping = WeightsMapper contract tests (model-agnostic core) | partial | none for mapping | PORT-CPU | C10 |
| multimodal/generation/* (25 tests + test_common) | 26 | VLM/ASR generation correctness per model | no | GPU + mm ckpts | SKIP-OTHERMODEL | — |
| multimodal/generation/vlm_utils/* (8: core, runners, builders, case_filtering, custom_inputs, model_utils, types) | 8 | VLM test harness framework (case definition, runners, builders) | partial | none (framework) | PORT-CPU | C1 |
| multimodal/pooling/* (16 model tests + __init__) | 17 | VLM pooling correctness | no | GPU | SKIP-OTHERMODEL | — |
| multimodal/processing/* (41: test_common, test_tensor_schema, transformers_backend, 38 model processors) | 41 | input-processor patterns; test_common/test_tensor_schema are model-agnostic processor-contract tests | partial | CPU for schema/common | PORT-CPU | C1 |
| quantization/* (10 tests: awq, fp8, fp8_per_channel, gpt_oss, gptq_marlin, modelopt, mxfp4, mxfp8, nvfp4, per_token_kv_cache) | 10 | quant-method correctness vs ground truth | no | quant-capable GPU | DEFER-GPU | — |
| qwen4_exp/test_config.py | 1 | tiny synthetic model-config validation (vocab=64, 2 layers; speculative config wiring, mamba hybrid state) | yes | none | PORT-CPU | C10 |
| qwen4_exp/* (6 others: test_hc_ops, test_ple, test_qsa_amd, test_qsa_pre_indexer, test_qsa_reference, __init__) | 6 | Qwen4-experimental kernels (QSA indexer, HC ops, PLE) | no | GPU | SKIP-OTHERMODEL | — |
| transformers/test_layer_registry.py | 1 | hw-agnostic layer resolution contract (monkeypatch-only) | yes | none | PORT-CPU | C1 |
| transformers/test_backend.py + fusers/* (linear, mla, moe, rms_norm) | 5 | transformers-backend layer tests | partial | torch CPU mostly | DEFER-GPU | — |
| fixtures/ | 0 | empty (data dir only) | — | — | SKIP | — |

## Notable PORT-CPU candidates

- **test_registry.py + registry.py** — registry consistency checks (every arch imports, text-gen/pooling/MM/speculative set coherence, HF example-model metadata) run fully on CPU; adapt to assert ds41rt's single-checkpoint registry consistency → C10.
- **test_utils.py** — WeightsMapper/AutoWeightsLoader unit tests (weight-name rename/stack/skip maps on synthetic modules) are the direct template for ds41rt weight/config validation → C10.
- **qwen4_exp/test_config.py** — the pattern of validating a model config with tiny synthetic dimensions (vocab=64, hidden=16, 2 layers) + mocked speculative config is exactly portable to ds41rt config validation → C10.
- **multimodal/test_mapping.py** — pure WeightsMapper contract tests (`hf_to_vllm_mapper.apply_list`), no GPU; reusable shape for ds41rt weight-map validation → C10.
- **multimodal/processing/test_common.py + test_tensor_schema.py** — model-agnostic input-processor contract tests (dummy options, batched-tensor schema, processor cache); the pattern (not the model cases) ports → C1.
- **transformers/test_layer_registry.py** — env-gated module-resolution contract tested entirely with monkeypatch/sys.modules injection; a clean template for ds41rt native-stack component resolution tests → C1.
- **test_adapters.py + test_language_model_cache_is_weak.py** — small CPU contract tests (adapter wrapping, weakref cache must-not-pin-weights) showing the "one focused invariant per file" style worth copying → C1.
- **multimodal/generation/vlm_utils/** — framework code (case types, runners, case filtering) showing how to structure correctness-case harnesses; reuse the structure for ds41rt prompt/response fixtures → C1.


<!-- ===== 16-vllm-executor-compile.md ===== -->

# Inventory 16 — vllm tests/model_executor + tests/compile

Census of 113 test files (62 model_executor, 51 compile). Sources read-only; counts are `grep -c "def test_"`.
Verdicts: PORT-CPU (logic portable, mocks/temp-files only) / DEFER-GPU (needs GPU execution) / SKIP-INFRA (machinery ds41rt does not share).
Targets per DS41RT component map (C1 API; C2 constrained; C3 sampler; C4 tokenizer; C5 KV cache; C6 spec decode; C7 MoE; C8 scheduler; C9 transport; C10 weights/quant + loader; C11 vision; C12 tool parsing).

## tests/compile (51)

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| compile/correctness_e2e/test_async_tp.py | 2 | async-TP compile correctness e2e | no | cuda, flashinfer, tp2 | DEFER-GPU | C9 |
| compile/correctness_e2e/test_sequence_parallel.py | 3 | sequence-parallel compile correctness e2e | no | cuda, tp | DEFER-GPU | C9 |
| compile/fullgraph/test_basic_correctness.py | 1 | fullgraph correctness across models/settings | no | GPU runner, HF models | DEFER-GPU | C2 |
| compile/fullgraph/test_full_cudagraph.py | 1 | full cudagraph mode e2e | no | LLM, GPU mem | DEFER-GPU | C2 |
| compile/fullgraph/test_full_graph.py | 2 | full-graph compile mode e2e | no | LLM, quant support check | DEFER-GPU | C2 |
| compile/fullgraph/test_multimodal_compile.py | 4 | multimodal model compilation | no | cuda skip, vllm_runner, forked | DEFER-GPU | C11 |
| compile/fullgraph/test_multiple_graphs.py | 1 | multi-graph splitting via decorators | partial | torch.compile machinery | SKIP-INFRA | — |
| compile/fullgraph/test_simple.py | 2 | compile decorator counter/warmup | partial | torch.compile machinery | SKIP-INFRA | — |
| compile/fullgraph/test_toy_llama.py | 1 | toy llama compile e2e | partial | triton, machinery | SKIP-INFRA | — |
| compile/fusions_e2e/test_tp1_quant.py | 2 | quant fusion e2e correctness | no | cuda, flashinfer fp8 | DEFER-GPU | C10 |
| compile/fusions_e2e/test_tp2_ar_rms.py | 3 | allreduce+rms fusion e2e, tp2 | no | 2 GPU, flashinfer | DEFER-GPU | C9 |
| compile/fusions_e2e/test_tp2_async_tp.py | 5 | async-TP fusion e2e, tp2 | no | 2 GPU, flashinfer, triton/fi attn | DEFER-GPU | C9 |
| compile/h100/test_startup.py | 2 | cold/warm compile startup timing | no | H100-specific, dynamo counters | SKIP-INFRA | — |
| compile/passes/distributed/test_async_tp.py | 2 | AsyncTPPass graph transform | no | 2 GPU, dist init | DEFER-GPU | C9 |
| compile/passes/distributed/test_fusion_all_reduce.py | 5 | allreduce fusion passes | no | 2 GPU, cuda, fp4/aiter | DEFER-GPU | C9 |
| compile/passes/distributed/test_sequence_parallelism.py | 2 | SequenceParallelismPass | no | 2 GPU, cuda | DEFER-GPU | C9 |
| compile/passes/ir/test_clone_cleanup.py | 17 | clone-elimination FX pass | yes | FX only | SKIP-INFRA | — |
| compile/passes/ir/test_inplace_functionalization.py | 5 | functionalization FX pass | partial | triton kernel reg | SKIP-INFRA | — |
| compile/passes/ir/test_lowering.py | 2 | IR lowering pass | no | cuda skip, triton | SKIP-INFRA | — |
| compile/passes/test_double_aiter_rms_quant_fusion.py | 1 | aiter rms-quant fusion | partial | rocm aiter | SKIP-INFRA | — |
| compile/passes/test_functionalization.py | 1 | fix-functionalization pass | partial | FX + TestFP8Layer | SKIP-INFRA | — |
| compile/passes/test_fuse_act_padding.py | 1 | act-padding fusion | partial | rocm gemm path | SKIP-INFRA | — |
| compile/passes/test_fuse_mla_dual_rms_norm.py | 2 | MLA dual-rms fusion | partial | aiter | SKIP-INFRA | — |
| compile/passes/test_fusion_attn.py | 1 | attention quant fusion | no | cuda, flashinfer, fp4 | SKIP-INFRA | — |
| compile/passes/test_fusion.py | 5 | generic fusion matcher | no | cuda, flashinfer, nvfp4 | SKIP-INFRA | — |
| compile/passes/test_mla_attn_quant_fusion.py | 1 | MLA attn quant fusion | no | cuda skip | SKIP-INFRA | — |
| compile/passes/test_mla_rope_kvcache_cat_fusion.py | 1 | MLA rope/kvcache cat fusion | partial | attention metadata fakes | SKIP-INFRA | — |
| compile/passes/test_noop_elimination.py | 2 | noop elimination pass | yes | FX only | SKIP-INFRA | — |
| compile/passes/test_pass_manager.py | 2 | PostGradPassManager | yes | config only | SKIP-INFRA | — |
| compile/passes/test_qk_norm_rope_fusion.py | 2 | qk-norm+rope fusion | partial | flashinfer refs | SKIP-INFRA | — |
| compile/passes/test_rmsnorm_reshape_fusion.py | 2 | rmsnorm reshape fusion | yes | FX only | SKIP-INFRA | — |
| compile/passes/test_rocm_aiter_qk_norm_rope_kvcache_fusion.py | 1 | rocm aiter rope/kvcache fusion | no | rocm aiter, version pin | SKIP-INFRA | — |
| compile/passes/test_rope_kvcache_fusion.py | 4 | rope/kvcache fusion pass | partial | triton, aiter checks | SKIP-INFRA | — |
| compile/passes/test_scatter_split_replace.py | 1 | scatter/split replace pass | partial | rotary embedding module | SKIP-INFRA | — |
| compile/passes/test_silu_mul_quant_fusion.py | 1 | silu-mul quant fusion | no | cuda, nvfp4 | SKIP-INFRA | — |
| compile/passes/test_silu_mul_quant_manual_fusion.py | 5 | manual silu-mul quant fusion | no | cuda, fused_act_quant layers | SKIP-INFRA | — |
| compile/passes/test_split_coalescing.py | 1 | split coalescing pass | yes | FX only | SKIP-INFRA | — |
| compile/passes/test_vllm_fusion_pattern_matcher_pass.py | 3 | pattern matcher framework | yes | FX only | SKIP-INFRA | — |
| compile/test_aot_compile.py | 27 | AOT compile cache hash/config | partial | cuda skips, mocks, envs | SKIP-INFRA | — |
| compile/test_codegen.py | 12 | execution-codegen split | yes | make_fx only | SKIP-INFRA | — |
| compile/test_compile_ranges.py | 4 | compilation range splitting | partial | silly_attention reg | SKIP-INFRA | — |
| compile/test_config.py | 34 | CompilationConfig pydantic validation | yes | some cuda/LLM tests inside | SKIP-INFRA | — |
| compile/test_decorator.py | 2 | compile decorators + cudagraph warmup | no | .cuda() tensors | SKIP-INFRA | — |
| compile/test_deepstack_input_contract.py | 1 | qwen3_vl deepstack input contract | partial | qwen3_vl model import | SKIP-INFRA | — |
| compile/test_dynamic_shapes_compilation.py | 3 | dynamic-shapes compilation e2e | no | LLM, logprobs compare | DEFER-GPU | C2 |
| compile/test_graph_partition.py | 13 | split_graph partition logic | yes | make_fx only | SKIP-INFRA | — |
| compile/test_inductor_fallback_allow_list_patch.py | 18 | inductor FALLBACK_ALLOW_LIST patch | yes | mock proxy, timing | SKIP-INFRA | — |
| compile/test_rotary_embedding_compile.py | 1 | rope under torch.compile | no | cuda required | DEFER-GPU | C5 |
| compile/test_sequence_parallelism_threshold.py | 10 | SP threshold logic | yes | pure unit | SKIP-INFRA | C9 |
| compile/test_structured_logging.py | 1 | compile structured logging | no | cuda required | SKIP-INFRA | — |
| compile/test_wrapper.py | 1 | TorchCompileWithNoGuardsWrapper | partial | bytecode hook | SKIP-INFRA | — |

## tests/model_executor (62)

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| model_executor/kernels/test_b12x_linear.py | 23 | b12x (Blackwell) linear kernel dispatch | no | B200-specific modules, importlib sweep | DEFER-GPU | C10 |
| model_executor/layers/test_fused_shared_expert.py | 38 | fused shared-expert layer | partial | heavy mocks, rocm refs | DEFER-GPU | C7 |
| model_executor/layers/test_mla_short_prefill_indexer.py | 6 | MLA sparse-attn indexer batch state | yes | mocked metadata | PORT-CPU | C5 |
| model_executor/layers/test_pooler_activations.py | 25 | pooler activation fns | yes | pure torch CPU | PORT-CPU | C2 |
| model_executor/layers/test_pooler_heads.py | 51 | pooler head classes | yes | pure torch CPU | PORT-CPU | C2 |
| model_executor/layers/test_pooler_methods.py | 37 | pooling methods + factories | yes | transformers config only | PORT-CPU | C2 |
| model_executor/layers/test_rocm_unquantized_gemm.py | 7 | rocm gemm dispatch | no | rocm gfx, triton | SKIP-INFRA | C10 |
| model_executor/model_loader/fastsafetensors_loader/test_fastsafetensors_loader.py | 1 | fastsafetensors loader download | no | vllm_runner, HF | DEFER-GPU | C10 |
| model_executor/model_loader/fastsafetensors_loader/test_weight_utils.py | 1 | fastsafetensors iterator | yes | temp files, mocked hf | PORT-CPU | C10 |
| model_executor/model_loader/instanttensor_loader/test_instanttensor_loader.py | 1 | instanttensor loader download | no | vllm_runner, HF | DEFER-GPU | C10 |
| model_executor/model_loader/instanttensor_loader/test_weight_utils.py | 1 | instanttensor iterator | yes | mocked files | PORT-CPU | C10 |
| model_executor/model_loader/runai_streamer_loader/test_runai_model_streamer_loader.py | 7 | runai loader selection/paths | partial | some vllm_runner cases | PORT-CPU | C10 |
| model_executor/model_loader/runai_streamer_loader/test_runai_model_streamer_s3.py | 1 | runai S3 path, streamer mock | yes | StreamerPatcher | PORT-CPU | C10 |
| model_executor/model_loader/runai_streamer_loader/test_runai_utils.py | 3 | runai URI/file listing utils | yes | pure logic | PORT-CPU | C10 |
| model_executor/model_loader/runai_streamer_loader/test_weight_utils.py | 2 | runai safetensors iterator | yes | temp safetensors files | PORT-CPU | C10 |
| model_executor/model_loader/tensorizer_loader/test_tensorizer.py | 14 | tensorizer loader | partial | tensorizer lib, subprocess | SKIP-INFRA | C10 |
| model_executor/model_loader/test_checkpoint_weight_patch.py | 1 | checkpoint weight patch (TP packing) | yes | cpu_test mark | PORT-CPU | C10 |
| model_executor/model_loader/test_ep_weight_filter.py | 37 | EP expert-id weight filtering | yes | temp safetensors | PORT-CPU | C10 |
| model_executor/model_loader/test_filter_duplicate_safetensors.py | 2 | duplicate-file filtering | yes | temp files | PORT-CPU | C10 |
| model_executor/model_loader/test_gpt_oss_weight_loading.py | 6 | gpt_oss expert weight mapping | yes | mocked namespace | PORT-CPU | C10 |
| model_executor/model_loader/test_modelexpress_loader.py | 5 | modelexpress loader w/ fake pkg | yes | monkeypatched module | PORT-CPU | C10 |
| model_executor/model_loader/test_mtp_validation.py | 1 | MTP completeness-check env scoping | yes | pure logic | PORT-CPU | C6 |
| model_executor/model_loader/test_registry.py | 6 | model-loader registry | yes | dummy loaders | PORT-CPU | C10 |
| model_executor/model_loader/test_reload.py | 25 | module/param reload + gc | yes | mocks, WeakKeyDictionary | PORT-CPU | C10 |
| model_executor/model_loader/test_sharded_state_loader.py | 2 | sharded state loader e2e | no | LLM, multiprocess | DEFER-GPU | C10 |
| model_executor/model_loader/test_weight_cache.py | 1 | IPC weight cache e2e | no | subprocess IPC | SKIP-INFRA | C10 |
| model_executor/model_loader/test_weight_tying.py | 3 | word-embedding retie logic | yes | cpu_test mark | PORT-CPU | C10 |
| model_executor/offloader/test_prefetch.py | 1 | offloader prefetch slot logic | yes | pure index math | PORT-CPU | C10 |
| model_executor/test_b12x_warmup.py | 5 | b12x warmup token-count coverage | yes | mocked, SimpleNamespace | PORT-CPU | C10 |
| model_executor/test_bailing_mrope.py | 4 | bailing mrope position construction | yes | cpu tensors, mocks | PORT-CPU | C11 |
| model_executor/test_cpu_unquantized_gemm_dispatch.py | 3 | CPU gemm dispatch (zentorch) | yes | monkeypatched zentorch | PORT-CPU | C10 |
| model_executor/test_deep_gemm_warmup.py | 4 | deep_gemm warmup config logic | yes | mocked layers | PORT-CPU | C10 |
| model_executor/test_eagle_quantization.py | 5 | draft-model quant config resolution | yes | fully mocked | PORT-CPU | C6 |
| model_executor/test_enabled_custom_ops.py | 4 | custom-op enable/disable registry | yes | rocm aiter refs only | PORT-CPU | — |
| model_executor/test_ernie45_vl_mrope.py | 3 | ernie45 vl mrope positions | yes | cpu default device forced | PORT-CPU | C11 |
| model_executor/test_flashinfer_autotune_warmup.py | 3 | flashinfer autotune warmup plan | yes | cpu_test, mocked MoE runner | PORT-CPU | C10 |
| model_executor/test_gemma_hidden_act.py | 3 | gemma activation-fn aliases | yes | pure fn checks | PORT-CPU | C2 |
| model_executor/test_jit_warmup_cutedsl_launcher.py | 2 | CuTeDSL launcher compile keys | partial | cutedsl infra | SKIP-INFRA | C10 |
| model_executor/test_jit_warmup.py | 26 | jit warmup config math/AST | yes | pure logic | PORT-CPU | C10 |
| model_executor/test_jit_warmup_triton_launcher.py | 12 | triton kernel launcher | no | triton.jit, GPU launch | DEFER-GPU | C10 |
| model_executor/test_keye_mrope.py | 3 | keye mrope positions | yes | cpu default device forced | PORT-CPU | C11 |
| model_executor/test_keye_vl1_5_mrope.py | 3 | keye vl1.5 mrope positions | yes | cpu default device forced | PORT-CPU | C11 |
| model_executor/test_mamba_triton_warmup.py | 1 | mamba batch-memcpy kernel compile | no | cuda required | DEFER-GPU | C2 |
| model_executor/test_minicpmv.py | 3 | minicpmv video-embed parsing | yes | object.__new__ trick, CPU | PORT-CPU | C11 |
| model_executor/test_mistral_large_3_eagle.py | 2 | mistral-large-3 eagle wiring | yes | cpu_test, dummy modules | PORT-CPU | C6 |
| model_executor/test_model_load_with_params.py | 3 | bert/roberta load-with-params | no | vllm_runner | DEFER-GPU | C10 |
| model_executor/test_nemotron_h_quantization.py | 2 | nemotron lm-head quant config | yes | fully mocked | PORT-CPU | C10 |
| model_executor/test_oink_integration.py | 2 | oink kernel availability | no | cuda only | SKIP-INFRA | — |
| model_executor/test_paddleocr_vl_mrope.py | 4 | paddleocr vl mrope positions | yes | cpu default device forced | PORT-CPU | C11 |
| model_executor/test_plamo3.py | 1 | plamo3 supports_eagle3 check | yes | cpu_test | PORT-CPU | C6 |
| model_executor/test_qwen3_5_quantization.py | 2 | qwen3.5 lm-head quant config | yes | fully mocked | PORT-CPU | C10 |
| model_executor/test_qwen3_asr_mrope.py | 1 | qwen3-asr mrope with audio stripped | yes | pure position math | PORT-CPU | C11 |
| model_executor/test_qwen3_omni.py | 6 | qwen3-omni input processing | yes | mocked processing ctx | PORT-CPU | C11 |
| model_executor/test_qwen3_vl_mrope.py | 1 | qwen3-vl mrope positions | yes | cpu default device forced | PORT-CPU | C11 |
| model_executor/test_qwen_triton_warmup.py | 1 | qwen GDN warmup config | no | cuda required | DEFER-GPU | C2 |
| model_executor/test_qwen_vl_triton_warmup.py | 3 | qwen-vl vision warmup path | no | torch.device("cuda") | DEFER-GPU | C11 |
| model_executor/test_replayssm_warmup.py | 3 | replay-ssm warmup planning | no | cuda-alike/flashinfer paths | DEFER-GPU | C2 |
| model_executor/test_routed_experts_capture.py | 15 | MoE routed-experts capture | yes | cpu_test, mocked EplbState | PORT-CPU | C7 |
| model_executor/test_sleep_mode_tensor_ownership.py | 1 | sleep-mode tensor ownership e2e | no | cuda, LLM, HF gemma3n | DEFER-GPU | C5 |
| model_executor/test_utils.py | 6 | parameter replace/copy logic | yes | pure torch CPU | PORT-CPU | C10 |
| model_executor/test_watermark_sample_warmup.py | 11 | watermark sampler warmup coverage | yes | mocked watermarker | PORT-CPU | C3 |
| model_executor/test_weight_utils.py | 19 | HF download/weight iterators | yes | temp files, hf hub mocks (some network) | PORT-CPU | C10 |

## Notable PORT-CPU candidates

- `model_executor/model_loader/test_ep_weight_filter.py` (37 tests) — expert-id parsing + safetensors filtering on temp files only; directly maps to ds41rt weight loading (C10).
- `model_executor/test_weight_utils.py` (19 tests) — HF download/iterator logic with mocked hub; core loader utility coverage (C10).
- `model_executor/test_routed_experts_capture.py` (15 tests, cpu_test) — MoE routed-experts capture against mocked EplbLayerState; cheapest C7 coverage in the tree.
- `model_executor/test_watermark_sample_warmup.py` (11 tests) — enumerates every sampler specialization the warmup must cover, fully mocked; ready-made C3 checklist.
- `model_executor/test_eagle_quantization.py` + `test_mistral_large_3_eagle.py` + `test_plamo3.py` + `model_loader/test_mtp_validation.py` (11 tests total) — spec-decode quant/wiring checks, mocked or cpu_test; all of C6's portable coverage in one batch.
- mrope CPU-forced family (`test_bailing_mrope`, `test_ernie45_vl_mrope`, `test_keye_mrope`, `test_keye_vl1_5_mrope`, `test_paddleocr_vl_mrope`, `test_qwen3_vl_mrope`, `test_qwen3_asr_mrope`, ~19 tests) — vl mrope position logic already forced to CPU by upstream; C11 patterns portable nearly verbatim.
- `model_executor/layers/test_pooler_*.py` (113 tests combined) — pure-CPU tensor math for pooler activations/heads/methods; bulk C2 coverage at zero GPU cost.
- `model_executor/model_loader/test_registry.py` + `test_modelexpress_loader.py` + `test_gpt_oss_weight_loading.py` (17 tests) — loader registry and per-format weight-mapping logic with fakes/mocks; no GPU, no network.


<!-- ===== 17-vllm-distributed-lora.md ===== -->

# Inventory 17 — vLLM tests/distributed + tests/lora

Census of `tests/distributed/` (59 py) and `tests/lora/` (40 py) in the upstream vLLM checkout. `n` = `grep -c "def test_"` (0 = script-style, helper-parametrized, or support file). Verdicts per task brief: comm tests DEFER-GPU; pure config/coordination math PORT-CPU; LoRA SKIP-OTHERMODEL / SKIP-INFRA (ds41rt has no LoRA path → C10 SKIP).

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| distributed/conftest.py | 0 | support | – | fixtures (tensor-parallel init) | SKIP-INFRA | – |
| distributed/eplb_utils.py | 0 | support | – | mp.spawn EPLB worker harness | SKIP-INFRA | – |
| distributed/__init__.py | 0 | support | – | – | SKIP-INFRA | – |
| distributed/test_ca_buffer_sharing.py | 0 | custom allreduce IPC | no | torchrun, p2p GPU access | DEFER-GPU | C9 |
| distributed/test_comm_ops.py | 14 | collective comm ops | no | multi-GPU, tensor-parallel init | DEFER-GPU | C9 |
| distributed/test_context_parallel.py | 1 | context parallel | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_custom_all_gather_reduce_scatter.py | 1 | custom AGA/RSA | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_custom_all_reduce.py | 5 | custom allreduce | no | multi-GPU, P2P | DEFER-GPU | C9 |
| distributed/test_dcp_a2a.py | 18 | DeepEP all-to-all | no | multi-GPU, DeepEP | DEFER-GPU | C9 |
| distributed/test_dcp_direct_a2a_lse_reduce.py | 21 | DeepEP fused LSE reduce | no | multi-GPU, DeepEP | DEFER-GPU | C9 |
| distributed/test_distributed_oot.py | 1 | out-of-tree distributed plugins | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_elastic_ep.py | 1 | elastic expert parallel | no | multi-GPU | DEFER-GPU | C9/C7 |
| distributed/test_engram_dp_shard.py | 6 | Engram datapal shard | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_eplb_algo.py | 13 | EPLB load-balance algorithm | yes | torch CPU tensors, numpy | PORT-CPU | C7 |
| distributed/test_eplb_events.py | 3 | EPLB event records | maybe | torch, mocked state | PORT-CPU | C7 |
| distributed/test_eplb_execute.py | 5 | EPLB execute path | no | multi-GPU via eplb_utils | DEFER-GPU | C7 |
| distributed/test_eplb_fused_moe_layer.py | 1 | EPLB + fused MoE layer | no | multi-GPU | DEFER-GPU | C7 |
| distributed/test_eplb_fused_moe_layer_dep_nvfp4.py | 1 | EPLB MoE NVFP4 | no | multi-GPU, NVFP4 | DEFER-GPU | C7 |
| distributed/test_eplb_quant_scale_consistency.py | 2 | EPLB quant scale | maybe | torch tensors, mocked | PORT-CPU | C7 |
| distributed/test_eplb_spec_decode.py | 2 | EPLB + spec decode | no | multi-GPU | DEFER-GPU | C7/C6 |
| distributed/test_eplb_utils.py | 5 | EPLB map commit utils | yes | torch CPU tensors, MagicMock | PORT-CPU | C7 |
| distributed/test_events.py | 10 | async event serialization | yes | cpu device | PORT-CPU | C8 |
| distributed/test_expert_parallel.py | 1 | expert parallel smoke | no | multi-GPU | DEFER-GPU | C7 |
| distributed/test_expert_placement.py | 3 | expert map round-robin layout | yes | pure index math | PORT-CPU | C7 |
| distributed/test_file_store.py | 3 | distributed file store | maybe | gloo, spawns processes | DEFER-GPU | C9 |
| distributed/test_flashinfer_pcie_ipc_all_reduce.py | 5 | FlashInfer allreduce | no | multi-GPU, flashinfer | DEFER-GPU | C9 |
| distributed/test_kimi_linear_context_parallel.py | 2 | Kimi linear CP | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_kv_cache_events.py | 12 | KV cache event schema | yes | cpu device, config mocks | PORT-CPU | C5 |
| distributed/test_kvlayout.py | 4 | KV connector cache layout | yes | DeviceConfig("cpu") | PORT-CPU | C5 |
| distributed/test_mnnvl_alltoall.py | 8 | MNNVL all-to-all | no | multi-node NVLink | DEFER-GPU | C9 |
| distributed/test_mq_connect_ip.py | 3 | MQ connect IP util | maybe | sockets, no GPU | PORT-CPU | C9 |
| distributed/test_multi_node_assignment.py | 1 | multi-node TP/PP assignment | maybe | gloo spawns | DEFER-GPU | C9 |
| distributed/test_multiproc_executor.py | 12 | MultiprocExecutor RPC | no | multi-GPU, opt-125m | DEFER-GPU | C9 |
| distributed/test_nccl_symm_mem.py | 3 | NCCL symmetric memory | no | multi-GPU, NCCL | DEFER-GPU | C9 |
| distributed/test_node_count.py | 0 | node count util | maybe | gloo script, torchrun | DEFER-GPU | C9 |
| distributed/test_packed_tensor.py | 21 | packed tensor comm | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_pipeline_parallel.py | 3 | pipeline parallel fwd/bwd | no | multi-GPU, PP | DEFER-GPU | C8 |
| distributed/test_pipeline_partition.py | 2 | PP layer partition math | yes | env monkeypatch, pure math | PORT-CPU | C8 |
| distributed/test_pp_cudagraph.py | 1 | PP + CUDA graph | no | multi-GPU | DEFER-GPU | C8 |
| distributed/test_pynccl.py | 15 | PyNccl communicator | no | multi-GPU, NCCL | DEFER-GPU | C9 |
| distributed/test_quick_all_reduce.py | 18 | quick allreduce kernels | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_ray_v2_executor_e2e.py | 3 | Ray executor e2e | no | ray, multi-GPU | DEFER-GPU | C9 |
| distributed/test_ray_v2_executor.py | 15 | Ray v2 executor | no | ray, multi-GPU | DEFER-GPU | C9 |
| distributed/test_rocm_aiter_custom_ar.py | 2 | ROCm AITER custom AR | no | ROCm GPU | DEFER-GPU | C9 |
| distributed/test_rocm_quick_reduce.py | 27 | ROCm quick reduce | no | ROCm GPU | DEFER-GPU | C9 |
| distributed/test_same_node.py | 0 | same-node detection | maybe | gloo script, torchrun | DEFER-GPU | C9 |
| distributed/test_sharded_rdt_plan.py | 94 | sharded RDT planner | maybe | large, spawns workers | DEFER-GPU | C9 |
| distributed/test_sharded_rdt_producer.py | 64 | sharded RDT producer | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_sharded_rdt_trainer.py | 25 | sharded RDT trainer | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_shm_broadcast.py | 17 | shm broadcast | maybe | multi-proc, shm | DEFER-GPU | C9 |
| distributed/test_shm_buffer.py | 6 | shm buffer | yes-ish | multi-proc, no CUDA | PORT-CPU* | C9 |
| distributed/test_shm_storage.py | 7 | shm storage backend | yes | cpu | PORT-CPU* | C9 |
| distributed/test_split_group.py | 3 | split_group coordinator | no | torch.distributed spawns | DEFER-GPU | C9 |
| distributed/test_symm_mem_allreduce.py | 2 | symmetric-memory allreduce | no | multi-GPU | DEFER-GPU | C9 |
| distributed/test_torchrun_example.py | 1 | torchrun example | no | torchrun, multi-GPU | DEFER-GPU | C9 |
| distributed/test_torchrun_example_moe.py | 1 | torchrun MoE example | no | torchrun, multi-GPU | DEFER-GPU | C9 |
| distributed/test_utils.py | 2 | distributed utils (StatelessPG) | no | ray, multi_gpu_test | DEFER-GPU | C9 |
| distributed/test_weight_transfer_nccl_uid.py | 14 | weight transfer NCCL uid | no | multi-GPU, NCCL | DEFER-GPU | C9 |
| distributed/test_weight_transfer.py | 88 | weight transfer layer map | maybe | mostly CPU map math, some NCCL | PORT-CPU* | C9 |
| lora/conftest.py | 0 | support | – | LoRA fixtures | SKIP-INFRA | C10 |
| lora/__init__.py | 0 | support | – | – | SKIP-INFRA | C10 |
| lora/utils.py | 0 | support | – | LoRA test utils | SKIP-INFRA | C10 |
| lora/test_add_lora.py | 1 | LoRA add utility | yes | cpu | SKIP-INFRA | C10 |
| lora/test_chatglm3_tp.py | 3 | ChatGLM3 TP LoRA | no | multi-GPU, other model | SKIP-OTHERMODEL | C10 |
| lora/test_deepseekv2_tp.py | 4 | DeepSeek-V2 TP LoRA | no | multi-GPU, other model | SKIP-OTHERMODEL | C10 |
| lora/test_default_mm_loras.py | 5 | default multimodal LoRAs | no | mm models | SKIP-OTHERMODEL | C10 |
| lora/test_fused_moe_lora_kernel.py | 8 | fused MoE LoRA kernel | no | GPU kernel | SKIP-INFRA | C10 |
| lora/test_gemma4_tp.py | 3 | Gemma4 TP LoRA | no | multi-GPU, other model | SKIP-OTHERMODEL | C10 |
| lora/test_gptoss_tp.py | 2 | GPT-OSS TP LoRA | no | multi-GPU, other model | SKIP-OTHERMODEL | C10 |
| lora/test_layers.py | 16 | LoRA layer wrappers | maybe | cpu tensors mostly | SKIP-INFRA | C10 |
| lora/test_layers_utils.py | 4 | LoRA layer utils | yes | cpu | SKIP-INFRA | C10 |
| lora/test_llama_tp.py | 4 | Llama TP LoRA | no | multi-GPU, other model | SKIP-OTHERMODEL | C10 |
| lora/test_lora_checkpoints.py | 4 | LoRA checkpoint load/save | maybe | cpu | SKIP-INFRA | C10 |
| lora/test_lora_functions.py | 2 | LoRA apply functions | maybe | cpu | SKIP-INFRA | C10 |
| lora/test_lora_huggingface.py | 1 | HF LoRA hub load | no | network/HF | SKIP-INFRA | C10 |
| lora/test_lora_manager.py | 20 | LoRA manager | maybe | cpu/mocks | SKIP-INFRA | C10 |
| lora/test_lora_utils.py | 9 | LoRA utils | yes | cpu | SKIP-INFRA | C10 |
| lora/test_lora_weights.py | 2 | LoRA weight packing | yes | cpu | SKIP-INFRA | C10 |
| lora/test_minicpmv_tp.py | 3 | MiniCPM-V TP LoRA | no | multi-GPU, other model | SKIP-OTHERMODEL | C10 |
| lora/test_mixtral.py | 1 | Mixtral LoRA | no | other model | SKIP-OTHERMODEL | C10 |
| lora/test_moe_lora_align_sum.py | 5 | MoE LoRA align/sum | no | GPU kernel | SKIP-INFRA | C10 |
| lora/test_moe_lora_ep_load.py | 1 | MoE LoRA EP load | no | multi-GPU | SKIP-INFRA | C10 |
| lora/test_olmoe_tp.py | 5 | OLMoE TP LoRA | no | multi-GPU, other model | SKIP-OTHERMODEL | C10 |
| lora/test_peft_helper.py | 3 | PEFT helper | yes | cpu | SKIP-INFRA | C10 |
| lora/test_punica_ops_fp8.py | 2 | Punica FP8 ops | no | GPU kernel | SKIP-INFRA | C10 |
| lora/test_punica_ops.py | 3 | Punica ops | no | GPU kernel | SKIP-INFRA | C10 |
| lora/test_punica_xpu_ops.py | 2 | Punica XPU ops | no | XPU | SKIP-INFRA | C10 |
| lora/test_quant_model.py | 2 | quant model LoRA | no | GPU | SKIP-INFRA | C10 |
| lora/test_qwen35_densemodel_lora.py | 2 | Qwen3.5 dense LoRA | no | other model | SKIP-OTHERMODEL | C10 |
| lora/test_qwen36_moe_lora.py | 2 | Qwen3.6 MoE LoRA | no | other model | SKIP-OTHERMODEL | C10 |
| lora/test_qwen3moe_tp.py | 3 | Qwen3MoE TP LoRA | no | multi-GPU, other model | SKIP-OTHERMODEL | C10 |
| lora/test_qwen3_unembed.py | 1 | Qwen3 unembed LoRA | no | other model | SKIP-OTHERMODEL | C10 |
| lora/test_qwen3_with_multi_loras.py | 4 | Qwen3 multi-LoRA | no | other model | SKIP-OTHERMODEL | C10 |
| lora/test_qwenvl.py | 6 | Qwen-VL LoRA | no | other model | SKIP-OTHERMODEL | C10 |
| lora/test_resolver.py | 4 | LoRA resolver | yes-ish | cpu, some HF | SKIP-INFRA | C10 |
| lora/test_transformers_model.py | 3 | transformers-model LoRA | no | other model | SKIP-OTHERMODEL | C10 |
| lora/test_utils.py | 8 | LoRA test utils | yes | cpu | SKIP-INFRA | C10 |
| lora/test_whisper.py | 2 | Whisper LoRA | no | other model | SKIP-OTHERMODEL | C10 |
| lora/test_worker.py | 1 | LoRA worker | no | multi-GPU | SKIP-INFRA | C10 |

`*` = PORT-CPU candidates that still spawn processes or touch NCCL in some tests — port only after checking per-test; counted as PORT-CPU for census purposes but lower priority.

## Notable PORT-CPU candidates

- `distributed/test_pipeline_partition.py` (n=2) — pure `get_pp_indices` layer-partition arithmetic over env-var strings; zero GPU. Directly mirrors ds41rt rank→layer-range mapping (C8).
- `distributed/test_eplb_algo.py` (n=13) — EPLB logical/physical expert map + load-stats math on CPU torch tensors; the whole expert-placement balancing concept maps to C7 even though the transport underneath differs.
- `distributed/test_eplb_utils.py` (n=5) — EPLB map-commit logic with mocked model state and CPU tensors; same C7 concept lane.
- `distributed/test_expert_placement.py` (n=3) — round-robin expert→rank index arithmetic, pure Python; canonical C7 coordination-math port.
- `distributed/test_kvlayout.py` (n=4) — KV connector cache-layout selection under `DeviceConfig("cpu")`; config-level C5 census value.
- `distributed/test_kv_cache_events.py` (n=12) — KV cache event schema/serialization on CPU; large n for the effort, C5.


<!-- ===== 18-vllm-quant-tok-config.md ===== -->

# Inventory 18 — vLLM quantization / tokenizer-utils / tokenizers / samplers / detokenizer / watermarking / config

Census of 83 py files across 7 dirs (actual counts differ slightly from the brief: quantization 36, transformers_utils 12, tokenizers_ 8, samplers 7, detokenizer 7, watermarking 5, config 8). `n` = `grep -c "def test_"`. Verdicts: PORT-CPU / DEFER-GPU (weights or GPU kernels) / SKIP (infra helper, no tests).

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| tests/quantization/__init__.py | 0 | pkg init | – | none | SKIP | – |
| tests/quantization/reference_mxfp4.py | 0 | MXFP4 reference impl (helper) | – | none | SKIP | C10 |
| tests/quantization/utils.py | 0 | shared fixtures (quant method probe) | – | allenai/OLMoE-1B pin | SKIP | C10 |
| tests/quantization/test_auto_awq.py | 9 | AWQ config unification + AWQ model load/run | partial | Qwen/Qwen2-1.5B-Instruct-AWQ | PORT-CPU (config) / DEFER-GPU (load) | C10 |
| tests/quantization/test_auto_gptq.py | 6 | auto_gptq config + GPTQ model load/run | partial | GPTQ models via utils | PORT-CPU (config) / DEFER-GPU (load) | C10 |
| tests/quantization/test_auto_round.py | 90 | AutoRound quant method setup + inference | no | many quantized HF models | DEFER-GPU | C10 |
| tests/quantization/test_blackwell_moe.py | 21 | Blackwell FP4 MoE via RemoteOpenAIServer | no | deepseek-ai/DeepSeek-V3.1, nvidia FP4 repos | DEFER-GPU | C10/C7 |
| tests/quantization/test_compressed_tensors.py | 27 | llmcompressor quant setup + weight loading | partial | nm-testing tiny-llama repos | DEFER-GPU (mock config cases portable) | C10 |
| tests/quantization/test_configs.py | 1 | Marlin-from-autogptq-config via ModelConfig | yes | none (config only) | PORT-CPU | C10 |
| tests/quantization/test_config_utils.py | 12 | quant config matching utils (Mock) | yes | none | PORT-CPU | C10 |
| tests/quantization/test_cpu_offload.py | 4 | quantized CPU offload vs baseline server | no | nm-testing + Qwen AWQ pins | DEFER-GPU | C10/C5 |
| tests/quantization/test_cpu_w8a8.py | 1 | CPU W8A8 quant model run | yes | RedHatAI/Qwen3-30B-A3B-...-w8a8 | PORT-CPU (native CPU quant) | C10 |
| tests/quantization/test_cpu_wna16.py | 1 | CPU WNA16 quant model run | yes | openai/gpt-oss-20b, Qwen GPTQ/FP8 pins | PORT-CPU (native CPU quant) | C10 |
| tests/quantization/test_cutlass_w4a16.py | 5 | Cutlass/Machete W4A16 Hopper kernels | no | nm-testing/tinyllama-oneshot-w4a16 | DEFER-GPU (CUDA kernel) | C10 |
| tests/quantization/test_experts_int8.py | 1 | experts_int8 startup + generation | no | HF_EXAMPLE_MODELS registry | DEFER-GPU | C10/C7 |
| tests/quantization/test_fp8.py | 14 | FP8 enablement, config + model runs | partial | facebook/opt-125m, allenai OLMoE, nm-testing FP8 | DEFER-GPU (config cases portable) | C10 |
| tests/quantization/test_fp8_per_channel.py | 2 | FP8 per-channel online quant via _custom_ops | no | none | DEFER-GPU (CUDA custom ops) | C10 |
| tests/quantization/test_gfx950_moe.py | 5 | MXFP4 MoE oracle backend selection (monkeypatched) | yes | none (stubbed configs) | PORT-CPU | C10/C7 |
| tests/quantization/test_gptq_dynamic.py | 1 | rejects unsupported GPTQ group actorder | yes | ModelCloud dynamic-cfg repo (config-only) | PORT-CPU | C10 |
| tests/quantization/test_gptq_v2.py | 2 | gptq_v2 checkpoint load + inference | no | gptq_v2 repos via runner | DEFER-GPU | C10 |
| tests/quantization/test_humming_ignore.py | 2 | humming is_layer_skipped w/ compressed-tensors | yes | none (real Kimi-K2.6 config JSON inline) | PORT-CPU | C10 |
| tests/quantization/test_humming_mxfp4_block_fp8.py | 6 | Humming MXFP4-GEMM + block-FP8 activation | no | none (in-memory tensors) | DEFER-GPU (fused CUDA kernel) | C10 |
| tests/quantization/test_int8_moe_oracle.py | 3 | INT8 W8A8 fused-MoE oracle dispatch (monkeypatched) | yes | none (stubbed configs) | PORT-CPU | C10/C7 |
| tests/quantization/test_lm_head.py | 1 | GPTQ quantized lm_head load | no | via load_model_without_vllm_runner | DEFER-GPU | C10 |
| tests/quantization/test_mixed_precision.py | 1 | quark MXFP4/FP8 mixed-precision models | no | amd/ Llama-2-70b / Qwen3 pins | DEFER-GPU | C10 |
| tests/quantization/test_modelopt.py | 27 | ModelOpt method setup + weight loading | partial | mock + runner mixes | DEFER-GPU (config cases portable) | C10 |
| tests/quantization/test_moe_wna16.py | 14 | MoE WNA16 config/oracle + weight transpose | yes | none (in-memory tensors) | PORT-CPU | C10/C7 |
| tests/quantization/test_online.py | 26 | online quant config validation + model runs | partial | nm-testing tinysmokeqwen3moe | DEFER-GPU (most tests portable config parsing) | C10 |
| tests/quantization/test_online_mxfp4.py | 5 | online MXFP4 backend/kernel selection | yes | none (ModelConfig only) | PORT-CPU | C10 |
| tests/quantization/test_per_token_kv_cache.py | 15 | per-token-head KV cache quant (INT4/INT8/FP8) | yes | none (MagicMock) | PORT-CPU | C5 |
| tests/quantization/test_quantization_config_args.py | 22 | QuantizationConfigArgs parsing | yes | none | PORT-CPU | C10 |
| tests/quantization/test_quark.py | 31 | quark method setup + weight loading | partial | amd/ quark repos | DEFER-GPU (config cases portable) | C10 |
| tests/quantization/test_register_quantization_config.py | 2 | register custom quant config; custom model forward | partial | meta-llama/Llama-3.2-1B-Instruct | PORT-CPU (register) / DEFER-GPU (custom model) | C10 |
| tests/quantization/test_torchao.py | 11 | torchao pre-quant + online quant runs | no | facebook/opt-125m, Qwen/Qwen3-0.6B | DEFER-GPU | C10 |
| tests/quantization/test_trtllm_nvfp4_hidden_dim_padding.py | 4 | TRT-LLM NVFP4 hidden-dim padding | no | none (flashinfer imports) | DEFER-GPU (flashinfer kernel) | C10 |
| tests/quantization/test_turboquant.py | 46 | TurboQuant KV-cache quant unit tests | yes | none (torch) | PORT-CPU | C5 |
| tests/transformers_utils/__init__.py | 0 | pkg init | – | none | SKIP | – |
| tests/transformers_utils/test_bailing_moe_v3_vl_config.py | 6 | Bailing MoE V3 VL config parsing | yes | none (AutoConfig JSON) | PORT-CPU | C4 |
| tests/transformers_utils/test_config_parser_registry.py | 2 | get_config_parser / register_config_parser | yes | none | PORT-CPU | C4 |
| tests/transformers_utils/test_config.py | 13 | HF config patches, EOS tokens, mrope dims | yes | meta-llama pin (metadata only) | PORT-CPU | C4 |
| tests/transformers_utils/test_dspark_mla_config.py | 4 | K3DSpark MLA config via ModelConfig | yes | none (inline config) | PORT-CPU | C4 |
| tests/transformers_utils/test_hf_overrides_model_type.py | 1 | hf_overrides → config class resolution | yes | none (tempfiles) | PORT-CPU | C4 |
| tests/transformers_utils/test_muse_glimmer_config.py | 7 | MuseGlimmer config normalization (math) | yes | none | PORT-CPU | C4 |
| tests/transformers_utils/test_processor.py | 2 | processor kwargs typing | yes | none | PORT-CPU | C4 |
| tests/transformers_utils/test_repo_utils.py | 5 | HF hub cache/repo utils (mocked hub) | yes | none (hub mocks) | PORT-CPU | C4 |
| tests/transformers_utils/test_speculators_dspark_config.py | 3 | SpeculatorsConfig parsing | yes | none | PORT-CPU | C4 |
| tests/transformers_utils/test_speculators_override.py | 6 | maybe_override_with_speculators merging | yes | none | PORT-CPU | C4 |
| tests/transformers_utils/test_utils.py | 4 | transformers_utils.utils helpers | yes | none | PORT-CPU | C4 |
| tests/tokenizers_/__init__.py | 0 | pkg init (renamed to avoid HF clash) | – | none | SKIP | – |
| tests/tokenizers_/test_basic.py | 3 | TokenizerLike protocol, get_tokenizer dispatch | yes | opt-125m / Mistral / DeepSeek-V3 tokenizer repos | PORT-CPU | C4 |
| tests/tokenizers_/test_deepseek_v41.py | 11 | DeepSeek V4.1 chat-template tokenizer (fake HF) | yes | deepseek-ai/DeepSeek-V4.1-Flash (fixture JSON) | PORT-CPU | C4/C1 |
| tests/tokenizers_/test_deepseek_v4.py | 20 | DeepSeek V4 chat parsing with FakeHfTokenizer | yes | none (fake tokenizer) | PORT-CPU | C4/C1 |
| tests/tokenizers_/test_detokenize.py | 7 | incremental detok vs transformers backends | yes | bloom/gpt-j/pythia/opt/llama tokenizer repos | PORT-CPU | C4 |
| tests/tokenizers_/test_hf.py | 2 | HF tokenizer wrapper, pickling | yes | tokenizer repos (small) | PORT-CPU | C4 |
| tests/tokenizers_/test_mistral.py | 26 | Mistral tokenizer, llguidance grammars | yes | mistralai repos (tokenizer only) | PORT-CPU | C4/C2 |
| tests/tokenizers_/test_registry.py | 6 | tokenizer registry/resolve args idempotence | yes | opt-125m, Mistral-Nemo tokenizer repos | PORT-CPU | C4 |
| tests/samplers/__init__.py | 0 | pkg init | – | none | SKIP | – |
| tests/samplers/test_beam_search_online.py | 3 | BeamSearchOnlineMixin with mock engine client | yes | none (mocked) | PORT-CPU | C3 |
| tests/samplers/test_beam_search.py | 5 | HF vs vLLM beam search parity | no | TinyLlama-1.1B, Qwen2-Audio pins | DEFER-GPU | C3 |
| tests/samplers/test_ignore_eos.py | 1 | ignore_eos incl. generation_config EOS | no | meta-llama/Llama-3.2-1B pin | DEFER-GPU | C3 |
| tests/samplers/test_logprobs.py | 1 | logprob ranks via vllm_runner | no | runner model | DEFER-GPU | C3 |
| tests/samplers/test_no_bad_words.py | 2 | bad_words filtering | no | tokenizer + LLM weights | DEFER-GPU | C3 |
| tests/samplers/test_non_finite_params.py | 4 | NaN/Inf rejected by SamplingParams | yes | none | PORT-CPU | C3 |
| tests/detokenizer/__init__.py | 0 | pkg init | – | none | SKIP | – |
| tests/detokenizer/test_check_stop_strings.py | 6 | check_stop_strings unit tests | yes | none | PORT-CPU | C4 |
| tests/detokenizer/test_disable_detokenization.py | 1 | disable detok flag through LLM | no | LLM weights | DEFER-GPU | C4 |
| tests/detokenizer/test_min_tokens.py | 1 | min_tokens with FastIncrementalDetokenizer | yes | opt-125m tokenizer repo | PORT-CPU | C4 |
| tests/detokenizer/test_stop_reason.py | 1 | finish_reason=stop situations | no | vllm_model.llm generate | DEFER-GPU | C4 |
| tests/detokenizer/test_stop_strings.py | 1 | stop strings via LLM | no | meta-llama/llama-2-7b-hf | DEFER-GPU | C4 |
| tests/detokenizer/test_stop_string_while_stop_model_terminates.py | 1 | stop-string race w/ BaseIncrementalDetokenizer | yes | none (fake detok) | PORT-CPU | C4 |
| tests/watermarking/test_detection.py | 2 | GumbelWatermarkDetector | yes | none | PORT-CPU | C4 |
| tests/watermarking/test_gumbel.py | 12 | gumbel scheme math, gamma survival, sampler | partial | none (torch; imports gpu sampler module) | PORT-CPU | C4/C3 |
| tests/watermarking/test_prf.py | 3 | PhiloxPRF | yes | none (torch CPU) | PORT-CPU | C4 |
| tests/watermarking/test_watermarking_e2e.py | 1 | watermarking through full LLM generate | no | opt-125m | DEFER-GPU | C4 |
| tests/watermarking/test_watermarking.py | 42 | WatermarkConfig validation + scheme math | yes | none (SimpleNamespace/numpy) | PORT-CPU | C4 |
| tests/config/test_bailing_mtp_config.py | 2 | Bailing MTP speculative config types | yes | none (PretrainedConfig) | PORT-CPU | C8/C6 |
| tests/config/test_config_generation.py | 3 | EngineArgs generation fallback, deep_compare | yes | deepseek-ai/DeepSeek-V2-Lite (config only) | PORT-CPU | C8 |
| tests/config/test_config_utils.py | 14 | CacheConfig/SchedulerConfig utils, hashing | yes | none | PORT-CPU | C8 |
| tests/config/test_model_arch_config.py | 20 | ModelArchitectureConfig + ModelConfig integration | yes | HF config JSONs (llama-68m etc.) | PORT-CPU | C8 |
| tests/config/test_mp_reducer.py | 1 | data-parallel reducer env | yes | none (mocked AsyncLLM) | PORT-CPU | C8 |
| tests/config/test_multimodal_config.py | 31 | multimodal + EC transfer config | yes | none (MagicMock) | PORT-CPU | C8/C11 |
| tests/config/test_speculative_draft_hf_overrides.py | 7 | draft config overrides for SpeculativeConfig | yes | none (MagicMock) | PORT-CPU | C8/C6 |
| tests/config/test_speculative_draft_max_position_embeddings.py | 5 | EAGLE draft max_position_embeddings override | yes | none (PretrainedConfig) | PORT-CPU | C8/C6 |

Totals: 83 files; 76 with tests; 716 test functions. PORT-CPU (fully): 48 files. DEFER-GPU (weights/kernels): 20 files. Mixed (config-portable + weight-load portions): 8 files. SKIP infra (5 `__init__.py` + `utils.py` + `reference_mxfp4.py`): 7 files.

## Notable PORT-CPU candidates

- `tests/quantization/test_quantization_config_args.py` (22 tests) — pure QuantizationConfigArgs parsing; direct model for ds41rt's python validators.
- `tests/quantization/test_config_utils.py` (12) — quant config matching utilities, all Mock-based; near-zero port friction.
- `tests/quantization/test_online_mxfp4.py` (5) — online MXFP4 backend/kernel selection without weight loading; overlaps ds41rt FP4 format logic.
- `tests/quantization/test_gfx950_moe.py` + `test_int8_moe_oracle.py` + `test_moe_wna16.py` (32 tests) — MoE quant oracle backend dispatch, fully monkeypatched; portable C7/C10 logic.
- `tests/quantization/test_turboquant.py` (46) — KV-cache quant math on torch CPU; largest single CPU-portable quant file, maps to C5.
- `tests/quantization/test_per_token_kv_cache.py` (15) — per-token KV quant schemes with MagicMock; clean C5 port.
- `tests/watermarking/test_watermarking.py` (42) + `test_gumbel.py` (12) + `test_prf.py` (3) — watermark scheme math (Gumbel/Philox) runs on torch CPU; 57 tests, zero weight deps.
- `tests/detokenizer/test_check_stop_strings.py` (6) + `test_stop_string_while_stop_model_terminates.py` (1) — pure incremental-detokenizer stop-string logic with fake tokenizers; core C4 behavior.
- `tests/samplers/test_non_finite_params.py` (4) + `test_beam_search_online.py` (3) — sampler param validation and beam-search mixin against a mock engine; no GPU.
- `tests/config/test_config_utils.py` (14) + `test_speculative_draft_hf_overrides.py` (7) + `test_speculative_draft_max_position_embeddings.py` (5) — engine/speculative config parsing, all mocked; C8/C6 anchor set.


<!-- ===== 19-vllm-misc.md ===== -->

# 19 — vLLM misc dirs: utils_, multimodal, benchmarks, plugins, evals, basic_correctness, tools, ir, jit_monitor, cuda, rust tests, vllm_test_utils

Census slice, read-only. Counts are `def test_` occurrences (python) / `#[test]`+`#[tokio::test]` attrs (rust). Actual file counts differ from the brief: utils_ has 19 test files, multimodal 15, benchmarks 17, plugins_tests 10, evals 3 test files (gpt_oss 1 + gsm8k 2) plus configs, basic_correctness 4, tools 2, ir 2 (+`ir_test_utils.py` helper), jit_monitor 3 (+conftest), cuda 3. Rust has exactly 4 test dirs as named. `vllm_test_utils` is a support package, not tests.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| tests/utils_/test_argparse_utils.py | 22 | CLI arg parsing (FlexibleArgumentParser) | yes | transformers import (one util) | PORT-CPU | C1 |
| tests/utils_/test_async_utils.py | 2 | async iterator merge | yes | – | PORT-CPU | C1 |
| tests/utils_/test_cache.py | 2 | LRUCache util (prefix/MM cache basis) | yes | – | PORT-CPU | C5 |
| tests/utils_/test_collection_utils.py | 3 | pure-python collection helpers | yes | – | PORT-CPU | – |
| tests/utils_/test_func_utils.py | 1 | kwarg introspection | yes | – | PORT-CPU | – |
| tests/utils_/test_gc_utils.py | 3 | GC debug helpers | yes | – | PORT-CPU | – |
| tests/utils_/test_gpu_sync_debug.py | 9 | implicit-GPU-sync detection | no | CUDA | DEFER-GPU | – |
| tests/utils_/test_hashing.py | 1 | sha256 tuple hashing (cache keys) | yes | – | PORT-CPU | C5 |
| tests/utils_/test_import_utils.py | 9 | PlaceholderModule / plugin import | yes | – | PORT-CPU | C1 |
| tests/utils_/test_jsontree.py | 1 | json leaf counting | yes | – | PORT-CPU | – |
| tests/utils_/test_mem_utils.py | 3 | memory profiling w/ cudaMalloc | no | CUDA, CudaRTLibrary | DEFER-GPU | – |
| tests/utils_/test_network_utils.py | 15 | sockets, zmq paths, open ports | yes | zmq | PORT-CPU | C9 |
| tests/utils_/test_numa_utils.py | 29 | NUMA affinity / PCT gates | yes | Linux /proc, NUMA hw | PORT-CPU | C8 |
| tests/utils_/test_ray_utils.py | 1 | ray bundle sorting (mocked ray) | yes | ray API surface | SKIP-INFRA | C8 |
| tests/utils_/test_serial_utils.py | 3 | tensor↔base64 serialization (IPC) | yes | pybase64, numpy | PORT-CPU | C9 |
| tests/utils_/test_spawn_decorator.py | 4 | test-harness subprocess decorator | yes | – | SKIP-INFRA | – |
| tests/utils_/test_system_utils.py | 2 | unique paths, spawn forcing | yes | – | PORT-CPU | – |
| tests/utils_/test_tensor_schema.py | 20 | TensorSchema validation on model input types | yes | imports glm4v/phi3v/granite schemas | PORT-CPU | C11 |
| tests/utils_/test_torch_utils.py | 9 | dtype/OMP/thread helpers, KV quant dtype fns | yes | torch CPU | PORT-CPU | C5 |
| tests/multimodal/test_audio.py | 54 | audio resample/normalize (pyav/scipy/torchaudio) | yes | pyav, scipy, torchaudio | PORT-CPU | C11 |
| tests/multimodal/test_cache.py | 10 | MM processor/receiver cache, shm store | yes (cpu_test) | shm, vllm config stack | PORT-CPU | C5/C11 |
| tests/multimodal/test_embedding_shape_validation.py | 13 | parser hidden_size rejection | yes | – | PORT-CPU | C11 |
| tests/multimodal/test_embedding_shape_validation_unit.py | 18 | Image/Audio/Video EmbeddingItems validation | yes | – | PORT-CPU | C11 |
| tests/multimodal/test_gpu_ipc_memory.py | 16 | MM GPU IPC memory pool | no | CUDA, pynvvideocodec consts | DEFER-GPU | C11 |
| tests/multimodal/test_hasher.py | 15 | MultiModalHasher (sha256/blake3) | yes (cpu_test) | local assets/ | PORT-CPU | C11 |
| tests/multimodal/test_image.py | 6 | PIL image mode convert/normalize | yes (cpu_test) | PIL, assets/ | PORT-CPU | C11 |
| tests/multimodal/test_inputs.py | 2 | PlaceholderRange embed counting | yes | – | PORT-CPU | C11 |
| tests/multimodal/test_parse.py | 7 | MultiModalDataParser items | partial | 1 subtest CUDA-skipif | PORT-CPU | C11 |
| tests/multimodal/test_processing.py | 26 | processor prompt replacement/matching, hashes | yes (cpu_test) | HF processors: llava-v1.6-mistral-7b, Qwen2-VL-2B | PORT-CPU | C11 |
| tests/multimodal/test_registry.py | 3 | registry supports_multimodal, dummy-input budget | yes (cpu_test) | mocked + model context | PORT-CPU | C11 |
| tests/multimodal/test_sparse_tensor_validation_unit.py | 7 | torch.sparse invariant checks | yes | – | PORT-CPU | C11 |
| tests/multimodal/test_utils.py | 5 | mm position sorting, URL encode, batching | yes | – | PORT-CPU | C11 |
| tests/multimodal/test_vidcom2.py | 8 | video token retention/pruning masks | yes | – | PORT-CPU | C11 |
| tests/multimodal/test_video.py | 44 | video backends, decode, metadata | partial (cpu_test; some CUDA-skipif backends) | HF public asset downloads | PORT-CPU | C11 |
| tests/benchmarks/test_audio_dataset.py | 7 | bench ASR dataset sampling | yes | soundfile | SKIP-INFRA | – |
| tests/benchmarks/test_bench_startup.py | 1 | `vllm bench startup` subprocess | no | vllm CLI | SKIP-INFRA | – |
| tests/benchmarks/test_bfcl_dataset.py | 7 | BFCL dataset loading | yes | mocked HF API | SKIP-INFRA | – |
| tests/benchmarks/test_custom_dataset_chat_template_kwargs.py | 2 | chat-template kwargs propagation | yes | tokenizer stub | SKIP-INFRA | – |
| tests/benchmarks/test_custom_dataset_seed.py | 2 | dataset seed determinism | yes | gpt2 tokenizer | SKIP-INFRA | – |
| tests/benchmarks/test_custom_image_dataset.py | 8 | image dataset request shaping | yes | PIL, pybase64 | SKIP-INFRA | – |
| tests/benchmarks/test_endpoint_request_func_timing.py | 6 | TPOT/ITL timing invariants | yes | fake clock/session | SKIP-INFRA | – |
| tests/benchmarks/test_latency_cli.py | 1 | `vllm bench latency` subprocess | no | Llama-3.2-1B, GPU | SKIP-INFRA | – |
| tests/benchmarks/test_plot_filters.py | 12 | sweep plot filter ops | yes | pandas | SKIP-INFRA | – |
| tests/benchmarks/test_random_dataset.py | 12 | random dataset sample gen | yes | gpt2 tokenizer | SKIP-INFRA | – |
| tests/benchmarks/test_random_multimodal_dataset_video.py | 11 | random MM video dataset | yes | cv2, gpt2 | SKIP-INFRA | – |
| tests/benchmarks/test_rust_bench_cli_parity.py | 1 | keeps rust cli_parity snapshot current | yes | rust snapshot file | SKIP-INFRA | C1 |
| tests/benchmarks/test_sampling_params.py | 16 | get_sampling_params mapping | yes | fake tokenizer | SKIP-INFRA | C3 |
| tests/benchmarks/test_serve_cli.py | 3 | `vllm bench serve` subprocess + TLS | no | Llama-3.2-1B server | SKIP-INFRA | – |
| tests/benchmarks/test_skip_tokenizer_init.py | 1 | skip-tokenizer-init regression | yes | Prithvi-EO model id | SKIP-INFRA | – |
| tests/benchmarks/test_throughput_cli.py | 11 | throughput CLI + LoRA/MMVU dispatch | yes | gpt2 tokenizer | SKIP-INFRA | – |
| tests/benchmarks/test_txt_slices_dataset.py | 1 | txt-slices dataset creator | yes | gpt2 tokenizer | SKIP-INFRA | – |
| tests/plugins_tests/test_bge_m3_sparse_io_processor_plugins.py | 3 | BGE-M3 sparse plugin e2e | no | BAAI/bge-m3 server | SKIP-INFRA | C11 |
| tests/plugins_tests/test_colbert_query_io_processor_plugins.py | 6 | ColBERT query plugin e2e | no | colbert model server | SKIP-INFRA | C11 |
| tests/plugins_tests/test_endpoint_plugins.py | 12 | endpoint plugin hooks (fake engine + e2e HTTP) | partial | dummy endpoint plugin pkg | SKIP-INFRA | C1 |
| tests/plugins_tests/test_io_processor_plugins.py | 4 | IOProcessor resolution (mocked) | yes | – | SKIP-INFRA | C1 |
| tests/plugins_tests/test_oot_registration_offline.py | 4 | out-of-tree model registration | no | dummy_opt weights, GPU | SKIP-INFRA | – |
| tests/plugins_tests/test_oot_registration_online.py | 1 | OOT registration via server | no | GPU server | SKIP-INFRA | – |
| tests/plugins_tests/test_platform_plugins.py | 2 | platform plugin loading via runpy | no | runs example, GPU platform | SKIP-INFRA | – |
| tests/plugins_tests/test_scheduler_plugins.py | 1 | custom v1 scheduler plugin | no | opt-125m engine | SKIP-INFRA | C8 |
| tests/plugins_tests/test_stats_logger_plugins.py | 4 | stat-logger plugin discovery | partial | dummy_stat_logger pkg | SKIP-INFRA | C1 |
| tests/plugins_tests/test_terratorch_io_processor_plugins.py | 2 | terratorch IO plugin e2e | no | terratorch (quarantined), server | SKIP-INFRA | C11 |
| tests/evals/gpt_oss/test_gpqa_correctness.py | 1 | GPQA eval vs server | no | tiktoken data, GPU model list | SKIP-INFRA | – |
| tests/evals/gsm8k/test_gsm8k_correctness.py | 1 | GSM8K eval vs server | no | GPU models, config lists | SKIP-INFRA | – |
| tests/evals/gsm8k/test_gsm8k_offloading.py | 1 | GSM8K CPU-KV-offload regression (5 arch families) | no | NemotronH/Gemma4/Qwen3.5/DSV4-Flash/DSV2-Lite, GPU | SKIP-INFRA | C5 |
| tests/basic_correctness/test_basic_correctness.py | 5 | HF vs vLLM greedy output compare | no | many models, 2-GPU tests | DEFER-GPU | C1 |
| tests/basic_correctness/test_cpu_offload.py | 3 | UVA/prefetch weight offload params | no | GPU runner | DEFER-GPU | C10 |
| tests/basic_correctness/test_mem.py | 12 | cumem allocator, profiling, sleep/wake mem | no | GPU, fp8 subtests | DEFER-GPU | C10 |
| tests/basic_correctness/test_prefetch_offload.py | 1 | prefetch offload output parity | no | Llama-3.2-1B, GPU | DEFER-GPU | C10 |
| tests/tools/test_config_validator.py | 1 | AST-check @config class conventions | yes | – | SKIP-INFRA | – |
| tests/tools/test_docker_build_metadata_args.py | 13 | docker build metadata shell helper | yes | bash scripts, buildkite dir | SKIP-INFRA | – |
| tests/ir/test_op.py | 32 | vllm IR op registry, impl dispatch, fx tracing | yes | torch.fx | PORT-CPU | compiler |
| tests/ir/test_inplace_op.py | 4 | inplace overload semantics | yes | torch.fx | PORT-CPU | compiler |
| tests/jit_monitor/test_hooks.py | 22 | JIT monitor hook unit tests (mocked backends) | yes (cpu_test) | – | PORT-CPU | C1 |
| tests/jit_monitor/test_hooks_gpu.py | 3 | JIT monitor e2e w/ real Triton kernel | no | CUDA + triton | DEFER-GPU | – |
| tests/jit_monitor/test_no_runtime_jit.py | 1 | no post-warmup JIT on popular models | no | module pytest.mark.skip; LLM e2e | SKIP-INFRA | – |
| tests/cuda/test_cuda_compatibility_path.py | 13 | CUDA forward-compat LD_LIBRARY_PATH logic | yes | monkeypatched env | PORT-CPU | – |
| tests/cuda/test_cuda_context.py | 5 | CUDA driver context per-thread behavior | no | libcuda, GPU | DEFER-GPU | – |
| tests/cuda/test_platform_no_cuda_init.py | 2 | import must not init CUDA (Ray multi-GPU) | no | libcuda present | DEFER-GPU | – |
| rust/src/engine-core-client/src/tests/client.rs | ~36 | engine-core-client protocol over ZMQ vs mock engine | yes | zeromq, tokio; python_compat fixtures | PORT-CPU | C1/C9 |
| rust/src/engine-core-client/src/tests/python_compat.py | – | Python mirror of wire structs (fixture generator) | yes | msgspec, msgpack | PORT-CPU | C1 |
| rust/src/llm/tests/generate.rs | 9 | `vllm_llm` generate/stream API vs mock engine | yes | tokio, TestTokenizer | PORT-CPU | C1 |
| rust/src/chat/tests/chat.rs | 17 | chat API: render, tools, streaming, output processor | yes | TestTokenizer (fake), mock engine | PORT-CPU | C1/C12 |
| rust/src/chat/tests/roundtrip.rs | 1 (parametrized ~10) | chat-template render → parse → re-render roundtrip | yes | real HF models: Qwen3-0.6B, Qwen3.5-4B, MiniMax-M2.5/M3, DeepSeek-V4-Flash, **DeepSeek-V4.1-Flash**, V3.2-Exp, GLM-4.5, GLM-4.7-Flash | PORT-CPU | C1/C12 |
| rust/src/bench/tests/cli_parity.rs | 3 | rust `vllm bench serve` flag parity vs Python snapshot | yes | python_serve_flags.txt | SKIP-INFRA | C1 |

## vllm_test_utils (support package, not tests)

`tests/vllm_test_utils/` is an installed helper package (`setup.py`) providing:
- `monitor.py` — `monitor(measure_func)` contextmanager: sys-tracing value monitor recording measured values + stack traces of every change (used by mem-profiling tests).
- `blame.py` — `blame(func)` contextmanager: sys-tracing helper finding the first call satisfying a predicate, with stack.
These are test-scaffolding only; port only if the mem-util tests above are ported.

## Rust tests — notes for ds41rt rust api crate (C1)

All four rust test dirs are CPU-only and mock-engine based (no GPU): they spin a fake engine-core over ZMQ and assert client/protocol behavior. Highest-value for ds41rt:
- `chat/tests/roundtrip.rs` is the standout: it exercises real chat-template rendering against **DeepSeek-V4.1-Flash** plus Qwen3.5, MiniMax M2.5/M3, DSV4/V3.2-Exp, GLM-4.5/4.7-Flash — a ready-made template+tool-parser compatibility matrix directly applicable to a ds41rt rust chat API.
- `chat/tests/chat.rs` covers tool-call parsing and streaming output processing with a fake tokenizer — directly maps to C12.
- `engine-core-client/tests/client.rs` is a thorough protocol/transport (msgpack-over-ZMQ) conformance suite — relevant to C9 transport and C1 client design.
- `llm/tests/generate.rs` covers the high-level generate API surface (logprobs, finish reasons, events).

## Notable PORT-CPU candidates

- **tests/ir/test_op.py (32) + test_inplace_op.py (4)** — self-contained compiler-IR registry tests, torch.fx only, no models/GPU; closest thing to a clean unit suite in this slice.
- **tests/multimodal/test_processing.py (26)** — the core prompt-replacement/matching engine for multimodal inputs, cpu_test-marked; only pins are llava-v1.6/Qwen2-VL HF processor configs (dummy-weight loads). Core C11 logic.
- **tests/multimodal/test_cache.py (10)** — MM processor/receiver cache semantics incl. shm eviction, all cpu_test; pairs with C5 work.
- **tests/utils_/test_torch_utils.py (9) + test_cache.py (2) + test_hashing.py (1)** — small pure-CPU utilities underlying KV-cache dtype handling and cache-key hashing.
- **tests/jit_monitor/test_hooks.py (22)** — fully mocked JIT-monitor unit tests, cpu_test; portable as-is.
- **tests/cuda/test_cuda_compatibility_path.py (13)** — pure env-parsing/dedup logic with monkeypatch, no GPU needed despite the dir name.
- **rust/src/chat/tests/roundtrip.rs** — real-model chat-template/tool-parser roundtrip matrix including DeepSeek-V4.1-Flash; the single most relevant file for ds41rt rust C1/C12.
- **rust/src/engine-core-client/src/tests/client.rs (~36)** — protocol conformance over ZMQ with a mock engine; blueprint for ds41rt rust api crate transport tests.


<!-- ===== 20-sglang-unit.md ===== -->

# 20 — sglang `test/registered/unit/` inventory

Census of `/home/turq/dev/upstream-test-inventory/sglang/test/registered/unit/` (691 py files, 38 subdirs).
Method: immediate subdirs with ≤12 files get one row per file (headers skimmed); larger dirs get one
rolled row per thematic group (3–5 representatives spot-read per group). sglang's own CI registration
markers (`register_cpu_ci` / `register_cuda_ci` / `register_npu_ci`) were counted per dir and used as
the primary CPU/GPU signal — they make every verdict below defensible without reading all 691 files.
Dir-wide marker counts: sampling 4/4 cpu · managers 68/70 cpu · mem_cache 84/105 cpu ·
layers 56/74 cpu · models 48/52 cpu · multimodal 34/37 cpu · spec 22/27 cpu · model_executor 26/27 cpu ·
disaggregation 26/26 cpu · entrypoints 23/23 cpu · function_call 20/20 cpu · parser 11/11 cpu ·
constrained 7/8 cpu · npu 1/8 (npu-ci) · hardware_backend/mlx 3/19 cpu.

n = approx test functions (summed for rolled rows). Verdicts: PORT-CPU / DEFER-GPU / SKIP-OTHERMODEL / SKIP-INFRA / SKIP-DUP.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| test_cache_hit_kit_metrics.py | 5 | cache-hit metrics | yes | cpu_ci | PORT-CPU | C5 |
| test_chain_read_ratchet.py | 3 | internal read-ratchet governance | yes | cpu_ci | SKIP-DUP | C8 |
| test_checkpoint_quantization.py | 10 | checkpoint quant plumbing | yes | cpu_ci | PORT-CPU | C10 |
| test_cuda_vmm_utils.py | 4 | CUDA VMM alloc utils | no | cuda | DEFER-GPU | C5 |
| test_dsa_tilelang_fp8_validation.py | 5 | tilelang FP8 kernel validation | partial | cpu_ci, tilelang | DEFER-GPU | C10 |
| test_environ.py | 13 | env-var config surface | yes | cpu_ci | PORT-CPU | C8 |
| test_eval_accuracy_kit_sgl_eval.py | 5 | eval harness glue | yes | cpu_ci | SKIP-INFRA | — |
| test_flashinfer_sparse_mla.py | 4 | flashinfer sparse MLA kernel | no | flashinfer | DEFER-GPU | C5 |
| test_fork_test_worker.py | 1 | forked-worker test infra | yes | — | SKIP-INFRA | — |
| test_global_config_read_ratchet.py | 2 | internal read-ratchet governance | yes | cpu_ci | SKIP-DUP | C8 |
| test_model_overrides.py | 101 | model-override config plumbing | yes | cpu_ci | PORT-CPU | C8 |
| test_platform_context.py | 8 | platform detection/context | yes | cpu_ci | SKIP-INFRA | — |
| test_precision_baseline_store.py | 42 | precision baseline registry | yes | cpu_ci | PORT-CPU | C10 |
| test_pre_publish_readers.py | 3 | internal publish governance | yes | cpu_ci | SKIP-DUP | C8 |
| test_quantization_post_load.py | 7 | quant config post-load checks | yes | — | PORT-CPU | C10 |
| test_ray_driver_reads_the_bags.py | 3 | Ray driver config | yes | cpu_ci, ray | SKIP-INFRA | — |
| test_rocm_pageable_h2d_staging.py | 3 | ROCm H2D staging | no | rocm | SKIP-INFRA | — |
| test_runai_utils.py | 9 | Run:ai streamer utils | yes | cpu_ci | SKIP-INFRA | C10 |
| test_runtime_context.py | 99 | runtime context config | yes | cpu_ci | PORT-CPU | C8 |
| test_runtime_context_config_bags.py | 18 | runtime context bags | yes | cpu_ci | PORT-CPU | C8 |
| test_runtime_context_override.py | 13 | runtime context overrides | yes | cpu_ci | PORT-CPU | C8 |
| test_server_args_cli_metadata.py | 3 | server-args CLI metadata | yes | cpu_ci | PORT-CPU | C8 |
| test_server_args_migration.py | 13 | server-args migration | yes | cpu_ci | PORT-CPU | C8 |
| test_server_args_namespaces.py | 5 | server-args namespaces | yes | cpu_ci | PORT-CPU | C8 |
| test_split_attention_backend_decisions.py | 9 | attention backend selection policy | yes | cpu_ci | PORT-CPU | C8 |
| test_supplied_instance_exposure_ratchet.py | 2 | internal exposure governance | yes | cpu_ci | SKIP-DUP | C8 |
| batch_invariant_ops/test_batch_invariant_ops.py | 7 | batch-invariant math ops | yes | cpu_ci | PORT-CPU | C3 |
| batch_overlap/test_tbo_children_dummy_token_mask.py | 6 | TBO batch children masking | yes | cpu_ci | PORT-CPU | C8 |
| batch_overlap/test_tbo_cuda_graph_num_token_device.py | 3 | TBO cuda-graph token counting | partial | cpu_ci | DEFER-GPU | C8 |
| batch_overlap/test_tbo_filter_batch_marker.py | 4 | TBO filter_batch plan marker | yes | cpu_ci, CPU-only per docstring | PORT-CPU | C8 |
| beam_search/test_beam_search_core.py | 10 | beam search core loop | yes | cpu_ci | PORT-CPU | C3 |
| beam_search/test_fork.py | 9 | beam search batch forking | yes | cpu_ci | PORT-CPU | C3 |
| beam_search/test_output_decode.py | 4 | beam search output decode | yes | cpu_ci | PORT-CPU | C3 |
| bench/test_mixed_prefix_gsm8k.py | 3 | benchmark harness (GSM8K) | no | bench runner | SKIP-INFRA | — |
| bench/test_mmmu_eval_utils.py | 3 | benchmark eval utils | yes | — | SKIP-INFRA | — |
| bench/test_simple_eval_gsm8k.py | 3 | benchmark harness (GSM8K) | no | subprocess | SKIP-INFRA | — |
| checkpoint_engine/test_checkpoint_engine_worker.py | 5 | checkpoint-engine device routing | yes | cpu_ci, mocks | SKIP-INFRA | C10 |
| cli/test_serve_backends.py | 12 | CLI serve backend dispatch | yes | cpu_ci, mocks | PORT-CPU | C8 |
| compilation/test_torch_compile_decoration.py | 3 | torch.compile decoration gating | yes | cpu_ci, mocks | PORT-CPU | C8 |
| configs (generic: model_config{,_scaling,_shapes}, parser_registry, embedding_model_spec, linear_attn_model_registry, multimodal_piecewise_cuda_graph) | 53 | model config parsing/validation | yes | cpu_ci 13/13 | PORT-CPU | C8 |
| configs (model-specific: cohere2_moe, laguna, locate_anything, minicpm, nano_nemotron_vl, zaya) | 41 | per-model config quirks | yes | cpu_ci | SKIP-OTHERMODEL | C8 |
| constrained/test_base_grammar_backend.py | 33 | grammar backend base/registry | yes | cpu_ci | PORT-CPU | C2 |
| constrained/test_e2e_constrained_reasoning.py | 7 | constrained reasoning e2e (mocked) | yes | cpu_ci | PORT-CPU | C2 |
| constrained/test_grammar_manager.py | 46 | grammar manager lifecycle | yes | cpu_ci | PORT-CPU | C2 |
| constrained/test_llguidance_batched_mask.py | 4 | llguidance batched masks | yes | llguidance lib | PORT-CPU | C2 |
| constrained/test_mistral_common_xgrammar.py | 4 | mistral-common/xgrammar compat | yes | xgrammar | PORT-CPU | C2 |
| constrained/test_reasoner_grammar_backend.py | 27 | reasoner grammar wrapper | yes | cpu_ci | PORT-CPU | C2 |
| constrained/test_token_filter_ops.py | 9 | token filter tensor ops | yes | cpu_ci | PORT-CPU | C2 |
| constrained/test_utils.py | 9 | constrained misc utils | yes | cpu_ci | PORT-CPU | C2 |
| disaggregation/ — wire: disaggregation_wire, dcp_pack, kv_events, register_to_bootstrap, decode_queue_cleanup | 88 | PD wire protocol, conn args, staging | yes | cpu_ci 26/26, mocks | PORT-CPU | C9 |
| disaggregation/ — encode/decode lifecycle: encode_{receiver,server,scheduler}, encoder_health, decode_{hicache_tree_core,req_to_token_pool}, deferred_decode_kv_release, prefill_abort_result_cleanup, kimi_k3_encoder_mode, minimax_sparse_disagg_state | 165 | prefill/decode scheduler, receiver, KV release | yes | cpu_ci, mocks | PORT-CPU | C9 |
| disaggregation/ — mooncake + nixl: mooncake_{custom_mem_pool_batch,transfer_batching}, nixl_{backend_basic,deferred_kv_release,sender_failure_cleanup} | 71 | external KV transfer engines | no | mooncake/nixl libs | SKIP-INFRA | C9 |
| disaggregation/ — kv-transfer misc: pp_hybrid_kv_transfer, kv_transfer_replica_metric, specv2_kvcache_offloading, staging_draft_kv_slots, unified_memory_move_gate | 48 | KV transfer batching/offload gates | yes | cpu_ci | PORT-CPU | C9 |
| distributed/test_cuda_wrapper.py | 2 | device wrapper routing | yes | cpu_ci | PORT-CPU | C9 |
| distributed/test_custom_all_reduce_v2_capability.py | 1 | custom all-reduce capability | no | cuda | DEFER-GPU | C9 |
| distributed/test_gated_launch.py | 7 | distributed launch gating | yes | cpu_ci | PORT-CPU | C9 |
| distributed/test_get_default_distributed_backend.py | 4 | backend selection | yes | cpu_ci | PORT-CPU | C9 |
| distributed/test_mmencoder_forwards_attn_cp_size.py | 1 | mm-encoder CP size plumbing | partial | mocks | DEFER-GPU | C11 |
| distributed/test_parallel_state.py | 5 | parallel state init | yes | cpu_ci | PORT-CPU | C9 |
| eplb/test_balanced_packing.py | 11 | expert-load packing | yes | cpu_ci | PORT-CPU | C7 |
| eplb/test_compute_logical_to_rank_dispatch_physical_map.py | 11 | EP rank/dispatch mapping | yes | cpu_ci | PORT-CPU | C7 |
| eplb/test_dispatch_dtype_preservation.py | 9 | dispatch dtype semantics | yes | cpu_ci | PORT-CPU | C7 |
| eplb/test_waterfill_eplb.py | 4 | waterfill EPLB algorithm | yes | cpu_ci | PORT-CPU | C7 |
| entrypoints/test_grpc_bridge.py | 2 | gRPC bridge runtime handle | yes | cpu_ci | PORT-CPU | C9 |
| entrypoints/test_http2_server_config.py | 1 | HTTP/2 server config | yes | cpu_ci | SKIP-INFRA | C1 |
| entrypoints/test_http_server_warmup.py | 0 | HTTP warmup harness | yes | — | SKIP-INFRA | C1 |
| entrypoints/test_rust_server_dp_ports.py | 3 | rust server DP port alloc | yes | cpu_ci | SKIP-INFRA | C1 |
| entrypoints/test_server_info.py | 30 | server info reporting | yes | cpu_ci | SKIP-INFRA | C1 |
| entrypoints/test_server_warmup.py | 3 | server warmup | yes | cpu_ci | SKIP-INFRA | C1 |
| entrypoints/test_ssl_cert_refresher.py | 6 | TLS cert reload | yes | cpu_ci | SKIP-INFRA | C1 |
| entrypoints/test_v1_loads_aggregate.py | 5 | load aggregation | yes | cpu_ci | SKIP-INFRA | C1 |
| entrypoints/anthropic/test_serving.py | 61 | Anthropic-protocol serving | yes | cpu_ci | SKIP-OTHERMODEL | C1 |
| entrypoints/openai/ — protocol core: protocol, responses_protocol, responses_custom_tools, audio_chunking, matched_stop, utils | 92 | OpenAI pydantic protocol models | yes | cpu_ci | PORT-CPU | C1 |
| entrypoints/openai/ — serving: serving_{chat,completions,embedding,responses,responses_stream,transcription}, transcription_adapters, whisper_adapter, exa_search | 305 | OpenAI endpoint handlers (mocked) | yes | cpu_ci | PORT-CPU | C1 |
| function_call/ (20 files: hermes/deepseekv4/dots/glm47/hunyuan/k2_v3/kimik3/llama32/minicpm5/minimax_m3/mistral/muse_glimmer/poolside_v1/spark25 detectors, parser, json_schema_constraint, normalize_schema_types, parallel_tool_calls, unknown_tool_name) | 581 | tool-call detectors + parser | yes | cpu_ci 20/20 | PORT-CPU | C12 |
| hardware_backend/mlx/ (19 files: attention patching, swiglu, quantization, runtime, scheduler mixin, swa radix pool, tp routing, windowed kv cache…) | ~223 | Apple-MLX backend | partial | mlx, 3/19 cpu_ci | SKIP-INFRA | — |
| layers/ — top-level logic (attn_residual, dsv4_kv_splits_heuristic, dsv4_nonpaged_indexer, flashinfer_comm_fusion, fp8_bpreshuffle_scale, gdn_mis_metadata, kda_decode_mtp_slot_stride, layer_communicator_fusion_gate, layer_scatter_modes, logprob_chunk_stitching, logprob_fast_input, mamba2_track_ssm_indices, minicpm_attention_adapter, minicpm_sparse_{cache,metadata}, moriep_mxfp8_dispatch, mova, pooler_score_and_pool, radix_attention, radix_linear_attention) | 159 | radix/linear-attn layer logic, logprob path | yes | mostly cpu_ci | PORT-CPU | C5+C3 |
| layers/ — top-level triton/kernels (flashattention_paged_mha, mamba_state_scatter_triton, fp8_bpreshuffle_{dense_linear,producer}_mi35x, layernorm_sp, conv_layer) | 58 | GPU kernels (triton/AMD) | no | cuda/hip | DEFER-GPU | C10 |
| layers/attention/ — logic (dsa_{head_gate_guard,mqa_logits_chunking}, encoder_decoder_varlen_gather, flashattention_{graph_metadata,pa_swa_prefill_lens_size}, gdn_{flashinfer_alignment,prefill_backend_policy}, index_topk_share, kv_translate_ownership, linear_attn_config, mamba_track_state_dtype, mla_decode_{forced_splits,geometry}, verify_mask, vision_{backend_selection,max_seqlen,seqlens,strided_qkv}) | 113 | backend metadata/geometry/selection | yes | 20/25 cpu_ci | PORT-CPU | C5+C11 |
| layers/attention/ — kernels (aiter_fp8_asm_gqa, dots_hybrid_backend, kda_helion_dispatcher, triton_{dense,mla}_prefill_gfx950, linear/kernels kda_{nvidia,ptx}) | 35 | attention kernels | no | cuda/hip/triton | DEFER-GPU | C5 |
| layers/moe/ — logic (copy_weight_views_before_h2d, fused_moe_common_utils, fused_shared_expert_scaling, hpc_ops_runner_guard, moe_runner_extensions, topk_correction_bias_cache) | 29 | MoE routing/topk/scaling logic | yes | cpu_ci | PORT-CPU | C7 |
| layers/moe/ — kernels+dispatchers (aiter_runner, deepep_v2_{buffer_lifecycle,masked_slab}, flashinfer_{a2a_wide_ep,dispatcher,megamoe}, fused_moe_{native,triton_config}, mega_moe_deepgemm_api, qwen35_flashinfer_fusion, w4afp8_deepep_{dtype,post_reorder}, w4afp8_requant_geometry) | 72 | MoE GEMM/dispatch kernels | no | cuda/hip | DEFER-GPU | C7 |
| layers/quantization/ — logic (compressed_tensors_*, fp4_kv_cache_quant_method, fp8_utils_mxfp4, humming_w4afp8_schemas, int8_linear_methods, marlin_utils_fp8, modelopt_nvfp4{,_moe_scales}, quark_config, quark_utils, unquant_apply_with_addend) | 106 | quant method/schema logic | yes | cpu_ci | PORT-CPU | C10 |
| layers/quantization/ — kernels+backends (bf16_splitk_gemm, deepgemm_ue8m0_requant, flashinfer_trtllm_fp8_fallback, fp8_blockwise_linear_backends, fp8_kernel_hip_max, fp8_moe_runner_{fallback,ownership}, mxfp4_{flashinfer_activation_prep,situ_output,situ_weight_layout,sm100_trtllm_gen,sm120_cutlass,sm90_cutlass}, nvfp4_{linear,moe}_backends) | 69 | quant GEMM backends | no | cuda/hip/cutlass | DEFER-GPU | C10 |
| lora/ — sgl-marlin kernels (experimental_sgl_marlin_{multi_prefill,policy,runtime_unit,shared_outer_reduce}) | 7 | marint LoRA kernels | no | cuda | DEFER-GPU | C10 |
| lora/ — logic (eviction_policy, inkling_linearized_lora_unit, laguna_hidden_dim_unit, lm_head_pruning, lora_manager_tied_lm_head, lora_moe_inplace_unit, lora_spec_verify_batch_info, mem_pool_ep_unit, uno_{inactive_lora_batch,lora_targets}) | 70 | LoRA manager/pool/batch logic | yes | cpu_ci | PORT-CPU | C7 |
| managers/scheduler_components/ (dp_attn, invariant_checker, output_sender) | 10 | scheduler subcomponents | yes | cpu_ci | PORT-CPU | C8 |
| managers/ — scheduler core (scheduler_*, schedule_batch_*, schedule_policy*, prefill_adder, prefill_delayer, retraction_order, priority_scheduling_disaggregation) | 163 | admission, scheduling policy, batch prep | yes | cpu_ci | PORT-CPU | C8 |
| managers/ — batch result + streamers (batch_result_processor_*, customized_info_streaming, output_streamer_*) | 29 | batch result processing, streaming | yes | cpu_ci | PORT-CPU | C8 |
| managers/ — multimodal (mm_embedding_length, mm_embed_scatter, mm_hashes, mm_process_config, mm_shm_error_consensus, mm_utils_split, multimimodal_abort_cleanup, embed_overrides) | 120 | mm input accounting/embed scatter | yes | cpu_ci | PORT-CPU | C11 |
| managers/ — tokenizer manager (multi_tokenizer_mixin, tokenizer_config_updates, tokenizer_manager_rid_cleanup) | 44 | tokenizer manager lifecycle | yes | cpu_ci | PORT-CPU | C4 |
| managers/ — spec/stop handling (finish_length_speculative, grammar_stop_speculative, stop_str_speculative, trim_matched_stop, vocab_boundary_finish, uno_request_validation, uno_token_accounting) | 55 | speculative finish + stop-string semantics | yes | cpu_ci | PORT-CPU | C6 |
| managers/ — io/comm/misc (io_struct, msgpack_ipc_roundtrip, fanout_communicator, loadstat_wire, data_parallel_controller, detailed_annotations, flat_raw_top_logprobs, generation_auxiliary_output, hidden_state_server_mode, hisparse_unit, kv_page_invariants, lora_update_result_merge, mamba_checkpoint_depth, pp_cp_rank_offsets, profile_merger_http_api, sampling_mask_validation, load_inquirer, load_snapshot_backends) | 227 | IPC wire formats, DP control, misc mgr logic | yes | cpu_ci | PORT-CPU | C9 |
| mem_cache/ — radix core (radix_cache_unit, radix_cache_cpp_unit, radix_cache_slru_accuracy, radix_force_miss, hiradix_cache_unit, pure_swa_radix_cache, pure_swa_chunk_cache, decode_radix_lock_ref) | 51 | python/cpp radix tree, eviction, lock-ref | yes | cpu-runnable, some cuda_ci | PORT-CPU | C5 |
| mem_cache/ — rust tree core (rust_tree_core{,_integration}, rust_unified_radix_cache_{unittest,bench}, tree_core_registry, rust_unified_tree_core_inspector) | 118 | **Rust TreeCore radix backend** | yes | cargo build | PORT-CPU | C5 |
| mem_cache/ — hicache (hicache_{dcp_host_pool,file_lru_unit,host_register,load_back_timing,nixl_cleaner,nixl_storage,staged_write_back_dispatch,dispatch…} + dsv4_hicache_l2, unified_radix_hicache_dispatch) | 106 | host-cache L2, file LRU, write-back | yes | mostly cpu_ci | PORT-CPU | C5 |
| mem_cache/ — swa pools (swa_alloc_extend_page_estimation, swa_cpu_copy_filter, swa_eviction_boundary, swa_locked_full_recover_unified, swa_lock_release_lifecycle, swa_pool_v_head_dim, swa_ring_page_return, swa_unittest, unified_swa_shared_virtual_ids) | 80 | sliding-window pool lifecycle | yes | mixed | PORT-CPU | C5 |
| mem_cache/ — unified pools (unified_{byte_accounting,byte_budget_sizing,cache_linker,capacity_memo,free_no_host_sync,handout_zeroing,mamba_views,mha_views,mla_block_table,mla_gpu_parity,mla_views,npool_sweep,radix_allocation_eviction,radix_cache_unittest,radix_cache_bench,radix_hicache_dispatch,radix_lock_ref,tri_pool} + inspectors) | 430 | unified MHA/MLA/SWA pool views + radix alloc/evict | mostly | 84/105 dir cpu_ci | PORT-CPU | C5 |
| mem_cache/ — state-space pools (mamba_unittest, mamba_{donated_alloc_ratio,path_state_cap,state_transfer_buffers}, kda_fused_decode_strided_state, flashkda_strided_state_access, inkling_sconv_strided_conv_state, replayssm_ring_accounting, retraction_mamba_backup) | 53 | mamba/KDA conv-state pools | yes | cpu_ci | SKIP-OTHERMODEL | C5 |
| mem_cache/ — host allocators (asymmetric_mha_pool_host, dsa_pool_host, mla_host_dedup_primitives, mem_pool_host, minimax_sparse_pool_host, umbp_host_allocator, mmap_allocator, buffer_mode_sidecar) | 114 | host-side alloc/dedup | yes | cpu_ci | PORT-CPU | C5 |
| mem_cache/ — paged allocators + eviction (paged_allocator_lazy_release, paged_free_segment, free_kv_row_coalesce, evict_policy, multi_ended_allocator, page_{interleave_shard,major_layout}, full_loc_fast_path, hisparse_allocator, hisparse_max_token_pool_size) | 276 | page alloc/eviction/layout | yes | cpu_ci | PORT-CPU | C5 |
| mem_cache/ — quant/DSv4/MLA pools (dsv4_{c4_state_lifecycle,compressed_pools,compress_write_pad,unified_fp8_pool}, dsa_layer_{shard_utils,split_broadcast}, mla_host_dedup, mxfp8_scale_transfer_buffers, qsa_kv_pool, quantized_kv_pool) | 49 | DSv4-family quantized pools | yes | cpu_ci | SKIP-OTHERMODEL | C5 |
| mem_cache/ — mooncake (mooncake_group_semantics, mooncake_standalone_dummy_mamba, mooncake_tenant_config) | 18 | mooncake store semantics | yes | cpu_ci, mocks | SKIP-INFRA | C9 |
| mem_cache/ — misc (mem_cache_utils, registry, kv_index_translator, layout_compat, linker_pool_assembler, hybrid_pool_assembler, session_token_share_unit, session_unified_radix_cache, streaming_session_unit, decode_retraction_backup, dllm_fdfo_kv_reuse, pd_envelope_transfer_layout) | 39 | registry/session/kv-index plumbing | yes | cpu_ci | PORT-CPU | C5 |
| model_executor/ — top-level logic (chunked_prefix_cache_gate, draft_runner_skips_lora, forward_metadata_plan_record, hisparse_pool_configurator, kv_canary_headroom, mlp_sync_pad_unpad, model_runner_decode_rows, num_token_non_padded_localization, pool_configurator, unified_out_cache_loc_rebind) | 76 | runner metadata/pool config | yes | cpu_ci | PORT-CPU | C8+C5 |
| model_executor/ — cuda-graph runner tests (prefill_cuda_graph_runner{,_helpers}, cuda_graph_buffer_registry) | 63 | prefill/decode cuda graph runners | no | cuda | DEFER-GPU | C8 |
| model_executor/runner{,_backend,_utils}/ (decode/prefill graph runners, shared_read_fence, flashinfer_autotune_sync, hidden_state_graph_recapture, graph_pool_borrow, full_cuda_graph_backend) | 61 | cuda graph capture/runner internals | partial | cpu_ci but graph-heavy | DEFER-GPU | C8 |
| model_executor/model_runner_components/ (attention_backend_setup, cuda_graph_setup, layer_setup, ngram_embedding_manager, spec_aux_hidden_state, startup_weight_load) | 36 | runner component setup | yes | cpu_ci | PORT-CPU | C8+C6 |
| models/ — deepseek (deepseek_mla_dispatch, deepseek_nextn_mm_embed, deepseek_v4_{amd_fused_mhc,amd_wo_a_bf16,mxfp4_shared_expert_requant,rope_policy,shared_expert_fusion,unified_fp8_q_pair}) | 51 | DeepSeek-V4 layer/quant logic | yes | cpu_ci 48/52 dir | PORT-CPU | C10+C7 |
| models/ — qwen (qwen3_5_{modelopt_fp4,packed_weight_loader,pipeline_parallel}, qwen3_embedding_registration, qwen3_vl_feature_materialization, qwen4_exp_ple_table, qwen_image_fp8_norm_quant) | 55 | Qwen-family model internals | yes | cpu_ci | SKIP-OTHERMODEL | C10 |
| models/ — kimi (kimi_k25, kimi_k3_{bfa_overlap,vision}, kimi_vl{,_moonvit}) | 79 | Kimi/Kimi-VL internals | yes | cpu_ci | SKIP-OTHERMODEL | C11 |
| models/ — glm (glm5_next_{dflash_capture,modelopt}, glm_moe_gate_fp32, glmmoedsa_correction_bias_fp32) | 14 | GLM-family internals | yes | cpu_ci | SKIP-OTHERMODEL | C10 |
| models/ — nemotron (nemotron_h_{mtp,shared_add,weight_loading}) | 17 | Nemotron internals | yes | cpu_ci | SKIP-OTHERMODEL | C10 |
| models/ — other (cosmos3{,_edge}, flux2, granite split-expert, hunyuan_v3/v4, interns1pro, llava{,_processor_pool}, locate_anything, longcat, mellum, moss_vl, nano_nemotron_vl, paddleocr_vl, radio, vit_pos_embed_interpolate, xllm, zaya_{cca,mod_tp}, draft_entry_hook_parity, shared_experts_fusion_gates) | 183 | assorted model internals | yes | cpu_ci | SKIP-OTHERMODEL | C10 |
| multimodal/ — processors (base_processor_{bad_input,image_decode}, pixtral, nano_nemotron_vl_processor, glm4v_mixed_offsets, dots_note_omni, evs, media_artifact_processor, mrope_encoder_utils, preprocess_cache, precomputed_embedding_validation, processor_{async_call_sites,clone_isolation,device_selection}, feature_materialization) | 112 | image decode/processor logic | yes | cpu_ci | PORT-CPU | C11 |
| multimodal/ — transports (cuda_ipc_{pool_budget,transport}, cuda_vmm_transport, gpu_feature_transport, tensor_transport_mode, vit_{cuda_graph_metadata_cuda,cuda_graph_runner,npu_graph_runner}, kimi_k3_gpu_preprocess) | 70 | GPU tensor transports/graphs | no | cuda | DEFER-GPU | C11 |
| multimodal/ — misc (audio_container_decode, media_url_security) | 18 | audio decode, URL security | yes | cpu_ci | PORT-CPU | C11 |
| multimodal/rust/ (shared: fetch, image_decode, partition_cores, rust_mm_gate, wrap_encoded; qwen: driver, e2e_parity, preprocess, rust_mm_processor, token_layout_mrope; inkling: bindings) | 35 | **Rust mm processors** | yes | cargo build; qwen/inkling model-specific | SKIP-OTHERMODEL | C11 |
| npu/ (attention ascend backends + mla_preprocess; quantization fp4_moe, mxfp8_linear; arch35 capability, sparsity offload config) | 238 | Ascend-NPU backend | no | npu_ci 8 files | SKIP-INFRA | — |
| observability/ (cpu_monitor, forward_pass_metrics, func_timer, label_transform, metrics_utils, ray_wrappers, req_time_stats, request_metrics_exporter, scheduler_stage_metrics, startup_func_log_and_timer, stat_loggers_di, trace) | 178 | metrics/logging infra | yes | cpu_ci 12/12 | SKIP-DUP | — |
| parser/test_code_completion_parser.py | 13 | code-completion parsing | yes | cpu_ci | PORT-CPU | C4 |
| parser/test_conversation.py | 76 | chat-template conversation | yes | cpu_ci | PORT-CPU | C4 |
| parser/test_harmony_parser.py | 43 | harmony format parsing | yes | cpu_ci | PORT-CPU | C4 |
| parser/test_hunyuan_reasoning.py | 1 | hunyuan reasoning tags | yes | cpu_ci | SKIP-OTHERMODEL | C4 |
| parser/test_inkling_renderer.py | 14 | inkling template render | yes | cpu_ci | SKIP-OTHERMODEL | C4 |
| parser/test_jinja_template_utils.py | 26 | jinja template utils | yes | cpu_ci | PORT-CPU | C4 |
| parser/test_k2_v3_reasoning_parser.py | 9 | K2-V3 reasoning parser | yes | cpu_ci | SKIP-OTHERMODEL | C4 |
| parser/test_kimik3_reasoning_parser.py | 25 | Kimi-K3 reasoning parser | yes | cpu_ci | SKIP-OTHERMODEL | C4 |
| parser/test_reasoning_content_without_parser.py | 1 | reasoning passthrough | yes | cpu_ci | PORT-CPU | C4 |
| parser/test_reasoning_parser.py | 134 | generic reasoning detectors | yes | cpu_ci | PORT-CPU | C4 |
| parser/test_template_manager.py | 48 | chat template manager | yes | cpu_ci | PORT-CPU | C4 |
| plugins/test_hook_registry.py | 20 | plugin hook registry | yes | cpu_ci | SKIP-DUP | C8 |
| plugins/test_load_plugins.py | 6 | plugin loading | yes | cpu_ci | SKIP-INFRA | C8 |
| platforms/test_mps_triton_stub.py | 2 | Apple MPS triton stub | yes | — | SKIP-INFRA | — |
| platforms/test_platform_interface.py | 72 | platform abstraction | yes | cpu_ci | SKIP-INFRA | — |
| sampling/test_custom_logit_processor.py | 36 | custom logit processors | yes | cpu_ci | PORT-CPU | C3 |
| sampling/test_penaltylib.py | 41 | presence/frequency penalties | yes | cpu_ci | PORT-CPU | C3 |
| sampling/test_sampling_batch_info.py | 46 | sampler batch state/merge | yes | cpu_ci, CPU per docstring | PORT-CPU | C3 |
| sampling/test_sampling_params.py | 76 | sampling params validation | yes | cpu_ci | PORT-CPU | C3 |
| scripted_runtime/test_background_http_poster.py | 9 | scripted-runtime http poster | yes | cpu_ci | SKIP-INFRA | C9 |
| scripted_runtime/test_http_server.py | 7 | scripted-runtime http server | yes | cpu_ci | SKIP-INFRA | C9 |
| scripted_runtime/test_scheduler_hook.py | 3 | scripted-runtime scheduler hook | yes | cpu_ci | SKIP-INFRA | C8 |
| scripted_runtime/test_scripted_runtime_utils.py | 11 | scripted-runtime utils | yes | cpu_ci | SKIP-INFRA | — |
| scripted_runtime/test_tokenizer_recv_proxy.py | 9 | tokenizer recv proxy | yes | cpu_ci | SKIP-INFRA | C4 |
| server_args/ — core + gates (server_args[235], unified_prefill_cuda_graph_gate, unified_tbo_gate, page_major_backend_allowlist, platform_prefill_cp_deprecation) | 255 | server-arg parsing/validation/gates | yes | cpu_ci | PORT-CPU | C8 |
| server_args/ — resolution framework (declared_fallbacks, model_config_cache, model_source_paths, record_holds_the_raw_input, resolution_{declarations,hook_registry,is_reproducible,reads_the_declarations}) | 55 | internal arg-resolution machinery | yes | cpu_ci | SKIP-DUP | C8 |
| spec/ — eagle + draft construction (eagle_{seeded_coins,worker_v2_topk1_fastpath,draft_extend_logits}, draft_construction_isolation, draft_per_runner_config) | 43 | EAGLE draft logic | yes | cpu_ci | PORT-CPU | C6 |
| spec/ — eagle cuda-graph runner (eagle_draft_cuda_graph_runner) | 3 | EAGLE graph capture | no | cuda | DEFER-GPU | C6 |
| spec/ — dflash/dspark (dflash_{domino,extra_buffer_lazy,logits,overlap_hostsync}, dspark_target_hidden_projection) | 38 | dflash draft algorithm | yes | cpu_ci | PORT-CPU | C6 |
| spec/ — uno tree (uno_{request_validation,tree_config,tree_sparse_sampling}) | 6 | uno tree config/validation | yes | cpu_ci | PORT-CPU | C6 |
| spec/ — ngram (ngram_corpus, ngram_mamba_verify_update) | 56 | ngram spec decode | yes | cpu_ci | PORT-CPU | C6 |
| spec/ — misc (adaptive_{runtime_state,spec_params}, decode_bookkeeping_ownership, decoupled_spec_io, fast_prefill_plan, plugin_hook_signatures, spec_registry, spec_utils_traverse_tree, suffix_attention_merge_dispatch, resolve_swa_kv_pool, spec_cpu_overlap_constraint) | 107 | spec algorithm registry/plumbing | yes | cpu_ci | PORT-CPU | C6 |
| state_capturer/test_routed_experts_scattered_a2a.py | 4 | DeepEP backend recognition | yes | cpu_ci, mocks | DEFER-GPU | C9 |
| tokenizer/test_mistral_empty_assistant.py | 2 | mistral template edge case | yes | cpu_ci | SKIP-OTHERMODEL | C4 |
| tokenizer/test_tekken_tokenizer_routing.py | 2 | tiktoken routing | yes | cpu_ci | PORT-CPU | C4 |
| tokenizer/test_tiktoken_tokenizer.py | 12 | tiktoken tokenizer | yes | cpu_ci | PORT-CPU | C4 |
| tools/ (amd_ci_install_dependency, docker_build_metadata_args, get_version_tag, slash_command_handler) | 15 | CI/release tooling | yes | cpu_ci | SKIP-INFRA | — |
| utils/ — generic (common, auth, http_server_auth, field_validators, invariants, json_response, patch_tokenizer, profile_merger, subprocess_watchdog, tensor_bridge, weight_checker{,_comparator}, weight_versions, gauge_histogram, diffusion_torch_fallback) | 251 | shared utils, weight checkers | yes | cpu_ci | PORT-CPU | C10+C4 |
| utils/ — hf_transformers (hf_transformers{,_fastokens,_loading}) | 72 | HF-transformers compat patches | yes | cpu_ci | SKIP-OTHERMODEL | C4 |

## Notable PORT-CPU candidates

- **mem_cache rust tree core + hicache (n≈224 combined)** — the Rust TreeCore radix backend and host-cache L2 (file LRU, staged write-back) are the closest existing analogue of ds41rt-hostcache (C5); CPU-only, cargo build pin.
- **mem_cache unified pools + paged allocators (n≈706)** — unified MHA/MLA/SWA pool views, radix alloc/eviction, page layout/eviction-policy logic; all pure-python/torch-CPU, ~84/105 files cpu_ci-registered (C5).
- **mem_cache radix core (n≈51)** — python RadixCache insert/match/evict/lock-ref semantics; direct prefix-reuse reference for ds41rt (C5).
- **managers scheduler core (n≈163)** — admission, HRRN/DFS schedule policy, prefill adder/delayer, retraction order; exactly ds41rt scheduler semantics (C8).
- **sampling/ (n≈199)** — sampling params validation, penalty lib, batch-info merge/fork; all four files explicitly cpu_ci "no server, no model loading" (C3).
- **spec/ (n≈250 logic)** — ngram corpus/verify, EAGLE draft logic, dflash/dspark, adaptive spec params, spec registry; only the cuda-graph-runner files need GPU (C6).
- **disaggregation wire + encode/decode lifecycle (n≈253)** — PD wire protocol, conn args, staging allocator, encode scheduler, deferred KV release; 26/26 cpu_ci-registered — the prime disagg-serving port (C9).
- **entrypoints/openai protocol + serving (n≈397)** — pydantic protocol models + mocked serving_chat/completions/responses handlers (C1).
- **function_call/ (n≈581)** — 20 detector/parser files, 20/20 cpu_ci; largest single CPU-verdict block in the tree (C12).
- **constrained/ (n≈139)** — grammar backend base/registry, grammar manager, token-filter ops — structured-output stack (C2).
- **parser/ generic files (n≈298: conversation, reasoning_parser, template_manager, jinja utils)** — chat-template and reasoning-detector semantics directly reusable (C4).
- **models deepseek (n≈51)** — DeepSeek-V4 RoPE policy, shared-expert fusion, FP8/quant plumbing: same architecture family ds41rt targets; cpu_ci-registered (C10+C7).


<!-- ===== 21-sglang-serving.md ===== -->

# 21 — sglang serving test inventory

Census of `sglang/test/registered/` serving-side dirs for the DS41RT coverage map.
Source tree read-only; one row per file (dirs ≤12 files) or rolled rows per subdir
(spec/, kv_canary/, and dirs with subdirs). `n` = `grep -c "def test_"` (n=0 files
usually compose test classes from shared kit mixins). CPU determination is anchored
on sglang's own CI registration (`sglang.test.ci.ci_register`): `register_cpu_ci` =
runs in the CPU (stage-b cpu-intel) suite; `register_cuda_ci`/`register_amd_ci` =
GPU suite; several GPU-registered unit files were spot-checked for `.cuda()` usage.

Verdicts: PORT-CPU / DEFER-GPU / SKIP-OTHERMODEL / SKIP-INFRA / SKIP-DUP.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| radix_cache/test_int8_mamba_checkpoint_e2e.py | 1 | mamba int8 checkpoint pool | no | GDN-hybrid mamba model, GPU server | SKIP-OTHERMODEL | C5 |
| radix_cache/test_mamba2_extra_buffer_kl.py | 0 | mamba2 extra_buffer KL | no | NemotronH mamba, GPU | SKIP-OTHERMODEL | C5 |
| radix_cache/test_radix_attention.py | 1 | radix cache + attention | partial | small model server (cpu-intel suite too) | DEFER-GPU | C5 |
| radix_cache/test_radix_cache_hit.py | 1 | multiturn radix cache-hit | yes | small model server; CPU-CI registered | DEFER-GPU | C5 |
| radix_cache/test_swa_radix_cache_kl.py | 0 | SWA radix KL | partial | SWA model, in-process KL harness | DEFER-GPU | C5 |
| radix_cache/unified_radix_tree/linker/ (2 files) | 10 | unified-tree direct-linker load-back | no | dsv4 / glm52, GPU server | DEFER-GPU | C5 |
| radix_cache/unified_radix_tree/test_unified_radix_cache_kl_*.py (11 files) | 11 | unified radix tree bitexact/KL guards | no | dsv4, glm52, mamba, mimo, SWA, MiMo pins | DEFER-GPU | C5 |
| hicache/test_hicache_spec_file_storage.py | 1 | hicache file storage + EAGLE3 | no | GPU server, file backend | DEFER-GPU | C5 |
| hicache/test_hicache_spec_mooncake_storage.py | 2 | hicache mooncake + EAGLE3 | no | GPU server, mooncake | SKIP-INFRA | C5 |
| hicache/test_hicache_storage_3fs_backend.py | 1 | hicache 3fs backend | no | 3FS store | SKIP-INFRA | C5 |
| hicache/test_hicache_storage_file_backend.py | 2 | hicache file backend e2e | no | GPU server (amd+cuda) | DEFER-GPU | C5 |
| hicache/test_hicache_storage_mooncake_backend.py | 1 | hicache mooncake backend | no | mooncake store | SKIP-INFRA | C5 |
| hicache/test_hicache_storage.py | 0 | hicache storage suite | no | GPU server | DEFER-GPU | C5 |
| hicache/test_hicache_storage_runtime_attach_detach.py | 2 | hicache runtime attach/detach | no | GPU server | DEFER-GPU | C5 |
| hicache/test_hicache_storage_umbp_backend.py | 1 | hicache UMBP backend | no | DeepSeek-V4 UMBP, amd-only CI | DEFER-GPU | C5 |
| hicache/test_hicache_variants.py | 0 | hicache config variants | no | GPU server | DEFER-GPU | C5 |
| hicache/test_pp_with_hicache.py | 1 | pipeline-parallel + hicache | no | GPU server | DEFER-GPU | C5 |
| hicache/test_qwen35_hicache.py | 2 | qwen3.5 hicache | no | Qwen3.5 pin | SKIP-OTHERMODEL | C5 |
| kv_canary/test_e2e_base.py | 2 | canary e2e harness base | yes | none (pure harness) | PORT-CPU | C5 |
| kv_canary/test_self_e2e_*.py (12 files) | 15 | mock-model canary e2e (baseline/perturb/PD/PP/PR-regressions) | no | mock weights, GPU runner | DEFER-GPU | C5 |
| kv_canary/test_self_unit_{capacities,e2e_base,plan_input,pool_patcher_utils,radix_walker,sweep_plan_builder,token_oracle,violation,pool_patcher,req_to_expected_token_ids_manager}.py (10 files) | 82 | canary units: config, planner, tree walk, violation logic | partial | torch CPU tensors, mocks; no `.cuda()` | PORT-CPU | C5 |
| kv_canary/test_self_unit_{buffer_alloc,endpoint,future_tensor,perturb,runner_health,runner_per_forward,runner_swa_divergence,runner_sweep}.py (8 files) | 75 | canary units: pool/tensor plumbing, per-forward runner | partial | torch CPU tensors; GPU-registered but no `.cuda()` | PORT-CPU | C5 |
| moe/test_cutedsl_moe.py | 6 | CuteDSL MoE kernels | no | B200, cuda | DEFER-GPU | C7 |
| moe/test_flashinfer_a2a_cutedsl_v2.py | 1 | FP4 MoE + flashinfer A2A | no | B200, Qwen3.5, DP attn | SKIP-OTHERMODEL | C7 |
| moe/test_flashinfer_a2a_cutlass.py | 1 | Cutlass BF16 MoE + A2A | no | B200, Qwen3.5 | SKIP-OTHERMODEL | C7 |
| moe/test_fused_append_remap_per_rank_shared_slots.py | 6 | fused append remap kernel | no | amd+cuda kernels | DEFER-GPU | C7 |
| moe/test_fused_append_shared_experts.py | 7 | fused shared-experts append | no | amd+cuda kernels | DEFER-GPU | C7 |
| moe/test_fused_append_shared_experts_top6.py | 3 | fused shared-experts top6 | no | amd+cuda kernels | DEFER-GPU | C7 |
| moe/test_fused_moe.py | 2 | fused MoE | no | cuda kernels | DEFER-GPU | C7 |
| moe/test_glm4_moe_models.py | 1 | glm4 MoE models | no | GLM pin | SKIP-OTHERMODEL | C7 |
| moe/test_hash_topk.py | 3 | hash-based top-k routing | yes | none (pure logic) | PORT-CPU | C7 |
| moe/test_hpc_ops_moe.py | 2 | HPC-ops FP8 MoE backend | no | hpc_ops backend | DEFER-GPU | C7 |
| moe/test_hybrid_dp_ep_tp_mtp.py | 60 | DP+EP+TP+MTP MoE matrix | no | GPU server, model pins | SKIP-OTHERMODEL | C7 |
| moe/test_moe_ep.py | 1 | MoE EP base | no | GPU server | DEFER-GPU | C7 |
| moe/test_moe_ep_extra.py | 1 | MoE EP FP8/DeepGEMM extra | no | GPU server | DEFER-GPU | C7 |
| moe/test_topk_padded_region.py | 9 | top-k padded region | no | amd+cuda kernels | DEFER-GPU | C7 |
| moe/test_topk_renormalize_degenerate.py | 7 | top-k renorm NaN regression | no | amd+cuda kernels | DEFER-GPU | C7 |
| moe/test_torch_compile_moe.py | 2 | torch.compile MoE | no | torch.compile | SKIP-INFRA | C7 |
| moe/test_triton_fused_moe.py | 1 | triton fused MoE | no | triton | DEFER-GPU | C7 |
| moe/test_triton_moe_channel_fp8_kernel.py | 1 | triton MoE channel FP8 | no | triton FP8 | DEFER-GPU | C7 |
| moe/test_zero_experts.py | 1 | zero-experts edge case | no | cuda | DEFER-GPU | C7 |
| spec/dspark/test_dspark_{block_accept_estimator,confidence_metrics,dp_tier,draft_path_default,info_dumper,scheduler,sps_profiler,sps_table,sts,ragged_verify}.py (10 files) | 146 | dSpark bookkeeping: SPS table/scheduler, accept estimator, STS, info dumper | yes | none (CPU-registered, torch CPU) | PORT-CPU | C6 |
| spec/dspark/test_dspark_{kernel_parity,stacked_ctx_kv_parity}.py + test_ragged_verify_backend_capability.py (3 files) | 9 | dSpark kernel/parity + backend capability | no | triton kernels, cuda | DEFER-GPU | C6 |
| spec/dflash/ (2 files) | 5 | DFLASH spec decode e2e | no | GPU server, dflash pins | DEFER-GPU | C6 |
| spec/eagle/ (13 files) | 8 | EAGLE3 spec-decode suite (chain/topk/page/parity/stress/fa3/triton) | no | EAGLE3 weights, llama pins | SKIP-OTHERMODEL | C6 |
| spec/uno/test_uno.py | 3 | UNO linear/tree decoding e2e | no | GPU server | SKIP-OTHERMODEL | C6 |
| spec/utils/test_build_eagle_tree.py | 2 | eagle tree build util | no | amd+cuda | DEFER-GPU | C6 |
| spec/test_{constrained_decoding_spec_reasoning,frozen_kv_mtp,mixed_chunk,ngram,ngram_extra,standalone,standalone_extra}.py (7 files) | 4 | model-free-ish spec paths (ngram/standalone/mixed-chunk/frozen-kv) | no | GPU server | DEFER-GPU | C6 |
| spec/test_gemma4_{dflash_31b,mtp_26b_a4b,mtp_31b}_extra.py (3 files) | 3 | gemma4 MTP/DFLASH extras | no | Gemma4 pins | SKIP-OTHERMODEL | C6 |
| sampling/test_deterministic_gumbel_u1.py | 2 | gumbel u==1.0 bucket edge (logprob path) | partial | `.cuda()` (trivial to de-GPU) | PORT-CPU | C3 |
| sampling/test_original_logprobs.py | 1 | sglang-vs-HF logprob alignment | no | GPU server + HF | DEFER-GPU | C3 |
| sampling/test_penalty.py | 10 | frequency/presence penalty | no | GPU server (amd+cuda) | DEFER-GPU | C3 |
| sampling/test_pytorch_sampling_backend.py | 2 | pytorch sampling backend | no | GPU server | DEFER-GPU | C3 |
| sampling/test_sampling_mask.py | 21 | sampling mask bookkeeping | no | GPU server (amd+cuda) | DEFER-GPU | C3 |
| eplb/test_lplb_distributed.py | 3 | LPLB solver, real multi-rank | no | multi-GPU; supplements CPU test_lplb | DEFER-GPU | C7 |
| expert_pack/test_expert_pack_mxfp4.py | 4 | MXFP4 expert-pack kernels | no | cuda | DEFER-GPU | C7 |
| expert_pack/test_expert_pack_runtime.py | 11 | expert-pack runtime logic | yes | none (CPU-registered) | PORT-CPU | C7 |
| expert_pack/test_kimi_k3_gguf.py | 9 | kimi-k3 GGUF weight layout | yes | kimi-k3 pins, CPU-registered | PORT-CPU | C7 |
| ep/test_deepep_large.py | 4 | DeepEP large | no | multi-GPU, DeepEP | DEFER-GPU | C7 |
| ep/test_deepep_small.py | 4 | DeepEP small | no | multi-GPU, DeepEP | DEFER-GPU | C7 |
| ep/test_deepep_small_extra.py | 3 | DeepEP small extra | no | multi-GPU, DeepEP | DEFER-GPU | C7 |
| ep/test_eplb_no_a2a.py | 1 | EPLB redundant experts, no A2A | no | GPU server | DEFER-GPU | C7 |
| ep/test_flashinfer_a2a.py | 3 | flashinfer all-to-all | no | multi-GPU | DEFER-GPU | C7 |
| ep/test_mooncake_ep_small.py | 3 | mooncake EP | no | mooncake store | SKIP-INFRA | C7 |
| ep/test_routed_experts_dp_readback.py | 1 | DP routed-expert readback parity | no | multi-GPU | DEFER-GPU | C7 |
| ep/test_tbo_shared_experts_fusion.py | 1 | TBO shared-experts fusion | no | GPU server | DEFER-GPU | C7 |
| disaggregation/test_disaggregation_basic.py | 14 | PD disaggregation core | no | 2-GPU server | DEFER-GPU | C9 |
| disaggregation/test_disaggregation_chunked_prefill_abort.py | 1 | PD chunked-prefill abort | no | 2-GPU server | DEFER-GPU | C9 |
| disaggregation/test_disaggregation_decode_offload.py | 1 | PD decode offload | no | 2-GPU | DEFER-GPU | C9 |
| disaggregation/test_disaggregation_decode_radix_cache.py | 3 | PD decode-side radix cache | no | 2-GPU | DEFER-GPU | C9 |
| disaggregation/test_disaggregation_decode_radix_cache_swa.py | 0 | PD decode radix cache SWA | no | gpt-oss-20b pin | SKIP-OTHERMODEL | C9 |
| disaggregation/test_disaggregation_different_tp.py | 8 | PD mixed TP | no | multi-GPU | DEFER-GPU | C9 |
| disaggregation/test_disaggregation_dp_attention.py | 1 | PD DP-attention | no | 2-GPU | DEFER-GPU | C9 |
| disaggregation/test_disaggregation_dsv4.py | 0 | PD dsv4 | no | dsv4 pin | DEFER-GPU | C9 |
| disaggregation/test_disaggregation_dwdp_gpt_oss.py | 1 | PD DWDP gpt-oss | no | gpt-oss pin | SKIP-OTHERMODEL | C9 |
| disaggregation/test_disaggregation_hisparse.py | 0 | PD hisparse | no | GLM hisparse pin | SKIP-OTHERMODEL | C9 |
| disaggregation/test_disaggregation_hybrid_attention.py | 5 | PD hybrid attention | no | hybrid-attn model | SKIP-OTHERMODEL | C9 |
| disaggregation/test_disaggregation_inkling_mxfp8.py | 0 | PD inkling MXFP8 | no | Inkling pin | SKIP-OTHERMODEL | C9 |
| disaggregation/test_disaggregation_kimi_linear.py | 0 | PD kimi-linear | no | kimi-linear pin | SKIP-OTHERMODEL | C9 |
| disaggregation/test_disaggregation_nixl.py | 4 | PD NIXL transport | no | NIXL | SKIP-INFRA | C9 |
| disaggregation/test_disaggregation_optimistic_prefill.py | 3 | PD optimistic prefill | no | 2-GPU | DEFER-GPU | C9 |
| disaggregation/test_disaggregation_pp.py | 2 | PD pipeline parallel | no | 2-GPU | DEFER-GPU | C9 |
| disaggregation/test_disaggregation_rust_server.py | 6 | PD embedded Rust server | no | rust server build | SKIP-INFRA | C9 |
| disaggregation/test_disaggregation_unified_memory.py | 0 | PD unified memory pool | no | unified-memory model pins | SKIP-OTHERMODEL | C9 |
| disaggregation/test_disaggregation_xpu.py | 2 | PD NIXL on Intel XPU | no | XPU | SKIP-INFRA | C9 |
| disaggregation/test_disaggregation_aarch64.py | 1 | PD aarch64 | no | ARM build | SKIP-INFRA | C9 |
| disaggregation/test_epd_disaggregation.py | 11 | EPD (encoder-prefill-decode) | no | multi-GPU, TBO | DEFER-GPU | C9 |
| disaggregation/test_kimi_linear_pd_dcp4.py | 5 | PD kimi-linear DCP4 | no | kimi-linear, Blackwell | SKIP-OTHERMODEL | C9 |
| prefill_only/test_embedding_models.py | 2 | embedding serving | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| prefill_only/test_multi_item_scoring.py | 20 | multi-item scoring (MIS) | no | GPU server | DEFER-GPU | C8 |
| prefill_only/test_openai_embedding.py | 10 | /v1/embeddings endpoint | partial | GPU server; also CPU-CI registered | DEFER-GPU | C8 |
| prefill_only/test_pooled_hidden_states.py | 20 | pooled hidden states scoring | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| prefill_only/test_reward_models.py | 1 | reward-model serving | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| prefill_only/test_score_api.py | 8 | /v1/score HTTP layer | no | GPU server | DEFER-GPU | C8 |
| prefill_only/test_score_engine.py | 19 | /v1/score engine pipeline | no | GPU server | DEFER-GPU | C8 |
| prefill_only/test_serving_rerank.py | 13 | rerank serving | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| chunked_prefill/test_mm_chunked_embedding_unit.py | 10 | per-item DataEmbeddingFunc | yes | none (CPU-registered) | PORT-CPU | C5 |
| chunked_prefill/test_scripted_core_1gpu.py | 7 | scripted chunked-prefill core | no | 1 GPU (amd+cuda) | DEFER-GPU | C5 |
| chunked_prefill/test_scripted_core_4gpu.py | 2 | scripted chunked-prefill 4-GPU | no | 4 GPU | DEFER-GPU | C5 |
| chunked_prefill/test_scripted_swa_1gpu.py | 1 | scripted chunked-prefill SWA | no | SWA model | SKIP-OTHERMODEL | C5 |
| cp/test_cp_strategy_unit.py | 24 | context-parallel strategy | yes | none (CPU-registered, mocks) | PORT-CPU | C9 |
| cp/test_deepseek_v3_cp_single_node.py | 1 | DSV3 CP single node | no | dsv3, GPU server | DEFER-GPU | C9 |
| cp/test_deepseek_v4_flash_fp4_b200_cp.py | 0 | DSV4-flash FP4 CP | no | B200, dsv4 | DEFER-GPU | C9 |
| cp/test_dsa_prefill_cp.py | 1 | DSA prefill CP | no | dsa kernels | DEFER-GPU | C9 |
| cp/test_gpt_oss_4gpu_mxfp4_cp.py | 1 | gpt-oss 4-GPU MXFP4 CP | no | gpt-oss | SKIP-OTHERMODEL | C9 |
| cp/test_gqa_prefill_cp.py | 3 | GQA prefill CP | no | GPU server | DEFER-GPU | C9 |
| dcp/test_dcp_layout_unit.py | 17 | DCP per-rank KV-length math | yes | none (explicit CPU unit test) | PORT-CPU | C9 |
| dcp/test_dsv31_dcp8_gsm8k.py | 3 | DSV3.1 DCP8 GSM8K | no | dsv3.1, multi-GPU | DEFER-GPU | C9 |
| dcp/test_kimi_linear_dcp4.py | 2 | kimi-linear DCP4 | no | kimi-linear, Blackwell | SKIP-OTHERMODEL | C9 |
| dcp/test_kimi_linear_dcp_dspark4.py | 1 | kimi-linear DCP + dSpark | no | kimi-linear, Blackwell | SKIP-OTHERMODEL | C9 |
| dcp/test_qwen3p5_triton_dcp.py | 1 | qwen3.5 triton DCP | no | qwen3.5 | SKIP-OTHERMODEL | C9 |
| dcp/test_reduce_scatter_along_dim.py | 1 | reduce_scatter_along_dim | no | multi-GPU | DEFER-GPU | C9 |
| dcp/test_trtllm_mla_family_dcp_metadata.py | 18 | DCP cuda-graph metadata (trtllm_mla family) | no | cuda-graph, MLA | DEFER-GPU | C9 |
| dp_attn/test_dp_attention.py | 2 | DP attention | no | 2-GPU server (amd+cuda) | DEFER-GPU | C9 |
| dp_attn/test_dp_attention_bcg_kl.py | 4 | DP attention BCG KL | no | BCG + GPU server | DEFER-GPU | C9 |
| mla/test_mla_fp8.py | 0 | MLA FP8 | no | dsv3-family, amd+cuda | DEFER-GPU | C5 |
| mla/test_mla_int8_deepseek_v3.py | 1 | MLA int8 DSV3 | no | dsv3 | DEFER-GPU | C5 |
| mem_cache/test_int8_checkpoint_store.py | 7 | int8-compressed linear-attn state store | yes | 1 CUDA-gated test skipped on CPU | PORT-CPU | C5 |
| mem_cache/test_post_capture_kv_sizing.py | 3 | post-capture KV sizing e2e | no | GPU server | DEFER-GPU | C5 |
| openai_server/basic/test_openai_server.py | 28 | OpenAI endpoint core | no | GPU server (amd+cuda) | DEFER-GPU | C1 |
| openai_server/basic/test_http2_server.py | 5 | HTTP/2 (Granian) endpoints | no | GPU server | DEFER-GPU | C1 |
| openai_server/basic/test_openai_completion_rust.py | 1 | rust-backed completion | no | rust server | SKIP-INFRA | C1 |
| openai_server/basic/test_serving_transcription.py | 10 | /v1/audio/transcriptions | no | Whisper pin | SKIP-OTHERMODEL | C1 |
| openai_server/features/test_openai_server_hidden_states.py | 4 | hidden-states feature flag | no | GPU server (amd+cuda) | DEFER-GPU | C1 |
| openai_server/function_call/test_openai_function_calling.py | 13 | OpenAI tool-calling e2e | no | GPU server (amd+cuda+npu) | DEFER-GPU | C1/C12 |
| openai_server/function_call/test_anthropic_tool_use.py | 10 | Anthropic /v1/messages tool use | partial | GPU server; also CPU-CI registered | DEFER-GPU | C1/C12 |
| openai_server/validation/test_request_length_validation.py | 8 | request length validation | no | GPU server | DEFER-GPU | C1 |
| openai_server/validation/test_large_max_new_tokens.py | 1 | max_new_tokens cap | partial | GPU server; CPU-CI registered | DEFER-GPU | C1 |
| openai_server/validation/test_matched_stop.py | 0 | matched stop sequences | partial | GPU server; CPU-CI registered | DEFER-GPU | C1 |
| tokenizer/test_multi_tokenizer.py | 2 | multi-tokenizer serving | no | GPU server (amd+cuda) | DEFER-GPU | C4 |
| tokenizer/test_skip_tokenizer_init.py | 6 | skip-tokenizer-init mode | no | GPU server (amd+cuda) | DEFER-GPU | C4 |
| function_call/test_kimik2_detector.py | 60 | Kimi-K2 tool-call/reasoning detector | yes | none (pure parsing) | PORT-CPU | C1/C12 |
| function_call/test_kimik3_detector.py | 14 | Kimi-K3 tool-call detector/format | yes | none (pure parsing) | PORT-CPU | C1/C12 |
| constrained_decoding/test_constrained_decoding.py | 0 | constrained decoding e2e (kit-composed) | no | GPU server (amd+cuda) | DEFER-GPU | C2 |
| reasoning/test_reasoning.py | 4 | reasoning parser e2e | no | GPU server (amd+cuda) | DEFER-GPU | C1/C12 |
| accuracy/models/test_text_models_gsm8k_eval.py | 1 | GSM8K accuracy eval harness | no | pinned HF models | DEFER-GPU | C1 |
| accuracy/models/test_vlms_mmmu_eval.py | 1 | MMMU VLM accuracy eval | no | VLM pins | SKIP-OTHERMODEL | C11 |
| observability/test_metrics.py | 10 | prometheus metrics | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| observability/test_priority_metrics.py | 6 | priority-scheduling metrics | partial | GPU server; CPU-CI registered | DEFER-GPU | C8 |
| observability/test_encoder_server_metrics.py | 1 | EPD encoder metrics | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| observability/test_tracing.py | 11 | OTLP tracing | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| observability/test_tracing_disaggregation.py | 1 | tracing under PD | no | 2-GPU | DEFER-GPU | C8 |
| scheduler/test_min_free_slots_delayer.py | 12 | admission min-free-slots delayer | yes | none (pure logic) | PORT-CPU | C8 |
| scheduler/test_abort_with_metrics.py | 2 | abort + ASGI dispatch metrics | yes | none (pure unit) | PORT-CPU | C8 |
| scheduler/test_load_snapshot_server.py | 4 | load-snapshot (zmq/dp combos) | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| scheduler/test_mixed_chunked_prefill.py | 0 | mixed chunked prefill | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| scheduler/test_prefill_delayer.py | 7 | prefill delayer | no | GPU server | DEFER-GPU | C8 |
| scheduler/test_priority_scheduling.py | 7 | priority scheduling | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| scheduler/test_retract_decode.py | 2 | decode retract | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| scheduler/test_retract_decode_logprob.py | 1 | retract logprob off-by-one regression | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| scheduler/test_routing_key_scheduling.py | 1 | routing-key scheduling | partial | GPU server; CPU-CI registered | DEFER-GPU | C8 |
| scheduler/test_scheduler_control.py | 8 | scheduler control plane | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| core/test_basic_sanity.py | 0 | stage-a server sanity (kit-composed) | no | small model server (amd+cuda) | DEFER-GPU | C8 |
| core/test_basic_sanity_dspark.py | 0 | sanity + dSpark | no | GPU server | DEFER-GPU | C8/C6 |
| core/test_basic_sanity_dflash.py | 0 | sanity + DFLASH | no | GPU server | DEFER-GPU | C8/C6 |
| core/test_basic_sanity_eagle3.py | 0 | sanity + EAGLE3 | no | GPU server (amd+cuda) | SKIP-OTHERMODEL | C8/C6 |
| core/test_engine_child_pids.py | 3 | engine child PID tracking | no | cuda | DEFER-GPU | C8 |
| core/test_gated_launch.py | 1 | gated launch | no | cuda | DEFER-GPU | C8 |
| core/test_hidden_states.py | 5 | hidden-states return | no | cuda | DEFER-GPU | C8 |
| core/test_no_extra_forked_cuda_context.py | 1 | no extra CUDA context fork | no | GPU server | DEFER-GPU | C8 |
| core/test_request_queue_validation.py | 3 | request queue validation | no | GPU server | DEFER-GPU | C8 |
| core/test_srt_empty_deps.py | 2 | runtime_base torch-free packaging guard | yes | none | PORT-CPU | C8 |
| core/test_srt_endpoint.py | 28 | srt endpoint matrix | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| core/test_srt_engine.py | 6 | engine API (sync/async/stream) | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| sessions/test_session_control.py | 5 | session control API | no | GPU server (amd+cuda) | DEFER-GPU | C4 |
| sessions/test_session_latency.py | 3 | session inter-turn latency bench | no | GPU server | DEFER-GPU | C4 |
| sessions/test_streaming_session.py | 0 | streaming session per-commit (kit) | no | GPU server (amd+cuda) | DEFER-GPU | C4 |
| sessions/test_streaming_session_extra.py | 0 | streaming session extras | no | GPU server (amd+cuda) | DEFER-GPU | C4 |
| sessions/test_streaming_session_swa.py | 0 | streaming session SWA | no | GPU server | SKIP-OTHERMODEL | C4 |
| sessions/test_streaming_session_swa_extra.py | 0 | streaming session SWA extras | partial | GPU server; CPU-CI registered | DEFER-GPU | C4 |
| mock_model/test_e2e_{pd,pp,tp}.py + test_e2e_spec_eagle.py (4 files) | 6 | mock-model e2e (PD/PP/TP/spec) | no | mock weights, GPU runner (amd+cuda) | DEFER-GPU | C8 |
| mock_model/test_self_e2e_perturb_next_token_swap.py | 1 | mock-model next-token-swap perturbation | no | GPU runner (amd+cuda) | DEFER-GPU | C8 |
| mock_model/test_self_unit_canary_mock_wiring.py | 8 | mock-model canary wiring | no | amd+cuda | DEFER-GPU | C8 |
| mock_model/test_self_unit_canary_perturb.py | 1 | mock-model canary perturb unit | yes | none (CPU-registered) | PORT-CPU | C8 |
| mock_model/test_self_unit_install.py | 3 | canary install into mock model | no | amd+cuda | DEFER-GPU | C8 |
| mock_model/test_self_unit_oracle.py | 7 | token oracle logic | no | amd+cuda | DEFER-GPU | C8 |
| mock_model/test_self_unit_oracle_torch_vs_ref.py | 2 | oracle torch-vs-ref | no | amd+cuda | DEFER-GPU | C8 |
| mock_model/test_self_unit_sampler_hookpoint.py | 1 | sampler hookpoint registration | no | amd+cuda | DEFER-GPU | C8 |
| model_loading/test_external_models.py | 1 | external model loading | partial | CPU-CI registered | DEFER-GPU | C10 |
| model_loading/test_load_weights_from_remote_instance.py | 2 | remote weight load | no | GPU server (amd+cuda) | DEFER-GPU | C10 |
| model_loading/test_load_weights_from_remote_instance_npu.py | 2 | remote weight load (NPU) | no | NPU | SKIP-INFRA | C10 |
| model_loading/test_prefetch_checkpoints_multi_gpu.py | 1 | checkpoint prefetch multi-GPU | no | multi-GPU | DEFER-GPU | C10 |
| model_loading/test_runai_model_loader.py | 1 | RunAI model loader | no | runai loader | SKIP-INFRA | C10 |
| model_loading/test_startup_weight_load.py | 1 | post-capture startup weight load parity | no | GPU server | DEFER-GPU | C10 |
| model_loading/test_utils_update_weights.py | 1 | update-weights util | no | cuda | DEFER-GPU | C10 |
| model_loading/test_weight_cache_daemon.py | 5 | weight cache daemon | no | GPU server | DEFER-GPU | C10 |
| model_loading/test_weight_loader_v2_e2e.py | 2 | weight loader v2 e2e | no | cuda | DEFER-GPU | C10 |
| cuda_graph/breakable/test_breakable_cuda_graph.py | 15 | breakable CUDA graph (BCG) runner | no | cuda (amd+cuda) | DEFER-GPU | C8 |
| cuda_graph/breakable/test_bcg_with_lora.py | 6 | BCG + LoRA | no | GPU server | SKIP-OTHERMODEL | C8 |
| cuda_graph/breakable/test_bcg_with_speculative_decoding.py | 0 | BCG + EAGLE3 | no | EAGLE3 | SKIP-OTHERMODEL | C8 |
| cuda_graph/full_prefill/test_full_cuda_graph_prefill.py | 5 | full prefill CUDA graph | no | GPU server | DEFER-GPU | C8 |
| cuda_graph/piecewise/test_pcg_with_speculative_decoding{,_dflash,_extra}.py (3 files) | 0 | piecewise CUDA graph + spec | no | EAGLE3/DFLASH pins | DEFER-GPU | C8 |
| cuda_graph/piecewise/test_piecewise_cuda_graph_support_1_gpu.py | 2 | piecewise CUDA graph 1-GPU | no | GPU server (amd+cuda) | DEFER-GPU | C8 |
| cuda_graph/test_cuda_piecewise_backend.py | 1 | piecewise backend unit | no | cuda | DEFER-GPU | C8 |
| backends/test_deepseek_v3_fp4_cutedsl_moe.py | 1 | CuteDSL MoE EP=TP=4 | no | dsv3, B200 | DEFER-GPU | C8 |
| backends/test_flashinfer_fusion_preflight.py | 2 | flashinfer allreduce-fusion preflight | no | multi-GPU | DEFER-GPU | C8 |
| backends/test_flashinfer_nvfp4_online_moe_backend.py | 1 | flashinfer NVFP4 online MoE | no | GPU server | DEFER-GPU | C8 |
| backends/test_flashinfer_trtllm_gen_moe_backend.py | 6 | flashinfer trtllm-gen MoE | no | GPU server | DEFER-GPU | C8 |
| backends/test_torch_compile.py | 1 | torch.compile e2e | no | GPU server (amd+cuda) | SKIP-INFRA | C8 |
| utils/test_log_utils.py | 3 | log utils | yes | none (CPU-registered) | PORT-CPU | — |
| utils/test_network_address.py | 54 | network/address helpers | yes | none (CPU-registered) | PORT-CPU | — |
| utils/test_numa_utils.py | 41 | NUMA topology utils | partial | CPU+cuda registered; NUMA hw | PORT-CPU | — |
| utils/test_socket_utils.py | 23 | socket/ZMQ helpers | yes | none (CPU-registered) | PORT-CPU | — |
| utils/test_stale_shm_cleanup.py | 8 | stale shm cleanup | yes | none (CPU-registered) | PORT-CPU | — |
| utils/test_type_based_dispatcher.py | 1 | type-based dispatcher e2e | yes | amd+cpu registered | PORT-CPU | — |
| utils/test_bench_typebaseddispatcher.py | 3 | dispatcher bench | yes | amd-only CI | PORT-CPU | — |
| utils/test_phase_checker.py | 32 | phase checker (subprocess phases) | no | amd+cuda | DEFER-GPU | — |
| layers/mamba/ (5 test files + conftest) | 20 | mamba kernels (conv1d, ssm, ssd, slot fused) | no | cuda/xpu | SKIP-OTHERMODEL | C10 |
| layers/test_layernorm_fusion.py | 3 | layernorm fusion | no | amd+cuda | DEFER-GPU | C10 |
| layers/test_fla_layernorm_guard.py | 2 | fla layernorm guard | no | cuda | DEFER-GPU | C10 |
| rotary/test_rope_cache_invalidation.py | 4 | rope cache invalidation | yes | none (CPU-registered) | PORT-CPU | C10 |
| rotary/test_mrope_axis_map.py | 5 | mrope axis mapping | yes | none (CPU-registered) | PORT-CPU | C10 |
| rotary/test_rope_rocm.py | 2 | rope on ROCm | no | AMD ROCm | SKIP-INFRA | C10 |

## Notable PORT-CPU candidates

- **spec/dspark CPU suite (10 files, n=146)** — sps_table (21), scheduler (25), info_dumper (27), block_accept_estimator (13), STS fit (10), confidence_metrics (10), ragged_verify (10): pure dSpark accept/bookkeeping logic; direct parity model for ds41rt's dSpark (C6).
- **kv_canary/test_self_unit_radix_walker.py (14)** — walks sglang's actual RadixCache/SWARadixCache/UnifiedTreeCore trees with mocks; the closest pure-CPU match/eviction-policy coverage for ds41rt-hostcache (C5).
- **kv_canary/test_self_unit_* remaining (17 files, n=143)** — capacities, perturb config, pool patcher, sweep plan builder, violation oracles: zero `.cuda()` hits despite GPU CI registration; CPU-tensor logic (C5).
- **scheduler/test_min_free_slots_delayer.py (12)** — admission-delay policy arithmetic, pure CPU; template for ds41rt scheduler admission gating (C8).
- **function_call/test_kimik2_detector.py (60) + test_kimik3_detector.py (14)** — tool-call/reasoning detector parsing with zero server/GPU deps; largest single pure-CPU block in the set (C1/C12).
- **sampling/test_deterministic_gumbel_u1.py (2)** — sampler logprob edge (u==1.0 gumbel bucket NaN/inf); only 4 `.cuda()` call-sites to de-GPU (C3).
- **sampling/test_sampling_mask.py (21)** — sampling-mask bookkeeping; server-based but the mask/logprob state-machine is the transferable core (C3).
- **dcp/test_dcp_layout_unit.py (17) + cp/test_cp_strategy_unit.py (24)** — per-rank KV-length/sharding math with mocks; both explicitly CPU (C9).
- **mem_cache/test_int8_checkpoint_store.py (7)** — int8-compressed state checkpoint store; one CUDA-gated test auto-skips on CPU; persistence-store analog for hostcache (C5).
- **rotary/test_rope_cache_invalidation.py (4) + test_mrope_axis_map.py (5)** — rope cache lifecycle/axis mapping, CPU-registered; cheap wins if ds41rt keeps sglang rope (C10).
- **moe/test_hash_topk.py (3) + expert_pack/test_expert_pack_runtime.py (11)** — MoE routing/packing logic pure CPU; relevant to DS41RT's 552B-MoE EP layout (C7).
- **chunked_prefill/test_mm_chunked_embedding_unit.py (10)** — per-item embedding funcs in the chunked-prefill path; CPU-registered unit block (C5).


<!-- ===== 22-sglang-kernels-hw.md ===== -->

# 22 — sglang kernels/hw test inventory

Scope: `/home/turq/dev/upstream-test-inventory/sglang/test/registered/` — kernels/, kernel/, attention/, npu/, amd/, xpu/, gb300/, musa/, ops/, gemm/, page_major/, input_embedding/, quant/, lora/, cpu/, debug_utils/, perf/, stress/, bench_fn/, vlm/, rl/, dllm/, jit/, scripted_runtime/, rust/, mlx/.

Method: mechanical census. `n` = summed `def test_*`/`check_*` (plus a few `bench_*`) counted per file by regex; approximate. Spot-read 2–3 files per subdir; cpu/ read file-by-file. No file-level coverage attempted for kernels/ or hardware dirs (rolled up per subdir).

Verdicts: PORT-CPU = pure-CPU/torch or mock-based, portable to DS41RT CPU CI. DEFER-GPU = needs CUDA/ROCm GPU kernels. DEFER-MODEL = CPU-runnable but launches a model server / needs weights. SKIP-OTHERMODEL = hardware-specific (NPU/AMD/XPU/GB300/MUSA/MLX/AMX/arm64). SKIP-INFRA = benchmark/stress/CI-infra scaffolding, not product coverage.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| kernels/ (top-level *.py) | ~112 | kernel integration: KDA/replay-SSM, AOT dispatch, lean-attn, allreduce | no | test_verify_hand.py(56), test_model_fast_paths.py(50), test_plan_hand.py(45) | DEFER-GPU | C6 |
| kernels/ops/ | ~1022 | kernel unit tests: attention, gemm, moe, quant, kvcache, layernorm, elementwise, mamba, diffusion, communication | no | test_c128_v2.py, test_fp8_blockwise_gemm.py, test_fused_topk_deepseek.py | DEFER-GPU | C5/C7/C10 |
| kernels/benchmark/ | ~10 | benchmark fns (spec_topk, ngram) | no | bench_spec_topk1.py, bench_ngram_update_token_table.py | SKIP-INFRA | — |
| kernel/attention/ | ~9 | fused attn backend verify (KDA, DSA replay, subblock sage fp8) | no | test_kda_fused_verify_backend.py, test_dsa_metadata_replay.py | DEFER-GPU | C5 |
| kernel/cuda_graph/ | ~2 | cuda graph dedup | no | test_cuda_graph_dedup.py | DEFER-GPU | C8 |
| kernel/diffusion/ | ~30 | VDN/Sana WM diffusion kernels | no | test_vdn_linear_branch.py(6), test_vdn_delta_factors.py | DEFER-GPU | C11 |
| kernel/embeddings/ | ~7 | Qwen4 ple offload embeddings | no | test_qwen4_ple_offload.py | DEFER-GPU | C4 |
| kernel/hyperconnection/ | ~3 | HC mix triton (MoE hyper-connections) | no | test_hc_mix_triton.py | DEFER-GPU | C7 |
| kernel/jit/ | ~14 | JIT kernels: fast_topk, hc_combine, grouped gemma rmsnorm | no | test_fast_topk.py(7), test_hc_combine.py | DEFER-GPU | C7 |
| kernel/moe/ | ~3 | JIT grouped topk | no | test_jit_grouped_topk.py | DEFER-GPU | C7 |
| kernel/ops/ | ~2 | KDA track state | no | test_kda_track_state.py | DEFER-GPU | C5 |
| kernel/qsa/ | ~39 | QSA indexer/strided kernels | no | test_qsa.py(33), test_qsa_strided_zero_fill.py | DEFER-GPU | C5 |
| kernel/quantization/ | ~1 | aiter fp8 utils (ROCm) | no | test_fp8_utils_aiter.py | SKIP-OTHERMODEL | C10 |
| kernel/speculative/ | ~8 | dflash domino, spec kv grid, verify commit triton | no | test_dflash_domino.py(4), test_spec_kv_indices_grid.py | DEFER-GPU | C6 |
| attention/ (top-level) | ~179 | attention backend correctness (chunk gated delta rule, deepseek_v4, aiter gluon fp8) | no | test_chunk_gated_delta_rule.py(31), test_deepseek_v4.py(22) | DEFER-GPU | C5 |
| attention/unittests/ | ~224 | per-backend attn unit tests (triton, DSA, dsv4) | no | test_deepseek_v4.py(22), test_dsa.py(16), test_triton.py(11) | DEFER-GPU | C5 |
| npu/ (all subdirs rolled) | ~173 | Ascend NPU: api, accuracy, llm/vlm/reward models, perf, radix/hierarchical cache | no | test_npu_api.py(32), test_npu_no_overlap_scheduler.py, test_npu_qwen3_vl_8b_* | SKIP-OTHERMODEL | C1/C5 |
| amd/ (all subdirs rolled) | ~202 | ROCm/MI35x: aiter kernels, mxfp4 accuracy evals, disaggregation (NIXL), perf | no | test_moriep_small.py(9), test_disaggregation_basic.py(6), test_qwen35_mxfp4_eval_mi35x.py | SKIP-OTHERMODEL | C5/C9/C10 |
| xpu/ (all subdirs rolled) | ~117 | Intel XPU: numa utils, breakable cuda graph, memory saver, llm models | no | test_numa_utils_xpu.py(26), test_breakable_cuda_graph.py(13) | SKIP-OTHERMODEL | C8 |
| gb300/ | ~8 | GB300 fp8/nvfp4 TP/DP server configs | no | test_qwen35_fp8_tp.py, test_kimi_k25_nvfp4_tp.py | SKIP-OTHERMODEL | C10 |
| musa/ | ~4 | MUSA smoke server | no | test_llm_server_smoke_musa.py | SKIP-OTHERMODEL | C1 |
| ops/ | ~27 | aiter (ROCm) greedy_sample / allreduce / allgather kernels | no | test_aiter_greedy_sample_amd.py(16), test_aiter_allreduce_fusion_amd.py | SKIP-OTHERMODEL | C3 |
| gemm/ | ~8 | HPC-ops bf16xfp32 router GEMM (sm90a), hopper gemv | no | test_linear_bf16_fp32_hpc.py(5), test_hopper_bf16_gemv.py | DEFER-GPU | C10 |
| page_major/ | ~2 | page-major KV layout e2e (qwen hybrid, gpt-oss) | no | test_page_major_qwen_hybrid.py, test_page_major_gpt_oss.py | DEFER-GPU | C5 |
| input_embedding/ | ~7 | input-embeddings e2e server (small HF model) | partial (cpu_ci registered) | test_input_embeddings.py(4), test_input_embeds_chunked.py | DEFER-MODEL | C4 |
| quant/ | ~61 | quant configs: quark mxfp4, fp8 utils, quant config parsing, nvfp4 embedding | partial (some cpu_ci) | test_quark_mxfp4.py(18), test_fp8_utils.py(6), test_quant_config_parsing.py | DEFER-GPU (parsing/is_layer_skipped CPU-portable) | C10 |
| lora/ | ~89 | LoRA API, overlap loading, virtual experts kernels, openai adapter resolution | partial (openai_api cpu_ci, mock-based) | test_lora_openai_api.py(28), test_lora_overlap_loading.py, test_virtual_experts_kernels.py | MIXED: 1 file PORT-CPU, rest DEFER-GPU | C10 |
| cpu/test_subblock_sparse_attention.py | ~28 | sparse-attn routing-plan/block-map logic, heavy mocking (sm90/sm100 paths) | yes | test_subblock_sparse_attention.py | PORT-CPU (partial; runner paths GPU-gated) | C5 |
| cpu/test_spec_kernels.py | ~25 | spec-decode utils: eagle_utils organize_draft_results, tree masks (pure torch) | yes | test_spec_kernels.py | PORT-CPU | C6 |
| cpu/test_rank_consensus_checker.py | ~23 | distributed rank-consensus checks, torch.distributed CPU | yes | test_rank_consensus_checker.py | PORT-CPU | C9 |
| cpu/test_norm.py | ~13 | rmsnorm/layernorm vs sgl_kernel CPU, fp16/bf16 | yes | test_norm.py | PORT-CPU | C10 |
| cpu/test_topk.py | ~9 | MoE topk native-torch references (grouped/biased/fused) | yes | test_topk.py | PORT-CPU | C7 |
| cpu/test_moe.py | ~9 | CPUQuantMethod MoE gemm (Intel AMX kernels) | yes (AMX-only) | test_moe.py | SKIP-OTHERMODEL (Intel AMX) | C7 |
| cpu/test_request_headers.py | ~9 | OpenAI routing-header overrides (pure fastapi) | yes | test_request_headers.py | PORT-CPU | C1 |
| cpu/test_request_decompression.py | ~7 | request decompression middleware | yes | test_request_decompression.py | PORT-CPU | C1 |
| cpu/test_gemm.py | ~6 | CPU w8a8/awq/gptq gemm + MXFP4 (AMX path) | yes (AMX-only) | test_gemm.py | SKIP-OTHERMODEL (Intel AMX) | C10 |
| cpu/test_server_args_backend.py | ~5 | server-args parsing/validation | yes | test_server_args_backend.py | PORT-CPU | C1 |
| cpu/test_rope.py | ~5 | rope vs CPU reference | yes | test_rope.py | PORT-CPU | C10 |
| cpu/test_extend.py | ~5 | extend/SDPA attention vs CPU kernel path | yes | test_extend.py | PORT-CPU | C5 |
| cpu/test_qwen3.py | ~3 | qwen3 model-op numerics on CPU (arm64+intel registered) | yes | test_qwen3.py | PORT-CPU | C10 |
| cpu/test_intel_amx_attention_backend_{a,b,c}.py | ~8 | full server launch + eval on Intel AMX backend | yes (AMX-only) | test_intel_amx_attention_backend_a.py | SKIP-OTHERMODEL (Intel AMX) | C5 |
| cpu/test_cpu_graph.py | ~2 | torch.compile CPU graph server MMLU eval | yes | test_cpu_graph.py | DEFER-MODEL | C8 |
| cpu/quant/test_autoround.py | ~3 | AutoRound INT4 CPU accuracy via AMX server | yes (AMX-only) | quant/test_autoround.py | SKIP-OTHERMODEL (Intel AMX) | C10 |
| cpu/ small torch-unit group | ~29 | flash_attn(3), causal_conv1d(4), mamba(3), shared_expert(3), store_cache/fp4-kv(3), comm(4), sampling(2), activation(2), mla(1), decode(1), bmm(1), qkv_proj_with_rope(3), binding/numa(4) | yes | test_store_cache.py, test_comm.py, test_binding.py | PORT-CPU | C5/C10 |
| cpu/arm64/test_moe.py | ~1 | arm64-specific MoE numerics | yes (arm64) | arm64/test_moe.py | SKIP-OTHERMODEL (arm64) | C7 |
| debug_utils/ (top-level) | ~309 | dumper, entrypoint, planner unit tests | mostly | test_dumper.py(238), test_entrypoint.py(135), test_planner.py(101) | SKIP-INFRA (debug tooling; some logic portable) | — |
| debug_utils/comparator/ | ~941 | output comparator: planner, entrypoint, comparator | mostly | test_entrypoint.py(135), test_planner.py(101), test_comparator.py(52) | SKIP-INFRA | — |
| debug_utils/source_patcher/ | ~44 | source editor / code patcher | yes | test_source_editor.py(29), test_code_patcher.py(13) | SKIP-INFRA (editor tooling; CPU-portable) | — |
| perf/ (top-level) | ~23 | bench_serving harness 1/2-GPU | no | test_bench_serving_1gpu_part1.py(8) | SKIP-INFRA | — |
| perf/models/ | ~2 | text/vlm perf gates | no | test_text_models_perf.py, test_vlms_perf.py | SKIP-INFRA | — |
| stress/models/ | ~4 | stress: qwen3-235b, kimi-k2, glm-4.6 | no | test_stress_qwen3_235b.py | SKIP-INFRA | — |
| bench_fn/ | ~59 | benchmark datasets api, bench_serving reasoning stream, steady-state | partial | test_benchmark_datasets_api.py(45), test_bench_serving_reasoning_stream.py | SKIP-INFRA | — |
| vlm/ | ~20 | VLM server e2e (openai vision, rust native mm, input format, video utils, token retokenize) | partial (video_utils/token_id_retokenize cpu_ci) | test_vision_openai_server_a.py(4), test_rust_native_mm_e2e.py, test_video_utils.py | DEFER-GPU (2 files CPU-portable) | C11 |
| rl/ | ~63 | RL: weight version spans, weight checker e2e, routed-experts return | no | test_weight_version_spans.py(17), test_weight_checker_e2e.py(9) | DEFER-GPU (server-based) | C9 |
| dllm/ | ~5 | diffusion-LLM server e2e (llada2 mini amd, fdfo batching) | no | test_dllm_batching_fdfo.py(2), test_llada2_mini_amd.py | DEFER-GPU | C6 |
| jit/ (top-level) | ~11 | hisparse spec kvcache kernels, flux2 gated resnorm | no | test_hisparse_spec.py(9), test_flux2_gated_resnorm.py | DEFER-GPU | C6 |
| jit/benchmark/ | 0 | hisparse spec bench | no | bench_hisparse_spec.py | SKIP-INFRA | — |
| scripted_runtime/ | ~42 | scripted runtime test harness (schedule_batch, chunked prefill drivers) | mostly | test_scripted_runtime_core.py(42) | PORT-CPU (harness; valuable for scheduler CI) | C8 |
| rust/ | ~17 | rust extension loader, radix tree rust tests | yes (cpu_ci) | test_rust_extension.py(15) | PORT-CPU (needs rust toolchain) | C5 |
| mlx/models_e2e/ | ~12 | MLX e2e correctness (gpt-oss, qwen3-moe, qwen2-moe) | no | test_gpt_oss_mlx_correctness.py(6) | SKIP-OTHERMODEL | C7 |

Notes:
- kernels/ registered count in the brief (267) is far below the mechanical `def test_*` count (~1144 incl. kernels/ops) — the brief figure likely counted CI-registered tests only. `n` here is the raw mechanical count.
- npu/llm_models, xpu/llm_models have 0 test fns (server-launch configs, model-gated).
- Hardware dirs rolled to single rows per brief; no file-level coverage attempted.

## Notable PORT-CPU candidates

- **cpu/test_rank_consensus_checker.py** (~23 fns) — torch.distributed CPU consensus checks; direct C9 (transport) coverage with no GPU.
- **cpu/test_subblock_sparse_attention.py** (~28 fns) — routing-plan/block-map logic, heavily mocked; most fns CPU-runnable (sm90/sm100 runner paths GPU-gated). C5.
- **cpu/test_spec_kernels.py** (~25 fns) — pure-torch EAGLE draft-tree/verify utils; clean C6 (spec decode) coverage.
- **lora/test_lora_openai_api.py** (~28 fns) — mock-based OpenAI LoRA adapter resolution; already `register_cpu_ci`. C10.
- **cpu/test_norm.py + test_topk.py + test_rope.py** (~27 fns combined) — torch-vs-CPU-kernel numerics; sgl_kernel CPU build required but no GPU. C10/C7.
- **scripted_runtime/test_scripted_runtime_core.py** (~42 fns) — scripted scheduler/chunked-prefill harness; strong C8 candidate as a CPU CI driver.
- **rust/test_rust_extension.py** (~15 fns) — rust extension loader, `register_cpu_ci`; needs rust toolchain only. C5.
- **cpu/test_request_headers.py + test_request_decompression.py** (~16 fns) — pure fastapi request-path unit tests, zero deps. C1.


<!-- ===== 23-sglang-manual-kits.md ===== -->

# 23 — sglang `test/manual/` + `python/sglang/test/` inventory

Census only (spot-read 2–3 files per group; heads + `def test_` counts). Targets use the
DS41RT component map: C1 API; C2 constrained; C3 sampler; C4 tokenizer; C5 KV
cache/radix/persistence; C6 spec decode; C7 MoE; C8 scheduler; C9 transport; C10
weights/quant; C11 vision; C12 tool parsing.

## `sglang/test/manual/` (255 py) — one row per thematic subdir / top-level group

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| manual/4-gpu-models/ | 4 | Qwen3-Next / Qwen3.5 FP4 multi-GPU launch + GSM8K/KL/prefix-cache kits | no | 4 GPU, specific Qwen weights | DEFER-GPU | C10, C5, C3 |
| manual/8-gpu-models/ | 3 | DeepSeek-V3.2 / GLM-5 FP8 TP8+DP8 sanity launches | no | 8 GPU, DSv3.2/GLM weights | SKIP-OTHERMODEL | C1, C7 |
| manual/ascend/ | 16 | NPU (Ascend) LLM+VLM model tests, disagg utils, MTP | no | Ascend NPU, modelscope weights | SKIP-OTHERMODEL | C1, C11, C6 |
| manual/attention/ | 6 | FA3 / FlashAttn / TRT-LLM MLA backend launches; prefix-chunk-info tensor UT | partial | SM90+; most launch servers | DEFER-GPU | C5, C6 |
| manual/beam_search/ | 3 | Beam-search parity vs HF, load/perf sweeps | no | GPU + HF reference + server | DEFER-GPU | C3 |
| manual/chunked_prefill/ | 33 | scripted-runtime chunked-prefill invariants + e2e (dp-attn, lora, pp, spec, radix) | partial | scripted runtime is in-process; e2e launch servers | DEFER-GPU | C5, C8 |
| manual/core/ | 1 | DynamicGradMode UT (torch inference/no_grad wrapper) | yes | torch CPU ok | PORT-CPU | C8 |
| manual/debug_utils/ | 4 | log_parser UT (pure), metric dump, retry/logit utils | yes | none (log_parser); get_logits_ut is torch CPU-possible | PORT-CPU | C8 |
| manual/distributed/ | 2 | DP-attention launches w/ constrained kits | no | multi-GPU, MLA models | DEFER-GPU | C2, C7 |
| manual/dsv4/ | 21 | DeepSeek-V4 cookbook launch-recipe tests (B200/GB300/B300, Flash/Pro, CP) + sgl-eval | no | 8 GPU, DSv4 weights, sgl-eval on PATH | DEFER-GPU | C1, C7, C6 (ds41rt-adjacent) |
| manual/entrypoints/http_server/ | 1 | abort_request HTTP integration | no | launched server | DEFER-GPU | C1, C8 |
| manual/ep/ | 14 | expert-parallel transports: DeepEP intra/low-latency, EPLB mori, mooncake, elastic scale | no | multi-GPU, deep_ep/mooncake libs | DEFER-GPU | C7, C9 |
| manual/eval/ | 1 | large accuracy eval (MMLU/HumanEval/MGSM) launcher | no | server + datasets | DEFER-GPU | C1, C3 |
| manual/hicache/ | 1 | HiCache disaggregation (prefill/decode split) | no | 2 servers | DEFER-GPU | C5 |
| manual/kernels/ | 3 | hisparse prefetch CUDA-graph tests; fused-op dispatch microbench | partial | CUDA (hisparse); dispatch bench CPU-fine | DEFER-GPU | C5, C10 |
| manual/kv_canary/ | 2 | KV-canary e2e baseline reuse (DSv4 pool configs) | no | server, registered-suite base | DEFER-GPU | C5 |
| manual/kv_transfer/ | 2 | Mooncake transfer-engine e2e / init | no | mooncake lib, 2 nodes | DEFER-GPU | C5, C9 |
| manual/lang_frontend/ | 6 | lang API: choices scoring UT (pure numpy), jump-forward, reasoning, openai backend | partial | test_choices.py is pure CPU; rest launch servers | PORT-CPU | C3, C1 |
| manual/layers/ | 11 | layer UTs (layernorm, fused ops), DSA triton kernel vs torch_fast, MoE kernel benches | no | CUDA/triton kernels; benches H100 | DEFER-GPU | C10, C7 |
| manual/lora/ | 10 | LoRA ops/backend/cuda-graph, tuning config | no | CUDA | DEFER-GPU | C10 |
| manual/minimax_m3/ | 2 | MiniMax-M3 NPU topk / memory pool (fake-module trick) | partial | NPU-oriented; topk test is torch-CPU-able | SKIP-OTHERMODEL | C7 |
| manual/mla/ | 5 | MLA launches w/ MGSM eval (DeepSeek-V3 MLA, FlashMLA archived) | no | GPU, MLA models | DEFER-GPU | C5 |
| manual/models/ | 16 | per-model basic launches (Mistral, Falcon, Llama4, Nemotron, Qwen3-ASR...) | no | GPU + each model's weights | SKIP-OTHERMODEL | C1, C11 |
| manual/nightly/ | 5 | nightly perf/VLM-VIT suites (piecewise cuda graph, flashinfer/cudnn) | no | GPU, nightly datasets | DEFER-GPU | C11, C1 |
| manual/openai_server/ | 4 | OpenAI-server feature tests: tool_choice, structural tag, usage stats, cache report | no | launched server | DEFER-GPU | C12, C1 |
| manual/perf/ | 1 | bench_one_batch 1-GPU | no | 1 GPU | DEFER-GPU | C1 |
| manual/piecewise_cuda_graph/ | 1 | archived piecewise-cuda-graph server tests (InternVL etc.) | no | GPU | DEFER-GPU | C11 |
| manual/piecewise_cudagraph/ | 1 | disaggregation piecewise cuda graph | no | 2 servers | DEFER-GPU | C5 |
| manual/prefill_only/ | 2 | cross-encoder / encoder-embedding models via HF+SRT runners | no | GPU + HF models | DEFER-GPU | C1 |
| manual/quant/ | 15 | quant launches (AWQ/FP8/FP4/NVFP4/block-fp8/autoround) + quant kernel UTs | no | CUDA, Blackwell some | DEFER-GPU | C10 |
| manual/scheduler/ | 3 | scheduler flags (no-chunked-prefill, no-overlap, scripted pp abort) | no | launched servers | DEFER-GPU | C8, C5 |
| manual/spec/ | 2 | spec-decode: ngram+FA3 kit, EAGLE3 basic | no | GPU + draft models | DEFER-GPU | C6 |
| manual/vlm/ | 4 | mm_utils Mock UT, anthropic vision, K25 equivalence verifies | partial | mm_utils UT is mocked/CPU-able | DEFER-GPU | C11 |
| manual/test_{deepseek_v31,glm_46_fp8,qwen3_235b,w4a8_deepseek_v3,mla_tp,torch_tp,srt_engine_with_quant_args,models_from_modelscope}.py | 8 | model-specific accuracy+perf launch runners (combined accuracy/perf runner) | no | GPU + named model weights (DeepSeek-V3.1, GLM-4.6, Qwen3-235B = non-ds41rt) | SKIP-OTHERMODEL | C10, C1 |
| manual/test_{aiter_unified_draft_extend_env,dsa_alias_cli_registry_env}.py | 2 | env/CLI flag-registration guards (pure argparse/env asserts) | yes | none (aiter test is flag-only, ROCm kernel not exercised) | PORT-CPU | C1, C6, C8 |
| manual/test_{async_dynamic_batch_tokenizer,tokenizer_batch_encode,tokenizer_manager}.py | 3 | TokenizerManager / async-batch-tokenizer UTs (asyncio + Mock) | yes | none | PORT-CPU | C4 |
| manual/test_deepseek_chat_templates.py | 1 | DeepSeek chat-template tool-call escaping (jinja, issue #11700) | yes | none | PORT-CPU | C12 |
| manual/test_{schedule_policy,weight_validation,config_integration}.py | 3 | scheduler-policy UT (RadixCache in-proc), weight-cache shard validation, ServerArgs config merger | yes | none (in-proc, tempfile) | PORT-CPU | C8, C10, C1 |
| manual/test_logprobs.py | 1 | logprobs accuracy vs recorded baseline | no | GPU precision-sensitive | DEFER-GPU | C3 |
| manual/test_{fim_completion,health_check,weight_version,sagemaker_server,vertex_endpoint,crusoe_backend}.py | 6 | server-feature integration (FIM, health, weight_version, SageMaker/Vertex adapters, Crusoe cloud backend) | no | launched server; crusoe needs API key | DEFER-GPU | C1 |
| manual/test_{custom_allreduce,quick_allreduce,ray_engine,cross_node_scheduler_info_sync}.py | 4 | custom allreduce (Ray-spawned), Ray engine/DP, cross-node scheduler sync | no | multi-GPU + Ray; cross-node 2 nodes | DEFER-GPU | C9, C8 |
| manual/test_{get_weights_by_name,weight_cache_e2e,weight_loader_v2_equiv,expert_location_updater,expert_distribution,moe_quant_once,modelopt,modelopt_fp8kvcache}.py | 8 | weight loading/cache/eplb-location/quant-once GPU tests | no | GPU, multi-proc distributed some | DEFER-GPU | C10, C7 |
| manual/test_{torch_flex_attention_backend,wave_attention_backend,triton_attention_rocm_mla,create_custom_4d_mask,two_batch_overlap,kda_spec_integration,kda_target_verify,triton_moe_wna16}.py | 8 | attention-backend launches + KDA (linear-attn) kernels + wna16 MoE kernels | no | CUDA (one ROCm-mla); kernels GPU | DEFER-GPU | C6, C5, C7 |
| manual/test_forward_split_prefill.py | 1 | forward-split-prefill in-proc scheduler batch UT | partial | imports ModelConfig/ParallelState; likely CPU-able but unverified | DEFER-GPU | C8, C5 |
| manual/test_{mori_transfer_engine_e2e,kv_events,forward_pass_metrics}.py | 3 | PD transfer e2e, KV-event ZMQ schema, FPM ZMQ pub/sub (schema roundtrip CPU) | partial | FPM/kv_events schema parts are CPU | DEFER-GPU | C9, C5 |
| manual/test_{vlm_accuracy,whisper_cuda_graph}.py | 2 | VLM embedding accuracy UT, Whisper cuda-graph | no | GPU | DEFER-GPU | C11 |

## `sglang/python/sglang/test/` (178 py) — harness machinery, one row per file/group

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| test/test_utils.py | 1 | THE shared harness: popen_launch_server, CustomTestCase, bench wrappers, model-name constants | partial | launches servers for most helpers | SKIP-INFRA | — |
| test/run_eval.py | 1 | unified eval entrypoint (gsm8k/mmlu/...) over HTTP | no | server | SKIP-INFRA | — |
| test/run_combined_tests.py + accuracy_test_runner.py + performance_test_runner.py | 3 | combined accuracy+perf runner framework | no | server | SKIP-INFRA | — |
| test/runners.py | 1 | HFRunner vs SRTRunner comparison harness | no | GPU | SKIP-INFRA | — |
| test/send_one.py | 1 | single-prompt bench client | no | server | SKIP-INFRA | — |
| test/test_deterministic.py + test_deterministic_utils.py | 2 | batch-invariance determinism driver + unittest wrapper | no | server | SKIP-INFRA | — |
| test/few_shot_gsm8k.py + few_shot_gsm8k_engine.py | 2 | deprecated GSM8K eval (routes to run_eval) | no | server | SKIP-INFRA | — |
| test/simple_eval_{common,humaneval,mgsm,mixed_prefix_gsm8k,mmlu,mmmu_vlm}.py | 6 | OpenAI simple-evals ports (CPU-side scoring; need server for gen) | partial | server for generation | SKIP-INFRA | — |
| test/tool_call_test_runner.py | 1 | OpenAI tool-call eval runner over server | no | server | SKIP-INFRA | — |
| test/cache_consistency_jitter.py | 1 | bitwise cache-consistency workload generator | no | server | SKIP-INFRA | — |
| test/chunked_prefill_test_utils.py | 1 | ChunkedTestBase class (extends PD fixture) | no | server | SKIP-INFRA | — |
| test/cpu_test_utils.py | 1 | pure-torch CPU reference helpers (tolerances, dequant fns) | yes | none | PORT-CPU | C10 |
| test/config_publishers.py | 1 | AST-derived registry of RuntimeContext.set_server_args publishers | yes | none | PORT-CPU | C8, C1 |
| test/doc_patch.py | 1 | doc-build monkey patches (ports, mem limits) | yes | none | SKIP-INFRA | — |
| test/gb300_utils.py | 1 | GB300 runner port constants | n/a | GB300 runner | SKIP-INFRA | — |
| test/gpt_oss_common.py | 1 | gpt-oss eval common (server + requests) | no | server | SKIP-INFRA | — |
| test/hicache_spec_storage_common.py | 1 | HiCache+EAGLE3 shared test helpers | no | servers | SKIP-INFRA | — |
| test/kl_test_utils.py + kl_multiturn_utils.py + logprob_test_utils.py | 3 | KL/logprob assertion helpers over live server | no | server | SKIP-INFRA | — |
| test/layer_ut_utils.py + quant_ref_utils.py | 2 | layer parity-UT fixture plumbing + hand-written quant references (deliberately srt-independent) | yes | torch only | PORT-CPU | C10 |
| test/lora_utils.py | 1 | LoRA reference sgmv ops (torch) | partial | torch; used vs GPU runners | SKIP-INFRA | — |
| test/mooncake_utils.py | 1 | mooncake master/server process mgmt | no | mooncake lib | SKIP-INFRA | — |
| test/nightly_utils.py + nightly_bench_utils.py + precision_baseline_store.py | 3 | nightly bench reporting + HF-dataset precision baseline store | partial | HF dataset access for store | SKIP-INFRA | — |
| test/otel_collector.py | 1 | in-process OTLP collector test double (gRPC/HTTP) | yes | none | PORT-CPU | C9 |
| test/scripted_runtime_chunked_helpers.py | 1 | scripted-runtime constants/helpers for chunked tests | yes | in-proc | SKIP-INFRA | — |
| test/test_deepep_utils.py | 1 | DeepEP test utils (init_dist, fp8 cast benches) | no | GPU + deep_ep | SKIP-INFRA | — |
| test/test_marlin_utils.py | 1 | marlin weight repack utils (from vLLM) | no | GPU kernels | SKIP-INFRA | — |
| test/test_programs.py | 1 | sglang program samples for unit tests | partial | sglang runtime | SKIP-INFRA | — |
| test/vlm_utils.py | 1 | VLM request helpers (base64 images, openai client) | no | server | SKIP-INFRA | — |
| test/__init__.py, long_prompt.txt | - | package marker + fixture text | — | — | SKIP-INFRA | — |
| test/kits/ (27 kit modules) | 27 | reusable test mixins: eval_accuracy, constrained (regex/json/EBNF/json-mode), KL, spec decoding, radix/unified-cache, abort/timeout, cache-hit, mmmu_vlm, reasoning, streaming, pd_parity, lm_eval, hellaswag, dsa_metadata, basic_api_contract/decode/scheduler-stress, fwd_occupancy, pause_generation, anthropic_messages, spec_server, prefix_cache_branching, matched_stop | no | all assume live server (base_url/process) | SKIP-INFRA | C1–C6 kits |
| test/kits/attention_unittest/ | 18 | attention-method unittest runners (runner_modes, attention_methods) | no | GPU attention paths | SKIP-INFRA | — |
| test/server_fixtures/ (11 modules) | 11 | server base classes: default, disaggregation (PD), eagle/spec, ngram, dsa_mtp, hybrid-attn, mmmu, pcg_spec, standalone, streaming_session | no | launches servers | SKIP-INFRA | — |
| test/scripted_runtime/ (19 py) | 19 | in-process scripted runtime: context, io_struct, req_handle, scheduler_hook, http_server, test_case, tokenizer proxy | partial | runs scheduler in-proc (still needs torch/CUDA for full use) | SKIP-INFRA | C8 |
| test/kv_canary/ (11 py) | 11 | KV-canary fixtures: consts, e2e_base, pd/pp fixtures, violation asserts, mode configs | no | servers; canary runner | SKIP-INFRA | C5 |
| test/kernels/ (12 py incl. deepseek_v4/, kv_canary/) | 12 | kernel test helpers: multigpu pytest launcher, DSv4 common, canary differential/invariant/fuzz drivers | no | GPU | SKIP-INFRA | — |
| test/ascend/ (19 py incl. e2e/) | 19 | Ascend/NPU kit: eval kits, disagg utils, e2e k8s scripts, NPU logging | no | NPU cluster | SKIP-OTHERMODEL | — |
| test/ci/ (6 py) | 6 | CI machinery: ci_register (register_cpu/cuda/amd decorators), ci_utils, fork_test_worker, stress utils, diffusion bridge | yes (register/AST parts) | none for registry | SKIP-INFRA | — |
| test/mock_model/ (3 py) | 3 | mock-model e2e perturbation base | no | server | SKIP-INFRA | — |
| test/observability/ (2 py) | 2 | fake_ray test double | yes | none | PORT-CPU | C9 |
| test/external_models/ (1 py) | 1 | custom_qwen2_vl model def for tests | no | model weights | SKIP-INFRA | — |
| test/xpu/ (3 py) | 3 | XPU (Intel) eval mixins/utils | no | XPU | SKIP-OTHERMODEL | — |
| test/manual/disaggregation/ (3 py) | 3 | manual disagg abort tests (kit-level) | no | servers | SKIP-INFRA | — |

Notes on the few CPU-reusable pieces in `python/sglang/test`: `cpu_test_utils.py`
(pure-torch reference math + tolerances), `quant_ref_utils.py` (hand-written quant
references, explicitly independent of `sglang.srt`), `config_publishers.py` (AST
analysis), `otel_collector.py` (in-process OTLP double), `observability/fake_ray.py`,
and the `ci/ci_register.py` decorator registry. Everything else launches servers,
GPU kernels, or vendor (NPU/XPU/GB300/mooncake/ray) infrastructure.

## Notable PORT-CPU candidates

- `test/manual/lang_frontend/test_choices.py` — 3 tests; pure-numpy scoring for `sglang.lang.choices` (greedy/normalized selection). No server, no GPU. **C3.**
- `test/manual/test_tokenizer_manager.py` + `test_tokenizer_batch_encode.py` + `test_async_dynamic_batch_tokenizer.py` — 29+ mocked asyncio unit tests over TokenizerManager input-format/result-extraction/batching logic. **C4.**
- `test/manual/test_schedule_policy.py` — 14 tests; in-proc `SchedulePolicy`/`RadixCache`/`Req` with array-based fake inputs (import of `sglang.srt.managers` must not drag CUDA — verify). **C8/C5.**
- `test/manual/debug_utils/test_log_parser.py` + `manual/core/test_dynamic_grad_mode.py` — pure-python log-line parser UT and torch-CPU grad-mode UT (plus `quant_ref_utils.py` / `cpu_test_utils.py` as reference oracles, and `test/manual/test_weight_validation.py` — 4 tests, tempfile-only shard validation). **C8/C10.**
- `test/manual/test_deepseek_chat_templates.py` — 4 tests; jinja chat-template tool-call argument escaping (issue #11700). Directly relevant to tool-parsing ports. **C12.**
- `test/manual/test_{dsa_alias_cli_registry_env,aiter_unified_draft_extend_env}.py` — env/CLI flag-registration guards; zero GPU work, catches renames/default flips. **C1/C6.**


<!-- ===== 24-sglang-rust-satellite.md ===== -->

# 24 — sglang satellite test trees (rust + python satellites)

Census of sglang's satellite test trees (radix tree, mm, router, kv-indexer, model-gateway,
minimax sparse ops, aot kernels, multimodal_gen, simulator). Paths relative to
`/home/turq/dev/upstream-test-inventory/sglang/`. `.claude/` dirs skipped (none present).
n = number of `#[test]`/`#[tokio::test]` fns (Rust) or `def test_*` (Python), counted mechanically.

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| rust/sglang-radix-tree/src/tests/unified_tree_core.rs | 341 | unified radix-tree core: insert, match_prefix, split, evict (device+host), lock/unlock refcounts, sessions/namespaces, chunked insert, priorities | yes | tch crate (libtorch CPU) via crate dep | PORT-CPU | C5 |
| rust/sglang-radix-tree/src/tests/node.rs | 124 | node arena, key hashing, value slots, page-aligned prefix pages | yes | tch (libtorch CPU) | PORT-CPU | C5 |
| rust/sglang-radix-tree/src/tests/components/swa.rs | 186 | sliding-window-attention tree component (SWA eviction/branching/seqlen) | yes | tch | PORT-CPU | C5 |
| rust/sglang-radix-tree/src/tests/components/full.rs | 98 | FULL-attention component lifecycle (insert overlap, eviction, host transfer bookkeeping) | yes | tch | PORT-CPU | C5 |
| rust/sglang-radix-tree/src/tests/components/mamba.rs | 63 | mamba/SSD component (mamba_value handling, state eviction) | yes | tch | PORT-CPU | C5 |
| rust/sglang-radix-tree/src/tests/unified_lru_list.rs | 51 | unified LRU list ordering/validation per component | yes | none direct (crate tch) | PORT-CPU | C5 |
| rust/sglang-radix-tree/src/tests/components/base.rs | 4 | TreeComponent trait default behaviors, ComponentType discriminants | yes | tch | PORT-CPU | C5 |
| rust/sglang-radix-tree/src/tests/test_utils.rs | — | shared test helpers (not a test file) | — | — | — | C5 |
| rust/sglang-mm/tests/rlib_is_single_threaded.rs | 1 | rlib single-threadedness guard | yes | cargo test harness | PORT-CPU | C11 |
| rust/sglang-mm/tests/test_golden.py + test_integration.py + test_resize_parity.py + test_hash_fetch.py | 10 | multimodal image/hash encode-decode parity vs golden | no | torch, PIL, soundfile, golden assets | DEFER-GPU | C11 |
| rust/sglang-mm/tests/generate_golden.py | — | golden-data generator (not a test) | — | — | SKIP-INFRA | — |
| experimental/sgl-router/tests/component/policies/bucket_domains.rs | 9 | bucket-domain routing policy | yes | tokio, in-process | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/workers/manager.rs | 10 | worker manager lifecycle/introspection | yes | tokio, mock workers | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/policies/decode.rs | 5 | decode-side routing policy | yes | tokio | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/policies/power_of_two.rs | 5 | power-of-two-choices LB | yes | tokio | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/health/circuit_breaker.rs | 4 | circuit-breaker open/half-open/recovery | yes | tokio | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/discovery/static_urls.rs | 3 | static-URL worker discovery | yes | tokio | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/policies/round_robin.rs | 3 | round-robin policy | yes | tokio | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/policies/fused_score.rs | 3 | fused-score policy | yes | tokio | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/workers/concurrent_state.rs | 2 | worker-registry concurrent state | yes | tokio | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/policies/kv_events_hash_parity.rs | 2 | KV-event hash parity (rust vs py generator) | yes | py-generated fixtures | PORT-CPU | C5/C9 |
| experimental/sgl-router/tests/component/policies/kv_events_tree_concurrent.rs | 2 | concurrent KV-event tree updates | yes | tokio | PORT-CPU | C5/C9 |
| experimental/sgl-router/tests/component/policies/kv_events_two_subscribers.rs | 2 | two-subscriber KV-event fanout | yes | tokio | PORT-CPU | C5/C9 |
| experimental/sgl-router/tests/component/tokenizer/parity.rs | 1 | tokenizer parity vs HF | yes | HF tokenizer fixture | PORT-CPU | C4 |
| experimental/sgl-router/tests/component/policies/cache_prefix_provider.rs | 1 | cache-prefix provider | yes | tokio | PORT-CPU | C5 |
| experimental/sgl-router/tests/component/policies/zmq_helpers.rs | 0 | zmq helper fns (no #[test]) | yes | zmq | PORT-CPU | C9 |
| experimental/sgl-router/tests/component/main.rs + mod.rs files | — | component-test entrypoint/modules | — | — | — | — |
| experimental/sgl-router/tests/proxy/chat_routing.rs | 35 | end-to-end proxy chat routing over mock workers | yes | axum/tower, mock workers | PORT-CPU | C1/C9 |
| experimental/sgl-router/tests/proxy/bucket_routing.rs | 14 | bucket routing through full proxy | yes | axum/tower | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/pd_pool_isolation.rs | 8 | prefill/decode pool isolation | yes | axum/tower | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/shared_prefill_admission.rs | 6 | shared-prefill admission control | yes | axum/tower | PORT-CPU | C5/C9 |
| experimental/sgl-router/tests/proxy/sticky_input_ids.rs | 5 | sticky routing by input ids | yes | axum/tower | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/sticky_routing.rs | 5 | sticky session routing | yes | axum/tower | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/pd_bootstrap_injection.rs | 5 | PD bootstrap config injection | yes | axum/tower | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/cache_aware_input_ids.rs | 4 | cache-aware routing on input ids | yes | axum/tower | PORT-CPU | C5/C9 |
| experimental/sgl-router/tests/proxy/roundrobin_input_ids.rs | 3 | round-robin on input ids | yes | axum/tower | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/failover.rs | 1 | failover when a worker dies mid-traffic (fault injection) | yes | axum/tower, 3 mock workers | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/timeout.rs | 1 | upstream timeout behavior (fault injection) | yes | axum/tower | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/graceful_shutdown.rs | 1 | graceful drain/shutdown | yes | axum/tower | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/header_forwarding.rs | 1 | header forwarding through proxy | yes | axum/tower | PORT-CPU | C9 |
| experimental/sgl-router/tests/proxy/radix_tree_routing.rs | 1 | radix-tree-based cache-aware routing through proxy | yes | axum/tower | PORT-CPU | C5/C9 |
| experimental/sgl-router/tests/proxy/external_indexer_routing.rs | 1 | routing against external KV indexer | yes | axum/tower | PORT-CPU | C5/C9 |
| experimental/sgl-router/tests/proxy/common/{mock_worker,streaming,cache_aware_fixture}.rs | — | mock-worker/streaming test fixtures | yes | — | PORT-CPU | C9 |
| experimental/sgl-router/tests/e2e/chat_completions/*.py | 3 | live-router e2e (load-based policy, validation) | no | live sglang router + model servers | SKIP-INFRA | C9 |
| experimental/sgl-router/tests/e2e/test_chat_smoke.py + test_tokenize_smoke.py | 2 | router smoke vs live backend | no | live backend | SKIP-INFRA | C9 |
| experimental/sgl-router/tests/e2e/infra/*.py | — | e2e harness (gateway/model_pool/model_specs + tests) | no | model pool | SKIP-INFRA | — |
| experimental/sgl-router/tests/e2e/k8s_integration/*.py | 11 | k8s discovery/lifecycle/reconciliation | no | kind/k8s cluster | SKIP-INFRA | C9 |
| experimental/sgl-router/tests/scripts/*.py | — | parity-fixture generators | — | — | SKIP-INFRA | — |
| experimental/sgl-router/sgl-kv-indexer/tests/memory_integration.rs | 14 | in-memory KV indexer backend: apply/match external KV batches, component reports (HBM/DRAM), hit counts | yes | tonic types only | PORT-CPU | C5 |
| experimental/sgl-router/sgl-kv-indexer/tests/grpc_contract.rs | 9 | gRPC contract of KV indexer service (tonic) | yes | tonic, in-process server | PORT-CPU | C5/C9 |
| experimental/sgl-router/sgl-kv-indexer/tests/common/{id,kv,net}.rs | — | indexer test fixtures | yes | — | PORT-CPU | C5 |
| sgl-model-gateway/tests/common/*.rs (11 files) | — | test harness: mock OpenAI/MCP/worker servers, redis test server, TLS certs, test app/config | yes | axum/tower, redis server bin | PORT-CPU | C9 |
| sgl-model-gateway/tests/api/*.rs | 78 | OpenAI-compatible API endpoints, parser endpoints, request formats, responses API, SSE streaming | yes | axum/tower in-process, mock workers | PORT-CPU | C1/C9 |
| sgl-model-gateway/tests/spec/*.rs | 90 | request/response spec types: chat_completion, chat_message, embedding, rerank, responses | yes | serde only | PORT-CPU | C1/C12 |
| sgl-model-gateway/tests/reliability/*.rs | 49 | circuit breaker, fault tolerance, rate limiting, retries, upstream cancel | yes | axum/tower, mock workers | PORT-CPU | C9 |
| sgl-model-gateway/tests/security/*.rs | 39 | auth (api key/jwt), auth integration, mTLS | yes | axum/tower, TLS certs | PORT-CPU | C9 |
| sgl-model-gateway/tests/routing/*.rs | 90 | LB, power-of-two, cache-aware backcompat, header forwarding, payload size, PD routing, policy registry, service/worker discovery & management | yes | axum/tower, mock workers | PORT-CPU | C9 |
| sgl-model-gateway/tests/mcp_test.rs | 12 | MCP tool-call protocol handling (mock MCP server) | yes | axum/tower, mock MCP | PORT-CPU | C12 |
| sgl-model-gateway/tests/wasm_test.rs | 8 | WASM plugin sandbox | yes | wasm runtime | PORT-CPU | C9 |
| sgl-model-gateway/tests/metrics_aggregator_test.rs | 5 | metrics aggregation | yes | — | PORT-CPU | C9 |
| sgl-model-gateway/tests/load_guard_raii_test.rs | 6 | RAII load guard | yes | — | PORT-CPU | C9 |
| sgl-model-gateway/tests/inflight_tracker_test.rs | 3 | inflight-request tracking | yes | — | PORT-CPU | C9 |
| sgl-model-gateway/tests/otel_tracing_test.rs | 2 | OTel tracing spans | yes | otel collector | PORT-CPU | C9 |
| sgl-model-gateway/tests/{api,reliability,routing,security,spec}_tests.rs | — | thin module-aggregator entrypoints (tests live in subdirs) | — | — | — | — |
| sgl-model-gateway/bindings/python/tests/*.py (3 files) | 81 | pyo3 python binding: arg parser, binding surface, startup sequence | yes | compiled pyo3 .so | SKIP-INFRA | C1/C9 |
| sgl-model-gateway/e2e_test/chat_completions/*.py | 49 | live e2e: OpenAI server, function calling, reasoning, thinking, validation | no | live gateway + model | DEFER-GPU | C1/C12 |
| sgl-model-gateway/e2e_test/responses/*.py | 59 | responses API e2e: CRUD, state, streaming, structured output, tools | no | live gateway + model | DEFER-GPU | C1/C12 |
| sgl-model-gateway/e2e_test/embeddings/*.py | 7 | embeddings e2e | no | live backend | DEFER-GPU | C1 |
| sgl-model-gateway/e2e_test/router/*.py | 12 | router MMLU/PD-MMLU eval + worker API | no | live models + eval harness | DEFER-GPU | C9 |
| sgl-model-gateway/e2e_test/benchmarks/*.py | 2 | perf benchmarks (regular + PD) | no | live GPUs | DEFER-GPU | C9 |
| sgl-model-gateway/e2e_test/k8s_integration/*.py | 14 | k8s multi-model/lifecycle/reconciliation/PD-type-change | no | k8s cluster | SKIP-INFRA | C9 |
| sgl-model-gateway/e2e_test/{fixtures,infra}/*.py | — | e2e harness: backend setup, model pool/specs, GPU allocator/monitor | no | — | SKIP-INFRA | — |
| python/sglang/srt/layers/attention/minimax_sparse_ops/tests/*.py (4 files) | 21 | MiniMax sparse attention ops: flash w/ topk idx, fp8 attn GEMM, MSA fp8 parity, sparse GQA | no | CUDA kernels, fp8 | DEFER-GPU | C5 (attn kernels) |
| python/sglang/kernels/aot/tests/test_*.py (27 files) | 71 | AOT CUDA kernels: activation, copy, norm, rope, GEMM (fp8/int8), flash attn/MLA/InfLLM, topk, sampling, kvcacheio, merge_state, gguf | no | CUDA GPU, sgl_kernel | DEFER-GPU | C3/C5/C7 |
| python/sglang/kernels/aot/tests/speculative/*.py (3 files) | — | eagle utils, ngram utils, speculative sampling kernels | no | CUDA | DEFER-GPU | C6 |
| python/sglang/kernels/aot/tests/spatial/*.py (1 file) | — | greenctx stream | no | CUDA | DEFER-GPU | C5 |
| python/sglang/kernels/aot/tests/utils.py + conftest.py | — | kernel-test utilities | — | — | — | — |
| python/sglang/multimodal_gen/test/unit/*.py (top level, ~173 files) | ~2100 | diffusion-model unit tests: loaders, weight readers/snapshots, VAE, attention backends, LoRA, configs, schedulers, quant — most instantiate DiT/VAE on GPU or load weights | mostly no | torch + diffusion model weights | DEFER-GPU | C11 |
| python/sglang/multimodal_gen/test/unit/realtime/*.py (8 files) | 136 | realtime denoising: causal denoise, control signals, output transport/materialization, VAE | no | GPU | DEFER-GPU | C11 |
| python/sglang/multimodal_gen/test/unit/sana_wm/*.py (7 files) | 121 | Sana-WAM streaming pipeline: stages, cached streaming, VAE, config | no | GPU | DEFER-GPU | C11 |
| python/sglang/multimodal_gen/test/unit/musa/layers/*.py (2 files) | — | MUSA rmsnorm/silu kernels | no | MUSA GPU | SKIP-OTHERMODEL | C11 |
| python/sglang/multimodal_gen/test/server/ascend/*.py | — | Ascend NPU server tests | no | Ascend NPU | SKIP-OTHERMODEL | C11 |
| python/sglang/multimodal_gen/test/{runner,scripts,partitioning.py,test_utils.py} | — | suite runners, CI-output/perf-baseline generators | — | — | SKIP-INFRA | — |
| tools/sglang-simulator/test/test_simulation_cache_hit_ratio.py | 2 | discrete-event sim: per-cache-tier hit ratio reporting (L1/L2/evict) | yes | sglang_simulator pkg + sglang.srt.server_args import | PORT-CPU | C5/C8 |
| tools/sglang-simulator/test/test_simulation_offline_blocking.py | 1 | sim: offline blocking vs request-rate timing tolerances | yes | sglang_simulator pkg | PORT-CPU | C8 |
| tools/sglang-simulator/test/test_simulation_sglang_runner.py | 1 | sim runner: paged decode simulation | yes | sglang.srt imports | PORT-CPU | C8 |
| tools/sglang-simulator/test/test_simulation_sglang_serving.py | 2 | sim serving loop under SIM_CONFIGS, decode metrics | yes | sglang_simulator pkg | PORT-CPU | C8 |

Totals: ~870 radix-tree Rust tests (all PORT-CPU/C5), ~150 router component+proxy Rust tests
(PORT-CPU/C9, C5 for cache-aware/KV-event ones), 23 kv-indexer tests (PORT-CPU/C5),
~376 gateway Rust tests (PORT-CPU, C1/C9/C12), ~81 pyo3 binding tests, ~230 gateway python e2e
(mostly DEFER-GPU/SKIP-INFRA), 21 minimax + ~75 aot kernel tests (DEFER-GPU), ~2360
multimodal_gen unit tests (DEFER-GPU/C11, mostly SKIP-OTHERMODEL), 6 simulator tests (PORT-CPU/C8).

## Notable PORT-CPU candidates

- **sglang-radix-tree `unified_tree_core.rs` (341 tests)** — the single highest-value Rust port batch in the whole inventory for ds41rt-hostcache: insert/match_prefix/split/evict/lockref/session semantics of the production radix tree, pure CPU + libtorch-CPU tensors.
- **sglang-radix-tree `components/swa.rs` (186) + `full.rs` (98) + `mamba.rs` (63)** — component-layer eviction and lifecycle invariants on top of the core; same hostcache target, same tch-only pin.
- **sglang-radix-tree `node.rs` (124) + `unified_lru_list.rs` (51)** — arena/key-hashing and LRU-ordering invariants; cheap, fast, directly reusable for ds41rt prefix-reuse correctness.
- **sgl-kv-indexer `memory_integration.rs` (14) + `grpc_contract.rs` (9)** — in-memory external-KV match/apply semantics with HBM/DRAM component reports; exactly the ds41rt C5 indexer surface, tonic-only pin.
- **sgl-router `proxy/failover.rs` + `timeout.rs` + `graceful_shutdown.rs`** — fault-injection patterns (worker death mid-traffic, upstream timeout, drain) that map directly onto ds41rt-transport fault handling; mock-worker harness is fully CPU.
- **sgl-router `component/health/circuit_breaker.rs` (4) + gateway `reliability/*.rs` (49)** — circuit-breaker open/half-open/recovery and retry/cancel behavior, CPU-runnable with in-process mocks; the reference semantics for ds41rt C9 resilience.
- **sgl-router `policies/kv_events_*.rs` (7) + `cache_prefix_provider.rs` + proxy `radix_tree_routing.rs` / `cache_aware_input_ids.rs` / `external_indexer_routing.rs`** — router-side prefix-cache routing and KV-event tree concurrency, C5∩C9 overlap with ds41rt prefix reuse.
- **sgl-router `component/policies/power_of_two.rs` + `round_robin.rs` + `fused_score.rs` (11)** — classic LB policy unit tests, trivially portable CPU reference behavior for C9.
- **model-gateway `spec/*.rs` (90)** — serde-level OpenAI/responses/rerank/embedding spec types; zero-service CPU tests useful for ds41rt C1 API contract coverage.
- **model-gateway `mcp_test.rs` (12)** — MCP tool-call protocol tests against a mock MCP server; direct input to ds41rt C12 tool-parsing coverage.


<!-- ===== 25-llamacpp.md ===== -->

# llama.cpp test-suite inventory

Census of llama.cpp test suites vs the ds41rt component map. Read-only on sources.
Target codes: C1 OpenAI API, C2 constrained/structured output (xgrammar), C3 sampler, C4 tokenizer/chat template, C5 KV cache, C6 spec decode, C9 transport, C10 weights/quant, C11 vision, C12 tool parsing.

## tests/ (C++ unit tests, registered in tests/CMakeLists.txt)

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| tests/test-alloc.cpp | ~14 | ggml tensor allocator | yes | buffer split/merge, reuse-and-free, prefer-already-alloc, max-size/too-many-tensors errors, multi buffer types, view-inplace, graph alloc deps, optimize | PORT-CPU (pattern only) | – |
| tests/test-arg-parser.cpp | ~20 | common arg parsing | yes | --repeat-last-n vs --dry-penalty-last-n, repeat-penalty flag forms, unknown-arg errors | SKIP (llama.cpp CLI surface, not ds41rt) | – |
| tests/test-autorelease.cpp | 1 | model autorelease | model | autorelease pool smoke | DEFER-GPU | – |
| tests/test-backend-ops.cpp | ~200 | ggml backend op correctness | gpu | per-op golden tests over all backends (unary/binary/cmul/conv/rope/flash-attn…) | DEFER-GPU | – |
| tests/test-backend-sampler.cpp | ~10 | sampler on backend | mixed | top-k/top-p/temp via ggml-backend graph vs CPU sampler chain | DEFER-GPU | C3 (CPU variant in test-sampling is the portable one) |
| tests/test-barrier.cpp | 3 | ggml backend barrier | yes | single/multi-graph barrier sync | SKIP-INFRA (ggml-internal) | – |
| tests/test-batch-alloc.cpp | 4 | llama batch alloc | yes | init, keep-tail, split, mrope batch layouts | PORT-CPU | C5 (concepts) |
| tests/test-chat-analysis.cpp | ~46 | chat-template auto-analysis | yes | tool-call analysis (full/pure-JSON/fn-name markers, edge cases), reasoning-format detection (nemotron/cohere/laguna/smollm3 variants), marker separation | PORT-CPU | C4/C12 |
| tests/test-chat-auto-parser.cpp | ~60 | chat auto-parser | yes | diff-split (prefix/suffix/overlap/tag boundaries), variant comparison, tool formats (nemotron/laguna/cohere/smollm3/seed-oss), reasoning detection, quote normalization | PORT-CPU | C4/C12 |
| tests/test-chat.cpp | ~200 | chat templates + formats | yes | template application vs expected outputs (205 embedded cases), oaicompat msg/tools JSON conversion, delimiters split, responses→chatcmpl, lfm2 parser, generation prompt, PEG parser compare | PORT-CPU | C4 |
| tests/test-chat-peg-parser.cpp | 6 | PEG chat parser | yes | native/qwen3-coder/qwen3 templates, prefix tool names, tagged parser, permutation | PORT-CPU | C4/C2 (peg→C2 per map) |
| tests/test-chat-template.cpp | 6 | jinja chat template | yes | template render + format extraction for key models | PORT-CPU | C4 |
| tests/test-col2im-1d.cpp | 2 | ggml col2im op | yes | col2im-1d forward vs reference | SKIP-INFRA (ggml op) | – |
| tests/test-double-float.cpp | 2 | ggml double-float | yes | float↔double rounding ops | SKIP-INFRA | – |
| tests/test-export-graph-ops.cpp | 2 | graph op export | yes | export/import graph op roundtrip | SKIP-INFRA | – |
| tests/test-fusion.cpp | 4 | op fusion | model | prefill/decode fused vs unfused NMSE | DEFER-GPU | – |
| tests/test-gbnf-validator.cpp | ~30 | GBNF validation | yes | grammar strings accepted/rejected (one case per line) | PORT-CPU | C2 |
| tests/test-gguf.cpp | 4 groups | GGUF container | yes | handcrafted file, roundtrip (file/buffer/callback read modes), set-kv — property-style, randomized | PORT-CPU | C10 |
| tests/test-gguf-model-data.cpp | 2 | GGUF model-data helper | yes | glm4moe/step35 key composition | PORT-CPU | C10 |
| tests/test-grammar-integration.cpp | ~12 | grammar build/accept | yes | simple/complex grammar, quantifiers, special chars, JSON-schema via grammar, left-recursion/missing-ref/missing-root failures, custom root | PORT-CPU | C2 |
| tests/test-grammar-llguidance.cpp | 7 | llguidance backend parity | yes | same grammar/schema corpus forced through llguidance sampler chain | PORT-CPU (xgrammar is the ds41rt analog) | C2 |
| tests/test-grammar-parser.cpp | 50 | GBNF parser | yes | rule/grammar string → parsed rule counts (root_1, expr_5, …) | PORT-CPU | C2 |
| tests/test-jinja.cpp | 22 | jinja engine | yes | literals, expressions, conditionals, loops, filters, string/array/object methods, macros, namespace, set, tests, whitespace control, comments, hasher, fuzzing, stats, template_cpp vs template_py parity | PORT-CPU | C4 |
| tests/test-json-schema.cpp | 14 | JSON-schema→GBNF output validation | yes | primitives, string/integer, array/tuple, object, any/anyOf/allOf, const/enum, $ref, may-be-string, value types, errors | PORT-CPU | C2 |
| tests/test-json-schema-to-grammar.cpp | ~170 | JSON-schema→GBNF golden corpus | yes | ~170 schema→grammar goldens (min/max, patterns, required, nesting) | PORT-CPU (top artifact for xgrammar port) | C2 |
| tests/test-llama-grammar.cpp | 13 | llama grammar engine | yes | stack-based accept/reject cases (expr/ident/num/root…) | PORT-CPU | C2 |
| tests/test-llama-archs.cpp | ~19 | arch backend parity | gpu | per-arch (gguf vocab models) backend vs CPU golden | DEFER-GPU | – |
| tests/test-log.cpp | 1 | logging | yes | log callback smoke | SKIP-INFRA | – |
| tests/test-model-load-cancel.cpp | 1 | model load cancel | model | interrupt mid-load, no crash | DEFER-GPU | – |
| tests/test-model-resolution.cpp | 2 | HF model resolution | yes | plan resolution, task assembly (httplib) | SKIP-INFRA (llama.cpp downloader) | – |
| tests/test-mtmd-impl.cpp | 2 | mtmd internals | yes | image preprocessor lfm2, temporal merge grouping | DEFER-GPU (vision impl) | C11 |
| tests/test-opt.cpp | ~5 | ggml optimizer (MNIST-ish) | yes | dataset split, grad check, forward-backward weights/results, epoch-vs-fit equivalence across optimizers | PORT-CPU (pattern) | – |
| tests/test-peg-parser.cpp | 1 | peg parser smoke | yes | wrapper entry | PORT-CPU | C2 (peg) |
| tests/peg-parser/test-basic.cpp | 1 fn (~40 cases) | PEG basics | yes | literal/seq/choice/rep/composite parsing | PORT-CPU | C2 |
| tests/peg-parser/test-gbnf-generation.cpp | 1 fn (~40 cases) | PEG→GBNF generation | yes | grammar generation from PEG defs, gbnf snapshots | PORT-CPU | C2 |
| tests/peg-parser/test-json-parser.cpp | 1 fn (~15 cases) | PEG JSON parser | yes | JSON parse/serialize roundtrips, errors | PORT-CPU | C2 |
| tests/peg-parser/test-json-serialization.cpp | 1 fn | PEG JSON serialization | yes | serialization format | PORT-CPU | C2 |
| tests/peg-parser/test-python-dict-parser.cpp | 1 fn (~30 cases) | PEG python-dict parser | yes | dict-literal parsing (tool args) | PORT-CPU | C2/C12 |
| tests/peg-parser/test-unicode.cpp | 1 fn (~30 cases) | PEG unicode | yes | unicode char/range/escapes in PEG | PORT-CPU | C2 |
| tests/test-quantize-fns.cpp | 2 | quant fn correctness | mixed | vec_dot f32 vs q-types vs reference error, per-type quantize/dequant error budget (CPU vs backend) | PORT-CPU (CPU side) | C10 |
| tests/test-quantize-perf.cpp | 1 | quant perf | yes | quant/dequant timing smoke | SKIP-INFRA (perf, not correctness) | – |
| tests/test-quantize-stats.cpp | 2 | quant stats | yes | roundtrip error on chunk/layer | PORT-CPU (pattern) | C10 |
| tests/test-quant-type-selection.cpp | ~45 | tensor-type selection | yes | per-arch/role type-selection snapshots (snapshots/*.schema), remote regen mode | PORT-CPU | C10 |
| tests/test-reasoning-budget.cpp | 6 | reasoning-budget parser | yes | budget counting, clone mid-counting/mid-forcing, end-match, force-manual, UTF-8 boundary detection | PORT-CPU | C4 |
| tests/test-recurrent-state-rollback.cpp | 2 (+4 variants) | recurrent-state rollback | mixed | rollback, multi-seq split replay; nemotron-h/dsv4/kimi-k3 model variants (needs models) | DEFER-GPU (model-gated) | C5 (concepts) |
| tests/test-rope.cpp | 4 | rope op | yes | rope f32/f16, context swap | SKIP-INFRA (ggml op; CPU-only though) | – |
| tests/test-rpc-multi-server.cpp + .sh | 1 | rpc backend isolation | gpu (2 rpc servers) | tensor alloc on server A must not be computable by server B; script orchestrates 2 rpc servers | DEFER-GPU | C9 (concepts) |
| tests/test-rset-release.cpp | 1 | memory reuse | model | alloc-after-free rset release smoke | DEFER-GPU | C5 |
| tests/test-sampling.cpp | ~13 fns (~60 calls) | sampler chain | yes | dist singleton RNG, temp/temp-ext, top-k, top-p, min-p, xtc, typical, penalties, dry, top-n-sigma, sampler-queue sequence handling, perf | PORT-CPU (top artifact for C3) | C3 |
| tests/test-save-load-state.cpp | 7 | session state save/load | mixed | num/seq_cp device+host+scatter/seq_rm_isolated/state_load/state_roundtrip (needs generated models) | PORT-CPU (CPU host paths) | C5 |
| tests/test-state-restore-fragmented.cpp | 1 | fragmented state restore | model | restore state with non-contiguous slots | DEFER-GPU (model-gated) | C5 |
| tests/test-thread-safety.cpp | 1 | thread safety | model | parallel decode smoke, ngl 99 | DEFER-GPU | – |
| tests/test-tokenizer-0.cpp | 16 registered | tokenizer roundtrip | yes | get_vocab/add_special/eol/assertion matrix over 16 ggml-vocab models (bert-bge…starcoder); drives test-tokenizer-0-* ctest names | PORT-CPU | C4 |
| tests/test-tokenizer-0.py + .sh, test-tokenizer-random.py, test-tokenizers-repo.sh | – | tokenizer repo sweep | scripts | full HF tokenizers repo diff vs llama.cpp (py driver) | PORT-CPU (harness pattern) | C4 |
| tests/test-tokenizer-1-bpe.cpp | 1 (mostly commented) | BPE merge-order | yes | tokenize/untokenize vs reference incl. --ignore-merges mode | PORT-CPU | C4 |
| tests/test-tokenizer-1-spm.cpp | 1 | SPM tokenization | yes | llama-spm roundtrip, pre-tokenization splits | PORT-CPU | C4 |
| tests/test-c.c | 1 | C API compile | yes | extern "C" compile smoke | SKIP-INFRA | – |
| tests/test-mtmd-c-api.c | 1 | mtmd C API | gpu | bitmap decode → tokens smoke | DEFER-GPU | C11 |
| tests/testing.h, gguf-model-data.{cpp,h}, fusion/MTL.csv, snapshots/*.schema | – | support/data | – | test harness; MTL fusion table; 13 chat-template/quant snapshots (deepseek-v3.1, gpt-oss-120b, qwen3.5-397b…) | data artifacts | C4/C10 |

## tools/server/tests (Python integration tests, pytest, utils.py server harness)

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| unit/test_basic.py | 7 | server boot/health | model | health, embedding flag, tokenize, slots, model info, OAI compat discovery | SKIP-INFRA (boot smoke) | C1 |
| unit/test_chat_completion.py | 24 | OpenAI chat API | model | chat completion (stream/non-stream, OpenAI lib), chat template + assistant prefill, continue-final-message (vllm compat, mutual exclusion), apply_chat_template, response_format JSON-schema, grammar, invalid grammar, logprobs (stream), logit_bias, context-size-exceeded, timings-per-token, n>1 choices, token counts, cached tokens | PORT-CPU as API-contract patterns (assertions portable; model-gated) | C1/C2 |
| unit/test_completion.py | 24 | legacy completions API | model | stream parity, seed determinism, batch-size invariance, cache vs nocache, tokens input, parallel slots, unified endpoint, n_probs (incl. post-backend-sampling), logit_bias, cancel request, prompt cache | PORT-CPU as contract patterns | C1/C3 |
| unit/test_tool_call.py | 9 | forced tool calling | model | required-tool tiny/real model, no-tool-call, weather/calc/thoughts/hello-world grammars | PORT-CPU | C12/C2 |
| unit/test_tools_builtin.py | 19 | built-in tool server | model | grep/read/write/edit file, edit rejects non-unique/overlapping, exec shell stream, cwd/runtime headers, injection rejection (ssh/container opts), docker cleanup, glob search limits | SKIP (llama.cpp built-in tool sandbox, not ds41rt) | C12 |
| unit/test_mcp_servers.py | 21 | MCP integration | model | tool listing, invocation, bad command, multi-server, retry/fail-once, json config, slot independence, concurrency, timeout, respawn after crash, notifications/burst, config errors | SKIP (MCP infra) | C12 |
| unit/test_speculative.py | 7 | speculative decoding | gpu (draft) | with/without draft, draft min/max, synth determinism, ignores target tokens, slot ctx not exceeded, ctx shift, parallel multi-request | DEFER-GPU | C6 |
| unit/test_router.py | 16 | model router | model | router props, streaming, unload, LRU eviction, queue coalescing/FIFO/disconnect/no-evict-busy, autoload, api-key, reload, dedup cache, remote preset, download/delete | SKIP-INFRA (llama.cpp router) | – |
| unit/test_security.py | 12 | API security | model | public endpoint access, static assets, api-key (incl. Anthropic header, OAI lib), CORS options/origins, proxy header forwarding | PORT-CPU as contract patterns | C1 |
| unit/test_compat_anthropic.py | 29 | Anthropic protocol | model | messages API mapping, streaming, tools, system blocks, thinking, errors | PORT-CPU as API-contract patterns | C1 |
| unit/test_compat_gcp.py | 2 | GCP protocol | model | vertex/openai-compatible surface | SKIP (GCP-specific) | C1 |
| unit/test_compat_oai_responses.py | 3 | OAI Responses API | model | responses endpoint mapping | PORT-CPU as contract patterns | C1 |
| unit/test_vision_api.py | 6 | vision API | gpu (mtmd) | multimodal capability flags, vision chat/completion/token count/embedding | DEFER-GPU | C11 |
| unit/test_embedding.py | test_embedding.py 15 / test_rerank.py 6 | embeddings/rerank | model | embedding consistency, dimensions, normalization; rerank scoring | SKIP (not ds41rt surface, verify) | C1 |
| unit/test_metrics.py | 9 | /metrics | model | prometheus counter/histogram sanity | SKIP-INFRA | – |
| unit/test_stream.py, test_tokenize.py, test_template.py, test_ctx_shift.py, test_sleep.py, test_lora.py, test_proxy.py, test_infill.py, test_ignore_eos.py, test_kv_keep_only_active.py, test_slot_save.py | 43 total | misc server behaviors | model | SSE stream events, tokenize/detokenize, template endpoint, ctx shift, sleep/wake, LoRA hot-load, auth proxy, infill, ignore_eos, KV keep-active, slot save/restore | MIXED: ctx_shift/slot_save/kv_keep → PORT-CPU (C5 concepts); rest SKIP | C1/C5 |
| fixtures/mcp_*.py, utils.py, conftest.py, pytest.ini, tests.sh | – | harness | – | python server fixtures (echo/burst/crash/slow/malformed MCP), server spawn utils, CI runner | SKIP-INFRA | – |

## gguf-py/tests (GGUF format tests)

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| test_gguf_reader_validation.py | 2 | reader robustness | yes | n_dims upper bound (evil 1e6-dim file rejected, no EOF overread); dims-product uint64 wraparound rejected | PORT-CPU (exact patterns apply to any GGUF loader) | C10 |
| test_metadata.py | 5 | metadata heuristics | yes | id→title, model-id components, metadata heuristic from model card / HF parameters / model dir | SKIP (HF-metadata naming heuristic, llama.cpp-specific) | – |
| test_quants.py | 0 test fns | quant type mapping | yes | ctypes ggml quant enum mapping table | PORT-CPU (reference table for C10 mapping) | C10 |

## tools/mtmd/tests (multimodal)

| path | n | area | cpu | pins | verdict | target |
|---|---|---|---|---|---|---|
| test-deepseek-ocr.py + test-1-positive.png + test-1-ground-truth.txt | 1 spec (many model/image cases) | OCR quality eval | gpu | mtmd-cli OCR output scored by CER/chrF vs HF reference scores with tolerances (DeepSeek-OCR v1/v2/unlimited, tile looping, DRY, grounding-strip) | DEFER-GPU (eval harness; CER/chrF method is reusable) | C11 |

## Notable PORT-CPU candidates

- **test-json-schema-to-grammar.cpp (~170 goldens)** — the single most valuable artifact; JSON-schema→grammar golden corpus to re-validate xgrammar's converter case-by-case.
- **test-json-schema.cpp (14)** — schema semantics pins (anyOf/allOf/$ref/const/tuple/errors) that any converter must satisfy; port as xgrammar assertions.
- **test-grammar-integration.cpp + test-grammar-llguidance.cpp** — same grammar/schema corpus run against two engines; the parity method transfers directly to an xgrammar-vs-expected harness.
- **test-grammar-parser.cpp (50) + test-llama-grammar.cpp (13) + test-gbnf-validator.cpp (~30)** — GBNF parse/accept/reject vectors; cheap to port as xgrammar grammar-compiler tests.
- **test-sampling.cpp (~60 calls)** — exact probability vectors for temp/top-k/top-p/min-p/xtc/typical/penalties/dry/top-n-sigma plus sampler-chain sequencing; re-implement against ds41rt's sampler (C3).
- **test-tokenizer-0.cpp + test-tokenizer-1-{bpe,spm}.cpp + test-tokenizers-repo.sh** — vocab-roundtrip matrix (16 vocabs, special tokens, BPE merge-order/ignore-merges, SPM splits) — direct C4 pattern.
- **test-jinja.cpp (22 fns) + test-chat-template.cpp + test-chat.cpp (~200 cases)** — jinja engine semantics + chat-template golden outputs; C4 core, mostly model-independent.
- **test-chat-analysis.cpp + test-chat-auto-parser.cpp (~100 cases)** — tool-call/reasoning-format auto-detection vectors; feeds C4/C12.
- **gguf-py/tests/test_gguf_reader_validation.py (2)** — adversarial container cases (dims bound, uint64 wraparound); apply verbatim to ds41rt's loader.
- **test-gguf.cpp (4 property groups)** — randomized handcrafted/roundtrip/set-kv GGUF property tests; re-target at ds41rt's GGUF writer/reader.
- **test-quant-type-selection.cpp + snapshots/*.schema** — per-arch tensor-type selection goldens incl. deepseek-v3.1 and qwen3.5 schemas; C10.
- **test-save-load-state.cpp / test-batch-alloc.cpp / test-recurrent-state-rollback.cpp** — KV/state semantics vectors (seq cp/rm/split, keep-tail, rollback) portable as CPU-side C5 concept tests.
