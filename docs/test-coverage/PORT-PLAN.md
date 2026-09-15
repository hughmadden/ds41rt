# PORT-PLAN — CPU-portable upstream tests, grouped by ds41rt target

Every row is an upstream test file (or rolled-up dir) inventoried as PORT-CPU.
Tranches pick from these groups; the remainder stays here as the port backlog.


## C1 — OpenAI protocol / API surface (339 rows)

| upstream path | n | pins |
|---|---|---|
| function_call/ (20 files: hermes/deepseekv4/dots/glm47/hunyuan/k2_v3/kimik3/llama32/minicpm5/minimax_m3/mistral/muse_glimmer/poolside_v1/spark25 detectors, parser, json_schema_constraint, normalize_schema_types, parallel_tool_calls, unknown_tool_name) | 581 | cpu_ci 20/20 |
| entrypoints/openai/ — serving: serving_{chat,completions,embedding,responses,responses_stream,transcription}, transcription_adapters, whisper_adapter, exa_search | 305 | cpu_ci |
| utils/ — generic (common, auth, http_server_auth, field_validators, invariants, json_response, patch_tokenizer, profile_merger, subprocess_watchdog, tensor_bridge, weight_checker{,_comparator}, weight_versions, gauge_histogram, diffusion_torch_fallback) | 251 | cpu_ci |
| tests/peg-parser/test-python-dict-parser.cpp | 1 fn (~30 cases) | dict-literal parsing (tool args) |
| tool_parsers/test_utils.py | 128 | none (pure py) |
| managers/ — multimodal (mm_embedding_length, mm_embed_scatter, mm_hashes, mm_process_config, mm_shm_error_consensus, mm_utils_split, multimimodal_abort_cleanup, embed_overrides) | 120 | cpu_ci |
| layers/attention/ — logic (dsa_{head_gate_guard,mqa_logits_chunking}, encoder_decoder_varlen_gather, flashattention_{graph_metadata,pa_swa_prefill_lens_size}, gdn_{flashinfer_alignment,prefill_backend_policy}, index_topk_share, kv_translate_ownership, linear_attn_config, mamba_track_state_dtype, mla_decode_{forced_splits,geometry}, verify_mask, vision_{backend_selection,max_seqlen,seqlens,strided_qkv}) | 113 | 20/25 cpu_ci |
| multimodal/ — processors (base_processor_{bad_input,image_decode}, pixtral, nano_nemotron_vl_processor, glm4v_mixed_offsets, dots_note_omni, evs, media_artifact_processor, mrope_encoder_utils, preprocess_cache, precomputed_embedding_validation, processor_{async_call_sites,clone_isolation,device_selection}, feature_materialization) | 112 | cpu_ci |
| rust/src/chat/tests/roundtrip.rs | 1 (parametrized ~10) | real HF models: Qwen3-0.6B, Qwen3.5-4B, MiniMax-M2.5/M3, DeepSeek-V4-Flash, **DeepSeek-V4.1-Flash**, V3.2-Exp, GLM-4.5, GLM-4.7-Flash |
| rust/src/chat/tests/roundtrip.rs | 1 (parametrized ~10) | real HF models: Qwen3-0.6B, Qwen3.5-4B, MiniMax-M2.5/M3, DeepSeek-V4-Flash, **DeepSeek-V4.1-Flash**, V3.2-Exp, GLM-4.5, GLM-4.7-Flash |
| layers/quantization/ — logic (compressed_tensors_*, fp4_kv_cache_quant_method, fp8_utils_mxfp4, humming_w4afp8_schemas, int8_linear_methods, marlin_utils_fp8, modelopt_nvfp4{,_moe_scales}, quark_config, quark_utils, unquant_apply_with_addend) | 106 | cpu_ci |
| cohere/test_serving_conversion.py | 100 | none |
| parser/engine/test_parser_engine.py | 97 | none (mock tokenizer) |
| entrypoints/openai/ — protocol core: protocol, responses_protocol, responses_custom_tools, audio_chunking, matched_stop, utils | 92 | cpu_ci |
| sgl-model-gateway/tests/spec/*.rs | 90 | serde only |
| sgl-model-gateway/tests/spec/*.rs | 90 | serde only |
| unit_tests/test_chat_utils.py | 81 | HF tokenizers, image/audio/video assets |
| unit_tests/test_chat_utils.py | 81 | HF tokenizers, image/audio/video assets |
| sgl-model-gateway/tests/api/*.rs | 78 | axum/tower in-process, mock workers |
| parser/engine/test_deepseek_v4.py | 68 | none (mock tokenizer) |
| `parser/test_harmony_utils.py` | 66 | pure unit |
| `parser/test_harmony_utils.py` | 66 | pure unit |
| parser/engine/test_inkling.py | 64 | none (mock tokenizer) |
| tool_parsers/test_gemma4_tool_parser.py | 63 | mock tok |
| parser/engine/test_qwen3_reasoning.py | 61 | none (mock tokenizer) |
| function_call/test_kimik2_detector.py | 60 | none (pure parsing) |
| function_call/test_kimik2_detector.py | 60 | none (pure parsing) |
| tests/test-chat-auto-parser.cpp | ~60 | diff-split (prefix/suffix/overlap/tag boundaries), variant comparison, tool formats (nemotron/laguna/cohere/smollm3/seed-oss), reasoning detection, quote normalization |
| parser/engine/test_gemma4_streaming_reasoning.py | 58 | none (mock tokenizer) |
| tests/multimodal/test_audio.py | 54 | pyav, scipy, torchaudio |
| tests/v1/metrics/test_perf_metrics.py | 51 | HF configs only |
| parser/engine/test_qwen3.py | 51 | none (mock tokenizer) |
| models/ — deepseek (deepseek_mla_dispatch, deepseek_nextn_mm_embed, deepseek_v4_{amd_fused_mhc,amd_wo_a_bf16,mxfp4_shared_expert_requant,rope_policy,shared_expert_fusion,unified_fp8_q_pair}) | 51 | cpu_ci 48/52 dir |
| tool_parsers/test_deepseekv32_tool_parser.py | 48 | mock tok; xgrammar structural tags |
| tests/test-chat-analysis.cpp | ~46 | tool-call analysis (full/pure-JSON/fn-name markers, edge cases), reasoning-format detection (nemotron/cohere/laguna/smollm3 variants), marker separation |
| tests/test-quant-type-selection.cpp | ~45 | per-arch/role type-selection snapshots (snapshots/*.schema), remote regen mode |
| parser/engine/test_engine.py | 44 | none (mock tokenizer) |
| tests/multimodal/test_video.py | 44 | HF public asset downloads |
| parser/mistral/test_tool_calls.py | 42 | none |
| test_precision_baseline_store.py | 42 | cpu_ci |
| multimodal/processing/* (41: test_common, test_tensor_schema, transformers_backend, 38 model processors) | 41 | CPU for schema/common |
| cohere/test_serving_streaming.py | 38 | none |
| model_executor/model_loader/test_ep_weight_filter.py | 37 | temp safetensors |
| tool_parsers/test_minicpm5xml_tool_parser.py | 36 | gpt2 tok |
| rust/src/engine-core-client/src/tests/client.rs | ~36 | zeromq, tokio; python_compat fixtures |
| `responses/test_serving_responses.py` | 35 | mocks / monkeypatch |
| `responses/test_serving_responses.py` | 35 | mocks / monkeypatch |
| parser/test_harmony.py | 35 | none |
| experimental/sgl-router/tests/proxy/chat_routing.rs | 35 | axum/tower, mock workers |
| ec_connector/unit/cpu/scheduler/test_embedding_cache.py | 34 | pure ints, no mmap/torch |
| `chat_completion/test_serving_chat.py` | 34 | mocks; gpt-oss server for 2 tests |
| `chat_completion/test_serving_chat.py` | 34 | mocks; gpt-oss server for 2 tests |
| pooling/embed/test_io_processor.py | 34 | none |
| renderers/test_sparse_tensor_validation.py | 34 | none |
| `responses/test_responses_utils.py` | 33 | pure unit |
| tests/v1/engine/test_admission_control.py | 31 | pure mocks; 503 mapping |
| tests/config/test_multimodal_config.py | 31 | none (MagicMock) |
| parser/engine/test_token_id_scanner.py | 30 | none (mock tokenizer) |
| tool_parsers/test_structural_tag_registry.py | 29 | xgrammar lib; mock/gpt2 toks |
| cpu/ small torch-unit group | ~29 | test_store_cache.py, test_comm.py, test_binding.py |
| unit/test_compat_anthropic.py | 29 | messages API mapping, streaming, tools, system blocks, thinking, errors |
| `completion/test_completion_error.py` | 28 | MagicMock engine, AsyncLLM spec |
| cohere/test_api_router.py | 28 | monkeypatch |
| ec_connector/unit/test_ec_example_connector.py | 27 | bulk CPU mocks; 1 test cuda-gated |
| tool_parsers/test_kimi_k2_tool_parser.py | 26 | HF kimi-k2 tok (trust_remote_code) |
| model_executor/test_jit_warmup.py | 26 | pure logic |
| tests/multimodal/test_processing.py | 26 | HF processors: llava-v1.6-mistral-7b, Qwen2-VL-2B |
| tool_use/test_kimi_k3_tool_parser.py | 25 | none |
| model_executor/model_loader/test_reload.py | 25 | mocks, WeakKeyDictionary |
| reasoning/test_base_thinking_reasoning_parser.py | 24 | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B tok |
| unit/test_chat_completion.py | 24 | chat completion (stream/non-stream, OpenAI lib), chat template + assistant prefill, continue-final-message (vllm compat, mutual exclusion), apply_chat_template, response_format JSON-schema, grammar, invalid grammar, logprobs (stream), logit_bias, context-size-exceeded, timings-per-token, n>1 choices, token counts, cached tokens |
| unit/test_completion.py | 24 | stream parity, seed determinism, batch-size invariance, cache vs nocache, tokens input, parallel slots, unified endpoint, n_probs (incl. post-backend-sampling), logit_bias, cancel request, prompt cache |
| tool_parsers/test_qwen3coder_tool_parser.py | 23 | HF qwen3 tok; xgrammar |
| unit_tests/test_context.py | 22 | openai_harmony, mocked outputs |
| reasoning/test_minimax_m3_reasoning_parser.py | 22 | none |
| tests/quantization/test_quantization_config_args.py | 22 | none |
| tests/utils_/test_argparse_utils.py | 22 | transformers import (one util) |
| tests/jit_monitor/test_hooks.py | 22 | – |
| ec_connector/unit/cpu/scheduler/test_scheduler.py | 21 | real ECSharedRegion mmap, CPU tensors |
| tool_parsers/test_minimax_m2_tool_parser.py | 21 | fake tok stub |
| tool_use/test_muse_glimmer.py | 21 | mock tok |
| parser/test_streaming.py | 20 | none |
| tests/tokenizers_/test_deepseek_v4.py | 20 | none (fake tokenizer) |
| tests/utils_/test_tensor_schema.py | 20 | imports glm4v/phi3v/granite schemas |
| pooling/embed/test_protocol.py | 19 | none |
| tool_parsers/test_apertus_tool_parser.py | 19 | mock tok |
| model_executor/test_weight_utils.py | 19 | temp files, hf hub mocks (some network) |
| reasoning/test_kimi_k3_reasoning_parser.py | 18 | none |
| tests/multimodal/test_embedding_shape_validation_unit.py | 18 | – |
| multimodal/ — misc (audio_container_decode, media_url_security) | 18 | cpu_ci |
| `responses/test_response_input_to_harmony.py` | 17 | pure unit |
| multimodal/llm/test_mm_processor_kwargs.py | 17 | monkeypatch |
| tool_parsers/test_step3p5_tool_parser.py | 17 | HF step3.5 tok |
| tool_use/test_responses_request_validations.py | 17 | none |
| parser/test_parse.py | 17 | none |
| parser/cohere/test_structural_tags.py | 17 | none |
| parser/engine/test_deepseek_v32.py | 17 | none (mock tokenizer) |
| rust/src/chat/tests/chat.rs | 17 | TestTokenizer (fake), mock engine |
| rust/src/chat/tests/chat.rs | 17 | TestTokenizer (fake), mock engine |
| `responses/test_harmony_utils.py` | 16 | pure unit |
| parser/test_include_reasoning.py | 16 | none |
| `kernels/test_deepseek_v4_cpu_kernels.py` | 16 | 1 |
| `chat_completion/test_chat_error.py` | 15 | MagicMock/AsyncMock engine |
| tool_parsers/test_llama3_json_tool_parser.py | 15 | HF llama-3 tok |
| tool_parsers/test_functiongemma_tool_parser.py | 15 | mock tok |
| tests/multimodal/test_hasher.py | 15 | local assets/ |
| `responses/test_function_call_parsing.py` | 14 | protocol only |
| speech_to_text/transcription/test_transcription_inter_chunk_spacing.py | 14 | none |
| tool_parsers/test_hy_v3_tool_parser.py | 14 | mock tok |
| parser/engine/test_minimax_m2.py | 14 | none (mock tokenizer) |
| parser/engine/test_nemotron_v3.py | 14 | none (mock tokenizer) |
| tests/quantization/test_moe_wna16.py | 14 | none (in-memory tensors) |
| function_call/test_kimik3_detector.py | 14 | none (pure parsing) |
| function_call/test_kimik3_detector.py | 14 | none (pure parsing) |
| ec_connector/unit/cpu/test_ec_shared_region.py | 13 | Linux mmap; /dev/shm residency checks |
| scale_out/token_in_token_out/test_mm_serde.py | 13 | CPU torch tensors, pydantic |
| tool_parsers/test_hermes_tool_parser.py | 13 | gpt2 tok |
| tool_parsers/test_glm47_moe_tool_parser.py | 13 | HF glm47 tok |
| tool_parsers/test_lfm2_tool_parser.py | 13 | HF LFM2.5 tok |
| tool_parsers/test_poolside_v1_tool_parser.py | 13 | stub tok |
| tests/multimodal/test_embedding_shape_validation.py | 13 | – |
| cpu/test_norm.py | ~13 | test_norm.py |
| ec_connector/unit/test_scheduler_nixl_consumer.py | 12 | CPU tensors; monkeypatched nixl fields |
| worker/test_encoder_runner.py | 12 | pytestmark cpu_test |
| serve/utils/test_api_utils.py | 12 | pure functions, Namespace args |
| tool_parsers/common_tests.py | 12 | gpt2 default tok via conftest |
| tests/quantization/test_config_utils.py | 12 | none |
| sgl-model-gateway/tests/mcp_test.rs | 12 | axum/tower, mock MCP |
| unit/test_security.py | 12 | public endpoint access, static assets, api-key (incl. Anthropic header, OAI lib), CORS options/origins, proxy header forwarding |
| worker/test_gpu_worker_weight_transfer.py | 11 | recording-engine mocks; CPU LoRA layers |
| serve/exception_handling/test_error_sanitization.py | 11 | pure function (sanitize_message), CVE-2026-22778 |
| serve/exception_handling/test_http_status_metrics.py | 11 | FastAPI build_app, mocked engine |
| serve/exception_handling/test_validation_exception_handler.py | 11 | fake request, no server |
| serve/utils/test_sse_keep_alive.py | 11 | pure async generator |
| cohere/test_protocol.py | 11 | none |
| tool_parsers/test_deepseekv4_tool_parser.py | 11 | mock tok; xgrammar structural tags |
| tool_parsers/test_k2_horizon_tool_parser.py | 11 | mock tok |
| parser/engine/test_replay.py | 11 | none (mock tokenizer) |
| parser/engine/test_seed_oss.py | 11 | none (mock tokenizer) |
| renderers/test_sparse_tensor_concurrent_race.py | 11 | facebook/opt-125m tok |
| tests/tokenizers_/test_deepseek_v41.py | 11 | deepseek-ai/DeepSeek-V4.1-Flash (fixture JSON) |
| `test_reasoning_enable_thinking.py` | 10 | protocol models only |
| `parser/test_harmony_render_parity.py` | 10 | pure unit, openai types |
| `responses/test_parsable_context_unit.py` | 10 | mocks |
| tool_parsers/test_dots_tool_parser.py | 10 | mock tok |
| tests/multimodal/test_cache.py | 10 | shm, vllm config stack |
| test_checkpoint_quantization.py | 10 | cpu_ci |
| `test_stop_token_ids.py` | 9 | protocol models only |
| `responses/test_mcp_tools.py` | 9 | mock MCP server; 1 RemoteOpenAIServer |
| serve/lora/test_serving_models.py | 9 | MagicMock engine/client |
| serve/utils/test_request_logger.py | 9 | MagicMock logger |
| cohere/test_cohere_chat_message.py | 9 | none |
| tests/quantization/test_auto_awq.py | 9 | Qwen/Qwen2-1.5B-Instruct-AWQ |
| tests/utils_/test_import_utils.py | 9 | – |
| rust/src/llm/tests/generate.rs | 9 | tokio, TestTokenizer |
| cpu/test_request_headers.py | ~9 | test_request_headers.py |
| unit/test_tool_call.py | 9 | required-tool tiny/real model, no-tool-call, weather/calc/thoughts/hello-world grammars |
| ec_connector/unit/test_epd_proxy_round_robin.py | 8 | loads real examples/ proxy |
| `test_render_parity.py` | 8 | mocked OnlineRenderer / serving |
| `responses/test_sampling_params.py` | 8 | unit (torch import only) |
| tool_parsers/test_rust_tool_parser.py | 8 | rust ext build; mock tok |
| tool_use/test_chat_completion_request_validations.py | 8 | none |
| tool_use/test_gemma4_responses_adjust_request.py | 8 | stub tok |
| tool_use/test_gemma4_responses_adjust_request.py | 8 | stub tok |
| reasoning/test_k2_horizon_reasoning_parser.py | 8 | none |
| reasoning/test_nemotron_v3_reasoning_parser.py | 8 | none |
| parser/cohere/test_tool_calls.py | 8 | none |
| multimodal/generation/vlm_utils/* (8: core, runners, builders, case_filtering, custom_inputs, model_utils, types) | 8 | none (framework) |
| tests/multimodal/test_vidcom2.py | 8 | – |
| ec_connector/unit/test_metadata.py | 7 | trivial constructors |
| `chat_completion/test_logprob_token_ids.py` | 7 | Qwen2.5-1.5B server (3 tests) |
| `responses/test_streaming_events.py` | 7 | pure unit |
| pooling/test_io_processor.py | 7 | monkeypatch |
| speech_to_text/test_upload_size_limit.py | 7 | none |
| cohere/test_registry_and_args.py | 7 | none |
| tool_parsers/test_xlam_tool_parser.py | 7 | HF xlam tok |
| parser/cohere/test_reasoning.py | 7 | none |
| parser/mistral/test_reasoning.py | 7 | mistral tokenizer |
| multimodal/conftest.py + root tests (test_mapping, test_conformer_encoder, test_cohere_asr, test_mimo_v2_omni, test_nano_nemotron_vl, test_openpangu_vl) | 7 | none for mapping |
| model_executor/model_loader/runai_streamer_loader/test_runai_model_streamer_loader.py | 7 | some vllm_runner cases |
| tests/multimodal/test_parse.py | 7 | 1 subtest CUDA-skipif |
| tests/multimodal/test_sparse_tensor_validation_unit.py | 7 | – |
| test_quantization_post_load.py | 7 | — |
| cpu/test_request_decompression.py | ~7 | test_request_decompression.py |
| `test_tool_choice_content_none.py` | 6 | dummy DelegatingParser |
| `chat_completion/test_thinking_token_budget_validation.py` | 6 | protocol models only |
| unit_tests/test_non_object_body_validation.py | 6 | direct pydantic validators, 8 protocols |
| tool_parsers/test_minimax_m3_tool_parser.py | 6 | fake tok stub |
| tool_parsers/test_glm4_moe_tool_parser.py | 6 | mock tok stub |
| reasoning/test_kimi_k2_reasoning_parser.py | 6 | none |
| parser/engine/test_ling3.py | 6 | none (mock tokenizer) |
| parser/engine/test_ling3.py | 6 | none (mock tokenizer) |
| renderers/test_process_multi_modal_uuids.py | 6 | llava-hf/llava-onevision-qwen2-0.5b-ov-hf tok |
| `kernels/test_awq_int4_to_int8.py` | 6 | 0 |
| model_executor/model_loader/test_gpt_oss_weight_loading.py | 6 | mocked namespace |
| model_executor/model_loader/test_registry.py | 6 | dummy loaders |
| model_executor/test_qwen3_omni.py | 6 | mocked processing ctx |
| model_executor/test_utils.py | 6 | pure torch CPU |
| tests/quantization/test_auto_gptq.py | 6 | GPTQ models via utils |
| tests/multimodal/test_image.py | 6 | PIL, assets/ |
| manual/lang_frontend/ | 6 | test_choices.py is pure CPU; rest launch servers |
| ec_connector/unit/cpu/test_connector.py | 5 | monkeypatched make_scheduler/worker |
| ec_connector/unit/test_ec_transfer_params.py | 5 | importorskip flash-attn-built tests.v1.core.utils |
| ec_connector/unit/test_epd_proxy_retry.py | 5 | loads real examples/ proxy; aiohttp+httpx |
| ec_connector/unit/test_scheduler_nixl_ctor.py | 5 | skipif NixlWrapper absent |
| spec_decode/test_dspark_topk.py | 5 | 3 CPU-monkeypatch tests; 2 tests `skipif(not cuda)` incl. cudagraph capture |
| `test_session_id.py` | 5 | starlette Request, protocol models |
| `test_tool_calls_serialization.py` | 5 | protocol models only |
| `test_tool_calls_serialization.py` | 5 | protocol models only |
| scale_out/test_factories.py | 5 | FastAPI app, Namespace args |
| tool_use/test_muse_glimmer_parse_delta.py | 5 | HF MuseGlimmer ckpt (trust_remote_code; skips if absent) |
| parser/cohere/test_citations.py | 5 | none |
| model_executor/model_loader/test_modelexpress_loader.py | 5 | monkeypatched module |
| model_executor/test_b12x_warmup.py | 5 | mocked, SimpleNamespace |
| tests/quantization/test_gfx950_moe.py | 5 | none (stubbed configs) |
| tests/quantization/test_online_mxfp4.py | 5 | none (ModelConfig only) |
| tests/multimodal/test_utils.py | 5 | – |
| rotary/test_mrope_axis_map.py | 5 | none (CPU-registered) |
| cpu/test_server_args_backend.py | ~5 | test_server_args_backend.py |
| cpu/test_rope.py | ~5 | test_rope.py |
| ec_connector/unit/test_ec_output_aggregator.py | 4 | cpu_test mark, pure aggregation |
| ec_connector/unit/test_scheduler_nixl_producer.py | 4 | CPU tensors; monkeypatched nixl fields |
| ec_connector/unit/test_worker_ec_connector.py | 4 | cpu_test mark, patched get_ec_transfer |
| worker/test_gpu_model_runner_mm_gather.py | 4 | pytestmark cpu_test |
| `test_render_token_offsets.py` | 4 | Mock ModelConfig |
| `completion/test_lora_resolvers.py` | 4 | mocks |
| `completion/test_lora_resolvers.py` | 4 | mocks |
| `completion/test_prompt_validation.py` | 4 | gpt2 server (2 tests); torch embeds |
| serve/utils/test_fingerprint.py | 4 | SimpleNamespace config |
| unit_tests/test_offline_utils.py | 4 | pure mixin, SamplingParams |
| pooling/basic/test_tiling_engine.py | 4 | none |
| tool_parsers/test_pythonic_tool_parser.py | 4 | gpt2 tok |
| tool_parsers/test_llama4_pythonic_tool_parser.py | 4 | gpt2 tok |
| tool_parsers/test_olmo3_tool_parser.py | 4 | gpt2 tok |
| tool_parsers/test_jamba_tool_parser.py | 4 | HF jamba tok |
| tool_parsers/test_granite_tool_parser.py | 4 | gpt2 tok |
| reasoning/test_holo2_reasoning_parser.py | 4 | HCompany/Holo2-4B tok |
| reasoning/test_hy_v3_reasoning_parser.py | 4 | none |
| parser/engine/test_deepseek_v41.py | 4 | none (mock tokenizer) |
| model_executor/test_bailing_mrope.py | 4 | cpu tensors, mocks |
| model_executor/test_deep_gemm_warmup.py | 4 | mocked layers |
| model_executor/test_paddleocr_vl_mrope.py | 4 | cpu default device forced |
| rotary/test_rope_cache_invalidation.py | 4 | none (CPU-registered) |
| tests/test-gguf.cpp | 4 groups | handcrafted file, roundtrip (file/buffer/callback read modes), set-kv — property-style, randomized |
| worker/test_mrope_prompt_embeds.py | 3 | fake model; CPU |
| `test_watermarking.py` | 3 | protocol models only |
| `responses/test_errors.py` | 3 | MagicMock engine |
| pooling/scoring/test_jina_ranking_io_processor_unit.py | 3 | none |
| pooling/test_factories.py | 3 | monkeypatch |
| pooling/test_utils.py | 3 | none |
| speech_to_text/test_speech_to_text_cancellation.py | 3 | none |
| tool_parsers/test_hunyuan_a13b_tool_parser.py | 3 | mock tok |
| tool_parsers/test_gigachat3_tool_parser.py | 3 | gpt2 tok |
| tool_parsers/test_ernie45_moe_tool_parser.py | 3 | HF ernie4.5 tok (trust_remote_code) |
| reasoning/test_deepseekv3_reasoning_parser.py | 3 | deepseek-ai/DeepSeek-V3.1 tok |
| reasoning/test_gemma4_reasoning_parser.py | 3 | none |
| reasoning/test_qwen3_reasoning_parser.py | 3 | Qwen/Qwen3-0.6B, Qwen3-4B-Thinking-2507 (…, tok params) |
| parser/engine/test_ufffd_reasoning_transition.py | 3 | none (mock tokenizer) |
| model_executor/model_loader/runai_streamer_loader/test_runai_utils.py | 3 | pure logic |
| model_executor/model_loader/test_weight_tying.py | 3 | cpu_test mark |
| model_executor/test_cpu_unquantized_gemm_dispatch.py | 3 | monkeypatched zentorch |
| model_executor/test_ernie45_vl_mrope.py | 3 | cpu default device forced |
| model_executor/test_flashinfer_autotune_warmup.py | 3 | cpu_test, mocked MoE runner |
| model_executor/test_keye_mrope.py | 3 | cpu default device forced |
| model_executor/test_keye_vl1_5_mrope.py | 3 | cpu default device forced |
| model_executor/test_minicpmv.py | 3 | object.__new__ trick, CPU |
| tests/quantization/test_int8_moe_oracle.py | 3 | none (stubbed configs) |
| tests/multimodal/test_registry.py | 3 | mocked + model context |
| cpu/test_qwen3.py | ~3 | test_qwen3.py |
| manual/test_{schedule_policy,weight_validation,config_integration}.py | 3 | none (in-proc, tempfile) |
| manual/test_{schedule_policy,weight_validation,config_integration}.py | 3 | none (in-proc, tempfile) |
| unit/test_compat_oai_responses.py | 3 | responses endpoint mapping |
| `chat_completion/test_non_object_body_validation.py` | 2 | pydantic ValidationError only |
| `responses/test_protocol.py` | 2 | pure unit |
| serve/middleware/test_authentication_middleware.py | 2 | FastAPI TestClient, no server |
| tool_parsers/test_internlm2_tool_parser.py | 2 | gpt2 tok |
| tool_parsers/test_kimi_k3_named_tool_choice.py | 2 | none obvious (registry-level) |
| tool_parsers/test_deepseekv31_tool_parser.py | 2 | gpt2/mock tok |
| reasoning/test_glm4_moe_reasoning_parser.py | 2 | zai-org/GLM-4.7 tok |
| reasoning/test_granite_reasoning_parser.py | 2 | facebook/opt-125m tok |
| parser/engine/test_delegating_replay.py | 2 | none (mock tokenizer) |
| `kernels/test_onednn.py` | 2 | CPU, oneDNN |
| model_executor/model_loader/runai_streamer_loader/test_weight_utils.py | 2 | temp safetensors files |
| model_executor/model_loader/test_filter_duplicate_safetensors.py | 2 | temp files |
| model_executor/test_nemotron_h_quantization.py | 2 | fully mocked |
| model_executor/test_qwen3_5_quantization.py | 2 | fully mocked |
| tests/quantization/test_humming_ignore.py | 2 | none (real Kimi-K2.6 config JSON inline) |
| tests/quantization/test_register_quantization_config.py | 2 | meta-llama/Llama-3.2-1B-Instruct |
| tests/utils_/test_async_utils.py | 2 | – |
| tests/multimodal/test_inputs.py | 2 | – |
| manual/test_{aiter_unified_draft_extend_env,dsa_alias_cli_registry_env}.py | 2 | none (aiter test is flag-only, ROCm kernel not exercised) |
| test/layer_ut_utils.py + quant_ref_utils.py | 2 | torch only |
| tests/test-gguf-model-data.cpp | 2 | glm4moe/step35 key composition |
| tests/test-quantize-fns.cpp | 2 | vec_dot f32 vs q-types vs reference error, per-type quantize/dequant error budget (CPU vs backend) |
| tests/test-quantize-stats.cpp | 2 | roundtrip error on chunk/layer |
| test_gguf_reader_validation.py | 2 | n_dims upper bound (evil 1e6-dim file rejected, no EOF overread); dims-product uint64 wraparound rejected |
| pooling/scoring/test_late_interaction_serving.py | 1 | none |
| pooling/scoring/test_io_processor_unit.py | 1 | none |
| pooling/scoring/test_utils.py | 1 | none |
| speech_to_text/transcription/test_chunk_timestamp_offset.py | 1 | none |
| tool_parsers/test_granite4_tool_parser.py | 1 | HF granite-4 tok |
| reasoning/test_deepseekr1_reasoning_parser.py | 1 | DeepSeek-R1-Distill-Qwen-1.5B tok |
| reasoning/test_ernie45_reasoning_parser.py | 1 | baidu/ERNIE-4.5-21B-A3B-Thinking tok |
| reasoning/test_gptoss_reasoning_parser.py | 1 | none |
| reasoning/test_hunyuan_reasoning_parser.py | 1 | tencent/Hunyuan-A13B-Instruct tok (trust_remote_code) |
| reasoning/test_minimax_m2_append_reasoning_parser.py | 1 | MiniMaxAI/MiniMax-M2 tok |
| reasoning/test_minimax_m2_reasoning_parser.py | 1 | MiniMaxAI/MiniMax-M2 tok |
| reasoning/test_olmo3_reasoning_parser.py | 1 | allenai/Olmo-3-7B-Think tok |
| reasoning/test_step3p5_reasoning_parser.py | 1 | stepfun-ai/Step-3.5-Flash tok |
| renderers/test_multimodal_hashes.py | 1 | llava-hf/llava-onevision-qwen2-0.5b-ov-hf tok |
| registry.py | 1 | HF transformers ver |
| utils.py | 1 | none |
| test_registry.py | 1 | none |
| test_utils.py | 1 | torch CPU |
| test_adapters.py | 1 | torch CPU |
| test_language_model_cache_is_weak.py | 1 | none |
| qwen4_exp/test_config.py | 1 | none |
| transformers/test_layer_registry.py | 1 | none |
| model_executor/model_loader/fastsafetensors_loader/test_weight_utils.py | 1 | temp files, mocked hf |
| model_executor/model_loader/instanttensor_loader/test_weight_utils.py | 1 | mocked files |
| model_executor/model_loader/runai_streamer_loader/test_runai_model_streamer_s3.py | 1 | StreamerPatcher |
| model_executor/model_loader/test_checkpoint_weight_patch.py | 1 | cpu_test mark |
| model_executor/offloader/test_prefetch.py | 1 | pure index math |
| model_executor/test_qwen3_asr_mrope.py | 1 | pure position math |
| model_executor/test_qwen3_vl_mrope.py | 1 | cpu default device forced |
| tests/quantization/test_configs.py | 1 | none (config only) |
| tests/quantization/test_cpu_w8a8.py | 1 | RedHatAI/Qwen3-30B-A3B-...-w8a8 |
| tests/quantization/test_cpu_wna16.py | 1 | openai/gpt-oss-20b, Qwen GPTQ/FP8 pins |
| tests/quantization/test_gptq_dynamic.py | 1 | ModelCloud dynamic-cfg repo (config-only) |
| manual/test_deepseek_chat_templates.py | 1 | none |
| test/cpu_test_utils.py | 1 | none |
| test/config_publishers.py | 1 | none |
| rust/sglang-mm/tests/rlib_is_single_threaded.rs | 1 | cargo test harness |
| tool_parsers/conftest.py | 0 | gpt2 tok (HF download/cache) |
| tool_parsers/__init__.py | 0 | — |
| tool_parsers/utils.py | 0 | gpt2 tok |
| `kernels/quant_utils.py` (helper module, 0 tests) | 0 | — |
| rust/src/engine-core-client/src/tests/python_compat.py | – | msgspec, msgpack |
| test_quants.py | 0 test fns | ctypes ggml quant enum mapping table |

## C2 — Constrained / structured output (50 rows)

| upstream path | n | pins |
|---|---|---|
| tests/test-json-schema-to-grammar.cpp | ~170 | ~170 schema→grammar goldens (min/max, patterns, required, nesting) |
| tests/peg-parser/test-basic.cpp | 1 fn (~40 cases) | literal/seq/choice/rep/composite parsing |
| tests/peg-parser/test-gbnf-generation.cpp | 1 fn (~40 cases) | grammar generation from PEG defs, gbnf snapshots |
| tests/peg-parser/test-python-dict-parser.cpp | 1 fn (~30 cases) | dict-literal parsing (tool args) |
| tests/peg-parser/test-unicode.cpp | 1 fn (~30 cases) | unicode char/range/escapes in PEG |
| tests/peg-parser/test-json-parser.cpp | 1 fn (~15 cases) | JSON parse/serialize roundtrips, errors |
| parser/engine/test_inkling.py | 64 | none (mock tokenizer) |
| model_executor/layers/test_pooler_heads.py | 51 | pure torch CPU |
| tests/test-grammar-parser.cpp | 50 | rule/grammar string → parsed rule counts (root_1, expr_5, …) |
| tool_parsers/test_deepseekv32_tool_parser.py | 48 | mock tok; xgrammar structural tags |
| constrained/test_grammar_manager.py | 46 | cpu_ci |
| model_executor/layers/test_pooler_methods.py | 37 | transformers config only |
| constrained/test_base_grammar_backend.py | 33 | cpu_ci |
| logits_processors/test_correctness.py | 30 | device_type |
| tests/test-gbnf-validator.cpp | ~30 | grammar strings accepted/rejected (one case per line) |
| tool_parsers/test_structural_tag_registry.py | 29 | xgrammar lib; mock/gpt2 toks |
| constrained/test_reasoner_grammar_backend.py | 27 | cpu_ci |
| renderers/test_inkling.py | 26 | none |
| tests/tokenizers_/test_mistral.py | 26 | mistralai repos (tokenizer only) |
| model_executor/layers/test_pooler_activations.py | 25 | pure torch CPU |
| unit/test_chat_completion.py | 24 | chat completion (stream/non-stream, OpenAI lib), chat template + assistant prefill, continue-final-message (vllm compat, mutual exclusion), apply_chat_template, response_format JSON-schema, grammar, invalid grammar, logprobs (stream), logit_bias, context-size-exceeded, timings-per-token, n>1 choices, token counts, cached tokens |
| tool_parsers/test_qwen3coder_tool_parser.py | 23 | HF qwen3 tok; xgrammar |
| structured_output/test_reasoning_structured_output.py | 19 | mock reasoner, StructuredOutputManager |
| tests/test-json-schema.cpp | 14 | primitives, string/integer, array/tuple, object, any/anyOf/allOf, const/enum, $ref, may-be-string, value types, errors |
| spec_decode/test_mtp_structured_output.py | 13 | parametrize backend xgrammar/guidance; StructuredOutputManager fakes |
| tests/test-llama-grammar.cpp | 13 | stack-based accept/reject cases (expr/ident/num/root…) |
| tests/test-grammar-integration.cpp | ~12 | simple/complex grammar, quantifiers, special chars, JSON-schema via grammar, left-recursion/missing-ref/missing-root failures, custom root |
| tool_parsers/test_deepseekv4_tool_parser.py | 11 | mock tok; xgrammar structural tags |
| structured_output/test_outlines_cache.py | 10 | `pytestmark = cpu_test` |
| constrained/test_token_filter_ops.py | 9 | cpu_ci |
| constrained/test_utils.py | 9 | cpu_ci |
| unit/test_tool_call.py | 9 | required-tool tiny/real model, no-tool-call, weather/calc/thoughts/hello-world grammars |
| `responses/test_sampling_params.py` | 8 | unit (torch import only) |
| constrained/test_e2e_constrained_reasoning.py | 7 | cpu_ci |
| tests/test-grammar-llguidance.cpp | 7 | same grammar/schema corpus forced through llguidance sampler chain |
| structured_output/test_validation.py | 6 | `pytestmark = cpu_test` |
| tests/test-chat-peg-parser.cpp | 6 | native/qwen3-coder/qwen3 templates, prefix tool names, tagged parser, permutation |
| tests/test-chat-peg-parser.cpp | 6 | native/qwen3-coder/qwen3 templates, prefix tool names, tagged parser, permutation |
| structured_output/test_regex_compilation_timeout.py | 5 | pure |
| structured_output/test_backend_guidance.py | 4 | guidance + tokenizer deps |
| structured_output/test_scheduler_speculative_padding.py | 4 | pure |
| structured_output/test_utils.py | 4 | `pytestmark = cpu_test`, xgrammar |
| constrained/test_llguidance_batched_mask.py | 4 | llguidance lib |
| constrained/test_mistral_common_xgrammar.py | 4 | xgrammar |
| structured_output/test_guidance_negative_draft_tokens.py | 3 | pure (fake tokenizer/matcher) |
| model_executor/test_gemma_hidden_act.py | 3 | pure fn checks |
| tool_parsers/test_kimi_k3_named_tool_choice.py | 2 | none obvious (registry-level) |
| structured_output/test_backend_xgrammar_stop_tokens.py | 1 | xgrammar |
| tests/test-peg-parser.cpp | 1 | wrapper entry |
| tests/peg-parser/test-json-serialization.cpp | 1 fn | serialization format |

## C3 — Sampler semantics (31 rows)

| upstream path | n | pins |
|---|---|---|
| tests/test-sampling.cpp | ~13 fns (~60 calls) | dist singleton RNG, temp/temp-ext, top-k, top-p, min-p, xtc, typical, penalties, dry, top-n-sigma, sampler-queue sequence handling, perf |
| layers/ — top-level logic (attn_residual, dsv4_kv_splits_heuristic, dsv4_nonpaged_indexer, flashinfer_comm_fusion, fp8_bpreshuffle_scale, gdn_mis_metadata, kda_decode_mtp_slot_stride, layer_communicator_fusion_gate, layer_scatter_modes, logprob_chunk_stitching, logprob_fast_input, mamba2_track_ssm_indices, minicpm_attention_adapter, minicpm_sparse_{cache,metadata}, moriep_mxfp8_dispatch, mova, pooler_score_and_pool, radix_attention, radix_linear_attention) | 159 | mostly cpu_ci |
| sampling/test_sampling_params.py | 76 | cpu_ci |
| sampling/test_sampling_batch_info.py | 46 | cpu_ci, CPU per docstring |
| sampling/test_penaltylib.py | 41 | cpu_ci |
| sampling/test_custom_logit_processor.py | 36 | cpu_ci |
| logits_processors/test_correctness.py | 30 | device_type |
| sample/test_topk_topp_sampler.py | 26 | large_gpu subset |
| unit/test_completion.py | 24 | stream parity, seed determinism, batch-size invariance, cache vs nocache, tokens input, parallel slots, unified endpoint, n_probs (incl. post-backend-sampling), logit_bias, cancel request, prompt cache |
| sample/test_rejection_sampler.py | 20 | device_type |
| tests/watermarking/test_gumbel.py | 12 | none (torch; imports gpu sampler module) |
| sample/test_trace_replay_params.py | 11 | — |
| model_executor/test_watermark_sample_warmup.py | 11 | mocked watermarker |
| beam_search/test_beam_search_core.py | 10 | cpu_ci |
| beam_search/test_fork.py | 9 | cpu_ci |
| sample/test_head_dtype.py | 8 | 1 cuda + 1 e2e LLM (core_model) |
| `responses/test_sampling_params.py` | 8 | unit (torch import only) |
| `chat_completion/test_logprob_token_ids.py` | 7 | Qwen2.5-1.5B server (3 tests) |
| batch_invariant_ops/test_batch_invariant_ops.py | 7 | cpu_ci |
| manual/lang_frontend/ | 6 | test_choices.py is pure CPU; rest launch servers |
| sample/test_sampler.py | 5 | device_type |
| spec_decode/test_synthetic_rejection_sampler_utils.py | 5 | pure |
| tests/samplers/test_non_finite_params.py | 4 | none |
| beam_search/test_output_decode.py | 4 | cpu_ci |
| tests/v1/engine/test_parallel_sampling.py | 3 | pure unit |
| tests/samplers/test_beam_search_online.py | 3 | none (mocked) |
| tests/v1/engine/test_logprobs_processor.py | 2 | numpy unit |
| sample/test_batched_count_greater_than.py | 2 | device_type |
| sample/test_thinking_budget_state.py | 2 | cpu-only |
| spec_decode/test_llm_base_proposer_sampling.py | 2 | `current_platform`, `set_random_seed` |
| sampling/test_deterministic_gumbel_u1.py | 2 | `.cuda()` (trivial to de-GPU) |

## C4 — Tokenizer / detok / chat templates (70 rows)

| upstream path | n | pins |
|---|---|---|
| utils/ — generic (common, auth, http_server_auth, field_validators, invariants, json_response, patch_tokenizer, profile_merger, subprocess_watchdog, tensor_bridge, weight_checker{,_comparator}, weight_versions, gauge_histogram, diffusion_torch_fallback) | 251 | cpu_ci |
| tests/test-chat.cpp | ~200 | template application vs expected outputs (205 embedded cases), oaicompat msg/tools JSON conversion, delimiters split, responses→chatcmpl, lfm2 parser, generation prompt, PEG parser compare |
| parser/test_reasoning_parser.py | 134 | cpu_ci |
| renderers/test_cohere.py | 91 | none |
| unit_tests/test_chat_utils.py | 81 | HF tokenizers, image/audio/video assets |
| parser/test_conversation.py | 76 | cpu_ci |
| tests/test-chat-auto-parser.cpp | ~60 | diff-split (prefix/suffix/overlap/tag boundaries), variant comparison, tool formats (nemotron/laguna/cohere/smollm3/seed-oss), reasoning detection, quote normalization |
| parser/test_template_manager.py | 48 | cpu_ci |
| tests/test-chat-analysis.cpp | ~46 | tool-call analysis (full/pure-JSON/fn-name markers, edge cases), reasoning-format detection (nemotron/cohere/laguna/smollm3 variants), marker separation |
| managers/ — tokenizer manager (multi_tokenizer_mixin, tokenizer_config_updates, tokenizer_manager_rid_cleanup) | 44 | cpu_ci |
| parser/test_harmony_parser.py | 43 | cpu_ci |
| tests/watermarking/test_watermarking.py | 42 | none (SimpleNamespace/numpy) |
| renderers/test_hf.py | 37 | Qwen/Qwen2-VL-2B-Instruct, NousResearch/Hermes-3-Llama-3.1-8B tok |
| scale_out/derender/test_derender.py | 35 | RemoteLaunchRenderServer (GPU-less) |
| scale_out/derender/test_derender_stream.py | 28 | tokenizer-only unit layer + render server |
| renderers/test_completions.py | 28 | openai-community/gpt2 tok |
| renderers/test_inkling.py | 26 | none |
| tests/tokenizers_/test_mistral.py | 26 | mistralai repos (tokenizer only) |
| parser/test_jinja_template_utils.py | 26 | cpu_ci |
| renderers/test_warmup.py | 24 | none |
| tests/test-jinja.cpp | 22 | literals, expressions, conditionals, loops, filters, string/array/object methods, macros, namespace, set, tests, whitespace control, comments, hasher, fuzzing, stats, template_cpp vs template_py parity |
| tests/tokenizers_/test_deepseek_v4.py | 20 | none (fake tokenizer) |
| renderers/test_gemma4_chat_template.py | 16 | none (jinja only, no vllm import) |
| renderers/test_kimi_k3.py | 16 | none |
| tests/test-tokenizer-0.cpp | 16 registered | get_vocab/add_special/eol/assertion matrix over 16 ggml-vocab models (bert-bge…starcoder); drives test-tokenizer-0-* ctest names |
| renderers/test_chat_utils_prompt_embeds.py | 14 | AutoTokenizer from request.params |
| tests/transformers_utils/test_config.py | 13 | meta-llama pin (metadata only) |
| parser/test_code_completion_parser.py | 13 | cpu_ci |
| renderers/test_token_offsets.py | 12 | openai-community/gpt2 tok |
| tests/watermarking/test_gumbel.py | 12 | none (torch; imports gpu sampler module) |
| tokenizer/test_tiktoken_tokenizer.py | 12 | cpu_ci |
| tests/v1/engine/test_output_processor.py | 11 | dummy test vectors |
| tests/tokenizers_/test_deepseek_v41.py | 11 | deepseek-ai/DeepSeek-V4.1-Flash (fixture JSON) |
| `parser/test_harmony_render_parity.py` | 10 | pure unit, openai types |
| renderers/inputs/test_preprocess.py | 9 | none |
| `test_render_parity.py` | 8 | mocked OnlineRenderer / serving |
| serve/tokenize/test_serving_tokenization.py | 8 | TestClient + AsyncMock renderer |
| tests/transformers_utils/test_muse_glimmer_config.py | 7 | none |
| tests/tokenizers_/test_detokenize.py | 7 | bloom/gpt-j/pythia/opt/llama tokenizer repos |
| tests/transformers_utils/test_bailing_moe_v3_vl_config.py | 6 | none (AutoConfig JSON) |
| tests/transformers_utils/test_speculators_override.py | 6 | none |
| tests/tokenizers_/test_registry.py | 6 | opt-125m, Mistral-Nemo tokenizer repos |
| tests/detokenizer/test_check_stop_strings.py | 6 | none |
| tests/test-chat-peg-parser.cpp | 6 | native/qwen3-coder/qwen3 templates, prefix tool names, tagged parser, permutation |
| tests/test-chat-template.cpp | 6 | template render + format extraction for key models |
| tests/test-reasoning-budget.cpp | 6 | budget counting, clone mid-counting/mid-forcing, end-match, force-manual, UTF-8 boundary detection |
| tests/transformers_utils/test_repo_utils.py | 5 | none (hub mocks) |
| tests/transformers_utils/test_dspark_mla_config.py | 4 | none (inline config) |
| tests/transformers_utils/test_utils.py | 4 | none |
| spec_decode/test_vocab_mapping.py | 3 | HF tokenizer downloads (meta-llama/Qwen) |
| tests/transformers_utils/test_speculators_dspark_config.py | 3 | none |
| tests/tokenizers_/test_basic.py | 3 | opt-125m / Mistral / DeepSeek-V3 tokenizer repos |
| tests/watermarking/test_prf.py | 3 | none (torch CPU) |
| manual/test_{async_dynamic_batch_tokenizer,tokenizer_batch_encode,tokenizer_manager}.py | 3 | none |
| tests/v1/engine/test_logprobs_processor.py | 2 | numpy unit |
| renderers/test_mistral.py | 2 | mistralai/Mistral-7B-Instruct-v0.3 tok |
| tests/transformers_utils/test_config_parser_registry.py | 2 | none |
| tests/transformers_utils/test_processor.py | 2 | none |
| tests/tokenizers_/test_hf.py | 2 | tokenizer repos (small) |
| tests/watermarking/test_detection.py | 2 | none |
| tokenizer/test_tekken_tokenizer_routing.py | 2 | cpu_ci |
| tests/v1/engine/test_fast_incdec_prefix_err.py | 1 | AutoTokenizer only |
| tests/transformers_utils/test_hf_overrides_model_type.py | 1 | none (tempfiles) |
| tests/detokenizer/test_min_tokens.py | 1 | opt-125m tokenizer repo |
| tests/detokenizer/test_stop_string_while_stop_model_terminates.py | 1 | none (fake detok) |
| parser/test_reasoning_content_without_parser.py | 1 | cpu_ci |
| experimental/sgl-router/tests/component/tokenizer/parity.rs | 1 | HF tokenizer fixture |
| tests/test-tokenizer-1-bpe.cpp | 1 (mostly commented) | tokenize/untokenize vs reference incl. --ignore-merges mode |
| tests/test-tokenizer-1-spm.cpp | 1 | llama-spm roundtrip, pre-tokenization splits |
| tests/test-tokenizer-0.py + .sh, test-tokenizer-random.py, test-tokenizers-repo.sh | – | full HF tokenizers repo diff vs llama.cpp (py driver) |

## C5 — KV cache / prefix reuse / hostcache (135 rows)

| upstream path | n | pins |
|---|---|---|
| mem_cache/ — unified pools (unified_{byte_accounting,byte_budget_sizing,cache_linker,capacity_memo,free_no_host_sync,handout_zeroing,mamba_views,mha_views,mla_block_table,mla_gpu_parity,mla_views,npool_sweep,radix_allocation_eviction,radix_cache_unittest,radix_cache_bench,radix_hicache_dispatch,radix_lock_ref,tri_pool} + inspectors) | 430 | 84/105 dir cpu_ci |
| rust/sglang-radix-tree/src/tests/unified_tree_core.rs | 341 | tch crate (libtorch CPU) via crate dep |
| mem_cache/ — paged allocators + eviction (paged_allocator_lazy_release, paged_free_segment, free_kv_row_coalesce, evict_policy, multi_ended_allocator, page_{interleave_shard,major_layout}, full_loc_fast_path, hisparse_allocator, hisparse_max_token_pool_size) | 276 | cpu_ci |
| rust/sglang-radix-tree/src/tests/components/swa.rs | 186 | tch |
| layers/ — top-level logic (attn_residual, dsv4_kv_splits_heuristic, dsv4_nonpaged_indexer, flashinfer_comm_fusion, fp8_bpreshuffle_scale, gdn_mis_metadata, kda_decode_mtp_slot_stride, layer_communicator_fusion_gate, layer_scatter_modes, logprob_chunk_stitching, logprob_fast_input, mamba2_track_ssm_indices, minicpm_attention_adapter, minicpm_sparse_{cache,metadata}, moriep_mxfp8_dispatch, mova, pooler_score_and_pool, radix_attention, radix_linear_attention) | 159 | mostly cpu_ci |
| `kernels/helion/` | 128 | helion pkg, CUDA |
| rust/sglang-radix-tree/src/tests/node.rs | 124 | tch (libtorch CPU) |
| mem_cache/ — rust tree core (rust_tree_core{,_integration}, rust_unified_radix_cache_{unittest,bench}, tree_core_registry, rust_unified_tree_core_inspector) | 118 | cargo build |
| mem_cache/ — host allocators (asymmetric_mha_pool_host, dsa_pool_host, mla_host_dedup_primitives, mem_pool_host, minimax_sparse_pool_host, umbp_host_allocator, mmap_allocator, buffer_mode_sidecar) | 114 | cpu_ci |
| layers/attention/ — logic (dsa_{head_gate_guard,mqa_logits_chunking}, encoder_decoder_varlen_gather, flashattention_{graph_metadata,pa_swa_prefill_lens_size}, gdn_{flashinfer_alignment,prefill_backend_policy}, index_topk_share, kv_translate_ownership, linear_attn_config, mamba_track_state_dtype, mla_decode_{forced_splits,geometry}, verify_mask, vision_{backend_selection,max_seqlen,seqlens,strided_qkv}) | 113 | 20/25 cpu_ci |
| tests/v1/core/test_prefix_caching.py | 109 | kv-events, offloading/hisparse connector mocks |
| mem_cache/ — hicache (hicache_{dcp_host_pool,file_lru_unit,host_register,load_back_timing,nixl_cleaner,nixl_storage,staged_write_back_dispatch,dispatch…} + dsv4_hicache_l2, unified_radix_hicache_dispatch) | 106 | mostly cpu_ci |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py | 102 | scheduler-side store/load policy: grouping, mamba, events |
| kv_offload/tiering/p2p/test_sessions.py | 101 | fake control connections only |
| tests/v1/core/test_kv_cache_utils.py | 98 | some subprocess/mmap, HiSparseConfig, LoRA |
| rust/sglang-radix-tree/src/tests/components/full.rs | 98 | tch |
| kv_canary/test_self_unit_{capacities,e2e_base,plan_input,pool_patcher_utils,radix_walker,sweep_plan_builder,token_oracle,violation,pool_patcher,req_to_expected_token_ids_manager}.py (10 files) | 82 | torch CPU tensors, mocks; no `.cuda()` |
| mem_cache/ — swa pools (swa_alloc_extend_page_estimation, swa_cpu_copy_filter, swa_eviction_boundary, swa_locked_full_recover_unified, swa_lock_release_lifecycle, swa_pool_v_head_dim, swa_ring_page_return, swa_unittest, unified_swa_shared_virtual_ids) | 80 | mixed |
| `attention/` — CPU/MLA/model-specific: test_cpu_attn.py, test_mla_decode_cpu.py, test_mla_cross_layer_kernel_equivalence.py, test_kimi_k3_mla_fused_epilogue.py, test_kimi_k3_mla_key_concat_kv_cache.py, test_minimax_m3.py, test_minimax_m3_msa_cutlass_sparse_decode.py, test_xpu_mla_sparse.py | 77 | CPU for 2 files; CUDA/ROCm/XPU for rest |
| model_executor/ — top-level logic (chunked_prefix_cache_gate, draft_runner_skips_lora, forward_metadata_plan_record, hisparse_pool_configurator, kv_canary_headroom, mlp_sync_pad_unpad, model_runner_decode_rows, num_token_non_padded_localization, pool_configurator, unified_out_cache_loc_rebind) | 76 | cpu_ci |
| kv_canary/test_self_unit_{buffer_alloc,endpoint,future_tensor,perturb,runner_health,runner_per_forward,runner_swa_divergence,runner_sweep}.py (8 files) | 75 | torch CPU tensors; GPU-registered but no `.cuda()` |
| kv_offload/tiering/p2p/test_manager.py | 65 | fake transports/sessions, no real net |
| rust/sglang-radix-tree/src/tests/components/mamba.rs | 63 | tch |
| mem_cache/ — radix core (radix_cache_unit, radix_cache_cpp_unit, radix_cache_slru_accuracy, radix_force_miss, hiradix_cache_unit, pure_swa_radix_cache, pure_swa_chunk_cache, decode_radix_lock_ref) | 51 | cpu-runnable, some cuda_ci |
| rust/sglang-radix-tree/src/tests/unified_lru_list.rs | 51 | none direct (crate tch) |
| tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py | 46 | dcp_world_size 1/2/4, HMA connector mock |
| tests/quantization/test_turboquant.py | 46 | none (torch) |
| kv_offload/cpu/test_shared_offload_region.py | 43 | Linux /dev/shm; threads+multiproc, no GPU |
| kv_offload/tiering/test_tiering_offloading.py | 42 | mock mmap region, CPU tensors |
| vllm/tests/v1/simple_kv_offload/test_scheduler.py | 41 | CPU offload scheduler: lookup, alloc, free, pin accounting |
| worker/test_utils.py | 39 | mocks; one monkeypatched is_cuda_alike |
| mem_cache/ — misc (mem_cache_utils, registry, kv_index_translator, layout_compat, linker_pool_assembler, hybrid_pool_assembler, session_token_share_unit, session_unified_radix_cache, streaming_session_unit, decode_retraction_backup, dllm_fdfo_kv_reuse, pd_envelope_transfer_layout) | 39 | cpu_ci |
| kv_offload/tiering/test_obj_tier.py | 35 | nixl module import; agent fully mocked |
| kv_offload/test_factory.py | 29 | pure config/registry logic, mocked ctors |
| kv_offload/tiering/test_fs_tier.py | 29 | real disk I/O; C-ext/O_DIRECT optional fallback |
| cpu/ small torch-unit group | ~29 | test_store_cache.py, test_comm.py, test_binding.py |
| vllm/tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py | 28 | P pulls KV from D; remote_block_ids lifecycle, partial coverage |
| kv_offload/cpu/test_manager.py | 28 | zero GPU touch; pure policy state machine |
| cpu/test_subblock_sparse_attention.py | ~28 | test_subblock_sparse_attention.py |
| kv_offload/tiering/p2p/test_data_transport.py | 25 | nixl agent mocked or patched to None |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_config.py | 22 | KV-cache-spec → offload config translation (mamba/hybrid/MLA) |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_metrics.py | 22 | offload prometheus metric defs, aggregation, stats |
| attention/test_mla_prefill_selector.py | 21 | — |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_events.py | 19 | BlockStored/BlockRemoved event tracking per group |
| tests/v1/core/test_contiguous_kv_packing.py | 19 | allocate_kv_cache on torch cpu device, LBNHC/BLHNC |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py | 18 | canonical page mapping across TP/DCP/MLA layouts |
| kv_offload/tiering/test_async_lookup.py | 17 | in-memory backend, threads only |
| tests/v1/core/test_single_type_kv_cache_manager.py | 17 | block_size 3584/4608, in-flight chunks |
| rust/ | ~17 | test_rust_extension.py(15) |
| vllm/tests/v1/kv_connector/unit/test_flexkv_connector.py | 16 | delegation + ImportError when flexkv missing (mocked) |
| tests/v1/core/test_kv_cache_metrics.py | 16 | pure unit, patch |
| `kernels/test_deepseek_v4_cpu_kernels.py` | 16 | CPU |
| `kernels/test_deepseek_v4_cpu_kernels.py` | 16 | 1 |
| tests/quantization/test_per_token_kv_cache.py | 15 | none (MagicMock) |
| kv_offload/test_file_mapper.py | 14 | pure path/namespace logic, no I/O |
| `kernels/ir/` (excl. test_activation.py): test_ir_ops.py, test_layernorm.py | 14 | kernel backend |
| experimental/sgl-router/sgl-kv-indexer/tests/memory_integration.rs | 14 | tonic types only |
| vllm/tests/v1/simple_kv_offload/test_kv_events.py | 13 | BlockStored/Removed medium + per-group metadata emission |
| vllm/tests/v1/simple_kv_offload/test_worker.py | 12 | GPU→CPU store cross-stream sync ordering (no stale reads) |
| tests/v1/metrics/test_stats.py | 12 | pure unit |
| worker/test_attn_utils.py | 12 | none (no markers; CPU tensors) |
| distributed/test_kv_cache_events.py | 12 | cpu device, config mocks |
| kv_offload/tiering/test_factory.py | 11 | MagicMock args only |
| vllm/tests/v1/kv_connector/unit/test_hidden_states_connector.py | 10 | hidden-state KV-cache-group resolution (hybrid specs) |
| vllm/tests/v1/kv_connector/unit/test_tp_mapping.py | 10 | TP mapping: source ranks, split handles, desc IDs, no GPU |
| tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py | 10 | dcp_world_size 1/2/4, kv-events |
| attention/test_kv_head_stride_canonicalization.py | 10 | — |
| tests/multimodal/test_cache.py | 10 | shm, vllm config stack |
| chunked_prefill/test_mm_chunked_embedding_unit.py | 10 | none (CPU-registered) |
| vllm/tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py | 9 | recovery paths when async KV load fails (recompute/resume) |
| vllm/tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py | 9 | remote-prefill request lifecycle, finish/retry paths |
| tests/v1/core/test_mamba_align_chunk_split.py | 9 | hybrid full+mamba specs, partial hits |
| attention/test_kpool_tail_slot_mapping.py | 9 | — |
| attention/test_mla_prefill_registry.py | 9 | — |
| tests/utils_/test_torch_utils.py | 9 | torch CPU |
| experimental/sgl-router/sgl-kv-indexer/tests/grpc_contract.rs | 9 | tonic, in-process server |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_worker.py | 8 | worker offload ops, load/store spec selection per backend |
| kv_offload/cpu/policies/test_factory.py | 8 | registry resolution only |
| kv_offload/cpu/test_gpu_worker.py | 8 | 5 pure-mock lifecycle tests; 3 need CUDA |
| vllm/tests/v1/kv_connector/unit/test_config.py | 7 | KVTransferConfig → connector factory translation |
| vllm/tests/v1/kv_connector/unit/test_nixl_heartbeat.py | 7 | scheduler-driven heartbeat / KV lease renewal semantics |
| tests/v1/core/test_swa_inflight_window_free.py | 7 | ChunkedLocalAttentionSpec, SlidingWindowSpec |
| attention/test_mla_prefill_quant_output.py | 7 | — |
| attention/test_sparse_indexer_decode_seq_lens.py | 7 | — |
| mem_cache/test_int8_checkpoint_store.py | 7 | 1 CUDA-gated test skipped on CPU |
| tests/test-save-load-state.cpp | 7 | num/seq_cp device+host+scatter/seq_rm_isolated/state_load/state_roundtrip (needs generated models) |
| kv_offload/cpu/test_canonical_layout.py | 6 | 4 pure-numpy CPU tests; 2 CUDA skipif |
| kv_offload/tiering/test_metrics.py | 6 | pure tracker logic |
| tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py | 6 | eagle_group, num_prefill_lookahead params |
| worker/test_gpu_model_runner_v2.py | 6 | runner via __new__; CPU |
| attention/test_mla_noncausal.py | 6 | cpu-device |
| model_executor/layers/test_mla_short_prefill_indexer.py | 6 | mocked metadata |
| experimental/sgl-router/tests/proxy/shared_prefill_admission.rs | 6 | axum/tower |
| vllm/tests/v1/simple_kv_offload/test_hip_mem_ops.py | 5 | HIP runtime version gating, memcpy attrs (monkeypatched) |
| tests/v1/engine/test_input_processor_trace_replay.py | 5 | mocked VllmConfig |
| attention/test_cpu_mla_backend.py | 5 | cpu-only |
| test_cache_hit_kit_metrics.py | 5 | cpu_ci |
| cpu/test_extend.py | ~5 | test_extend.py |
| vllm/tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py | 4 | worker metadata aggregation across jobs/workers |
| vllm/tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py | 4 | remote-decode request lifecycle via example connector |
| attention/test_indexer_native_next_n.py | 4 | — |
| distributed/test_kvlayout.py | 4 | DeviceConfig("cpu") |
| rust/sglang-radix-tree/src/tests/components/base.rs | 4 | tch |
| experimental/sgl-router/tests/proxy/cache_aware_input_ids.rs | 4 | axum/tower |
| tests/test-batch-alloc.cpp | 4 | init, keep-tail, split, mrope batch layouts |
| vllm/tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py | 3 | invalid-block recompute/free semantics, no caching after fail |
| vllm/tests/v1/kv_connector/unit/test_nixl_simple_cpu_offload.py | 3 | Nixl + SimpleCPUOffload delegation, HMA detection, metadata agg |
| vllm/tests/v1/kv_connector/unit/test_scheduler_kv_connector_override.py | 3 | plugin/factory override of scheduler connector instance |
| tests/v1/core/test_kv_sharing.py | 3 | torch cpu tensors |
| worker/test_kv_cache_allocation_scope.py | 3 | mock scope/context managers |
| attention/test_group_head_counts.py | 3 | cpu-only |
| spec_decode/test_dflash_lookahead.py | 3 | Scheduler/KV-cache fakes, StructuredOutputManager |
| vllm/tests/v1/kv_connector/unit/test_error_propagation.py | 2 | connector load failure propagates to request abort |
| vllm/tests/v1/kv_connector/unit/test_hma_auto_config.py | 2 | HMA auto-disable when KV transfer connector configured |
| vllm/tests/v1/kv_connector/unit/test_output_aggregator.py | 2 | merge finished_sending/recving/invalid/failed across outputs |
| vllm/tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py | 2 | connector-level wrapper behavior over offload scheduler |
| tests/v1/e2e/test_cpu_linear_attn_chunked_prefix.py | 2 | cpu_model mark; KV space 1 |
| tests/v1/engine/test_init_error_messaging.py | 2 | mocked config |
| worker/test_dsv4_packed_zeroer_geometry.py | 2 | pytestmark cpu_test |
| worker/test_gpu_kv_connector.py | 2 | Mock backend; no markers |
| tests/utils_/test_cache.py | 2 | – |
| kv_canary/test_e2e_base.py | 2 | none (pure harness) |
| experimental/sgl-router/tests/component/policies/kv_events_hash_parity.rs | 2 | py-generated fixtures |
| experimental/sgl-router/tests/component/policies/kv_events_tree_concurrent.rs | 2 | tokio |
| experimental/sgl-router/tests/component/policies/kv_events_two_subscribers.rs | 2 | tokio |
| tools/sglang-simulator/test/test_simulation_cache_hit_ratio.py | 2 | sglang_simulator pkg + sglang.srt.server_args import |
| vllm/tests/v1/kv_connector/unit/test_cache_pollution_prevention.py | 1 | failed sync-load evicts invalid blocks from prefix cache |
| vllm/tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py | 1 | init/shutdown lifecycle of KVTransferState group |
| attention/test_chunked_local_attention.py | 1 | device_type |
| attention/test_group_sliding_window.py | 1 | — |
| tests/utils_/test_hashing.py | 1 | – |
| experimental/sgl-router/tests/component/policies/cache_prefix_provider.rs | 1 | tokio |
| experimental/sgl-router/tests/proxy/radix_tree_routing.rs | 1 | axum/tower |
| experimental/sgl-router/tests/proxy/external_indexer_routing.rs | 1 | axum/tower |
| experimental/sgl-router/sgl-kv-indexer/tests/common/{id,kv,net}.rs | — | — |

## C6 — Spec decode (dSpark) (49 rows)

| upstream path | n | pins |
|---|---|---|
| spec/dspark/test_dspark_{block_accept_estimator,confidence_metrics,dp_tier,draft_path_default,info_dumper,scheduler,sps_profiler,sps_table,sts,ragged_verify}.py (10 files) | 146 | none (CPU-registered, torch CPU) |
| spec/ — misc (adaptive_{runtime_state,spec_params}, decode_bookkeeping_ownership, decoupled_spec_io, fast_prefill_plan, plugin_hook_signatures, spec_registry, spec_utils_traverse_tree, suffix_attention_merge_dispatch, resolve_swa_kv_pool, spec_cpu_overlap_constraint) | 107 | cpu_ci |
| spec/ — ngram (ngram_corpus, ngram_mamba_verify_update) | 56 | cpu_ci |
| managers/ — spec/stop handling (finish_length_speculative, grammar_stop_speculative, stop_str_speculative, trim_matched_stop, vocab_boundary_finish, uno_request_validation, uno_token_accounting) | 55 | cpu_ci |
| spec/ — eagle + draft construction (eagle_{seeded_coins,worker_v2_topk1_fastpath,draft_extend_logits}, draft_construction_isolation, draft_per_runner_config) | 43 | cpu_ci |
| spec/ — dflash/dspark (dflash_{domino,extra_buffer_lazy,logits,overlap_hostsync}, dspark_target_hidden_projection) | 38 | cpu_ci |
| model_executor/model_runner_components/ (attention_backend_setup, cuda_graph_setup, layer_setup, ngram_embedding_manager, spec_aux_hidden_state, startup_weight_load) | 36 | cpu_ci |
| `completion/test_completion_error.py` | 28 | MagicMock engine, AsyncLLM spec |
| cpu/test_spec_kernels.py | ~25 | test_spec_kernels.py |
| sample/test_rejection_sampler.py | 20 | device_type |
| spec_decode/test_dynamic_sd.py | 17 | Scheduler fakes, caplog |
| spec_decode/test_mtp_structured_output.py | 13 | parametrize backend xgrammar/guidance; StructuredOutputManager fakes |
| worker/test_gpu_autoregressive_speculator.py | 11 | no skipif; mock-based; graph-replay tests may need CUDA |
| spec_decode/test_request_acceptance.py | 11 | pure data classes |
| spec_decode/test_eagle.py | 10 | `DEVICE_TYPE = current_platform.device_type`; test_load_model + 2 propose tests need Llama weights/kernels |
| spec_decode/test_extract_hidden_states.py | 9 | mocked model, `DEVICE_TYPE` from platform |
| spec_decode/test_adaptive_verification.py | 8 | monkeypatched fakes; imports gpu worker modules |
| spec_decode/test_backup_token_async_spec.py | 7 | pure fake batch/request classes |
| tests/config/test_speculative_draft_hf_overrides.py | 7 | none (MagicMock) |
| spec_decode/test_dynamic_sd_cug.py | 6 | `pytestmark = cpu_test`, monkeypatch |
| spec/ — uno tree (uno_{request_validation,tree_config,tree_sparse_sampling}) | 6 | cpu_ci |
| worker/test_spec_decode_embed_sharing_pp.py | 5 | fake PP groups; CPU |
| worker/test_workspace.py | 5 | stub config; CPU |
| spec_decode/test_dflash2.py | 5 | meta-device model init, monkeypatch |
| spec_decode/test_dflash_causality.py | 5 | config-level only |
| spec_decode/test_dspark_topk.py | 5 | 3 CPU-monkeypatch tests; 2 tests `skipif(not cuda)` incl. cudagraph capture |
| spec_decode/test_synthetic_rejection_sampler_utils.py | 5 | pure |
| model_executor/test_eagle_quantization.py | 5 | fully mocked |
| tests/config/test_speculative_draft_max_position_embeddings.py | 5 | none (PretrainedConfig) |
| worker/test_gpu_extract_hidden_states_speculator.py | 4 | none; cpu device in tests |
| spec_decode/test_eagle_draft_attn_metadata.py | 4 | asserts bound `.device.type == "cpu"` |
| structured_output/test_scheduler_speculative_padding.py | 4 | pure |
| spec_decode/test_dflash_lookahead.py | 3 | Scheduler/KV-cache fakes, StructuredOutputManager |
| spec_decode/test_draft_attention_backend_override.py | 3 | monkeypatched `load_eagle_model` |
| spec_decode/test_draft_moe_backend_override.py | 3 | monkeypatched `load_eagle_model` |
| spec_decode/test_llm_base_proposer.py | 3 | monkeypatch only |
| spec_decode/test_vocab_mapping.py | 3 | HF tokenizer downloads (meta-llama/Qwen) |
| structured_output/test_guidance_negative_draft_tokens.py | 3 | pure (fake tokenizer/matcher) |
| tests/v1/e2e/test_cpu_spec_decode.py | 2 | cpu_model; Triton gate |
| worker/test_eagle3_aux_hidden_states_pp.py | 2 | none |
| worker/test_gpu_rejection_sampler_chunking.py | 2 | 1/2 skipif CUDA |
| spec_decode/test_llm_base_proposer_sampling.py | 2 | `current_platform`, `set_random_seed` |
| spec_decode/test_mtp.py | 2 | mocked `get_model`/layers/pp_group |
| spec_decode/test_ngram.py | 2 | pure |
| model_executor/test_mistral_large_3_eagle.py | 2 | cpu_test, dummy modules |
| tests/config/test_bailing_mtp_config.py | 2 | none (PretrainedConfig) |
| manual/test_{aiter_unified_draft_extend_env,dsa_alias_cli_registry_env}.py | 2 | none (aiter test is flag-only, ROCm kernel not exercised) |
| model_executor/model_loader/test_mtp_validation.py | 1 | pure logic |
| model_executor/test_plamo3.py | 1 | cpu_test |

## C7 — MoE routing / expert placement (22 rows)

| upstream path | n | pins |
|---|---|---|
| lora/ — logic (eviction_policy, inkling_linearized_lora_unit, laguna_hidden_dim_unit, lm_head_pruning, lora_manager_tied_lm_head, lora_moe_inplace_unit, lora_spec_verify_batch_info, mem_pool_ep_unit, uno_{inactive_lora_batch,lora_targets}) | 70 | cpu_ci |
| models/ — deepseek (deepseek_mla_dispatch, deepseek_nextn_mm_embed, deepseek_v4_{amd_fused_mhc,amd_wo_a_bf16,mxfp4_shared_expert_requant,rope_policy,shared_expert_fusion,unified_fp8_q_pair}) | 51 | cpu_ci 48/52 dir |
| layers/moe/ — logic (copy_weight_views_before_h2d, fused_moe_common_utils, fused_shared_expert_scaling, hpc_ops_runner_guard, moe_runner_extensions, topk_correction_bias_cache) | 29 | cpu_ci |
| model_executor/test_routed_experts_capture.py | 15 | cpu_test, mocked EplbState |
| tests/quantization/test_moe_wna16.py | 14 | none (in-memory tensors) |
| distributed/test_eplb_algo.py | 13 | torch CPU tensors, numpy |
| eplb/test_balanced_packing.py | 11 | cpu_ci |
| eplb/test_compute_logical_to_rank_dispatch_physical_map.py | 11 | cpu_ci |
| expert_pack/test_expert_pack_runtime.py | 11 | none (CPU-registered) |
| eplb/test_dispatch_dtype_preservation.py | 9 | cpu_ci |
| expert_pack/test_kimi_k3_gguf.py | 9 | kimi-k3 pins, CPU-registered |
| cpu/test_topk.py | ~9 | test_topk.py |
| distributed/test_eplb_utils.py | 5 | torch CPU tensors, MagicMock |
| tests/quantization/test_gfx950_moe.py | 5 | none (stubbed configs) |
| worker/test_gpu_model_runner_v2_eplb.py | 4 | Fake memory profiler/EPLB state; no GPU |
| eplb/test_waterfill_eplb.py | 4 | cpu_ci |
| spec_decode/test_draft_moe_backend_override.py | 3 | monkeypatched `load_eagle_model` |
| distributed/test_eplb_events.py | 3 | torch, mocked state |
| distributed/test_expert_placement.py | 3 | pure index math |
| tests/quantization/test_int8_moe_oracle.py | 3 | none (stubbed configs) |
| moe/test_hash_topk.py | 3 | none (pure logic) |
| distributed/test_eplb_quant_scale_consistency.py | 2 | torch tensors, mocked |

## C8 — Scheduler / admission / lifecycle (110 rows)

| upstream path | n | pins |
|---|---|---|
| server_args/ — core + gates (server_args[235], unified_prefill_cuda_graph_gate, unified_tbo_gate, page_major_backend_allowlist, platform_prefill_cp_deprecation) | 255 | cpu_ci |
| managers/ — scheduler core (scheduler_*, schedule_batch_*, schedule_policy*, prefill_adder, prefill_delayer, retraction_order, priority_scheduling_disaggregation) | 163 | cpu_ci |
| tests/v1/core/test_scheduler.py | 113 | mocks + MockKVConfig, some hisparse/DCP cases |
| test_model_overrides.py | 101 | cpu_ci |
| test_runtime_context.py | 99 | cpu_ci |
| model_executor/ — top-level logic (chunked_prefix_cache_gate, draft_runner_skips_lora, forward_metadata_plan_record, hisparse_pool_configurator, kv_canary_headroom, mlp_sync_pad_unpad, model_runner_decode_rows, num_token_non_padded_localization, pool_configurator, unified_out_cache_loc_rebind) | 76 | cpu_ci |
| configs (generic: model_config{,_scaling,_shapes}, parser_registry, embedding_model_spec, linear_attn_model_registry, multimodal_piecewise_cuda_graph) | 53 | cpu_ci 13/13 |
| worker/test_gpu_profiler.py | 51 | proton tests skipif CUDA; config tests CPU |
| scripted_runtime/ | ~42 | test_scripted_runtime_core.py(42) |
| worker/test_gpu_model_runner.py | 40 | only 2 CUDA-only tests (FlashInfer, mamba views); gloo init |
| model_executor/model_runner_components/ (attention_backend_setup, cuda_graph_setup, layer_setup, ngram_embedding_manager, spec_aux_hidden_state, startup_weight_load) | 36 | cpu_ci |
| tests/v1/engine/test_admission_control.py | 31 | pure mocks; 503 mapping |
| tests/config/test_multimodal_config.py | 31 | none (MagicMock) |
| launchers/test_cli_args.py | 30 | pure argparse, no engine |
| tests/utils_/test_numa_utils.py | 29 | Linux /proc, NUMA hw |
| managers/ — batch result + streamers (batch_result_processor_*, customized_info_streaming, output_streamer_*) | 29 | cpu_ci |
| launchers/test_run_batch.py | 25 | MagicMock connection, subprocess bits |
| worker/test_gpu_ubatch_slicing.py | 22 | ~8 tests skipif CUDA (DBO exec, triton, graph pool); slicing pins CPU |
| attention/test_attention_splitting.py | 22 | cpu-device |
| ec_connector/unit/cpu/scheduler/test_scheduler.py | 21 | real ECSharedRegion mmap, CPU tensors |
| tests/v1/engine/test_engine_core_client.py | 20 | DPLB pure-mock; fork procs |
| tests/v1/core/test_encoder_cache_manager.py | 20 | pure mocks, no GPU |
| launchers/test_dp_supervisor.py | 20 | fake aiohttp children, no GPU per docstring |
| tests/config/test_model_arch_config.py | 20 | HF config JSONs (llama-68m etc.) |
| tests/v1/core/test_repetition_detection.py | 19 | pure unit + integration w/ mocks |
| test_runtime_context_config_bags.py | 18 | cpu_ci |
| spec_decode/test_dynamic_sd.py | 17 | Scheduler fakes, caplog |
| worker/test_gpu_pcp_manager.py | 15 | 1/15 skipif CUDA (GPU-kernel test); rest CPU math |
| tests/config/test_config_utils.py | 14 | none |
| tests/v1/core/test_deferred_block_free.py | 13 | pytestmark cpu_test, opt-125m, kv-connector mocks |
| worker/test_gpu_model_runner_v2_cudagraph_profiling.py | 13 | runner built via __new__, GPU helpers faked |
| test_environ.py | 13 | cpu_ci |
| test_runtime_context_override.py | 13 | cpu_ci |
| test_server_args_migration.py | 13 | cpu_ci |
| tests/v1/metrics/test_stats.py | 12 | pure unit |
| tests/v1/core/test_async_scheduler.py | 12 | mocks, pp 1/3, num_spec 0–3 |
| cli/test_serve_backends.py | 12 | cpu_ci, mocks |
| scheduler/test_min_free_slots_delayer.py | 12 | none (pure logic) |
| tests/v1/engine/test_output_processor.py | 11 | dummy test vectors |
| launchers/test_launch_cli.py | 11 | pure subcommand parser |
| worker/test_gpu_batch_ordering.py | 10 | none; CPU tensors |
| distributed/test_events.py | 10 | cpu device |
| managers/scheduler_components/ (dp_attn, invariant_checker, output_sender) | 10 | cpu_ci |
| vllm/tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py | 9 | recovery paths when async KV load fails (recompute/resume) |
| vllm/tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py | 9 | remote-prefill request lifecycle, finish/retry paths |
| cudagraph/test_cudagraph_manager.py | 9 | pytestmark cpu_test |
| test_split_attention_backend_decisions.py | 9 | cpu_ci |
| tests/v1/streaming_input/test_scheduler_streaming.py | 8 | scheduler-level, torch tensors, mocks |
| attention/test_backend_per_kind.py | 8 | — |
| spec_decode/test_adaptive_verification.py | 8 | monkeypatched fakes; imports gpu worker modules |
| vllm/tests/v1/kv_connector/unit/test_config.py | 7 | KVTransferConfig → connector factory translation |
| vllm/tests/v1/kv_connector/unit/test_nixl_heartbeat.py | 7 | scheduler-driven heartbeat / KV lease renewal semantics |
| worker/test_gpu_worker.py | 7 | SimpleNamespace worker; no GPU |
| attention/test_mamba_update_block_table.py | 7 | cpu-device |
| tests/config/test_speculative_draft_hf_overrides.py | 7 | none (MagicMock) |
| tests/v1/metrics/test_histogram_buckets.py | 6 | hard-coded literals |
| tests/v1/core/test_output.py | 6 | pure cpu |
| spec_decode/test_dynamic_sd_cug.py | 6 | `pytestmark = cpu_test`, monkeypatch |
| batch_overlap/test_tbo_children_dummy_token_mask.py | 6 | cpu_ci |
| tests/v1/engine/test_dp_placement_node_allowlist.py | 5 | env-var unit tests |
| tests/v1/engine/test_input_processor_trace_replay.py | 5 | mocked VllmConfig |
| tests/v1/metrics/test_metrics_reader.py | 5 | cpu_test mark |
| worker/test_gpu_input_batch.py | 5 | parametrized over current_platform device; CPU-capable |
| tests/config/test_speculative_draft_max_position_embeddings.py | 5 | none (PretrainedConfig) |
| test_server_args_namespaces.py | 5 | cpu_ci |
| vllm/tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py | 4 | remote-decode request lifecycle via example connector |
| ec_connector/unit/test_ec_output_aggregator.py | 4 | cpu_test mark, pure aggregation |
| tests/v1/engine/test_engine_args.py | 4 | CLI parsing unit tests |
| tests/v1/engine/test_iteration_logging.py | 4 | FakeEngineCore |
| worker/test_gpu_input_batch_v2.py | 4 | none; CPU |
| attention/test_linear_attention_metadata_builder.py | 4 | cpu-device |
| unit_tests/test_offline_utils.py | 4 | pure mixin, SamplingParams |
| batch_overlap/test_tbo_filter_batch_marker.py | 4 | cpu_ci, CPU-only per docstring |
| manual/debug_utils/ | 4 | none (log_parser); get_logits_ut is torch CPU-possible |
| vllm/tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py | 3 | invalid-block recompute/free semantics, no caching after fail |
| vllm/tests/v1/kv_connector/unit/test_scheduler_kv_connector_override.py | 3 | plugin/factory override of scheduler connector instance |
| tests/v1/engine/test_core_engine_actor_manager.py | 3 | mocked sockets/paths |
| tests/v1/engine/test_llm_engine_finalizer_is_weak.py | 3 | gc only |
| tests/v1/engine/test_startup_watch_processes.py | 3 | mocked zmq |
| tests/v1/core/test_worker_slot_overflow.py | 3 | scheduler-level, mocks |
| attention/test_gdn_metadata_builder.py | 3 | cpu-device |
| attention/test_replayssm_metadata_builder.py | 3 | cpu-device |
| unit_tests/test_remote_vllm_server.py | 3 | pure arg-list transform |
| tests/config/test_config_generation.py | 3 | deepseek-ai/DeepSeek-V2-Lite (config only) |
| test_server_args_cli_metadata.py | 3 | cpu_ci |
| compilation/test_torch_compile_decoration.py | 3 | cpu_ci, mocks |
| manual/test_{schedule_policy,weight_validation,config_integration}.py | 3 | none (in-proc, tempfile) |
| vllm/tests/v1/kv_connector/unit/test_error_propagation.py | 2 | connector load failure propagates to request abort |
| tests/v1/engine/test_init_error_messaging.py | 2 | mocked config |
| tests/v1/streaming_input/test_async_llm_streaming.py | 2 | AsyncMock/MagicMock, no GPU |
| worker/test_cp_utils.py | 2 | none |
| worker/test_mixed_warmup_gate.py | 2 | callback-must-not-run asserts; CPU |
| attention/test_attention_backends_selection.py | 2 | — |
| distributed/test_pipeline_partition.py | 2 | env monkeypatch, pure math |
| tests/config/test_bailing_mtp_config.py | 2 | none (PretrainedConfig) |
| scheduler/test_abort_with_metrics.py | 2 | none (pure unit) |
| core/test_srt_empty_deps.py | 2 | none |
| manual/test_{aiter_unified_draft_extend_env,dsa_alias_cli_registry_env}.py | 2 | none (aiter test is flag-only, ROCm kernel not exercised) |
| tools/sglang-simulator/test/test_simulation_cache_hit_ratio.py | 2 | sglang_simulator pkg + sglang.srt.server_args import |
| tools/sglang-simulator/test/test_simulation_sglang_serving.py | 2 | sglang_simulator pkg |
| vllm/tests/v1/kv_connector/unit/test_cache_pollution_prevention.py | 1 | failed sync-load evicts invalid blocks from prefix cache |
| tests/v1/core/test_priority_preemption_bug.py | 1 | scheduler-level, no GPU |
| tests/v1/core/test_priority_scheduler_random.py | 1 | seeded random, spec tokens param |
| attention/test_batch_reordering.py | 1 | — |
| tests/config/test_mp_reducer.py | 1 | none (mocked AsyncLLM) |
| mock_model/test_self_unit_canary_perturb.py | 1 | none (CPU-registered) |
| manual/core/ | 1 | torch CPU ok |
| test/config_publishers.py | 1 | none |
| tools/sglang-simulator/test/test_simulation_offline_blocking.py | 1 | sglang_simulator pkg |
| tools/sglang-simulator/test/test_simulation_sglang_runner.py | 1 | sglang.srt imports |

## C9 — Transport / RPC / failure injection (98 rows)

| upstream path | n | pins |
|---|---|---|
| managers/ — io/comm/misc (io_struct, msgpack_ipc_roundtrip, fanout_communicator, loadstat_wire, data_parallel_controller, detailed_annotations, flat_raw_top_logprobs, generation_auxiliary_output, hidden_state_server_mode, hisparse_unit, kv_page_invariants, lora_update_result_merge, mamba_checkpoint_depth, pp_cp_rank_offsets, profile_merger_http_api, sampling_mask_validation, load_inquirer, load_snapshot_backends) | 227 | cpu_ci |
| disaggregation/ — encode/decode lifecycle: encode_{receiver,server,scheduler}, encoder_health, decode_{hicache_tree_core,req_to_token_pool}, deferred_decode_kv_release, prefill_abort_result_cleanup, kimi_k3_encoder_mode, minimax_sparse_disagg_state | 165 | cpu_ci, mocks |
| kv_offload/tiering/p2p/test_sessions.py | 101 | fake control connections only |
| sgl-model-gateway/tests/routing/*.rs | 90 | axum/tower, mock workers |
| distributed/test_weight_transfer.py | 88 | mostly CPU map math, some NCCL |
| disaggregation/ — wire: disaggregation_wire, dcp_pack, kv_events, register_to_bootstrap, decode_queue_cleanup | 88 | cpu_ci 26/26, mocks |
| sgl-model-gateway/tests/api/*.rs | 78 | axum/tower in-process, mock workers |
| kv_offload/tiering/p2p/test_manager.py | 65 | fake transports/sessions, no real net |
| sgl-model-gateway/tests/reliability/*.rs | 49 | axum/tower, mock workers |
| disaggregation/ — kv-transfer misc: pp_hybrid_kv_transfer, kv_transfer_replica_metric, specv2_kvcache_offloading, staging_draft_kv_slots, unified_memory_move_gate | 48 | cpu_ci |
| sgl-model-gateway/tests/security/*.rs | 39 | axum/tower, TLS certs |
| ec_connector/unit/test_session.py | 36 | MagicMock data transport |
| rust/src/engine-core-client/src/tests/client.rs | ~36 | zeromq, tokio; python_compat fixtures |
| scale_out/derender/test_derender.py | 35 | RemoteLaunchRenderServer (GPU-less) |
| experimental/sgl-router/tests/proxy/chat_routing.rs | 35 | axum/tower, mock workers |
| scale_out/render/test_render.py | 32 | mocked ServingRender + GPU-less server |
| vllm/tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py | 28 | P pulls KV from D; remote_block_ids lifecycle, partial coverage |
| scale_out/derender/test_derender_stream.py | 28 | tokenizer-only unit layer + render server |
| kv_offload/tiering/p2p/test_data_transport.py | 25 | nixl agent mocked or patched to None |
| cp/test_cp_strategy_unit.py | 24 | none (CPU-registered, mocks) |
| cpu/test_rank_consensus_checker.py | ~23 | test_rank_consensus_checker.py |
| ec_connector/unit/test_control.py | 20 | mocked dealer sockets |
| tests/v1/engine/test_engine_core_client.py | 20 | DPLB pure-mock; fork procs |
| dcp/test_dcp_layout_unit.py | 17 | none (explicit CPU unit test) |
| kv_offload/tiering/p2p/test_zmq_transport.py | 16 | needs pyzmq; 127.0.0.1 loopback |
| tests/utils_/test_network_utils.py | 15 | zmq |
| vllm/tests/v1/kv_connector/unit/test_nixl_desc_geometry.py | 14 | transfer byte-range invariants under P/D block geometry |
| ec_connector/unit/test_data.py | 14 | NixlWrapper fully mocked |
| experimental/sgl-router/tests/proxy/bucket_routing.rs | 14 | axum/tower |
| scale_out/token_in_token_out/test_mm_serde.py | 13 | CPU torch tensors, pydantic |
| ec_connector/unit/test_scheduler_nixl_consumer.py | 12 | CPU tensors; monkeypatched nixl fields |
| scale_out/token_in_token_out/test_generate_stream.py | 12 | AsyncMock AsyncLLM, no server |
| executor/test_vllm_net_devices.py | 11 | pure parsing |
| vllm/tests/v1/kv_connector/unit/test_tp_mapping.py | 10 | TP mapping: source ranks, split handles, desc IDs, no GPU |
| experimental/sgl-router/tests/component/workers/manager.rs | 10 | tokio, mock workers |
| experimental/sgl-router/tests/component/policies/bucket_domains.rs | 9 | tokio, in-process |
| experimental/sgl-router/sgl-kv-indexer/tests/grpc_contract.rs | 9 | tonic, in-process server |
| ec_connector/unit/test_epd_proxy_round_robin.py | 8 | loads real examples/ proxy |
| scale_out/render/test_launch_render.py | 8 | RemoteLaunchRenderServer |
| scale_out/token_in_token_out/test_protocol.py | 8 | direct protocol objects |
| experimental/sgl-router/tests/proxy/pd_pool_isolation.rs | 8 | axum/tower |
| sgl-model-gateway/tests/wasm_test.rs | 8 | wasm runtime |
| distributed/test_shm_storage.py | 7 | cpu |
| distributed/test_gated_launch.py | 7 | cpu_ci |
| distributed/test_shm_buffer.py | 6 | multi-proc, no CUDA |
| experimental/sgl-router/tests/proxy/shared_prefill_admission.rs | 6 | axum/tower |
| sgl-model-gateway/tests/load_guard_raii_test.rs | 6 | — |
| ec_connector/unit/test_epd_proxy_retry.py | 5 | loads real examples/ proxy; aiohttp+httpx |
| ec_connector/unit/test_scheduler_nixl_ctor.py | 5 | skipif NixlWrapper absent |
| executor/test_multiproc_executor_timeout.py | 5 | pure Python futures + monotonic clock |
| scale_out/test_factories.py | 5 | FastAPI app, Namespace args |
| distributed/test_parallel_state.py | 5 | cpu_ci |
| experimental/sgl-router/tests/component/policies/decode.rs | 5 | tokio |
| experimental/sgl-router/tests/component/policies/power_of_two.rs | 5 | tokio |
| experimental/sgl-router/tests/proxy/sticky_input_ids.rs | 5 | axum/tower |
| experimental/sgl-router/tests/proxy/sticky_routing.rs | 5 | axum/tower |
| experimental/sgl-router/tests/proxy/pd_bootstrap_injection.rs | 5 | axum/tower |
| sgl-model-gateway/tests/metrics_aggregator_test.rs | 5 | — |
| vllm/tests/v1/kv_connector/unit/test_transfer_topology_sharded.py | 4 | sharded TransferTopology registration of engine info |
| ec_connector/unit/test_scheduler_nixl_producer.py | 4 | CPU tensors; monkeypatched nixl fields |
| ec_connector/unit/test_utils.py | 4 | msgspec only |
| worker/test_pp_utils.py | 4 | numpy + Mock batch; CPU |
| distributed/test_get_default_distributed_backend.py | 4 | cpu_ci |
| experimental/sgl-router/tests/component/health/circuit_breaker.rs | 4 | tokio |
| experimental/sgl-router/tests/proxy/cache_aware_input_ids.rs | 4 | axum/tower |
| vllm/tests/v1/kv_connector/unit/test_handshake_pp_aggregation.py | 3 | handshake metadata aggregation across PP ranks |
| ec_connector/unit/test_protocol.py | 3 | msgspec only |
| tests/v1/engine/test_core_engine_actor_manager.py | 3 | mocked sockets/paths |
| tests/v1/engine/test_startup_watch_processes.py | 3 | mocked zmq |
| distributed/test_mq_connect_ip.py | 3 | sockets, no GPU |
| tests/utils_/test_serial_utils.py | 3 | pybase64, numpy |
| experimental/sgl-router/tests/component/discovery/static_urls.rs | 3 | tokio |
| experimental/sgl-router/tests/component/policies/round_robin.rs | 3 | tokio |
| experimental/sgl-router/tests/component/policies/fused_score.rs | 3 | tokio |
| experimental/sgl-router/tests/proxy/roundrobin_input_ids.rs | 3 | axum/tower |
| sgl-model-gateway/tests/inflight_tracker_test.rs | 3 | — |
| vllm/tests/v1/kv_connector/unit/test_output_aggregator.py | 2 | merge finished_sending/recving/invalid/failed across outputs |
| executor/test_multiproc_executor.py | 2 | fake queue; CPU |
| executor/test_ray_utils.py | 2 | numpy only; CPU |
| scale_out/token_in_token_out/test_tokens_logprobs.py | 2 | pure ServingTokens static method |
| distributed/test_cuda_wrapper.py | 2 | cpu_ci |
| entrypoints/test_grpc_bridge.py | 2 | cpu_ci |
| test/observability/ (2 py) | 2 | none |
| experimental/sgl-router/tests/component/workers/concurrent_state.rs | 2 | tokio |
| experimental/sgl-router/tests/component/policies/kv_events_hash_parity.rs | 2 | py-generated fixtures |
| experimental/sgl-router/tests/component/policies/kv_events_tree_concurrent.rs | 2 | tokio |
| experimental/sgl-router/tests/component/policies/kv_events_two_subscribers.rs | 2 | tokio |
| sgl-model-gateway/tests/otel_tracing_test.rs | 2 | otel collector |
| test/otel_collector.py | 1 | none |
| experimental/sgl-router/tests/proxy/failover.rs | 1 | axum/tower, 3 mock workers |
| experimental/sgl-router/tests/proxy/timeout.rs | 1 | axum/tower |
| experimental/sgl-router/tests/proxy/graceful_shutdown.rs | 1 | axum/tower |
| experimental/sgl-router/tests/proxy/header_forwarding.rs | 1 | axum/tower |
| experimental/sgl-router/tests/proxy/radix_tree_routing.rs | 1 | axum/tower |
| experimental/sgl-router/tests/proxy/external_indexer_routing.rs | 1 | axum/tower |
| experimental/sgl-router/tests/component/policies/zmq_helpers.rs | 0 | zmq |
| experimental/sgl-router/tests/proxy/common/{mock_worker,streaming,cache_aware_fixture}.rs | — | — |
| sgl-model-gateway/tests/common/*.rs (11 files) | — | axum/tower, redis server bin |

## C10 — Weights / quant formats (0 rows)

| upstream path | n | pins |
|---|---|---|

## C11 — Vision / multimodal (0 rows)

| upstream path | n | pins |
|---|---|---|

## C12 — Tool-call parsing (0 rows)

| upstream path | n | pins |
|---|---|---|
