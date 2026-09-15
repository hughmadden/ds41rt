# DEFERRED — fleet-dependent upstream tests (DO NOT RUN while the fleet is live)

These upstream tests are RELEVANT to ds41rt but require GPU weights, a served model,
kernels, or multi-node fleet state. They are recorded here for the fleet-clear window.
Hard rule: nothing in this document may be executed against the live fleet; each entry
names the unblocking condition. Owner runs them only in the designated window.

## Unblocking conditions

- **UC-1 coordinator free**: RTX coordinator host idle, ds41rt images available. Covers single-node GPU tests (kernels, served-model API tests, sampler/GPU parity).
- **UC-2 Sparks released**: 4x Spark workers free (GLM EXL3 stopped per standing instruction). Covers TP4/multi-node transport, expert-parallel, disaggregation tests.
- **UC-3 full fleet**: coordinator + Sparks with the qualified ds41rt v2 images. Covers end-to-end throughput/accuracy/persistence-soak tests.
- **UC-4 weights present**: specific HF checkpoints downloaded (vision/STT/pooling models ds41rt does not normally stage).
- **UC-5 reference packages**: `b12x` reference package + generated fixtures (e.g. tests/fixtures/nvfp4/real_tensor_decode.json) available on the GPU dev host.

## Semantic-parity questions (need real model output to resolve)

Characterized by the ported tool-call suite (`upstream_tool_calls.rs`, MAPPING
DIVERGENCE comments); do NOT change parser semantics until validated against real
DeepSeek-V4.1-Flash outputs in the fleet window (**UC-3**):

1. **Missing `<｜DSML｜tool_calls>` wrapper recovery** — vLLM recovers a bare
   `<｜DSML｜invoke>` block (vllm#48931); ds41rt does not (non-stream stays content;
   stream leaks raw invoke syntax into content deltas).
2. **`arguments`-wrapper nesting** — vLLM unwraps `<｜DSML｜parameter name="arguments" ...>`;
   ds41rt's schema-free parser nests one level (`{"arguments":{...}}`). Could bite real
   tool consumers; consistent with ds41rt's no-schema design today.
3. **Stream parser anchor strictness** — ds41rt anchors on `"\n\n<｜DSML｜tool_calls"`
   (leading blank line required); vLLM detects the bare marker.

Each has a pinned-behavior test in `upstream_tool_calls.rs`; if the fleet window shows
real outputs that trip these, fix the parser and flip the tests to the vLLM semantics.

4. **Unset temperature is greedy** — ds41rt treats an absent `temperature` as 0.0
   (greedy), diverging from the OpenAI/vLLM default of 1.0; the 1.0/0.95/50 sampling
   defaults in `request_sampling_params` are unreachable unless temperature is set
   explicitly. Pinned by tests in `upstream_sampler.rs`. Likely intentional for the
   current eval harnesses, but an OpenAI-compatibility question for the owner before
   broad client exposure; validate against real client traffic in the fleet window.


## C1 — OpenAI protocol / API surface (186 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| python/sglang/multimodal_gen/test/unit/*.py (top level, ~173 files) | ~2100 | torch + diffusion model weights | UC-1 |
| kernels/ops/ | ~1022 | test_c128_v2.py, test_fp8_blockwise_gemm.py, test_fused_topk_deepseek.py | UC-1 |
| `kernels/quantization/` (test_aiter_hipb_mm_linear_kernel, test_allspark_gemm, test_awq, test_awq_triton, test_block_fp8, test_cpu_fp8_scaled_mm, test_cutlass_scaled_mm, test_cutlass_w4a8, test_cutlass_w4a8_moe, test_flashinfer_mxfp8_trtllm, test_flashinfer_nvfp4_scaled_mm, test_flashinfer_scaled_mm, test_fp8_min_max_helper, test_fp8_quant, test_fp8_quant_group, test_gptq, test_hadacore, test_int4_emulation_moe, test_int8_kernel, test_int8_quant, test_machete_mm, test_marlin_gemm, test_marlin_tile_padding, test_mxfp4_kernel_selection, test_mxfp4_qutlass, test_mxfp4_triton_ep, test_mxfp6_kernel_selection, test_nvfp4_emulation, test_nvfp4_kernel_selection, test_nvfp4_quant, test_nvfp4_qutlass, test_nvfp4_scaled_mm, test_per_token_group_quant, test_quantized_embedding, test_quant_op_schema, test_rdna3_compile_guards, test_rdna3_moe_w4a16, test_rdna3_w4a16, test_rdna3_w4a16_selection, test_rdna_hybrid_w4a16, test_rocm_aiter_grouped_quant, test_rocm_compressed_tensors_w4a16, test_rocm_fp8, test_rocm_mxfp4, test_rocm_mxfp8_linear, test_rocm_skinny_gemms, test_scaled_mm_kernel_selection, test_silu_mul_nvfp4_quant, test_triton_scaled_mm, test_triton_w4a16, test_w4a16_kernel_selection; helpers `nvfp4_utils.py`, `quant_utils.py`) | 258 | 102 skipif | UC-1 |
| python/sglang/multimodal_gen/test/unit/realtime/*.py (8 files) | 136 | GPU | UC-1 |
| `kernels/helion/` (helpers, test_autotune, test_benchmark_script, test_case_key, test_config_manager, test_dynamic_per_token_scaled_fp8_quant, test_fused_qk_norm_rope, test_helion_available, test_pattern_matching, test_per_token_group_fp8_quant, test_register, test_rms_norm_dynamic_per_token_quant, test_rms_norm_per_block_quant, test_silu_and_mul_per_block_quant, test_silu_mul_fp8, test_utils; `utils.py`) | 128 | 6 skipif | UC-1 |
| python/sglang/multimodal_gen/test/unit/sana_wm/*.py (7 files) | 121 | GPU | UC-1 |
| `kernels/core/` (test_activation, test_apply_rotary_emb, test_batched_weight_rms_norm, test_cpu_activation, test_fused_allreduce_gemma_rms_norm, test_fused_embed_norm, test_fused_qk_norm_rope, test_fused_q_kv_rmsnorm, test_fused_quant_layernorm, test_fused_rms_norm_gated, test_fused_silu_mul_block_quant, test_layernorm, test_minimax_reduce_rms, test_mrope, test_opcheck, test_pos_encoding, test_rocm_aiter_ops, test_rocm_misc_ops, test_rotary_embedding, test_rotary_embedding_mla_cache_fused, test_uva, test_vit_bilinear_pos_embed, test_vit_fp8_attn, test_vit_fp8_quant, test_vit_fp8_scaling, test_vocab_parallel_embedding) | 94 | 35 skipif | UC-1 |
| tests/quantization/test_auto_round.py | 90 | many quantized HF models | UC-1 |
| `kernels/core/` (excl. activation-owned): test_apply_rotary_emb.py, test_batched_weight_rms_norm.py, test_fused_allreduce_gemma_rms_norm.py, test_fused_embed_norm.py, test_fused_qk_norm_rope.py, test_fused_q_kv_rmsnorm.py, test_fused_quant_layernorm.py, test_fused_rms_norm_gated.py, test_fused_silu_mul_block_quant.py, test_layernorm.py, test_minimax_reduce_rms.py, test_mrope.py, test_opcheck.py, test_pos_encoding.py, test_rocm_aiter_ops.py, test_rocm_misc_ops.py, test_rotary_embedding_mla_cache_fused.py, test_rotary_embedding.py, test_uva.py, test_vit_bilinear_pos_embed.py, test_vit_fp8_attn.py, test_vit_fp8_quant.py, test_vit_fp8_scaling.py, test_vocab_parallel_embedding.py | 81 | CUDA/triton; ROCm files; UVA pinned mem | UC-1 |
| multimodal/ — transports (cuda_ipc_{pool_budget,transport}, cuda_vmm_transport, gpu_feature_transport, tensor_transport_mode, vit_{cuda_graph_metadata_cuda,cuda_graph_runner,npu_graph_runner}, kimi_k3_gpu_preprocess) | 70 | cuda | UC-1 |
| layers/quantization/ — kernels+backends (bf16_splitk_gemm, deepgemm_ue8m0_requant, flashinfer_trtllm_fp8_fallback, fp8_blockwise_linear_backends, fp8_kernel_hip_max, fp8_moe_runner_{fallback,ownership}, mxfp4_{flashinfer_activation_prep,situ_output,situ_weight_layout,sm100_trtllm_gen,sm120_cutlass,sm90_cutlass}, nvfp4_{linear,moe}_backends) | 69 | cuda/hip/cutlass | UC-1 |
| quant/ | ~61 | test_quark_mxfp4.py(18), test_fp8_utils.py(6), test_quant_config_parsing.py | UC-1 |
| sgl-model-gateway/e2e_test/responses/*.py | 59 | live gateway + model | UC-1 |
| sgl-model-gateway/e2e_test/responses/*.py | 59 | live gateway + model | UC-1 |
| layers/ — top-level triton/kernels (flashattention_paged_mha, mamba_state_scatter_triton, fp8_bpreshuffle_{dense_linear,producer}_mi35x, layernorm_sp, conv_layer) | 58 | cuda/hip | UC-1 |
| sgl-model-gateway/e2e_test/chat_completions/*.py | 49 | live gateway + model | UC-1 |
| sgl-model-gateway/e2e_test/chat_completions/*.py | 49 | live gateway + model | UC-1 |
| cudagraph/test_encoder_cudagraph.py | 44 | 3 no-GPU classes; capture/replay classes CUDA/ROCm | UC-1 |
| `chat_completion/test_chat.py` | 35 | zephyr-7b-beta server | UC-1 |
| `responses/test_harmony.py` | 32 | gpt-oss harmony server | UC-1 |
| `responses/test_harmony.py` | 32 | gpt-oss harmony server | UC-1 |
| tests/quantization/test_quark.py | 31 | amd/ quark repos | UC-1 |
| kernel/diffusion/ | ~30 | test_vdn_linear_branch.py(6), test_vdn_delta_factors.py | UC-1 |
| openai_server/basic/test_openai_server.py | 28 | GPU server (amd+cuda) | UC-1 |
| tests/quantization/test_compressed_tensors.py | 27 | nm-testing tiny-llama repos | UC-4 |
| tests/quantization/test_modelopt.py | 27 | mock + runner mixes | UC-1 |
| tests/quantization/test_online.py | 26 | nm-testing tinysmokeqwen3moe | UC-4 |
| model_executor/kernels/test_b12x_linear.py | 23 | B200-specific modules, importlib sweep | UC-1 |
| ec_connector/unit/cpu/worker/test_worker.py | 22 | 6 lifecycle CPU; 16 need CUDA/XPU accel | UC-1 |
| `completion/test_completion.py` | 21 | opt-125m server | UC-1 |
| tests/quantization/test_blackwell_moe.py | 21 | deepseek-ai/DeepSeek-V3.1, nvidia FP4 repos | UC-1 |
| manual/dsv4/ | 21 | 8 GPU, DSv4 weights, sgl-eval on PATH | UC-1 |
| vlm/ | ~20 | test_vision_openai_server_a.py(4), test_rust_native_mm_e2e.py, test_video_utils.py | UC-4 |
| tests/v1/engine/test_async_llm.py | 18 | CUDA-only module skip | UC-1 |
| `kernels/test_compressor_kv_cache.py` | 18 | 10 | UC-1 |
| tests/multimodal/test_gpu_ipc_memory.py | 16 | CUDA, pynvvideocodec consts | UC-1 |
| manual/quant/ | 15 | CUDA, Blackwell some | UC-1 |
| tests/v1/e2e/general/test_streaming_input.py | 14 | opt-125m | UC-1 |
| tests/quantization/test_fp8.py | 14 | facebook/opt-125m, allenai OLMoE, nm-testing FP8 | UC-1 |
| openai_server/function_call/test_openai_function_calling.py | 13 | GPU server (amd+cuda+npu) | UC-1 |
| openai_server/function_call/test_openai_function_calling.py | 13 | GPU server (amd+cuda+npu) | UC-1 |
| `kernels/test_fused_inv_rope_fp8_quant.py` | 12 | 0 | UC-1 |
| model_executor/test_jit_warmup_triton_launcher.py | 12 | triton.jit, GPU launch | UC-1 |
| tests/basic_correctness/test_mem.py | 12 | GPU, fp8 subtests | UC-1 |
| attention/test_mm_prefix.py | 11 | cuda+fa4 | UC-1 |
| multimodal/openai/chat_completion/test_vision.py | 11 | RemoteOpenAIServer | UC-1 |
| tests/quantization/test_torchao.py | 11 | facebook/opt-125m, Qwen/Qwen3-0.6B | UC-4 |
| manual/layers/ | 11 | CUDA/triton kernels; benches H100 | UC-1 |
| `responses/test_simple.py` | 10 | Qwen3-8B server | UC-4 |
| serve/lora/test_lora_adapters.py | 10 | RemoteOpenAIServer, Qwen3-0.6B | UC-4 |
| openai_server/function_call/test_anthropic_tool_use.py | 10 | GPU server; also CPU-CI registered | UC-1 |
| openai_server/function_call/test_anthropic_tool_use.py | 10 | GPU server; also CPU-CI registered | UC-1 |
| manual/lora/ | 10 | CUDA | UC-1 |
| rust/sglang-mm/tests/test_golden.py + test_integration.py + test_resize_parity.py + test_hash_fetch.py | 10 | torch, PIL, soundfile, golden assets | UC-1 |
| serve/middleware/test_optional_middleware.py | 9 | RemoteOpenAIServer, e5-small | UC-1 |
| tool_use/test_tool_choice_required.py | 8 | server+model | UC-1 |
| `kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | 8 | 3 | UC-1 |
| openai_server/validation/test_request_length_validation.py | 8 | GPU server | UC-1 |
| gemm/ | ~8 | test_linear_bf16_fp32_hpc.py(5), test_hopper_bf16_gemv.py | UC-1 |
| manual/test_{get_weights_by_name,weight_cache_e2e,weight_loader_v2_equiv,expert_location_updater,expert_distribution,moe_quant_once,modelopt,modelopt_fp8kvcache}.py | 8 | GPU, multi-proc distributed some | UC-1 |
| `chat_completion/test_batched_chat_completions.py` | 7 | Qwen2.5-1.5B server | UC-4 |
| `chat_completion/test_chat_completion_with_prompt_embeds.py` | 7 | opt-125m + chatml.jinja template, torch | UC-1 |
| `responses/test_stateful.py` | 7 | conftest client (store enabled) | UC-1 |
| serve/instrumentator/test_basic.py | 7 | RemoteOpenAIServer, Qwen3-0.6B | UC-4 |
| `kernels/test_relu2_fp8_quant.py` | 7 | CUDA | UC-1 |
| `kernels/test_relu2_fp8_quant.py` | 7 | 1 | UC-1 |
| lora/ — sgl-marlin kernels (experimental_sgl_marlin_{multi_prefill,policy,runtime_unit,shared_outer_reduce}) | 7 | cuda | UC-1 |
| sgl-model-gateway/e2e_test/embeddings/*.py | 7 | live backend | UC-1 |
| `chat_completion/test_chat_completion.py` | 6 | Qwen2.5-1.5B server | UC-4 |
| `chat_completion/test_include_reasoning.py` | 6 | Qwen3-0.6B + qwen3 reasoning parser | UC-4 |
| `responses/test_basic.py` | 6 | conftest client (Qwen3-1.7B server) | UC-4 |
| `responses/test_function_call.py` | 6 | conftest client | UC-1 |
| tool_use/mistral/test_mistral_tool_calls.py | 6 | server+mistral model | UC-1 |
| `kernels/test_awq_int4_to_int8.py` | 6 | CUDA + CPU oneDNN paths | UC-1 |
| `kernels/test_cp_gather_fp8.py` | 6 | 0 | UC-1 |
| tests/quantization/test_humming_mxfp4_block_fp8.py | 6 | none (in-memory tensors) | UC-1 |
| manual/test_{fim_completion,health_check,weight_version,sagemaker_server,vertex_endpoint,crusoe_backend}.py | 6 | launched server; crusoe needs API key | UC-1 |
| unit/test_vision_api.py | 6 | multimodal capability flags, vision chat/completion/token count/embedding | UC-4 |
| `chat_completion/test_chat_logit_bias_validation.py` | 5 | Qwen2.5-1.5B server + ModelConfig | UC-4 |
| `chat_completion/test_completion_with_function_calling.py` | 5 | Qwen3-0.6B server | UC-4 |

(106 more rows in UPSTREAM-INVENTORY.md)


## C2 — Constrained / structured output (22 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| `kernels/mamba/` (test_causal_conv1d, test_cpu_short_conv, test_gdn_forward_core_split, test_gdn_fused_mtp, test_gdn_prefill_cutedsl, test_gdn_prefill_flashinfer, test_mamba_mixer2, test_mamba_ssm_configs, test_mamba_ssm, test_mamba_ssm_ssd, test_memcpy_u64_tiled, test_precopy_mamba_align, test_replayssm_prefill_decode_equivalence_mamba2, test_replayssm_standard_decode_mamba2, test_ssu_dispatch; `cpu/`, `utils.py`) | 51 | 22 skipif | UC-1 |
| `chat_completion/test_chat.py` | 35 | zephyr-7b-beta server | UC-1 |
| `kernels/ir/` (test_activation, test_ir_ops, test_layernorm) | 24 | 17 skipif | UC-1 |
| `kernels/test_engram.py` | 15 | 11 | UC-1 |
| `kernels/test_fla_layernorm_guard.py` | 9 | 0 | UC-1 |
| `chat_completion/test_chat_completion.py` | 6 | Qwen2.5-1.5B server | UC-4 |
| llm/test_struct_output_generate.py | 6 | vllm_runner + jsonschema/regex/xgrammar | UC-4 |
| `kernels/test_kpool_decode_update_batched.py` | 6 | 2 | UC-1 |
| `kernels/test_fused_indexer_q_rope_quant.py` | 5 | 6 | UC-1 |
| `kernels/test_fused_gdn_post_conv.py` | 4 | 0 | UC-1 |
| `kernels/test_fused_recurrent_packed_decode.py` | 3 | 1 | UC-1 |
| compile/test_dynamic_shapes_compilation.py | 3 | LLM, logprobs compare | UC-1 |
| model_executor/test_replayssm_warmup.py | 3 | cuda-alike/flashinfer paths | UC-1 |
| `responses/test_structured_output.py` | 2 | conftest client | UC-1 |
| `kernels/test_fused_sigmoid_gating_delta_rule.py` | 2 | 0 | UC-1 |
| compile/fullgraph/test_full_graph.py | 2 | LLM, quant support check | UC-1 |
| manual/distributed/ | 2 | multi-GPU, MLA models | UC-2 |
| compile/fullgraph/test_basic_correctness.py | 1 | GPU runner, HF models | UC-1 |
| compile/fullgraph/test_full_cudagraph.py | 1 | LLM, GPU mem | UC-1 |
| model_executor/test_mamba_triton_warmup.py | 1 | cuda required | UC-1 |
| model_executor/test_qwen_triton_warmup.py | 1 | cuda required | UC-1 |
| constrained_decoding/test_constrained_decoding.py | 0 | GPU server (amd+cuda) | UC-1 |

## C3 — Sampler semantics (43 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| python/sglang/kernels/aot/tests/test_*.py (27 files) | 71 | CUDA GPU, sgl_kernel | UC-1 |
| `chat_completion/test_chat.py` | 35 | zephyr-7b-beta server | UC-1 |
| sample/test_logprobs.py | 26 | large_gpu, HfRunner/VllmRunner | UC-4 |
| `kernels/test_top_k_per_row.py` | 23 | CUDA SM100/SM103 | UC-1 |
| `completion/test_completion.py` | 21 | opt-125m server | UC-1 |
| sampling/test_sampling_mask.py | 21 | GPU server (amd+cuda) | UC-1 |
| spec_decode/test_rejection_sampler_utils.py | 18 | module-level `skipif(not cuda)` + `importorskip triton` | UC-1 |
| worker/test_gpu_thinking_budget.py | 13 | importorskip triton + CUDA skip | UC-1 |
| tests/v1/determinism/test_rms_norm_batch_invariant.py | 11 | CUDA, dtype/eps/seed sweeps | UC-1 |
| worker/test_gpu_gumbel_sample.py | 10 | importorskip triton + CUDA skip | UC-1 |
| sampling/test_penalty.py | 10 | GPU server (amd+cuda) | UC-1 |
| tests/test-backend-sampler.cpp | ~10 | top-k/top-p/temp via ggml-backend graph vs CPU sampler chain | UC-1 |
| worker/test_gpu_batch_shard.py | 8 | DEVICE=cuda; per-test requires_cuda | UC-1 |
| sample/test_sampling_params_e2e.py | 8 | LLM tiny-random | UC-1 |
| tests/v1/e2e/general/test_min_tokens.py | 7 | opt-125m small | UC-1 |
| worker/test_gpu_trace_replay.py | 7 | importorskip triton + CUDA skip | UC-1 |
| `chat_completion/test_batched_chat_completions.py` | 7 | Qwen2.5-1.5B server | UC-4 |
| tests/v1/determinism/test_batch_invariance.py | 5 | CUDA, FLASH_ATTN, flaky reruns, 1000s timeout | UC-1 |
| `chat_completion/test_chat_logit_bias_validation.py` | 5 | Qwen2.5-1.5B server + ModelConfig | UC-4 |
| tests/samplers/test_beam_search.py | 5 | TinyLlama-1.1B, Qwen2-Audio pins | UC-4 |
| tests/v1/engine/test_llm_engine.py | 4 | opt-125m runner boot | UC-1 |
| worker/test_gpu_bad_words.py | 4 | importorskip triton + module-level CUDA skip | UC-1 |
| worker/test_gpu_logit_bias.py | 4 | importorskip triton + CUDA skip | UC-1 |
| logits_processors/test_custom_offline.py | 4 | LLM | UC-1 |
| `test_return_token_ids.py` | 4 | Qwen2.5-1.5B + hermes tool parser | UC-4 |
| manual/4-gpu-models/ | 4 | 4 GPU, specific Qwen weights | UC-2 |
| tests/v1/determinism/test_matmul_batch_invariant.py | 3 | CUDA platform-gated | UC-1 |
| worker/test_gpu_sampler_flags.py | 3 | importorskip triton + CUDA skip | UC-1 |
| `chat_completion/test_chat_echo.py` | 3 | Qwen2-1.5B server | UC-4 |
| manual/beam_search/ | 3 | GPU + HF reference + server | UC-1 |
| sample/test_logprobs_e2e.py | 2 | LLM+RemoteOpenAIServer | UC-1 |
| logits_processors/test_custom_online.py | 2 | RemoteOpenAIServerCustom | UC-1 |
| `kernels/test_apply_repetition_penalties.py` | 2 | CUDA | UC-1 |
| tests/samplers/test_no_bad_words.py | 2 | tokenizer + LLM weights | UC-1 |
| sampling/test_pytorch_sampling_backend.py | 2 | GPU server | UC-1 |
| tests/v1/e2e/general/test_sharded_sampling.py | 1 | multi-GPU; tolerance-based | UC-2 |
| tests/v1/e2e/spec_decode/test_sharded_sampling.py | 1 | LLM; multi-GPU | UC-2 |
| tests/v1/determinism/test_online_batch_invariance.py | 1 | RemoteOpenAIServer + model | UC-1 |
| tests/samplers/test_ignore_eos.py | 1 | meta-llama/Llama-3.2-1B pin | UC-4 |
| tests/samplers/test_logprobs.py | 1 | runner model | UC-1 |
| sampling/test_original_logprobs.py | 1 | GPU server + HF | UC-1 |
| manual/eval/ | 1 | server + datasets | UC-1 |
| manual/test_logprobs.py | 1 | GPU precision-sensitive | UC-1 |

## C4 — Tokenizer / detok / chat templates (19 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| serve/tokenize/test_tokenization.py | 9 | RemoteOpenAIServer, SmolLM2-135M | UC-1 |
| kernel/embeddings/ | ~7 | test_qwen4_ple_offload.py | UC-4 |
| input_embedding/ | ~7 | test_input_embeddings.py(4), test_input_embeds_chunked.py | UC-1 |
| tokenizer/test_skip_tokenizer_init.py | 6 | GPU server (amd+cuda) | UC-1 |
| llm/test_chat.py | 5 | vllm_runner Llama-3.2-1B | UC-4 |
| sessions/test_session_control.py | 5 | GPU server (amd+cuda) | UC-1 |
| worker/test_prompt_embeds_state.py | 4 | importorskip triton + CUDA skip | UC-1 |
| sessions/test_session_latency.py | 3 | GPU server | UC-1 |
| tokenizer/test_multi_tokenizer.py | 2 | GPU server (amd+cuda) | UC-1 |
| `completion/test_token_in_token_out.py` | 1 | Qwen3-0.6B tokenizer-only | UC-4 |
| serve/tokenize/test_tokenization_vlm.py | 1 | RemoteOpenAIServer, Qwen2.5-VL-3B | UC-4 |
| serve/tokenize/test_tokenize_then_chat_vlm.py | 1 | RemoteOpenAIServer, Qwen2.5-VL-3B | UC-4 |
| tests/detokenizer/test_disable_detokenization.py | 1 | LLM weights | UC-1 |
| tests/detokenizer/test_stop_reason.py | 1 | vllm_model.llm generate | UC-4 |
| tests/detokenizer/test_stop_strings.py | 1 | meta-llama/llama-2-7b-hf | UC-4 |
| tests/watermarking/test_watermarking_e2e.py | 1 | opt-125m | UC-1 |
| sessions/test_streaming_session.py | 0 | GPU server (amd+cuda) | UC-1 |
| sessions/test_streaming_session_extra.py | 0 | GPU server (amd+cuda) | UC-1 |
| sessions/test_streaming_session_swa_extra.py | 0 | GPU server; CPU-CI registered | UC-1 |

## C5 — KV cache / prefix reuse / hostcache (126 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| kernels/ops/ | ~1022 | test_c128_v2.py, test_fp8_blockwise_gemm.py, test_fused_topk_deepseek.py | UC-1 |
| attention/unittests/ | ~224 | test_deepseek_v4.py(22), test_dsa.py(16), test_triton.py(11) | UC-1 |
| attention/ (top-level) | ~179 | test_chunk_gated_delta_rule.py(31), test_deepseek_v4.py(22) | UC-1 |
| `kernels/core/` (test_activation, test_apply_rotary_emb, test_batched_weight_rms_norm, test_cpu_activation, test_fused_allreduce_gemma_rms_norm, test_fused_embed_norm, test_fused_qk_norm_rope, test_fused_q_kv_rmsnorm, test_fused_quant_layernorm, test_fused_rms_norm_gated, test_fused_silu_mul_block_quant, test_layernorm, test_minimax_reduce_rms, test_mrope, test_opcheck, test_pos_encoding, test_rocm_aiter_ops, test_rocm_misc_ops, test_rotary_embedding, test_rotary_embedding_mla_cache_fused, test_uva, test_vit_bilinear_pos_embed, test_vit_fp8_attn, test_vit_fp8_quant, test_vit_fp8_scaling, test_vocab_parallel_embedding) | 94 | 35 skipif | UC-1 |
| `kernels/core/` (excl. activation-owned): test_apply_rotary_emb.py, test_batched_weight_rms_norm.py, test_fused_allreduce_gemma_rms_norm.py, test_fused_embed_norm.py, test_fused_qk_norm_rope.py, test_fused_q_kv_rmsnorm.py, test_fused_quant_layernorm.py, test_fused_rms_norm_gated.py, test_fused_silu_mul_block_quant.py, test_layernorm.py, test_minimax_reduce_rms.py, test_mrope.py, test_opcheck.py, test_pos_encoding.py, test_rocm_aiter_ops.py, test_rocm_misc_ops.py, test_rotary_embedding_mla_cache_fused.py, test_rotary_embedding.py, test_uva.py, test_vit_bilinear_pos_embed.py, test_vit_fp8_attn.py, test_vit_fp8_quant.py, test_vit_fp8_scaling.py, test_vocab_parallel_embedding.py | 81 | CUDA/triton; ROCm files; UVA pinned mem | UC-1 |
| python/sglang/kernels/aot/tests/test_*.py (27 files) | 71 | CUDA GPU, sgl_kernel | UC-1 |
| attention/test_sparse_mla_backends.py | 59 | cuda | UC-1 |
| `attention/` — Flash/FlashMLA/TRT-LLM/cutlass family: test_flash_attn.py, test_flashinfer.py, test_flashinfer_mla_decode.py, test_flashinfer_trtllm_attention.py, test_flashmla.py, test_flashmla_sparse.py, test_cascade_flash_attn.py, test_cutlass_mla_decode.py, test_deepgemm_attention.py, test_prefix_prefill.py, test_mha_attn.py, test_trtllm_kvfp8_dequant.py, test_mixed_causal_attn.py, test_attention.py | 59 | CUDA, SM90/100, flashinfer, flashmla, trtllm | UC-1 |
| `attention/` — selector & backend choice: test_attention_selector.py, test_rocm_attention_selector.py, test_use_trtllm_attention.py, test_amx_mla.py | 53 | CUDA/ROCm mocks; AMX tile CPU | UC-1 |
| `kernels/test_bf16_skinny_gemm.py` | 45 | CUDA | UC-1 |
| kernel/qsa/ | ~39 | test_qsa.py(33), test_qsa_strided_zero_fill.py | UC-1 |
| `kernels/test_ll_bf16_gemm.py` | 35 | CUDA | UC-1 |
| layers/attention/ — kernels (aiter_fp8_asm_gqa, dots_hybrid_backend, kda_helion_dispatcher, triton_{dense,mla}_prefill_gfx950, linear/kernels kda_{nvidia,ptx}) | 35 | cuda/hip/triton | UC-1 |
| manual/chunked_prefill/ | 33 | scripted runtime is in-process; e2e launch servers | UC-1 |
| `attention/` — Triton attention: test_triton_decode_attention.py, test_triton_prefill_attention.py, test_triton_unified_attention.py, test_triton_unified_attention_diffkv.py, test_merge_attn_states.py, test_lightning_attn.py, test_pack_unpack_triton.py | 27 | Triton, CUDA/ROCm | UC-1 |
| `kernels/test_mhc_kernels.py` | 27 | CUDA, tilelang/triton | UC-1 |
| tests/test-recurrent-state-rollback.cpp | 2 (+4 variants) | rollback, multi-seq split replay; nemotron-h/dsv4/kimi-k3 model variants (needs models) | UC-1 |
| vllm/tests/v1/kv_connector/unit/test_multi_connector.py | 23 | N-connector delegation: first-wins load, store-to-all, E2E LLM | UC-1 |
| python/sglang/srt/layers/attention/minimax_sparse_ops/tests/*.py (4 files) | 21 | CUDA kernels, fp8 | UC-1 |
| `attention/` — KV cache ops: test_cache.py | 18 | CUDA primary, triton | UC-1 |
| `kernels/test_compressor_kv_cache.py` | 18 | CUDA | UC-1 |
| `kernels/test_compressor_kv_cache.py` | 18 | 10 | UC-1 |
| attention/test_indexer_dcp_localize.py | 17 | cuda+cutedsl | UC-1 |
| attention/test_mla_backends.py | 15 | device_type | UC-1 |
| `kernels/test_engram.py` | 15 | CUDA (one CPU-side mask util) | UC-1 |
| kv_canary/test_self_e2e_*.py (12 files) | 15 | mock weights, GPU runner | UC-1 |
| attention/test_attention_backends.py | 14 | device_type | UC-1 |
| `kernels/test_fused_deepseek_v32_norm_rope.py` | 13 | CUDA | UC-1 |
| `kernels/test_fused_deepseek_v32_norm_rope.py` | 13 | 2 | UC-1 |
| attention/test_b12x.py | 12 | device-cap check | UC-1 |
| `kernels/test_fused_inv_rope_fp8_quant.py` | 12 | CUDA | UC-1 |
| attention/test_rocm_aiter_mla_fp8_decode_routing.py | 11 | rocm | UC-1 |
| radix_cache/unified_radix_tree/test_unified_radix_cache_kl_*.py (11 files) | 11 | dsv4, glm52, mamba, mimo, SWA, MiMo pins | UC-1 |
| `kernels/test_flex_attention.py` | 10 | torch ≥2.7, CUDA | UC-1 |
| `kernels/test_shuffle_rows.py` | 10 | CUDA | UC-1 |
| radix_cache/unified_radix_tree/linker/ (2 files) | 10 | dsv4 / glm52, GPU server | UC-1 |
| worker/test_kv_block_zeroer.py | 9 | all tests skipif CUDA | UC-1 |
| attention/test_deepseek_v4_swa_visible.py | 9 | cuda | UC-1 |
| attention/test_mla_context_chunks.py | 9 | cuda | UC-1 |
| attention/test_rocm_glm5next_sparse.py | 9 | rocm+triton | UC-1 |
| `kernels/test_fla_layernorm_guard.py` | 9 | CUDA triton | UC-1 |
| kernel/attention/ | ~9 | test_kda_fused_verify_backend.py, test_dsa_metadata_replay.py | UC-1 |
| attention/test_indexer_deepseek_v4_slot_mapping.py | 8 | cuda | UC-1 |
| `kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | 8 | CUDA | UC-1 |
| `kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | 8 | 3 | UC-1 |
| manual/test_{torch_flex_attention_backend,wave_attention_backend,triton_attention_rocm_mla,create_custom_4d_mask,two_batch_overlap,kda_spec_integration,kda_target_verify,triton_moe_wna16}.py | 8 | CUDA (one ROCm-mla); kernels GPU | UC-1 |
| chunked_prefill/test_scripted_core_1gpu.py | 7 | 1 GPU (amd+cuda) | UC-1 |
| worker/test_gpu_block_table.py | 6 | pytestmark skipif !is_cuda | UC-1 |
| worker/test_gpu_warmup_blocks.py | 6 | pytestmark skipif !is_cuda (accelerator sync) | UC-1 |
| attention/test_flashinfer_mla_sparse_sm90.py | 6 | cuda sm90 | UC-1 |
| `kernels/test_concat_mla_q.py` | 6 | CUDA | UC-1 |
| `kernels/test_fp32_router_gemm.py` | 6 | CUDA | UC-1 |
| `kernels/test_kpool_decode_update_batched.py` | 6 | CUDA | UC-1 |
| `kernels/test_concat_mla_q.py` | 6 | 0 | UC-1 |
| manual/attention/ | 6 | SM90+; most launch servers | UC-1 |
| vllm/tests/v1/kv_connector/unit/test_offloading_connector.py | 5 | E2E LLM offload with zmq events, block-size multiple constraint | UC-1 |
| tests/v1/e2e/test_replayssm_decode.py | 5 | Nemotron-3 4B; 40GB | UC-1 |
| attention/test_flashinfer_sparse_mla_sm120_api.py | 5 | cuda sm120 | UC-1 |
| `kernels/test_fused_indexer_q_rope_quant.py` | 5 | CUDA | UC-1 |
| `kernels/test_mhc_jit_warmup.py` | 5 | CUDA, tilelang | UC-1 |
| manual/mla/ | 5 | GPU, MLA models | UC-1 |
| vllm/tests/v1/simple_kv_offload/test_integration.py | 4 | real-model E2E offload correctness (CUDA/ROCm-gated) | UC-1 |
| tests/v1/e2e/general/test_mamba_prefix_cache.py | 4 | Qwen3-Next-80B; datasets | UC-4 |
| attention/test_flashinfer_mla_dcp.py | 4 | cuda+flashinfer-mla | UC-1 |
| `kernels/test_fused_gdn_post_conv.py` | 4 | CUDA | UC-1 |
| `kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py` | 4 | CUDA | UC-1 |
| tests/quantization/test_cpu_offload.py | 4 | nm-testing + Qwen AWQ pins | UC-4 |
| test_cuda_vmm_utils.py | 4 | cuda | UC-1 |
| test_flashinfer_sparse_mla.py | 4 | flashinfer | UC-1 |
| manual/4-gpu-models/ | 4 | 4 GPU, specific Qwen weights | UC-2 |
| vllm/tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py | 3 | multi-GPU LLM hidden-states extraction to file | UC-2 |
| attention/test_dspark_noncausal_sparse_mla.py | 3 | cuda | UC-1 |
| attention/test_sparse_mla_mask.py | 3 | cuda | UC-1 |
| `kernels/test_fused_recurrent_packed_decode.py` | 3 | CUDA | UC-1 |
| mem_cache/test_post_capture_kv_sizing.py | 3 | GPU server | UC-1 |
| manual/kernels/ | 3 | CUDA (hisparse); dispatch bench CPU-fine | UC-1 |
| manual/scheduler/ | 3 | launched servers | UC-1 |
| manual/test_{mori_transfer_engine_e2e,kv_events,forward_pass_metrics}.py | 3 | FPM/kv_events schema parts are CPU | UC-1 |
| kv_offload/cpu/test_swap_blocks_batch.py | 2 | CUDA-only kernel op | UC-1 |
| tests/v1/e2e/general/test_attention_backend_per_kind.py | 2 | gemma-3-1b; CUDA-only | UC-1 |

(46 more rows in UPSTREAM-INVENTORY.md)


## C6 — Spec decode (dSpark) (47 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| kernels/ (top-level *.py) | ~112 | test_verify_hand.py(56), test_model_fast_paths.py(50), test_plan_hand.py(45) | UC-1 |
| manual/dsv4/ | 21 | 8 GPU, DSv4 weights, sgl-eval on PATH | UC-1 |
| spec_decode/test_rejection_sampler_utils.py | 18 | module-level `skipif(not cuda)` + `importorskip triton` | UC-1 |
| attention/test_rocm_aiter_mla_mtp_split.py | 16 | rocm | UC-1 |
| `kernels/test_engram.py` | 15 | CUDA (one CPU-side mask util) | UC-1 |
| `kernels/test_engram.py` | 15 | CUDA (one CPU-side mask util) | UC-1 |
| jit/ (top-level) | ~11 | test_hisparse_spec.py(9), test_flux2_gated_resnorm.py | UC-1 |
| tests/v1/e2e/spec_decode/draft_model/test_draft_model.py | 10 | Qwen3-0.6B; 1-2 GPU | UC-4 |
| spec/dspark/test_dspark_{kernel_parity,stacked_ctx_kv_parity}.py + test_ragged_verify_backend_capability.py (3 files) | 9 | triton kernels, cuda | UC-1 |
| kernel/speculative/ | ~8 | test_dflash_domino.py(4), test_spec_kv_indices_grid.py | UC-1 |
| manual/test_{torch_flex_attention_backend,wave_attention_backend,triton_attention_rocm_mla,create_custom_4d_mask,two_batch_overlap,kda_spec_integration,kda_target_verify,triton_moe_wna16}.py | 8 | CUDA (one ROCm-mla); kernels GPU | UC-1 |
| unit/test_speculative.py | 7 | with/without draft, draft min/max, synth determinism, ignores target tokens, slot ctx not exceeded, ctx shift, parallel multi-request | UC-1 |
| spec_decode/test_acceptance_estimator.py | 6 | hardcodes `torch.device("cuda")`; pure tensor math | UC-1 |
| manual/attention/ | 6 | SM90+; most launch servers | UC-1 |
| spec/dflash/ (2 files) | 5 | GPU server, dflash pins | UC-1 |
| dllm/ | ~5 | test_dllm_batching_fdfo.py(2), test_llada2_mini_amd.py | UC-1 |
| spec/test_{constrained_decoding_spec_reasoning,frozen_kv_mtp,mixed_chunk,ngram,ngram_extra,standalone,standalone_extra}.py (7 files) | 4 | GPU server | UC-1 |
| tests/v1/e2e/spec_decode/ngram_suffix/test_ngram_suffix.py | 3 | single-GPU | UC-1 |
| spec_decode/test_dflash_prepare_inputs.py | 3 | module-level `skipif(not cuda)` | UC-1 |
| spec_decode/test_eagle_step_kernel.py | 3 | module-level skip unless CUDA/XPU; `importorskip triton` | UC-1 |
| spec_decode/test_max_len.py | 3 | `vllm_runner` fixture, real models | UC-5 |
| spec/ — eagle cuda-graph runner (eagle_draft_cuda_graph_runner) | 3 | cuda | UC-1 |
| tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py | 2 | gsm8k eval; single-GPU | UC-1 |
| tests/v1/e2e/spec_decode/eagle/test_eagle_correctness.py | 2 | single-GPU; gsm8k | UC-1 |
| tests/v1/e2e/spec_decode/mtp/test_mtp.py | 2 | single-GPU; gsm8k | UC-1 |
| tests/v1/e2e/spec_decode/test_mtp_parallel_load.py | 2 | multi-GPU marks | UC-2 |
| spec_decode/test_speculators_correctness.py | 2 | `LLM(...)`, `cleanup_dist_env_and_memory` | UC-1 |
| distributed/test_eplb_spec_decode.py | 2 | multi-GPU | UC-2 |
| spec/utils/test_build_eagle_tree.py | 2 | amd+cuda | UC-1 |
| manual/spec/ | 2 | GPU + draft models | UC-1 |
| tests/v1/e2e/spec_decode/acceptance_rates/dspark/test_dspark.py | 1 | gsm8k eval; CUDA | UC-1 |
| tests/v1/e2e/spec_decode/acceptance_rates/mtp_other/test_medusa.py | 1 | vicuna-7b medusa ckpt | UC-4 |
| tests/v1/e2e/spec_decode/acceptance_rates/mtp_other/test_mtp.py | 1 | vllm_runner; mrv2 param | UC-4 |
| tests/v1/e2e/spec_decode/acceptance_rates/mtp_other/test_synthetic.py | 1 | single-GPU; acceptance 1.875 | UC-1 |
| tests/v1/e2e/spec_decode/draft_model/test_async.py | 1 | eagle3 llama 1B | UC-4 |
| tests/v1/e2e/spec_decode/draft_model/test_lora.py | 1 | vllm_runner | UC-4 |
| tests/v1/e2e/spec_decode/eagle/test_eagle3_pp.py | 1 | multi-GPU(2); llama 1B | UC-2 |
| tests/v1/e2e/spec_decode/speculators/test_speculators.py | 1 | single-GPU | UC-1 |
| tests/v1/e2e/spec_decode/test_sharded_sampling.py | 1 | LLM; multi-GPU | UC-2 |
| worker/test_gpu_rejection_sampler_i64.py | 1 | ~5 GiB GPU per case; CUDA tensors | UC-1 |
| distributed/test_eagle_dp.py | 1 | DP engines + attn backends; SM90 gate | UC-1 |
| attention/test_flashinfer_dcp_spec_reorder.py | 1 | cuda | UC-1 |
| spec_decode/test_acceptance_length.py | 1 | VllmRunner, `Llama-3.2-1B` + eagle3, metrics API | UC-4 |
| spec_decode/test_speculators_eagle3.py | 1 | `vllm_runner`, `skipif` non-cuda-alike | UC-4 |
| core/test_basic_sanity_dspark.py | 0 | GPU server | UC-1 |
| core/test_basic_sanity_dflash.py | 0 | GPU server | UC-1 |
| python/sglang/kernels/aot/tests/speculative/*.py (3 files) | — | CUDA | UC-1 |

## C7 — MoE routing / expert placement (50 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| kernels/ops/ | ~1022 | test_c128_v2.py, test_fp8_blockwise_gemm.py, test_fused_topk_deepseek.py | UC-1 |
| `kernels/moe/` (conftest, test_b12x, test_batched_deepgemm, test_batched_moe, test_block_fp8, test_block_int8, test_count_expert_num_tokens, test_cpu_fused_moe, test_cpu_int4_moe, test_cpu_quant_fused_moe, test_cutedsl_moe, test_cutlass_moe, test_deepep_deepgemm_moe, test_deepep_moe, test_deepep_v2_async_finalize, test_deepep_v2_moe, test_deepgemm, test_flashinfer_b12x_moe, test_flashinfer_cutedsl_layout, test_flashinfer_cutedsl_nvfp4_moe, test_flashinfer_moe, test_flashinfer, test_flydsl_moe, test_fused_topk, test_gemma4router, test_gpt_oss_triton_kernels, test_grouped_topk, test_marlin_vs_trtllm_mxint4, test_modular_kernel_combinations, test_modular_oai_triton_moe, test_moe_align_block_size, test_moe_fused_mul_sum, test_moe_kernel_oracle, test_moe_layer, test_moe_permute_unpermute, test_moe, test_moe_weight_loading_padded, test_mxfp4_moe, test_mxfp8_aiter_backend_selection, test_nvfp4_moe, test_ocp_mx_moe, test_profile_modular_kernel, test_rocm_aiter_moe, test_rocm_aiter_topk, test_routed_experts_capture_monolithic, test_routing, test_routing_simulator, test_shared_fused_moe_routed_transform, test_silu_mul_fp8_quant_deep_gemm, test_silu_mul_per_token_group_quant_fp8_colmajor, test_situ_mul_fp8_quant, test_topk_softplus_sqrt, test_triton_moe_no_act_mul, test_triton_moe_ptpc_fp8, test_trtllm_bf16_moe, test_trtllm_nvfp4_moe, test_unquantized_backend_selection, test_zen_cpu_fused_moe, test_zen_cpu_int8_moe, test_zero_expert_moe; `modular_kernel_tools/`, `parallel_utils.py`) | 315 | 100 skipif | UC-1 |
| layers/moe/ — kernels+dispatchers (aiter_runner, deepep_v2_{buffer_lifecycle,masked_slab}, flashinfer_{a2a_wide_ep,dispatcher,megamoe}, fused_moe_{native,triton_config}, mega_moe_deepgemm_api, qwen35_flashinfer_fusion, w4afp8_deepep_{dtype,post_reorder}, w4afp8_requant_geometry) | 72 | cuda/hip | UC-1 |
| python/sglang/kernels/aot/tests/test_*.py (27 files) | 71 | CUDA GPU, sgl_kernel | UC-1 |
| model_executor/layers/test_fused_shared_expert.py | 38 | heavy mocks, rocm refs | UC-1 |
| `kernels/test_top_k_per_row.py` | 23 | 26 | UC-1 |
| tests/quantization/test_blackwell_moe.py | 21 | deepseek-ai/DeepSeek-V3.1, nvidia FP4 repos | UC-1 |
| manual/dsv4/ | 21 | 8 GPU, DSv4 weights, sgl-eval on PATH | UC-1 |
| kernel/jit/ | ~14 | test_fast_topk.py(7), test_hc_combine.py | UC-1 |
| manual/ep/ | 14 | multi-GPU, deep_ep/mooncake libs | UC-2 |
| manual/layers/ | 11 | CUDA/triton kernels; benches H100 | UC-1 |
| `kernels/test_shuffle_rows.py` | 10 | 0 | UC-1 |
| moe/test_topk_padded_region.py | 9 | amd+cuda kernels | UC-1 |
| manual/test_{get_weights_by_name,weight_cache_e2e,weight_loader_v2_equiv,expert_location_updater,expert_distribution,moe_quant_once,modelopt,modelopt_fp8kvcache}.py | 8 | GPU, multi-proc distributed some | UC-1 |
| manual/test_{torch_flex_attention_backend,wave_attention_backend,triton_attention_rocm_mla,create_custom_4d_mask,two_batch_overlap,kda_spec_integration,kda_target_verify,triton_moe_wna16}.py | 8 | CUDA (one ROCm-mla); kernels GPU | UC-1 |
| moe/test_fused_append_shared_experts.py | 7 | amd+cuda kernels | UC-1 |
| moe/test_topk_renormalize_degenerate.py | 7 | amd+cuda kernels | UC-1 |
| moe/test_cutedsl_moe.py | 6 | B200, cuda | UC-1 |
| moe/test_fused_append_remap_per_rank_shared_slots.py | 6 | amd+cuda kernels | UC-1 |
| distributed/test_eplb_execute.py | 5 | multi-GPU via eplb_utils | UC-2 |
| expert_pack/test_expert_pack_mxfp4.py | 4 | cuda | UC-1 |
| ep/test_deepep_large.py | 4 | multi-GPU, DeepEP | UC-2 |
| ep/test_deepep_small.py | 4 | multi-GPU, DeepEP | UC-2 |
| tests/v1/determinism/test_cutlass_batch_invariance.py | 3 | CUDA, cutlass, fp8/nvfp4, MoE kernels | UC-1 |
| moe/test_fused_append_shared_experts_top6.py | 3 | amd+cuda kernels | UC-1 |
| eplb/test_lplb_distributed.py | 3 | multi-GPU; supplements CPU test_lplb | UC-2 |
| ep/test_deepep_small_extra.py | 3 | multi-GPU, DeepEP | UC-2 |
| ep/test_flashinfer_a2a.py | 3 | multi-GPU | UC-2 |
| kernel/hyperconnection/ | ~3 | test_hc_mix_triton.py | UC-1 |
| kernel/moe/ | ~3 | test_jit_grouped_topk.py | UC-1 |
| distributed/test_eplb_spec_decode.py | 2 | multi-GPU | UC-2 |
| moe/test_fused_moe.py | 2 | cuda kernels | UC-1 |
| moe/test_hpc_ops_moe.py | 2 | hpc_ops backend | UC-1 |
| manual/distributed/ | 2 | multi-GPU, MLA models | UC-2 |
| distributed/test_dbo.py | 1 | RemoteOpenAIServer, deep_ep, 2+ GPUs | UC-1 |
| `test_return_routed_experts.py` | 1 | tiny-mixtral server (8 experts, top-2) | UC-1 |
| scale_out/token_in_token_out/test_return_routed_experts.py | 1 | RemoteOpenAIServer, tiny-mixtral | UC-1 |
| distributed/test_elastic_ep.py | 1 | multi-GPU | UC-2 |
| distributed/test_eplb_fused_moe_layer.py | 1 | multi-GPU | UC-2 |
| distributed/test_eplb_fused_moe_layer_dep_nvfp4.py | 1 | multi-GPU, NVFP4 | UC-2 |
| distributed/test_expert_parallel.py | 1 | multi-GPU | UC-2 |
| tests/quantization/test_experts_int8.py | 1 | HF_EXAMPLE_MODELS registry | UC-1 |
| moe/test_moe_ep.py | 1 | GPU server | UC-1 |
| moe/test_moe_ep_extra.py | 1 | GPU server | UC-1 |
| moe/test_triton_fused_moe.py | 1 | triton | UC-1 |
| moe/test_triton_moe_channel_fp8_kernel.py | 1 | triton FP8 | UC-1 |
| moe/test_zero_experts.py | 1 | cuda | UC-1 |
| ep/test_eplb_no_a2a.py | 1 | GPU server | UC-1 |
| ep/test_routed_experts_dp_readback.py | 1 | multi-GPU | UC-2 |
| ep/test_tbo_shared_experts_fusion.py | 1 | GPU server | UC-1 |

## C8 — Scheduler / admission / lifecycle (92 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| model_executor/ — cuda-graph runner tests (prefill_cuda_graph_runner{,_helpers}, cuda_graph_buffer_registry) | 63 | cuda | UC-1 |
| model_executor/runner{,_backend,_utils}/ (decode/prefill graph runners, shared_read_fence, flashinfer_autotune_sync, hidden_state_graph_recapture, graph_pool_borrow, full_cuda_graph_backend) | 61 | cpu_ci but graph-heavy | UC-1 |
| manual/chunked_prefill/ | 33 | scripted runtime is in-process; e2e launch servers | UC-1 |
| core/test_srt_endpoint.py | 28 | GPU server (amd+cuda) | UC-1 |
| prefill_only/test_multi_item_scoring.py | 20 | GPU server | UC-1 |
| prefill_only/test_pooled_hidden_states.py | 20 | GPU server (amd+cuda) | UC-1 |
| cudagraph/test_breakable_cudagraph.py | 19 | ~5 pure-CPU state tests; most need cuda_capture_stream | UC-1 |
| prefill_only/test_score_engine.py | 19 | GPU server | UC-1 |
| tests/v1/engine/test_async_llm.py | 18 | CUDA-only module skip | UC-1 |
| cuda_graph/breakable/test_breakable_cuda_graph.py | 15 | cuda (amd+cuda) | UC-1 |
| tests/v1/e2e/general/test_streaming_input.py | 14 | opt-125m | UC-1 |
| attention/test_rocm_attention_backends_selection.py | 14 | rocm | UC-1 |
| prefill_only/test_serving_rerank.py | 13 | GPU server (amd+cuda) | UC-1 |
| distributed/test_async_llm_dp.py | 12 | spawns DP engines; DP_SIZE=2 default | UC-1 |
| observability/test_tracing.py | 11 | GPU server (amd+cuda) | UC-1 |
| tests/v1/engine/test_engine_core.py | 10 | in-proc dummy cores; TP 2-GPU | UC-2 |
| prefill_only/test_openai_embedding.py | 10 | GPU server; also CPU-CI registered | UC-1 |
| observability/test_metrics.py | 10 | GPU server (amd+cuda) | UC-1 |
| cudagraph/test_cudagraph_dispatch.py | 8 | several tests skipif CUDA; spawn per test | UC-1 |
| prefill_only/test_score_api.py | 8 | GPU server | UC-1 |
| scheduler/test_scheduler_control.py | 8 | GPU server (amd+cuda) | UC-1 |
| mock_model/test_self_unit_canary_mock_wiring.py | 8 | amd+cuda | UC-1 |
| tests/v1/e2e/general/test_min_tokens.py | 7 | opt-125m small | UC-1 |
| launchers/test_shutdown.py | 7 | RemoteOpenAIServer, psutil | UC-1 |
| scheduler/test_prefill_delayer.py | 7 | GPU server | UC-1 |
| scheduler/test_priority_scheduling.py | 7 | GPU server (amd+cuda) | UC-1 |
| mock_model/test_self_unit_oracle.py | 7 | amd+cuda | UC-1 |
| observability/test_priority_metrics.py | 6 | GPU server; CPU-CI registered | UC-1 |
| core/test_srt_engine.py | 6 | GPU server (amd+cuda) | UC-1 |
| mock_model/test_e2e_{pd,pp,tp}.py + test_e2e_spec_eagle.py (4 files) | 6 | mock weights, GPU runner (amd+cuda) | UC-1 |
| backends/test_flashinfer_trtllm_gen_moe_backend.py | 6 | GPU server | UC-1 |
| attention/test_cuda_backend_probe_errors.py | 5 | cuda | UC-1 |
| core/test_hidden_states.py | 5 | cuda | UC-1 |
| cuda_graph/full_prefill/test_full_cuda_graph_prefill.py | 5 | GPU server | UC-1 |
| tests/v1/engine/test_llm_engine.py | 4 | opt-125m runner boot | UC-1 |
| serve/dev/rlhf/state_transitions/test_pause_resume.py | 4 | RemoteOpenAIServer, conftest above | UC-1 |
| scheduler/test_load_snapshot_server.py | 4 | GPU server (amd+cuda) | UC-1 |
| manual/test_{custom_allreduce,quick_allreduce,ray_engine,cross_node_scheduler_info_sync}.py | 4 | multi-GPU + Ray; cross-node 2 nodes | UC-2 |
| tests/v1/e2e/general/test_async_scheduling.py | 3 | VllmRunner; large-GPU 16GB | UC-4 |
| tests/v1/shutdown/test_delete.py | 3 | real model, TP2, wait_for_gpu_memory_to_clear | UC-2 |
| distributed/test_pipeline_parallel.py | 3 | multi-GPU, PP | UC-2 |
| batch_overlap/test_tbo_cuda_graph_num_token_device.py | 3 | cpu_ci | UC-1 |
| core/test_engine_child_pids.py | 3 | cuda | UC-1 |
| core/test_request_queue_validation.py | 3 | GPU server | UC-1 |
| mock_model/test_self_unit_install.py | 3 | amd+cuda | UC-1 |
| manual/scheduler/ | 3 | launched servers | UC-1 |
| tests/v1/e2e/general/test_context_length.py | 2 | VllmRunner; ValueError semantics | UC-4 |
| tests/v1/e2e/general/test_pooling_chunked_prefill.py | 2 | vllm_runner; CUDA-only | UC-4 |
| tests/v1/e2e/test_hybrid_chunked_prefill.py | 2 | Qwen3.5-4B; 30-80GB | UC-4 |
| tests/v1/metrics/test_engine_logger_apis.py | 2 | distilgpt2 engine boot | UC-1 |
| tests/v1/fault_tolerance/test_fault_tolerance_e2e.py | 2 | nixl_ep FT hardware gate | UC-1 |
| tests/v1/core/test_scheduler_e2e.py | 2 | full engine, model weights | UC-1 |
| tests/v1/streaming_input/test_gpu_model_runner_streaming.py | 2 | pinned (UVA) memory needs CUDA device | UC-1 |
| tests/v1/streaming_input/test_gpu_model_runner_v2_streaming.py | 2 | pinned (UVA) memory needs CUDA device | UC-1 |
| tests/v1/shutdown/test_forward_error.py | 2 | real Llama model, TP1/TP2 | UC-2 |
| tests/v1/shutdown/test_startup_error.py | 2 | real Llama model, TP1/TP2 | UC-2 |
| cudagraph/test_cudagraph_mode.py | 2 | LLM() with FA3/FA2/FlashInfer backends | UC-1 |
| `test_chunked_prompt.py` | 2 | Qwen3-0.6B server, `--enable-chunked-prefill` | UC-4 |
| prefill_only/test_embedding_models.py | 2 | GPU server (amd+cuda) | UC-1 |
| scheduler/test_retract_decode.py | 2 | GPU server (amd+cuda) | UC-1 |
| mock_model/test_self_unit_oracle_torch_vs_ref.py | 2 | amd+cuda | UC-1 |
| cuda_graph/piecewise/test_piecewise_cuda_graph_support_1_gpu.py | 2 | GPU server (amd+cuda) | UC-1 |
| backends/test_flashinfer_fusion_preflight.py | 2 | multi-GPU | UC-2 |
| kernel/cuda_graph/ | ~2 | test_cuda_graph_dedup.py | UC-1 |
| cpu/test_cpu_graph.py | ~2 | test_cpu_graph.py | UC-1 |
| tests/v1/engine/test_abort_final_step.py | 1 | CUDA-only module skip | UC-1 |
| tests/v1/engine/test_preprocess_error_handling.py | 1 | engine boot; fork | UC-1 |
| tests/v1/shutdown/test_processor_error.py | 1 | real model via AsyncEngineArgs | UC-1 |
| worker/test_jit_warmup_migration.py | 1 | skipif !is_cuda_alike | UC-1 |
| distributed/test_dense_dp_world_size.py | 1 | RemoteOpenAIServer, DPxTP GPUs | UC-2 |
| llm/test_gpu_utilization.py | 1 | 3× vllm_runner opt-125m | UC-4 |
| distributed/test_pp_cudagraph.py | 1 | multi-GPU | UC-2 |
| prefill_only/test_reward_models.py | 1 | GPU server (amd+cuda) | UC-1 |
| observability/test_encoder_server_metrics.py | 1 | GPU server (amd+cuda) | UC-1 |
| observability/test_tracing_disaggregation.py | 1 | 2-GPU | UC-2 |
| scheduler/test_retract_decode_logprob.py | 1 | GPU server (amd+cuda) | UC-1 |
| scheduler/test_routing_key_scheduling.py | 1 | GPU server; CPU-CI registered | UC-1 |
| core/test_gated_launch.py | 1 | cuda | UC-1 |
| core/test_no_extra_forked_cuda_context.py | 1 | GPU server | UC-1 |
| mock_model/test_self_e2e_perturb_next_token_swap.py | 1 | GPU runner (amd+cuda) | UC-1 |

(12 more rows in UPSTREAM-INVENTORY.md)


## C9 — Transport / RPC / failure injection (94 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| distributed/test_sharded_rdt_plan.py | 94 | large, spawns workers | UC-1 |
| distributed/test_sharded_rdt_producer.py | 64 | multi-GPU | UC-2 |
| rl/ | ~63 | test_weight_version_spans.py(17), test_weight_checker_e2e.py(9) | UC-1 |
| distributed/test_rocm_quick_reduce.py | 27 | ROCm GPU | UC-1 |
| distributed/test_sharded_rdt_trainer.py | 25 | multi-GPU | UC-2 |
| distributed/test_dcp_direct_a2a_lse_reduce.py | 21 | multi-GPU, DeepEP | UC-2 |
| distributed/test_packed_tensor.py | 21 | multi-GPU | UC-2 |
| distributed/test_dcp_a2a.py | 18 | multi-GPU, DeepEP | UC-2 |
| distributed/test_quick_all_reduce.py | 18 | multi-GPU | UC-2 |
| dcp/test_trtllm_mla_family_dcp_metadata.py | 18 | cuda-graph, MLA | UC-1 |
| distributed/test_shm_broadcast.py | 17 | multi-proc, shm | UC-1 |
| distributed/test_pynccl.py | 15 | multi-GPU, NCCL | UC-2 |
| distributed/test_ray_v2_executor.py | 15 | ray, multi-GPU | UC-2 |
| distributed/test_comm_ops.py | 14 | multi-GPU, tensor-parallel init | UC-2 |
| distributed/test_weight_transfer_nccl_uid.py | 14 | multi-GPU, NCCL | UC-2 |
| disaggregation/test_disaggregation_basic.py | 14 | 2-GPU server | UC-2 |
| manual/ep/ | 14 | multi-GPU, deep_ep/mooncake libs | UC-2 |
| distributed/test_multiproc_executor.py | 12 | multi-GPU, opt-125m | UC-2 |
| sgl-model-gateway/e2e_test/router/*.py | 12 | live models + eval harness | UC-1 |
| disaggregation/test_epd_disaggregation.py | 11 | multi-GPU, TBO | UC-2 |
| tests/v1/metrics/test_ray_metrics.py | 10 | ray.init; num_gpus=1 | UC-1 |
| scale_out/token_in_token_out/test_serving_tokens.py | 9 | RemoteOpenAIServer, Qwen3-0.6B | UC-4 |
| executor/test_executor.py | 8 | base/agg tests CPU; AsyncLLM/LLMEngine cases need GPU/model | UC-1 |
| distributed/test_mnnvl_alltoall.py | 8 | multi-node NVLink | UC-2 |
| disaggregation/test_disaggregation_different_tp.py | 8 | multi-GPU | UC-2 |
| `kernels/test_cp_gather_fp8.py` | 6 | CUDA | UC-1 |
| `kernels/test_cp_gather_fp8.py` | 6 | 0 | UC-1 |
| distributed/test_engram_dp_shard.py | 6 | multi-GPU | UC-2 |
| distributed/test_internal_lb_dp.py | 5 | 2-node server managers | UC-1 |
| scale_out/derender/test_derender_parity.py | 5 | RemoteOpenAIServer full GPU serve | UC-1 |
| compile/fusions_e2e/test_tp2_async_tp.py | 5 | 2 GPU, flashinfer, triton/fi attn | UC-1 |
| compile/passes/distributed/test_fusion_all_reduce.py | 5 | 2 GPU, cuda, fp4/aiter | UC-1 |
| distributed/test_custom_all_reduce.py | 5 | multi-GPU, P2P | UC-2 |
| distributed/test_flashinfer_pcie_ipc_all_reduce.py | 5 | multi-GPU, flashinfer | UC-2 |
| distributed/test_pp_dp_v2.py | 4 | requires 4 GPUs (DP2 PP2) | UC-2 |
| scale_out/token_in_token_out/test_serving_multimodal_tokens.py | 4 | RemoteOpenAIServer, Qwen3-VL-2B | UC-4 |
| state_capturer/test_routed_experts_scattered_a2a.py | 4 | cpu_ci, mocks | UC-1 |
| dp_attn/test_dp_attention_bcg_kl.py | 4 | BCG + GPU server | UC-1 |
| manual/test_{custom_allreduce,quick_allreduce,ray_engine,cross_node_scheduler_info_sync}.py | 4 | multi-GPU + Ray; cross-node 2 nodes | UC-2 |
| distributed/test_external_lb_dp.py | 3 | multiple server procs + openai client | UC-1 |
| distributed/test_hybrid_lb_dp.py | 3 | 4 DP ranks across 2 nodes | UC-1 |
| attention/test_dcp_a2a_pack_mask.py | 3 | cuda-alike | UC-1 |
| serve/dev/rpc/test_collective_rpc.py | 3 | RemoteOpenAIServer, Qwen3-0.6B | UC-4 |
| compile/correctness_e2e/test_sequence_parallel.py | 3 | cuda, tp | UC-2 |
| compile/fusions_e2e/test_tp2_ar_rms.py | 3 | 2 GPU, flashinfer | UC-1 |
| distributed/test_file_store.py | 3 | gloo, spawns processes | UC-1 |
| distributed/test_nccl_symm_mem.py | 3 | multi-GPU, NCCL | UC-2 |
| distributed/test_ray_v2_executor_e2e.py | 3 | ray, multi-GPU | UC-2 |
| distributed/test_split_group.py | 3 | torch.distributed spawns | UC-1 |
| disaggregation/test_disaggregation_decode_radix_cache.py | 3 | 2-GPU | UC-2 |
| disaggregation/test_disaggregation_optimistic_prefill.py | 3 | 2-GPU | UC-2 |
| cp/test_gqa_prefill_cp.py | 3 | GPU server | UC-1 |
| dcp/test_dsv31_dcp8_gsm8k.py | 3 | dsv3.1, multi-GPU | UC-2 |
| manual/test_{mori_transfer_engine_e2e,kv_events,forward_pass_metrics}.py | 3 | FPM/kv_events schema parts are CPU | UC-1 |
| tests/v1/fault_tolerance/test_fault_tolerance_e2e.py | 2 | nixl_ep FT hardware gate | UC-1 |
| scale_out/render/test_render_multimodal.py | 2 | RemoteOpenAIServer, Qwen3-VL-2B | UC-4 |
| compile/correctness_e2e/test_async_tp.py | 2 | cuda, flashinfer, tp2 | UC-2 |
| compile/passes/distributed/test_async_tp.py | 2 | 2 GPU, dist init | UC-1 |
| compile/passes/distributed/test_sequence_parallelism.py | 2 | 2 GPU, cuda | UC-1 |
| distributed/test_kimi_linear_context_parallel.py | 2 | multi-GPU | UC-2 |
| distributed/test_rocm_aiter_custom_ar.py | 2 | ROCm GPU | UC-1 |
| distributed/test_symm_mem_allreduce.py | 2 | multi-GPU | UC-2 |
| distributed/test_utils.py | 2 | ray, multi_gpu_test | UC-1 |
| disaggregation/test_disaggregation_pp.py | 2 | 2-GPU | UC-2 |
| dp_attn/test_dp_attention.py | 2 | 2-GPU server (amd+cuda) | UC-2 |
| manual/kv_transfer/ | 2 | mooncake lib, 2 nodes | UC-1 |
| sgl-model-gateway/e2e_test/benchmarks/*.py | 2 | live GPUs | UC-1 |
| ec_connector/integration/test_nixl_failure.py | 1 | 2 CUDA GPUs + nixl pkg | UC-1 |
| tests/v1/e2e/general/test_hisparse.py | 1 | forked procs; CUDA | UC-1 |
| worker/test_worker_memory_snapshot.py | 1 | spawns real TP=2 Workers, dummy weights | UC-2 |
| serve/instrumentator/test_uds.py | 1 | RemoteOpenAIServer, --uds | UC-1 |
| llm/test_collective_rpc.py | 1 | vllm_runner, torch.accelerator, tp1/2 | UC-2 |
| scale_out/token_in_token_out/test_return_routed_experts.py | 1 | RemoteOpenAIServer, tiny-mixtral | UC-1 |
| distributed/test_context_parallel.py | 1 | multi-GPU | UC-2 |
| distributed/test_custom_all_gather_reduce_scatter.py | 1 | multi-GPU | UC-2 |
| distributed/test_distributed_oot.py | 1 | multi-GPU | UC-2 |
| distributed/test_elastic_ep.py | 1 | multi-GPU | UC-2 |
| distributed/test_multi_node_assignment.py | 1 | gloo spawns | UC-1 |
| distributed/test_torchrun_example.py | 1 | torchrun, multi-GPU | UC-2 |
| distributed/test_torchrun_example_moe.py | 1 | torchrun, multi-GPU | UC-2 |

(14 more rows in UPSTREAM-INVENTORY.md)


## C? — other (16 rows)

| upstream path | n | pins | likely UC |
|---|---|---|---|
| tests/test-backend-ops.cpp | ~200 | per-op golden tests over all backends (unary/binary/cmul/conv/rope/flash-attn…) | UC-1 |
| utils/test_phase_checker.py | 32 | amd+cuda | UC-1 |
| language/pooling/* (23: conftest, embed_utils, 21 pooling tests) | 23 | GPU for runner tests | UC-1 |
| tests/test-llama-archs.cpp | ~19 | per-arch (gguf vocab models) backend vs CPU golden | UC-1 |
| quantization/* (10 tests: awq, fp8, fp8_per_channel, gpt_oss, gptq_marlin, modelopt, mxfp4, mxfp8, nvfp4, per_token_kv_cache) | 10 | quant-capable GPU | UC-1 |
| tests/utils_/test_gpu_sync_debug.py | 9 | CUDA | UC-1 |
| transformers/test_backend.py + fusers/* (linear, mla, moe, rms_norm) | 5 | torch CPU mostly | UC-1 |
| tests/cuda/test_cuda_context.py | 5 | libcuda, GPU | UC-1 |
| tests/test-fusion.cpp | 4 | prefill/decode fused vs unfused NMSE | UC-1 |
| tests/utils_/test_mem_utils.py | 3 | CUDA, CudaRTLibrary | UC-1 |
| tests/jit_monitor/test_hooks_gpu.py | 3 | CUDA + triton | UC-1 |
| tests/cuda/test_platform_no_cuda_init.py | 2 | libcuda present | UC-1 |
| test_initialization.py | 1 | GPU for init | UC-1 |
| tests/test-autorelease.cpp | 1 | autorelease pool smoke | UC-1 |
| tests/test-model-load-cancel.cpp | 1 | interrupt mid-load, no crash | UC-1 |
| tests/test-thread-safety.cpp | 1 | parallel decode smoke, ngl 99 | UC-1 |
