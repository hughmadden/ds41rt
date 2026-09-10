use ds41rt_core::{
    deterministic_tiny_completion, plan_prefill_chunks, DecodeStep, KvCacheAllocator,
    KvCacheConfig, LayerWave, ModelFacts, PrefillChunkPolicy, Priority,
};
use std::time::Instant;

use crate::metrics::BackendMetrics;
use crate::{duration_ms, BackendCompletion};

fn tiny_kv_cache_config(max_tokens: usize) -> KvCacheConfig {
    KvCacheConfig::deepseek_v4_hybrid_bf16(max_tokens, &ModelFacts::default())
}

pub(crate) fn tiny_backend_completion(
    prompt: &str,
    prompt_tokens: usize,
    max_tokens: usize,
) -> BackendCompletion {
    let prefill_tokens = prompt_tokens.max(1);
    let mut kv_allocator = KvCacheAllocator::new(tiny_kv_cache_config(prefill_tokens + 1));
    let kv_reservation_id = kv_allocator
        .reserve("tiny", prefill_tokens + 1)
        .expect("tiny prefill reservation fits in tiny KV allocator");
    let policy = PrefillChunkPolicy::latency_smoke(16);

    let prefill_start = Instant::now();
    let prefill_rows = plan_prefill_chunks(
        "tiny-prefill",
        "tiny",
        0,
        prefill_tokens,
        kv_reservation_id,
        Priority(10),
        &policy,
        "deepseek-v4-tiny",
    )
    .into_iter()
    .map(LayerWave::prefill)
    .map(|wave| wave.num_rows())
    .sum::<usize>();
    let prefill_ms = duration_ms(prefill_start.elapsed());

    let decode_start = Instant::now();
    let decode_wave = LayerWave::decode(DecodeStep::new(
        "tiny-decode",
        "tiny",
        0,
        prefill_tokens as u64,
        Some(kv_reservation_id),
        Priority(0),
        "deepseek-v4-tiny",
    ));
    let content = deterministic_tiny_completion(prompt, max_tokens);
    let decode_ms = duration_ms(decode_start.elapsed());

    BackendCompletion {
        content,
        reasoning_content: None,
        completion_tokens: None,
        stream_chunks: None,
        metrics: BackendMetrics {
            cache_load_ms: 0.0,
            prefill_ms,
            time_to_first_token_ms: None,
            decode_ms,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            prefill_tokens,
            prefill_chunk_count: prefill_rows.div_ceil(policy.chunk_tokens),
            layerwave_prefill_rows: prefill_rows,
            layerwave_decode_rows: decode_wave.num_rows(),
            real_full: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::tiny_kv_cache_config;
    use ds41rt_core::{KvLayout, MlaKvCacheRepresentation, ModelFacts};

    #[test]
    fn tiny_backend_uses_deepseek_v4_flash_kv_geometry() {
        let facts = ModelFacts::default();
        let config = tiny_kv_cache_config(257);

        assert_eq!(config.layout, KvLayout::DeepseekV4HybridBf16);
        assert_eq!(config.layers, facts.num_hidden_layers);
        assert_eq!(config.key_value_width, facts.head_dim);
        assert_eq!(config.dsa_index_head_dim, facts.index_head_dim);
        assert_eq!(
            config.mla_representation,
            MlaKvCacheRepresentation::NormalizedRotated
        );
        assert_eq!(config.max_tokens, 257);
    }
}
