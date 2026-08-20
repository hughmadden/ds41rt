use std::collections::{BTreeMap, BTreeSet};

use ds4rt_core::{DType, KvCacheConfig, LayerId, TensorCatalog, TensorRole};

use super::types::{RealFullAttentionKvBindingDryRun, RealFullAttentionKvIoDryRun};

const COMMON_ATTENTION_SUFFIXES: [&str; 7] = [
    ".self_attn.q_a_proj.weight",
    ".self_attn.q_a_layernorm.weight",
    ".self_attn.q_b_proj.weight",
    ".self_attn.kv_a_proj_with_mqa.weight",
    ".self_attn.kv_a_layernorm.weight",
    ".self_attn.kv_b_proj.weight",
    ".self_attn.o_proj.weight",
];
const DSA_INDEXER_SUFFIXES: [&str; 5] = [
    ".self_attn.indexer.k_norm.bias",
    ".self_attn.indexer.k_norm.weight",
    ".self_attn.indexer.weights_proj.weight",
    ".self_attn.indexer.wk.weight",
    ".self_attn.indexer.wq_b.weight",
];
const DEEPSEEK_V4_ATTENTION_SUFFIXES: [&str; 13] = [
    ".attn.attn_sink",
    ".attn.kv_norm.weight",
    ".attn.q_norm.weight",
    ".attn.wkv.scale",
    ".attn.wkv.weight",
    ".attn.wo_a.scale",
    ".attn.wo_a.weight",
    ".attn.wo_b.scale",
    ".attn.wo_b.weight",
    ".attn.wq_a.scale",
    ".attn.wq_a.weight",
    ".attn.wq_b.scale",
    ".attn.wq_b.weight",
];
const DEEPSEEK_V4_INDEXER_SUFFIXES: [&str; 7] = [
    ".attn.indexer.compressor.ape",
    ".attn.indexer.compressor.norm.weight",
    ".attn.indexer.compressor.wgate.weight",
    ".attn.indexer.compressor.wkv.weight",
    ".attn.indexer.weights_proj.weight",
    ".attn.indexer.wq_b.scale",
    ".attn.indexer.wq_b.weight",
];

#[derive(Debug, Default)]
struct AttentionLayerStats {
    common_suffixes: BTreeSet<&'static str>,
    indexer_suffixes: BTreeSet<&'static str>,
    tensor_count: usize,
    bf16_tensor_count: usize,
    byte_count: u64,
}

pub(super) fn real_full_attention_kv_binding_dry_run(
    catalog: &TensorCatalog,
    kv_config: &KvCacheConfig,
    kv_io: &RealFullAttentionKvIoDryRun,
) -> RealFullAttentionKvBindingDryRun {
    let layer_count = catalog.facts.num_hidden_layers;
    let native_deepseek_v4 = catalog.facts.model_type == "deepseek_v4";
    let required_attention_suffixes: &[&'static str] = if native_deepseek_v4 {
        &DEEPSEEK_V4_ATTENTION_SUFFIXES
    } else {
        &COMMON_ATTENTION_SUFFIXES
    };
    let required_indexer_suffixes: &[&'static str] = if native_deepseek_v4 {
        &DEEPSEEK_V4_INDEXER_SUFFIXES
    } else {
        &DSA_INDEXER_SUFFIXES
    };
    let mut stats_by_layer = BTreeMap::<usize, AttentionLayerStats>::new();
    let mut attention_tensors = 0_usize;
    let mut bf16_attention_tensors = 0_usize;
    let mut common_attention_tensors = 0_usize;
    let mut indexer_attention_tensors = 0_usize;
    let mut attention_tensor_bytes = 0_u64;

    for tensor in &catalog.tensors {
        if tensor.role != TensorRole::Attention && tensor.role != TensorRole::AttentionIndexer {
            continue;
        }
        let Some(layer_id) = tensor.layer_id.map(|layer_id| layer_id as usize) else {
            continue;
        };
        if layer_id >= layer_count {
            continue;
        }
        let stats = stats_by_layer.entry(layer_id).or_default();
        if tensor.role == TensorRole::Attention {
            stats.tensor_count += 1;
            stats.byte_count += tensor.byte_length;
            attention_tensors += 1;
            attention_tensor_bytes += tensor.byte_length;
            if tensor.dtype == DType::Bf16 {
                stats.bf16_tensor_count += 1;
                bf16_attention_tensors += 1;
            }
            if let Some(suffix) = matching_suffix(&tensor.name, required_attention_suffixes) {
                stats.common_suffixes.insert(suffix);
                common_attention_tensors += 1;
            }
        }
        if let Some(suffix) = matching_suffix(&tensor.name, required_indexer_suffixes) {
            stats.indexer_suffixes.insert(suffix);
            indexer_attention_tensors += 1;
        }
    }

    // This report audits the 78 target layers; layer 78's MTP-only indexer is
    // accounted for by the serving KV configuration and execution path.
    let config_dsa_layer_ids = kv_config
        .dsa_indexer_layer_ids()
        .iter()
        .copied()
        .filter(|layer_id| *layer_id < layer_count)
        .collect::<Vec<_>>();
    let config_dsa_layer_set = config_dsa_layer_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let catalog_dsa_layer_ids = stats_by_layer
        .iter()
        .filter_map(|(layer_id, stats)| (!stats.indexer_suffixes.is_empty()).then_some(*layer_id))
        .collect::<Vec<_>>();
    let catalog_dsa_layer_set = catalog_dsa_layer_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

    let mut common_layers_with_required_tensors = 0_usize;
    let mut dsa_indexer_layers_with_required_tensors = 0_usize;
    let mut non_dsa_layers_without_indexer_tensors = 0_usize;
    let mut attention_layers = 0_usize;

    for layer_id in 0..layer_count {
        let Some(stats) = stats_by_layer.get(&layer_id) else {
            continue;
        };
        attention_layers += 1;
        common_layers_with_required_tensors +=
            usize::from(stats.common_suffixes.len() == required_attention_suffixes.len());
        if config_dsa_layer_set.contains(&layer_id) {
            dsa_indexer_layers_with_required_tensors +=
                usize::from(stats.indexer_suffixes.len() == required_indexer_suffixes.len());
        } else {
            non_dsa_layers_without_indexer_tensors +=
                usize::from(stats.indexer_suffixes.is_empty());
        }
    }

    let kv_layer_bytes_sum = (0..layer_count)
        .map(|layer_id| kv_config.layer_bytes_per_token(LayerId(layer_id as u32)))
        .sum::<usize>();
    let non_dsa_layer_id = (0..layer_count)
        .find(|layer_id| !kv_config.layer_has_dsa_indexer(LayerId(*layer_id as u32)))
        .unwrap_or_default();
    let dsa_layer_id = config_dsa_layer_ids.first().copied().unwrap_or_default();
    let config_dsa_layer_count = config_dsa_layer_ids.len();
    RealFullAttentionKvBindingDryRun {
        status: "attention-kv-binding-dry-run",
        scope: "bind real full-model attention tensor coverage to compressed KV layer-byte accounting and LayerWave KV I/O",
        attention_layers,
        attention_tensors,
        bf16_attention_tensors,
        common_attention_tensors,
        indexer_attention_tensors,
        attention_tensor_bytes,
        common_layers_with_required_tensors,
        dsa_indexer_layers: catalog_dsa_layer_ids.len(),
        dsa_indexer_layers_with_required_tensors,
        non_dsa_layers: layer_count - config_dsa_layer_count,
        non_dsa_layers_without_indexer_tensors,
        catalog_dsa_indexer_layer_ids: catalog_dsa_layer_ids,
        config_dsa_indexer_layer_ids: config_dsa_layer_ids,
        catalog_dsa_indexer_layers_match_kv_config: catalog_dsa_layer_set == config_dsa_layer_set,
        dsa_layer_bytes_per_token: kv_config.layer_bytes_per_token(LayerId(dsa_layer_id as u32)),
        non_dsa_layer_bytes_per_token: kv_config
            .layer_bytes_per_token(LayerId(non_dsa_layer_id as u32)),
        kv_bytes_per_token: kv_config.bytes_per_token(),
        kv_layer_bytes_sum,
        kv_io_layer_count: kv_io.layer_count,
        kv_io_prefill_writes: kv_io.prefix_prefill_wave_writes + kv_io.later_prefill_wave_writes,
        kv_io_decode_writes: kv_io.decode_wave_writes,
        kv_io_tentative_mtp_writes: kv_io.mtp_tentative_wave_writes,
        kv_io_prefix_read_blocks: kv_io.later_prefill_prefix_read_blocks
            + kv_io.decode_prefix_read_blocks
            + kv_io.mtp_prefix_read_blocks,
        kv_io_backed_bytes_after_discard: kv_io.backed_bytes_after_discard,
        all_attention_layers_bound_to_kv: attention_layers == layer_count
            && common_layers_with_required_tensors == layer_count
            && dsa_indexer_layers_with_required_tensors == config_dsa_layer_count
            && non_dsa_layers_without_indexer_tensors
                == layer_count - config_dsa_layer_count
            && (native_deepseek_v4 || bf16_attention_tensors == attention_tensors)
            && catalog_dsa_layer_set == config_dsa_layer_set
            && kv_layer_bytes_sum == kv_config.bytes_per_token()
            && kv_io.layer_count == layer_count,
    }
}

fn matching_suffix(name: &str, suffixes: &[&'static str]) -> Option<&'static str> {
    suffixes
        .iter()
        .copied()
        .find(|suffix| name.ends_with(suffix))
}
