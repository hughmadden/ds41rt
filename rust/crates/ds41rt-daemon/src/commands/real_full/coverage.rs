use ds41rt_core::{ModelVariant, TensorCatalog, TensorRole};
use std::collections::BTreeMap;

use super::types::FullModelTensorCoverage;

pub(super) fn tensor_coverage(catalog: &TensorCatalog) -> FullModelTensorCoverage {
    let mut layers_with_any_tensor = BTreeMap::new();
    let mut sparse_layers_with_routed_experts = BTreeMap::new();
    let mut dense_layers_with_dense_mlp = BTreeMap::new();
    let mut coverage = FullModelTensorCoverage {
        hidden_layers_with_any_tensor: 0,
        sparse_layers_with_routed_experts: 0,
        dense_layers_with_dense_mlp: 0,
        routed_expert_tensors: 0,
        routed_quant_metadata_tensors: 0,
        attention_tensors: 0,
        router_tensors: 0,
        shared_expert_tensors: 0,
        embedding_tensors: 0,
        lm_head_tensors: 0,
    };
    for tensor in &catalog.tensors {
        if let Some(layer_id) = tensor.layer_id {
            if (layer_id as usize) < catalog.facts.num_hidden_layers {
                layers_with_any_tensor.insert(layer_id, ());
            }
        }
        match tensor.role {
            TensorRole::RoutedExpert => {
                coverage.routed_expert_tensors += 1;
                if tensor.is_quantization_metadata {
                    coverage.routed_quant_metadata_tensors += 1;
                }
                if let Some(layer_id) = tensor
                    .layer_id
                    .filter(|layer_id| (*layer_id as usize) < catalog.facts.num_hidden_layers)
                {
                    sparse_layers_with_routed_experts.insert(layer_id, ());
                }
            }
            TensorRole::DenseMlp => {
                if let Some(layer_id) = tensor
                    .layer_id
                    .filter(|layer_id| (*layer_id as usize) < catalog.facts.num_hidden_layers)
                {
                    dense_layers_with_dense_mlp.insert(layer_id, ());
                }
            }
            TensorRole::Attention => coverage.attention_tensors += 1,
            TensorRole::Router => coverage.router_tensors += 1,
            TensorRole::SharedExpert => coverage.shared_expert_tensors += 1,
            TensorRole::Embedding => coverage.embedding_tensors += 1,
            TensorRole::LmHead => coverage.lm_head_tensors += 1,
            _ => {}
        }
    }
    coverage.hidden_layers_with_any_tensor = layers_with_any_tensor.len();
    coverage.sparse_layers_with_routed_experts = sparse_layers_with_routed_experts.len();
    coverage.dense_layers_with_dense_mlp = dense_layers_with_dense_mlp.len();
    coverage
}

pub(in crate::commands::real_full) fn catalog_supports_default_sparse_router_routed_nvfp4(
    catalog: &TensorCatalog,
) -> bool {
    let coverage = SparseExpertCatalogCoverage::from_catalog(catalog);
    coverage.supports_sparse_router_routed_nvfp4()
}

pub(in crate::commands::real_full) fn catalog_supports_sparse_routed_and_shared_experts(
    catalog: &TensorCatalog,
) -> bool {
    let coverage = SparseExpertCatalogCoverage::from_catalog(catalog);
    coverage.supports_sparse_router_routed_nvfp4() && coverage.supports_sparse_shared_experts()
}

pub(in crate::commands::real_full) fn catalog_supports_full_model_execution_tensors(
    catalog: &TensorCatalog,
) -> bool {
    catalog
        .tensors
        .iter()
        .any(|tensor| tensor.role == TensorRole::LmHead)
        && DensePrefixCatalogCoverage::from_catalog(catalog).supports_dense_prefix()
        && catalog_supports_sparse_routed_and_shared_experts(catalog)
}

struct DensePrefixCatalogCoverage {
    first_sparse_layer: usize,
    post_attention_norm_by_layer: BTreeMap<u32, usize>,
    dense_mlp_by_layer: BTreeMap<u32, usize>,
}

impl DensePrefixCatalogCoverage {
    fn from_catalog(catalog: &TensorCatalog) -> Self {
        let mut coverage = Self {
            first_sparse_layer: catalog.facts.first_k_dense_replace,
            post_attention_norm_by_layer: BTreeMap::new(),
            dense_mlp_by_layer: BTreeMap::new(),
        };

        for tensor in &catalog.tensors {
            match tensor.role {
                TensorRole::Norm if tensor.name.ends_with(".post_attention_layernorm.weight") => {
                    if let Some(layer_id) = tensor.layer_id {
                        *coverage
                            .post_attention_norm_by_layer
                            .entry(layer_id)
                            .or_default() += 1;
                    }
                }
                TensorRole::DenseMlp => {
                    if let Some(layer_id) = tensor.layer_id {
                        *coverage.dense_mlp_by_layer.entry(layer_id).or_default() += 1;
                    }
                }
                _ => {}
            }
        }

        coverage
    }

    fn supports_dense_prefix(&self) -> bool {
        (0..self.first_sparse_layer as u32).all(|layer_id| {
            self.post_attention_norm_by_layer
                .get(&layer_id)
                .copied()
                .unwrap_or(0)
                >= 1
                && self.dense_mlp_by_layer.get(&layer_id).copied().unwrap_or(0) >= 3
        })
    }
}

struct SparseExpertCatalogCoverage {
    first_sparse_layer: usize,
    num_hidden_layers: usize,
    routed_experts: usize,
    routed_quant_metadata_per_expert: usize,
    shared_by_layer: BTreeMap<u32, usize>,
    router_by_layer: BTreeMap<u32, usize>,
    routed_weight_by_expert: BTreeMap<(u32, u32), usize>,
    routed_quant_metadata_by_expert: BTreeMap<(u32, u32), usize>,
}

impl SparseExpertCatalogCoverage {
    fn from_catalog(catalog: &TensorCatalog) -> Self {
        let mut coverage = Self {
            first_sparse_layer: catalog.facts.first_k_dense_replace,
            num_hidden_layers: catalog.facts.num_hidden_layers,
            routed_experts: catalog.facts.routed_experts,
            routed_quant_metadata_per_expert: if catalog.facts.variant == ModelVariant::Flash
                && catalog.facts.quantization_recipe == "deepseek_v4_native_fp4_fp8_mixed_v1"
            {
                3
            } else {
                9
            },
            shared_by_layer: BTreeMap::new(),
            router_by_layer: BTreeMap::new(),
            routed_weight_by_expert: BTreeMap::new(),
            routed_quant_metadata_by_expert: BTreeMap::new(),
        };

        for tensor in &catalog.tensors {
            match tensor.role {
                TensorRole::SharedExpert => {
                    if let Some(layer_id) = tensor.layer_id {
                        *coverage.shared_by_layer.entry(layer_id).or_default() += 1;
                    }
                }
                TensorRole::Router => {
                    if let Some(layer_id) = tensor.layer_id {
                        *coverage.router_by_layer.entry(layer_id).or_default() += 1;
                    }
                }
                TensorRole::RoutedExpert if tensor.is_quantization_metadata => {
                    if let (Some(layer_id), Some(expert_id)) = (tensor.layer_id, tensor.expert_id) {
                        *coverage
                            .routed_quant_metadata_by_expert
                            .entry((layer_id, expert_id))
                            .or_default() += 1;
                    }
                }
                TensorRole::RoutedExpert => {
                    if let (Some(layer_id), Some(expert_id)) = (tensor.layer_id, tensor.expert_id) {
                        *coverage
                            .routed_weight_by_expert
                            .entry((layer_id, expert_id))
                            .or_default() += 1;
                    }
                }
                _ => {}
            }
        }

        coverage
    }

    fn supports_sparse_router_routed_nvfp4(&self) -> bool {
        for layer_id in self.first_sparse_layer as u32..self.num_hidden_layers as u32 {
            if self.router_by_layer.get(&layer_id).copied().unwrap_or(0) < 2 {
                return false;
            }
            for expert_id in 0..self.routed_experts as u32 {
                let expert_key = (layer_id, expert_id);
                if self
                    .routed_weight_by_expert
                    .get(&expert_key)
                    .copied()
                    .unwrap_or(0)
                    < 3
                {
                    return false;
                }
                if self
                    .routed_quant_metadata_by_expert
                    .get(&expert_key)
                    .copied()
                    .unwrap_or(0)
                    < self.routed_quant_metadata_per_expert
                {
                    return false;
                }
            }
        }
        true
    }

    fn supports_sparse_shared_experts(&self) -> bool {
        (self.first_sparse_layer as u32..self.num_hidden_layers as u32)
            .all(|layer_id| self.shared_by_layer.get(&layer_id).copied().unwrap_or(0) >= 3)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        catalog_supports_default_sparse_router_routed_nvfp4,
        catalog_supports_full_model_execution_tensors,
        catalog_supports_sparse_routed_and_shared_experts,
    };
    use ds41rt_core::{
        DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole, DS4_FLASH_NUM_HIDDEN_LAYERS,
        DS4_FLASH_ROUTED_EXPERTS,
    };
    use ds41rt_loader::DEEPSEEK_V4_EXL3_RECIPE;

    const DS4_FLASH_FIRST_K_DENSE_REPLACE: usize = 0;

    #[test]
    fn default_real_probe_coverage_gates_require_complete_tensor_families() {
        let mut catalog = full_default_real_probe_catalog();
        assert!(catalog_supports_full_model_execution_tensors(&catalog));
        assert!(catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));
        assert!(catalog_supports_sparse_routed_and_shared_experts(&catalog));
        assert!(catalog_supports_full_model_execution_tensors(&catalog));

        let sparse_layer = DS4_FLASH_FIRST_K_DENSE_REPLACE as u32;
        let missing_quant = remove_first_matching(&mut catalog, |tensor| {
            tensor.role == TensorRole::RoutedExpert
                && tensor.is_quantization_metadata
                && tensor.layer_id == Some(sparse_layer)
                && tensor.expert_id == Some(0)
        });
        assert!(!catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));
        assert!(!catalog_supports_sparse_routed_and_shared_experts(&catalog));
        assert!(!catalog_supports_full_model_execution_tensors(&catalog));
        catalog.tensors.push(missing_quant);
        assert!(catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));

        let missing_shared = remove_first_matching(&mut catalog, |tensor| {
            tensor.role == TensorRole::SharedExpert && tensor.layer_id == Some(sparse_layer)
        });
        assert!(catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));
        assert!(!catalog_supports_sparse_routed_and_shared_experts(&catalog));
        assert!(!catalog_supports_full_model_execution_tensors(&catalog));
        catalog.tensors.push(missing_shared);
        assert!(catalog_supports_sparse_routed_and_shared_experts(&catalog));

        assert!(catalog_supports_sparse_routed_and_shared_experts(&catalog));
        assert!(catalog_supports_full_model_execution_tensors(&catalog));

        let missing_lm_head =
            remove_first_matching(&mut catalog, |tensor| tensor.role == TensorRole::LmHead);
        assert!(!catalog_supports_full_model_execution_tensors(&catalog));
        catalog.tensors.push(missing_lm_head);
        assert!(catalog_supports_full_model_execution_tensors(&catalog));
    }

    #[test]
    fn canonical_exl3_trellises_satisfy_routed_weight_coverage() {
        let mut facts = ModelFacts::default();
        facts.num_hidden_layers = 1;
        facts.dspark_target_layer_ids.clear();
        facts.routed_experts = 1;
        facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
        let mut tensors = (0..2)
            .map(|_| tensor("router.weight", TensorRole::Router, Some(0), None, false))
            .collect::<Vec<_>>();
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for suffix in ["trellis", "suh", "svh", "mcg"] {
                tensors.push(tensor(
                    &format!("model.layers.0.mlp.experts.0.{projection}.{suffix}"),
                    TensorRole::RoutedExpert,
                    Some(0),
                    Some(0),
                    suffix != "trellis",
                ));
            }
        }
        let mut catalog = TensorCatalog {
            model_id: "tpurtell/test-canonical-exl3-coverage".to_owned(),
            snapshot_path: "/tmp/test-canonical-exl3-coverage".to_owned(),
            facts,
            tensors,
        };

        assert!(catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));
        catalog
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name.ends_with("gate_proj.trellis"))
            .unwrap()
            .is_quantization_metadata = true;
        assert!(!catalog_supports_default_sparse_router_routed_nvfp4(
            &catalog
        ));
    }

    fn full_default_real_probe_catalog() -> TensorCatalog {
        let mut tensors = vec![tensor(
            "lm_head.weight",
            TensorRole::LmHead,
            None,
            None,
            false,
        )];
        for layer_id in 0..DS4_FLASH_FIRST_K_DENSE_REPLACE as u32 {
            tensors.push(tensor(
                &format!("model.layers.{layer_id}.post_attention_layernorm.weight"),
                TensorRole::Norm,
                Some(layer_id),
                None,
                false,
            ));
            for _ in 0..3 {
                tensors.push(tensor(
                    "dense.weight",
                    TensorRole::DenseMlp,
                    Some(layer_id),
                    None,
                    false,
                ));
            }
        }
        for layer_id in DS4_FLASH_FIRST_K_DENSE_REPLACE as u32..DS4_FLASH_NUM_HIDDEN_LAYERS as u32 {
            for _ in 0..2 {
                tensors.push(tensor(
                    "router.weight",
                    TensorRole::Router,
                    Some(layer_id),
                    None,
                    false,
                ));
            }
            for _ in 0..3 {
                tensors.push(tensor(
                    "shared.weight",
                    TensorRole::SharedExpert,
                    Some(layer_id),
                    None,
                    false,
                ));
            }
            for expert_id in 0..DS4_FLASH_ROUTED_EXPERTS as u32 {
                for _ in 0..3 {
                    tensors.push(tensor(
                        "routed.weight",
                        TensorRole::RoutedExpert,
                        Some(layer_id),
                        Some(expert_id),
                        false,
                    ));
                }
                for _ in 0..3 {
                    tensors.push(tensor(
                        "routed.weight_scale",
                        TensorRole::RoutedExpert,
                        Some(layer_id),
                        Some(expert_id),
                        true,
                    ));
                }
            }
        }
        TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: "/tmp/ds41rt-test".to_owned(),
            facts: ModelFacts::default(),
            tensors,
        }
    }

    fn tensor(
        name: &str,
        role: TensorRole,
        layer_id: Option<u32>,
        expert_id: Option<u32>,
        is_quantization_metadata: bool,
    ) -> TensorInfo {
        TensorInfo {
            name: name.to_owned(),
            file: "model.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: vec![1],
            byte_offset: 0,
            byte_length: 2,
            role,
            layer_id,
            expert_id,
            is_quantization_metadata,
        }
    }

    fn remove_first_matching(
        catalog: &mut TensorCatalog,
        mut predicate: impl FnMut(&TensorInfo) -> bool,
    ) -> TensorInfo {
        let index = catalog
            .tensors
            .iter()
            .position(|tensor| predicate(tensor))
            .expect("test catalog contains matching tensor");
        catalog.tensors.remove(index)
    }
}
