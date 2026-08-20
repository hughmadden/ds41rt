use anyhow::{Context, Result};
use ds4rt_core::{
    DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole, DS4_EXPERT_TP_WORLD_SIZE,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;

pub const DEEPSEEK_V4_EXL3_RECIPE_V2: &str = "deepseek_v4_exl3_trellis_2bpw_v2";
pub const DEEPSEEK_V4_EXL3_RECIPE_V3: &str =
    "deepseek_v4_exl3_trellis_2bpw_v3_flash_activation_pilot";
pub const DEEPSEEK_V4_EXL3_RECIPE_V4: &str = "deepseek_v4_exl3_trellis_2bpw_v4_flash_natural_route";
pub const DEEPSEEK_V4_EXL3_RECIPE_K3_V4: &str =
    "deepseek_v4_exl3_trellis_3bpw_v4_flash_natural_route";
pub const DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1: &str = "deepseek_v4_exl3_trellis_mixed_k2_k3_v1";
/// Production runtime identity for the shared K=2 EXL3 tensor layout.
pub const DEEPSEEK_V4_EXL3_RECIPE: &str = DEEPSEEK_V4_EXL3_RECIPE_V4;
pub const DEEPSEEK_V4_EXL3_SCHEMA: &str = "ds4rt.exl3.expert-trellis";
pub const DEEPSEEK_V4_EXL3_SCHEMA_VERSION: usize = 1;
pub const DEEPSEEK_V4_EXL3_TRELLIS_BITS: usize = 2;
pub const DEEPSEEK_V4_EXL3_CODEBOOK: &str = "mcg";
pub const DEEPSEEK_V4_EXL3_TENSOR_FORMAT: &str = "exllamav3_trellis_mcg";
pub const DEEPSEEK_V4_EXL3_SOURCE_FORMAT: &str = "fp4_e8m0_k32";
/// Resident SQG-XOR-Cheb T12 decode table required by the b12x trellis ABI.
/// The table is generated once per model block and is not checkpoint payload.
pub const DEEPSEEK_V4_EXL3_T12_LUT_BYTES: usize = 1 << 12;
pub const EXLLAMAV3_REPOSITORY: &str = "https://github.com/turboderp-org/exllamav3.git";
pub const EXLLAMAV3_REVISION: &str = "0b9745c526a13d5b30f1b58a864efc1932d3d9eb";
pub const EXLLAMAV3_SOURCE_TREE_SHA256: &str =
    "8c2f94e3a7335e304c47dad85a5254428950c324acf74263fb7bff59fe057dec";

pub fn is_deepseek_v4_exl3_recipe(recipe: &str) -> bool {
    matches!(
        recipe,
        DEEPSEEK_V4_EXL3_RECIPE_V2
            | DEEPSEEK_V4_EXL3_RECIPE_V3
            | DEEPSEEK_V4_EXL3_RECIPE_V4
            | DEEPSEEK_V4_EXL3_RECIPE_K3_V4
            | DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1
    )
}

pub fn is_deepseek_v4_mixed_exl3_recipe(recipe: &str) -> bool {
    recipe == DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1
}

pub fn exl3_trellis_bits_for_recipe(recipe: &str) -> Result<usize> {
    match recipe {
        DEEPSEEK_V4_EXL3_RECIPE_V2 | DEEPSEEK_V4_EXL3_RECIPE_V3 | DEEPSEEK_V4_EXL3_RECIPE_V4 => {
            Ok(2)
        }
        DEEPSEEK_V4_EXL3_RECIPE_K3_V4 => Ok(3),
        _ => anyhow::bail!("unsupported DeepSeek V4 EXL3 recipe {recipe:?}"),
    }
}

/// Exact resident K2/K3 layout for one strict-TP4 Spark rank and one model block.
///
/// Every rank stores every expert id but only its quarter of the routed
/// intermediate dimension. Hidden-axis rotations remain replicated; rotations
/// on the intermediate axis are sliced. No field includes a widened or
/// dequantized weight copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Exl3Tp4ResidentGeometry {
    pub trellis_bits: usize,
    pub hidden_size: usize,
    pub global_intermediate_size: usize,
    pub local_intermediate_size: usize,
    pub experts: usize,
    pub top_k: usize,
    pub projection_trellis_bytes: u64,
    pub w13_trellis_bytes: u64,
    pub w2_trellis_bytes: u64,
    pub hidden_rotation_table_bytes: u64,
    pub intermediate_rotation_bytes: u64,
    pub scalar_metadata_bytes: u64,
}

impl Exl3Tp4ResidentGeometry {
    pub fn from_model_facts(facts: &ModelFacts) -> Result<Self> {
        let trellis_bits = exl3_trellis_bits_for_recipe(&facts.quantization_recipe)?;
        Self::from_model_facts_with_trellis_bits(facts, trellis_bits)
    }

    pub fn from_model_facts_with_trellis_bits(
        facts: &ModelFacts,
        trellis_bits: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            facts.model_type == "deepseek_v4",
            "EXL3 TP4 geometry requires model_type=deepseek_v4, got {:?}",
            facts.model_type
        );
        anyhow::ensure!(
            facts.hidden_size > 0 && facts.hidden_size % 128 == 0,
            "EXL3 hidden size {} must be a positive multiple of Hadamard H128",
            facts.hidden_size
        );
        anyhow::ensure!(
            facts.moe_intermediate_size > 0
                && facts.moe_intermediate_size % (DS4_EXPERT_TP_WORLD_SIZE * 128) == 0,
            "EXL3 intermediate size {} must be divisible by TP{} Hadamard alignment {}",
            facts.moe_intermediate_size,
            DS4_EXPERT_TP_WORLD_SIZE,
            DS4_EXPERT_TP_WORLD_SIZE * 128
        );
        anyhow::ensure!(
            facts.routed_experts > 0 && facts.top_k > 0 && facts.top_k <= facts.routed_experts,
            "invalid EXL3 routed geometry: experts={} top_k={}",
            facts.routed_experts,
            facts.top_k
        );
        anyhow::ensure!(
            matches!(trellis_bits, 2 | 3),
            "EXL3 TP4 geometry supports K2 or K3, got K{trellis_bits}"
        );
        let hidden = facts.hidden_size as u64;
        let experts = facts.routed_experts as u64;
        let local_intermediate_size = facts.moe_intermediate_size / DS4_EXPERT_TP_WORLD_SIZE;
        let local_intermediate = local_intermediate_size as u64;
        // K=2 uses exactly two bits per logical matrix scalar.
        let projection_trellis_bytes = hidden
            .checked_mul(local_intermediate)
            .and_then(|bits| bits.checked_mul(trellis_bits as u64))
            .and_then(|bits| bits.checked_div(8))
            .context("EXL3 TP4 projection trellis byte count overflow")?;
        let w13_trellis_bytes = projection_trellis_bytes
            .checked_mul(experts)
            .and_then(|bytes| bytes.checked_mul(2))
            .context("EXL3 TP4 W13 trellis byte count overflow")?;
        let w2_trellis_bytes = projection_trellis_bytes
            .checked_mul(experts)
            .context("EXL3 TP4 W2 trellis byte count overflow")?;
        let hidden_rotation_table_bytes = experts
            .checked_mul(hidden)
            .and_then(|values| values.checked_mul(2))
            .context("EXL3 TP4 hidden rotation byte count overflow")?;
        // Gate SVH, up SVH, and down SUH are the three intermediate-axis
        // tables stored contiguously after slicing to this rank.
        let intermediate_rotation_bytes = experts
            .checked_mul(3)
            .and_then(|values| values.checked_mul(local_intermediate))
            .and_then(|values| values.checked_mul(2))
            .context("EXL3 TP4 intermediate rotation byte count overflow")?;
        let scalar_metadata_bytes = (facts.routed_experts + 1)
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|bytes| bytes.checked_add(16))
            .and_then(|bytes| {
                facts
                    .routed_experts
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|scale_bytes| bytes.checked_add(scale_bytes))
            })
            .and_then(|bytes| bytes.checked_add(DEEPSEEK_V4_EXL3_T12_LUT_BYTES))
            .context("EXL3 TP4 scalar metadata byte count overflow")?
            as u64;

        Ok(Self {
            trellis_bits,
            hidden_size: facts.hidden_size,
            global_intermediate_size: facts.moe_intermediate_size,
            local_intermediate_size,
            experts: facts.routed_experts,
            top_k: facts.top_k,
            projection_trellis_bytes,
            w13_trellis_bytes,
            w2_trellis_bytes,
            hidden_rotation_table_bytes,
            intermediate_rotation_bytes,
            scalar_metadata_bytes,
        })
    }

    pub fn resident_weight_bytes(self) -> u64 {
        self.w13_trellis_bytes + self.w2_trellis_bytes
    }

    pub fn resident_rotation_bytes(self) -> u64 {
        3 * self.hidden_rotation_table_bytes + self.intermediate_rotation_bytes
    }

    pub fn resident_total_bytes(self) -> u64 {
        self.resident_weight_bytes() + self.resident_rotation_bytes() + self.scalar_metadata_bytes
    }
}

#[derive(Debug, Deserialize)]
struct Exl3QuantizationConfig {
    quant_method: String,
    version: String,
    bits: f64,
    codebook: String,
    ds4rt: Ds4rtExl3Config,
}

#[derive(Debug, Deserialize)]
struct Ds4rtExl3Config {
    schema: String,
    schema_version: usize,
    recipe: String,
    scope: String,
    source_format: String,
    tensor_format: String,
    expert_tp_world_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Exl3RecipeContract {
    pub recipe: &'static str,
}

fn validate_gptqmodel_serving_quantization_config(value: &Value) -> Result<Exl3RecipeContract> {
    for field in ["quant_method", "method", "format", "checkpoint_format"] {
        anyhow::ensure!(
            value
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(|method| method.eq_ignore_ascii_case("exl3")),
            "GPTQModel EXL3 requires quantization_config.{field}=exl3"
        );
    }
    let bits = value
        .get("bits")
        .and_then(Value::as_f64)
        .filter(|bits| bits.is_finite() && *bits >= 2.0 && *bits <= 3.0)
        .context("GPTQModel EXL3 requires K2, K3, or a declared mixed K2/K3 payload")?;
    let mixed_plan = value.pointer("/meta/ds4rt_expert_bit_plan").filter(|plan| {
        plan.get("recipe").and_then(Value::as_str) == Some(DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1)
            && plan.get("schema").and_then(Value::as_str) == Some("ds4rt.exl3-mixed-k2-k3-v1")
    });
    let integer_bits = bits.fract() == 0.0 && matches!(bits as usize, 2 | 3);
    anyhow::ensure!(
        integer_bits || (bits > 2.0 && bits < 3.0 && mixed_plan.is_some()),
        "fractional EXL3 bits require a bound DS4RT K2/K3 expert plan"
    );
    anyhow::ensure!(
        value.get("codebook").and_then(Value::as_str) == Some(DEEPSEEK_V4_EXL3_CODEBOOK)
            && value.get("out_scales").and_then(Value::as_str) == Some("auto")
            && value.get("group_size").and_then(Value::as_i64) == Some(-1)
            && value.get("desc_act").and_then(Value::as_bool) == Some(false),
        "GPTQModel EXL3 requires the integer-tier MCG/auto-scale storage contract"
    );
    anyhow::ensure!(
        value
            .get("tensor_storage")
            .and_then(Value::as_object)
            .is_some_and(|storage| !storage.is_empty()),
        "GPTQModel EXL3 tensor_storage cannot be empty"
    );

    Ok(Exl3RecipeContract {
        recipe: if !integer_bits {
            DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1
        } else if bits as usize == 2 {
            DEEPSEEK_V4_EXL3_RECIPE_V4
        } else {
            DEEPSEEK_V4_EXL3_RECIPE_K3_V4
        },
    })
}

pub(crate) fn exl3_contract_from_quantization_config(
    quantization_config: Option<&Value>,
) -> Result<Option<Exl3RecipeContract>> {
    let Some(value) = quantization_config else {
        return Ok(None);
    };
    let quant_method = value
        .get("quant_method")
        .or_else(|| value.get("quant_algo"))
        .and_then(Value::as_str)
        .unwrap_or("unquantized");
    if !quant_method.eq_ignore_ascii_case("exl3") && !quant_method.eq_ignore_ascii_case("trellis") {
        return Ok(None);
    }
    anyhow::ensure!(
        quant_method.eq_ignore_ascii_case("exl3"),
        "legacy quant_method={quant_method:?} is not a DS4RT EXL3 artifact; expected exl3"
    );
    if value.get("tensor_storage").is_some() {
        return validate_gptqmodel_serving_quantization_config(value).map(Some);
    }
    validate_legacy_ds4rt_quantization_config(value).map(Some)
}

pub(crate) fn exl3_recipe_from_quantization_config(
    quantization_config: Option<&Value>,
) -> Result<Option<&'static str>> {
    Ok(exl3_contract_from_quantization_config(quantization_config)?.map(|value| value.recipe))
}

fn validate_legacy_ds4rt_quantization_config(value: &Value) -> Result<Exl3RecipeContract> {
    let config: Exl3QuantizationConfig =
        serde_json::from_value(value.clone()).context("parsing DS4RT EXL3 quantization_config")?;
    anyhow::ensure!(config.quant_method.eq_ignore_ascii_case("exl3"));
    anyhow::ensure!(
        config.version == "1.3.0",
        "DS4RT EXL3 requires exllamav3 version 1.3.0, got {:?}",
        config.version
    );
    anyhow::ensure!(
        (config.bits - DEEPSEEK_V4_EXL3_TRELLIS_BITS as f64).abs() < f64::EPSILON,
        "DS4RT EXL3 requires a native 2-bpw trellis payload, got {}",
        config.bits
    );
    anyhow::ensure!(
        config.codebook == DEEPSEEK_V4_EXL3_CODEBOOK,
        "DS4RT EXL3 requires the MCG codebook, got {:?}",
        config.codebook
    );
    let recipe = config.ds4rt.recipe.as_str();
    let validated_recipe = match recipe {
        DEEPSEEK_V4_EXL3_RECIPE_V2 => DEEPSEEK_V4_EXL3_RECIPE_V2,
        DEEPSEEK_V4_EXL3_RECIPE_V3 => DEEPSEEK_V4_EXL3_RECIPE_V3,
        DEEPSEEK_V4_EXL3_RECIPE_V4 => DEEPSEEK_V4_EXL3_RECIPE_V4,
        _ => "",
    };
    let ds4rt = config.ds4rt;
    anyhow::ensure!(
        ds4rt.schema == DEEPSEEK_V4_EXL3_SCHEMA
            && ds4rt.schema_version == DEEPSEEK_V4_EXL3_SCHEMA_VERSION,
        "unsupported DS4RT EXL3 schema {:?} version {}",
        ds4rt.schema,
        ds4rt.schema_version
    );
    anyhow::ensure!(
        matches!(
            ds4rt.recipe.as_str(),
            DEEPSEEK_V4_EXL3_RECIPE_V2 | DEEPSEEK_V4_EXL3_RECIPE_V3 | DEEPSEEK_V4_EXL3_RECIPE_V4
        ),
        "unsupported DS4RT EXL3 recipe {:?}",
        ds4rt.recipe
    );
    anyhow::ensure!(
        ds4rt.scope == "routed_experts",
        "DS4RT EXL3 quantizes only routed_experts, got {:?}",
        ds4rt.scope
    );
    anyhow::ensure!(
        ds4rt.source_format == DEEPSEEK_V4_EXL3_SOURCE_FORMAT
            && ds4rt.tensor_format == DEEPSEEK_V4_EXL3_TENSOR_FORMAT,
        "unsupported DS4RT EXL3 source/tensor format {:?}/{:?}",
        ds4rt.source_format,
        ds4rt.tensor_format
    );
    anyhow::ensure!(
        ds4rt.expert_tp_world_size == DS4_EXPERT_TP_WORLD_SIZE,
        "DS4RT EXL3 requires strict expert TP{}, got TP{}",
        DS4_EXPERT_TP_WORLD_SIZE,
        ds4rt.expert_tp_world_size
    );
    Ok(Exl3RecipeContract {
        recipe: validated_recipe,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Exl3ProjectionKind {
    Gate,
    Up,
    Down,
}

impl Exl3ProjectionKind {
    fn checkpoint_stem(self) -> &'static str {
        match self {
            Self::Gate => "w1",
            Self::Down => "w2",
            Self::Up => "w3",
        }
    }

    fn gptqmodel_stem(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Down => "down_proj",
            Self::Up => "up_proj",
        }
    }

    fn logical_shape(self, catalog: &TensorCatalog) -> (usize, usize) {
        match self {
            Self::Gate | Self::Up => (
                catalog.facts.hidden_size,
                catalog.facts.moe_intermediate_size,
            ),
            Self::Down => (
                catalog.facts.moe_intermediate_size,
                catalog.facts.hidden_size,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Exl3TensorLayout {
    CheckpointNative,
    GptqModel,
}

#[derive(Clone, Copy, Debug)]
pub struct Exl3Projection<'a> {
    pub kind: Exl3ProjectionKind,
    pub trellis: &'a TensorInfo,
    pub suh: &'a TensorInfo,
    pub svh: &'a TensorInfo,
    pub mcg: &'a TensorInfo,
    pub input_features: usize,
    pub output_features: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Exl3Expert<'a> {
    pub layer_id: usize,
    pub expert_id: usize,
    pub tensor_layout: Exl3TensorLayout,
    pub gate: Exl3Projection<'a>,
    pub up: Exl3Projection<'a>,
    pub down: Exl3Projection<'a>,
}

impl<'a> Exl3Expert<'a> {
    /// Projection-major EXL3 is already in SparkInfer kernel order `[gate; up]`.
    ///
    /// Native FP4 uses `[up; gate]` as input to its ModelOpt repacker, but the
    /// zero-copy Trellis path has no such row rotation.
    pub fn sparkinfer_w13(self) -> [Exl3Projection<'a>; 2] {
        [self.gate, self.up]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Exl3CatalogSummary {
    pub transformer_blocks: usize,
    pub experts_per_block: usize,
    pub expert_tensors: usize,
    pub trellis_bytes: u64,
    pub rotation_bytes: u64,
}

pub fn exl3_expert(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
) -> Result<Exl3Expert<'_>> {
    anyhow::ensure!(
        layer_id < catalog.facts.total_transformer_blocks(),
        "EXL3 layer {layer_id} exceeds {} transformer/dSpark blocks",
        catalog.facts.total_transformer_blocks()
    );
    anyhow::ensure!(
        expert_id < catalog.facts.routed_experts,
        "EXL3 expert {expert_id} exceeds {} routed experts",
        catalog.facts.routed_experts
    );
    let tensor_layout = exl3_tensor_layout(catalog, layer_id, expert_id)?;
    exl3_expert_with_layout(catalog, layer_id, expert_id, tensor_layout)
}

pub fn exl3_expert_trellis_bits(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
) -> Result<usize> {
    let expert = exl3_expert(catalog, layer_id, expert_id)?;
    let logical_values = expert
        .gate
        .input_features
        .checked_mul(expert.gate.output_features)
        .context("EXL3 logical expert size overflow")?;
    let stored_bits = usize::try_from(expert.gate.trellis.byte_length)
        .context("EXL3 trellis byte length exceeds usize")?
        .checked_mul(8)
        .context("EXL3 trellis bit length overflow")?;
    anyhow::ensure!(
        stored_bits % logical_values == 0,
        "EXL3 expert layer {layer_id} expert {expert_id} has a fractional trellis tier"
    );
    let bits = stored_bits / logical_values;
    anyhow::ensure!(
        matches!(bits, 2 | 3),
        "EXL3 expert layer {layer_id} expert {expert_id} has unsupported K{bits}"
    );
    for projection in [expert.up, expert.down] {
        let values = projection
            .input_features
            .checked_mul(projection.output_features)
            .context("EXL3 projection size overflow")?;
        anyhow::ensure!(
            usize::try_from(projection.trellis.byte_length)? * 8 == values * bits,
            "EXL3 expert layer {layer_id} expert {expert_id} mixes projection tiers"
        );
    }
    Ok(bits)
}

fn exl3_tensor_layout(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
) -> Result<Exl3TensorLayout> {
    let native = if layer_id < catalog.facts.num_hidden_layers {
        format!("layers.{layer_id}.ffn.experts.{expert_id}.w1.trellis")
    } else {
        format!(
            "mtp.{}.ffn.experts.{expert_id}.w1.trellis",
            layer_id - catalog.facts.num_hidden_layers
        )
    };
    let gptqmodel = if layer_id < catalog.facts.num_hidden_layers {
        format!("model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.trellis")
    } else {
        format!(
            "mtp.{}.mlp.experts.{expert_id}.gate_proj.trellis",
            layer_id - catalog.facts.num_hidden_layers
        )
    };
    let contains = |name: &str| {
        catalog
            .tensors
            .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
            .is_ok()
    };
    match (contains(&native), contains(&gptqmodel)) {
        (true, false) => Ok(Exl3TensorLayout::CheckpointNative),
        (false, true) => Ok(Exl3TensorLayout::GptqModel),
        (false, false) => {
            anyhow::bail!("missing EXL3 layout anchor; tried {native} and {gptqmodel}")
        }
        (true, true) => {
            anyhow::bail!("ambiguous EXL3 tensor layout exposes both {native} and {gptqmodel}")
        }
    }
}

fn exl3_expert_with_layout(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    tensor_layout: Exl3TensorLayout,
) -> Result<Exl3Expert<'_>> {
    let uniform_trellis_bits =
        (!is_deepseek_v4_mixed_exl3_recipe(&catalog.facts.quantization_recipe))
            .then(|| exl3_trellis_bits_for_recipe(&catalog.facts.quantization_recipe))
            .transpose()?;
    let block = match (tensor_layout, layer_id < catalog.facts.num_hidden_layers) {
        (Exl3TensorLayout::CheckpointNative, true) => {
            format!("layers.{layer_id}.ffn")
        }
        (Exl3TensorLayout::CheckpointNative, false) => {
            format!("mtp.{}.ffn", layer_id - catalog.facts.num_hidden_layers)
        }
        (Exl3TensorLayout::GptqModel, true) => {
            format!("model.layers.{layer_id}.mlp")
        }
        (Exl3TensorLayout::GptqModel, false) => {
            format!("mtp.{}.mlp", layer_id - catalog.facts.num_hidden_layers)
        }
    };
    let projection = |kind: Exl3ProjectionKind| -> Result<Exl3Projection<'_>> {
        let stem = match tensor_layout {
            Exl3TensorLayout::CheckpointNative => kind.checkpoint_stem(),
            Exl3TensorLayout::GptqModel => kind.gptqmodel_stem(),
        };
        let base = format!("{block}.experts.{expert_id}.{stem}");
        let find = |suffix: &str| -> Result<&TensorInfo> {
            let name = format!("{base}.{suffix}");
            catalog
                .tensors
                .binary_search_by(|tensor| tensor.name.as_str().cmp(name.as_str()))
                .ok()
                .map(|index| &catalog.tensors[index])
                .with_context(|| format!("missing DS4RT EXL3 tensor {name}"))
        };
        let (input_features, output_features) = kind.logical_shape(catalog);
        let trellis = find("trellis")?;
        let logical_values = input_features
            .checked_mul(output_features)
            .context("EXL3 logical projection size overflow")?;
        let stored_bits = usize::try_from(trellis.byte_length)?
            .checked_mul(8)
            .context("EXL3 trellis bit length overflow")?;
        anyhow::ensure!(
            stored_bits % logical_values == 0,
            "EXL3 projection {base} has a fractional stored trellis tier"
        );
        let trellis_bits = uniform_trellis_bits.unwrap_or(stored_bits / logical_values);
        let value = Exl3Projection {
            kind,
            trellis,
            suh: find("suh")?,
            svh: find("svh")?,
            mcg: find("mcg")?,
            input_features,
            output_features,
        };
        validate_projection(value, trellis_bits)?;
        Ok(value)
    };
    Ok(Exl3Expert {
        layer_id,
        expert_id,
        tensor_layout,
        gate: projection(Exl3ProjectionKind::Gate)?,
        up: projection(Exl3ProjectionKind::Up)?,
        down: projection(Exl3ProjectionKind::Down)?,
    })
}

pub fn validate_exl3_expert_catalog(catalog: &TensorCatalog) -> Result<Exl3CatalogSummary> {
    anyhow::ensure!(
        is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe),
        "EXL3 validation requires a recognized DeepSeek V4 EXL3 recipe, got {}",
        catalog.facts.quantization_recipe
    );
    if !is_deepseek_v4_mixed_exl3_recipe(&catalog.facts.quantization_recipe) {
        let _resident_geometry = Exl3Tp4ResidentGeometry::from_model_facts(&catalog.facts)?;
    }
    let mut expected_names = BTreeSet::new();
    let mut trellis_bytes = 0_u64;
    let mut rotation_bytes = 0_u64;
    let tensor_layout =
        exl3_tensor_layout(catalog, 0, 0).context("detecting artifact-wide EXL3 tensor layout")?;
    for layer_id in 0..catalog.facts.total_transformer_blocks() {
        for expert_id in 0..catalog.facts.routed_experts {
            let expert = exl3_expert_with_layout(catalog, layer_id, expert_id, tensor_layout)
                .with_context(|| format!("validating EXL3 layer {layer_id} expert {expert_id}"))?;
            for projection in [expert.gate, expert.up, expert.down] {
                for tensor in [
                    projection.trellis,
                    projection.suh,
                    projection.svh,
                    projection.mcg,
                ] {
                    expected_names.insert(tensor.name.as_str());
                }
                trellis_bytes = trellis_bytes
                    .checked_add(projection.trellis.byte_length)
                    .context("EXL3 trellis byte total overflow")?;
                rotation_bytes = rotation_bytes
                    .checked_add(projection.suh.byte_length)
                    .and_then(|value| value.checked_add(projection.svh.byte_length))
                    .context("EXL3 rotation byte total overflow")?;
            }
        }
    }
    let actual_names = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::RoutedExpert)
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        actual_names == expected_names,
        "EXL3 routed tensor set mismatch: expected {}, found {}; missing={:?}; unexpected={:?}",
        expected_names.len(),
        actual_names.len(),
        expected_names
            .difference(&actual_names)
            .take(8)
            .collect::<Vec<_>>(),
        actual_names
            .difference(&expected_names)
            .take(8)
            .collect::<Vec<_>>()
    );
    Ok(Exl3CatalogSummary {
        transformer_blocks: catalog.facts.total_transformer_blocks(),
        experts_per_block: catalog.facts.routed_experts,
        expert_tensors: expected_names.len(),
        trellis_bytes,
        rotation_bytes,
    })
}

fn validate_projection(projection: Exl3Projection<'_>, trellis_bits: usize) -> Result<()> {
    let expected_trellis = vec![
        projection.input_features / 16,
        projection.output_features / 16,
        16 * trellis_bits,
    ];
    validate_tensor(
        projection.trellis,
        DType::I16,
        &expected_trellis,
        (expected_trellis.iter().product::<usize>() * 2) as u64,
        false,
    )?;
    validate_tensor(
        projection.suh,
        DType::F16,
        &[projection.input_features],
        (projection.input_features * 2) as u64,
        true,
    )?;
    validate_tensor(
        projection.svh,
        DType::F16,
        &[projection.output_features],
        (projection.output_features * 2) as u64,
        true,
    )?;
    validate_tensor(projection.mcg, DType::I32, &[], 4, true)?;
    Ok(())
}

fn validate_tensor(
    tensor: &TensorInfo,
    expected_dtype: DType,
    expected_shape: &[usize],
    expected_bytes: u64,
    expected_quantization_metadata: bool,
) -> Result<()> {
    anyhow::ensure!(
        tensor.role == TensorRole::RoutedExpert
            && tensor.dtype == expected_dtype
            && tensor.shape == expected_shape
            && tensor.byte_length == expected_bytes
            && tensor.is_quantization_metadata == expected_quantization_metadata,
        "invalid EXL3 tensor {}: role={:?} dtype={:?} shape={:?} bytes={} quantization_metadata={}, expected role=RoutedExpert dtype={:?} shape={:?} bytes={expected_bytes} quantization_metadata={expected_quantization_metadata}",
        tensor.name,
        tensor.role,
        tensor.dtype,
        tensor.shape,
        tensor.byte_length,
        tensor.is_quantization_metadata,
        expected_dtype,
        expected_shape
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4rt_core::{ModelFacts, ModelVariant};

    fn valid_quantization_config() -> Value {
        serde_json::json!({
            "quant_method": "exl3",
            "version": "1.3.0",
            "bits": 2.0,
            "codebook": "mcg",
            "ds4rt": {
                "schema": "ds4rt.exl3.expert-trellis",
                "schema_version": 1,
                "recipe": "deepseek_v4_exl3_trellis_2bpw_v2",
                "scope": "routed_experts",
                "source_format": "fp4_e8m0_k32",
                "tensor_format": "exllamav3_trellis_mcg",
                "expert_tp_world_size": 4
            }
        })
    }

    fn valid_activation_quantization_config() -> Value {
        let mut config = valid_quantization_config();
        *config.pointer_mut("/ds4rt/recipe").unwrap() =
            serde_json::json!(DEEPSEEK_V4_EXL3_RECIPE_V3);
        config
    }

    fn valid_natural_route_quantization_config() -> Value {
        let mut config = valid_quantization_config();
        *config.pointer_mut("/ds4rt/recipe").unwrap() =
            serde_json::json!(DEEPSEEK_V4_EXL3_RECIPE_V4);
        config
    }

    fn valid_gptqmodel_quantization_config() -> Value {
        serde_json::json!({
            "bits": 2.0,
            "group_size": -1,
            "desc_act": false,
            "method": "exl3",
            "quant_method": "exl3",
            "format": "exl3",
            "checkpoint_format": "exl3",
            "out_scales": "auto",
            "codebook": "mcg",
            "tensor_storage": {
                "model.layers.0.mlp.experts.0.gate_proj": {
                    "stored_tensors": {"trellis": "trellis"}
                }
            }
        })
    }

    #[test]
    fn legacy_schema_checks_only_the_k2_serving_layout() {
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&valid_quantization_config())).unwrap(),
            Some(DEEPSEEK_V4_EXL3_RECIPE_V2)
        );
        for (pointer, value, expected) in [
            ("/bits", serde_json::json!(3.0), "native 2-bpw"),
            (
                "/ds4rt/expert_tp_world_size",
                serde_json::json!(2),
                "strict expert TP4",
            ),
            (
                "/ds4rt/tensor_format",
                serde_json::json!("different"),
                "source/tensor format",
            ),
        ] {
            let mut config = valid_quantization_config();
            *config.pointer_mut(pointer).unwrap() = value;
            let error = exl3_recipe_from_quantization_config(Some(&config))
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }

        let mut extra_quantization_provenance = valid_quantization_config();
        extra_quantization_provenance["calibration"] =
            serde_json::json!({"method": "publication-side-only"});
        extra_quantization_provenance["ds4rt"]["quantizer_source"] =
            serde_json::json!({"revision": "not-a-runtime-field"});
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&extra_quantization_provenance)).unwrap(),
            Some(DEEPSEEK_V4_EXL3_RECIPE_V2)
        );
    }

    #[test]
    fn legacy_recipe_ids_do_not_trigger_calibration_audits() {
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&valid_activation_quantization_config()))
                .unwrap(),
            Some(DEEPSEEK_V4_EXL3_RECIPE_V3)
        );
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&valid_natural_route_quantization_config()))
                .unwrap(),
            Some(DEEPSEEK_V4_EXL3_RECIPE_V4)
        );
    }

    #[test]
    fn gptqmodel_schema_accepts_uniform_k2_and_k3_serving_layouts() {
        let config = valid_gptqmodel_quantization_config();
        assert_eq!(
            exl3_contract_from_quantization_config(Some(&config)).unwrap(),
            Some(Exl3RecipeContract {
                recipe: DEEPSEEK_V4_EXL3_RECIPE_V4,
            })
        );
        let mut k3 = valid_gptqmodel_quantization_config();
        k3["bits"] = serde_json::json!(3.0);
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&k3)).unwrap(),
            Some(DEEPSEEK_V4_EXL3_RECIPE_K3_V4)
        );
        let mut mixed = valid_gptqmodel_quantization_config();
        mixed["bits"] = serde_json::json!(2.1);
        mixed["meta"] = serde_json::json!({
            "ds4rt_expert_bit_plan": {
                "schema": "ds4rt.exl3-mixed-k2-k3-v1",
                "recipe": DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1
            }
        });
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&mixed)).unwrap(),
            Some(DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1)
        );
        for (pointer, value, expected) in [
            (
                "/bits",
                serde_json::json!(2.5),
                "bound DS4RT K2/K3 expert plan",
            ),
            (
                "/codebook",
                serde_json::json!("different"),
                "integer-tier MCG",
            ),
            ("/tensor_storage", serde_json::json!({}), "cannot be empty"),
        ] {
            let mut invalid = valid_gptqmodel_quantization_config();
            *invalid.pointer_mut(pointer).unwrap() = value;
            let error = exl3_recipe_from_quantization_config(Some(&invalid))
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }

        let mut provenance = valid_gptqmodel_quantization_config();
        provenance["meta"] = serde_json::json!({
            "ds4rt_error_ledger": {
                "arbitrary": "publication-side-only"
            }
        });
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&provenance)).unwrap(),
            Some(DEEPSEEK_V4_EXL3_RECIPE_V4)
        );
    }

    fn test_tensor_for_block(
        facts: &ModelFacts,
        block: &str,
        layer_id: usize,
        stem: &str,
        suffix: &str,
    ) -> TensorInfo {
        test_tensor_for_layout(
            facts,
            block,
            layer_id,
            stem,
            suffix,
            Exl3TensorLayout::CheckpointNative,
        )
    }

    fn test_tensor_for_layout(
        facts: &ModelFacts,
        block: &str,
        layer_id: usize,
        stem: &str,
        suffix: &str,
        tensor_layout: Exl3TensorLayout,
    ) -> TensorInfo {
        let kind = match stem {
            "w1" | "gate_proj" => Exl3ProjectionKind::Gate,
            "w2" | "down_proj" => Exl3ProjectionKind::Down,
            "w3" | "up_proj" => Exl3ProjectionKind::Up,
            _ => unreachable!(),
        };
        let (input, output) = kind.logical_shape(&TensorCatalog {
            model_id: String::new(),
            snapshot_path: String::new(),
            facts: facts.clone(),
            tensors: vec![],
        });
        let (dtype, shape, bytes) = match suffix {
            "trellis" => {
                let shape = vec![input / 16, output / 16, 32];
                let bytes = (shape.iter().product::<usize>() * 2) as u64;
                (DType::I16, shape, bytes)
            }
            "suh" => (DType::F16, vec![input], (input * 2) as u64),
            "svh" => (DType::F16, vec![output], (output * 2) as u64),
            "mcg" => (DType::I32, vec![], 4),
            _ => unreachable!(),
        };
        TensorInfo {
            name: format!(
                "{block}.{}.experts.0.{stem}.{suffix}",
                match tensor_layout {
                    Exl3TensorLayout::CheckpointNative => "ffn",
                    Exl3TensorLayout::GptqModel => "mlp",
                }
            ),
            file: "model.safetensors".to_owned(),
            dtype,
            shape,
            byte_offset: 0,
            byte_length: bytes,
            role: TensorRole::RoutedExpert,
            layer_id: Some(layer_id as u32),
            expert_id: Some(0),
            is_quantization_metadata: suffix != "trellis",
        }
    }

    fn test_tensor(facts: &ModelFacts, stem: &str, suffix: &str) -> TensorInfo {
        test_tensor_for_block(facts, "layers.0", 0, stem, suffix)
    }

    fn test_catalog(hidden: usize, intermediate: usize, variant: ModelVariant) -> TensorCatalog {
        let mut facts = ModelFacts::default();
        facts.variant = variant;
        facts.num_hidden_layers = 1;
        facts.dspark_target_layer_ids.clear();
        facts.routed_experts = 1;
        facts.top_k = 1;
        facts.hidden_size = hidden;
        facts.moe_intermediate_size = intermediate;
        facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
        let mut tensors = Vec::new();
        for stem in ["w1", "w2", "w3"] {
            for suffix in ["trellis", "suh", "svh", "mcg"] {
                tensors.push(test_tensor(&facts, stem, suffix));
            }
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        TensorCatalog {
            model_id: "deepseek-ai/test-exl3".to_owned(),
            snapshot_path: "/tmp/test-exl3".to_owned(),
            facts,
            tensors,
        }
    }

    #[test]
    fn flash_and_pro_experts_validate_with_equal_tp4_hadamard_slices() {
        for (hidden, intermediate, variant) in [
            (4096, 2048, ModelVariant::Flash),
            (7168, 3072, ModelVariant::Pro),
        ] {
            let catalog = test_catalog(hidden, intermediate, variant);
            let summary = validate_exl3_expert_catalog(&catalog).unwrap();
            assert_eq!(summary.expert_tensors, 12);
            assert_eq!(intermediate / DS4_EXPERT_TP_WORLD_SIZE % 128, 0);
            let expert = exl3_expert(&catalog, 0, 0).unwrap();
            assert_eq!(expert.sparkinfer_w13()[0].kind, Exl3ProjectionKind::Gate);
            assert_eq!(expert.down.input_features, intermediate);
        }
    }

    #[test]
    fn exl3_catalog_requires_all_integrated_dspark_routed_blocks() {
        let mut facts = ModelFacts::default();
        facts.variant = ModelVariant::Flash;
        facts.num_hidden_layers = 1;
        facts.dspark_target_layer_ids = vec![40, 41, 42];
        facts.routed_experts = 1;
        facts.top_k = 1;
        facts.hidden_size = 4_096;
        facts.moe_intermediate_size = 2_048;
        facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
        let mut tensors = Vec::new();
        for (layer_id, block) in ["layers.0", "mtp.0", "mtp.1", "mtp.2"]
            .into_iter()
            .enumerate()
        {
            for stem in ["w1", "w2", "w3"] {
                for suffix in ["trellis", "suh", "svh", "mcg"] {
                    tensors.push(test_tensor_for_block(&facts, block, layer_id, stem, suffix));
                }
            }
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        let catalog = TensorCatalog {
            model_id: "deepseek-ai/test-exl3-dspark".to_owned(),
            snapshot_path: "/tmp/test-exl3-dspark".to_owned(),
            facts,
            tensors,
        };

        let summary = validate_exl3_expert_catalog(&catalog).unwrap();
        assert_eq!(summary.transformer_blocks, 4);
        assert_eq!(summary.expert_tensors, 48);
        assert_eq!(
            exl3_expert(&catalog, 3, 0).unwrap().up.trellis.name,
            "mtp.2.ffn.experts.0.w3.trellis"
        );
    }

    #[test]
    fn gptqmodel_publication_names_validate_for_base_and_dspark_blocks() {
        let mut facts = ModelFacts::default();
        facts.variant = ModelVariant::Flash;
        facts.num_hidden_layers = 1;
        facts.dspark_target_layer_ids = vec![40, 41, 42];
        facts.routed_experts = 1;
        facts.top_k = 1;
        facts.hidden_size = 4_096;
        facts.moe_intermediate_size = 2_048;
        facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
        let mut tensors = Vec::new();
        for (layer_id, block) in ["model.layers.0", "mtp.0", "mtp.1", "mtp.2"]
            .into_iter()
            .enumerate()
        {
            for stem in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in ["trellis", "suh", "svh", "mcg"] {
                    tensors.push(test_tensor_for_layout(
                        &facts,
                        block,
                        layer_id,
                        stem,
                        suffix,
                        Exl3TensorLayout::GptqModel,
                    ));
                }
            }
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        let catalog = TensorCatalog {
            model_id: "tpurtell/test-gptqmodel-exl3".to_owned(),
            snapshot_path: "/tmp/test-gptqmodel-exl3".to_owned(),
            facts,
            tensors,
        };

        let summary = validate_exl3_expert_catalog(&catalog).unwrap();
        assert_eq!(summary.transformer_blocks, 4);
        assert_eq!(summary.expert_tensors, 48);
        assert_eq!(summary.trellis_bytes, 25_165_824);
        assert_eq!(summary.rotation_bytes, 147_456);
        let base = exl3_expert(&catalog, 0, 0).unwrap();
        assert_eq!(base.tensor_layout, Exl3TensorLayout::GptqModel);
        assert!(!base.gate.trellis.is_quantization_metadata);
        assert!(base.gate.suh.is_quantization_metadata);
        assert!(base.gate.svh.is_quantization_metadata);
        assert!(base.gate.mcg.is_quantization_metadata);
        assert_eq!(
            base.gate.trellis.name,
            "model.layers.0.mlp.experts.0.gate_proj.trellis"
        );
        assert_eq!(
            exl3_expert(&catalog, 3, 0).unwrap().up.trellis.name,
            "mtp.2.mlp.experts.0.up_proj.trellis"
        );
    }

    #[test]
    fn exl3_catalog_rejects_mixed_publication_namespaces() {
        let mut catalog = test_catalog(4_096, 2_048, ModelVariant::Flash);
        let mut gptq_alias = catalog
            .tensors
            .iter()
            .find(|tensor| tensor.name == "layers.0.ffn.experts.0.w1.trellis")
            .unwrap()
            .clone();
        gptq_alias.name = "model.layers.0.mlp.experts.0.gate_proj.trellis".to_owned();
        catalog.tensors.push(gptq_alias);
        catalog
            .tensors
            .sort_by(|left, right| left.name.cmp(&right.name));

        let error = format!("{:#}", validate_exl3_expert_catalog(&catalog).unwrap_err());
        assert!(error.contains("both"), "unexpected error: {error}");
    }

    #[test]
    fn exl3_catalog_rejects_trellis_misclassified_as_metadata() {
        let mut catalog = test_catalog(4_096, 2_048, ModelVariant::Flash);
        catalog
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "layers.0.ffn.experts.0.w1.trellis")
            .unwrap()
            .is_quantization_metadata = true;

        let error = format!("{:#}", validate_exl3_expert_catalog(&catalog).unwrap_err());
        assert!(
            error.contains("quantization_metadata=true")
                && error.contains("quantization_metadata=false"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn flash_and_pro_rank_local_residency_has_no_widened_weight_copy() {
        for (hidden, intermediate, experts, expected_projection, expected_weight, expected_total) in [
            (4_096, 2_048, 256, 524_288, 402_653_184, 409_737_236),
            (7_168, 3_072, 384, 1_376_256, 1_585_446_912, 1_603_738_644),
        ] {
            let mut facts = ModelFacts::default();
            facts.hidden_size = hidden;
            facts.moe_intermediate_size = intermediate;
            facts.routed_experts = experts;
            facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
            let geometry = Exl3Tp4ResidentGeometry::from_model_facts(&facts).unwrap();
            assert_eq!(geometry.local_intermediate_size, intermediate / 4);
            assert_eq!(geometry.projection_trellis_bytes, expected_projection);
            assert_eq!(geometry.resident_weight_bytes(), expected_weight);
            assert_eq!(geometry.resident_total_bytes(), expected_total);
        }
    }

    #[test]
    fn explicit_k3_rank_local_geometry_scales_only_trellis_storage() {
        let mut facts = ModelFacts::default();
        facts.hidden_size = 4_096;
        facts.moe_intermediate_size = 2_048;
        facts.routed_experts = 256;
        facts.top_k = 6;
        facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1.to_owned();

        let k2 = Exl3Tp4ResidentGeometry::from_model_facts_with_trellis_bits(&facts, 2).unwrap();
        let k3 = Exl3Tp4ResidentGeometry::from_model_facts_with_trellis_bits(&facts, 3).unwrap();

        assert_eq!(
            k3.projection_trellis_bytes,
            k2.projection_trellis_bytes * 3 / 2
        );
        assert_eq!(
            k3.resident_weight_bytes(),
            k2.resident_weight_bytes() * 3 / 2
        );
        assert_eq!(k3.resident_rotation_bytes(), k2.resident_rotation_bytes());
        assert!(Exl3Tp4ResidentGeometry::from_model_facts_with_trellis_bits(&facts, 4).is_err());
    }
}
