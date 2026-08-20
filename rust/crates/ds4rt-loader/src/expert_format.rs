use anyhow::{Context, Result};
use ds4rt_core::{DType, TensorCatalog, TensorInfo, TensorRole, DS4_EXPERT_TP_WORLD_SIZE};
use std::collections::BTreeSet;

pub const NATIVE_FP4_K_BLOCK: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeFp4ProjectionKind {
    Gate,
    Up,
    Down,
}

impl NativeFp4ProjectionKind {
    fn checkpoint_stem(self) -> &'static str {
        match self {
            Self::Gate => "w1",
            Self::Down => "w2",
            Self::Up => "w3",
        }
    }

    fn logical_shape(self, catalog: &TensorCatalog) -> (usize, usize) {
        match self {
            Self::Gate | Self::Up => (
                catalog.facts.moe_intermediate_size,
                catalog.facts.hidden_size,
            ),
            Self::Down => (
                catalog.facts.hidden_size,
                catalog.facts.moe_intermediate_size,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NativeFp4Projection<'a> {
    pub kind: NativeFp4ProjectionKind,
    pub weight: &'a TensorInfo,
    pub scale: &'a TensorInfo,
    pub logical_rows: usize,
    pub logical_columns: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeFp4Expert<'a> {
    pub layer_id: usize,
    pub expert_id: usize,
    pub gate: NativeFp4Projection<'a>,
    pub up: NativeFp4Projection<'a>,
    pub down: NativeFp4Projection<'a>,
}

impl<'a> NativeFp4Expert<'a> {
    /// SparkInfer's `w13` source layout is `[up; gate]`, not checkpoint order.
    pub fn sparkinfer_w13(self) -> [NativeFp4Projection<'a>; 2] {
        [self.up, self.gate]
    }

    pub fn tp_shard(self, rank: usize) -> Result<NativeFp4TpExpertShard<'a>> {
        Ok(NativeFp4TpExpertShard {
            layer_id: self.layer_id,
            expert_id: self.expert_id,
            rank,
            world_size: DS4_EXPERT_TP_WORLD_SIZE,
            gate: native_fp4_tp_projection(self.gate, rank)?,
            up: native_fp4_tp_projection(self.up, rank)?,
            down: native_fp4_tp_projection(self.down, rank)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NativeFp4TpTensorWindow<'a> {
    pub tensor: &'a TensorInfo,
    pub row_start: usize,
    pub row_count: usize,
    pub column_start_bytes: usize,
    pub row_width_bytes: usize,
}

impl NativeFp4TpTensorWindow<'_> {
    pub fn source_bytes(self) -> Result<u64> {
        (self.row_count as u64)
            .checked_mul(self.row_width_bytes as u64)
            .context("native FP4 TP tensor source byte count overflow")
    }

    pub fn is_contiguous(self) -> bool {
        self.column_start_bytes == 0
            && self.tensor.shape.get(1).copied() == Some(self.row_width_bytes)
    }

    pub fn source_offset_for_row(self, local_row: usize) -> Result<u64> {
        anyhow::ensure!(
            local_row < self.row_count,
            "native FP4 TP local row {local_row} exceeds {} rows",
            self.row_count
        );
        let source_row_width = self
            .tensor
            .shape
            .get(1)
            .copied()
            .context("native FP4 TP tensor has no packed row width")?;
        let source_row = self
            .row_start
            .checked_add(local_row)
            .context("native FP4 TP source row overflow")?;
        let row_offset = source_row
            .checked_mul(source_row_width)
            .and_then(|offset| offset.checked_add(self.column_start_bytes))
            .context("native FP4 TP source row byte offset overflow")?;
        self.tensor
            .byte_offset
            .checked_add(row_offset as u64)
            .context("native FP4 TP absolute source offset overflow")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NativeFp4TpProjectionShard<'a> {
    pub kind: NativeFp4ProjectionKind,
    pub weight: NativeFp4TpTensorWindow<'a>,
    pub scale: NativeFp4TpTensorWindow<'a>,
    pub local_intermediate_size: usize,
}

impl NativeFp4TpProjectionShard<'_> {
    pub fn source_bytes(self) -> Result<u64> {
        self.weight
            .source_bytes()?
            .checked_add(self.scale.source_bytes()?)
            .context("native FP4 TP projection source byte count overflow")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NativeFp4TpExpertShard<'a> {
    pub layer_id: usize,
    pub expert_id: usize,
    pub rank: usize,
    pub world_size: usize,
    pub gate: NativeFp4TpProjectionShard<'a>,
    pub up: NativeFp4TpProjectionShard<'a>,
    pub down: NativeFp4TpProjectionShard<'a>,
}

impl<'a> NativeFp4TpExpertShard<'a> {
    pub fn sparkinfer_w13(self) -> [NativeFp4TpProjectionShard<'a>; 2] {
        [self.up, self.gate]
    }

    pub fn source_bytes(self) -> Result<u64> {
        let w13_bytes = self
            .gate
            .source_bytes()?
            .checked_add(self.up.source_bytes()?)
            .context("native FP4 TP W13 source byte count overflow")?;
        w13_bytes
            .checked_add(self.down.source_bytes()?)
            .context("native FP4 TP expert source byte count overflow")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeFp4CatalogSummary {
    pub transformer_blocks: usize,
    pub experts_per_block: usize,
    pub expert_tensors: usize,
    pub packed_weight_bytes: u64,
    pub e8m0_scale_bytes: u64,
}

pub fn native_fp4_expert<'a>(
    catalog: &'a TensorCatalog,
    layer_id: usize,
    expert_id: usize,
) -> Result<NativeFp4Expert<'a>> {
    anyhow::ensure!(
        layer_id < catalog.facts.total_transformer_blocks(),
        "DeepSeek V4 expert layer {layer_id} exceeds {} transformer blocks",
        catalog.facts.total_transformer_blocks()
    );
    anyhow::ensure!(
        expert_id < catalog.facts.routed_experts,
        "DeepSeek V4 expert id {expert_id} exceeds {} routed experts",
        catalog.facts.routed_experts
    );
    let gate = native_fp4_projection(catalog, layer_id, expert_id, NativeFp4ProjectionKind::Gate)?;
    let up = native_fp4_projection(catalog, layer_id, expert_id, NativeFp4ProjectionKind::Up)?;
    let down = native_fp4_projection(catalog, layer_id, expert_id, NativeFp4ProjectionKind::Down)?;
    Ok(NativeFp4Expert {
        layer_id,
        expert_id,
        gate,
        up,
        down,
    })
}

pub fn validate_native_fp4_expert_catalog(
    catalog: &TensorCatalog,
) -> Result<NativeFp4CatalogSummary> {
    anyhow::ensure!(
        catalog.facts.model_type == "deepseek_v4",
        "native FP4 expert validation requires model_type=deepseek_v4"
    );
    anyhow::ensure!(
        catalog.facts.expert_dtype == "fp4",
        "native FP4 expert validation requires expert_dtype=fp4, got {:?}",
        catalog.facts.expert_dtype
    );
    anyhow::ensure!(
        catalog.facts.quantization_recipe == "deepseek_v4_native_fp4_fp8_mixed_v1",
        "native FP4 expert validation cannot consume recipe {:?}",
        catalog.facts.quantization_recipe
    );
    anyhow::ensure!(
        catalog.facts.hidden_size % NATIVE_FP4_K_BLOCK == 0
            && catalog.facts.moe_intermediate_size % NATIVE_FP4_K_BLOCK == 0,
        "native FP4 dimensions hidden={} intermediate={} must be divisible by {NATIVE_FP4_K_BLOCK}",
        catalog.facts.hidden_size,
        catalog.facts.moe_intermediate_size
    );

    let transformer_blocks = catalog.facts.total_transformer_blocks();
    anyhow::ensure!(
        transformer_blocks > 0,
        "checkpoint has no transformer blocks"
    );
    anyhow::ensure!(
        catalog.facts.routed_experts > 0,
        "checkpoint has no routed experts"
    );

    let mut expected_names = BTreeSet::new();
    let mut packed_weight_bytes = 0_u64;
    let mut e8m0_scale_bytes = 0_u64;
    for layer_id in 0..transformer_blocks {
        for expert_id in 0..catalog.facts.routed_experts {
            let expert = native_fp4_expert(catalog, layer_id, expert_id).with_context(|| {
                format!("validating native FP4 layer {layer_id} expert {expert_id}")
            })?;
            for projection in [expert.gate, expert.up, expert.down] {
                expected_names.insert(projection.weight.name.as_str());
                expected_names.insert(projection.scale.name.as_str());
                packed_weight_bytes = packed_weight_bytes
                    .checked_add(projection.weight.byte_length)
                    .context("native FP4 packed weight byte total overflow")?;
                e8m0_scale_bytes = e8m0_scale_bytes
                    .checked_add(projection.scale.byte_length)
                    .context("native FP4 scale byte total overflow")?;
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
        "native FP4 routed tensor set mismatch: expected {} tensors, found {}; missing={:?}; unexpected={:?}",
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

    Ok(NativeFp4CatalogSummary {
        transformer_blocks,
        experts_per_block: catalog.facts.routed_experts,
        expert_tensors: expected_names.len(),
        packed_weight_bytes,
        e8m0_scale_bytes,
    })
}

fn native_fp4_tp_projection(
    projection: NativeFp4Projection<'_>,
    rank: usize,
) -> Result<NativeFp4TpProjectionShard<'_>> {
    anyhow::ensure!(
        rank < DS4_EXPERT_TP_WORLD_SIZE,
        "native FP4 expert TP rank {rank} exceeds world size {DS4_EXPERT_TP_WORLD_SIZE}"
    );
    let intermediate_size = match projection.kind {
        NativeFp4ProjectionKind::Gate | NativeFp4ProjectionKind::Up => projection.logical_rows,
        NativeFp4ProjectionKind::Down => projection.logical_columns,
    };
    let alignment = DS4_EXPERT_TP_WORLD_SIZE * NATIVE_FP4_K_BLOCK;
    anyhow::ensure!(
        intermediate_size % alignment == 0,
        "native FP4 intermediate size {intermediate_size} must be divisible by TP4 K/32 alignment {alignment}"
    );
    let local_intermediate_size = intermediate_size / DS4_EXPERT_TP_WORLD_SIZE;
    let intermediate_start = local_intermediate_size
        .checked_mul(rank)
        .context("native FP4 TP intermediate start overflow")?;

    let (row_start, row_count, weight_column_start, weight_row_width) = match projection.kind {
        NativeFp4ProjectionKind::Gate | NativeFp4ProjectionKind::Up => (
            intermediate_start,
            local_intermediate_size,
            0,
            projection.logical_columns / 2,
        ),
        NativeFp4ProjectionKind::Down => (
            0,
            projection.logical_rows,
            intermediate_start / 2,
            local_intermediate_size / 2,
        ),
    };
    let (scale_column_start, scale_row_width) = match projection.kind {
        NativeFp4ProjectionKind::Gate | NativeFp4ProjectionKind::Up => {
            (0, projection.logical_columns / NATIVE_FP4_K_BLOCK)
        }
        NativeFp4ProjectionKind::Down => (
            intermediate_start / NATIVE_FP4_K_BLOCK,
            local_intermediate_size / NATIVE_FP4_K_BLOCK,
        ),
    };
    let weight = NativeFp4TpTensorWindow {
        tensor: projection.weight,
        row_start,
        row_count,
        column_start_bytes: weight_column_start,
        row_width_bytes: weight_row_width,
    };
    let scale = NativeFp4TpTensorWindow {
        tensor: projection.scale,
        row_start,
        row_count,
        column_start_bytes: scale_column_start,
        row_width_bytes: scale_row_width,
    };
    validate_tp_window(weight)?;
    validate_tp_window(scale)?;
    Ok(NativeFp4TpProjectionShard {
        kind: projection.kind,
        weight,
        scale,
        local_intermediate_size,
    })
}

fn validate_tp_window(window: NativeFp4TpTensorWindow<'_>) -> Result<()> {
    let source_rows = window
        .tensor
        .shape
        .first()
        .copied()
        .context("native FP4 TP tensor has no row count")?;
    let source_row_width = window
        .tensor
        .shape
        .get(1)
        .copied()
        .context("native FP4 TP tensor has no row width")?;
    anyhow::ensure!(
        window.row_start <= source_rows
            && window.row_count <= source_rows.saturating_sub(window.row_start),
        "native FP4 TP row window {}+{} exceeds {} rows for {}",
        window.row_start,
        window.row_count,
        source_rows,
        window.tensor.name
    );
    anyhow::ensure!(
        window.column_start_bytes <= source_row_width
            && window.row_width_bytes <= source_row_width.saturating_sub(window.column_start_bytes),
        "native FP4 TP column window {}+{} exceeds {} bytes for {}",
        window.column_start_bytes,
        window.row_width_bytes,
        source_row_width,
        window.tensor.name
    );
    Ok(())
}

fn native_fp4_projection<'a>(
    catalog: &'a TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    kind: NativeFp4ProjectionKind,
) -> Result<NativeFp4Projection<'a>> {
    let prefix = native_expert_prefix(catalog, layer_id, expert_id)?;
    let stem = kind.checkpoint_stem();
    let weight_name = format!("{prefix}.{stem}.weight");
    let scale_name = format!("{prefix}.{stem}.scale");
    let weight = catalog_tensor(catalog, &weight_name)?;
    let scale = catalog_tensor(catalog, &scale_name)?;
    let (logical_rows, logical_columns) = kind.logical_shape(catalog);
    let expected_weight_shape = [logical_rows, logical_columns / 2];
    let expected_scale_shape = [logical_rows, logical_columns / NATIVE_FP4_K_BLOCK];
    validate_native_tensor(
        weight,
        layer_id,
        expert_id,
        DType::I8,
        &expected_weight_shape,
        false,
    )?;
    validate_native_tensor(
        scale,
        layer_id,
        expert_id,
        DType::F8E8M0,
        &expected_scale_shape,
        true,
    )?;
    Ok(NativeFp4Projection {
        kind,
        weight,
        scale,
        logical_rows,
        logical_columns,
    })
}

fn native_expert_prefix(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
) -> Result<String> {
    if layer_id < catalog.facts.num_hidden_layers {
        return Ok(format!("layers.{layer_id}.ffn.experts.{expert_id}"));
    }
    let block_id = layer_id - catalog.facts.num_hidden_layers;
    anyhow::ensure!(
        block_id < catalog.facts.dspark_target_layer_ids.len(),
        "dSpark block {block_id} exceeds {} checkpoint blocks",
        catalog.facts.dspark_target_layer_ids.len()
    );
    Ok(format!("mtp.{block_id}.ffn.experts.{expert_id}"))
}

fn validate_native_tensor(
    tensor: &TensorInfo,
    layer_id: usize,
    expert_id: usize,
    expected_dtype: DType,
    expected_shape: &[usize],
    quantization_metadata: bool,
) -> Result<()> {
    anyhow::ensure!(
        tensor.dtype == expected_dtype,
        "tensor {} has dtype {:?}, expected {:?}",
        tensor.name,
        tensor.dtype,
        expected_dtype
    );
    anyhow::ensure!(
        tensor.shape == expected_shape,
        "tensor {} has shape {:?}, expected {:?}",
        tensor.name,
        tensor.shape,
        expected_shape
    );
    anyhow::ensure!(
        tensor.role == TensorRole::RoutedExpert
            && tensor.layer_id == Some(layer_id as u32)
            && tensor.expert_id == Some(expert_id as u32),
        "tensor {} has inconsistent routed identity role={:?} layer={:?} expert={:?}",
        tensor.name,
        tensor.role,
        tensor.layer_id,
        tensor.expert_id
    );
    anyhow::ensure!(
        tensor.is_quantization_metadata == quantization_metadata,
        "tensor {} quantization metadata flag is {}, expected {}",
        tensor.name,
        tensor.is_quantization_metadata,
        quantization_metadata
    );
    let expected_bytes = expected_shape.iter().try_fold(1_u64, |size, dim| {
        size.checked_mul(*dim as u64)
            .context("native FP4 tensor byte size overflow")
    })?;
    anyhow::ensure!(
        tensor.byte_length == expected_bytes,
        "tensor {} records {} bytes, expected {}",
        tensor.name,
        tensor.byte_length,
        expected_bytes
    );
    Ok(())
}

fn catalog_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
        .ok()
        .and_then(|index| catalog.tensors.get(index))
        .with_context(|| format!("native FP4 tensor {name} not found in catalog"))
}
