use serde::{Deserialize, Serialize};

use crate::{
    DeepseekV4AttentionLayerPlan, DeepseekV4AttentionLayerSource, DeepseekV4AttentionPlan,
    Ds4rtError, ModelFacts,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeepseekV4CompressorKind {
    Main,
    Indexer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4CompressorSpec {
    pub kind: DeepseekV4CompressorKind,
    pub compress_ratio: usize,
    pub overlap: bool,
    pub coefficient: usize,
    pub head_dim: usize,
    pub projected_width: usize,
    pub state_rows: usize,
    pub state_width: usize,
}

impl DeepseekV4CompressorSpec {
    pub fn paired_state_bytes(&self) -> Option<usize> {
        self.state_rows
            .checked_mul(self.state_width)?
            .checked_mul(std::mem::size_of::<f32>())?
            .checked_mul(2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4CompressorLayerExecutionPlan {
    pub logical_layer_id: usize,
    pub source: DeepseekV4AttentionLayerSource,
    pub main: Option<DeepseekV4CompressorSpec>,
    pub indexer: Option<DeepseekV4CompressorSpec>,
    /// One load-time concatenated BF16 projection emits KV and gate channels
    /// for every compressor attached to the layer.
    pub joint_projection_width: usize,
}

impl DeepseekV4CompressorLayerExecutionPlan {
    pub fn for_layer(
        layer: &DeepseekV4AttentionLayerPlan,
        facts: &ModelFacts,
    ) -> Result<Self, Ds4rtError> {
        if !layer.uses_compressor() {
            return Ok(Self {
                logical_layer_id: layer.logical_layer_id,
                source: layer.source,
                main: None,
                indexer: None,
                joint_projection_width: 0,
            });
        }
        let main = compressor_spec(
            DeepseekV4CompressorKind::Main,
            layer.compress_ratio,
            facts.head_dim,
        )?;
        let indexer = if layer.uses_indexer() {
            Some(compressor_spec(
                DeepseekV4CompressorKind::Indexer,
                layer.compress_ratio,
                facts.index_head_dim,
            )?)
        } else {
            None
        };
        let joint_projection_width = projection_pair_width(&main)?
            .checked_add(
                indexer
                    .as_ref()
                    .map(projection_pair_width)
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or_else(|| invalid_compressor("joint projection width overflow"))?;
        Ok(Self {
            logical_layer_id: layer.logical_layer_id,
            source: layer.source,
            main: Some(main),
            indexer,
            joint_projection_width,
        })
    }

    pub fn paired_state_bytes(&self) -> Option<usize> {
        self.main
            .as_ref()
            .and_then(DeepseekV4CompressorSpec::paired_state_bytes)
            .unwrap_or(0)
            .checked_add(
                self.indexer
                    .as_ref()
                    .and_then(DeepseekV4CompressorSpec::paired_state_bytes)
                    .unwrap_or(0),
            )
    }

    pub fn decode_step(
        &self,
        logical_position: usize,
    ) -> Result<Option<DeepseekV4CompressorDecodeStep>, Ds4rtError> {
        let Some(main) = self.main.as_ref() else {
            return Ok(None);
        };
        let lane = logical_position % main.compress_ratio;
        let emits = lane + 1 == main.compress_ratio;
        let rope_position = if emits {
            Some(
                logical_position
                    .checked_add(1)
                    .and_then(|end| end.checked_sub(main.compress_ratio))
                    .ok_or_else(|| invalid_compressor("decode RoPE position overflow"))?,
            )
        } else {
            None
        };
        Ok(Some(DeepseekV4CompressorDecodeStep {
            logical_position,
            ape_row: lane,
            state_row: logical_position % main.state_rows,
            emits,
            compressed_slot: emits.then_some(logical_position / main.compress_ratio),
            rope_position,
            rolls_current_window_to_previous: false,
        }))
    }

    pub fn prefill(
        &self,
        source_tokens: usize,
    ) -> Result<Option<DeepseekV4CompressorPrefillPlan>, Ds4rtError> {
        let Some(main) = self.main.as_ref() else {
            return Ok(None);
        };
        if source_tokens == 0 {
            return Err(invalid_compressor(
                "compressor prefill must contain at least one source token",
            ));
        }
        let ratio = main.compress_ratio;
        let complete_groups = source_tokens / ratio;
        let cutoff = complete_groups
            .checked_mul(ratio)
            .ok_or_else(|| invalid_compressor("prefill cutoff overflow"))?;
        let remainder = source_tokens - cutoff;
        let previous_window = if main.overlap && complete_groups > 0 {
            Some(DeepseekV4CompressorStateFill {
                source_start: cutoff - ratio,
                rows: ratio,
                state_row_start: 0,
                ape_row_start: 0,
            })
        } else {
            None
        };
        let current_window = (remainder > 0).then_some(DeepseekV4CompressorStateFill {
            source_start: cutoff,
            rows: remainder,
            state_row_start: if main.overlap { ratio } else { 0 },
            ape_row_start: 0,
        });
        Ok(Some(DeepseekV4CompressorPrefillPlan {
            source_tokens,
            compress_ratio: ratio,
            complete_groups,
            remainder,
            output_slot_start: 0,
            output_rope_position_start: 0,
            output_rope_position_stride: ratio,
            reset_state_before_projection: true,
            first_overlap_half_is_inactive: main.overlap && complete_groups > 0,
            previous_window,
            current_window,
        }))
    }

    pub fn continuation_prefill(
        &self,
        logical_start: usize,
        source_tokens: usize,
    ) -> Result<Option<DeepseekV4CompressorContinuationPlan>, Ds4rtError> {
        let Some(main) = self.main.as_ref() else {
            return Ok(None);
        };
        if source_tokens == 0 {
            return Err(invalid_compressor(
                "compressor continuation must contain at least one source token",
            ));
        }
        let logical_end = logical_start
            .checked_add(source_tokens)
            .ok_or_else(|| invalid_compressor("continuation logical end overflow"))?;
        let ratio = main.compress_ratio;
        let initial_group_index = logical_start / ratio;
        let initial_group_start = initial_group_index
            .checked_mul(ratio)
            .ok_or_else(|| invalid_compressor("continuation group start overflow"))?;
        let terminal_group_index = logical_end / ratio;
        let terminal_group_start = terminal_group_index
            .checked_mul(ratio)
            .ok_or_else(|| invalid_compressor("continuation cutoff overflow"))?;
        let complete_groups = terminal_group_index - initial_group_index;
        let first_output_group_start = (complete_groups > 0).then_some(initial_group_start);
        let carried_remainder = logical_start - initial_group_start;
        let terminal_remainder = logical_end - terminal_group_start;
        let carried_previous_window = if main.overlap && initial_group_start >= ratio {
            Some(DeepseekV4CompressorStateFill {
                source_start: initial_group_start - ratio,
                rows: ratio,
                state_row_start: 0,
                ape_row_start: 0,
            })
        } else {
            None
        };
        let carried_current_window =
            (carried_remainder > 0).then_some(DeepseekV4CompressorStateFill {
                source_start: initial_group_start,
                rows: carried_remainder,
                state_row_start: if main.overlap { ratio } else { 0 },
                ape_row_start: 0,
            });
        let terminal_previous_window = if main.overlap && terminal_group_start >= ratio {
            Some(DeepseekV4CompressorStateFill {
                source_start: terminal_group_start - ratio,
                rows: ratio,
                state_row_start: 0,
                ape_row_start: 0,
            })
        } else {
            None
        };
        let terminal_current_window =
            (terminal_remainder > 0).then_some(DeepseekV4CompressorStateFill {
                source_start: terminal_group_start,
                rows: terminal_remainder,
                state_row_start: if main.overlap { ratio } else { 0 },
                ape_row_start: 0,
            });
        Ok(Some(DeepseekV4CompressorContinuationPlan {
            logical_start,
            source_tokens,
            logical_end,
            compress_ratio: ratio,
            complete_groups,
            carried_remainder,
            terminal_remainder,
            first_output_group_start,
            output_slot_start: first_output_group_start.map(|start| start / ratio),
            output_rope_position_stride: ratio,
            carried_previous_window,
            carried_current_window,
            terminal_previous_window,
            terminal_current_window,
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4CompressorDecodeStep {
    pub logical_position: usize,
    pub ape_row: usize,
    pub state_row: usize,
    pub emits: bool,
    pub compressed_slot: Option<usize>,
    pub rope_position: Option<usize>,
    pub rolls_current_window_to_previous: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4CompressorStateFill {
    pub source_start: usize,
    pub rows: usize,
    pub state_row_start: usize,
    pub ape_row_start: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4CompressorPrefillPlan {
    pub source_tokens: usize,
    pub compress_ratio: usize,
    pub complete_groups: usize,
    pub remainder: usize,
    pub output_slot_start: usize,
    pub output_rope_position_start: usize,
    pub output_rope_position_stride: usize,
    pub reset_state_before_projection: bool,
    pub first_overlap_half_is_inactive: bool,
    pub previous_window: Option<DeepseekV4CompressorStateFill>,
    pub current_window: Option<DeepseekV4CompressorStateFill>,
}

impl DeepseekV4CompressorPrefillPlan {
    pub fn output_rope_position(&self, output_index: usize) -> Option<usize> {
        if output_index >= self.complete_groups {
            return None;
        }
        self.output_rope_position_start
            .checked_add(output_index.checked_mul(self.output_rope_position_stride)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4CompressorContinuationPlan {
    pub logical_start: usize,
    pub source_tokens: usize,
    pub logical_end: usize,
    pub compress_ratio: usize,
    pub complete_groups: usize,
    pub carried_remainder: usize,
    pub terminal_remainder: usize,
    pub first_output_group_start: Option<usize>,
    pub output_slot_start: Option<usize>,
    pub output_rope_position_stride: usize,
    /// Logical rows already resident in the previous half of C=4 state.
    pub carried_previous_window: Option<DeepseekV4CompressorStateFill>,
    /// Logical rows already resident in the incomplete current group.
    pub carried_current_window: Option<DeepseekV4CompressorStateFill>,
    /// Exact previous C=4 window required after the continuation completes.
    pub terminal_previous_window: Option<DeepseekV4CompressorStateFill>,
    /// Exact incomplete group required after the continuation completes.
    pub terminal_current_window: Option<DeepseekV4CompressorStateFill>,
}

impl DeepseekV4CompressorContinuationPlan {
    pub fn output_group_start(&self, output_index: usize) -> Option<usize> {
        if output_index >= self.complete_groups {
            return None;
        }
        self.first_output_group_start?
            .checked_add(output_index.checked_mul(self.output_rope_position_stride)?)
    }

    pub fn output_slot(&self, output_index: usize) -> Option<usize> {
        if output_index >= self.complete_groups {
            return None;
        }
        self.output_slot_start?.checked_add(output_index)
    }

    pub fn output_rope_position(&self, output_index: usize) -> Option<usize> {
        self.output_group_start(output_index)
    }
}

pub fn deepseek_v4_compressor_execution_plans(
    facts: &ModelFacts,
) -> Result<Vec<DeepseekV4CompressorLayerExecutionPlan>, Ds4rtError> {
    DeepseekV4AttentionPlan::from_model_facts(facts)?
        .layers
        .iter()
        .map(|layer| DeepseekV4CompressorLayerExecutionPlan::for_layer(layer, facts))
        .collect()
}

fn compressor_spec(
    kind: DeepseekV4CompressorKind,
    compress_ratio: usize,
    head_dim: usize,
) -> Result<DeepseekV4CompressorSpec, Ds4rtError> {
    if !matches!(compress_ratio, 4 | 128) {
        return Err(invalid_compressor(format!(
            "compressor ratio must be 4 or 128, got {compress_ratio}"
        )));
    }
    if head_dim == 0 {
        return Err(invalid_compressor("compressor head dimension is zero"));
    }
    let overlap = compress_ratio == 4;
    let coefficient: usize = if overlap { 2 } else { 1 };
    let projected_width = coefficient
        .checked_mul(head_dim)
        .ok_or_else(|| invalid_compressor("compressor projected width overflow"))?;
    let state_rows = coefficient
        .checked_mul(compress_ratio)
        .and_then(|rows| rows.checked_mul(2))
        .ok_or_else(|| invalid_compressor("compressor state row count overflow"))?;
    Ok(DeepseekV4CompressorSpec {
        kind,
        compress_ratio,
        overlap,
        coefficient,
        head_dim,
        projected_width,
        state_rows,
        state_width: projected_width,
    })
}

fn projection_pair_width(spec: &DeepseekV4CompressorSpec) -> Result<usize, Ds4rtError> {
    spec.projected_width
        .checked_mul(2)
        .ok_or_else(|| invalid_compressor("compressor KV/gate projection width overflow"))
}

fn invalid_compressor(reason: impl Into<String>) -> Ds4rtError {
    Ds4rtError::InvalidDeepseekV4KvLayout {
        reason: reason.into(),
    }
}
