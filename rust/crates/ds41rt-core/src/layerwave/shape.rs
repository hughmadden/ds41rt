use serde::{Deserialize, Serialize};

use crate::{ModelFacts, PositionId, RequestId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LayerWaveMode {
    Decode,
    Prefill,
    MtpVerify,
    Benchmark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphBucket {
    pub row_capacity: usize,
}

impl GraphBucket {
    pub fn new(row_capacity: usize) -> Self {
        Self {
            row_capacity: row_capacity.max(1),
        }
    }

    pub fn decode() -> Self {
        Self::new(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HiddenShape {
    pub rows: usize,
    pub hidden_dim: usize,
    pub bytes_per_row: usize,
}

impl HiddenShape {
    pub fn bf16_rows(rows: usize, hidden_dim: usize) -> Self {
        Self {
            rows,
            hidden_dim,
            bytes_per_row: hidden_dim * std::mem::size_of::<u16>(),
        }
    }

    pub fn for_model_bf16(rows: usize, facts: &ModelFacts) -> Self {
        Self::bf16_rows(rows, facts.hidden_size)
    }

    pub fn ds4_flash_bf16_rows(rows: usize) -> Self {
        Self::for_model_bf16(rows, &ModelFacts::default())
    }

    pub fn payload_bytes(self) -> usize {
        self.rows * self.bytes_per_row
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RowSourceKind {
    DecodeStep,
    PrefillChunk,
    MtpVerifyBlock,
    Benchmark,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowSource {
    pub kind: RowSourceKind,
    pub request_id: RequestId,
    pub sequence_id: String,
    pub token_start: PositionId,
    pub row_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteMetadataPlaceholder {
    pub top_k: usize,
    pub routed_experts: usize,
}

impl Default for RouteMetadataPlaceholder {
    fn default() -> Self {
        Self::for_model(&ModelFacts::default())
    }
}

impl RouteMetadataPlaceholder {
    pub fn for_model(facts: &ModelFacts) -> Self {
        Self {
            top_k: facts.top_k,
            routed_experts: facts.routed_experts,
        }
    }
}
