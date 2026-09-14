//! Persisted object model and the namespace fingerprint.
//!
//! One object per snapshot: `Meta` (tokens or their hash chain, end, kind, draft present, image
//! keys, page refs per compressor), `tail` (`TAIL_BYTES`), optional `draft` (`DRAFT_BYTES`),
//! `scores` (one logit row). Pages are content-addressed (`sha256` of the page bytes) so shared
//! prefixes dedupe on disk exactly as refcounts share them in memory. Every part carries its
//! checksum; a decoder rejects a part whose checksum fails before returning any byte.
use serde::{Deserialize, Serialize};

/// Everything that changes the cache layout; a mismatch is a clean miss, never a load.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Namespace {
    pub model_revision: String,
    pub quant: String,
    pub engine_build: String,
    pub aot_build: String,
    pub layout_version: u32,
}
impl Namespace {
    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(serde_json::to_vec(self).expect("namespace serialises"));
        hex(&h.finalize())
    }
}
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SnapshotKind {
    Prompt,
    Turn,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageRef {
    pub compressor: u8,
    pub logical: u32,
    pub hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Meta {
    pub key: String,
    pub tokens: Vec<u32>,
    pub end: u64,
    pub kind: SnapshotKind,
    pub has_draft: bool,
    pub rows_per_compressor: [u32; 4],
    pub pages: Vec<PageRef>,
    pub tail_sha256: String,
    pub draft_sha256: Option<String>,
    pub scores_sha256: String,
    pub created_unix: u64,
}
