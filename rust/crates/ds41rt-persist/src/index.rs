//! The durable index (Phase 1, packet P1-1): which snapshots are on disk, their key-space
//! tokens, and the page reference counts that make content-addressed pages safe to delete.
//!
//! Two views of one truth:
//! - **Durable rows** in sqlite (`objects`, `pages`). Every mutation is one transaction, so a
//!   crash between statements leaves the rows consistent with each other. The rows are a cache
//!   of what `Storage` holds, never the authority: [`Index::rebuild`] regenerates them from the
//!   published `meta.json` files when the database is missing or fails its open check.
//! - **In-RAM views** loaded at open: the durable-key set ([`Index::contains`] touches neither
//!   sqlite nor storage) and a [`Retention`] radix over the stored token sequences, so
//!   [`Index::lookup_longest`] applies exactly the reuse rule the in-memory cache applies and a
//!   disk hit is only ever a snapshot the in-memory tier would itself have chosen.
//!
//! Accounting: every stored page is the same size ([`PAGE_STORED_BYTES`]) and is counted once
//! however many objects reference it; an object's own parts are counted in its
//! [`Entry::part_bytes`]. `bytes_used` is therefore exact under sharing and never drifts.
//!
//! Concurrency contract: an implementation is `Send`; the store serialises access behind one
//! lock. `contains` and `lookup_longest` are sub-millisecond hash and radix operations;
//! `insert`, `remove` and `touch_restored` each cost one sqlite transaction. Keeping the
//! scheduler's hot path off that transaction is the store's job (Phase 3), not the index's.
//!
//! Memory: the radix holds every stored token sequence with shared prefixes deduplicated; the
//! quota (Phase 1, P1-3) bounds it indirectly. A hashed-chain index replaces it only if measured
//! RAM exceeds budget.
use crate::object::PageRef;
use anyhow::Result;
use ds41rt_core::prefix::{Retention, SnapshotKind};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use ds41rt_core::prefix::Retention as PrefixRetention;

/// Bytes one page occupies on disk: its rows plus the trailing checksum.
pub const PAGE_STORED_BYTES: u64 = (crate::PAGE_BYTES + crate::CHECKSUM_BYTES) as u64;

/// Why an index mutation was refused. Every variant names the key, so the caller can log it
/// without reconstructing context.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum IndexError {
    /// `insert` of a key that is already present. A duplicate put is a logic error in the
    /// caller (it must check `contains` first), never a race to paper over.
    #[error("index already holds {key}")]
    Duplicate { key: String },
    /// `remove` or `touch_restored` of a key that is not present.
    #[error("index does not hold {key}")]
    NotFound { key: String },
    /// The database exists but failed its open check; the caller deletes it and rebuilds.
    #[error("index database unusable: {reason}")]
    Corrupt { reason: String },
}

/// What the index records about one stored snapshot: enough to answer lookups, order eviction
/// and account bytes without opening the object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// [`snapshot_key`] of the namespace fingerprint and `tokens`.
    pub key: String,
    /// Key-space tokens: image spans are already folded into ids by the engine.
    pub tokens: Vec<u32>,
    /// Tokens the snapshot covers; equals `tokens.len()` for a whole-prefix snapshot.
    pub end: u64,
    pub kind: SnapshotKind,
    /// Encoded bytes of the object's own parts (meta, tail, draft, scores). Pages are not
    /// included: they are accounted once each, in the page table.
    pub part_bytes: u64,
    /// Every page the object references, in `(compressor, logical)` order.
    pub pages: Vec<PageRef>,
    pub created_unix: u64,
    /// Zero until the first restore. Eviction takes the smallest first.
    pub last_restored_unix: u64,
}

/// A reusable stored prefix. `common` query tokens match; the snapshot's frontier is
/// `frontier` tokens. `common == frontier` is an exact ancestor; otherwise the caller replays
/// from the aligned window the shared rule defines (`Reusable::skipped` in
/// `ds41rt_core::prefix`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    pub common: usize,
    pub frontier: usize,
    pub key: String,
    pub kind: SnapshotKind,
}

/// Content-derived key of a snapshot: lowercase hex sha256 over the namespace fingerprint's
/// bytes followed by every token as little-endian `u32`. Equal prompts under one namespace
/// share one object; a namespace change changes every key.
pub fn snapshot_key(namespace_fingerprint: &str, tokens: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(namespace_fingerprint.as_bytes());
    for token in tokens {
        h.update(token.to_le_bytes());
    }
    crate::object::hex(&h.finalize())
}

/// The index operations. Invariants every implementation keeps:
/// - a key is present exactly when `insert` succeeded for it and no `remove` has since;
/// - the reference count of a page equals the number of present entries that list it, and a
///   page row exists exactly when that count is positive;
/// - `bytes_used == Σ part_bytes over present entries + page rows × PAGE_STORED_BYTES`;
/// - after `open` of a database that some earlier instance mutated, every query answers as the
///   earlier instance would have answered after its last successful mutation.
pub trait Index: Send {
    /// Record a stored object and add one reference to each of its pages (a page seen for the
    /// first time gets a row with one reference). Atomic. [`IndexError::Duplicate`] if present.
    fn insert(&mut self, entry: Entry) -> Result<()>;
    /// Forget an object and release its page references; returns the pages whose count reached
    /// zero, in the order they appeared in the entry (the caller deletes those files after
    /// this returns). Atomic. [`IndexError::NotFound`] if absent.
    fn remove(&mut self, key: &str) -> Result<Vec<PageRef>>;
    /// Is exactly this key present? Answered from RAM: no sqlite, no storage.
    fn contains(&self, key: &str) -> bool;
    fn entry(&self, key: &str) -> Result<Option<Entry>>;
    /// The stored prefix that saves the most work for `tokens` under the shared radix rule;
    /// `None` when nothing saves any work. Does not change eviction order.
    fn lookup_longest(&mut self, tokens: &[u32]) -> Option<Match>;
    /// Mark a successful restore; eviction order follows this. [`IndexError::NotFound`] if absent.
    fn touch_restored(&mut self, key: &str, now_unix: u64) -> Result<()>;
    /// Present entries that reference `page`; zero for an unknown page.
    fn page_refs(&self, page: &PageRef) -> u32;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn bytes_used(&self) -> u64;
    /// Every present key, least recently restored first (never restored sorts before every
    /// restored key), then oldest created, then by key so the order is total.
    fn eviction_candidates(&self) -> Vec<String>;
    /// Replace every row with `entries` (the boot-time scan of published objects). Atomic: a
    /// failure leaves the previous rows and views untouched.
    fn rebuild(&mut self, entries: &mut dyn Iterator<Item = Entry>) -> Result<()>;
}

/// The radix view the implementations share: one bank per [`SnapshotKind`], holding the key
/// of each stored sequence, with no entry limit (the quota bounds the store instead).
pub fn unbounded_retention() -> Retention<String> {
    Retention::new(usize::MAX)
}
