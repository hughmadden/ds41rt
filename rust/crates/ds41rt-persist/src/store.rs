//! The snapshot store (Phase 1): objects + pages + index + quota over `Storage`.
use crate::object::{Meta, Namespace};
use anyhow::Result;

pub trait SnapshotStore {
    fn namespace(&self) -> &Namespace;
    /// Longest stored prefix of `tokens` under the radix rule: returns `(common, frontier, key)`.
    fn lookup_longest(&mut self, tokens: &[u32]) -> Option<(usize, usize, String)>;
    /// Cheap negative: is any object with exactly this key present? Zero storage I/O.
    fn contains(&self, key: &str) -> bool;
    fn meta(&self, key: &str) -> Result<Meta>;
    fn bytes_used(&self) -> u64;
    fn quota(&self) -> u64;
}
