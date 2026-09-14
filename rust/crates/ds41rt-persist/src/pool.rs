//! Page pool: the shared, refcounted, copy-on-write source pages of one compressor.
//!
//! Mirrors `v41_compressor/source_cache.rs` + `ownership.rs`: pages of `PAGE_ROWS` rows; each of
//! 16 request slots owns an ordered page list; `reserve(appends)` atomically claims free pages
//! (and replacement pages for shared tails that will be written: copy-on-write) or fails with
//! `PoolExhausted { work_index, needed, available }` naming the first append that did not fit;
//! `apply(plan)` publishes the plan (retain new pages, release replaced ones); a `SourcePrefix`
//! retains pages by refcount and may be truncated to fewer rows; releasing a slot releases its
//! pages; a page is free only when its refcount is zero.
use anyhow::Result;
use thiserror::Error;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("compressed KV pool exhausted at work item {work_index}: need {needed} pages, {available} available")]
pub struct PoolExhausted {
    pub work_index: usize,
    pub needed: usize,
    pub available: usize,
}

/// An append: `(slot, old_rows, new_rows)`.
pub type Append = (usize, usize, usize);

/// A reserved-but-unpublished plan. Dropping without `apply` releases the reservation.
pub trait Plan {
    /// Physical destination row for `(slot, logical_row)` under this plan.
    fn destination(&self, slot: usize, row: usize) -> Result<u64>;
}

/// A retained, immutable view of the first `rows` rows of a slot: the pages it references stay
/// alive while the prefix exists.
pub trait SourcePrefix: Clone {
    fn rows(&self) -> usize;
    fn pages(&self) -> &[u32];
    fn truncate(&self, rows: usize) -> Result<Self>;
}

pub trait PagePool {
    type Plan: Plan;
    type Prefix: SourcePrefix;
    fn pages(&self) -> usize;
    fn free_pages(&self) -> usize;
    fn references(&self, page: u32) -> usize;
    fn slot_rows(&self, slot: usize) -> usize;
    fn reserve(&mut self, appends: &[Append]) -> Result<Self::Plan, PoolExhausted>;
    fn apply(&mut self, plan: Self::Plan) -> Result<()>;
    fn write_rows(
        &mut self,
        plan: &Self::Plan,
        slot: usize,
        first_row: usize,
        bytes: &[u8],
    ) -> Result<()>;
    fn read_rows(&self, slot: usize, first_row: usize, rows: usize) -> Result<Vec<u8>>;
    fn retain_prefix(&mut self, slot: usize, rows: usize) -> Result<Self::Prefix>;
    /// Attach a retained prefix to an empty slot (restore).
    fn restore_prefix(&mut self, slot: usize, prefix: &Self::Prefix) -> Result<()>;
    fn release_slot(&mut self, slot: usize) -> Result<()>;
    /// Drop a prefix, releasing its page references.
    fn release_prefix(&mut self, prefix: Self::Prefix);
}
