//! Identity of one completed, token-bound backbone query execution.
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QueryBinding {
    snapshot: u64,
    layer: usize,
}
impl QueryBinding {
    pub fn new(layer: usize) -> Result<Self> {
        Ok(Self {
            snapshot: NEXT
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .ok()
                .context("query snapshots exhausted")?,
            layer,
        })
    }
    pub fn layer(self) -> usize {
        self.layer
    }
}
