//! Snapshot arena: fixed-stride device slots for backbone tails and draft rings.
//!
//! Mirrors `v41_memory/snapshot.rs::SnapshotPool`: allocated once before serving, `take(bytes)`
//! refuses `bytes > stride` and fails when exhausted, a slot returns to the free list when its
//! storage drops without touching the allocator, outstanding storage keeps the arena alive.
use crate::copy::DeviceBuf;
use anyhow::Result;

/// Fixed-stride device slots for snapshot tails. Invariant: `slots() == free_slots() + outstanding`
/// and no slot is handed out twice while outstanding.
pub trait SnapshotArena {
    fn stride(&self) -> usize;
    fn slots(&self) -> usize;
    fn free_slots(&self) -> usize;
    /// Take a slot of at most `stride` bytes.
    fn take(&mut self, bytes: usize) -> Result<ArenaSlot>;
    /// Return a slot. Callers must have drained every stream touching it. Fails on a slot that is
    /// not outstanding (double return, or a slot from another arena).
    fn give_back(&mut self, slot: ArenaSlot) -> Result<()>;
}

/// An outstanding arena slot; `buf` is the slot's device range.
#[derive(Debug)]
pub struct ArenaSlot {
    pub index: usize,
    pub buf: DeviceBuf,
}
