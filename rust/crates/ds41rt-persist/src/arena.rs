//! Snapshot arena: fixed-stride device slots for backbone tails and draft rings.
//!
//! Mirrors `v41_memory/snapshot.rs::SnapshotPool`: allocated once before serving, `take(bytes)`
//! refuses `bytes > stride` and fails when exhausted, a slot returns to the free list when its
//! storage drops without touching the allocator, outstanding storage keeps the arena alive.
use crate::copy::DeviceBuf;
use anyhow::Result;

pub trait SnapshotArena {
    fn stride(&self) -> usize;
    fn slots(&self) -> usize;
    fn free_slots(&self) -> usize;
    /// Take a slot of at most `stride` bytes.
    fn take(&mut self, bytes: usize) -> Result<ArenaSlot>;
    /// Return a slot. Callers must have drained every stream touching it.
    fn give_back(&mut self, slot: ArenaSlot);
}

#[derive(Debug)]
pub struct ArenaSlot {
    pub index: usize,
    pub buf: DeviceBuf,
}
