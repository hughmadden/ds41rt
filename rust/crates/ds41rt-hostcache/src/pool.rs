//! Pinned slab pool (packet HC-1): all pinned host memory is allocated at construction to the
//! configured quota in fixed chunks and never grows. Each chunk is carved for one size class
//! (page, tail, draft, scores); each class keeps a free list; a class claims a free chunk when
//! its list is empty and releases a chunk when every slab in it is free. Allocation and release
//! are O(1); bytes in use is exact and equals slabs held × class size.
use serde::Serialize;
use thiserror::Error;

/// The four slab sizes a snapshot needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Class {
    Page,
    Tail,
    Draft,
    Scores,
}

impl Class {
    pub const ALL: [Class; 4] = [Class::Page, Class::Tail, Class::Draft, Class::Scores];
}

/// Slab sizes in bytes. The engine layout uses the crate constants for pages, tails and drafts;
/// the scores row size comes from the engine at boot (verified in the daemon binding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Layout {
    pub page: usize,
    pub tail: usize,
    pub draft: usize,
    pub scores: usize,
}

impl Layout {
    pub fn engine(scores: usize) -> Self {
        Self {
            page: crate::PAGE_BYTES,
            tail: crate::TAIL_BYTES,
            draft: crate::DRAFT_BYTES,
            scores,
        }
    }
    pub fn size(&self, class: Class) -> usize {
        match class {
            Class::Page => self.page,
            Class::Tail => self.tail,
            Class::Draft => self.draft,
            Class::Scores => self.scores,
        }
    }
}

/// One pinned chunk as the memory provider knows it: an id the copy engine maps to an address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct HostChunk {
    pub id: u32,
    pub bytes: usize,
}

/// Where pinned memory comes from: the CUDA engine registers it for DMA; the stub fakes it.
pub trait PinnedMemory {
    fn allocate_chunk(&mut self, bytes: usize) -> anyhow::Result<HostChunk>;
    fn release_chunk(&mut self, chunk: HostChunk) -> anyhow::Result<()>;
}

/// A slab handle: which chunk, which index within it. Copy engines address it through
/// [`SlabPool::location`]. Handles are plain data; the pool is the authority on validity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Slab {
    pub class: Class,
    pub chunk: u32,
    pub index: u32,
}

/// A byte range inside a pinned chunk, the unit every copy is expressed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct HostRange {
    pub chunk: u32,
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("pinned pool exhausted for {class:?}: {free_bytes} bytes free, {needed_bytes} needed")]
pub struct PoolExhausted {
    pub class: Class,
    pub needed_bytes: usize,
    pub free_bytes: u64,
}

/// Per-class occupancy for the metrics export.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ClassOccupancy {
    pub slabs_in_use: u64,
    pub slabs_free: u64,
    pub chunks: u32,
}

/// The pool. Invariants: `bytes_used() <= quota()`; a `Slab` handed out by `take` is valid until
/// its `give_back`; `give_back` of a slab not held is a logic error (debug-asserted); chunks
/// carved for a class hold `chunk_bytes / size` slabs and the remainder is unused.
pub struct SlabPool {
    _private: (),
}

impl SlabPool {
    /// Allocate `quota_bytes / chunk_bytes` chunks up front through `memory`. Fails if the
    /// provider cannot supply them; a partial allocation is released before returning.
    pub fn new(
        quota_bytes: u64,
        chunk_bytes: usize,
        layout: Layout,
        memory: &mut dyn PinnedMemory,
    ) -> anyhow::Result<Self> {
        let _ = (quota_bytes, chunk_bytes, layout, memory);
        unimplemented!("HC-1")
    }
    pub fn take(&mut self, class: Class) -> Result<Slab, PoolExhausted> {
        let _ = class;
        unimplemented!("HC-1")
    }
    pub fn give_back(&mut self, slab: Slab) {
        let _ = slab;
        unimplemented!("HC-1")
    }
    /// The chunk and byte offset of a held slab; its length is the class size.
    pub fn location(&self, slab: Slab) -> HostRange {
        let _ = slab;
        unimplemented!("HC-1")
    }
    pub fn layout(&self) -> Layout {
        unimplemented!("HC-1")
    }
    pub fn bytes_used(&self) -> u64 {
        unimplemented!("HC-1")
    }
    pub fn quota(&self) -> u64 {
        unimplemented!("HC-1")
    }
    /// Bytes a `take` of `class` could still satisfy from free slabs and free chunks.
    pub fn free_bytes(&self, class: Class) -> u64 {
        let _ = class;
        unimplemented!("HC-1")
    }
    pub fn occupancy(&self) -> [(Class, ClassOccupancy); 4] {
        unimplemented!("HC-1")
    }
    /// Release every chunk back to the provider (drop order: the engine outlives the pool).
    pub fn release_all(self, memory: &mut dyn PinnedMemory) -> anyhow::Result<()> {
        let _ = memory;
        unimplemented!("HC-1")
    }
}
