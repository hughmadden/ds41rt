//! Storage: the filesystem beneath the store, with fault injection for tests.
//!
//! Publication order for every object part: write to a temporary path → fsync file → atomic
//! rename → fsync directory. A reader never observes a torn object. `FaultInjector` can crash
//! (return an error and leave the temporary file) between any two of those steps, corrupt bytes
//! on read, report `ENOSPC`, or add latency per byte to model a slow device.
use anyhow::Result;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    CrashBeforeRename,
    CrashAfterRename,
    CorruptOnRead { byte: usize },
    NoSpace,
    Slow { nanos_per_byte: u64 },
}

pub trait Storage {
    fn root(&self) -> &Path;
    /// Atomically publish `bytes` at `rel`; parents created as needed.
    fn publish(&mut self, rel: &Path, bytes: &[u8]) -> Result<()>;
    fn read(&self, rel: &Path) -> Result<Vec<u8>>;
    fn exists(&self, rel: &Path) -> bool;
    fn delete(&mut self, rel: &Path) -> Result<()>;
    fn list(&self, rel_dir: &Path) -> Result<Vec<PathBuf>>;
    fn bytes_used(&self) -> Result<u64>;
    /// Arm one fault for the next matching operation (tests only; production ignores).
    fn inject(&mut self, fault: Fault);
}
