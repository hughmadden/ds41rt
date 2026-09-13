//! Bounded snapshot storage allocated once, before serving becomes ready.
use super::*;
use anyhow::{ensure, Context};
use std::{cell::RefCell, rc::Rc};

struct Arena<'a> {
    allocation: DeviceAllocation<'a>,
    stride: usize,
    free: RefCell<Vec<usize>>,
}

#[derive(Clone)]
pub(crate) struct SnapshotPool<'a>(Rc<Arena<'a>>);
impl<'a> SnapshotPool<'a> {
    pub fn new(library: &'a NativeLibrary, bytes: usize, slots: usize) -> Result<Self> {
        ensure!(bytes > 0 && slots > 0, "empty snapshot arena");
        let stride = bytes.checked_add(255).context("snapshot stride overflow")? / 256 * 256;
        let allocation = DeviceAllocation::new(library,
            stride.checked_mul(slots).context("snapshot arena overflow")?)?;
        Ok(Self(Rc::new(Arena { allocation, stride, free: RefCell::new((0..slots).rev().collect()) })))
    }
    pub fn device_bytes(&self) -> usize { self.0.allocation.buffer.bytes }
    pub fn take(&self, bytes: usize) -> Result<SnapshotStorage<'a>> {
        ensure!(bytes > 0 && bytes <= self.0.stride, "snapshot exceeds arena slot");
        let slot = self.0.free.borrow_mut().pop().context("snapshot arena exhausted")?;
        let mut buffer = self.0.allocation.buffer;
        buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(slot * self.0.stride).cast() };
        buffer.bytes = bytes;
        Ok(SnapshotStorage { buffer, _owned: None, pooled: Some((self.clone(), slot)) })
    }
}

/// The owner must drain every reader/writer before dropping this storage.
/// A pooled slot returns to the free list without calling the CUDA allocator.
pub(crate) struct SnapshotStorage<'a> {
    pub buffer: Ds41rtDeviceBuffer,
    _owned: Option<DeviceAllocation<'a>>,
    pooled: Option<(SnapshotPool<'a>, usize)>,
}
impl<'a> SnapshotStorage<'a> {
    pub fn new(library: &'a NativeLibrary, bytes: usize, pool: Option<&SnapshotPool<'a>>) -> Result<Self> {
        if let Some(pool) = pool { return pool.take(bytes); }
        let owned = DeviceAllocation::new(library, bytes)?;
        Ok(Self { buffer: owned.buffer, _owned: Some(owned), pooled: None })
    }
}
impl Drop for SnapshotStorage<'_> {
    fn drop(&mut self) {
        if let Some((pool, slot)) = self.pooled.take() { pool.0.free.borrow_mut().push(slot); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA"]
    fn native_snapshot_arena_reuses_only_released_slots() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let pool = SnapshotPool::new(&lib, 513, 2)?;
        assert!(pool.take(769).is_err());
        let first = pool.take(513)?;
        let second = pool.take(129)?;
        assert_ne!(first.buffer.ptr, second.buffer.ptr);
        assert!(pool.take(1).is_err());
        lib.copy_h2d(first.buffer, &[17; 513])?;
        lib.copy_h2d(second.buffer, &[29; 129])?;
        let address = first.buffer.ptr;
        drop(first);
        let reused = pool.take(513)?;
        assert_eq!(reused.buffer.ptr, address);
        lib.copy_h2d(reused.buffer, &[41; 513])?;
        // Outstanding storage owns the arena even when its allocator is dropped.
        drop(pool);
        let mut actual = [0; 129];
        lib.copy_d2h(&mut actual, second.buffer)?;
        assert_eq!(actual, [29; 129]);
        Ok(())
    }
}
