//! Read-only, demand-paged checkpoint rows for engram tables and scale tables.
//!
//! Mapping does not read the payload or allocate a table-sized staging buffer.
use anyhow::{ensure, Context, Result};
use std::collections::BTreeSet;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;

/// An immutable checkpoint tensor viewed as fixed-width byte rows.
///
/// The caller must keep the underlying file immutable for the mapping lifetime.
/// Truncating a mapped checkpoint can cause SIGBUS even with a read-only mapping.
pub struct MappedRows {
    base: NonNull<u8>,
    mapped_len: usize,
    data_offset: usize,
    rows: u64,
    row_bytes: usize,
    page_bytes: usize,
    _file: File,
}

// SAFETY: Only immutable access is exposed; munmap runs after the final owner drops.
unsafe impl Send for MappedRows {}
unsafe impl Sync for MappedRows {}

impl MappedRows {
    /// Map a tensor payload whose offset is absolute within a checkpoint shard.
    ///
    /// # Safety
    /// The file must not be modified or truncated while this mapping exists.
    pub unsafe fn open(path: &Path, offset: u64, rows: u64, row_bytes: usize) -> Result<Self> {
        ensure!(
            rows > 0 && row_bytes > 0,
            "mapped tensor dimensions must be nonzero"
        );
        let payload = rows
            .checked_mul(row_bytes as u64)
            .context("mapped tensor size overflow")?;
        let end = offset
            .checked_add(payload)
            .context("mapped tensor end overflow")?;
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        ensure!(
            end <= file.metadata()?.len(),
            "mapped tensor exceeds checkpoint shard length"
        );
        let page_bytes = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        ensure!(page_bytes > 0, "cannot determine system page size");
        let page_bytes = page_bytes as usize;
        let aligned = offset / page_bytes as u64 * page_bytes as u64;
        let data_offset = usize::try_from(offset - aligned)?;
        let mapped_len = usize::try_from(end - aligned)?;
        ensure!(
            mapped_len <= isize::MAX as usize,
            "mapped tensor exceeds addressable slice size"
        );
        let file_offset = libc::off_t::try_from(aligned)?;
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                file_offset,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error()).context("mapping checkpoint tensor");
        }
        // mmap with a null hint on supported Linux hosts returns a non-null mapping.
        let Some(base) = NonNull::new(raw.cast::<u8>()) else {
            unsafe {
                libc::munmap(raw, mapped_len);
            }
            anyhow::bail!("checkpoint mapping returned a null address");
        };
        let result = Self {
            base,
            mapped_len,
            data_offset,
            rows,
            row_bytes,
            page_bytes,
            _file: file,
        };
        // Avoid kernel sequential read-ahead across a hundreds-of-GB hash table.
        if unsafe { libc::madvise(raw, mapped_len, libc::MADV_RANDOM) } != 0 {
            return Err(std::io::Error::last_os_error())
                .context("setting random checkpoint access");
        }
        Ok(result)
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    fn row_offset(&self, row: u64) -> Result<usize> {
        ensure!(
            row < self.rows,
            "mapped row {row} exceeds table rows {}",
            self.rows
        );
        // Use checked 64-bit arithmetic before conversion, including for high row IDs.
        let offset = row
            .checked_mul(self.row_bytes as u64)
            .context("row offset overflow")?;
        self.data_offset
            .checked_add(usize::try_from(offset)?)
            .context("mapped row offset overflow")
    }

    /// Copy selected rows in input order, retaining duplicates and allocating no staging memory.
    pub fn gather_into(&self, rows: &[u64], output: &mut [u8]) -> Result<()> {
        let size = rows
            .len()
            .checked_mul(self.row_bytes)
            .context("gather size overflow")?;
        ensure!(
            output.len() == size,
            "gather output size does not match requested rows"
        );
        // Validate the whole batch before modifying any output.
        for &row in rows {
            self.row_offset(row)?;
        }
        for (&row, destination) in rows.iter().zip(output.chunks_exact_mut(self.row_bytes)) {
            let offset = self.row_offset(row)?;
            let source = unsafe {
                std::slice::from_raw_parts(self.base.as_ptr().add(offset), self.row_bytes)
            };
            destination.copy_from_slice(source);
        }
        Ok(())
    }

    /// Advise only the deduplicated pages touched by this batch, with a hard page budget.
    ///
    /// WILLNEED starts OS read-ahead but does not guarantee residency or completion;
    /// gather_into remains the synchronization point for rows that fault in late.
    /// Call this on an I/O worker, not on a CUDA replay thread.
    pub fn prefetch(&self, rows: &[u64], max_pages: usize) -> Result<usize> {
        let mut pages = BTreeSet::new();
        for &row in rows {
            let start = self.row_offset(row)?;
            let last = (start + self.row_bytes - 1) / self.page_bytes;
            for page in start / self.page_bytes..=last {
                pages.insert(page);
                ensure!(
                    pages.len() <= max_pages,
                    "prefetch batch exceeds page budget {max_pages}"
                );
            }
        }
        // Coalesce adjacent pages into a single syscall, never including unrelated gaps.
        let mut pages = pages.iter().copied().peekable();
        let count = pages.len();
        while let Some(first) = pages.next() {
            let mut last = first;
            while pages.peek().is_some_and(|next| *next == last + 1) {
                last = pages.next().unwrap();
            }
            let start = first * self.page_bytes;
            let end = ((last + 1) * self.page_bytes).min(self.mapped_len);
            if unsafe {
                libc::madvise(
                    self.base.as_ptr().add(start).cast(),
                    end - start,
                    libc::MADV_WILLNEED,
                )
            } != 0
            {
                return Err(std::io::Error::last_os_error()).context("prefetching checkpoint rows");
            }
        }
        Ok(count)
    }
}

impl Drop for MappedRows {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.as_ptr().cast(), self.mapped_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn unaligned_payload_gather_preserves_order_and_rejects_bad_batches() -> Result<()> {
        let mut shard = tempfile::NamedTempFile::new()?;
        shard.write_all(&[99; 13])?;
        shard.write_all(&[1, 2, 3, 4, 5, 6, 7, 8])?;
        let table = unsafe { MappedRows::open(shard.path(), 13, 4, 2)? };
        assert_eq!(table.prefetch(&[3, 1, 3], 1)?, 1);
        let mut out = [0; 6];
        table.gather_into(&[3, 1, 3], &mut out)?;
        assert_eq!(out, [7, 8, 3, 4, 7, 8]);
        assert!(table.gather_into(&[0, 4, 1], &mut out).is_err());
        assert_eq!(out, [7, 8, 3, 4, 7, 8]);
        assert!(table.prefetch(&[0], 0).is_err());
        assert_eq!(table.prefetch(&[], 0)?, 0);
        assert!(unsafe { MappedRows::open(shard.path(), 14, 4, 2) }.is_err());
        Ok(())
    }

    #[test]
    fn sparse_checkpoint_rows_past_two_gib_use_wide_offsets() -> Result<()> {
        let mut shard = tempfile::NamedTempFile::new()?;
        let row_bytes = 256;
        let high = (1_u64 << 31) / row_bytes as u64 + 7;
        shard.as_file().set_len((high + 1) * row_bytes as u64)?;
        shard.seek(SeekFrom::Start(high * row_bytes as u64))?;
        shard.write_all(&[0x5a; 256])?;
        let table = unsafe { MappedRows::open(shard.path(), 0, high + 1, row_bytes)? };
        assert_eq!(table.prefetch(&[high, high], 1)?, 1);
        let mut out = [0; 256];
        table.gather_into(&[high], &mut out)?;
        assert_eq!(out, [0x5a; 256]);
        assert!(table.prefetch(&[0, high], 1).is_err());
        Ok(())
    }
}
