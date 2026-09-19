//! Canonical slot read/write reservations shared by native GPU windows and CPU contract tests.
use anyhow::{ensure, Result};
use std::{cell::Cell, rc::Rc};

// Reservations outlive the short host borrow while a chain reads ring slots.
// Other request slots may be committed, but a live read slot cannot be rewritten
// or recycled until the chain has drained its GPU work.
#[derive(Clone, Default)]
pub(crate) struct SlotAccess(pub(super) Rc<Cell<[u32; 16]>>);
pub(crate) struct ReadReservation {
    slots: SlotAccess,
    used: [bool; 16],
}
pub(super) const WRITE_RESERVED: u32 = u32::MAX;
pub(crate) struct WriteReservation {
    slots: SlotAccess,
    used: [bool; 16],
}
impl SlotAccess {
    pub(crate) fn writable(&self, slot: usize) -> Result<()> {
        ensure!(
            self.0.get()[slot] == 0,
            "dSpark cache slot has outstanding access"
        );
        Ok(())
    }
    pub(crate) fn readable(&self, slot: usize) -> Result<()> {
        ensure!(
            self.0.get()[slot] != WRITE_RESERVED,
            "dSpark cache write is unpublished"
        );
        Ok(())
    }
    pub(crate) fn reserve_write(&self, used: [bool; 16]) -> Result<WriteReservation> {
        let mut counts = self.0.get();
        for (i, active) in used.iter().enumerate() {
            if *active {
                self.writable(i)?;
                counts[i] = WRITE_RESERVED;
            }
        }
        self.0.set(counts);
        Ok(WriteReservation {
            slots: self.clone(),
            used,
        })
    }
    pub(crate) fn reserve(&self, used: [bool; 16]) -> Result<ReadReservation> {
        let mut counts = self.0.get();
        for (i, active) in used.iter().enumerate() {
            if *active {
                ensure!(
                    counts[i] < WRITE_RESERVED - 1,
                    "dSpark slot is reserved or reader count exhausted"
                );
                counts[i] += 1;
            }
        }
        self.0.set(counts);
        Ok(ReadReservation {
            slots: self.clone(),
            used,
        })
    }
}
impl Drop for ReadReservation {
    fn drop(&mut self) {
        let mut counts = self.slots.0.get();
        for (i, active) in self.used.iter().enumerate() {
            if *active {
                counts[i] -= 1;
            }
        }
        self.slots.0.set(counts);
    }
}
impl Drop for WriteReservation {
    fn drop(&mut self) {
        let mut counts = self.slots.0.get();
        for (i, active) in self.used.iter().enumerate() {
            if *active {
                debug_assert_eq!(counts[i], WRITE_RESERVED);
                counts[i] = 0;
            }
        }
        self.slots.0.set(counts);
    }
}
