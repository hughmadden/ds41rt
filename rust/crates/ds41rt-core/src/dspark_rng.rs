//! Request-owned Philox subsequences for five-position native dSpark sampling.
/// Each position uses 256 independent thread subsequences. Request compaction or
/// alternating waves do not affect draws, which depend on this seed and range.
#[derive(Debug)]
pub struct DsparkRng {
    seed: u64,
    next_subsequence: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsparkRngReservation {
    pub seed: u64,
    pub first_subsequence: u64,
}
impl DsparkRng {
    pub const SUBSEQUENCES_PER_DRAFT: u64 = 5 * 256;
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            next_subsequence: 0,
        }
    }
    pub fn can_reserve(&self) -> bool {
        self.next_subsequence
            .checked_add(Self::SUBSEQUENCES_PER_DRAFT)
            .is_some()
    }
    /// Reserve before submission. Cancellation consumes the reservation, so a new
    /// attempt cannot reuse its draws; an explicit replay may reuse the ticket.
    /// Exhaustion leaves state unchanged and requires a new request/seed policy.
    pub fn reserve(&mut self) -> Option<DsparkRngReservation> {
        let next = self
            .next_subsequence
            .checked_add(Self::SUBSEQUENCES_PER_DRAFT)?;
        let reservation = DsparkRngReservation {
            seed: self.seed,
            first_subsequence: self.next_subsequence,
        };
        self.next_subsequence = next;
        Some(reservation)
    }
}
