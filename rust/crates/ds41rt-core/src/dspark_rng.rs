//! Request-owned Philox subsequences for native dSpark sampling.
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
    pub fn can_reserve(&self) -> bool { self.can_reserve_width(5) }
    pub fn can_reserve_width(&self, width: usize) -> bool {
        (1..=crate::MAX_DSPARK_PROPOSALS).contains(&width)
            && self.next_subsequence.checked_add(width as u64 * 256).is_some()
    }
    /// Reserve before submission. Cancellation consumes the reservation, so a new
    /// attempt cannot reuse its draws; an explicit replay may reuse the ticket.
    /// Exhaustion leaves state unchanged and requires a new request/seed policy.
    pub fn reserve(&mut self) -> Option<DsparkRngReservation> { self.reserve_width(5) }
    /// Reserve a width-specific range without changing the legacy K5 sequence.
    pub fn reserve_width(&mut self, width: usize) -> Option<DsparkRngReservation> {
        if !self.can_reserve_width(width) { return None; }
        let next = self.next_subsequence.checked_add(width as u64 * 256)?;
        let reservation = DsparkRngReservation {
            seed: self.seed,
            first_subsequence: self.next_subsequence,
        };
        self.next_subsequence = next;
        Some(reservation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seven_position_ranges_do_not_overlap_on_retry_or_width_change() {
        let mut request = DsparkRng::new(41);
        let first = request.reserve_width(7).unwrap();
        let retry = request.reserve_width(7).unwrap();
        let legacy = request.reserve().unwrap();
        assert_eq!(retry.first_subsequence, first.first_subsequence + 7 * 256);
        assert_eq!(legacy.first_subsequence, retry.first_subsequence + 7 * 256);
        let before = request.next_subsequence;
        assert!(request.reserve_width(0).is_none());
        assert!(request.reserve_width(8).is_none());
        assert_eq!(request.next_subsequence, before);
        request.next_subsequence = u64::MAX - 7 * 256 + 1;
        assert!(!request.can_reserve_width(7));
        assert!(request.reserve_width(7).is_none());
        assert!(request.can_reserve());
    }
    #[test]
    fn cancellation_does_not_recycle_reserved_draws() {
        let mut request = DsparkRng::new(41);
        let cancelled = request.reserve().unwrap();
        let retry = request.reserve().unwrap();
        assert_eq!(cancelled.seed, retry.seed);
        assert!(
            retry.first_subsequence
                >= cancelled.first_subsequence + DsparkRng::SUBSEQUENCES_PER_DRAFT
        );
    }
    #[test]
    fn exhausted_range_cannot_wrap_or_advance() {
        let mut request = DsparkRng {
            seed: 99,
            next_subsequence: u64::MAX - 1279,
        };
        let before = request.next_subsequence;
        assert!(!request.can_reserve());
        assert_eq!(request.reserve(), None);
        assert_eq!(request.next_subsequence, before);
    }
    #[test]
    fn reservations_follow_requests_across_batch_order() {
        let mut first = DsparkRng::new(7);
        let mut second = DsparkRng::new(9);
        let a = first.reserve().unwrap();
        let b = second.reserve().unwrap();
        let d = second.reserve().unwrap();
        let c = first.reserve().unwrap();
        assert_eq!((a.seed, c.seed), (7, 7));
        assert_eq!((b.seed, d.seed), (9, 9));
        assert_eq!(c.first_subsequence, d.first_subsequence);
        assert_ne!(a.first_subsequence, c.first_subsequence);
    }
}
