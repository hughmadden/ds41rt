//! Counters every phase exposes; the fleet bench asserts on them.
use std::cell::Cell;

/// The counter fields of [`Counters`], so a caller can name one without a closure per field.
///
/// The variant order is the field order of [`Counters`]: `Counter as usize` indexes the same field
/// that [`Counters::get`] and [`Counters::add`] address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counter {
    Stores,
    StoreBytes,
    StoreDeclinedPressure,
    StoreDeclinedGate,
    Restores,
    RestoreBytes,
    RestoreCancelled,
    Lookups,
    LookupHits,
    LookupMissesFast,
    IdleFlushCandidates,
    IdleFlushEnqueued,
    EvictionsDisk,
    ChecksumFailures,
}

/// The counters every phase exposes; the fleet bench asserts on them.
///
/// Invariant: every counter is monotone non-decreasing over a run. [`Counters::merge`] and
/// [`Counters::add`] saturate at `u64::MAX` rather than wrapping, so a counter can never appear to
/// go backwards.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub stores: u64,
    pub store_bytes: u64,
    pub store_declined_pressure: u64,
    pub store_declined_gate: u64,
    pub restores: u64,
    pub restore_bytes: u64,
    pub restore_cancelled: u64,
    pub lookups: u64,
    pub lookup_hits: u64,
    pub lookup_misses_fast: u64,
    pub idle_flush_candidates: u64,
    pub idle_flush_enqueued: u64,
    pub evictions_disk: u64,
    pub checksum_failures: u64,
}

impl Counters {
    /// Add `other` into `self`, field by field, saturating at `u64::MAX`.
    ///
    /// Invariant: `merge` is associative and commutative, and `Counters::default()` is its
    /// identity, so counters from parallel workers can be folded in any order. Saturation keeps
    /// every counter monotone even at the type's limit.
    pub fn merge(&mut self, other: &Counters) {
        for (field, value) in self.fields_mut().into_iter().zip(other.fields()) {
            *field = field.saturating_add(value);
        }
    }

    /// The field-by-field difference `self - earlier`.
    ///
    /// Returns `None` when any counter went backwards, which means `earlier` is not a snapshot of
    /// this counter set and the difference would be meaningless.
    pub fn delta(&self, earlier: &Counters) -> Option<Counters> {
        let mut delta = Counters::default();
        for (out, (now, then)) in delta
            .fields_mut()
            .into_iter()
            .zip(self.fields().into_iter().zip(earlier.fields()))
        {
            *out = now.checked_sub(then)?;
        }
        Some(delta)
    }

    /// The value of one named counter.
    pub fn get(&self, counter: Counter) -> u64 {
        self.fields()[counter as usize]
    }

    /// Add `amount` to one named counter, saturating at `u64::MAX`.
    pub fn add(&mut self, counter: Counter, amount: u64) {
        let field = &mut self.fields_mut()[counter as usize];
        **field = field.saturating_add(amount);
    }

    /// Every counter, in [`Counter`] order.
    fn fields(&self) -> [u64; 14] {
        [
            self.stores,
            self.store_bytes,
            self.store_declined_pressure,
            self.store_declined_gate,
            self.restores,
            self.restore_bytes,
            self.restore_cancelled,
            self.lookups,
            self.lookup_hits,
            self.lookup_misses_fast,
            self.idle_flush_candidates,
            self.idle_flush_enqueued,
            self.evictions_disk,
            self.checksum_failures,
        ]
    }

    /// Mutable references to every counter, in [`Counter`] order.
    fn fields_mut(&mut self) -> [&mut u64; 14] {
        [
            &mut self.stores,
            &mut self.store_bytes,
            &mut self.store_declined_pressure,
            &mut self.store_declined_gate,
            &mut self.restores,
            &mut self.restore_bytes,
            &mut self.restore_cancelled,
            &mut self.lookups,
            &mut self.lookup_hits,
            &mut self.lookup_misses_fast,
            &mut self.idle_flush_candidates,
            &mut self.idle_flush_enqueued,
            &mut self.evictions_disk,
            &mut self.checksum_failures,
        ]
    }
}

/// Counts one named counter around a closure.
///
/// Invariant: the counter is incremented exactly once per call, whether the closure returns a value
/// or an error, so a failed operation is still visible in the counters.
#[derive(Debug)]
pub struct Metered<'a> {
    counters: &'a Cell<Counters>,
    counter: Counter,
}

impl<'a> Metered<'a> {
    /// Meter `counter` on `counters`.
    pub fn new(counters: &'a Cell<Counters>, counter: Counter) -> Self {
        Self { counters, counter }
    }

    /// Run `body`, incrementing the metered counter by one.
    pub fn count<T>(&self, body: impl FnOnce() -> T) -> T {
        self.count_by(1, body)
    }

    /// Run `body`, incrementing the metered counter by `amount`.
    pub fn count_by<T>(&self, amount: u64, body: impl FnOnce() -> T) -> T {
        let mut counters = self.counters.get();
        counters.add(self.counter, amount);
        self.counters.set(counters);
        body()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_is_the_field_wise_sum() {
        let mut a = Counters {
            stores: 1,
            store_bytes: 10,
            ..Counters::default()
        };
        let b = Counters {
            stores: 2,
            lookups: 7,
            ..Counters::default()
        };
        a.merge(&b);
        assert_eq!(a.stores, 3);
        assert_eq!(a.store_bytes, 10);
        assert_eq!(a.lookups, 7);
    }

    #[test]
    fn merge_with_default_is_the_identity() {
        let a = Counters {
            restores: 4,
            ..Counters::default()
        };
        let mut b = a;
        b.merge(&Counters::default());
        assert_eq!(b, a);
    }

    #[test]
    fn delta_inverts_merge() {
        let before = Counters {
            stores: 3,
            store_bytes: 30,
            ..Counters::default()
        };
        let mut after = before;
        after.merge(&Counters {
            stores: 2,
            store_bytes: 5,
            ..Counters::default()
        });
        assert_eq!(
            after.delta(&before),
            Some(Counters {
                stores: 2,
                store_bytes: 5,
                ..Counters::default()
            })
        );
    }

    #[test]
    fn delta_rejects_a_counter_that_went_backwards() {
        let before = Counters {
            stores: 3,
            ..Counters::default()
        };
        let after = Counters {
            stores: 2,
            ..Counters::default()
        };
        assert_eq!(after.delta(&before), None);
    }

    #[test]
    fn named_access_round_trips() {
        let mut counters = Counters::default();
        counters.add(Counter::LookupHits, 5);
        assert_eq!(counters.get(Counter::LookupHits), 5);
        assert_eq!(counters.get(Counter::LookupMissesFast), 0);
    }

    #[test]
    fn metered_counts_once_per_call() {
        let counters = Cell::new(Counters::default());
        let metered = Metered::new(&counters, Counter::Stores);
        metered.count(|| 1);
        metered.count_by(4, || 2);
        assert_eq!(counters.get().stores, 5);
    }

    #[test]
    fn metered_counts_a_failing_body() {
        let counters = Cell::new(Counters::default());
        let metered = Metered::new(&counters, Counter::ChecksumFailures);
        let result: Result<(), &str> = metered.count(|| Err("bad checksum"));
        assert!(result.is_err());
        assert_eq!(counters.get().checksum_failures, 1);
    }
}
