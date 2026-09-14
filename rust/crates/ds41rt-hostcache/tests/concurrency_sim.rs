//! Concurrency (interleaving) suite for the simulator (HC-4). The crate is single-threaded by
//! design, so "concurrency" here means many requests in flight on eight lane slots, their steps
//! interleaved by the seeded scheduler while store copies complete asynchronously a few ticks
//! after issue. Every failure carries its seed, and the schedule log reproduces it exactly.
use ds41rt_hostcache::cache::{
    DeviceSnapshot, EvictDecision, RestoreOutcome, RestoreTarget, StoreOutcome, StoreTicket,
    TickReport,
};
use ds41rt_hostcache::sim::testing::RecordingCache;
use ds41rt_hostcache::sim::{CacheOps, EngineModel, Simulator, Workload};
use ds41rt_hostcache::snapshot::{DevicePageId, Hit, Key};

/// The churn workload both tests drive: eight lane slots, far more conversations than the
/// device banks hold, so every stage of the request life cycle is exercised concurrently.
const SEED: u64 = 0xC0FFEE;
fn churn() -> Workload {
    Workload::Churn {
        sessions: 48,
        turns: 6,
        context_tokens: 2048,
        live_ratio: 1.0,
    }
}

#[test]
fn eight_lanes_interleave_with_pending_copies() {
    let model = EngineModel::default();
    let workload = churn();
    let mut sim = Simulator::new(model, RecordingCache::with_completion_ticks(3), SEED);
    let report = sim.run(&workload);
    assert!(
        report.invariant_failures.is_empty(),
        "interleaved run must be clean, got {:?}; log tail: {:?}",
        report.invariant_failures,
        report
            .schedule_log
            .iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>()
    );
    let cache = sim.cache();
    assert!(
        cache.max_pending >= 2,
        "store copies must overlap the lanes: max_pending={}",
        cache.max_pending
    );
    assert!(
        cache.stores.len() >= 8,
        "many snapshots are stored during the run: {}",
        cache.stores.len()
    );
    // Eight slots admitted in the first wave: the opening of the log names eight sessions.
    let admitted = report
        .schedule_log
        .iter()
        .filter(|line| line.contains("admit"))
        .take(8)
        .count();
    assert_eq!(admitted, 8, "the first wave fills all eight lane slots");
}

#[test]
fn seeded_schedule_reproduces_under_interleaving() {
    let workload = churn();
    let run = || {
        Simulator::new(
            EngineModel::default(),
            RecordingCache::with_completion_ticks(3),
            SEED,
        )
        .run(&workload)
    };
    let first = run();
    let second = run();
    assert_eq!(first, second, "the same seed must reproduce report and log");
}

/// A cache that plants a fault: on the `inject_at_tick`-th `tick` it reports a ticket that was
/// never issued. The simulator's "every ticket reported once" invariant must trip, and the seed
/// must reproduce the failure.
struct InjectedFaultCache {
    inner: RecordingCache,
    inject_at_tick: u64,
    ticks: u64,
}

impl CacheOps for InjectedFaultCache {
    type Payload = ();
    fn store(&mut self, snapshot: &DeviceSnapshot, (): ()) -> StoreOutcome {
        self.inner.store(snapshot, ())
    }
    fn tick(&mut self) -> TickReport {
        self.ticks += 1;
        let mut report = self.inner.tick();
        if self.ticks == self.inject_at_tick {
            report.completed.push(StoreTicket(999_999));
        }
        report
    }
    fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        self.inner.before_device_evict(ticket)
    }
    fn lookup(&mut self, tokens: &[u32]) -> Option<Hit> {
        self.inner.lookup(tokens)
    }
    fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome {
        self.inner.restore(key, target)
    }
    fn device_page_freed(&mut self, id: DevicePageId) {
        self.inner.device_page_freed(id);
    }
}

#[test]
fn planted_fault_reproduces_from_its_seed() {
    let workload = churn();
    let inject_at_tick = 5;
    let run = || {
        Simulator::new(
            EngineModel::default(),
            InjectedFaultCache {
                inner: RecordingCache::new(),
                inject_at_tick,
                ticks: 0,
            },
            SEED,
        )
        .run(&workload)
    };

    // Without the injection the workload is clean: the fault is the injection, not the model.
    let clean = Simulator::new(EngineModel::default(), RecordingCache::new(), SEED).run(&workload);
    assert!(
        clean.invariant_failures.is_empty(),
        "workload must be clean without the planted fault: {:?}",
        clean.invariant_failures
    );

    let first = run();
    assert_eq!(
        first.invariant_failures.len(),
        1,
        "the planted double-report must trip exactly one invariant, got {:?}",
        first.invariant_failures
    );
    assert!(
        first.invariant_failures[0].contains("not outstanding"),
        "the failure names the bogus ticket: {:?}",
        first.invariant_failures
    );

    let second = run();
    assert_eq!(first, second, "same seed, same planted failure, same log");
    // The log pinpoints where the schedule was when the fault landed.
    let violation_step = first
        .schedule_log
        .iter()
        .find(|line| line.contains("INVARIANT VIOLATION"))
        .expect("the log records the failing step");
    let second_violation_step = second
        .schedule_log
        .iter()
        .find(|line| line.contains("INVARIANT VIOLATION"))
        .expect("the log records the failing step");
    assert_eq!(violation_step, second_violation_step);
}
