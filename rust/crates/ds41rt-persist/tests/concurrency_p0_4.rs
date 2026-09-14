//! Concurrency suite for P0-4: the interleaver must find real races, and must find them
//! reproducibly.
//!
//! The toy below is the smallest program with a lost update: two tasks read a shared counter, yield
//! at the shared-state boundary, then write back what they read. Under a schedule that runs both
//! reads before either write, one increment is lost. Removing the yield removes the race, which is
//! what proves the interleaver is exploring interleavings rather than just running tasks in order.
use ds41rt_persist::sched::{explore, Failure, Interleaver, Shared, Task};

/// How many seeds the search is allowed before it must have found the race.
const SEED_BUDGET: u64 = 64;

/// The value both tasks start from and each increments once.
const START: u64 = 0;

/// A task that reads a shared counter, yields, then writes back `read + 1`.
///
/// With the yield, the read and the write are separate steps, so another task can interleave
/// between them and the update is lost. Without it, the whole increment is one step and cannot be
/// split.
struct Increment {
    name: &'static str,
    counter: Shared<u64>,
    yields: bool,
    read: Option<u64>,
}

impl Increment {
    fn new(name: &'static str, counter: &Shared<u64>, yields: bool) -> Self {
        Self {
            name,
            counter: counter.handle(),
            yields,
            read: None,
        }
    }

    /// Read the counter and write back `read + 1` without yielding in between.
    fn increment_atomically(&mut self) {
        let read = self.counter.get();
        *self.counter.borrow_mut() = read + 1;
    }
}

impl Task for Increment {
    fn step(&mut self) -> bool {
        if !self.yields {
            // No shared-state boundary: the read and the write are one indivisible step.
            self.increment_atomically();
            return false;
        }
        match self.read {
            // First step: read, then yield at the shared-state boundary.
            None => {
                self.read = Some(self.counter.get());
                self.counter.yield_point();
                true
            }
            // Second step: write back what was read, then finish.
            Some(read) => {
                *self.counter.borrow_mut() = read + 1;
                false
            }
        }
    }

    fn name(&self) -> &str {
        self.name
    }
}

/// Two incrementing tasks over one shared counter, with or without the yield point.
fn racy_tasks(yields: bool) -> (Shared<u64>, impl FnMut() -> Vec<Box<dyn Task>>) {
    let counter = Shared::new(START);
    let factory = {
        let counter = counter.handle();
        move || {
            vec![
                Box::new(Increment::new("a", &counter, yields)) as Box<dyn Task>,
                Box::new(Increment::new("b", &counter, yields)) as Box<dyn Task>,
            ]
        }
    };
    (counter, factory)
}

/// Run one schedule against a fresh pair of tasks and report the counter's final value.
fn run_schedule(schedule: &[usize], yields: bool) -> u64 {
    let (counter, mut factory) = racy_tasks(yields);
    let mut interleaver = Interleaver::new(factory(), 0);
    interleaver
        .replay(schedule)
        .expect("the schedule is a legal run");
    counter.get()
}

/// The property the search runs: both increments must survive.
fn both_increments_land(schedule: &[usize], yields: bool) -> Result<(), String> {
    let final_value = run_schedule(schedule, yields);
    if final_value == START + 2 {
        Ok(())
    } else {
        Err(format!(
            "lost update: counter ended at {final_value}, expected {}",
            START + 2
        ))
    }
}

#[test]
fn the_racy_toy_loses_an_update_within_the_seed_budget() {
    let (_, mut factory) = racy_tasks(true);
    let failure = explore(0..SEED_BUDGET, &mut factory, |_, schedule| {
        both_increments_land(schedule, true)
    })
    .expect("a read-yield-write pair must lose an update under some schedule");
    assert!(
        failure.message.contains("lost update"),
        "the failure must name the lost update: {}",
        failure.message
    );
    assert!(failure.seed < SEED_BUDGET);
}

#[test]
fn the_failing_schedule_replays_to_the_same_failure() {
    let (_, mut factory) = racy_tasks(true);
    let failure = explore(0..SEED_BUDGET, &mut factory, |_, schedule| {
        both_increments_land(schedule, true)
    })
    .expect("the race is found");
    let replayed = both_increments_land(&failure.schedule, true);
    assert_eq!(
        replayed,
        Err(failure.message.clone()),
        "the recorded schedule must reproduce it"
    );
}

#[test]
fn the_failing_schedule_interleaves_both_reads_before_either_write() {
    let (_, mut factory) = racy_tasks(true);
    let failure = explore(0..SEED_BUDGET, &mut factory, |_, schedule| {
        both_increments_land(schedule, true)
    })
    .expect("the race is found");
    // Each task's first step is its read and its second is its write, so a lost update is exactly
    // a schedule whose first two steps are the two distinct tasks and whose last two are their
    // writes, in either order.
    assert_eq!(failure.schedule.len(), 4);
    let (first, second) = (failure.schedule[0], failure.schedule[1]);
    assert_ne!(
        first, second,
        "the lost update needs both reads before either write: {:?}",
        failure.schedule
    );
    let mut writes = [failure.schedule[2], failure.schedule[3]];
    writes.sort_unstable();
    let mut reads = [first, second];
    reads.sort_unstable();
    assert_eq!(
        writes, reads,
        "each task writes exactly once, after both reads"
    );
}

#[test]
fn removing_the_yield_removes_the_race() {
    let (_, mut factory) = racy_tasks(false);
    let outcome = explore(0..SEED_BUDGET, &mut factory, |_, schedule| {
        both_increments_land(schedule, false)
    });
    assert!(
        outcome.is_none(),
        "without a yield point the increments cannot interleave: {outcome:?}"
    );
}

#[test]
fn the_race_is_found_at_the_same_seed_every_time() {
    let find = || {
        let (_, mut factory) = racy_tasks(true);
        explore(0..SEED_BUDGET, &mut factory, |_, schedule| {
            both_increments_land(schedule, true)
        })
        .expect("the race is found")
    };
    let first: Failure = find();
    let second = find();
    assert_eq!(first, second, "the search must be deterministic");
}

#[test]
fn a_serial_schedule_never_loses_an_update() {
    // 0,0,1,1 runs each task to completion in turn, so no read straddles another task's write.
    assert_eq!(both_increments_land(&[0, 0, 1, 1], true), Ok(()));
    assert_eq!(both_increments_land(&[1, 1, 0, 0], true), Ok(()));
}

#[test]
fn the_interleaver_replays_the_racy_schedule_bit_for_bit() {
    let (_, mut factory) = racy_tasks(true);
    let schedule = Interleaver::new(factory(), 7).run();
    let mut replay = Interleaver::new(factory(), 0);
    replay
        .replay(&schedule)
        .expect("the recorded schedule replays");
    assert_eq!(replay.schedule(), schedule.as_slice());
}

#[test]
fn every_seed_produces_a_legal_schedule() {
    let (_, mut factory) = racy_tasks(true);
    for seed in 0..SEED_BUDGET {
        let schedule = Interleaver::new(factory(), seed).run();
        let mut replay = Interleaver::new(factory(), 0);
        replay
            .replay(&schedule)
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
    }
}

#[test]
fn a_shared_counter_under_a_serial_schedule_is_exact() {
    // The same toy, driven by every serial schedule, must always reach two.
    for schedule in [[0, 0, 1, 1], [1, 1, 0, 0]] {
        assert_eq!(run_schedule(&schedule, true), START + 2);
    }
}
