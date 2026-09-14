//! Deterministic interleaver for concurrency suites.
//!
//! Cooperative tasks yield at every shared-state boundary; the interleaver picks the next task from
//! a seeded PRNG so a run is reproducible from its seed. A failing property prints the seed and the
//! schedule. Sixteen lanes plus two retention lanes plus the sink and restore workers are the
//! production population.
use std::cell::RefCell;
use std::rc::Rc;

/// Test support shared by this module's unit tests, the functional suite and the bench.
#[doc(hidden)]
pub mod testing;

pub trait Task {
    /// Run until the next yield point. Returns `false` when finished.
    fn step(&mut self) -> bool;
    /// The task's name, used in error messages and in the recorded schedule's diagnostics.
    fn name(&self) -> &str;
}

/// A task that has run to completion and can no longer be scheduled.
#[derive(Debug, thiserror::Error)]
#[error("task {index} ({name}) has already finished")]
pub struct TaskFinished {
    /// Index of the finished task in the interleaver's task list.
    pub index: usize,
    /// The finished task's name.
    pub name: String,
}

/// A schedule that does not describe a run of this task set.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// The schedule names a task that does not exist.
    #[error("schedule step {step} names task {index}, but only {tasks} tasks exist")]
    UnknownTask {
        /// Position in the schedule.
        step: usize,
        /// The out-of-range task index.
        index: usize,
        /// How many tasks the interleaver holds.
        tasks: usize,
    },
    /// The schedule names a task that had already finished.
    #[error("schedule step {step} names task {index} ({name}), which had already finished")]
    TaskFinished {
        /// Position in the schedule.
        step: usize,
        /// The finished task's index.
        index: usize,
        /// The finished task's name.
        name: String,
    },
    /// The schedule ran out before every task finished.
    #[error("schedule ended after {steps} steps with {unfinished} task(s) still unfinished")]
    Incomplete {
        /// How many steps the schedule held.
        steps: usize,
        /// How many tasks were still unfinished.
        unfinished: usize,
    },
}

/// A property failure found by [`Interleaver::explore`]: the seed that produced it and the exact
/// schedule that reproduces it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// The seed whose run failed.
    pub seed: u64,
    /// The schedule the failing run followed.
    pub schedule: Vec<usize>,
    /// The property's own description of the failure.
    pub message: String,
}

/// A deterministic, seeded scheduler over a fixed set of cooperative tasks.
///
/// Invariant: the schedule is a pure function of the seed and the tasks' yield behaviour, so
/// [`Interleaver::replay`] of a recorded schedule reproduces the run bit-for-bit.
pub struct Interleaver {
    tasks: Vec<Box<dyn Task>>,
    rng: XorShift64,
    schedule: Vec<usize>,
    finished: Vec<bool>,
    remaining: usize,
}

impl Interleaver {
    /// Build an interleaver over `tasks`, seeded with `seed`.
    pub fn new(tasks: Vec<Box<dyn Task>>, seed: u64) -> Self {
        let remaining = tasks.len();
        Self {
            tasks,
            rng: XorShift64::new(seed),
            schedule: Vec::new(),
            finished: vec![false; remaining],
            remaining,
        }
    }

    /// How many tasks the interleaver holds.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether the interleaver holds no tasks.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// How many tasks have not yet finished.
    pub fn remaining(&self) -> usize {
        self.remaining
    }

    /// The schedule recorded so far.
    pub fn schedule(&self) -> &[usize] {
        &self.schedule
    }

    /// The name of task `index`.
    pub fn name(&self, index: usize) -> Option<&str> {
        self.tasks.get(index).map(|task| task.name())
    }

    /// Run one step of a uniformly chosen unfinished task, recording its index.
    ///
    /// Returns `false` once every task has finished.
    pub fn step(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        let index = self.pick_unfinished();
        self.step_task(index);
        true
    }

    /// Run every task to completion, returning the recorded schedule.
    pub fn run(&mut self) -> Vec<usize> {
        while self.step() {}
        self.schedule.clone()
    }

    /// Run `schedule` against the tasks, reproducing the recorded run exactly.
    ///
    /// Fails if the schedule names an unknown or already-finished task, or stops early.
    pub fn replay(&mut self, schedule: &[usize]) -> Result<(), ReplayError> {
        for (step, &index) in schedule.iter().enumerate() {
            if index >= self.tasks.len() {
                return Err(ReplayError::UnknownTask {
                    step,
                    index,
                    tasks: self.tasks.len(),
                });
            }
            if self.finished[index] {
                return Err(ReplayError::TaskFinished {
                    step,
                    index,
                    name: self.tasks[index].name().to_owned(),
                });
            }
            self.step_task(index);
        }
        if self.remaining > 0 {
            return Err(ReplayError::Incomplete {
                steps: schedule.len(),
                unfinished: self.remaining,
            });
        }
        Ok(())
    }

    /// Run one step of task `index`, recording it. `index` must be unfinished.
    fn step_task(&mut self, index: usize) {
        self.schedule.push(index);
        if !self.tasks[index].step() {
            self.finished[index] = true;
            self.remaining -= 1;
        }
    }

    /// Choose uniformly among the unfinished tasks.
    fn pick_unfinished(&mut self) -> usize {
        let mut choice = self.rng.below(self.remaining as u64) as usize;
        for (index, done) in self.finished.iter().enumerate() {
            if *done {
                continue;
            }
            if choice == 0 {
                return index;
            }
            choice -= 1;
        }
        unreachable!("remaining counts exactly the unfinished tasks")
    }
}

/// Run `property` over `seeds`, returning the first failure with the seed and schedule that
/// reproduce it.
///
/// `tasks` builds a fresh task set for each seed, so a run cannot leak state into the next one. The
/// property receives the seed and the schedule the run followed, so a failure can be replayed
/// without re-running the search.
pub fn explore<T, F>(
    seeds: impl IntoIterator<Item = u64>,
    mut tasks: T,
    mut property: F,
) -> Option<Failure>
where
    T: FnMut() -> Vec<Box<dyn Task>>,
    F: FnMut(u64, &[usize]) -> Result<(), String>,
{
    for seed in seeds {
        let schedule = Interleaver::new(tasks(), seed).run();
        if let Err(message) = property(seed, &schedule) {
            return Some(Failure {
                seed,
                schedule,
                message,
            });
        }
    }
    None
}

/// A shared cell that tasks touch only across a [`Shared::yield_point`].
///
/// Invariant: the value is reachable only through `borrow`/`borrow_mut`, so a task that yields
/// while holding a borrow is a panic in the test, not a silently lost interleaving.
#[derive(Debug)]
pub struct Shared<T> {
    value: Rc<RefCell<T>>,
}

impl<T> Shared<T> {
    /// Wrap `value` for sharing between tasks.
    pub fn new(value: T) -> Self {
        Self {
            value: Rc::new(RefCell::new(value)),
        }
    }

    /// A second handle to the same cell.
    pub fn handle(&self) -> Self {
        Self {
            value: Rc::clone(&self.value),
        }
    }

    /// Read the value. The borrow must end before the next yield point.
    pub fn borrow(&self) -> std::cell::Ref<'_, T> {
        self.value.borrow()
    }

    /// Mutate the value. The borrow must end before the next yield point.
    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, T> {
        self.value.borrow_mut()
    }

    /// Mark a shared-state boundary: the interleaver may switch tasks here.
    ///
    /// The marker is a no-op at runtime; it exists so every boundary is visible in the source and
    /// so a task cannot forget to declare one.
    pub fn yield_point(&self) {}
}

impl<T: Clone> Shared<T> {
    /// A copy of the value, taken without holding a borrow across the caller's next yield.
    pub fn get(&self) -> T {
        self.value.borrow().clone()
    }
}

/// XorShift64: the smallest generator with a long enough period for schedule search.
///
/// Invariant: state is never zero, so the sequence has period `2^64 - 1`.
#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Seed the generator. A zero seed is remapped, since zero is the sequence's fixed point.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// The next value in the sequence.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// A uniform value in `[0, bound)`.
    ///
    /// # Panics
    ///
    /// Panics if `bound` is zero: there is no value in an empty range, and a silent zero would
    /// make the schedule search loop forever.
    pub fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "XorShift64::below requires a non-zero bound");
        self.next_u64() % bound
    }
}

#[cfg(test)]
mod tests {
    use super::testing::yielders;
    use super::*;

    #[test]
    fn xorshift_never_reaches_its_fixed_point() {
        let mut rng = XorShift64::new(0);
        for _ in 0..1_000 {
            assert_ne!(rng.next_u64(), 0);
        }
    }

    #[test]
    fn xorshift_is_deterministic_per_seed() {
        let mut a = XorShift64::new(42);
        let mut b = XorShift64::new(42);
        let mut c = XorShift64::new(43);
        let (x, y, z) = (a.next_u64(), b.next_u64(), c.next_u64());
        assert_eq!(x, y);
        assert_ne!(x, z);
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = XorShift64::new(7);
        for _ in 0..1_000 {
            assert!(rng.below(3) < 3);
        }
    }

    #[test]
    #[should_panic(expected = "non-zero bound")]
    fn below_zero_is_a_documented_panic() {
        let mut rng = XorShift64::new(7);
        let _ = rng.below(0);
    }

    #[test]
    fn a_run_finishes_every_task() {
        let mut interleaver = Interleaver::new(yielders(&[2, 0, 5]), 1);
        let schedule = interleaver.run();
        assert_eq!(interleaver.remaining(), 0);
        // Each task costs one step per yield plus one final step that reports it finished.
        assert_eq!(
            schedule.len(),
            3 + 1 + 6,
            "each task's yields plus its finishing step"
        );
    }

    #[test]
    fn an_empty_interleaver_finishes_immediately() {
        let mut interleaver = Interleaver::new(Vec::new(), 1);
        assert!(interleaver.is_empty());
        assert_eq!(interleaver.run(), Vec::<usize>::new());
        assert!(!interleaver.step());
    }

    #[test]
    fn replay_reproduces_the_schedule() {
        let schedule = Interleaver::new(yielders(&[3, 4, 5]), 9).run();
        let mut replay = Interleaver::new(yielders(&[3, 4, 5]), 0);
        replay
            .replay(&schedule)
            .expect("the recorded schedule replays");
        assert_eq!(replay.schedule(), schedule.as_slice());
    }

    #[test]
    fn replay_rejects_an_unknown_task() {
        let mut interleaver = Interleaver::new(yielders(&[1]), 1);
        assert!(matches!(
            interleaver.replay(&[4]),
            Err(ReplayError::UnknownTask {
                step: 0,
                index: 4,
                tasks: 1
            })
        ));
    }

    #[test]
    fn replay_rejects_a_finished_task() {
        let mut interleaver = Interleaver::new(yielders(&[0]), 1);
        assert!(matches!(
            interleaver.replay(&[0, 0]),
            Err(ReplayError::TaskFinished {
                step: 1,
                index: 0,
                ..
            })
        ));
    }

    #[test]
    fn replay_rejects_a_short_schedule() {
        let mut interleaver = Interleaver::new(yielders(&[2]), 1);
        assert!(matches!(
            interleaver.replay(&[0]),
            Err(ReplayError::Incomplete {
                steps: 1,
                unfinished: 1
            })
        ));
    }

    #[test]
    fn explore_returns_the_first_failing_seed() {
        let failure = explore(
            0..64,
            || yielders(&[2, 3]),
            |seed, schedule| {
                if seed == 5 {
                    Err(format!("seed {seed} failed after {} steps", schedule.len()))
                } else {
                    Ok(())
                }
            },
        )
        .expect("seed 5 fails");
        assert_eq!(failure.seed, 5);
        assert!(failure.message.contains("seed 5 failed"));
        assert!(!failure.schedule.is_empty());
    }

    #[test]
    fn explore_returns_none_when_every_seed_passes() {
        assert!(explore(0..64, || yielders(&[2, 3]), |_, _| Ok(())).is_none());
    }

    #[test]
    fn shared_hands_out_independent_handles() {
        let shared = Shared::new(1u64);
        let other = shared.handle();
        *other.borrow_mut() += 1;
        shared.yield_point();
        assert_eq!(shared.get(), 2);
    }
}
