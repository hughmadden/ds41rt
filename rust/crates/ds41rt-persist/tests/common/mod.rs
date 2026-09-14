// Shared fixtures for the P0-4 suites and bench.
//
// The yield-N-times task is the smallest stand-in for a lane's shared-state boundaries. Keeping
// one copy here means the functional suite, the scheduler's unit tests and the bench all exercise
// the same task shape.
//
// This file is compiled into several targets, each of which uses a subset of it, so unused items
// are expected rather than dead code; every including module allows dead code.

use ds41rt_persist::sched::Task;

/// The production task population: sixteen decode lanes plus two retention lanes.
pub const PRODUCTION_TASKS: usize = 18;

/// A task that yields `yields` times before finishing.
pub struct Yielder {
    name: String,
    yields: usize,
}

impl Yielder {
    /// A task named `name` that yields `yields` times before finishing.
    pub fn new(name: impl Into<String>, yields: usize) -> Self {
        Self {
            name: name.into(),
            yields,
        }
    }
}

impl Task for Yielder {
    fn step(&mut self) -> bool {
        if self.yields == 0 {
            return false;
        }
        self.yields -= 1;
        true
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// One yielder per entry in `yields`, named `t0`, `t1`, ...
pub fn yielders(yields: &[usize]) -> Vec<Box<dyn Task>> {
    yields
        .iter()
        .enumerate()
        .map(|(index, &yields)| {
            Box::new(Yielder::new(format!("t{index}"), yields)) as Box<dyn Task>
        })
        .collect()
}

/// `count` yielders named `{prefix}0`, `{prefix}1`, ..., each yielding `yields` times.
pub fn lanes(prefix: &str, count: usize, yields: usize) -> Vec<Box<dyn Task>> {
    (0..count)
        .map(|index| Box::new(Yielder::new(format!("{prefix}{index}"), yields)) as Box<dyn Task>)
        .collect()
}
