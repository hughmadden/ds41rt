//! Test support shared by the functional suite, the bench and the scheduler's unit tests: the
//! smallest task shapes that exercise a lane's shared-state boundaries. Not production API.
use super::Task;

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
