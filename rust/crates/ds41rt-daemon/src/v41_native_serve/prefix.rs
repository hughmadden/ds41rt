use super::*;
use crate::v41_backbone_cache::{BackbonePrefix, CacheLease};
use crate::v41_requests::RequestPrefix;
use speculative::DraftPrefix;
use std::collections::BTreeMap;

struct Node<T> {
    edge: Vec<u32>,
    value: Option<(u64, T)>,
    children: BTreeMap<u32, Node<T>>,
}
impl<T> Node<T> {
    fn empty(edge: Vec<u32>) -> Self {
        Self {
            edge,
            value: None,
            children: BTreeMap::new(),
        }
    }
    fn insert(&mut self, tokens: &[u32], value: T, clock: u64) -> bool {
        if tokens.is_empty() {
            return self.value.replace((clock, value)).is_none();
        }
        let child = self
            .children
            .entry(tokens[0])
            .or_insert_with(|| Self::empty(tokens.to_vec()));
        let common = child
            .edge
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if common < child.edge.len() {
            let mut old = std::mem::replace(child, Self::empty(tokens[..common].to_vec()));
            old.edge.drain(..common);
            child.children.insert(old.edge[0], old);
        }
        child.insert(&tokens[common..], value, clock)
    }
    fn lookup<'a>(
        &'a mut self,
        tokens: &[u32],
        position: usize,
        clock: u64,
    ) -> Option<(usize, &'a T)> {
        if let Some(first) = tokens.first() {
            if let Some(child) = self.children.get_mut(first) {
                if tokens.starts_with(&child.edge) {
                    let length = child.edge.len();
                    if let Some(found) = child.lookup(&tokens[length..], position + length, clock) {
                        return Some(found);
                    }
                }
            }
        }
        self.value.as_mut().map(|(used, value)| {
            *used = clock;
            (position, &*value)
        })
    }
    fn oldest(&self) -> Option<u64> {
        self.value
            .as_ref()
            .map(|v| v.0)
            .into_iter()
            .chain(self.children.values().filter_map(Self::oldest))
            .min()
    }
    fn exact_clock(&self, tokens: &[u32]) -> Option<u64> {
        if tokens.is_empty() {
            return self.value.as_ref().map(|v| v.0);
        }
        let child = self.children.get(&tokens[0])?;
        tokens
            .strip_prefix(child.edge.as_slice())
            .and_then(|rest| child.exact_clock(rest))
    }
    fn evict(&mut self, clock: u64) {
        if self.value.as_ref().is_some_and(|v| v.0 == clock) {
            self.value = None;
        }
        for child in self.children.values_mut() {
            child.evict(clock);
        }
        self.children
            .retain(|_, child| child.value.is_some() || !child.children.is_empty());
        for child in self.children.values_mut() {
            while child.value.is_none() && child.children.len() == 1 {
                let (_, next) = child.children.pop_first().unwrap();
                child.edge.extend(next.edge);
                child.value = next.value;
                child.children = next.children;
            }
        }
    }
}
pub(super) struct Radix<T> {
    root: Node<T>,
    clock: u64,
    entries: usize,
    limit: usize,
}
impl<T> Radix<T> {
    pub fn new(limit: usize) -> Self {
        Self {
            root: Node::empty(Vec::new()),
            clock: 0,
            entries: 0,
            limit,
        }
    }
    pub fn insert(&mut self, tokens: &[u32], value: T) {
        if self.limit == 0 || tokens.is_empty() {
            return;
        }
        self.clock = self
            .clock
            .checked_add(1)
            .expect("prefix access clock exhausted");
        self.entries += usize::from(self.root.insert(tokens, value, self.clock));
        while self.entries > self.limit {
            self.evict_one();
        }
    }
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<(usize, &T)> {
        self.clock = self
            .clock
            .checked_add(1)
            .expect("prefix access clock exhausted");
        self.root.lookup(tokens, 0, self.clock)
    }
    pub fn evict_one(&mut self) -> bool {
        let Some(oldest) = self.root.oldest() else {
            return false;
        };
        self.root.evict(oldest);
        self.entries -= 1;
        true
    }
    fn remove_exact(&mut self, tokens: &[u32]) -> bool {
        let Some(clock) = self.root.exact_clock(tokens) else {
            return false;
        };
        self.root.evict(clock);
        self.entries -= 1;
        true
    }
}
struct Saved<'a> {
    target: RequestPrefix<'a>,
    draft: Option<DraftPrefix<'a>>,
    next: u32,
}
pub(super) struct PrefixCache<'a> {
    radix: Radix<Saved<'a>>,
}
impl<'a> PrefixCache<'a> {
    pub fn new(limit: usize) -> Self {
        Self {
            radix: Radix::new(limit),
        }
    }
    pub fn retain(
        &mut self,
        tokens: &[u32],
        next: u32,
        id: u64,
        lease: CacheLease,
        requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a>>,
    ) -> Result<()> {
        if self.radix.limit == 0 {
            return Ok(());
        }
        let end = requests.cache().committed_end(lease)?;
        ensure!(
            end > 0 && end as usize <= tokens.len(),
            "retained token frontier differs"
        );
        // Evict before allocating another tail, keeping peak retained residency
        // within the configured number of completed states.
        if !self.radix.remove_exact(&tokens[..end as usize])
            && self.radix.entries >= self.radix.limit
        {
            self.radix.evict_one();
        }
        let target = requests.retain_prefix(lease, BackbonePrefix::device_bytes())?;
        let draft = draft.map(|d| d.retain_prefix(id, end)).transpose()?;
        self.radix.insert(
            &tokens[..end as usize],
            Saved {
                target,
                draft,
                next,
            },
        );
        Ok(())
    }
    pub fn restore(
        &mut self,
        tokens: &[u32],
        id: u64,
        lease: CacheLease,
        requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a>>,
    ) -> Result<Option<(usize, u32)>> {
        let Some((end, saved)) = self.radix.lookup(tokens) else {
            return Ok(None);
        };
        ensure!(
            saved.target.end() == end as u64 && saved.draft.is_some() == draft.is_some(),
            "retained execution mode or token frontier differs"
        );
        requests.restore_prefix(lease, &saved.target)?;
        if let (Some(draft), Some(saved)) = (draft, saved.draft.as_ref()) {
            draft.restore_prefix(id, end as u64, saved)?;
        }
        Ok(Some((end, saved.next)))
    }
    pub fn make_room(&mut self, requests: &Requests<'a>, work: &[(CacheLease, u32)]) -> Result<()> {
        loop {
            match requests.cache().check_append_capacity(work) {
                Ok(()) => return Ok(()),
                Err(error) => {
                    if !self.radix.evict_one() {
                        return Err(error);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};
    #[test]
    fn token_radix_splits_edges_and_returns_longest_retained_frontier() {
        let mut radix = Radix::new(16);
        radix.insert(&[10, 20, 30, 40], 4);
        radix.insert(&[10, 20], 2);
        radix.insert(&[10, 20, 50], 3);
        assert_eq!(radix.lookup(&[10, 20, 30, 40, 60]), Some((4, &4)));
        assert_eq!(radix.lookup(&[10, 20, 30]), Some((2, &2)));
        assert_eq!(radix.lookup(&[10, 20, 50]), Some((3, &3)));
        assert!(radix.lookup(&[10]).is_none());
        assert!(radix.lookup(&[99, 20]).is_none());
        radix.insert(&[10, 20], 99);
        assert_eq!(radix.entries, 3);
        assert_eq!(radix.lookup(&[10, 20]), Some((2, &99)));
    }
    #[test]
    fn radix_eviction_drops_saved_owners_and_preserves_recent_branch() {
        struct Owner(Rc<Cell<usize>>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let dropped = Rc::new(Cell::new(0));
        let mut radix = Radix::new(2);
        radix.insert(&[1, 2], Owner(dropped.clone()));
        radix.insert(&[1, 2, 3], Owner(dropped.clone()));
        radix.lookup(&[1, 2]);
        radix.insert(&[1, 4], Owner(dropped.clone()));
        assert_eq!(dropped.get(), 1);
        assert_eq!(radix.lookup(&[1, 2, 3]).unwrap().0, 2);
        assert!(radix.evict_one());
        assert!(radix.evict_one());
        assert!(!radix.evict_one());
        assert_eq!(dropped.get(), 3);
        assert!(radix.root.children.is_empty());
    }
}
