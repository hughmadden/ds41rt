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
#[derive(Clone, Copy)]
struct Reusable {
    common: usize,
    frontier: usize,
    clock: u64,
}
impl Reusable {
    fn skipped(self) -> usize {
        if self.common == self.frontier {
            self.common
        } else {
            (self.common / 2 * 2).saturating_sub(128)
        }
    }
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
    fn any_frontier(&self, position: usize) -> Option<Reusable> {
        self.value
            .as_ref()
            .map(|&(clock, _)| Reusable {
                common: position,
                frontier: position,
                clock,
            })
            .or_else(|| {
                self.children
                    .values()
                    .find_map(|child| child.any_frontier(position + child.edge.len()))
            })
    }
    fn find_reusable(&self, tokens: &[u32], position: usize) -> Option<Reusable> {
        let mut best = self.value.as_ref().map(|&(clock, _)| Reusable {
            common: position,
            frontier: position,
            clock,
        });
        if let Some(child) = tokens.first().and_then(|first| self.children.get(first)) {
            let common = child
                .edge
                .iter()
                .zip(tokens)
                .take_while(|(a, b)| a == b)
                .count();
            let candidate = if common == child.edge.len() {
                child.find_reusable(&tokens[common..], position + common)
            } else {
                child
                    .any_frontier(position + child.edge.len())
                    .map(|found| Reusable {
                        common: position + common,
                        ..found
                    })
            };
            if let Some(candidate) = candidate {
                if best.is_none_or(|old| candidate.skipped() > old.skipped()) {
                    best = Some(candidate);
                }
            }
        }
        if best.is_none() && position > 0 {
            best = self.any_frontier(position).map(|found| Reusable {
                common: position,
                ..found
            });
        }
        best.filter(|found| found.skipped() > 0)
    }
    fn refresh(&mut self, old: u64, new: u64) -> Option<&T> {
        if let Some((clock, value)) = self.value.as_mut() {
            if *clock == old {
                *clock = new;
                return Some(value);
            }
        }
        self.children
            .values_mut()
            .find_map(|child| child.refresh(old, new))
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
    /// Prefer the most computation saved: a populated exact ancestor can beat
    /// a slightly longer partial match which needs a complete replay window.
    pub fn lookup_reusable(&mut self, tokens: &[u32]) -> Option<(usize, usize, &T)> {
        let found = self.root.find_reusable(tokens, 0)?;
        self.clock = self
            .clock
            .checked_add(1)
            .expect("prefix access clock exhausted");
        let value = self
            .root
            .refresh(found.clock, self.clock)
            .expect("selected retained frontier");
        Some((found.common, found.frontier, value))
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
/// Prompt repeats and completed agentic turns have separate bounded banks.
/// A turn slot cannot be consumed by its own prompt snapshot.
#[derive(Clone, Copy)]
pub(super) enum SnapshotKind { Prompt, Turn }
struct Retention<T> {
    prompts: Radix<T>,
    turns: Radix<T>,
}
impl<T> Retention<T> {
    fn new(limit: usize) -> Self {
        Self { prompts: Radix::new(limit), turns: Radix::new(limit) }
    }
    fn bank_mut(&mut self, kind: SnapshotKind) -> &mut Radix<T> {
        match kind { SnapshotKind::Prompt => &mut self.prompts, SnapshotKind::Turn => &mut self.turns }
    }
    fn lookup_reusable(&mut self, tokens: &[u32]) -> Option<(usize, usize, &T)> {
        let prompt = self.prompts.root.find_reusable(tokens, 0);
        let turn = self.turns.root.find_reusable(tokens, 0);
        // Prefer completed turns on a tie. Refresh only the chosen bank's LRU.
        let kind = match (prompt, turn) {
            (Some(p), Some(t)) if p.skipped() > t.skipped() => SnapshotKind::Prompt,
            (_, Some(_)) => SnapshotKind::Turn,
            (Some(_), None) => SnapshotKind::Prompt,
            (None, None) => return None,
        };
        self.bank_mut(kind).lookup_reusable(tokens)
    }
    fn evict_one(&mut self) -> bool {
        // Under global-page pressure, reclaim prompt snapshots before completed
        // turns; their pages may remain shared until the turn also expires.
        self.prompts.evict_one() || self.turns.evict_one()
    }
}
struct Saved<'a> {
    target: RequestPrefix<'a>,
    draft: Option<DraftPrefix<'a>>,
    next: u32,
}
pub(super) struct PrefixCache<'a> {
    retained: Retention<Saved<'a>>,
}
impl<'a> PrefixCache<'a> {
    pub fn new(limit: usize) -> Self {
        Self {
            retained: Retention::new(limit),
        }
    }
    pub fn retain(
        &mut self,
        kind: SnapshotKind,
        tokens: &[u32],
        next: u32,
        id: u64,
        lease: CacheLease,
        requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a>>,
    ) -> Result<()> {
        let bank = self.retained.bank_mut(kind);
        if bank.limit == 0 {
            return Ok(());
        }
        let end = requests.cache().committed_end(lease)?;
        ensure!(
            end > 0 && end as usize <= tokens.len(),
            "retained token frontier differs"
        );
        // Evict before allocating another tail, keeping peak retained residency
        // within the configured number of completed states.
        if !bank.remove_exact(&tokens[..end as usize])
            && bank.entries >= bank.limit
        {
            bank.evict_one();
        }
        let target = requests.retain_prefix(lease, BackbonePrefix::device_bytes())?;
        let draft = draft.map(|d| d.retain_prefix(id, end)).transpose()?;
        bank.insert(
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
        let Some((end, frontier, saved)) = self.retained.lookup_reusable(tokens) else {
            return Ok(None);
        };
        ensure!(
            saved.target.end() == frontier as u64 && saved.draft.is_some() == draft.is_some(),
            "retained execution mode or token frontier differs"
        );
        if end != frontier {
            let start =
                requests.restore_encoder_prefix(lease, &saved.target, end / 2 * 2, tokens)?;
            // Draft rings stay fresh until decoder replay seeds the final window.
            // The saved next token belongs to a different frontier and is unused.
            return Ok(Some((start, 0)));
        }
        if tokens.len() - end >= 128 {
            requests.restore_encoder_continuation(lease, &saved.target, tokens.len() as u64)?;
            // Every final decoder/draft row comes from the new encoder suffix.
            return Ok(Some((end, 0)));
        }
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
                    if !self.retained.evict_one() {
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
    fn twenty_four_completed_turns_do_not_compete_with_prompt_snapshots() {
        let mut retained = Retention::new(24);
        for i in 100..124 {
            retained.bank_mut(SnapshotKind::Prompt).insert(&[i, 1], (false, i));
            retained.bank_mut(SnapshotKind::Turn).insert(&[i, 1, 2], (true, i));
        }
        assert_eq!((retained.prompts.entries, retained.turns.entries), (24, 24));
        for i in 100..124 {
            assert_eq!(retained.lookup_reusable(&[i, 1]), Some((2, 2, &(false, i))));
            assert_eq!(retained.lookup_reusable(&[i, 1, 2, 3]), Some((3, 3, &(true, i))));
        }
        for i in 200..224 {
            retained.bank_mut(SnapshotKind::Prompt).insert(&[i, 1], (false, i));
        }
        for i in 100..124 {
            assert_eq!(retained.lookup_reusable(&[i, 1, 2, 3]), Some((3, 3, &(true, i))));
        }
        retained.bank_mut(SnapshotKind::Turn).insert(&[999, 1, 2], (true, 999));
        assert!(retained.lookup_reusable(&[100, 1, 2, 3]).is_none());
        assert_eq!(retained.turns.entries, 24);
        for _ in 0..24 { assert!(retained.evict_one()); }
        assert_eq!((retained.prompts.entries, retained.turns.entries), (0, 24));
        for _ in 0..24 { assert!(retained.evict_one()); }
        assert!(!retained.evict_one());
        let mut disabled = Retention::new(0);
        disabled.bank_mut(SnapshotKind::Turn).insert(&[1], (true, 1));
        assert!(disabled.lookup_reusable(&[1]).is_none());
    }

    #[test]
    fn partial_radix_match_accounts_for_alignment_replay_and_exact_ancestors() {
        let tokens: Vec<u32> = (1..=512).collect();
        let mut radix = Radix::new(16);
        radix.insert(&tokens, 512);
        let mut partial = tokens[..451].to_vec();
        partial.push(9999);
        assert_eq!(radix.lookup_reusable(&partial), Some((451, 512, &512)));
        radix.insert(&tokens[..384], 384);
        // Replaying at 450 only skips 322 tokens; the populated 384-token
        // ancestor skips more work and preserves its complete window state.
        assert_eq!(radix.lookup_reusable(&partial), Some((384, 384, &384)));
        assert_eq!(radix.lookup_reusable(&tokens), Some((512, 512, &512)));
        assert!(radix.lookup_reusable(&tokens[..128]).is_none());
        assert!(radix.lookup_reusable(&[9999]).is_none());
        assert_eq!(
            radix.lookup_reusable(&tokens[..258]),
            Some((258, 384, &384))
        );
        assert!(radix.evict_one());
        assert_eq!(radix.entries, 1);
        assert_eq!(
            radix.lookup_reusable(&tokens[..258]),
            Some((258, 384, &384))
        );
        assert!(radix.evict_one());
        assert_eq!(radix.entries, 0);
        assert!(radix.root.children.is_empty());
    }

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
