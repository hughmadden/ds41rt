use super::*;
use crate::v41_backbone_cache::{BackbonePrefix, CacheLease};
use crate::v41_requests::RequestPrefix;
use speculative::DraftPrefix;
mod images;
pub(super) use images::ImageKeys;
use images::ImageKeySpace;
pub(super) use ds41rt_core::prefix::{Radix, Retention, SnapshotKind};
struct Saved<'a> {
    _images: ImageKeys,
    target: RequestPrefix<'a>,
    draft: Option<DraftPrefix<'a>>,
    next: TokenScores,
}
pub(super) struct PrefixCache<'a> {
    retained: Retention<Saved<'a>>,
    images: ImageKeySpace,
    pending: [Option<PendingRetention>; 2],
}
struct PendingRetention {
    kind: SnapshotKind,
    keys: Vec<u32>,
    images: ImageKeys,
    next: TokenScores,
    id: u64,
    lease: CacheLease,
    draft: bool,
}
impl<'a> PrefixCache<'a> {
    pub fn new(limit: usize) -> Self {
        Self {
            retained: Retention::new(limit),
            images: ImageKeySpace::default(),
            pending: [None, None],
        }
    }
    pub fn prepare_key(&mut self, tokens: &[u32], images: &[ds41rt_loader::V41ImageSpan]) -> Result<ImageKeys> {
        self.images.prepare(tokens, images)
    }
    pub fn retain(
        &mut self,
        kind: SnapshotKind,
        tokens: &[u32],
        images: &ImageKeys,
        next: &TokenScores,
        id: u64,
        lease: CacheLease,
        requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a>>,
    ) -> Result<()> {
        let bank = self.retained.bank_mut(kind);
        if bank.limit() == 0 {
            return Ok(());
        }
        let end = requests.cache().committed_end(lease)?;
        ensure!(
            end > 0 && end as usize <= tokens.len(),
            "retained token frontier differs"
        );
        let keys = images.encode(&tokens[..end as usize])?;
        // Evict before allocating another tail, keeping peak retained residency
        // within the configured number of completed states.
        if bank.remove_exact(&keys).is_none() && bank.entries() >= bank.limit() {
            bank.evict_one();
        }
        let target = requests.retain_prefix(lease, BackbonePrefix::device_bytes())?;
        let draft = draft.map(|d| d.retain_prefix(id, end)).transpose()?;
        bank.insert(
            &keys,
            Saved {
                _images: images.through(end as usize),
                target,
                draft,
                next: next.clone(),
            },
        );
        Ok(())
    }
    pub fn queue_retain(&mut self, lane: usize, kind: SnapshotKind, tokens: &[u32],
        images: &ImageKeys, next: &TokenScores, id: u64, lease: CacheLease,
        requests: &mut Requests<'a>, mut draft: Option<&mut DraftRuntime<'_, 'a>>) -> Result<bool> {
        ensure!(self.pending.get(lane).context("invalid retention lane")?.is_none(), "retention lane occupied");
        let bank = self.retained.bank_mut(kind);
        if bank.limit() == 0 { return Ok(false); }
        let end = requests.cache().committed_end(lease)?;
        ensure!(end > 0 && end as usize <= tokens.len(), "retained token frontier differs");
        let keys = images.encode(&tokens[..end as usize])?;
        if bank.remove_exact(&keys).is_none() && bank.entries() >= bank.limit() { bank.evict_one(); }
        requests.queue_prefix(lane, lease)?;
        if let Some(draft) = draft.as_deref_mut() {
            if let Err(error) = draft.queue_prefix(lane, id, end) {
                if let Err(cleanup) = requests.abort_prefix(lane) {
                    tracing::error!(%cleanup, "draining target snapshot after draft failure");
                }
                return Err(error);
            }
        }
        self.pending[lane] = Some(PendingRetention { kind, keys: keys.into_owned(), images: images.through(end as usize),
            next: next.clone(), id, lease, draft: draft.is_some() });
        Ok(true)
    }
    pub fn poll_retain(&mut self, lane: usize, requests: &mut Requests<'a>,
        mut draft: Option<&mut DraftRuntime<'_, 'a>>) -> Result<bool> {
        let pending = self.pending.get(lane).and_then(Option::as_ref).context("retention is not pending")?;
        ensure!(pending.draft == draft.is_some(), "pending retention execution mode differs");
        if !requests.prefix_ready(lane, pending.lease)? { return Ok(false); }
        if let Some(draft) = draft.as_deref() {
            if !draft.prefix_ready(lane, pending.id)? { return Ok(false); }
        }
        let target = requests.finish_prefix(lane, pending.lease)?;
        let saved_draft = draft.as_deref_mut().map(|d| d.finish_prefix(lane, pending.id)).transpose()?;
        let pending = self.pending[lane].take().unwrap();
        // Another lane may have inserted while these copies ran. Radix insertion
        // enforces the bank limit again; two extra arena slots cover both pending copies.
        self.retained.bank_mut(pending.kind).insert(&pending.keys, Saved {
            _images: pending.images, target, draft: saved_draft, next: pending.next,
        });
        Ok(true)
    }
    pub fn abort_retain(&mut self, lane: usize, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a>>) -> Result<()> {
        let target = requests.abort_prefix(lane);
        let speculative = draft.map(|d| d.abort_prefix(lane)).transpose();
        if let Some(pending) = self.pending.get_mut(lane) { *pending = None; }
        target.and(speculative.map(|_| ()))
    }
    pub fn restore(
        &mut self,
        tokens: &[u32],
        images: &ImageKeys,
        id: u64,
        lease: CacheLease,
        requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a>>,
    ) -> Result<Option<(usize, Option<TokenScores>)>> {
        let keys = images.encode(tokens)?;
        let Some((end, frontier, saved)) = self.retained.lookup_reusable(&keys) else {
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
            return Ok(Some((start, None)));
        }
        if tokens.len() - end >= 128 {
            requests.restore_encoder_continuation(lease, &saved.target, tokens.len() as u64)?;
            // Every final decoder/draft row comes from the new encoder suffix.
            return Ok(Some((end, None)));
        }
        requests.restore_prefix(lease, &saved.target)?;
        if let (Some(draft), Some(saved)) = (draft, saved.draft.as_ref()) {
            draft.restore_prefix(id, end as u64, saved)?;
        }
        Ok(Some((end, Some(saved.next.clone()))))
    }
    pub fn make_room(&mut self, requests: &Requests<'a>, work: &[(CacheLease, u32)]) -> Result<()> {
        loop {
            match requests.cache().check_append_capacity(work) {
                Ok(()) => return Ok(()),
                Err(error) => {
                    if error.downcast_ref::<crate::v41_compressor::SourcePoolExhausted>().is_none()
                        || !self.retained.evict_one() {
                        return Err(error);
                    }
                }
            }
        }
    }
}
