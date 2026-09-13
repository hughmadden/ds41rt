//! Preserve raw scores at retained frontiers so a new grammar can select a
//! different first token without replaying an otherwise exact KV prefix.
use anyhow::{ensure, Result};
use std::sync::Arc;

pub(super) const VOCAB: usize = 129_280;
pub(super) const ROW_BYTES: usize = VOCAB * 4;

#[derive(Clone)]
pub(super) struct TokenScores {
    bytes: Arc<[u8]>,
    best: u32,
}
impl TokenScores {
    pub fn new(bytes: Vec<u8>) -> Result<Self> {
        ensure!(bytes.len() == ROW_BYTES, "invalid retained logit row");
        let best = argmax(&bytes, None)?;
        Ok(Self { bytes: bytes.into(), best })
    }
    pub fn select(&self, mask: Option<&[u32]>) -> Result<u32> {
        match mask {
            None => Ok(self.best),
            Some(mask) => argmax(&self.bytes, Some(mask)),
        }
    }
}

pub(super) struct BatchScores {
    pub best: Vec<u32>,
    bytes: Vec<u8>,
}
impl BatchScores {
    pub fn has_full_logits(&self) -> bool { !self.bytes.is_empty() }
    pub fn retain_downloaded(&self, row: usize, bytes: &[u8]) -> Result<TokenScores> {
        ensure!(row < self.best.len() && bytes.len() == ROW_BYTES, "downloaded retained logit extent differs");
        let best = argmax(bytes, None)?;
        ensure!(best == self.best[row], "GPU and retained CPU greedy selection differ");
        Ok(TokenScores { bytes: Arc::from(bytes), best })
    }
    pub fn new(bytes: Vec<u8>) -> Result<Self> {
        ensure!(bytes.len() % ROW_BYTES == 0, "invalid target logit batch");
        let best = bytes.chunks_exact(ROW_BYTES).map(|row| argmax(row, None)).collect::<Result<_>>()?;
        Ok(Self { bytes, best })
    }
    pub fn from_greedy(values: Vec<(u32, f32)>) -> Result<Self> {
        ensure!(!values.is_empty() && values.len() <= 80, "invalid compact target rows");
        ensure!(values.iter().all(|&(id, score)| (id as usize) < VOCAB && score.is_finite()),
            "invalid token or non-finite target logit");
        Ok(Self { best: values.into_iter().map(|v| v.0).collect(), bytes: Vec::new() })
    }
    /// The completed head buffer stays unchanged until commit returns. Fetch only
    /// a finishing request's frontier; a later grammar still sees all vocabulary scores.
    pub fn retain_from_device(&self, lib: &ds41rt_ffi::NativeLibrary,
        mut logits: ds41rt_ffi::Ds41rtDeviceBuffer, row: usize) -> Result<TokenScores> {
        if !self.bytes.is_empty() { return self.retain(row); }
        ensure!(row < self.best.len() && logits.bytes == self.best.len() * ROW_BYTES,
            "compact retained logit extent differs");
        logits.ptr = unsafe { logits.ptr.cast::<u8>().add(row * ROW_BYTES).cast() };
        logits.bytes = ROW_BYTES;
        let mut bytes = vec![0; ROW_BYTES];
        lib.copy_d2h(&mut bytes, logits)?;
        self.retain_downloaded(row, &bytes)
    }
    pub fn select(&self, row: usize, mask: Option<&[u32]>) -> Result<u32> {
        ensure!(row < self.best.len(), "selected logit row is outside batch");
        match mask {
            None => Ok(self.best[row]),
            Some(mask) => {
                ensure!(self.bytes.len() == self.best.len() * ROW_BYTES, "grammar requires full logits");
                argmax(&self.bytes[row * ROW_BYTES..(row + 1) * ROW_BYTES], Some(mask))
            },
        }
    }
    // Diagnostic only: scores are already on the host. Callers gate the scan
    // behind the logit trace target so normal serving incurs no extra work.
    pub fn top_two(&self, row: usize) -> Result<[(u32, f32); 2]> {
        ensure!(row < self.best.len(), "diagnostic logit row is outside batch");
        ensure!(self.bytes.len() == self.best.len() * ROW_BYTES, "diagnostics require full logits");
        let mut top = [(0, f32::NEG_INFINITY); 2];
        for (token, bytes) in self.bytes[row * ROW_BYTES..(row + 1) * ROW_BYTES]
            .chunks_exact(4).enumerate() {
            let value = f32::from_ne_bytes(bytes.try_into().unwrap());
            if value > top[0].1 {
                top[1] = top[0];
                top[0] = (token as u32, value);
            } else if value > top[1].1 {
                top[1] = (token as u32, value);
            }
        }
        Ok(top)
    }
    // Copy only a finishing request's committed frontier, never every decode row.
    pub fn retain(&self, row: usize) -> Result<TokenScores> {
        ensure!(row < self.best.len(), "retained logit row is outside batch");
        ensure!(self.bytes.len() == self.best.len() * ROW_BYTES, "retention requires full logits");
        Ok(TokenScores {
            bytes: Arc::from(&self.bytes[row * ROW_BYTES..(row + 1) * ROW_BYTES]),
            best: self.best[row],
        })
    }
}

fn argmax(bytes: &[u8], mask: Option<&[u32]>) -> Result<u32> {
    if let Some(mask) = mask {
        ensure!(mask.len() == VOCAB.div_ceil(32), "invalid grammar mask width");
    }
    let mut best = None;
    let mut maximum = f32::NEG_INFINITY;
    for (i, bytes) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_ne_bytes(bytes.try_into().unwrap());
        ensure!(value.is_finite(), "non-finite target logit");
        if mask.is_none_or(|words| words[i / 32] & (1 << (i % 32)) != 0) && value > maximum {
            maximum = value;
            best = Some(i as u32);
        }
    }
    best.ok_or_else(|| anyhow::anyhow!("grammar allows no target token"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(winner: usize) -> Vec<u8> {
        let mut scores = vec![0.0f32; VOCAB];
        scores[winner] = 4.;
        scores[17] = 3.;
        scores.into_iter().flat_map(f32::to_ne_bytes).collect()
    }
    #[test]
    fn compact_scores_reject_invalid_rows_and_require_logits_for_masks() {
        let scores = BatchScores::from_greedy(vec![(17, -2.), (91, 3.)]).unwrap();
        assert_eq!(scores.select(0, None).unwrap(), 17);
        assert!(scores.select(2, None).is_err());
        assert!(scores.select(0, Some(&vec![u32::MAX; VOCAB.div_ceil(32)])).is_err());
        assert!(scores.retain(0).is_err());
        assert!(scores.top_two(0).is_err());
        assert!(BatchScores::from_greedy(vec![(0, f32::NAN)]).is_err());
        assert!(BatchScores::from_greedy(vec![(0, f32::INFINITY)]).is_err());
        assert!(BatchScores::from_greedy(vec![(VOCAB as u32, 1.)]).is_err());
        assert!(BatchScores::from_greedy(vec![]).is_err());
    }
    #[test]
    fn new_constraint_reselects_an_exact_cached_frontier() {
        let scores = TokenScores::new(row(91)).unwrap();
        let shared = scores.clone();
        assert!(Arc::ptr_eq(&scores.bytes, &shared.bytes));
        assert_eq!(scores.select(None).unwrap(), 91);
        let mut mask = vec![0; VOCAB.div_ceil(32)];
        mask[0] = 1 << 17;
        assert_eq!(shared.select(Some(&mask)).unwrap(), 17);
        mask[0] = 0;
        assert!(shared.select(Some(&mask)).is_err());
        assert_eq!(scores.select(None).unwrap(), 91);
    }
    #[test]
    fn downloaded_frontier_keeps_full_scores_and_checks_gpu_selection() {
        let compact = BatchScores::from_greedy(vec![(91, 4.)]).unwrap();
        assert!(!compact.has_full_logits());
        let bytes = row(91);
        let retained = compact.retain_downloaded(0, &bytes).unwrap();
        assert!(compact.retain_downloaded(1, &bytes).is_err());
        assert!(compact.retain_downloaded(0, &row(93)).is_err());
        let mut invalid = bytes.clone();
        invalid[..4].copy_from_slice(&f32::NAN.to_ne_bytes());
        assert!(compact.retain_downloaded(0, &invalid).is_err());
        drop(compact); drop(bytes);
        let mut mask = vec![0; VOCAB.div_ceil(32)]; mask[0] = 1 << 17;
        assert_eq!(retained.select(None).unwrap(), 91);
        assert_eq!(retained.select(Some(&mask)).unwrap(), 17);
    }
    #[test]
    fn retained_scores_survive_batch_drop_and_reject_invalid_rows() {
        let batch = BatchScores::new([row(91), row(93)].concat()).unwrap();
        assert_eq!(batch.best, [91, 93]);
        assert!(batch.retain(2).is_err());
        let scores = batch.retain(1).unwrap();
        drop(batch);
        assert_eq!(scores.select(None).unwrap(), 93);
        assert!(TokenScores::new(vec![0; 4]).is_err());
        let mut invalid = row(91);
        invalid[..4].copy_from_slice(&f32::NAN.to_ne_bytes());
        assert!(TokenScores::new(invalid).is_err());
    }
    #[test]
    fn diagnostic_top_two_preserves_greedy_ties_and_negative_scores() {
        let mut values = vec![-10.0f32; VOCAB];
        values[17] = -2.;
        values[91] = -2.;
        let batch = BatchScores::new(values.into_iter().flat_map(f32::to_ne_bytes).collect()).unwrap();
        assert_eq!(batch.best, [17]);
        assert_eq!(batch.top_two(0).unwrap(), [(17, -2.), (91, -2.)]);
        assert!(batch.top_two(1).is_err());
    }
}
