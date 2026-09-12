//! Preserve raw scores at retained frontiers so a new grammar can select a
//! different first token without replaying an otherwise exact KV prefix.
use anyhow::{ensure, Result};
use std::sync::Arc;

pub(super) const VOCAB: usize = 129_280;
const ROW_BYTES: usize = VOCAB * 4;

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
    pub fn new(bytes: Vec<u8>) -> Result<Self> {
        ensure!(bytes.len() % ROW_BYTES == 0, "invalid target logit batch");
        let best = bytes.chunks_exact(ROW_BYTES).map(|row| argmax(row, None)).collect::<Result<_>>()?;
        Ok(Self { bytes, best })
    }
    pub fn select(&self, row: usize, mask: Option<&[u32]>) -> Result<u32> {
        ensure!(row < self.best.len(), "selected logit row is outside batch");
        match mask {
            None => Ok(self.best[row]),
            Some(mask) => argmax(&self.bytes[row * ROW_BYTES..(row + 1) * ROW_BYTES], Some(mask)),
        }
    }
    // Copy only a finishing request's committed frontier, never every decode row.
    pub fn retain(&self, row: usize) -> Result<TokenScores> {
        ensure!(row < self.best.len(), "retained logit row is outside batch");
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
}
