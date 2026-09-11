//! Official V4.1 engram addressing with transactional, request-owned history.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EngramError(&'static str);
type Result<T> = std::result::Result<T, EngramError>;
macro_rules! ensure {
    ($condition:expr, $message:expr $(,)?) => {
        if !$condition {
            return Err(EngramError($message));
        }
    };
}
use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock,
};

pub const ENGRAM_LAYERS: [u32; 2] = [1, 14];
pub const ENGRAM_ROWS: [u64; 2] = [384_006_168, 384_016_682];
pub const ENGRAM_COMPRESSED_VOCAB: u32 = 99_092;
// NumPy default_rng(10007 * layer).integers(0, (INT64_MAX / 99092) / 2, 4) * 2 + 1.
const MULTIPLIERS: [[u64; 4]; 2] = [
    [
        76632096046245,
        4839876093313,
        35959672319349,
        73987337458391,
    ],
    [
        67716810739261,
        51510806800915,
        30921347202721,
        82619226485591,
    ],
];

fn primes() -> &'static [[u64; 24]; 2] {
    static PRIMES: OnceLock<[[u64; 24]; 2]> = OnceLock::new();
    PRIMES.get_or_init(|| {
        let mut result = [[0; 24]; 2];
        let mut candidate = 16_000_000_u64;
        for layer in &mut result {
            for prime in layer {
                loop {
                    candidate += 1;
                    if candidate % 2 != 0
                        && (3..)
                            .step_by(2)
                            .take_while(|d| d * d <= candidate)
                            .all(|d| candidate % d != 0)
                    {
                        *prime = candidate;
                        break;
                    }
                }
            }
        }
        result
    })
}

pub type EngramHashes = [[u64; 24]; 2];

fn hash(tokens: [u32; 4]) -> EngramHashes {
    let mut result = [[0; 24]; 2];
    for layer in 0..2 {
        let mut rolling = tokens[0] as u64 * MULTIPLIERS[layer][0];
        let mut offset = 0;
        for order in 1..4 {
            rolling ^= tokens[order] as u64 * MULTIPLIERS[layer][order];
            for head in 0..8 {
                let column = (order - 1) * 8 + head;
                let prime = primes()[layer][column];
                result[layer][column] = rolling % prime + offset;
                offset += prime;
            }
        }
    }
    result
}

/// Only three preceding compressed IDs are needed, regardless of context length.
/// None is a sequence/image barrier, which pads this and every older lookback.
#[derive(Debug)]
pub struct EngramHistory {
    owner: u64,
    generation: u64,
    position: u64,
    recent: [Option<u32>; 3],
    pad: u32,
}

pub struct EngramBatch {
    owner: u64,
    generation: u64,
    position: u64,
    recent: [Option<u32>; 3],
    tokens: Vec<Option<u32>>,
    hashes: Vec<EngramHashes>,
}

/// A private preparation frontier for known prompt tokens. Advancing this cursor
/// never accepts tokens into the request history. Batches must still commit in
/// order against the original history after execution succeeds.
pub struct EngramPrefillCursor {
    history: EngramHistory,
}
impl EngramPrefillCursor {
    /// Use this history with the ordinary token-map/prefetch preparation path.
    pub fn history(&self) -> &EngramHistory { &self.history }
    /// Reserve a fully known prompt batch; partial speculative acceptance must
    /// use the original history and discard dependent prepared batches.
    pub fn advance_full(&mut self, batch: &EngramBatch) -> Result<()> {
        ensure!(!batch.tokens.is_empty(), "empty engram prefill reservation");
        self.history.commit(batch, batch.tokens.len())
    }
}

impl EngramBatch {
    pub fn hashes(&self) -> &[EngramHashes] {
        &self.hashes
    }
    pub fn is_image(&self, row: usize) -> Option<bool> {
        self.tokens.get(row).map(Option::is_none)
    }
    /// Excludes image rows because their engram residual contribution is zero.
    pub fn prefetch_rows(&self, layer: usize) -> Result<Vec<u64>> {
        ensure!(
            layer < ENGRAM_LAYERS.len(),
            "engram layer index out of range"
        );
        Ok(self
            .hashes
            .iter()
            .zip(&self.tokens)
            .filter(|(_, token)| token.is_some())
            .flat_map(|(hashes, _)| hashes[layer])
            .collect())
    }
}

impl EngramHistory {
    pub fn new(pad: u32) -> Result<Self> {
        ensure!(
            pad < ENGRAM_COMPRESSED_VOCAB,
            "engram padding ID out of range"
        );
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| EngramError("engram request identity exhausted"))?;
        Ok(Self {
            owner,
            generation: 0,
            position: 0,
            recent: [None; 3],
            pad,
        })
    }

    pub fn prefill_cursor(&self) -> EngramPrefillCursor {
        EngramPrefillCursor { history: Self {
            owner: self.owner, generation: self.generation, position: self.position,
            recent: self.recent, pad: self.pad,
        } }
    }

    pub fn pad_id(&self) -> u32 {
        self.pad
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    /// Prepare prefill, decode, or verification rows without changing committed history.
    pub fn prepare(
        &self,
        start: u64,
        tokens: &[Option<u32>],
        max_rows: usize,
    ) -> Result<EngramBatch> {
        ensure!(
            start == self.position,
            "engram batch position differs from committed history"
        );
        ensure!(
            tokens.len() <= max_rows,
            "engram batch exceeds row capacity"
        );
        ensure!(
            tokens
                .iter()
                .flatten()
                .all(|&token| token < ENGRAM_COMPRESSED_VOCAB),
            "compressed engram ID out of range"
        );
        ensure!(
            start.checked_add(tokens.len() as u64).is_some(),
            "engram position overflow"
        );
        let mut recent = self.recent;
        let mut hashes = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let mut padded = [self.pad; 4];
            let mut blocked = false;
            for (i, value) in std::iter::once(token).chain(recent).enumerate() {
                blocked |= value.is_none();
                if !blocked {
                    padded[i] = value.unwrap();
                }
            }
            hashes.push(hash(padded));
            recent = [token, recent[0], recent[1]];
        }
        Ok(EngramBatch {
            owner: self.owner,
            generation: self.generation,
            position: self.position,
            recent: self.recent,
            tokens: tokens.to_vec(),
            hashes,
        })
    }

    /// Commit only the accepted prefix; rejected rows leave no history behind.
    pub fn commit(&mut self, batch: &EngramBatch, accepted: usize) -> Result<()> {
        self.validate_commit(batch, accepted)?;
        for &token in &batch.tokens[..accepted] {
            self.recent = [token, self.recent[0], self.recent[1]];
        }
        self.position += accepted as u64;
        self.generation += 1;
        Ok(())
    }

    /// Validate every request in a physical wave before mutating any history.
    pub fn validate_commit(&self, batch: &EngramBatch, accepted: usize) -> Result<()> {
        ensure!(
            batch.owner == self.owner && batch.generation == self.generation
                && batch.position == self.position && batch.recent == self.recent,
            "stale or foreign engram batch"
        );
        ensure!(
            accepted <= batch.tokens.len(),
            "engram acceptance exceeds batch length"
        );
        self.generation
            .checked_add(1)
            .ok_or_else(|| EngramError("engram generation overflow"))?;
        self.position
            .checked_add(accepted as u64)
            .ok_or_else(|| EngramError("engram position overflow"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefill_lookahead_matches_sequential_hashes_without_early_acceptance() -> Result<()> {
        for width in [1, 3, 64, 65, 128, 2048] {
            let tokens = (0..width * 3 + 1).map(|i| {
                if i % 67 == 0 { None } else { Some((i * 29 % 99092) as u32) }
            }).collect::<Vec<_>>();
            let mut history = EngramHistory::new(2)?;
            let full = history.prepare(0, &tokens, tokens.len())?;
            let mut cursor = history.prefill_cursor();
            let mut batches = Vec::new();
            for chunk in tokens.chunks(width) {
                let start = cursor.history().position();
                let batch = cursor.history().prepare(start, chunk, width)?;
                assert_eq!(batch.hashes(), &full.hashes()[start as usize..start as usize + chunk.len()]);
                cursor.advance_full(&batch)?;
                batches.push(batch);
                assert_eq!(history.position(), 0);
            }
            assert!(history.commit(&batches[1], batches[1].tokens.len()).is_err());
            for batch in &batches { history.commit(batch, batch.tokens.len())?; }
            assert_eq!(history.position(), tokens.len() as u64);
            let next = [Some(42)];
            assert_eq!(history.prepare(history.position(), &next, 1)?.hashes(),
                cursor.history().prepare(history.position(), &next, 1)?.hashes());
        }
        Ok(())
    }
    #[test]
    fn lookahead_rejects_partial_or_divergent_predecessors_and_foreign_history() -> Result<()> {
        for accepted in [0, 1, 2] {
            let mut history = EngramHistory::new(2)?;
            let mut cursor = history.prefill_cursor();
            let first = cursor.history().prepare(0, &[Some(1), Some(2)], 2)?;
            cursor.advance_full(&first)?;
            let second = cursor.history().prepare(2, &[Some(3)], 1)?;
            let alternate = history.prepare(0, &[Some(8), Some(9)], 2)?;
            history.commit(&alternate, accepted)?;
            // Full alternate acceptance has the same generation and position,
            // so the predecessor-context check is essential in this case.
            assert!(history.commit(&second, 1).is_err());
            let mut foreign = EngramHistory::new(2)?;
            assert!(foreign.commit(&first, 2).is_err());
            assert!(cursor.advance_full(&first).is_err());
        }
        Ok(())
    }
    #[test]
    fn official_prime_ranges_match_checkpoint_rows() {
        for (layer, rows) in primes().iter().zip(ENGRAM_ROWS) {
            assert_eq!(layer.iter().sum::<u64>(), rows);
        }
    }
    #[test]
    fn chunking_acceptance_images_and_request_reuse_preserve_history() -> Result<()> {
        let tokens = [Some(17), Some(42), None, Some(99), Some(7), Some(31)];
        let mut history = EngramHistory::new(2)?;
        let full = history.prepare(0, &tokens, 16)?;
        let first = history.prepare(0, &tokens[..2], 16)?;
        history.commit(&first, 1)?;
        assert!(history.commit(&first, 1).is_err());
        let tail = history.prepare(1, &tokens[1..], 16)?;
        assert_eq!(&full.hashes()[1..], tail.hashes());
        let mut unrelated = EngramHistory::new(2)?;
        let foreign = unrelated.prepare(0, &tokens, 16)?;
        assert!(history.commit(&foreign, 1).is_err());
        let after_image = unrelated.prepare(0, &[Some(99)], 16)?;
        assert_eq!(full.hashes()[3], after_image.hashes()[0]);
        unrelated.commit(&after_image, 0)?;
        assert_eq!(unrelated.position(), 0);
        assert!(unrelated.commit(&after_image, 0).is_err());
        assert_eq!(full.prefetch_rows(0)?.len(), 5 * 24);
        assert_eq!(full.is_image(2), Some(true));
        assert!(history.prepare(0, &[], 16).is_err());
        assert!(history.prepare(1, &[Some(99092)], 16).is_err());
        Ok(())
    }
}
