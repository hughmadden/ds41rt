use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CacheStage {
    Full,
    Encoder,
    Replay,
}
impl CacheStage {
    pub(super) fn windows(self) -> std::ops::Range<usize> {
        match self {
            Self::Full => 0..40,
            Self::Encoder => 0..20,
            Self::Replay => 20..40,
        }
    }
    pub(super) fn source_count(self) -> usize {
        if self == Self::Replay {
            0
        } else {
            4
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CachePhase {
    Full,
    Encoder { target: u64 },
    Replay { target: u64, end: u64 },
}
impl CachePhase {
    pub(super) fn stage(self) -> CacheStage {
        match self {
            Self::Full => CacheStage::Full,
            Self::Encoder { .. } => CacheStage::Encoder,
            Self::Replay { .. } => CacheStage::Replay,
        }
    }
    pub(super) fn position(self, end: u64) -> u64 {
        match self {
            Self::Replay { end, .. } => end,
            _ => end,
        }
    }
    pub(super) fn window_end(self, layer: usize, end: u64) -> u64 {
        if layer < 20 {
            return end;
        }
        match self {
            Self::Full => end,
            Self::Encoder { .. } => 0,
            Self::Replay { end, .. } => end,
        }
    }
    pub(super) fn validate_tokens(self, position: u64, tokens: u32) -> Result<()> {
        let target = match self {
            Self::Full => 1048576,
            Self::Encoder { target } | Self::Replay { target, .. } => target,
        };
        ensure!(
            position
                .checked_add(u64::from(tokens))
                .is_some_and(|end| end <= target),
            "cache batch exceeds phase extent"
        );
        Ok(())
    }
    pub(super) fn advance(&mut self, published: &mut u64, next: u64) {
        match *self {
            Self::Replay { target, .. } if next == target => *self = Self::Full,
            Self::Replay { target, .. } => *self = Self::Replay { target, end: next },
            _ => *published = next,
        }
    }
}
impl BackboneCache<'_> {
    /// Select CED for a fresh admission before encoder work is planned.
    pub fn begin_encoder(&mut self, lease: CacheLease, prompt_end: u64) -> Result<()> {
        ensure!(
            (1..=1048576).contains(&prompt_end),
            "invalid encoder prompt extent"
        );
        ensure!(
            self.committed_end(lease)? == 0,
            "encoder prefill requires fresh admission"
        );
        let r = self.request(lease)?;
        ensure!(
            r.phase == CachePhase::Full && r.version == 0,
            "encoder already started"
        );
        let r = self.requests[lease.slot].as_mut().unwrap();
        r.phase = CachePhase::Encoder { target: prompt_end };
        r.version += 1;
        Ok(())
    }
    /// Start the final-window decoder replay only after every encoder/global
    /// source has reached the prompt end. Failure revokes the whole admission.
    pub fn begin_decoder_replay(&mut self, lease: CacheLease) -> Result<u64> {
        let end = self.committed_end(lease)?;
        let r = self.request(lease)?;
        let CachePhase::Encoder { target } = r.phase else {
            anyhow::bail!("decoder replay requires encoder phase");
        };
        ensure!(end == target, "encoder prefill is incomplete");
        let version = r
            .version
            .checked_add(1)
            .context("cache version exhausted")?;
        let windows = r.windows;
        let start = end.saturating_sub(128);
        for layer in 20..40 {
            if let Err(error) = self.windows[layer].begin_replay(windows[layer], start) {
                if let Err(cleanup) = self.release(&[lease]) {
                    tracing::error!(%cleanup, "revoking failed decoder replay initialization");
                }
                return Err(error);
            }
        }
        let r = self.requests[lease.slot].as_mut().unwrap();
        r.phase = CachePhase::Replay { target, end: start };
        r.version = version;
        Ok(start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ced_progress_preserves_global_history_until_replay_finishes() -> Result<()> {
        for target in [1, 127, 128, 129, 2048, 16410, 1048576] {
            let mut phase = CachePhase::Encoder { target };
            let mut published = 0;
            assert!(phase.validate_tokens(0, target as u32 + 1).is_err());
            phase.advance(&mut published, target);
            assert_eq!(phase.window_end(19, published), target);
            assert_eq!(phase.window_end(20, published), 0);
            let start = target.saturating_sub(128);
            phase = CachePhase::Replay { target, end: start };
            phase.validate_tokens(start, (target - start) as u32)?;
            assert!(phase
                .validate_tokens(start, (target - start + 1) as u32)
                .is_err());
            if target - start > 1 {
                phase.advance(&mut published, start + 1);
                assert_eq!(published, target);
                assert_eq!(phase.window_end(20, published), start + 1);
                assert_eq!(phase.stage().source_count(), 0);
            }
            phase.advance(&mut published, target);
            assert_eq!(phase, CachePhase::Full);
            assert_eq!(published, target);
            assert_eq!(phase.window_end(39, published), target);
        }
        assert_eq!(CacheStage::Encoder.windows().len(), 20);
        assert_eq!(CacheStage::Replay.windows().len(), 20);
        assert_eq!(CacheStage::Encoder.source_count(), 4);
        Ok(())
    }
}
