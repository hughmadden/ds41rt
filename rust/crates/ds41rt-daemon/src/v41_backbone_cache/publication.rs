//! Prompt-only early KV publication; completion remains a separate transaction.
use super::*;

pub(super) struct EncoderPublication {
    pub batch: u64,
    pub start: u64,
    pub end: u64,
    pub reserved: bool,
    pub windows: u64,
    pub sources: u8,
}
impl BackboneCache<'_> {
    /// Reserve contiguous known prompt rows, with at most sixteen outstanding
    /// chunks per request. Failed planning changes no participant. Releasing a
    /// request cancels every reservation; completion must follow reservation order.
    pub fn reserve_encoder(&mut self, work: &[CacheWork]) -> Result<CacheBatch> {
        let batch = self.plan_stage(work, false, true)?;
        for r in &batch.requests {
            self.requests[r.work.lease.slot].as_mut().expect("validated request")
                .publication.push_back(EncoderPublication {
                    batch: batch.identity, start: r.position,
                    end: r.position + u64::from(r.work.tokens), reserved: true,
                    windows: 0, sources: 0,
                });
        }
        Ok(batch)
    }
    pub(super) fn publication_masks(&self, batch: &CacheBatch) -> Result<(u64, u8)> {
        let mut masks = None;
        for r in &batch.requests {
            let current = self.request(r.work.lease)?.publication.iter().find(|p| p.batch == batch.identity)
                .map_or((0, 0), |p| (p.windows, p.sources));
            ensure!(masks.is_none_or(|prior| prior == current),
                "encoder publication participants differ");
            masks = Some(current);
        }
        masks.context("empty publication batch")
    }
    fn validate_publication(&self, batch: &CacheBatch) -> Result<(u64, u8)> {
        self.validate_batch(batch)?;
        ensure!(batch.stage == CacheStage::Encoder, "early KV publication requires encoder prefill");
        for r in &batch.requests {
            r.version.checked_add(1).context("cache version exhausted")?;
        }
        self.publication_masks(batch)
    }
    fn finish_publication(&mut self, batch: &CacheBatch, window: u64, source: u8,
        result: Result<()>) -> Result<()> {
        if let Err(error) = result {
            let leases = batch.requests.iter().map(|r| r.work.lease).collect::<Vec<_>>();
            if let Err(cleanup) = self.release(&leases) {
                tracing::error!(%cleanup, "revoking failed encoder publication");
            }
            return Err(error);
        }
        for r in &batch.requests {
            let live = self.requests[r.work.lease.slot].as_mut().expect("validated request");
            if live.publication.is_empty() {
                live.publication.push_back(EncoderPublication {
                    batch: batch.identity, start: r.position,
                    end: r.position + u64::from(r.work.tokens), reserved: false,
                    windows: 0, sources: 0,
                });
            }
            let p = live.publication.iter_mut().find(|p| p.batch == batch.identity)
                .expect("validated publication reservation");
            p.windows |= window;
            p.sources |= source;
        }
        Ok(())
    }
    /// Publish completed prompt KV after all consumers of the private window
    /// have drained. This invalidates that window proposal; it does not accept
    /// the chunk's FFN output or advance Engram/request completion.
    pub fn publish_encoder_window(&mut self, batch: &CacheBatch, layer: usize,
        wave: &mut WindowWave<'_, '_>) -> Result<()> {
        let (windows, _) = self.validate_publication(batch)?;
        ensure!(layer < 20 && windows & (1u64 << layer) == 0,
            "invalid or already published encoder window");
        wave.validate_batch(&self.windows[layer], &batch.window_chunks(layer)?)?;
        let counts = batch.requests.iter().map(|r| r.work.tokens).collect::<Vec<_>>();
        let result = wave.commit(&mut self.windows[layer], &counts);
        self.finish_publication(batch, 1u64 << layer, 0, result)
    }
    /// Publish the full prompt source proposal, including its ratio-two carry.
    /// Later consumers must use a committed view with a retained causal snapshot.
    pub fn publish_encoder_source(&mut self, batch: &CacheBatch, layer: usize,
        wave: &mut CompressorWave<'_, '_>) -> Result<()> {
        let (_, sources) = self.validate_publication(batch)?;
        let i = SOURCES.iter().position(|&l| l == layer).context("invalid source layer")?;
        ensure!(sources & (1 << i) == 0, "encoder source already published");
        wave.validate_batch(&self.sources[i], &batch.source_chunks(layer)?)?;
        let counts = batch.requests.iter().map(|r| r.work.tokens).collect::<Vec<_>>();
        let result = wave.commit(&mut self.sources[i], &counts).map(|_| ());
        self.finish_publication(batch, 0, 1 << i, result)
    }
}
