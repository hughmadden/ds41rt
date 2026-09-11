use super::*;

struct EncoderPairGuard<'r, 'a> {
    requests: &'r mut Requests<'a>,
    batches: [&'r mut RequestBatch; 2],
    complete: bool,
}
impl Drop for EncoderPairGuard<'_, '_> {
    fn drop(&mut self) {
        if !self.complete {
            for batch in &mut self.batches { self.requests.revoke_batch(batch); }
        }
    }
}
impl<'w, 'a> TargetPass<'w, 'a> {
    /// Execute two contiguous chunks of one prompt, retaining independent lanes
    /// through all encoder layers. The caller commits both chunks in order.
    /// # Safety
    /// Both passes/transport waves belong to this CUDA device with independent
    /// mutable storage and shared immutable official weights.
    pub async unsafe fn execute_encoder_pair(&mut self, other: &mut Self,
        requests: &mut Requests<'a>, batches: [&mut RequestBatch; 2],
        transports: [&mut NativeTp4Wave<'a>; 2], suffix: &mut EncoderSuffix<'a>,
    ) -> Result<()> {
        let mut guard = EncoderPairGuard { requests, batches, complete: false };
        let [first, second] = &mut guard.batches;
        let [transport0, transport1] = transports;
        for (pass, batch) in [(&mut *self, &mut **first), (&mut *other, &mut **second)] {
            guard.requests.validate(batch)?;
            ensure!(batch.cache()?.is_reserved(), "encoder pair requires reserved chunks");
            pass.state.begin()?;
            pass.execution.restart_for(CacheStage::Encoder);
            pass.lane.restart()?; pass.index.restart()?; pass.taps.reset();
            unsafe { guard.requests.begin_text(batch, &mut pass.embedding, &mut pass.lane)?; }
        }
        for layer in 0..20 {
            if layer != 0 {
                for (pass, batch) in [(&mut *self, &mut **first), (&mut *other, &mut **second)] {
                    pass.lane.advance()?;
                    if let Some(gate) = [1, 14].iter().position(|&l| l == layer) {
                        let start = Instant::now();
                        while !unsafe { guard.requests.poll_engram(batch, &mut pass.upload,
                            &mut pass.gates[gate], &mut pass.lane)? } {
                            ensure!(start.elapsed() < pass.engram_timeout, "paired engram gather timed out");
                            tokio::time::sleep(Duration::from_millis(1)).await;
                        }
                    }
                    unsafe { pass.lane.begin_prepared()?; }
                }
            }
            unsafe { guard.requests.execute_encoder_pair_layer([first, second],
                [&mut self.execution, &mut other.execution], [&mut self.lane, &mut other.lane],
                [&mut self.index, &mut other.index], [transport0, transport1]).await?; }
        }
        for (pass, batch) in [(&mut *self, &mut **first), (&mut *other, &mut **second)] {
            suffix.capture(&pass.lane.output()?)?;
            pass.lane.advance()?;
            unsafe {
                pass.lane.begin_prepared()?;
                guard.requests.publish_encoder_boundary(batch, &mut pass.execution, &pass.lane)?;
            }
            pass.state = State::Encoded(batch.cache()?.identity());
        }
        guard.complete = true;
        Ok(())
    }
}
