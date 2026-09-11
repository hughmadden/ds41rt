use super::*;
use std::cell::RefCell;
use tokio::sync::Notify;

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
        {
            // These futures are polled on the CUDA owner. Cache borrows are
            // synchronous and never retained through an await or an FFN.
            let requests = RefCell::new(&mut *guard.requests);
            let published: [Notify; 20] = std::array::from_fn(|_| Notify::new());
            tokio::try_join!(
                biased;
                unsafe { self.execute_encoder_chunk(&requests, first, transport0, &published, true) },
                unsafe { other.execute_encoder_chunk(&requests, second, transport1, &published, false) },
            )?;
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

    /// Each following attention waits for its predecessor's publication at that
    /// layer, not for the predecessor's expert output. Completed chunks retain
    /// their lane output until the caller captures and commits them in order.
    async unsafe fn execute_encoder_chunk(&mut self,
        requests: &RefCell<&mut Requests<'a>>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, published: &[Notify; 20], leading: bool,
    ) -> Result<()> {
        for layer in 0..20 {
            if !leading { published[layer].notified().await; }
            unsafe { self.prepare_encoder_query(requests, batch, layer).await?; }
            let prepared = unsafe { requests.borrow_mut().prepare_encoder_layer(batch,
                &mut self.execution, &mut self.lane, &mut self.index)? };
            // prepare_encoder_layer finishes the attention readers and publishes
            // this layer's window/source before releasing the cache borrow.
            if leading { published[layer].notify_one(); }
            let done = unsafe { prepared.execute(transport, 0, batch.image_mask()).await? };
            unsafe { self.execution.complete_layer(batch.cache()?, &mut self.lane, done)?; }
        }
        Ok(())
    }

    /// Prepare a chunk's next query only when its execution lane is available.
    async unsafe fn prepare_encoder_query(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, layer: usize) -> Result<()> {
        if layer == 0 { return Ok(()); }
        self.lane.advance()?;
        if let Some(gate) = [1, 14].iter().position(|&l| l == layer) {
            let start = Instant::now();
            loop {
                let ready = unsafe { requests.borrow().poll_engram(batch, &mut self.upload,
                    &mut self.gates[gate], &mut self.lane)? };
                if ready { break; }
                ensure!(start.elapsed() < self.engram_timeout, "paired engram gather timed out");
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
        unsafe { self.lane.begin_prepared()?; }
        Ok(())
    }

}
