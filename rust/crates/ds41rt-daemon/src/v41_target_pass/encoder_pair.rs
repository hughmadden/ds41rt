use super::*;
use std::cell::RefCell;
use tokio::sync::Notify;

struct EncoderGuard<'r, 'a, const N: usize> {
    requests: &'r mut Requests<'a>,
    batches: [&'r mut RequestBatch; N],
    complete: bool,
}
impl<const N: usize> Drop for EncoderGuard<'_, '_, N> {
    fn drop(&mut self) {
        if !self.complete {
            for batch in &mut self.batches { self.requests.revoke_batch(batch); }
        }
    }
}
impl<'w, 'a> TargetPass<'w, 'a> {
    /// Execute a final reserved chunk after preceding pairs have committed.
    /// # Safety
    /// The pass and transport must belong to the same CUDA device.
    pub async unsafe fn execute_reserved_encoder(&mut self, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, transport: &mut NativeTp4Wave<'a>,
        suffix: &mut EncoderSuffix<'a>,
    ) -> Result<()> {
        let mut guard = EncoderGuard { requests, batches: [batch], complete: false };
        let batch = &mut *guard.batches[0];
        guard.requests.validate(batch)?;
        ensure!(batch.cache()?.is_reserved(), "reserved encoder requires a reserved chunk");
        self.state.begin()?;
        self.execution.restart_for(CacheStage::Encoder);
        self.lane.restart()?; self.index.restart()?; self.taps.reset();
        unsafe { guard.requests.begin_text(batch, &mut self.embedding, &mut self.lane)?; }
        {
            let requests = RefCell::new(&mut *guard.requests);
            unsafe { self.execute_encoder_chunk(&requests, batch, transport, None, None).await?; }
        }
        suffix.capture(&self.lane.output()?)?;
        self.lane.advance()?;
        unsafe {
            self.lane.begin_prepared()?;
            guard.requests.publish_encoder_boundary(batch, &mut self.execution, &self.lane)?;
        }
        self.state = State::Encoded(batch.cache()?.identity());
        guard.complete = true;
        Ok(())
    }

    /// Execute two contiguous chunks of one prompt, retaining independent lanes
    /// through all encoder layers. The caller commits both chunks in order.
    /// # Safety
    /// Both passes/transport waves belong to this CUDA device with independent
    /// mutable storage and shared immutable official weights.
    pub async unsafe fn execute_encoder_pair(&mut self, other: &mut Self,
        requests: &mut Requests<'a>, batches: [&mut RequestBatch; 2],
        transports: [&mut NativeTp4Wave<'a>; 2], suffix: &mut EncoderSuffix<'a>,
    ) -> Result<()> {
        let mut guard = EncoderGuard { requests, batches, complete: false };
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
                unsafe { self.execute_encoder_chunk(&requests, first, transport0, None, Some(&published)) },
                unsafe { other.execute_encoder_chunk(&requests, second, transport1, Some(&published), None) },
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
        transport: &mut NativeTp4Wave<'a>, predecessor: Option<&[Notify; 20]>,
        successor: Option<&[Notify; 20]>,
    ) -> Result<()> {
        for layer in 0..20 {
            if let Some(published) = predecessor { published[layer].notified().await; }
            unsafe { self.prepare_encoder_query(requests, batch, layer).await?; }
            let prepared = unsafe { requests.borrow_mut().prepare_encoder_layer(batch,
                &mut self.execution, &mut self.lane, &mut self.index)? };
            // prepare_encoder_layer finishes the attention readers and publishes
            // this layer's window/source before releasing the cache borrow.
            if let Some(published) = successor { published[layer].notify_one(); }
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
