//! Bind reserved prompt rows to a private Engram preparation frontier.
use super::*;
impl Requests<'_> {
    /// # Safety
    /// The prepared layer-20 boundary and execution owner belong to this batch.
    pub unsafe fn publish_encoder_boundary(&mut self, batch: &RequestBatch,
        execution: &mut BackboneExecution<'_, '_>, lane: &BackboneLane<'_, '_>) -> Result<()> {
        self.validate(batch)?;
        unsafe { execution.publish_encoder_boundary(&mut self.cache, batch.cache()?, lane) }
    }

    /// # Safety
    /// The lane/query/index owners identify this batch and have completed their
    /// producers. Poll and complete the returned owner on the CUDA thread.
    pub unsafe fn prepare_encoder_layer<'l, 'lw, 'la>(&mut self, batch: &RequestBatch,
        execution: &mut BackboneExecution<'_, '_>, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut crate::v41_index_lane::IndexLane<'_, '_>)
        -> Result<crate::v41_backbone_execution::PreparedLayer<'l, 'lw, 'la>> {
        self.validate(batch)?;
        unsafe { execution.prepare_encoder_layer(&mut self.cache, batch.cache()?, lane, index) }
    }

    pub fn reserve_encoder(&mut self, requests: &[RequestTokens<'_>]) -> Result<RequestBatch> {
        ensure!(!requests.is_empty() && requests.len() <= 16, "invalid encoder request count");
        let mut prepared = Vec::with_capacity(requests.len());
        let mut work = Vec::with_capacity(requests.len());
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        for (i, r) in requests.iter().enumerate() {
            ensure!(!requests[..i].iter().any(|p| p.lease == r.lease), "duplicate encoder request");
            ensure!(r.kind == ExpertV2SourceKind::Prefill && !r.tokens.is_empty()
                && r.image_mask.is_none_or(|m| m.len() == r.tokens.len()),
                "invalid encoder tokens, kind or image mask");
            let live = self.request(r.lease)?;
            ensure!(live.history.position() == self.cache.committed_end(r.lease)?,
                "encoder accepted histories differ");
            let history = live.prefill.as_ref().map_or(&live.history, |p| p.history());
            ensure!(history.position() == self.cache.encoder_prepared_end(r.lease)?,
                "encoder preparation frontiers differ");
            prepared.push(history.prefill_cursor());
            work.push(CacheWork { lease: r.lease, tokens: u32::try_from(r.tokens.len())?, kind: r.kind });
            tokens.extend_from_slice(r.tokens);
            mask.extend((0..r.tokens.len()).map(|i| u8::from(r.image_mask.is_some_and(|m| m[i]))));
        }
        let inputs = requests.iter().zip(&prepared).map(|(r, p)| EngramRequestTokens {
            history: p.history(), token_ids: r.tokens, image_mask: r.image_mask,
        }).collect::<Vec<_>>();
        let mut engram = self.pipeline.prepare(&inputs)?;
        let advanced = prepared.iter().zip(engram.batches()).map(|(p, b)| {
            let mut next = p.history().prefill_cursor();
            next.advance_full(b)?;
            Ok(next)
        }).collect::<Result<Vec<_>>>()?;
        // No request preparation frontier changes unless every cache reservation
        // succeeds. Rejected capacity/extent cancels the speculative I/O work.
        let cache = match self.cache.reserve_encoder(&work) {
            Ok(cache) => cache,
            Err(error) => { engram.cancel(); return Err(error); }
        };
        for (r, next) in requests.iter().zip(advanced) {
            self.slots.iter_mut().flatten().find(|live| live.lease == r.lease)
                .expect("validated encoder request").prefill = Some(next);
        }
        Ok(RequestBatch { cache, prepared, engram: Some(engram),
            leases: requests.iter().map(|r| r.lease).collect(), tokens, image_mask: mask, finished: false })
    }
}

struct PairGuard<'r, 'a> {
    requests: &'r mut Requests<'a>,
    batches: [&'r mut RequestBatch; 2],
    complete: bool,
}
impl Drop for PairGuard<'_, '_> {
    fn drop(&mut self) {
        if !self.complete {
            for batch in &mut self.batches { self.requests.revoke_batch(batch); }
        }
    }
}
impl Requests<'_> {
    /// Run one layer of two ordered encoder chunks. The second attention is
    /// prepared while the first remote FFN is pending; both lanes retain their
    /// own query/index/residual storage. Cancellation revokes both admissions.
    /// # Safety
    /// Both prepared queries are on the owning CUDA thread/device; chunks are
    /// ordered, and the two lanes, executions and transports are independent.
    pub async unsafe fn execute_encoder_pair_layer(&mut self,
        batches: [&mut RequestBatch; 2],
        executions: [&mut BackboneExecution<'_, '_>; 2],
        lanes: [&mut BackboneLane<'_, '_>; 2],
        indices: [&mut crate::v41_index_lane::IndexLane<'_, '_>; 2],
        transports: [&mut crate::v41_experts::coordinator::NativeTp4Wave<'_>; 2],
    ) -> Result<()> {
        let mut guard = PairGuard { requests: self, batches, complete: false };
        let [first, second] = &mut guard.batches;
        let [exec0, exec1] = executions;
        let [lane0, lane1] = lanes;
        let [index0, index1] = indices;
        let [transport0, transport1] = transports;
        guard.requests.validate(first)?;
        guard.requests.validate(second)?;
        let prepared = unsafe { guard.requests.prepare_encoder_layer(first, exec0, lane0, index0)? };
        let (done0, done1) = tokio::try_join!(
            unsafe { prepared.execute(transport0, 0, first.image_mask()) },
            async {
                let prepared = unsafe { guard.requests.prepare_encoder_layer(second, exec1, lane1, index1)? };
                unsafe { prepared.execute(transport1, 0, second.image_mask()).await }
            },
        )?;
        unsafe {
            exec0.complete_layer(first.cache()?, lane0, done0)?;
            exec1.complete_layer(second.cache()?, lane1, done1)?;
        }
        guard.complete = true;
        Ok(())
    }
}
