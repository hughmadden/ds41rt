use super::*;
use crate::v41_dspark_cache::{DsparkWindow, WindowLease};
use crate::v41_experts::dspark::{DsparkChain, DsparkMainContext, DsparkWeights};
use crate::v41_requests::RequestBatch;

/// Execution workspaces share a request-indexed bank of persistent draft state.
pub(crate) struct DraftRuntime<'w, 'a> {
    main: DsparkMainContext<'w, 'a>,
    chain: DsparkChain<'w, 'a>,
    windows: [DsparkWindow<'a>; 3],
    requests: std::collections::BTreeMap<u64, DraftRequest>,
    captured: std::collections::BTreeSet<usize>,
}
struct DraftRequest {
    leases: [WindowLease; 3],
    rng: ds41rt_core::DsparkRng,
    slot: usize,
}
impl<'w, 'a> DraftRuntime<'w, 'a> {
    pub fn new(
        lib: &'a NativeLibrary,
        weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>,
        head: &'w VocabularyHead<'a>,
        capacity: u32,
    ) -> Result<Self> {
        let window = || DsparkWindow::new(lib, 16, capacity, DsparkWindow::device_bytes(16, capacity)?);
        Ok(Self {
            main: weights.main_context(capacity, DsparkMainContext::device_bytes(lib, capacity)?)?,
            chain: weights.draft(table, head, 16, weights.draft_bytes(16)?)?,
            windows: [window()?, window()?, window()?],
            requests: Default::default(),
            captured: Default::default(),
        })
    }
    pub fn admit(&mut self, id: u64) -> Result<()> {
        ensure!(!self.requests.contains_key(&id), "draft request already admitted");
        let slot = (0..16).find(|slot| self.requests.values().all(|request| request.slot != *slot))
            .context("draft request capacity exhausted")?;
        let mut leases = Vec::new();
        for stage in 0..3 {
            match self.windows[stage].begin_request(slot, id) {
                Ok(lease) => leases.push(lease),
                Err(error) => {
                    for (stage, lease) in leases.into_iter().enumerate() {
                        let _ = self.windows[stage].release(lease);
                    }
                    return Err(error);
                }
            }
        }
        self.requests.insert(id, DraftRequest {
            leases: leases.try_into().ok().context("draft admission incomplete")?,
            rng: ds41rt_core::DsparkRng::new(id),
            slot,
        });
        Ok(())
    }
    pub fn release(&mut self, id: u64) -> Result<()> {
        let mut failure = None;
        if let Some(request) = self.requests.remove(&id) {
            for (window, lease) in self.windows.iter_mut().zip(request.leases) {
                if window.request_id(lease).is_ok() {
                    if let Err(error) = window.release(lease) {
                        failure = Some(error);
                    }
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
    #[cfg(test)]
    pub(crate) fn validate_position(&self, id: u64, end: u64) -> Result<()> {
        let request = self.requests.get(&id).context("draft request not admitted")?;
        for (window, lease) in self.windows.iter().zip(request.leases) {
            ensure!(window.request_id(lease)? == id, "draft request identity differs");
            ensure!(window.committed_end(lease)? == Some(end), "draft position differs");
        }
        Ok(())
    }
    pub fn commit(
        &mut self,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: u32,
    ) -> Result<()> {
        self.commit_batch(pass, requests, batch, &[accepted])
    }
    pub fn commit_batch(
        &mut self,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: &[u32],
    ) -> Result<()> {
        let ids = batch.cache()?.request_ids();
        ensure!(ids.len() == accepted.len(), "draft acceptance count differs");
        let leases = ids.iter().map(|id| self.requests.get(id)
            .map(|request| request.leases).context("draft request not admitted"))
            .collect::<Result<Vec<_>>>()?;
        let stage_leases: [Vec<WindowLease>; 3] = std::array::from_fn(|stage|
            leases.iter().map(|request| request[stage]).collect());
        let taps = pass.taps(batch)?;
        let mut proposal = unsafe {
            self.main
                .execute_rows(taps.values(), taps.batch_identity(), taps.rows())?
        };
        let [a, b, c] = &mut self.windows;
        unsafe {
            pass.commit_with_dspark(
                requests,
                batch,
                &mut proposal,
                &mut [a, b, c],
                [&stage_leases[0], &stage_leases[1], &stage_leases[2]],
                accepted,
            )
        }
    }
    /// Inputs are (request identity, anchor, committed end, remaining output budget).
    /// Returned rows retain input order; short histories/budgets use the anchor only.
    pub fn propose(&mut self, lib: &'a NativeLibrary,
        inputs: &[(u64, u32, u64, usize)],
    ) -> Result<Vec<Vec<u32>>> {
        ensure!(!inputs.is_empty() && inputs.len() <= 16, "invalid draft batch size");
        let mut seen = std::collections::BTreeSet::new();
        for &(id, anchor, _, remaining) in inputs {
            ensure!(seen.insert(id) && self.requests.contains_key(&id), "invalid draft request identity");
            ensure!(anchor < 129280 && remaining > 0, "invalid draft request input");
        }
        let active: Vec<_> = inputs.iter().enumerate()
            .filter(|(_, (_, _, end, remaining))| *end >= 2 && *remaining > 1).collect();
        let mut outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
        if active.is_empty() { return Ok(outputs); }
        let count = active.len();
        let tokens: Vec<_> = active.iter().map(|(_, (_, anchor, _, _))| *anchor as i32).collect();
        let bindings: [Vec<_>; 3] = std::array::from_fn(|stage| active.iter()
            .map(|(_, (id, _, end, _))| (self.requests[id].leases[stage], *end)).collect());
        self.chain.set_tokens(&tokens)?;
        let mut rngs: Vec<_> = self.requests.iter_mut().filter_map(|(id, request)|
            active.iter().position(|(_, (active_id, _, _, _))| active_id == id)
                .map(|index| (index, &mut request.rng))).collect();
        rngs.sort_by_key(|(index, _)| *index);
        self.chain.prepare_sampling(&mut rngs.into_iter().map(|(_, rng)| rng).collect::<Vec<_>>(),
            &vec![0.0; count])?;
        let windows = self.windows.each_ref();
        let bindings = bindings.each_ref().map(|rows| rows.as_slice());
        if !self.captured.contains(&count) {
            unsafe { self.chain.capture(windows, bindings)?; }
            self.captured.insert(count);
        }
        unsafe { self.chain.replay(windows, bindings)?; }
        let buffer = self.chain.draft_output()?[0];
        let mut bytes = vec![0; buffer.bytes];
        lib.copy_d2h(&mut bytes, buffer)?;
        ensure!(bytes.len() == 6 * count * 4, "draft token extent differs");
        let packed: Vec<_> = bytes.chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap())).collect();
        for (row, &(output, &(_, anchor, _, remaining))) in active.iter().enumerate() {
            let tokens: Vec<_> = (0..6).map(|step| packed[step * count + row]).collect();
            ensure!(tokens[0] == anchor && tokens.iter().all(|&token| token < 129280), "invalid draft tokens");
            outputs[output] = tokens[..remaining.min(6)].to_vec();
        }
        Ok(outputs)
    }
    pub fn verify(
        &mut self,
        lib: &'a NativeLibrary,
        runtime: &tokio::runtime::Runtime,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        transport: &mut NativeTp4Wave<'a>,
        lease: crate::v41_backbone_cache::CacheLease,
        anchor: u32,
        remaining: usize,
        job: &NativeRequest,
    ) -> Result<Vec<u32>> {
        ensure!(!job.events.is_closed(), "client disconnected");
        let start = Instant::now();
        let end = requests.cache().committed_end(lease)?;
        let id = requests.cache().request_id(lease)?;
        let inputs = self.propose(lib, &[(id, anchor, end, remaining)])?
            .pop().context("draft proposal missing")?;
        let draft_us = start.elapsed().as_micros() as u64;
        ensure!(!job.events.is_closed(), "client disconnected");
        let mut batch = requests.prepare(&[RequestTokens {
            lease,
            tokens: &inputs,
            image_mask: None,
            kind: ExpertV2SourceKind::MtpVerify,
        }])?;
        let result = (|| -> Result<Vec<u32>> {
            let selected = (0..inputs.len()).collect::<Vec<_>>();
            let logits = runtime
                .block_on(unsafe { pass.execute(requests, &mut batch, transport, 0, &selected) })?;
            let mut bytes = vec![0; logits.logits.bytes];
            lib.copy_d2h(&mut bytes, logits.logits)?;
            let mut next = Vec::new();
            for row in bytes.chunks_exact(129280 * 4) {
                let mut best = (0, f32::NEG_INFINITY);
                for (i, b) in row.chunks_exact(4).enumerate() {
                    let value = f32::from_ne_bytes(b.try_into().unwrap());
                    ensure!(value.is_finite(), "non-finite verification logit");
                    if value > best.1 {
                        best = (i as u32, value);
                    }
                }
                next.push(best.0);
            }
            let decision = ds41rt_core::verify_dspark_greedy(&inputs, &next, 1, remaining)
                .map_err(anyhow::Error::msg)?;
            ensure!(!job.events.is_closed(), "client disconnected");
            self.commit(pass, requests, &mut batch, decision.accepted_inputs)?;
            tracing::debug!(target: "ds41rt::timing", proposed=inputs.len()-1, accepted=decision.accepted_inputs-1, emitted=decision.emitted.len(), draft_us, total_us=start.elapsed().as_micros() as u64, "speculative step");
            Ok(decision.emitted)
        })();
        if result.is_err() {
            pass.discard(&mut batch)?;
        }
        result
    }
}
