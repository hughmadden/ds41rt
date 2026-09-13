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
    request_limit: usize,
    draft_limit: usize,
    // Downloaded only for the experimental adaptive policy or explicit diagnostics.
    confidence_trace: std::collections::BTreeMap<u64, Vec<f32>>,
    adaptive: Option<ds41rt_core::DsparkRouteHistory>,
    confidence_cutoff: Option<f64>,
}
struct DraftRequest {
    leases: [WindowLease; 3],
    rng: ds41rt_core::DsparkRng,
    slot: usize,
}
pub(crate) struct DraftPrefix<'a> {
    windows: Vec<crate::v41_dspark_cache::DsparkPrefix<'a>>,
}
impl<'w, 'a> DraftRuntime<'w, 'a> {
    pub fn new(
        lib: &'a NativeLibrary,
        weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>,
        head: &'w VocabularyHead<'a>,
        capacity: u32,
    ) -> Result<Self> {
        Self::with_requests(lib, weights, table, head, capacity, 16)
    }
    pub fn with_requests(
        lib: &'a NativeLibrary, weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>, head: &'w VocabularyHead<'a>,
        capacity: u32, requests: u32,
    ) -> Result<Self> {
        ensure!((1..=16).contains(&requests), "invalid draft request limit");
        let window = || DsparkWindow::new(lib, requests as usize, capacity,
            DsparkWindow::device_bytes(requests as usize, capacity)?);
        Ok(Self {
            main: weights.main_context(capacity, DsparkMainContext::device_bytes(lib, capacity)?)?,
            chain: weights.draft(table, head, requests, weights.draft_bytes(requests)?)?,
            windows: [window()?, window()?, window()?],
            requests: Default::default(),
            captured: Default::default(),
            request_limit: requests as usize,
            draft_limit: 5,
            confidence_trace: Default::default(),
            adaptive: None,
            confidence_cutoff: None,
        })
    }
    pub fn admit(&mut self, id: u64) -> Result<()> {
        ensure!(!self.requests.contains_key(&id), "draft request already admitted");
        let slot = (0..self.request_limit).find(|slot| self.requests.values().all(|request| request.slot != *slot))
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
    pub fn set_draft_limit(&mut self, limit: u8) -> Result<()> {
        ensure!((1..=5).contains(&limit), "draft limit must be one through five");
        self.draft_limit = limit as usize;
        Ok(())
    }
    pub fn set_adaptive(&mut self, enabled: bool) {
        self.adaptive = enabled.then(ds41rt_core::DsparkRouteHistory::default);
    }
    pub fn set_confidence_cutoff(&mut self, threshold: Option<f64>) {
        self.confidence_cutoff = threshold;
    }
    pub fn confidence_prefix(&self, id: u64, maximum: usize) -> Result<usize> {
        let Some(threshold) = self.confidence_cutoff else { return Ok(maximum); };
        if maximum == 0 { return Ok(0); }
        let logits = self.confidence_trace(id).context("missing draft confidence")?;
        ensure!(maximum <= logits.len(), "draft confidence prefix exceeds output");
        let probabilities: Vec<_> = logits[..maximum].iter().map(|&x| {
            let x = f64::from(x);
            if x >= 0. { 1. / (1. + (-x).exp()) } else { x.exp() / (1. + x.exp()) }
        }).collect();
        // Preserve the existing minimum of one draft: anchor-only numerical
        // specialization needs its own qualification before voluntary selection.
        ds41rt_core::select_dspark_confidence_prefix(&probabilities, threshold, 1)
            .map_err(anyhow::Error::msg)
    }
    pub fn adaptive_enabled(&self) -> bool { self.adaptive.is_some() }
    pub fn observe_accepted_routes(&mut self, id: u64, offset: usize, accepted: usize,
        routes: &[Vec<[u32; 6]>]) -> Result<()> {
        let Some(history) = &mut self.adaptive else { return Ok(()); };
        ensure!(routes.len() == 40 && routes.iter().all(|r| offset + accepted <= r.len()),
            "adaptive route capture is incomplete");
        for (layer, rows) in routes.iter().enumerate() {
            history.observe_accepted(id, layer, &rows[offset..offset + accepted])
                .map_err(anyhow::Error::msg)?;
        }
        Ok(())
    }
    pub fn select_prefixes(&self, requests: &[(u64, usize, usize)], draft_us: u64)
        -> Result<Option<Vec<usize>>> {
        let started = std::time::Instant::now();
        let Some(history) = &self.adaptive else { return Ok(None); };
        let Some(forecast) = history.forecast(requests) else { return Ok(None); };
        let mut probabilities = Vec::with_capacity(requests.len());
        for &(id, _, maximum) in requests {
            let Some(confidence) = self.confidence_trace(id) else { return Ok(None); };
            probabilities.push(confidence[..maximum].iter().map(|&x| {
                let x = f64::from(x);
                if x >= 0. { 1. / (1. + (-x).exp()) } else { x.exp() / (1. + x.exp()) }
            }).collect::<Vec<_>>());
        }
        // Preliminary corrected-path fit in microseconds. Both lanes' expert
        // unions are counted separately. These coefficients are experimental;
        // missing routing history keeps the full fixed-length policy.
        let cost = |lengths: &[usize]| draft_us as f64 + 1000. + 19864.
            + 803. * (requests.len() + lengths.iter().sum::<usize>()) as f64
            + 636. * forecast.mean_unique_experts(lengths)
            + 3168. * (forecast.lane_count() - 1) as f64;
        let full: Vec<_> = requests.iter().map(|r| r.2).collect();
        let expected_full: f64 = probabilities.iter().map(|p| {
            let mut product = 1.;
            1. + p.iter().map(|v| { product *= v; product }).sum::<f64>()
        }).sum();
        let full_cost = cost(&full);
        // The pilot qualifies 2..6 verifier rows. M1 uses separate numerical
        // specializations and is not entered voluntarily by this policy.
        let minimum: Vec<_> = probabilities.iter().map(|p| usize::from(!p.is_empty())).collect();
        let result = ds41rt_core::select_dspark_prefixes_bounded(
            &probabilities.iter().map(Vec::as_slice).collect::<Vec<_>>(), &minimum, cost)
            .map_err(anyhow::Error::msg)?;
        let enabled = result.expected_tokens / result.cost_us > 1.02 * expected_full / full_cost;
        tracing::debug!(target: "ds41rt::adaptive_policy", full=?full, selected=?result.lengths,
            predicted_full_us=full_cost, predicted_selected_us=result.cost_us,
            expected_full, expected_selected=result.expected_tokens, enabled,
            evaluated_shapes=result.evaluated_shapes, selection_us=started.elapsed().as_micros() as u64,
            "native adaptive prefix selection");
        Ok(enabled.then_some(result.lengths))
    }
    pub fn release(&mut self, id: u64) -> Result<()> {
        if let Some(history) = &mut self.adaptive { history.release(id); }
        self.confidence_trace.remove(&id);
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
    pub fn retain_prefix(&mut self, id: u64, end: u64) -> Result<DraftPrefix<'a>> {
        let request = self.requests.get(&id).context("draft request not admitted")?;
        for (window, lease) in self.windows.iter().zip(request.leases) {
            ensure!(window.committed_end(lease)? == Some(end), "draft and target prefix frontiers differ");
        }
        let windows = self.windows.iter_mut().zip(request.leases)
            .map(|(window, lease)| window.retain_prefix(lease)).collect::<Result<Vec<_>>>()?;
        Ok(DraftPrefix { windows })
    }
    pub fn restore_prefix(&mut self, id: u64, end: u64, prefix: &DraftPrefix<'a>) -> Result<()> {
        let request = self.requests.get(&id).context("draft request not admitted")?;
        ensure!(prefix.windows.len() == 3 && prefix.windows.iter().all(|p| p.end() == end),
            "retained draft and target frontiers differ");
        let leases = request.leases;
        for (stage, saved) in prefix.windows.iter().enumerate() {
            if let Err(error) = self.windows[stage].restore_prefix(leases[stage], saved) {
                if let Err(cleanup) = self.release(id) {
                    tracing::error!(%cleanup, "releasing failed draft prefix restore");
                }
                return Err(error);
            }
        }
        Ok(())
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
        for &(id, _, _, _) in inputs {
            self.confidence_trace.remove(&id);
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
        if self.adaptive.is_some() || self.confidence_cutoff.is_some() || tracing::enabled!(target: "ds41rt::draft_policy", tracing::Level::DEBUG) {
            let confidence = self.chain.draft_output()?[2];
            ensure!(confidence.bytes == 5 * count * 4, "draft confidence extent differs");
            let mut bytes = vec![0; confidence.bytes];
            lib.copy_d2h(&mut bytes, confidence)?;
            let values: Vec<_> = bytes.chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();
            for (row, &(_, &(id, _, _, _))) in active.iter().enumerate() {
                self.confidence_trace.insert(id,
                    (0..5).map(|step| values[step * count + row]).collect());
            }
        }
        for (row, &(output, &(_, anchor, _, remaining))) in active.iter().enumerate() {
            let tokens: Vec<_> = (0..6).map(|step| packed[step * count + row]).collect();
            ensure!(tokens[0] == anchor && tokens.iter().all(|&token| token < 129280), "invalid draft tokens");
            outputs[output] = tokens[..remaining.min(self.draft_limit + 1)].to_vec();
        }
        Ok(outputs)
    }

    pub fn confidence_trace(&self, id: u64) -> Option<&[f32]> {
        self.confidence_trace.get(&id).map(Vec::as_slice)
    }
}
