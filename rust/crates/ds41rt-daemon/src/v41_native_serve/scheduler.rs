use super::*;
mod independent;
use super::scores::BatchScores;
use crate::v41_backbone_cache::CacheLease;
use crate::v41_requests::RequestBatch;
use super::prefix::{ImageKeys, PrefixCache, SnapshotKind};

struct Active<'a> {
    constraint: Option<super::constraints::State<'a>>,
    id: u64,
    lease: CacheLease,
    job: NativeRequest,
    decoder: ds41rt_loader::StreamingTokenDecoder,
    anchor: u32,
    generated: usize,
    buffered: usize,
    lane: usize,
    finished: bool,
    cacheable: bool,
    tokens: Vec<u32>,
    image_keys: ImageKeys,
    next_after_commit: Option<TokenScores>,
}
impl Active<'_> {
    fn emit_one(&mut self, token: u32) -> Result<[Option<InferenceChunk>; 3]> {
        ensure!(!self.job.events.is_closed(), "client disconnected");
        if let Some(constraint) = &mut self.constraint { constraint.accept(token)?; }
        self.anchor = token;
        self.tokens.push(token);
        self.generated += 1;
        self.buffered += 1;
        let mut chunks = [None, None, None];
        let mut count = 0;
        let mut push = |chunk| { chunks[count] = Some(chunk); count += 1; };
        if token != 1 {
            if let Some(content) = self.decoder.step(token)? {
                push(InferenceChunk::Text { content, content_tokens: self.buffered });
                self.buffered = 0;
            }
        }
        if token == 1 || self.generated == self.job.max_tokens {
            if let Some(content) = self.decoder.finish()? {
                push(InferenceChunk::Text { content, content_tokens: self.buffered });
                self.buffered = 0;
            }
            if self.buffered > 0 {
                push(InferenceChunk::Text { content: String::new(), content_tokens: self.buffered });
                self.buffered = 0;
            }
            push(InferenceChunk::Finish { finish_reason: if token == 1 { InferenceFinishReason::Stop }
                else { InferenceFinishReason::Length } });
            self.finished = true;
            self.cacheable = true;
        }
        Ok(chunks)
    }
    fn emit(&mut self, tokens: &[u32]) -> Result<()> {
        for &token in tokens {
            for chunk in self.emit_one(token)?.into_iter().flatten() { self.job.events.blocking_send(Ok(chunk))?; }
            if self.finished { break; }
        }
        Ok(())
    }
}

pub(super) fn serve<'w, 'a>(lib: &'a NativeLibrary, args: &crate::cli::NativeServeArgs,
    runtime: &tokio::runtime::Runtime, receive: &mut mpsc::Receiver<NativeRequest>,
    first: &mut TargetPass<'w, 'a>, second: &mut TargetPass<'w, 'a>,
    requests: &mut Requests<'a>, first_transport: &mut NativeTp4Wave<'a>,
    second_transport: &mut NativeTp4Wave<'a>, mut draft: Option<&mut DraftRuntime<'_, 'a>>,
    vision: &mut crate::v41_vision::VisionRuntime<'a>,
) -> Result<()> {
    let mut active: Vec<Option<Active<'a>>> = (0..args.concurrency).map(|_| None).collect();
    let mut compiler = super::constraints::Compiler::new(lib, args.snapshot.join("tokenizer.json"));
    let mut id = 0u64;
    let mut closed = false;
    let mut prefixes = PrefixCache::new(args.prefix_cache_entries as usize);
    let limits = ds41rt_api::native_v41::NativeLimits::new(args.max_context_tokens, args.max_output_tokens)?;
    loop {
        // This point is reached only after both complete stacks have drained and
        // committed. No cache owner is migrated or retired inside a layer stack.
        for entry in &mut active {
            if entry.as_ref().is_some_and(|r| r.finished || r.job.events.is_closed()) {
                let request = entry.take().unwrap();
                if request.cacheable && requests.cache().request_id(request.lease).is_ok() {
                    if let Err(error) = prefixes.retain(SnapshotKind::Turn, &request.tokens, &request.image_keys, request.next_after_commit.as_ref().context("finished request has no retained logits")?,
                        request.id, request.lease, requests, draft.as_deref_mut()) {
                        tracing::warn!(%error, "completed request prefix was not retained");
                    }
                }
                requests.release_if_present(request.lease)?;
                if let Some(draft) = draft.as_deref_mut() { draft.release(request.id)?; }
            }
        }
        let mut loads = [0usize; 2];
        for request in active.iter().flatten() { loads[request.lane] += 1; }
        while loads[0].abs_diff(loads[1]) > 1 {
            let heavy = usize::from(loads[1] > loads[0]);
            let request = active.iter_mut().flatten().find(|r| r.lane == heavy).unwrap();
            request.lane = 1 - heavy;
            loads[heavy] -= 1; loads[1 - heavy] += 1;
        }
        // Admit available work at a completed boundary. Prefill currently owns
        // both lanes; mixed prefill/decode interleaving is a subsequent policy.
        while let Some(slot) = active.iter().position(Option::is_none) {
            let mut job = if active.iter().all(Option::is_none) && !closed {
                match receive.blocking_recv() { Some(job) => job, None => { closed = true; break; } }
            } else {
                match receive.try_recv() {
                    Ok(job) => job,
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => { closed = true; break; }
                }
            };
            if job.events.is_closed() { continue; }
            id = id.checked_add(1).context("request ID exhausted")?;
            let lease = requests.admit(slot, id)?;
            let lane = usize::from(loads[1] < loads[0]);
            let events = job.events.clone();
            let result = (|| -> Result<Active<'a>> {
                let mut constraint = job.constraint.as_ref().map(|spec| compiler.matcher(spec)).transpose()?;
                ensure!(!job.events.is_closed(), "client disconnected");
                let prompt = ds41rt_loader::encode_tokenizer_text(&args.snapshot, &job.prompt, false)?.token_ids;
                let (prompt, images) = if job.images.is_empty() { (prompt, Vec::new()) } else {
                    let expanded = ds41rt_loader::V41VisionPrompt::expand(&prompt,
                        std::mem::take(&mut job.images), limits.context() as usize)?;
                    (expanded.tokens, expanded.images)
                };
                job.max_tokens = limits.output_for_prompt(prompt.len(), job.max_tokens)?;
                let decoder = ds41rt_loader::streaming_token_decoder(&args.snapshot, false)?;
                if let Some(draft) = draft.as_deref_mut() { draft.admit(id)?; }
                let image_keys = prefixes.prepare_key(&prompt, &images)?;
                if !images.is_empty() {
                    requests.attach_images(lease, crate::v41_requests::RequestImages::new(&images)?)?;
                }
                let hit = prefixes.restore(&prompt, &image_keys, id, lease, requests, draft.as_deref_mut())?;
                let cached = hit.as_ref().map_or(0, |(end, _)| *end);
                let source_end = requests.cache().committed_end(lease)? as usize;
                prefixes.make_room(requests, &[(lease, (prompt.len() - source_end) as u32)])?;
                if !images.is_empty() {
                    let start = if requests.cache().stage(lease)? == crate::v41_backbone_cache::CacheStage::EncoderReplay {
                        requests.cache().history_end(lease)? as usize
                    } else { source_end };
                    let needed = requests.images(lease)?.needed(start, prompt.len())?;
                    let started = Instant::now();
                    for &index in &needed {
                        ensure!(!job.events.is_closed(), "client disconnected");
                        let features = vision.encode(&images[index].image)?;
                        let mut bytes = vec![0; features.bytes];
                        lib.copy_d2h(&mut bytes, features)?;
                        requests.install_image_features(lease, index, bytes)?;
                    }
                    tracing::info!(request_id=id, images=images.len(), encoded_images=needed.len(),
                        encoder_ms=started.elapsed().as_secs_f64()*1000.0, "native vision preparation");
                }
                drop(images);
                job.events.blocking_send(Ok(InferenceChunk::Ready {
                    system_fingerprint: Some(if draft.is_some() { "ds41rt-native-fp4-kv-dspark" }
                        else { "ds41rt-native-fp4-kv" }.into()),
                    prompt_usage: PromptUsage { prompt_tokens: prompt.len(), prompt_cache_hit_tokens: cached },
                }))?;
                first_transport.begin_request(); second_transport.begin_request();
                let scores = if cached == prompt.len() { hit.expect("complete prefix hit").1.context("exact prefix has no logits")? }
                else { prefill(lib, runtime, first, second, requests, first_transport,
                    second_transport, lease, &prompt, args.prefill_batch_tokens as usize, &job,
                    draft.as_deref_mut())? };
                if cached != prompt.len() {
                  if let Err(error) = prefixes.retain(SnapshotKind::Prompt, &prompt, &image_keys, &scores, id, lease, requests, draft.as_deref_mut()) {
                    tracing::warn!(%error, "prompt prefix was not retained");
                  }
                }
                tracing::debug!(request_id=id, prompt_tokens=prompt.len(), cached_tokens=cached, "native prefix admission");
                let mask = constraint.as_mut().map(|state| state.mask()).transpose()?.flatten();
                let anchor = scores.select(mask)?;
                Ok(Active { constraint, id, lease, job, decoder, anchor, generated: 0, buffered: 0, lane,
                    finished: false, cacheable: false, tokens: prompt, image_keys, next_after_commit: Some(scores) })
            })();
            match result {
                Ok(mut request) => {
                    if let Err(error) = request.emit(&[request.anchor]) {
                        let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                        request.finished = true;
                    }
                    active[slot] = Some(request); loads[lane] += 1;
                }
                Err(error) => {
                    // Other completed requests retain their caches.
                    let failure = error.downcast_ref::<ds41rt_api::native_v41::NativeFailure>()
                        .cloned().unwrap_or_else(|| format!("{error:#}").into());
                    let _ = events.blocking_send(Err(failure));
                    tracing::warn!(%error, "native request admission failed");
                    requests.release_if_present(lease)?;
                    if let Some(draft) = draft.as_deref_mut() { draft.release(id)?; }
                    first_transport.reset_connections(); second_transport.reset_connections();
                }
            }
        }
        if active.iter().all(Option::is_none) { if closed { break; } else { continue; } }
        let members: [Vec<usize>; 2] = std::array::from_fn(|lane| active.iter().enumerate()
            .filter_map(|(slot, request)| request.as_ref().filter(|r| r.lane == lane && !r.finished
                && !r.job.events.is_closed()).map(|_| slot)).collect());
        if members.iter().all(Vec::is_empty) { continue; }
        let capacity: Vec<_> = members.iter().flatten().map(|&slot| {
            let r = active[slot].as_ref().unwrap();
            (r.lease, if draft.is_some() { (r.job.max_tokens - r.generated).min(6) as u32 } else { 1 })
        }).collect();
        let room = prefixes.make_room(requests, &capacity);
        if let Err(error) = &room {
            if let Some(pressure) = error.downcast_ref::<crate::v41_compressor::SourcePoolExhausted>() {
                let slot = *members.iter().flatten().nth(pressure.work_index)
                    .context("pool pressure references an invalid append participant")?;
                let request = active[slot].as_mut().unwrap();
                let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                request.finished = true;
                request.cacheable = false;
                // No layer stack started. The next completed-boundary cleanup
                // releases only this owner's pages, then retries remaining work.
                continue;
            }
        }
        // With one active lane there is no peer to overlap. Retain the
        // ordinary C1 path and avoid shared-bank/async-delivery overhead.
        let result = room.and_then(|_| if args.independent_decode_lanes
            && members.iter().all(|lane| !lane.is_empty()) {
            independent::run(lib, runtime, first, second, requests, first_transport,
                second_transport, &mut active, draft.as_deref_mut(), &mut prefixes, receive)
        } else {
            round(lib, runtime, first, second, requests, first_transport, second_transport,
                &mut active, &members, draft.as_deref_mut())
        });
        if let Err(error) = result {
            first_transport.reset_connections(); second_transport.reset_connections();
            for request in active.iter_mut().flatten() {
                let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                request.finished = true;
                request.cacheable = false;
            }
        }
    }
    Ok(())
}

async fn execute_logits<'a>(lib: &'a NativeLibrary, pass: &mut TargetPass<'_, 'a>,
    requests: &Requests<'a>, batch: &mut Option<RequestBatch>, transport: &mut NativeTp4Wave<'a>,
    capture_routes: bool,
) -> Result<BatchScores> {
    let Some(batch) = batch else { return BatchScores::new(Vec::new()); };
    pass.set_route_capture(capture_routes);
    let result = async {
        let selected: Vec<_> = (0..batch.cache()?.positions().len()).collect();
        let logits = unsafe { pass.execute(requests, batch, transport, 0, &selected).await? };
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        BatchScores::new(bytes)
    }.await;
    pass.set_route_capture(false);
    result
}

fn prepare_decode_lane<'a>(requests: &mut Requests<'a>, active: &[Option<Active<'a>>],
    members: &[usize], inputs: &[Vec<u32>], speculative: bool) -> Result<RequestBatch> {
    let work: Vec<_> = members.iter().zip(inputs).map(|(&slot, tokens)| RequestTokens {
        lease: active[slot].as_ref().unwrap().lease, tokens, image_mask: None,
        kind: if speculative { ExpertV2SourceKind::MtpVerify } else { ExpertV2SourceKind::Decode },
    }).collect();
    requests.prepare(&work)
}

fn round<'w, 'a>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    first: &mut TargetPass<'w, 'a>, second: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    first_transport: &mut NativeTp4Wave<'a>, second_transport: &mut NativeTp4Wave<'a>,
    active: &mut [Option<Active<'a>>], members: &[Vec<usize>; 2], mut draft: Option<&mut DraftRuntime<'_, 'a>>,
) -> Result<()> {
    let started = Instant::now();
    let speculative = draft.is_some();
    let adaptive = draft.as_deref().is_some_and(DraftRuntime::adaptive_enabled);
    let capture_routes = draft.as_deref().is_some_and(DraftRuntime::capture_routes);
    let mut inputs = [Vec::new(), Vec::new()];
    let mut batches = [None, None];
    let mut draft_us = 0u64;
    let mut prepare_us = 0u64;
    for lane in 0..2 {
        ensure!(members[lane].len() <= 8, "decode lane exceeds eight requests");
        let seeds = members[lane].iter().map(|&slot| {
            let r = active[slot].as_ref().unwrap();
            Ok((r.id, r.anchor, requests.cache().committed_end(r.lease)?, r.job.max_tokens - r.generated))
        }).collect::<Result<Vec<_>>>()?;
        let draft_start = Instant::now();
        inputs[lane] = if seeds.is_empty() { Vec::new() }
            else if let Some(draft) = draft.as_deref_mut() { draft.propose(lib, &seeds)? }
            else { seeds.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect() };
        for (&slot, input) in members[lane].iter().zip(&mut inputs[lane]) {
            if let Some(constraint) = &active[slot].as_ref().unwrap().constraint {
                constraint.truncate_proposal(input)?;
            } else if let Some(draft) = draft.as_deref() {
                // Select immediately for this request, before preparing this lane's
                // Engram reads; no other lane's proposals or history are needed.
                let length = draft.confidence_prefix(active[slot].as_ref().unwrap().id, input.len() - 1)?;
                input.truncate(length + 1);
            }
        }
        if draft.as_deref().is_some_and(DraftRuntime::reuse_enabled)
            && !members[lane].is_empty() && members[lane].iter().all(|&slot| active[slot].as_ref().unwrap().constraint.is_none()) {
            if let Some(draft) = draft.as_deref() {
                let candidates: Vec<_> = members[lane].iter().zip(&inputs[lane])
                    .map(|(&slot, input)| (active[slot].as_ref().unwrap().id, lane, input.len() - 1)).collect();
                if let Some(lengths) = draft.select_reuse_prefixes(&candidates)? {
                    for (input, length) in inputs[lane].iter_mut().zip(lengths) { input.truncate(length + 1); }
                }
            }
        }
        draft_us += draft_start.elapsed().as_micros() as u64;
        if !members[lane].is_empty() && !adaptive {
            // Token IDs are now known: start this lane's mapped Engram reads
            // while the other lane generates its draft proposals. RequestBatch
            // cancels its I/O on drop if subsequent preparation fails.
            let prepare_start = Instant::now();
            batches[lane] = Some(prepare_decode_lane(requests, active, &members[lane], &inputs[lane], speculative)?);
            prepare_us += prepare_start.elapsed().as_micros() as u64;
        }
    }
    if adaptive {
        // Both proposals must be available to price their joint row/expert cost.
        // Keep the existing early Engram preparation unchanged for fixed mode.
        let identities: Vec<_> = (0..2).flat_map(|lane| members[lane].iter().zip(&inputs[lane])
            .map(move |(&slot, input)| (slot, lane, input.len() - 1))).collect();
        let candidates: Vec<_> = identities.iter().map(|&(slot, lane, length)|
            (active[slot].as_ref().unwrap().id, lane, length)).collect();
        // Grammar-conditioned acceptance has not been calibrated yet.
        if identities.iter().all(|&(slot, _, _)| active[slot].as_ref().unwrap().constraint.is_none()) {
            if let Some(lengths) = draft.as_deref().unwrap().select_prefixes(&candidates, draft_us)? {
                for (input, length) in inputs.iter_mut().flatten().zip(lengths) { input.truncate(length + 1); }
            }
        }
        for lane in 0..2 {
            if !members[lane].is_empty() {
                let start = Instant::now();
                batches[lane] = Some(prepare_decode_lane(requests, active, &members[lane], &inputs[lane], speculative)?);
                prepare_us += start.elapsed().as_micros() as u64;
            }
        }
    }
    let prepared_us = started.elapsed().as_micros() as u64;
    let proposed: usize = inputs.iter().flatten().map(|tokens| tokens.len() - 1).sum();
    let [a, b] = &mut batches;
    // Drain both futures even if one fails, before discarding private device state.
    let results = runtime.block_on(async { tokio::join!(
        async {
            let result = execute_logits(lib, first, requests, a, first_transport, capture_routes).await;
            (result, started.elapsed().as_micros() as u64)
        },
        async {
            let result = execute_logits(lib, second, requests, b, second_transport, capture_routes).await;
            (result, started.elapsed().as_micros() as u64)
        },
    ) });
    let executed_us = started.elapsed().as_micros() as u64;
    let lane_done_us = [results.0.1, results.1.1];
    let result = (|| -> Result<()> {
        let mut accepted_drafts = 0u32;
        let mut emitted = 0usize;
        let next = [results.0.0?, results.1.0?];
        for (lane, pass) in [&mut *first, &mut *second].into_iter().enumerate() {
            let (accepted, count, emissions) = commit_lane(lane, pass, requests, active, &members[lane],
                &inputs[lane], &mut batches[lane], &next[lane], draft.as_deref_mut(),
                capture_routes, executed_us-prepared_us)?;
            accepted_drafts += accepted; emitted += count;
            for (&slot, tokens) in members[lane].iter().zip(emissions) {
                let request = active[slot].as_mut().unwrap();
                if let Err(error) = request.emit(&tokens) {
                    let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                    request.finished = true;
                }
            }
        }
        tracing::debug!(target: "ds41rt::timing", speculative,
            requests=members[0].len()+members[1].len(), lane0=members[0].len(), lane1=members[1].len(),
            proposed, accepted=accepted_drafts, emitted, draft_us, prepare_us,
            lane0_verify_done_us=lane_done_us[0], lane1_verify_done_us=lane_done_us[1],
            lane0_join_wait_us=if members[0].is_empty() { 0 } else { executed_us-lane_done_us[0] },
            lane1_join_wait_us=if members[1].is_empty() { 0 } else { executed_us-lane_done_us[1] },
            verify_us=executed_us-prepared_us, total_us=started.elapsed().as_micros() as u64,
            "native scheduler round");
        Ok(())
    })();
    if result.is_err() {
        // Cleanup is performed by the caller after both full execution futures
        // have returned. Successful commits no longer have a live batch.
        for (pass, batch) in [first, second].into_iter().zip(&mut batches) {
            if let Some(batch) = batch { pass.discard(batch)?; }
        }
    }
    result
}

fn commit_lane<'w, 'a>(lane: usize, pass: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    active: &mut [Option<Active<'a>>], members: &[usize], inputs: &[Vec<u32>],
    owned_batch: &mut Option<RequestBatch>, next: &BatchScores,
    mut draft: Option<&mut DraftRuntime<'_, 'a>>, capture_routes: bool, verify_us: u64,
) -> Result<(u32, usize, Vec<Vec<u32>>)> {
    let mut accepted_drafts = 0u32;
    let mut emitted = 0usize;
    let Some(batch) = owned_batch else { return Ok((0, 0, Vec::new())); };
    let mut offset = 0;
    let mut accepted = Vec::new();
    let mut emissions = Vec::new();
    let mut next_after_commit = Vec::new();
    for (&slot, input) in members.iter().zip(inputs) {
        let request = active[slot].as_ref().unwrap();
        let constrained = request.constraint.as_ref().map(|state|
            state.select_verification(next, offset, input)).transpose()?;
        let selected = constrained.as_deref().unwrap_or(&next.best[offset..offset + input.len()]);
        let decision = ds41rt_core::verify_dspark_greedy(input,
            selected, 1, request.job.max_tokens - request.generated)
            .map_err(anyhow::Error::msg)?;
        if tracing::enabled!(target: "ds41rt::logit_trace", tracing::Level::DEBUG) {
            let top_two = (offset..offset + input.len())
                .map(|row| next.top_two(row)).collect::<Result<Vec<_>>>()?;
            tracing::debug!(target: "ds41rt::logit_trace",
                request_id=request.id, lane, generated=request.generated,
                context_tokens=requests.cache().committed_end(request.lease)?,
                input=?input, selected=?selected, top_two=?top_two,
                accepted_inputs=decision.accepted_inputs,
                emitted=?decision.emitted,
                constrained=request.constraint.is_some(),
                "native verification logits");
        }
        if let Some(confidence) = draft.as_deref()
            .and_then(|draft| draft.confidence_trace(request.id)) {
            // Agreement after the first mismatch is conditional on a
            // rejected history and must not be treated as acceptance.
            let matched = input.iter().skip(1).zip(selected.iter())
                .take_while(|(proposal, target)| proposal == target).count();
            tracing::debug!(target: "ds41rt::draft_policy",
                request_id=request.id, lane, generated=request.generated,
                context_tokens=requests.cache().committed_end(request.lease)?,
                verifier_rows=input.len(), lane_rows=inputs.iter().map(Vec::len).sum::<usize>(),
                constrained=request.constraint.is_some(), raw_confidence=?confidence,
                matched_prefix=matched, accepted_inputs=decision.accepted_inputs,
                eos=decision.eos, length_limit=decision.length_limit,
                verify_us,
                "native draft policy observation");
        }
        accepted_drafts += decision.accepted_inputs - 1;
        emitted += decision.emitted.len();
        let finishing = decision.emitted.contains(&1)
            || request.generated + decision.emitted.len() >= request.job.max_tokens;
        next_after_commit.push(if finishing {
            Some(next.retain(offset + decision.accepted_inputs as usize - 1)?)
        } else { None });
        offset += input.len(); accepted.push(decision.accepted_inputs); emissions.push(decision.emitted);
    }
    if let Some(draft) = draft.as_deref_mut() {
        draft.commit_batch(pass, requests, batch, &accepted)?;
        if capture_routes {
            let mut offset = 0;
            for ((&slot, input), &count) in members.iter().zip(inputs).zip(&accepted) {
                draft.observe_accepted_routes(active[slot].as_ref().unwrap().id, offset,
                    count as usize, pass.captured_routes())?;
                offset += input.len();
            }
        }
    }
    else { pass.commit(requests, batch, &accepted)?; }
    *owned_batch = None; // Only successful commits relinquish cleanup ownership.
    for (&slot, next_token) in members.iter().zip(next_after_commit) {
        active[slot].as_mut().unwrap().next_after_commit = next_token;
    }
    Ok((accepted_drafts, emitted, emissions))
}
