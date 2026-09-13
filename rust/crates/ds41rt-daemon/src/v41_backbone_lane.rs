//! Reusable backbone execution storage; request caches and index selections live outside the lane.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_attention_output::{AttentionOutputWave, AttentionOutputWeights};
use crate::v41_attention_query::{AttentionQueryOutput, AttentionQueryWave, AttentionQueryWeights};
use crate::v41_backbone_hc::BackboneHcWeights;
use crate::v41_backbone_router::{BackboneRouterWave, BackboneRouterWeights, ExpertRow};
use crate::v41_backbone_shared::{BackboneSharedWave, BackboneSharedWeights, SharedOutput};
use crate::v41_block::{BackboneBlockWave, BlockOutput, FfnInput, PreparedBlockInput};
use crate::v41_engram::{layer::EngramGate, EngramDeviceView};
use crate::v41_experts::coordinator::{NativeFfnOutput, NativeTp4Wave};
use crate::v41_index_selection::IndexSelectionOutput;
use crate::v41_sparse_attention::{AttentionRequest, SparseAttentionWave};
use crate::v41_target_embedding::TargetEmbedding;
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::OfficialV41Catalog;

struct LayerWeights<'a> {
    hc: BackboneHcWeights<'a>,
    query: AttentionQueryWeights<'a>,
    projection: AttentionOutputWeights<'a>,
    shared: BackboneSharedWeights<'a>,
    router: BackboneRouterWeights<'a>,
}

/// All 40 layers' mHC, query, output, shared-expert and router weights. This
/// excludes window/compressor/index weights, engram, vision and dSpark.
pub(crate) struct BackboneLaneWeights<'a> {
    library: &'a NativeLibrary,
    layers: Vec<LayerWeights<'a>>,
}
impl<'a> BackboneLaneWeights<'a> {
    fn layer_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<[usize; 5]> {
        Ok([
            BackboneHcWeights::device_bytes(catalog, layer)?,
            AttentionQueryWeights::device_bytes(library, catalog, layer)?,
            AttentionOutputWeights::device_bytes(library, catalog, layer)?,
            BackboneSharedWeights::device_bytes(library, catalog, layer)?,
            BackboneRouterWeights::device_bytes(catalog, layer)?,
        ])
    }
    /// Conservative device budget including each loader's transient peak.
    pub fn device_bytes(library: &NativeLibrary, catalog: &OfficialV41Catalog) -> Result<usize> {
        (0..40).try_fold(0usize, |total, layer| {
            Self::layer_bytes(library, catalog, layer)?
                .into_iter()
                .try_fold(total, |n, b| {
                    n.checked_add(b)
                        .ok_or_else(|| anyhow::anyhow!("backbone weight budget overflow"))
                })
        })
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog)? <= budget,
            "backbone lane weights exceed budget"
        );
        let mut layers = Vec::with_capacity(40);
        for layer in 0..40 {
            let [hc, query, projection, shared, router] =
                Self::layer_bytes(library, catalog, layer)?;
            layers.push(LayerWeights {
                hc: BackboneHcWeights::load(library, catalog, layer, hc, staging)?,
                query: AttentionQueryWeights::load(library, catalog, layer, query, staging)?,
                projection: AttentionOutputWeights::load(
                    library, catalog, layer, projection, staging,
                )?,
                shared: BackboneSharedWeights::load(library, catalog, layer, shared, staging)?,
                router: BackboneRouterWeights::load(library, catalog, layer, router, staging)?,
            });
        }
        Ok(Self { library, layers })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Prepared,
    Query,
    Ffn,
    SharedReady,
    Complete,
    Invalid,
}

/// Dispatch routed work from input before executing the shared contribution on
/// RTX. Both consume the same preserved normalized rows and execution identity.
pub(crate) struct LaneFfn<'s, 'w, 'a> {
    pub input: FfnInput<'s>,
    shared: &'s mut BackboneSharedWave<'w, 'a>,
    router: &'s mut BackboneRouterWave<'w, 'a>,
    library: &'a NativeLibrary,
    phase: &'s mut Phase,
    route_capture: Option<&'s mut Vec<Vec<[u32; 6]>>>,
}
impl LaneFfn<'_, '_, '_> {
    /// # Safety
    /// Metadata and mask identify the actual requests/modality in input order.
    /// All external producers are complete and no writes race this lane.
    pub async unsafe fn execute_tp4<'t>(
        &mut self,
        transport: &'t mut NativeTp4Wave<'_>,
        placement: u64,
        image_mask: &[u8],
        rows: &[ExpertRow],
    ) -> Result<NativeFfnOutput<'t>> {
        let input = &self.input;
        let router = &mut self.router;
        let shared = &mut self.shared;
        let library = self.library;
        let route_capture = &mut self.route_capture;
        complete_ffn(self.phase, async {
            let timing = std::time::Instant::now();
            let routed = unsafe { router.execute_ffn(input, image_mask)? };
            let request = unsafe { routed.expert_request(library, placement, rows)? };
            if let Some(capture) = route_capture.as_deref_mut() {
                let output = &mut capture[input.layer];
                output.clear();
                output.extend(request.request().routes.chunks_exact(6)
                    .map(|routes| std::array::from_fn(|i| routes[i].expert_id)));
            }
            let routed_us = timing.elapsed().as_micros() as u64;
            let pending = transport.dispatch_ffn(&request).await?;
            let dispatched_us = timing.elapsed().as_micros() as u64;
            let contribution = unsafe { shared.execute_ffn(input)? };
            let shared_us = timing.elapsed().as_micros() as u64;
            let result = unsafe { pending.finish(&contribution).await }?;
            tracing::debug!(target: "ds41rt::timing", layer=input.layer, rows=rows.len(), routed_us, dispatch_us=dispatched_us-routed_us, shared_us=shared_us-dispatched_us, collect_us=timing.elapsed().as_micros() as u64-shared_us, "target experts");
            if tracing::enabled!(target: "ds41rt::route_policy", tracing::Level::DEBUG) {
                let ffn_us = timing.elapsed().as_micros() as u64;
                // The dispatch request already owns these CPU-side routes.
                // No extra device read or worker instrumentation is needed.
                let route_ids: Vec<_> = request.request().routes.iter().map(|r| r.expert_id).collect();
                let owners: Vec<_> = rows.iter().map(|r| (r.request_id, r.position)).collect();
                let unique_experts = route_ids.iter().collect::<std::collections::BTreeSet<_>>().len();
                tracing::debug!(target: "ds41rt::route_policy", layer=input.layer,
                    rows=rows.len(), unique_experts, ffn_us, routed_us,
                    remote_and_shared_us=ffn_us-routed_us,
                    owners=?owners, route_ids=?route_ids,
                    "native route policy observation");
            }
            Ok(result)
        })
        .await
    }
    /// # Safety
    /// No external writes race the preserved FFN input or shared workspace.
    pub unsafe fn execute_shared(&mut self) -> Result<SharedOutput<'_>> {
        let prior = std::mem::replace(self.phase, Phase::Invalid);
        ensure!(prior == Phase::Ffn, "lane shared FFN is not pending");
        let output = unsafe { self.shared.execute_ffn(&self.input)? };
        *self.phase = Phase::SharedReady;
        Ok(output)
    }
}

/// Invalidate before the first operation is polled. Dropping a polled future
/// therefore requires lane restart; successful completion alone republishes it.
async fn complete_ffn<T>(
    phase: &mut Phase,
    work: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    let prior = std::mem::replace(phase, Phase::Invalid);
    ensure!(prior == Phase::Ffn, "lane FFN work is not pending");
    let result = work.await?;
    *phase = Phase::SharedReady;
    Ok(result)
}

pub(crate) struct BackboneLane<'w, 'a> {
    weights: &'w BackboneLaneWeights<'a>,
    block: BackboneBlockWave<'w, 'a>,
    query: AttentionQueryWave<'w, 'a>,
    projection: AttentionOutputWave<'w, 'a>,
    shared: BackboneSharedWave<'w, 'a>,
    router: BackboneRouterWave<'w, 'a>,
    sparse: SparseAttentionWave<'a>,
    layer: usize,
    phase: Phase,
    capture_routes: bool,
    route_capture: Vec<Vec<[u32; 6]>>,
}
impl<'w, 'a> BackboneLane<'w, 'a> {
    pub fn workspace_bytes(library: &NativeLibrary, capacity: u32) -> Result<[usize; 6]> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid backbone lane capacity"
        );
        Ok([
            BackboneBlockWave::device_bytes(capacity as usize)?,
            AttentionQueryWave::device_bytes(library, capacity)?,
            AttentionOutputWave::device_bytes(library, capacity)?,
            BackboneSharedWave::device_bytes(library, capacity)?,
            SparseAttentionWave::device_bytes(capacity as usize)?,
            BackboneRouterWave::device_bytes(capacity)?,
        ])
    }
    /// One lane owns one allocation of each workspace group. Independent lanes
    /// borrow the same immutable weights and allocate distinct mutable storage.
    pub fn new(weights: &'w BackboneLaneWeights<'a>, capacity: u32, budget: usize) -> Result<Self> {
        let sizes = Self::workspace_bytes(weights.library, capacity)?;
        let total = sizes.into_iter().try_fold(0usize, |n, b| {
            n.checked_add(b)
                .ok_or_else(|| anyhow::anyhow!("backbone lane budget overflow"))
        })?;
        ensure!(total <= budget, "backbone lane exceeds budget");
        ensure!(
            weights.layers.len() == 40,
            "backbone lane requires all 40 layers"
        );
        let first = &weights.layers[0];
        Ok(Self {
            weights,
            block: first.hc.block(capacity as usize, sizes[0])?,
            query: first.query.wave(capacity, sizes[1])?,
            projection: first.projection.wave(capacity, sizes[2])?,
            shared: first.shared.wave(capacity, sizes[3])?,
            sparse: SparseAttentionWave::new(weights.library, capacity as usize, sizes[4])?,
            router: first.router.wave(capacity, sizes[5])?,
            layer: 0,
            phase: Phase::Idle,
            capture_routes: false,
            route_capture: Vec::new(),
        })
    }
    fn enter(&mut self, expected: Phase) -> Result<()> {
        let prior = std::mem::replace(&mut self.phase, Phase::Invalid);
        ensure!(
            prior == expected,
            "backbone lane phase differs: expected {expected:?}, got {prior:?}"
        );
        Ok(())
    }
    /// Cancel or start another wave. External consumers must have finished;
    /// borrowed outputs prevent safe callers from restarting during consumption.
    pub fn restart(&mut self) -> Result<()> {
        self.phase = Phase::Invalid;
        self.block.reset();
        let first = &self.weights.layers[0];
        self.query.rebind(&first.query)?;
        self.projection.rebind(&first.projection)?;
        self.shared.rebind(&first.shared)?;
        self.router.rebind(&first.router)?;
        self.block.restart(&first.hc)?;
        self.layer = 0;
        self.phase = Phase::Idle;
        Ok(())
    }
    pub fn restart_decoder(&mut self, encoder: &BlockOutput<'_>) -> Result<()> {
        self.phase = Phase::Invalid;
        self.block.reset();
        let decoder = &self.weights.layers[20];
        self.query.rebind(&decoder.query)?;
        self.projection.rebind(&decoder.projection)?;
        self.shared.rebind(&decoder.shared)?;
        self.router.rebind(&decoder.router)?;
        self.block.initialize_decoder(&decoder.hc, encoder)?;
        self.layer = 20;
        self.phase = Phase::Prepared;
        Ok(())
    }
    /// # Safety
    /// Same finite, completed embedding and exclusive-storage contract as the
    /// component owners. This text entry does not implement vision replacement.
    pub unsafe fn begin_embedded(
        &mut self,
        embedding: &TargetEmbedding<'_>,
    ) -> Result<AttentionQueryOutput<'_>> {
        self.enter(Phase::Idle)?;
        ensure!(self.layer == 0, "embedding requires layer zero");
        let result = unsafe {
            self.block
                .begin_embedded_attention(&mut self.query, embedding)?
        };
        self.phase = Phase::Query;
        Ok(result)
    }
    pub fn pending_engram(&self) -> Result<(usize, &[u64])> {
        ensure!(self.phase == Phase::Prepared, "backbone lane input not prepared");
        self.block.pending_engram()
    }
    pub fn prepared_input(&self) -> Result<PreparedBlockInput<'_>> {
        ensure!(
            self.phase == Phase::Prepared,
            "backbone lane input not prepared"
        );
        self.block.prepared_input()
    }
    /// # Safety
    /// Gathered rows/history are associated with this lane's exact request order,
    /// and all producers/consumers are complete as required by EngramGate.
    pub unsafe fn apply_engram(
        &mut self,
        gate: &mut EngramGate<'_, '_>,
        rows: &EngramDeviceView,
    ) -> Result<()> {
        self.enter(Phase::Prepared)?;
        unsafe {
            self.block.apply_engram(gate, rows)?;
        }
        self.phase = Phase::Prepared;
        Ok(())
    }
    /// # Safety
    /// Required dSpark tap consumers must finish before this call. The block
    /// checks that engram has completed on layers 1 and 14.
    pub unsafe fn begin_prepared(&mut self) -> Result<AttentionQueryOutput<'_>> {
        self.enter(Phase::Prepared)?;
        let result = unsafe { self.block.begin_prepared_attention(&mut self.query)? };
        self.phase = Phase::Query;
        Ok(result)
    }
    pub fn query_output(&self) -> Result<AttentionQueryOutput<'_>> {
        ensure!(
            self.phase == Phase::Query,
            "backbone lane query unavailable"
        );
        self.query.output()
    }
    /// # Safety
    /// Request windows, sources, selected rows and sink obey the sparse attention
    /// owner's matching-layer and completed-producer contract. The returned FFN
    /// work keeps its input stable while routed dispatch and shared work proceed.
    pub unsafe fn attention_ffn(
        &mut self,
        sink: Ds41rtDeviceBuffer,
        requests: &[AttentionRequest<'_>],
        selection: Option<&IndexSelectionOutput<'_>>,
    ) -> Result<LaneFfn<'_, 'w, 'a>> {
        self.enter(Phase::Query)?;
        let query = self.query.output()?;
        let timing = std::time::Instant::now();
        let attention = unsafe {
            self.sparse
                .execute_query(&query, sink, requests, selection)?
        };
        let sparse_us = timing.elapsed().as_micros() as u64;
        let projected = unsafe { self.projection.execute_attention(&attention)? };
        let projected_us = timing.elapsed().as_micros() as u64;
        let input = unsafe { self.block.begin_ffn(&projected)? };
        tracing::debug!(target: "ds41rt::timing", layer=self.layer, rows=projected.rows,
            sparse_us, projection_us=projected_us-sparse_us,
            ffn_prepare_us=timing.elapsed().as_micros() as u64-projected_us,
            "target attention stages");
        self.phase = Phase::Ffn;
        Ok(LaneFfn {
            input,
            shared: &mut self.shared,
            router: &mut self.router,
            library: self.weights.library,
            phase: &mut self.phase,
            route_capture: if self.capture_routes { Some(&mut self.route_capture) } else { None },
        })
    }
    /// Produce learned index selections from this lane's completed query.
    /// # Safety
    /// Cache proposals correspond to the same admitted query batch, with all
    /// producers complete and no external writes racing these owners.
    pub unsafe fn select_index(
        &self,
        index: &mut crate::v41_index_lane::IndexLane<'_, '_>,
        cache: &crate::v41_backbone_cache::CacheAttention<'_>,
    ) -> Result<()> {
        let query = self.query_output()?;
        unsafe { index.select(&query, cache) }
    }
    /// Reuse the nearest learned selection, checking it against this cache batch.
    /// # Safety
    /// Same completed-producer and matching-sink contract as attention_cached_ffn.
    pub unsafe fn attention_indexed_ffn(
        &mut self,
        sink: Ds41rtDeviceBuffer,
        cache: &crate::v41_backbone_cache::CacheAttention<'_>,
        index: &crate::v41_index_lane::IndexLane<'_, '_>,
    ) -> Result<LaneFfn<'_, 'w, 'a>> {
        let selection = if self.layer >= 2 { Some(index.output(self.layer, cache)?) } else { None };
        unsafe { self.attention_cached_ffn(sink, cache, selection.as_ref()) }
    }
    /// Execute attention from one cache-bank batch. The bank view pins all
    /// request leases and proposal buffers through the completed attention call.
    /// # Safety
    /// Sink and selection match this layer; producers have drained and no
    /// external writes race the cache/query buffers (as for attention_ffn).
    pub unsafe fn attention_cached_ffn(
        &mut self,
        sink: Ds41rtDeviceBuffer,
        cache: &crate::v41_backbone_cache::CacheAttention<'_>,
        selection: Option<&IndexSelectionOutput<'_>>,
    ) -> Result<LaneFfn<'_, 'w, 'a>> {
        let requests = cache.attention_requests();
        unsafe { self.attention_ffn(sink, &requests, selection) }
    }
    /// # Safety
    /// Result is the completed shared plus routed reduction for the exact FFN
    /// binding and row order published by attention_ffn.
    pub unsafe fn finish_ffn(
        &mut self,
        binding: QueryBinding,
        result: Ds41rtDeviceBuffer,
    ) -> Result<BlockOutput<'_>> {
        self.enter(Phase::SharedReady)?;
        let output = unsafe { self.block.finish_ffn(binding, result)? };
        self.phase = Phase::Complete;
        Ok(output)
    }
    pub fn output(&self) -> Result<BlockOutput<'_>> {
        ensure!(
            self.phase == Phase::Complete,
            "backbone lane output unavailable"
        );
        self.block.output()
    }
    pub fn set_route_capture(&mut self, enabled: bool) {
        self.capture_routes = enabled;
        if enabled {
            self.query.enable_small_graph_shapes();
            self.projection.enable_small_graph_shapes();
            self.shared.enable_small_graph_shapes();
            self.router.enable_small_graph_shapes();
            self.route_capture.resize_with(40, Vec::new);
            for rows in &mut self.route_capture { rows.clear(); }
        }
    }
    pub fn captured_routes(&self) -> &[Vec<[u32; 6]>] { &self.route_capture }
    /// Opt-in diagnostic at a completed layer boundary; never used by normal serving.
    pub fn trace_output(&self, directory: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let output = self.output()?;
        for (name, buffer) in [("residual", output.residual), ("pre", output.pre)] {
            let path = directory.join(format!("layer{}-{name}.bin", output.layer));
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
            let mut bytes = vec![0; buffer.bytes];
            self.weights.library.copy_d2h(&mut bytes, buffer)?;
            file.write_all(&bytes)?;
        }
        Ok(())
    }
    pub fn advance(&mut self) -> Result<()> {
        self.enter(Phase::Complete)?;
        ensure!(self.layer < 39, "backbone lane is at final layer");
        let next = &self.weights.layers[self.layer + 1];
        self.query.rebind(&next.query)?;
        self.projection.rebind(&next.projection)?;
        self.shared.rebind(&next.shared)?;
        self.router.rebind(&next.router)?;
        self.block.advance(&next.hc)?;
        self.layer += 1;
        self.phase = Phase::Prepared;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds41rt_loader::{read_official_v41_catalog, OFFICIAL_V41_MODEL_ID};

    #[test]
    fn lane_ffn_cancellation_and_failure_do_not_publish_completion() {
        use std::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        let mut context = Context::from_waker(Waker::noop());
        let mut phase = Phase::Ffn;
        // An unpolled future has dispatched nothing and leaves work available.
        drop(complete_ffn(
            &mut phase,
            std::future::pending::<Result<()>>(),
        ));
        assert_eq!(phase, Phase::Ffn);
        let mut work = Box::pin(complete_ffn(
            &mut phase,
            std::future::pending::<Result<()>>(),
        ));
        assert!(work.as_mut().poll(&mut context).is_pending());
        drop(work);
        assert_eq!(phase, Phase::Invalid);
        let touched = std::cell::Cell::new(false);
        let mut work = Box::pin(complete_ffn(&mut phase, async {
            touched.set(true);
            Ok(())
        }));
        assert!(matches!(
            work.as_mut().poll(&mut context),
            Poll::Ready(Err(_))
        ));
        drop(work);
        assert!(!touched.get());
        phase = Phase::Ffn;
        let mut work = Box::pin(complete_ffn(&mut phase, async {
            anyhow::bail!("shared/reduction failure")
        }));
        let result: Poll<Result<()>> = work.as_mut().poll(&mut context);
        assert!(matches!(result, Poll::Ready(Err(_))));
        drop(work);
        assert_eq!(phase, Phase::Invalid);
        phase = Phase::Ffn;
        let mut work = Box::pin(complete_ffn(&mut phase, async { Ok(17) }));
        assert!(matches!(
            work.as_mut().poll(&mut context),
            Poll::Ready(Ok(17))
        ));
        drop(work);
        assert_eq!(phase, Phase::SharedReady);
    }

    #[test]
    fn official_lane_budget_rejects_before_device_allocation() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_LANE_PLAN_LIBRARY") else {
            eprintln!("skip lane planning test: DS41RT_LANE_PLAN_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_LANE_PLAN_MODEL")
            .ok_or_else(|| anyhow::anyhow!("DS41RT_LANE_PLAN_MODEL required"))?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog =
            read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, std::path::Path::new(&model))?;
        // Checkpoint payload extents plus the native AOT packed-scale metadata.
        let weight_bytes = BackboneLaneWeights::device_bytes(&library, &catalog)?;
        assert_eq!(weight_bytes, 6_896_423_360);
        let rejected = BackboneLaneWeights::load(&library, &catalog, weight_bytes - 1, 1024 * 1024);
        assert!(rejected
            .err()
            .unwrap()
            .to_string()
            .contains("weights exceed budget"));
        let empty = BackboneLaneWeights {
            library: &library,
            layers: vec![],
        };
        for (capacity, expected) in [(1, 2_301_989usize), (80, 80_607_196), (4096, 2_985_263_116)] {
            let groups = BackboneLane::workspace_bytes(&library, capacity)?;
            let total: usize = groups.iter().sum();
            assert_eq!(total, expected);
            // Both paths must reject without attempting a CUDA allocation. The
            // CPU-only qualification container exposes no GPU devices.
            let rejected = BackboneLane::new(&empty, capacity, total - 1);
            assert!(rejected
                .err()
                .unwrap()
                .to_string()
                .contains("lane exceeds budget"));
            let rejected = BackboneLane::new(&empty, capacity, total);
            assert!(rejected
                .err()
                .unwrap()
                .to_string()
                .contains("all 40 layers"));
            eprintln!(
                "PASS capacity={capacity} groups={groups:?} workspace_bytes={total} budget_guard"
            );
        }
        for capacity in [0, 4097, u32::MAX] {
            assert!(BackboneLane::workspace_bytes(&library, capacity).is_err());
        }
        eprintln!("PASS official forty-layer weight budget={weight_bytes}; CPU-only planning and allocation guards");
        Ok(())
    }
}
