//! Route forecasts use only caller-confirmed accepted target inputs.
use std::collections::{BTreeMap, VecDeque};
type ExpertSet = [u64; 6];
type LayerSets = [ExpertSet; 40];
#[derive(Default)]
pub struct DsparkRouteHistory {
    requests: BTreeMap<u64, Vec<VecDeque<[u32; 6]>>>,
}
pub struct DsparkRouteForecast {
    lanes: Vec<usize>,
    prefixes: Vec<Vec<LayerSets>>,
}
// Seven independent 9-bit counters per word. Each lane has at most eight
// requests, each with eight history rows and six routes: even duplicate route IDs
// cannot exceed 384, so addition has no carry between fields. The +15 group
// rounding below also remains below 512.
const COUNT_WORDS: usize = 384usize.div_ceil(7);
type LayerCounts = [[u64; COUNT_WORDS]; 40];
#[cfg(test)]
const COUNT_LOW: u64 = (1 << 0) | (1 << 9) | (1 << 18) | (1 << 27)
    | (1 << 36) | (1 << 45) | (1 << 54);

pub struct DsparkWorkForecast {
    lanes: Vec<usize>,
    rows: Vec<Vec<[[u32; 6]; 40]>>,
}

#[cfg(test)]
fn packed_work(counts: u64) -> (u32, u32) {
    // ceil(count/16) fits five bits (maximum 24 groups). The guard bit
    // isolates zero detection; multiplication sums the seven 9-bit fields.
    let groups = ((counts + COUNT_LOW * 15) >> 4) & (COUNT_LOW * 31);
    let unique = (((groups | (COUNT_LOW << 8)) - COUNT_LOW) & (COUNT_LOW << 8)).count_ones();
    let total = (groups.wrapping_mul(COUNT_LOW) >> 54) & 511;
    (unique, total as u32)
}

pub struct DsparkWorkEvaluator<'a> {
    forecast: &'a DsparkWorkForecast,
    counts: [LayerCounts; 2],
    lengths: Vec<usize>,
    unique: u32,
    groups: u32,
}
impl DsparkWorkForecast {
    /// Start with mandatory anchors. Subsequent calls incrementally apply only
    /// changed suffix rows; every candidate still sees the complete joint cost.
    pub fn evaluator(&self) -> DsparkWorkEvaluator<'_> {
        let mut value = DsparkWorkEvaluator {
            forecast: self, counts: [[[0u64; COUNT_WORDS]; 40]; 2],
            lengths: vec![0; self.rows.len()], unique: 0, groups: 0,
        };
        for request in 0..self.rows.len() { value.adjust::<true>(request, 0); }
        value
    }
    /// Position-first, request-second admission within one lane. Forecasts are
    /// accepted-history estimates, not routes from the unexecuted verifier.
    pub fn select_confidence_prefixes(&self, probabilities: &[&[f64]], minimum: &[usize],
        reused_cutoff: f64, new_cutoff: f64) -> Result<Vec<usize>, &'static str> {
        if self.lanes.iter().any(|lane| *lane != self.lanes[0])
            || probabilities.len() != self.rows.len() || minimum.len() != self.rows.len()
            || !reused_cutoff.is_finite() || !new_cutoff.is_finite()
            || reused_cutoff <= 0. || reused_cutoff > new_cutoff || new_cutoff > 1. {
            return Err("invalid lane-local reuse policy parameters");
        }
        for ((p, &min), rows) in probabilities.iter().zip(minimum).zip(&self.rows) {
            if p.len() + 1 != rows.len() || min > p.len()
                || p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
                return Err("invalid reuse confidence extent");
            }
        }
        let mut lengths = minimum.to_vec();
        let mut cumulative = vec![1.; lengths.len()];
        let mut evaluator = self.evaluator();
        let (mut unique, _) = evaluator.mean_expert_work(&lengths);
        for position in 1..=crate::MAX_DSPARK_PROPOSALS {
            for request in 0..lengths.len() {
                let p = probabilities[request];
                if position > p.len() { continue; }
                cumulative[request] *= p[position - 1];
                if position <= minimum[request] || lengths[request] != position - 1 { continue; }
                lengths[request] += 1;
                let (candidate_unique, _) = evaluator.mean_expert_work(&lengths);
                // One row adds at most six experts per layer. Existing experts
                // reduce the threshold, but even full reuse has positive cost.
                let fraction = ((candidate_unique - unique) / 6.).clamp(0., 1.);
                let cutoff = reused_cutoff + (new_cutoff - reused_cutoff) * fraction;
                if cumulative[request] >= cutoff {
                    unique = candidate_unique;
                } else {
                    lengths[request] -= 1;
                    evaluator.mean_expert_work(&lengths);
                }
            }
        }
        Ok(lengths)
    }
    pub fn mean_expert_work(&self, lengths: &[usize]) -> (f64, f64) {
        self.evaluator().mean_expert_work(lengths)
    }
}
impl DsparkWorkEvaluator<'_> {
    fn adjust<const ADD: bool>(&mut self, request: usize, row: usize) {
        let lane = self.forecast.lanes[request];
        for (layer, routes) in self.forecast.rows[request][row].iter().enumerate() {
            for &expert in routes {
                let word = &mut self.counts[lane][layer][expert as usize / 7];
                let shift = (expert as usize % 7) * 9;
                let count = (*word >> shift) & 511;
                let delta = 1u64 << shift;
                if ADD {
                    self.unique += u32::from(count == 0);
                    self.groups += u32::from(count % 16 == 0);
                    *word += delta;
                } else {
                    debug_assert!(count > 0);
                    self.unique -= u32::from(count == 1);
                    self.groups -= u32::from((count - 1) % 16 == 0);
                    *word -= delta;
                }
            }
        }
    }
    /// Sum lane means for unique experts and 16-route work groups. Requests may
    /// change in any order, including restoring a previous or full candidate.
    pub fn mean_expert_work(&mut self, lengths: &[usize]) -> (f64, f64) {
        assert_eq!(lengths.len(), self.lengths.len());
        for (request, &length) in lengths.iter().enumerate() {
            assert!(length < self.forecast.rows[request].len());
            while self.lengths[request] > length {
                self.adjust::<false>(request, self.lengths[request]);
                self.lengths[request] -= 1;
            }
            while self.lengths[request] < length {
                self.lengths[request] += 1;
                self.adjust::<true>(request, self.lengths[request]);
            }
        }
        (self.unique as f64 / 40., self.groups as f64 / 40.)
    }
}

impl DsparkRouteHistory {
    /// Optional richer forecast; the current unique-only serving policy does
    /// not allocate or evaluate these counters. Preserve the scheduler's eight
    /// requests per lane bound so packed additions cannot overflow a field.
    pub fn forecast_work(&self, requests: &[(u64, usize, usize)]) -> Option<DsparkWorkForecast> {
        if requests.is_empty() || requests.len() > 16 { return None; }
        let mut lanes = [0usize; 2];
        let mut rows = Vec::with_capacity(requests.len());
        for (i, &(id, lane, maximum)) in requests.iter().enumerate() {
            if lane > 1 || maximum > crate::MAX_DSPARK_PROPOSALS || requests[..i].iter().any(|r| r.0 == id) { return None; }
            lanes[lane] += 1;
            if lanes[lane] > 8 { return None; }
            let history = self.requests.get(&id)?;
            if history.iter().any(|h| h.len() < maximum + 1) { return None; }
            rows.push((0..=maximum).map(|n| std::array::from_fn(|layer|
                history[layer][history[layer].len() - 1 - n])).collect());
        }
        Some(DsparkWorkForecast { lanes: requests.iter().map(|r| r.1).collect(), rows })
    }
    pub fn release(&mut self, request: u64) { self.requests.remove(&request); }
    pub fn observe_accepted(&mut self, request: u64, layer: usize, routes: &[[u32; 6]]) -> Result<(), &'static str> {
        if layer >= 40 || routes.len() > crate::MAX_DSPARK_PROPOSALS + 1 || routes.iter().flatten().any(|&e| e >= 384) {
            return Err("invalid accepted route observation");
        }
        let history = self.requests.entry(request).or_insert_with(|| (0..40).map(|_| VecDeque::with_capacity(crate::MAX_DSPARK_PROPOSALS + 1)).collect());
        for &row in routes {
            if history[layer].len() == crate::MAX_DSPARK_PROPOSALS + 1 { history[layer].pop_front(); }
            history[layer].push_back(row);
        }
        Ok(())
    }
    /// Inputs are (request identity, lane, maximum draft length). A forecast
    /// requires enough accepted history for every full prefix; otherwise callers
    /// retain the fixed policy until history is available.
    pub fn forecast(&self, requests: &[(u64, usize, usize)]) -> Option<DsparkRouteForecast> {
        if requests.is_empty() || requests.len() > 16 { return None; }
        let mut prefixes = Vec::with_capacity(requests.len());
        for &(id, lane, maximum) in requests {
            if lane > 1 || maximum > crate::MAX_DSPARK_PROPOSALS { return None; }
            let history = self.requests.get(&id)?;
            if history.iter().any(|h| h.len() < maximum + 1) { return None; }
            let mut curve = Vec::with_capacity(maximum + 1);
            let mut sets = [[0u64; 6]; 40];
            for n in 0..=maximum {
                for layer in 0..40 {
                    for &e in &history[layer][history[layer].len() - 1 - n] {
                        sets[layer][e as usize / 64] |= 1 << (e % 64);
                    }
                }
                curve.push(sets);
            }
            prefixes.push(curve);
        }
        Some(DsparkRouteForecast { lanes: requests.iter().map(|r| r.1).collect(), prefixes })
    }
}
impl DsparkRouteForecast {
    pub fn lane_count(&self) -> usize {
        usize::from(self.lanes.contains(&0)) + usize::from(self.lanes.contains(&1))
    }
    /// Sum of each lane's mean unique experts over all forty layers. Shared
    /// histories deduplicate within a lane; separately executing lanes each pay.
    pub fn mean_unique_experts(&self, lengths: &[usize]) -> f64 {
        self.unique_experts_by_layer(lengths).iter().sum::<usize>() as f64 / 40.0
    }
    /// Keep layer identity so callers can price the configured expert backend.
    /// Requests share experts within a lane; independent lanes each pay for them.
    pub fn unique_experts_by_layer(&self, lengths: &[usize]) -> [usize; 40] {
        assert_eq!(lengths.len(), self.prefixes.len());
        let mut sets = [[[0u64; 6]; 40]; 2];
        for ((&lane, curve), &length) in self.lanes.iter().zip(&self.prefixes).zip(lengths) {
            for (output, input) in sets[lane].iter_mut().zip(&curve[length]) {
                for (out, &value) in output.iter_mut().zip(input) { *out |= value; }
            }
        }
        std::array::from_fn(|layer| sets.iter().map(|lane|
            lane[layer].iter().map(|w| w.count_ones() as usize).sum::<usize>()).sum())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expert_forecast_preserves_layer_costs_and_lane_independence() {
        let mut history = DsparkRouteHistory::default();
        for id in [1, 2] {
            for layer in 0..40 {
                let older = if layer < 20 { [0; 6] } else { [1; 6] };
                history.observe_accepted(id, layer, &[older, [0; 6]]).unwrap();
            }
        }
        let shared = history.forecast(&[(1, 0, 1), (2, 0, 1)]).unwrap();
        let costs = shared.unique_experts_by_layer(&[1, 1]);
        assert_eq!(&costs[..20], &[1; 20]);
        assert_eq!(&costs[20..], &[2; 20]);
        assert_eq!(shared.unique_experts_by_layer(&[0, 0]), [1; 40]);
        let separate = history.forecast(&[(1, 0, 1), (2, 1, 1)]).unwrap();
        assert_eq!(separate.unique_experts_by_layer(&[1, 1]), costs.map(|n| n * 2));
        assert_eq!(shared.mean_unique_experts(&[1, 1]), 1.5);
    }
    #[test]
    fn seven_draft_history_stays_bounded_and_packed_counts_do_not_carry() {
        let mut history = DsparkRouteHistory::default();
        for id in 0..8 {
            for layer in 0..40 {
                history.observe_accepted(id, layer, &[[383; 6]; 8]).unwrap();
            }
        }
        let requests: Vec<_> = (0..8).map(|id| (id, 0, 7)).collect();
        assert_eq!(history.forecast(&requests).unwrap().mean_unique_experts(&[7; 8]), 1.);
        let forecast = history.forecast_work(&requests).unwrap();
        let mut evaluator = forecast.evaluator();
        assert_eq!(evaluator.mean_expert_work(&[7; 8]), (1., 24.));
        assert_eq!(evaluator.mean_expert_work(&[0; 8]), (1., 3.));
        assert_eq!(evaluator.mean_expert_work(&[7; 8]), (1., 24.));
        for layer in 0..40 {
            history.observe_accepted(0, layer, &[[0; 6]; 8]).unwrap();
        }
        assert_eq!(history.forecast(&[(0, 0, 7)]).unwrap().mean_unique_experts(&[7]), 1.);
        assert!(history.forecast(&[(0, 0, 8)]).is_none());
    }
    #[test]
    #[ignore = "explicit optimized-build host cost probe"]
    fn work_forecast_joint_search_cost_probe() {
        use std::{hint::black_box, time::Instant};
        let mut history = DsparkRouteHistory::default();
        for id in 0..16u64 {
            for layer in 0..40 {
                let rows: Vec<_> = (0..6).map(|row| std::array::from_fn(|route|
                    ((id * 17 + layer as u64 * 11 + row * 3 + route as u64) % 384) as u32)).collect();
                history.observe_accepted(id, layer, &rows).unwrap();
            }
        }
        let requests: Vec<_> = (0..16).map(|id| (id, id as usize / 8, 5)).collect();
        let probabilities = [&[0.9, 0.8, 0.7, 0.6, 0.5][..]; 16];
        let mut timings = Vec::new();
        let mut calls = 0;
        for _ in 0..25 {
            let start = Instant::now();
            let forecast = history.forecast_work(black_box(&requests)).unwrap();
            let mut evaluator = forecast.evaluator();
            let result = crate::select_dspark_prefixes_bounded(&probabilities, &[1; 16], |lengths| {
                let (unique, groups) = evaluator.mean_expert_work(black_box(lengths));
                20_000. + 250. * (16 + lengths.iter().sum::<usize>()) as f64
                    + 800. * unique + 4000. * (groups - unique)
            }).unwrap();
            calls = result.evaluated_shapes;
            black_box(result);
            timings.push(start.elapsed().as_micros());
        }
        timings.sort_unstable();
        eprintln!("C16 forecast+joint search: median_us={} max_us={} cost_calls={}", timings[12], timings[24], calls);
    }

    #[test]
    fn packed_group_counts_match_scalar_boundaries_and_mixed_fields() {
        for slot in 0..7 {
            for count in 0u64..=288 {
                assert_eq!(packed_work(count << (slot * 9)),
                    (u32::from(count != 0), count.div_ceil(16) as u32));
            }
        }
        let mut seed = 4197u64;
        for _ in 0..4096 {
            let (mut packed, mut unique, mut groups) = (0u64, 0u32, 0u32);
            for slot in 0..7 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let count = (seed >> 32) % 289;
                packed |= count << (slot * 9);
                unique += u32::from(count != 0);
                groups += count.div_ceil(16) as u32;
            }
            assert_eq!(packed_work(packed), (unique, groups));
        }
    }

    #[test]
    fn work_forecast_preserves_lane_groups_bounds_and_accepted_history() {
        let mut h = DsparkRouteHistory::default();
        for id in 0..16 {
            for layer in 0..40 {
                h.observe_accepted(id, layer, &[[383; 6]; 6]).unwrap();
                h.observe_accepted(id, layer, &[[0, 1, 2, 3, 4, 5]; 6]).unwrap();
            }
        }
        let requests: Vec<_> = (0..16).map(|id| (id, id as usize / 8, 5)).collect();
        let forecast = h.forecast_work(&requests).unwrap();
        assert_eq!(forecast.mean_expert_work(&[1; 16]), (12., 12.));
        assert_eq!(forecast.mean_expert_work(&[2; 16]), (12., 24.));
        assert_eq!(forecast.mean_expert_work(&[5; 16]), (12., 36.));
        for id in 0..16 {
            for layer in 0..40 { h.observe_accepted(id, layer, &[[383; 6]; 6]).unwrap(); }
        }
        assert_eq!(h.forecast_work(&requests).unwrap().mean_expert_work(&[5; 16]), (2., 36.));
        // Forecasts are immutable snapshots even as accepted history changes.
        assert_eq!(forecast.mean_expert_work(&[5; 16]), (12., 36.));
        let crowded: Vec<_> = (0..9).map(|id| (id, 0, 5)).collect();
        assert!(h.forecast_work(&crowded).is_none());
        assert!(h.forecast_work(&[(0, 0, 5), (0, 1, 5)]).is_none());
        h.release(0);
        assert!(h.forecast_work(&requests).is_none());
    }

    #[test]
    fn work_forecast_matches_scalar_route_multiplicities() {
        let mut h = DsparkRouteHistory::default();
        let mut histories = Vec::new();
        for id in 0..16u64 {
            let mut history = [[[0u32; 6]; 6]; 40];
            for layer in 0..40 {
                for row in 0..6 {
                    for route in 0..6 {
                        history[layer][row][route] = if route == 0 { 383 }
                            else { ((id as usize * 37 + layer * 13 + row * 5 + route) % 384) as u32 };
                    }
                }
                h.observe_accepted(id, layer, &history[layer]).unwrap();
            }
            histories.push(history);
        }
        let requests: Vec<_> = (0..16).map(|id| (id, id as usize % 2, 5)).collect();
        let forecast = h.forecast_work(&requests).unwrap();
        let mut evaluator = forecast.evaluator();
        for cycle in 0..12 {
            let lengths: Vec<_> = (0..16).map(|i| (i + cycle) % 6).collect();
            let (mut unique, mut groups) = (0usize, 0usize);
            for lane in 0..2 {
                for layer in 0..40 {
                    let mut counts = [0usize; 384];
                    for id in (lane..16).step_by(2) {
                        for row in &histories[id][layer][5-lengths[id]..] {
                            for &expert in row { counts[expert as usize] += 1; }
                        }
                    }
                    unique += counts.iter().filter(|&&n| n != 0).count();
                    groups += counts.iter().map(|n| n.div_ceil(16)).sum::<usize>();
                }
            }
            assert_eq!(forecast.mean_expert_work(&lengths), (unique as f64 / 40., groups as f64 / 40.));
            assert_eq!(evaluator.mean_expert_work(&lengths), (unique as f64 / 40., groups as f64 / 40.));
        }
    }

    #[test]
    fn accepted_history_is_bounded_and_shared_routes_deduplicate_per_lane() {
        let mut h = DsparkRouteHistory::default();
        for id in [1, 2] {
            for layer in 0..40 {
                h.observe_accepted(id, layer, &[[383; 6]]).unwrap();
                h.observe_accepted(id, layer, &[[0,1,2,3,4,5]; 6]).unwrap();
            }
        }
        let f = h.forecast(&[(1,0,5), (2,0,5)]).unwrap();
        assert_eq!(f.mean_unique_experts(&[5,5]), 6.);
        assert_eq!(f.mean_unique_experts(&[0,0]), 6.);
        assert_eq!(h.forecast(&[(1,0,5),(2,1,5)]).unwrap().mean_unique_experts(&[5,5]), 12.);
        h.release(1);
        assert!(h.forecast(&[(1,0,5)]).is_none());
    }
    #[test]
    fn missing_layers_or_insufficient_history_cannot_predict() {
        let mut h = DsparkRouteHistory::default();
        h.observe_accepted(1, 0, &[[0;6];6]).unwrap();
        assert!(h.forecast(&[(1,0,5)]).is_none());
        assert!(h.observe_accepted(1, 40, &[[0;6]]).is_err());
        assert!(h.observe_accepted(1, 0, &[[384;6]]).is_err());
    }
}

#[cfg(test)]
mod incremental_reuse_tests {
    use super::*;
    fn forecast(second_lane: usize) -> DsparkWorkForecast {
        let mut h = DsparkRouteHistory::default();
        // Reversed history predicts anchor=0..5, first draft=6..11,
        // second draft=12..17, identically for both requests.
        for id in 0..2 {
            for layer in 0..40 {
                h.observe_accepted(id, layer, &[[12,13,14,15,16,17],
                    [6,7,8,9,10,11],[0,1,2,3,4,5]]).unwrap();
            }
        }
        h.forecast_work(&[(0,0,2),(1,second_lane,2)]).unwrap()
    }
    #[test]
    fn newly_admitted_token_immediately_discounts_later_request() {
        let f = forecast(0);
        // Request zero buys six new experts at position two; request one then
        // reuses those experts and passes despite lower cumulative confidence.
        assert_eq!(f.select_confidence_prefixes(&[&[1.,0.9], &[1.,0.4]], &[1,1], 0.2,0.8), Ok(vec![2,2]));
        // A rejected token contributes no experts to the following request.
        assert_eq!(f.select_confidence_prefixes(&[&[1.,0.4], &[1.,0.4]], &[1,1], 0.2,0.8), Ok(vec![1,1]));
    }
    #[test]
    fn cross_lane_and_invalid_cutoffs_are_rejected() {
        assert!(forecast(1).select_confidence_prefixes(&[&[1.,1.][..];2], &[1,1],0.2,0.8).is_err());
        assert!(forecast(0).select_confidence_prefixes(&[&[1.,1.][..];2], &[1,1],0.,0.8).is_err());
        assert!(forecast(0).select_confidence_prefixes(&[&[1.,1.][..];2], &[1,1],0.9,0.8).is_err());
    }
}
