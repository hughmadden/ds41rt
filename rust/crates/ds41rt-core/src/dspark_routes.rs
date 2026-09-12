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
impl DsparkRouteHistory {
    pub fn release(&mut self, request: u64) { self.requests.remove(&request); }
    pub fn observe_accepted(&mut self, request: u64, layer: usize, routes: &[[u32; 6]]) -> Result<(), &'static str> {
        if layer >= 40 || routes.len() > 6 || routes.iter().flatten().any(|&e| e >= 384) {
            return Err("invalid accepted route observation");
        }
        let history = self.requests.entry(request).or_insert_with(|| (0..40).map(|_| VecDeque::with_capacity(6)).collect());
        for &row in routes {
            if history[layer].len() == 6 { history[layer].pop_front(); }
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
            if lane > 1 || maximum > 5 { return None; }
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
        assert_eq!(lengths.len(), self.prefixes.len());
        let mut sets = [[[0u64; 6]; 40]; 2];
        for ((&lane, curve), &length) in self.lanes.iter().zip(&self.prefixes).zip(lengths) {
            for (output, input) in sets[lane].iter_mut().zip(&curve[length]) {
                for (out, &value) in output.iter_mut().zip(input) { *out |= value; }
            }
        }
        sets.iter().flatten().flatten().map(|w| w.count_ones() as f64).sum::<f64>() / 40.0
    }
}
#[cfg(test)]
mod tests {
    use super::*;
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
