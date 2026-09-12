//! Bounded joint suffix selection. Cost calibration and route prediction belong
//! to the caller; this module performs no device work and claims no global optimum.
#[derive(Debug, Clone, PartialEq)]
pub struct DsparkPrefixSelection {
    /// Draft tokens retained per request, excluding the mandatory anchor.
    pub lengths: Vec<usize>,
    pub expected_tokens: f64,
    pub cost_us: f64,
    pub evaluated_shapes: usize,
}

/// Select among the full batch and a greedy suffix-removal trajectory.
/// Confidence at j means acceptance conditional on all earlier proposals matching.
/// The cost includes draft generation, both verifier lanes, and serving overhead.
/// Every candidate is rescored against the current *joint* prefix lengths.
/// Continue even through a temporary rate loss: crossing a kernel bucket can
/// require several removals. Return the best visited shape, never a predicted
/// regression from the original full prefixes. At most 1 + 5*16*16 cost calls.
pub fn select_dspark_prefixes(
    confidence: &[&[f64]],
    mut cost_us: impl FnMut(&[usize]) -> f64,
) -> Result<DsparkPrefixSelection, &'static str> {
    if confidence.is_empty() || confidence.len() > 16 {
        return Err("policy requires one to sixteen requests");
    }
    let mut expected = Vec::with_capacity(confidence.len());
    for probabilities in confidence {
        if probabilities.len() > 5 || probabilities.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
            return Err("invalid conditional draft confidence");
        }
        let mut row = vec![1.0]; // Correction/bonus from the mandatory anchor.
        let mut product = 1.0;
        for &p in *probabilities {
            product *= p;
            row.push(row.last().unwrap() + product);
        }
        expected.push(row);
    }
    let mut lengths: Vec<_> = confidence.iter().map(|p| p.len()).collect();
    let mut tokens: f64 = expected.iter().zip(&lengths).map(|(e, &n)| e[n]).sum();
    let valid_cost = |value: f64| if value.is_finite() && value > 0.0 { Ok(value) }
        else { Err("invalid predicted verification cost") };
    let cost = valid_cost(cost_us(&lengths))?;
    let mut best = DsparkPrefixSelection { lengths: lengths.clone(), expected_tokens: tokens, cost_us: cost, evaluated_shapes: 1 };
    let mut evaluations = 1;
    while lengths.iter().any(|&n| n != 0) {
        let mut choice: Option<(usize, f64, f64)> = None;
        for i in 0..lengths.len() {
            let n = lengths[i];
            if n == 0 { continue; }
            let candidate_tokens = tokens - expected[i][n] + expected[i][n - 1];
            lengths[i] -= 1;
            let candidate_cost = valid_cost(cost_us(&lengths))?;
            lengths[i] += 1;
            evaluations += 1;
            if choice.is_none_or(|(_, t, c)| candidate_tokens / candidate_cost > t / c) {
                choice = Some((i, candidate_tokens, candidate_cost));
            }
        }
        let (i, candidate_tokens, candidate_cost) = choice.unwrap();
        lengths[i] -= 1;
        tokens = candidate_tokens;
        if tokens / candidate_cost > best.expected_tokens / best.cost_us {
            best.lengths.clone_from(&lengths);
            best.expected_tokens = tokens;
            best.cost_us = candidate_cost;
        }
    }
    best.evaluated_shapes = evaluations;
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn confidence_is_conditional_and_anchor_is_always_retained() {
        let p = select_dspark_prefixes(&[&[0.8, 0.5]], |_| 10.).unwrap();
        assert_eq!(p.lengths, [2]);
        assert!((p.expected_tokens - 2.2).abs() < 1e-12);
        let p = select_dspark_prefixes(&[&[0., 0.], &[]], |n| 1. + n.iter().sum::<usize>() as f64).unwrap();
        assert_eq!(p.lengths, [0, 0]);
        assert_eq!(p.expected_tokens, 2.);
    }
    #[test]
    fn crosses_joint_bucket_even_when_first_trim_loses_rate() {
        let p = select_dspark_prefixes(&[&[0.1], &[0.1]], |n| if n.iter().sum::<usize>() == 0 { 1. } else { 10. }).unwrap();
        assert_eq!(p.lengths, [0, 0]);
        assert_eq!(p.cost_us, 1.);
        assert_eq!(p.evaluated_shapes, 4);
    }
    #[test]
    fn preserves_full_prefix_on_ties_and_bounds_work() {
        let probabilities = [1.; 5];
        let inputs = [&probabilities[..]; 16];
        let p = select_dspark_prefixes(&inputs, |n| (16 + n.iter().sum::<usize>()) as f64).unwrap();
        assert_eq!(p.lengths, vec![5; 16]);
        assert!(p.evaluated_shapes <= 1281);
    }
    #[test]
    fn rejects_invalid_predictions() {
        assert!(select_dspark_prefixes(&[], |_| 1.).is_err());
        assert!(select_dspark_prefixes(&[&[f64::NAN]], |_| 1.).is_err());
        assert!(select_dspark_prefixes(&[&[1.1]], |_| 1.).is_err());
        assert!(select_dspark_prefixes(&[&[0.5]], |_| 0.).is_err());
        assert!(select_dspark_prefixes(&[&[0.5]], |n| if n[0] == 1 { 1. } else { f64::INFINITY }).is_err());
    }
}
