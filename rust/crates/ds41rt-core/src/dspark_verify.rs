//! Greedy verification of an already emitted anchor and up to five draft tokens.
#[derive(Debug, PartialEq, Eq)]
pub struct GreedyVerification {
    /// Input rows to publish, including the anchor. A correction/bonus token
    /// has not been evaluated as input and must remain the next pending anchor.
    pub accepted_inputs: u32,
    /// Newly generated tokens; excludes the previously emitted anchor.
    pub emitted: Vec<u32>,
    pub eos: bool,
    pub length_limit: bool,
}

/// `inputs` is anchor followed by drafts. `target_next[i]` is the greedy target
/// prediction after input row i, using causal verification of the same prefix.
/// The anchor must already have been emitted, and must not be EOS.
pub fn verify_dspark_greedy(
    inputs: &[u32],
    target_next: &[u32],
    eos: u32,
    remaining: usize,
) -> Result<GreedyVerification, &'static str> {
    if inputs.is_empty() || inputs.len() > 6 || inputs.len() != target_next.len() {
        return Err("verification requires one to six corresponding target rows");
    }
    if remaining == 0 || inputs[0] == eos {
        return Err("completed request cannot verify another anchor");
    }
    if eos >= 129280 || inputs.iter().chain(target_next).any(|&id| id >= 129280) {
        return Err("verification token outside official vocabulary");
    }
    let mut result = GreedyVerification {
        accepted_inputs: 1,
        emitted: Vec::with_capacity(inputs.len()),
        eos: false,
        length_limit: false,
    };
    for (i, &token) in target_next.iter().enumerate() {
        let matched = inputs.get(i + 1) == Some(&token);
        result.emitted.push(token);
        if matched {
            result.accepted_inputs += 1;
        }
        result.eos = token == eos;
        result.length_limit = result.emitted.len() == remaining;
        if !matched || result.eos || result.length_limit {
            break;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_prefix_correction_bonus_and_stopping() {
        let inputs = [10, 11, 12, 13, 14, 15];
        for mismatch in 0..6 {
            let mut target = [11, 12, 13, 14, 15, 16];
            target[mismatch] = 99;
            let result = verify_dspark_greedy(&inputs, &target, 1, 100).unwrap();
            assert_eq!(result.accepted_inputs, mismatch as u32 + 1);
            assert_eq!(result.emitted, target[..=mismatch]);
            assert!(!result.eos && !result.length_limit);
        }
        let result = verify_dspark_greedy(&inputs, &[11, 12, 13, 14, 15, 16], 1, 100).unwrap();
        assert_eq!(result.accepted_inputs, 6);
        assert_eq!(result.emitted, [11, 12, 13, 14, 15, 16]);
        let stopped = verify_dspark_greedy(&inputs, &[11, 12, 13, 14, 15, 16], 1, 2).unwrap();
        assert_eq!(stopped.accepted_inputs, 3);
        assert_eq!(stopped.emitted, [11, 12]);
        assert!(stopped.length_limit);
        for draft in [1, 12] {
            let stopped = verify_dspark_greedy(&[10, draft, 13], &[1, 13, 14], 1, 100).unwrap();
            assert_eq!(stopped.emitted, [1]);
            assert!(stopped.eos);
            assert_eq!(stopped.accepted_inputs, if draft == 1 { 2 } else { 1 });
        }
        assert!(verify_dspark_greedy(&[1], &[10], 1, 1).is_err());
        assert!(verify_dspark_greedy(&[10], &[11], 1, 0).is_err());
        assert!(verify_dspark_greedy(&[10], &[], 1, 1).is_err());
        assert!(verify_dspark_greedy(&[10], &[129280], 1, 1).is_err());
    }
}
