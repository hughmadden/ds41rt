//! V4.1 model limits, shared by HTTP conversion and tokenized admission.
use anyhow::{ensure, Result};

pub const MAX_CONTEXT_TOKENS: u32 = 1_048_576;
pub const MAX_OUTPUT_TOKENS: u32 = 393_216;

#[derive(Clone, Copy, Debug)]
pub struct NativeLimits {
    context: u32,
    output: u32,
}

impl Default for NativeLimits {
    fn default() -> Self {
        Self {
            context: MAX_CONTEXT_TOKENS,
            output: MAX_OUTPUT_TOKENS,
        }
    }
}

impl NativeLimits {
    pub fn new(context: u32, output: u32) -> Result<Self> {
        ensure!(
            (1..=MAX_CONTEXT_TOKENS).contains(&context),
            "context limit must be 1..={MAX_CONTEXT_TOKENS}"
        );
        ensure!(
            (1..=MAX_OUTPUT_TOKENS).contains(&output),
            "output limit must be 1..={MAX_OUTPUT_TOKENS}"
        );
        Ok(Self { context, output })
    }

    pub fn context(self) -> u32 {
        self.context
    }
    pub fn output(self) -> u32 {
        self.output
    }

    pub fn requested_output(self, requested: Option<u32>) -> Result<usize> {
        let requested = requested.unwrap_or(self.output);
        ensure!(requested > 0, "max_tokens must be positive");
        Ok(requested.min(self.output) as usize)
    }

    pub fn output_for_prompt(self, prompt_tokens: usize, requested: usize) -> Result<usize> {
        ensure!(
            prompt_tokens > 0 && prompt_tokens < self.context as usize,
            "native prompt must contain 1..{} tokens to leave room for output",
            self.context
        );
        ensure!(requested > 0, "max_tokens must be positive");
        Ok(requested
            .min(self.output as usize)
            .min(self.context as usize - prompt_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_and_development_limits_bound_output_without_overflow() {
        let limits = NativeLimits::default();
        assert_eq!(limits.requested_output(None).unwrap(), 393_216);
        assert_eq!(limits.requested_output(Some(u32::MAX)).unwrap(), 393_216);
        assert_eq!(limits.output_for_prompt(1_048_575, usize::MAX).unwrap(), 1);
        assert_eq!(limits.output_for_prompt(1, usize::MAX).unwrap(), 393_216);
        for prompt in [0, 1_048_576, usize::MAX] {
            assert!(limits.output_for_prompt(prompt, 1).is_err());
        }
        let small = NativeLimits::new(256, 128).unwrap();
        assert_eq!(small.requested_output(None).unwrap(), 128);
        assert_eq!(small.output_for_prompt(240, 128).unwrap(), 16);
        assert_eq!(small.output_for_prompt(240, 8).unwrap(), 8);
        assert!(small.requested_output(Some(0)).is_err());
        assert!(small.output_for_prompt(1, 0).is_err());
        for (context, output) in [(0, 1), (1, 0), (1_048_577, 1), (1, 393_217)] {
            assert!(NativeLimits::new(context, output).is_err());
        }
    }
}
