mod real_full;
mod real_slice;
mod synthetic_ds4;
mod tiny;

pub(crate) use real_full::{
    real_ds4_full_completion, resolve_real_full_prompt_token_ids,
    try_real_ds4_full_streaming_response,
};
pub(crate) use real_slice::real_ds4_slice_completion;
pub(crate) use synthetic_ds4::synthetic_ds4_layer_completion;
pub(crate) use tiny::tiny_backend_completion;
