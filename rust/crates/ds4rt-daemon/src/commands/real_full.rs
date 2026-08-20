mod attention;
mod constants;
mod constraint;
mod coordinator_kernels;
mod coverage;
mod dense;
mod dspark;
mod embedding;
mod entry;
mod execution_plan;
mod expert_probe;
mod experts;
mod intermediate_sharding;
mod kv;
mod prefix_cache;
mod preflight;
mod probe_env;
mod rdma_reduction;
mod residency;
mod residual;
mod sampling;
mod scheduler;
mod sparse_mlp;
mod target_sampling;
mod types;

pub(crate) use entry::{load_real_full_serving, run_real_ds4_full_preflight};
#[cfg(test)]
pub(crate) use expert_probe::REAL_NVFP4_PROTOCOL_V2_EXECUTOR;
pub(crate) use expert_probe::{
    real_nvfp4_cuda_reference_kernels_enabled, RealNvfp4ProtocolV2Executor,
    RealNvfp4ResidentPreloadPlan, REAL_NVFP4_CUDA_REFERENCE_KERNELS_ENV,
};
pub(crate) use intermediate_sharding::{
    expert_intermediate_shard_count_from_env, spark_expert_intermediate_shard_from_env,
    spark_expert_owner_reduction_config_from_env, ExpertIntermediateShard,
    EXPERT_INTERMEDIATE_SHARDS_ENV,
};

#[cfg(test)]
mod tests;
