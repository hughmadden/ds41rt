pub const DEFAULT_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4-Flash-0731";
pub const DS4_FLASH_MODEL_ID: &str = DEFAULT_MODEL_ID;
pub const DS4_PRO_PREVIEW_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4-Pro-DSpark";
pub const SUPPORTED_MODEL_IDS: [&str; 2] = [DS4_FLASH_MODEL_ID, DS4_PRO_PREVIEW_MODEL_ID];
pub const COORDINATOR_HOST: &str = "coordinator";
pub const EXPERT_HOSTS: [&str; 4] = ["spark-0", "spark-1", "spark-2", "spark-3"];
pub const DS4_EXPERT_TP_WORLD_SIZE: usize = EXPERT_HOSTS.len();

pub const DS4_FLASH_HIDDEN_SIZE: usize = 4096;
pub const DS4_FLASH_HIDDEN_BF16_BYTES: usize = DS4_FLASH_HIDDEN_SIZE * 2;
pub const DS4_FLASH_NUM_HIDDEN_LAYERS: usize = 43;
pub const DS4_FLASH_ROUTED_EXPERTS: usize = 256;
pub const DS4_FLASH_TOP_K: usize = 6;
pub const DS4_FLASH_MOE_INTERMEDIATE_SIZE: usize = 2048;
pub const DS4_FLASH_Q_LORA_RANK: usize = 1024;
pub const DS4_FLASH_DSPARK_MARKOV_RANK: usize = 256;
pub const DS4_FLASH_COMPRESS_RATIOS: [usize; 46] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
];

pub const DS4_PRO_HIDDEN_SIZE: usize = 7168;
pub const DS4_PRO_HIDDEN_BF16_BYTES: usize = DS4_PRO_HIDDEN_SIZE * 2;
pub const DS4_PRO_NUM_HIDDEN_LAYERS: usize = 61;
pub const DS4_PRO_ROUTED_EXPERTS: usize = 384;
pub const DS4_PRO_TOP_K: usize = 6;
pub const DS4_PRO_MOE_INTERMEDIATE_SIZE: usize = 3072;
pub const DS4_PRO_Q_LORA_RANK: usize = 1536;
pub const DS4_PRO_DSPARK_MARKOV_RANK: usize = 512;
pub const DS4_PRO_COMPRESS_RATIOS: [usize; 64] = [
    128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
];

pub const DS4_NUM_HASH_LAYERS: usize = 3;
pub const DS4_NUM_SHARED_EXPERTS: usize = 1;
pub const DS4_HEAD_DIM: usize = 512;
pub const DS4_QK_ROPE_HEAD_DIM: usize = 64;
pub const DS4_KV_HEADS: usize = 1;
pub const DS4_INDEX_HEAD_DIM: usize = 128;
pub const DS4_INDEX_HEADS: usize = 64;
pub const DS4_SLIDING_WINDOW: usize = 128;
pub const DS4_HC_MULT: usize = 4;
pub const DS4_HC_SINKHORN_ITERS: usize = 20;
pub const DS4_HC_EPS: f32 = 1.0e-6;
pub const DS4_O_LORA_RANK: usize = 1024;
pub const DS4_DSPARK_BLOCK_SIZE: usize = 5;
pub const DS4_DSPARK_NOISE_TOKEN_ID: usize = 128_799;
pub const DS4_ROPE_THETA: f32 = 10_000.0;
pub const DS4_COMPRESS_ROPE_THETA: f32 = 160_000.0;
pub const DS4_ROPE_SCALING_FACTOR: f32 = 16.0;
pub const DS4_ORIGINAL_MAX_POSITION_EMBEDDINGS: usize = 65_536;
pub const DS4_ROPE_BETA_FAST: f32 = 32.0;
pub const DS4_ROPE_BETA_SLOW: f32 = 1.0;

// The imported GLM execution path is retained temporarily as a migration
// oracle. New DS4 code must use ModelFacts or DS4_* constants instead.
pub const GLM52_HIDDEN_SIZE: usize = 6144;
pub const GLM52_NUM_HIDDEN_LAYERS: usize = 78;
pub const GLM52_NUM_MTP_LAYERS: usize = 1;
pub const GLM52_MTP_LAYER_ID: usize = GLM52_NUM_HIDDEN_LAYERS;
pub const GLM52_TOTAL_LAYERS_WITH_MTP: usize = GLM52_NUM_HIDDEN_LAYERS + GLM52_NUM_MTP_LAYERS;
pub const GLM52_FIRST_K_DENSE_REPLACE: usize = 3;
pub const GLM52_ROUTED_EXPERTS: usize = 256;
pub const GLM52_TOP_K: usize = 8;
pub const GLM52_ROUTED_SCALING_FACTOR: f32 = 2.5;
pub const GLM52_HIDDEN_BF16_BYTES: usize = GLM52_HIDDEN_SIZE * 2;
pub const GLM52_MLA_KV_LORA_RANK: usize = 512;
pub const GLM52_MLA_QK_ROPE_HEAD_DIM: usize = 64;
pub const GLM52_MLA_ROPE_THETA: f32 = 8_000_000.0;
pub const GLM52_MLA_FP8_DS_SCALE_BYTES_PER_TOKEN: usize = 16;
pub const GLM52_MLA_FP8_DS_BYTES_PER_TOKEN: usize = GLM52_MLA_KV_LORA_RANK
    + GLM52_MLA_FP8_DS_SCALE_BYTES_PER_TOKEN
    + GLM52_MLA_QK_ROPE_HEAD_DIM * 2;
pub const GLM52_MLA_MXFP4_BLOCK_SIZE: usize = 16;
pub const GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN: usize = GLM52_MLA_KV_LORA_RANK / 2;
pub const GLM52_MLA_MXFP4_SCALE_BYTES_PER_TOKEN: usize =
    GLM52_MLA_KV_LORA_RANK / GLM52_MLA_MXFP4_BLOCK_SIZE;
pub const GLM52_MLA_MXFP4_PADDING_BYTES_PER_TOKEN: usize = 16;
pub const GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN: usize = GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN
    + GLM52_MLA_MXFP4_SCALE_BYTES_PER_TOKEN
    + GLM52_MLA_MXFP4_PADDING_BYTES_PER_TOKEN
    + GLM52_MLA_QK_ROPE_HEAD_DIM * 2;
pub const GLM52_DSA_INDEXER_LAYERS: usize = 21;
pub const GLM52_DSA_INDEXER_LAYER_IDS: [usize; GLM52_DSA_INDEXER_LAYERS] = [
    0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
];
pub const GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP: [usize;
    GLM52_DSA_INDEXER_LAYERS + GLM52_NUM_MTP_LAYERS] = [
    0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78,
];
pub const GLM52_DSA_INDEX_HEAD_DIM: usize = 128;
pub const GLM52_COMPRESSED_MAIN_MLA_BF16_BYTES_PER_TOKEN: usize =
    GLM52_NUM_HIDDEN_LAYERS * (GLM52_MLA_KV_LORA_RANK + GLM52_MLA_QK_ROPE_HEAD_DIM) * 2;
pub const GLM52_COMPRESSED_DSA_BF16_BYTES_PER_TOKEN: usize =
    GLM52_DSA_INDEXER_LAYERS * GLM52_DSA_INDEX_HEAD_DIM * 2;
pub const GLM52_COMPRESSED_KV_BF16_BYTES_PER_TOKEN: usize =
    GLM52_COMPRESSED_MAIN_MLA_BF16_BYTES_PER_TOKEN + GLM52_COMPRESSED_DSA_BF16_BYTES_PER_TOKEN;
pub const GLM52_EXPANDED_DEBUG_KV_BF16_BYTES_PER_TOKEN: usize =
    GLM52_NUM_HIDDEN_LAYERS * 2 * GLM52_HIDDEN_SIZE * 2;
