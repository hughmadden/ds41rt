//! Export the exact startup token map for reference qualification.
use anyhow::{Context, Result};
fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: engram_token_map TOKENIZER_JSON")?;
    let tokenizer =
        tokenizers::Tokenizer::from_file(path).map_err(|error| anyhow::anyhow!("{error}"))?;
    let map = ds41rt_loader::EngramTokenMap::from_tokenizer(&tokenizer)?;
    println!("{}", serde_json::to_string(map.compressed_ids())?);
    Ok(())
}
