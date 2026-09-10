//! Reproduce request hash transactions for the official-reference qualification tool.
use anyhow::{Context, Result};
use ds41rt_core::EngramHistory;
use ds41rt_loader::EngramTokenMap;
use serde::Deserialize;
#[derive(Deserialize)]
struct Event {
    request: usize,
    tokens: Vec<u32>,
    images: Vec<bool>,
    accept: usize,
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let tokenizer =
        tokenizers::Tokenizer::from_file(args.get(1).context("missing tokenizer path")?)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
    let map = EngramTokenMap::from_tokenizer(&tokenizer)?;
    let events: Vec<Event> =
        serde_json::from_slice(&std::fs::read(args.get(2).context("missing event file")?)?)?;
    let mut histories: Vec<_> = (0..16)
        .map(|_| EngramHistory::new(map.pad_id()))
        .collect::<Result<_, _>>()?;
    let mut output = Vec::new();
    for event in events {
        let history = histories
            .get_mut(event.request)
            .context("invalid request ID")?;
        let batch = map.prepare_batch(
            history,
            history.position(),
            &event.tokens,
            Some(&event.images),
            1024,
        )?;
        history.commit(&batch, event.accept)?;
        output.push(serde_json::json!({"hashes": batch.hashes(), "position":history.position()}));
    }
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
