//! JSON-line qualification bridge. Serving calls the same loader API directly.
use anyhow::Result;
use ds41rt_loader::{V41Image, V41ImageGrid};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{self, BufRead};

fn run(body: Value) -> Result<Value> {
    if let Some(path) = body["path"].as_str() {
        let image = V41Image::decode(&std::fs::read(path)?)?;
        if let Some(path) = body["patch_output"].as_str() {
            std::fs::write(path, image.patches())?;
        }
        Ok(json!({"grid":image.grid(), "tokens":image.grid().tokens(),
            "patch_sha256":format!("{:x}",Sha256::digest(image.patches())),
            "identity":image.identity(), "patch_bytes":image.patches().len()}))
    } else {
        let width = u32::try_from(
            body["width"]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("missing width"))?,
        )?;
        let height = u32::try_from(
            body["height"]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("missing height"))?,
        )?;
        Ok(json!({"grid":V41ImageGrid::plan(width,height)?}))
    }
}
fn main() -> Result<()> {
    for line in io::stdin().lock().lines() {
        let output = match serde_json::from_str(&line?)
            .map_err(anyhow::Error::from)
            .and_then(run)
        {
            Ok(value) => value,
            Err(error) => json!({"error":format!("{error:#}")}),
        };
        println!("{output}");
    }
    Ok(())
}
