use anyhow::{Context, Result};
use ds4rt_core::{ModelFacts, TensorCatalog};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotResolution {
    pub model_id: String,
    pub cache_root: PathBuf,
    pub model_cache: PathBuf,
    pub snapshot_path: Option<PathBuf>,
    pub snapshots: Vec<PathBuf>,
}

pub fn default_hf_home() -> PathBuf {
    env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface")))
        .unwrap_or_else(|| PathBuf::from("/root/.cache/huggingface"))
}

pub fn model_cache_dir(hf_home: &Path, model_id: &str) -> PathBuf {
    hf_home
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")))
}

pub fn resolve_snapshot(model_id: &str, hf_home: Option<&Path>) -> Result<SnapshotResolution> {
    resolve_snapshot_at_revision(model_id, hf_home, None)
}

pub fn resolve_snapshot_at_revision(
    model_id: &str,
    hf_home: Option<&Path>,
    revision: Option<&str>,
) -> Result<SnapshotResolution> {
    let cache_root = hf_home
        .map(Path::to_path_buf)
        .unwrap_or_else(default_hf_home);
    let model_cache = model_cache_dir(&cache_root, model_id);
    let snapshots_root = model_cache.join("snapshots");
    let mut snapshots = Vec::new();
    if snapshots_root.is_dir() {
        for entry in fs::read_dir(&snapshots_root)
            .with_context(|| format!("reading {}", snapshots_root.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                snapshots.push(entry.path());
            }
        }
    }
    snapshots.sort();
    let validate_revision = |revision: &str, source: &str| -> Result<()> {
        anyhow::ensure!(
            !revision.is_empty()
                && Path::new(revision)
                    .components()
                    .all(|component| matches!(component, Component::Normal(_)))
                && Path::new(revision).components().count() == 1,
            "{source} contains invalid revision {revision:?}",
        );
        Ok(())
    };
    let select_revision = |revision: &str, source: &str| -> Result<PathBuf> {
        validate_revision(revision, source)?;
        let selected = snapshots_root.join(revision);
        anyhow::ensure!(
            snapshots.iter().any(|snapshot| snapshot == &selected),
            "{source} selects missing snapshot {}",
            selected.display(),
        );
        Ok(selected)
    };
    let snapshot_path = if let Some(revision) = revision {
        Some(select_revision(revision, "explicit Hugging Face revision")?)
    } else {
        let main_ref = model_cache.join("refs/main");
        let main_ref_metadata = match fs::symlink_metadata(&main_ref) {
            Ok(metadata) => Some(metadata),
            Err(error) if error.kind() == ErrorKind::NotFound => None,
            Err(error) => {
                return Err(error).with_context(|| format!("reading {}", main_ref.display()))
            }
        };
        if let Some(metadata) = main_ref_metadata {
            anyhow::ensure!(
                metadata.file_type().is_file(),
                "Hugging Face main ref {} is not a regular file",
                main_ref.display()
            );
            let revision = fs::read_to_string(&main_ref)
                .with_context(|| format!("reading {}", main_ref.display()))?;
            let revision = revision.trim();
            Some(select_revision(
                revision,
                &format!("Hugging Face main ref {}", main_ref.display()),
            )?)
        } else {
            snapshots.last().cloned()
        }
    };
    Ok(SnapshotResolution {
        model_id: model_id.to_owned(),
        cache_root,
        model_cache,
        snapshot_path,
        snapshots,
    })
}

pub fn empty_catalog_for_snapshot(model_id: &str, snapshot_path: &Path) -> TensorCatalog {
    TensorCatalog {
        model_id: model_id.to_owned(),
        snapshot_path: snapshot_path.display().to_string(),
        facts: ModelFacts {
            model_id: model_id.to_owned(),
            ..ModelFacts::default()
        },
        tensors: Vec::new(),
    }
}
