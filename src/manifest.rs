use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

/// Records what `prepare` built, so `run` can refuse a stale cache instead of
/// silently falling back to CPU.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub source_sha256: String,
    pub backbone: String,
    pub width: u32,
    pub height: u32,
    pub ratio: f32,
    pub gpu_arch: String,
    pub rocm_version: String,
}

impl Manifest {
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("manifest.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Manifest> {
        let path = dir.join("manifest.json");
        let json = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "no prepared model at {}; run `matting prepare` first",
                path.display()
            )
        })?;
        Ok(serde_json::from_str(&json)?)
    }

    /// `self` is what is on disk; `other` is what the current run needs.
    pub fn check_matches(&self, other: &Manifest) -> Result<()> {
        let mut diffs = Vec::new();
        macro_rules! cmp {
            ($field:ident) => {
                if self.$field != other.$field {
                    diffs.push(format!(
                        "{}: cached {:?} but need {:?}",
                        stringify!($field),
                        self.$field,
                        other.$field
                    ));
                }
            };
        }
        cmp!(source_sha256);
        cmp!(backbone);
        cmp!(width);
        cmp!(height);
        cmp!(gpu_arch);
        cmp!(rocm_version);
        if (self.ratio - other.ratio).abs() > f32::EPSILON {
            diffs.push(format!(
                "ratio: cached {} but need {}",
                self.ratio, other.ratio
            ));
        }
        if diffs.is_empty() {
            return Ok(());
        }
        Err(anyhow!(
            "prepared model is stale, re-run `matting prepare`:\n  {}",
            diffs.join("\n  ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Manifest {
        Manifest {
            source_sha256: "abc123".into(),
            backbone: "resnet50".into(),
            width: 1024,
            height: 576,
            ratio: 0.5,
            gpu_arch: "gfx1151".into(),
            rocm_version: "7.2.4".into(),
        }
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("matting-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        sample().save(&dir).unwrap();
        assert_eq!(Manifest::load(&dir).unwrap(), sample());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn matching_manifests_pass() {
        assert!(sample().check_matches(&sample()).is_ok());
    }

    #[test]
    fn resolution_mismatch_is_reported() {
        let mut other = sample();
        other.width = 1280;
        let err = sample().check_matches(&other).unwrap_err().to_string();
        assert!(err.contains("width"), "error should name the field: {err}");
    }

    #[test]
    fn gpu_arch_mismatch_is_reported() {
        let mut other = sample();
        other.gpu_arch = "gfx1100".into();
        let err = sample().check_matches(&other).unwrap_err().to_string();
        assert!(err.contains("gpu_arch"), "error should name the field: {err}");
    }
}
