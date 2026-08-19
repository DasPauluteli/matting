use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use onnx_protobuf::ModelProto;
use ort::{ep, session::Session};
use protobuf::Message;
use sha2::{Digest, Sha256};

use crate::cli::{Backbone, PrepareArgs};
use crate::freeze::{freeze_model, read_state_channels};
use crate::manifest::Manifest;

/// Everything `prepare` produces lives here.
pub fn cache_dir() -> PathBuf {
    dirs_cache().join("matting")
}

fn dirs_cache() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME must be set")).join(".cache")
        })
}

pub fn frozen_model_path(dir: &Path) -> PathBuf {
    dir.join("model_frozen.onnx")
}

pub fn mxr_dir(dir: &Path) -> PathBuf {
    dir.join("mxr")
}

/// MIGraphX aborts session init if this is unset, writing to `""/<hash>.mxr`.
pub fn set_migraphx_cache_env(dir: &Path) -> Result<()> {
    let mxr = mxr_dir(dir);
    std::fs::create_dir_all(&mxr)?;
    std::env::set_var("ORT_MIGRAPHX_MODEL_CACHE_PATH", &mxr);
    Ok(())
}

pub fn gpu_arch() -> String {
    std::process::Command::new("rocminfo")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8(o.stdout).ok().and_then(|s| {
                s.lines()
                    .find_map(|l| {
                        l.trim()
                            .strip_prefix("Name:")
                            .map(str::trim)
                            .filter(|n| n.starts_with("gfx"))
                    })
                    .map(str::to_string)
            })
        })
        .unwrap_or_else(|| "unknown".into())
}

pub fn rocm_version() -> String {
    std::fs::read_to_string("/opt/rocm/.info/version")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

pub fn build_manifest(
    source: &Path,
    backbone: Backbone,
    width: u32,
    height: u32,
    ratio: f32,
) -> Result<Manifest> {
    let bytes = std::fs::read(source).with_context(|| format!("reading {}", source.display()))?;
    let sha = format!("{:x}", Sha256::digest(&bytes));
    Ok(Manifest {
        source_sha256: sha,
        backbone: format!("{backbone:?}").to_lowercase(),
        width,
        height,
        ratio,
        gpu_arch: gpu_arch(),
        rocm_version: rocm_version(),
    })
}

pub fn run(args: &PrepareArgs) -> Result<()> {
    args.validate().map_err(anyhow::Error::msg)?;
    let source = Path::new(&args.from);
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;

    println!("Reading {}", source.display());
    let bytes = std::fs::read(source).with_context(|| format!("reading {}", source.display()))?;
    let model = ModelProto::parse_from_bytes(&bytes).context("not a valid ONNX model")?;

    // Trust the model over the flag, but tell the user when they disagree.
    let channels = read_state_channels(&model)?;
    if channels != args.model.expected_channels() {
        anyhow::bail!(
            "--model {:?} expects state channels {:?} but this model declares {:?}; \
             pass the matching --model",
            args.model,
            args.model.expected_channels(),
            channels
        );
    }

    println!(
        "Freezing graph to {}x{} at ratio {}",
        args.width, args.height, args.ratio
    );
    let frozen = freeze_model(model, args.width, args.height, args.ratio)?;
    let frozen_path = frozen_model_path(&dir);
    std::fs::write(&frozen_path, frozen.write_to_bytes()?)?;
    println!("Wrote {}", frozen_path.display());

    set_migraphx_cache_env(&dir)?;
    println!("Compiling for MIGraphX. First run takes a few minutes.");
    let start = std::time::Instant::now();
    ort::init().commit();
    // `with_execution_providers` returns ort's recoverable Error<SessionBuilder>,
    // which is neither Send nor Sync, so it cannot cross `?` into anyhow.
    let _session = Session::builder()?
        .with_execution_providers([ep::MIGraphX::default().with_fp16(true).build()])
        .map_err(|e| anyhow::anyhow!("registering the MIGraphX provider: {e}"))?
        .commit_from_file(&frozen_path)
        .context("MIGraphX failed to compile the frozen model")?;
    println!("Compiled in {:.0}s", start.elapsed().as_secs_f64());

    build_manifest(source, args.model, args.width, args.height, args.ratio)?.save(&dir)?;
    println!("Ready. Cache at {}", dir.display());
    Ok(())
}
