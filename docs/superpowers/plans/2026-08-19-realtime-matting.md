# Realtime AMD GPU Webcam Matting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `matting`, a Rust CLI that routes a V4L2 webcam through Robust Video Matting on an AMD GPU and publishes alpha, greenscreen, or image-composited output to a v4l2loopback device at ≥25 fps.

**Architecture:** Two subcommands. `prepare` rewrites a stock RVM ONNX into a fully static graph (demoting `downsample_ratio` to a constant initializer and pinning all input shapes) and forces MIGraphX to compile it into a `.mxr` cache. `run` streams frames single-threaded: V4L2 capture → YUYV→RGB conversion → RVM inference via ONNX Runtime's MIGraphX execution provider → mode-dependent compositing → v4l2loopback write.

**Tech Stack:** Rust 1.97, `ort` 2.0.0-rc.13 (load-dynamic + migraphx), system ONNX Runtime 1.28, ROCm 7.2.4 / MIGraphX 2.15, `v4l` 0.14, `onnx-protobuf` 0.2.3, `clap` 4, `image` 0.25.

**Spec:** `docs/superpowers/specs/2026-08-19-realtime-matting-design.md`

## Global Constraints

- Target GPU is **AMD Radeon 8060S, gfx1151**. The only usable execution provider is `MIGraphXExecutionProvider`. Never add CUDA or ROCm EPs.
- **`ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so`** must be set for every `cargo build`, `cargo test`, and binary run. Without it `ort`'s `load-dynamic` feature cannot find the runtime.
- **`ORT_MIGRAPHX_MODEL_CACHE_PATH` must name an existing directory** before any MIGraphX session is created. Unset, the EP aborts session init trying to write `""/<hash>.mxr`. The program must create this directory itself.
- `ort` features are exactly: `load-dynamic,migraphx,ndarray,half`, with `default-features = false`.
- Source device format is **YUYV 4:2:2, 1024×576, 30 fps**. Default model is **resnet50**, default `--ratio` is **0.5**, default `--mode` is **alpha**.
- Never hardcode RVM recurrent-state channel widths. MobileNetV3 is `[16,20,40,64]`; ResNet50 is `[16,32,64,128]`. They must be read from the model's `r1o..r4o` output declarations.
- First MIGraphX compile takes ~120 s. Cached reload is ~0.8 s. Never do the compile inside `run`.
- The spec names `prost` for protobuf; this plan uses `onnx-protobuf` 0.2.3 (rust-protobuf 3.4 backend) instead, which ships the generated ONNX types directly.

---

### Task 1: Project scaffolding and CLI

**Files:**
- Create: `Cargo.toml`
- Create: `src/main.rs`
- Create: `src/cli.rs`
- Create: `.cargo/config.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: `cli::Cli`, `cli::Command::{Prepare, Run}`, `cli::Mode::{Alpha, Greenscreen, Image}`, `cli::Fit::{Cover, Stretch, Contain}`, `cli::Backbone::{Resnet50, Mobilenetv3}`, `cli::parse_color(&str) -> Result<[u8;3], String>`.

- [ ] **Step 1: Create `Cargo.toml`**

```toml
[package]
name = "matting"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
image = "0.25"
onnx-protobuf = "0.2.3"
protobuf = "3.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
v4l = "0.14"

[dependencies.ort]
version = "2.0.0-rc.13"
default-features = false
features = ["load-dynamic", "migraphx", "ndarray", "half"]
```

- [ ] **Step 2: Create `.cargo/config.toml` so the ORT path is always set**

```toml
[env]
ORT_DYLIB_PATH = "/usr/lib/libonnxruntime.so"
```

- [ ] **Step 3: Write the failing CLI tests**

Create `src/cli.rs` with only this test module at the bottom (no implementation yet):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn defaults_to_alpha_mode_and_resnet50() {
        let cli = Cli::parse_from(["matting", "run"]);
        let Command::Run(args) = cli.command else { panic!("expected run") };
        assert_eq!(args.mode, Mode::Alpha);
        assert_eq!(args.model, Backbone::Resnet50);
        assert_eq!(args.input, "/dev/video0");
    }

    #[test]
    fn parses_hex_color() {
        assert_eq!(parse_color("#00FF00").unwrap(), [0, 255, 0]);
        assert_eq!(parse_color("00ff00").unwrap(), [0, 255, 0]);
    }

    #[test]
    fn parses_triplet_color() {
        assert_eq!(parse_color("0,255,0").unwrap(), [0, 255, 0]);
    }

    #[test]
    fn rejects_malformed_color() {
        assert!(parse_color("#GGGGGG").is_err());
        assert!(parse_color("1,2").is_err());
        assert!(parse_color("300,0,0").is_err());
    }

    #[test]
    fn image_mode_requires_image_path() {
        let cli = Cli::parse_from(["matting", "run", "--mode", "image"]);
        let Command::Run(args) = cli.command else { panic!("expected run") };
        assert!(args.validate().is_err(), "image mode without --image must fail");
    }
}
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cargo test --lib cli`
Expected: FAIL — `Cli`, `Command`, `Mode`, `Backbone`, `parse_color` are not defined.

- [ ] **Step 5: Implement `src/cli.rs`**

Put this above the test module:

```rust
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(name = "matting", about = "Realtime webcam background matting on AMD GPUs")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Freeze an RVM ONNX model and compile the MIGraphX cache. Run once.
    Prepare(PrepareArgs),
    /// Stream the webcam through the model to a virtual camera.
    Run(RunArgs),
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Alpha,
    Greenscreen,
    Image,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    Cover,
    Stretch,
    Contain,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backbone {
    Resnet50,
    Mobilenetv3,
}

impl Backbone {
    /// Channel widths this backbone is expected to declare, used only to
    /// cross-check what we read from the model. Never used to build shapes.
    pub fn expected_channels(self) -> [i64; 4] {
        match self {
            Backbone::Mobilenetv3 => [16, 20, 40, 64],
            Backbone::Resnet50 => [16, 32, 64, 128],
        }
    }
}

#[derive(Args, Debug)]
pub struct PrepareArgs {
    /// Path to a stock RVM ONNX export.
    #[arg(long)]
    pub from: String,
    #[arg(long, value_enum, default_value_t = Backbone::Resnet50)]
    pub model: Backbone,
    #[arg(long, default_value_t = 1024)]
    pub width: u32,
    #[arg(long, default_value_t = 576)]
    pub height: u32,
    #[arg(long, default_value_t = 0.5)]
    pub ratio: f32,
}

#[derive(Args, Debug)]
pub struct RunArgs {
    #[arg(long, default_value = "/dev/video0")]
    pub input: String,
    #[arg(long, default_value = "/dev/video9")]
    pub output: String,
    #[arg(long, value_enum, default_value_t = Mode::Alpha)]
    pub mode: Mode,
    #[arg(long, value_enum, default_value_t = Backbone::Resnet50)]
    pub model: Backbone,
    /// Greenscreen colour as #RRGGBB or r,g,b.
    #[arg(long, default_value = "#00FF00")]
    pub color: String,
    /// Background image for --mode image.
    #[arg(long)]
    pub image: Option<String>,
    #[arg(long, value_enum, default_value_t = Fit::Cover)]
    pub fit: Fit,
}

impl RunArgs {
    pub fn validate(&self) -> Result<(), String> {
        if self.mode == Mode::Image && self.image.is_none() {
            return Err("--mode image requires --image <path>".into());
        }
        parse_color(&self.color)?;
        Ok(())
    }
}

/// Accepts `#RRGGBB`, `RRGGBB`, or `r,g,b`.
pub fn parse_color(s: &str) -> Result<[u8; 3], String> {
    let s = s.trim();
    if s.contains(',') {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err(format!("expected r,g,b but got {s:?}"));
        }
        let mut out = [0u8; 3];
        for (i, p) in parts.iter().enumerate() {
            out[i] = p
                .trim()
                .parse::<u8>()
                .map_err(|_| format!("invalid colour component {p:?} in {s:?}"))?;
        }
        return Ok(out);
    }
    let hex = s.strip_prefix('#').unwrap_or(s);
    if hex.len() != 6 {
        return Err(format!("expected 6 hex digits but got {s:?}"));
    }
    let mut out = [0u8; 3];
    for i in 0..3 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| format!("invalid hex colour {s:?}"))?;
    }
    Ok(out)
}
```

- [ ] **Step 6: Create `src/main.rs`**

```rust
mod cli;

use clap::Parser;
use cli::{Cli, Command};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Prepare(_) => anyhow::bail!("prepare not implemented yet"),
        Command::Run(args) => {
            args.validate().map_err(anyhow::Error::msg)?;
            anyhow::bail!("run not implemented yet")
        }
    }
}
```

Add `pub mod cli;` to a new `src/lib.rs` so tests can reach it:

```rust
pub mod cli;
```

And change `src/main.rs`'s first line from `mod cli;` to `use matting::cli;`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test --lib cli`
Expected: PASS, 5 tests.

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock .cargo/config.toml src/
git commit -m "feat: add CLI skeleton with mode, colour and backbone parsing"
```

---

### Task 2: ONNX graph surgery

This is the task that makes the GPU work at all. It converts a stock RVM export into a graph MIGraphX can compile.

**Files:**
- Create: `src/freeze.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `cli::Backbone`.
- Produces:
  - `freeze::StateShapes { channels: [i64; 4], spatial: [(i64, i64); 4] }`
  - `freeze::read_state_channels(model: &ModelProto) -> anyhow::Result<[i64; 4]>`
  - `freeze::state_shapes(model: &ModelProto, width: u32, height: u32, ratio: f32) -> anyhow::Result<StateShapes>`
  - `freeze::freeze_model(model: ModelProto, width: u32, height: u32, ratio: f32) -> anyhow::Result<ModelProto>`

- [ ] **Step 1: Write the failing tests**

Create `src/freeze.rs` containing only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use protobuf::Message;

    /// Stock RVM exports live outside the repo. Skip rather than fail when absent.
    fn load(path: &str) -> Option<onnx_protobuf::ModelProto> {
        let bytes = std::fs::read(path).ok()?;
        onnx_protobuf::ModelProto::parse_from_bytes(&bytes).ok()
    }

    const MNV3: &str = concat!(env!("HOME"), "/Downloads/rvm_mobilenetv3_fp32.onnx");
    const R50: &str = concat!(
        env!("HOME"),
        "/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx"
    );

    #[test]
    fn reads_mobilenetv3_channels_from_model() {
        let Some(m) = load(MNV3) else { return };
        assert_eq!(read_state_channels(&m).unwrap(), [16, 20, 40, 64]);
    }

    #[test]
    fn reads_resnet50_channels_from_model() {
        let Some(m) = load(R50) else { return };
        assert_eq!(read_state_channels(&m).unwrap(), [16, 32, 64, 128]);
    }

    #[test]
    fn computes_state_spatial_dims() {
        let Some(m) = load(MNV3) else { return };
        let s = state_shapes(&m, 1024, 576, 0.5).unwrap();
        // Downsampled is 512x288; states sit at /2, /4, /8, /16 of that.
        assert_eq!(s.spatial, [(144, 256), (72, 128), (36, 64), (18, 32)]);
    }

    #[test]
    fn frozen_model_has_no_downsample_ratio_input() {
        let Some(m) = load(MNV3) else { return };
        let f = freeze_model(m, 1024, 576, 0.5).unwrap();
        let g = f.graph.as_ref().unwrap();
        assert!(
            !g.input.iter().any(|i| i.name == "downsample_ratio"),
            "downsample_ratio must not remain a graph input"
        );
        assert!(
            g.initializer.iter().any(|i| i.name == "downsample_ratio"),
            "downsample_ratio must become an initializer"
        );
    }

    #[test]
    fn frozen_model_pins_all_input_dims() {
        let Some(m) = load(MNV3) else { return };
        let f = freeze_model(m, 1024, 576, 0.5).unwrap();
        let g = f.graph.as_ref().unwrap();
        for input in &g.input {
            let dims = &input.type_.as_ref().unwrap().tensor_type().shape.dim;
            for d in dims {
                assert!(
                    d.has_dim_value() && d.dim_value() > 0,
                    "input {} has an unpinned dim",
                    input.name
                );
            }
        }
        let src = g.input.iter().find(|i| i.name == "src").unwrap();
        let dims: Vec<i64> = src.type_.as_ref().unwrap().tensor_type().shape.dim
            .iter().map(|d| d.dim_value()).collect();
        assert_eq!(dims, vec![1, 3, 576, 1024]);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib freeze`
Expected: FAIL — `read_state_channels`, `state_shapes`, `freeze_model` are not defined.

- [ ] **Step 3: Implement `src/freeze.rs`**

Put this above the test module:

```rust
use anyhow::{anyhow, Context, Result};
use onnx_protobuf::{tensor_proto::DataType, ModelProto, TensorProto};

/// Concrete shapes for RVM's four recurrent state tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateShapes {
    pub channels: [i64; 4],
    /// (height, width) per state level.
    pub spatial: [(i64, i64); 4],
}

/// RVM declares its recurrent channel widths statically on the r1o..r4o
/// outputs even when everything else is dynamic. Read them from there rather
/// than assuming a backbone: MobileNetV3 and ResNet50 differ.
pub fn read_state_channels(model: &ModelProto) -> Result<[i64; 4]> {
    let graph = model.graph.as_ref().context("model has no graph")?;
    let mut channels = [0i64; 4];
    for (i, ch) in channels.iter_mut().enumerate() {
        let name = format!("r{}o", i + 1);
        let out = graph
            .output
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| anyhow!("model has no output {name}; is this an RVM export?"))?;
        let dim = &out
            .type_
            .as_ref()
            .context("output has no type")?
            .tensor_type()
            .shape
            .dim;
        if dim.len() != 4 {
            return Err(anyhow!("{name} should be rank 4, got {}", dim.len()));
        }
        if !dim[1].has_dim_value() || dim[1].dim_value() <= 0 {
            return Err(anyhow!("{name} has no static channel dim"));
        }
        *ch = dim[1].dim_value();
    }
    Ok(channels)
}

pub fn state_shapes(model: &ModelProto, width: u32, height: u32, ratio: f32) -> Result<StateShapes> {
    let channels = read_state_channels(model)?;
    let dw = (width as f32 * ratio) as i64;
    let dh = (height as f32 * ratio) as i64;
    let mut spatial = [(0i64, 0i64); 4];
    for (i, s) in spatial.iter_mut().enumerate() {
        let stride = 1i64 << (i + 1); // /2, /4, /8, /16 of the downsampled size
        *s = (div_ceil(dh, stride), div_ceil(dw, stride));
    }
    Ok(StateShapes { channels, spatial })
}

fn div_ceil(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// Rewrite a stock RVM graph into one MIGraphX can compile:
/// demote `downsample_ratio` to a constant initializer and pin every input dim.
pub fn freeze_model(mut model: ModelProto, width: u32, height: u32, ratio: f32) -> Result<ModelProto> {
    let shapes = state_shapes(&model, width, height, ratio)?;
    let graph = model.graph.as_mut().context("model has no graph")?;

    // 1. downsample_ratio: graph input -> initializer. This is what makes the
    //    internal Resize ops constant, which MIGraphX requires.
    let before = graph.input.len();
    graph.input.retain(|i| i.name != "downsample_ratio");
    if graph.input.len() == before {
        return Err(anyhow!("model has no downsample_ratio input; already frozen?"));
    }
    let mut ratio_init = TensorProto::new();
    ratio_init.name = "downsample_ratio".to_string();
    ratio_init.data_type = DataType::FLOAT as i32;
    ratio_init.dims = vec![1];
    ratio_init.float_data = vec![ratio];
    graph.initializer.push(ratio_init);

    // 2/3. Pin src and the four recurrent states.
    let mut wanted: Vec<(String, Vec<i64>)> =
        vec![("src".to_string(), vec![1, 3, height as i64, width as i64])];
    for i in 0..4 {
        let (h, w) = shapes.spatial[i];
        wanted.push((format!("r{}i", i + 1), vec![1, shapes.channels[i], h, w]));
    }

    for (name, dims) in wanted {
        let input = graph
            .input
            .iter_mut()
            .find(|i| i.name == name)
            .ok_or_else(|| anyhow!("model has no input {name}"))?;
        // `tensor_type` is a protobuf oneof, so it is reached via the generated
        // `mut_tensor_type()` accessor rather than as a plain field.
        let shape = input
            .type_
            .as_mut()
            .context("input has no type")?
            .mut_tensor_type()
            .shape
            .mut_or_insert_default();
        if shape.dim.len() != dims.len() {
            return Err(anyhow!(
                "{name} should be rank {}, got {}",
                dims.len(),
                shape.dim.len()
            ));
        }
        for (d, v) in shape.dim.iter_mut().zip(dims) {
            d.clear_dim_param();
            d.set_dim_value(v);
        }
    }

    Ok(model)
}
```

Add to `src/lib.rs`:

```rust
pub mod freeze;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib freeze`
Expected: PASS, 5 tests. If the RVM exports are absent the tests return early and still pass — check the output mentions 5 tests running.

- [ ] **Step 5: Verify against a real model end to end**

Run:
```bash
cargo run --release -- --help
```
Expected: help text listing `prepare` and `run`. (Full `prepare` wiring lands in Task 3.)

- [ ] **Step 6: Commit**

```bash
git add src/freeze.rs src/lib.rs
git commit -m "feat: freeze RVM ONNX graphs for MIGraphX compilation"
```

---

### Task 3: The `prepare` command — cache compilation and manifest

**Files:**
- Create: `src/manifest.rs`
- Create: `src/prepare.rs`
- Modify: `src/lib.rs`, `src/main.rs`

**Interfaces:**
- Consumes: `freeze::freeze_model`, `cli::{PrepareArgs, Backbone}`.
- Produces:
  - `manifest::Manifest { source_sha256: String, backbone: String, width: u32, height: u32, ratio: f32, gpu_arch: String, rocm_version: String }`
  - `manifest::Manifest::load(dir: &Path) -> Result<Manifest>` / `save(&self, dir: &Path) -> Result<()>`
  - `manifest::Manifest::check_matches(&self, other: &Manifest) -> Result<()>`
  - `prepare::cache_dir() -> PathBuf`, `prepare::frozen_model_path(dir: &Path) -> PathBuf`
  - `prepare::run(args: &PrepareArgs) -> Result<()>`

- [ ] **Step 1: Write the failing manifest tests**

Create `src/manifest.rs` containing only:

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib manifest`
Expected: FAIL — `Manifest` is not defined.

- [ ] **Step 3: Implement `src/manifest.rs`**

```rust
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
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("no prepared model at {}; run `matting prepare` first", path.display()))?;
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
            diffs.push(format!("ratio: cached {} but need {}", self.ratio, other.ratio));
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
```

Add `pub mod manifest;` to `src/lib.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib manifest`
Expected: PASS, 4 tests.

- [ ] **Step 5: Implement `src/prepare.rs`**

```rust
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
                    .find_map(|l| l.trim().strip_prefix("Name:").map(str::trim).filter(|n| n.starts_with("gfx")))
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

pub fn build_manifest(source: &Path, backbone: Backbone, width: u32, height: u32, ratio: f32) -> Result<Manifest> {
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

    println!("Freezing graph to {}x{} at ratio {}", args.width, args.height, args.ratio);
    let frozen = freeze_model(model, args.width, args.height, args.ratio)?;
    let frozen_path = frozen_model_path(&dir);
    std::fs::write(&frozen_path, frozen.write_to_bytes()?)?;
    println!("Wrote {}", frozen_path.display());

    set_migraphx_cache_env(&dir)?;
    println!("Compiling for MIGraphX. First run takes about two minutes.");
    let start = std::time::Instant::now();
    ort::init().commit();
    let _session = Session::builder()?
        .with_execution_providers([ep::MIGraphX::default().with_fp16(true).build()])?
        .commit_from_file(&frozen_path)
        .context("MIGraphX failed to compile the frozen model")?;
    println!("Compiled in {:.0}s", start.elapsed().as_secs_f64());

    build_manifest(source, args.model, args.width, args.height, args.ratio)?.save(&dir)?;
    println!("Ready. Cache at {}", dir.display());
    Ok(())
}
```

Add `pub mod prepare;` to `src/lib.rs`, and wire `src/main.rs`:

```rust
Command::Prepare(args) => matting::prepare::run(&args),
```

- [ ] **Step 6: Run `prepare` against a real model**

Run:
```bash
cargo run --release -- prepare \
  --from ~/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx \
  --model resnet50
```
Expected: prints the freeze step, then "Compiling for MIGraphX", then "Compiled in ~120s", then "Ready." Verify `~/.cache/matting/manifest.json`, `model_frozen.onnx`, and a non-empty `mxr/` exist.

- [ ] **Step 7: Verify the cache makes a second run fast**

Run the same command again.
Expected: "Compiled in" reports roughly 1 second, not 120.

- [ ] **Step 8: Commit**

```bash
git add src/manifest.rs src/prepare.rs src/lib.rs src/main.rs
git commit -m "feat: add prepare command with MIGraphX cache and staleness manifest"
```

---

### Task 4: YUYV ↔ RGB conversion

**Files:**
- Create: `src/convert.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `convert::yuyv_to_rgb_f32_nchw(yuyv: &[u8], width: usize, height: usize, out: &mut [f32])`
  - `convert::rgb_to_yuyv(rgb: &[u8], width: usize, height: usize, out: &mut [u8])`
  - `convert::yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3]`
  - `convert::rgb_to_yuv(r: u8, g: u8, b: u8) -> [u8; 3]`

- [ ] **Step 1: Write the failing tests**

Create `src/convert.rs` containing only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_round_trips() {
        // Mid grey: equal RGB should map to neutral chroma and back.
        let [y, u, v] = rgb_to_yuv(128, 128, 128);
        let [r, g, b] = yuv_to_rgb(y, u, v);
        assert!((r as i32 - 128).abs() <= 2, "r={r}");
        assert!((g as i32 - 128).abs() <= 2, "g={g}");
        assert!((b as i32 - 128).abs() <= 2, "b={b}");
    }

    #[test]
    fn primaries_round_trip_within_tolerance() {
        for colour in [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0], [255, 255, 255]] {
            let [y, u, v] = rgb_to_yuv(colour[0], colour[1], colour[2]);
            let got = yuv_to_rgb(y, u, v);
            for i in 0..3 {
                assert!(
                    (got[i] as i32 - colour[i] as i32).abs() <= 4,
                    "{colour:?} -> {got:?} channel {i}"
                );
            }
        }
    }

    #[test]
    fn yuyv_to_nchw_is_planar_and_normalized() {
        // 2x1 image, one YUYV macropixel of white.
        let [y, u, v] = rgb_to_yuv(255, 255, 255);
        let yuyv = [y, u, y, v];
        let mut out = vec![0f32; 3 * 2];
        yuyv_to_rgb_f32_nchw(&yuyv, 2, 1, &mut out);
        for value in &out {
            assert!(*value > 0.95, "expected near 1.0, got {value}");
            assert!(*value <= 1.0, "must be normalized, got {value}");
        }
    }

    #[test]
    fn rgb_to_yuyv_round_trips_through_nchw() {
        let width = 4;
        let height = 2;
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i * 7 % 256) as u8).collect();
        let mut yuyv = vec![0u8; width * height * 2];
        rgb_to_yuyv(&rgb, width, height, &mut yuyv);
        let mut back = vec![0f32; 3 * width * height];
        yuyv_to_rgb_f32_nchw(&yuyv, width, height, &mut back);
        // 4:2:2 discards half the chroma, so only luma is tightly preserved.
        for p in 0..width * height {
            let orig_y = rgb_to_yuv(rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2])[0];
            let r = (back[p] * 255.0) as i32;
            let g = (back[width * height + p] * 255.0) as i32;
            let b = (back[2 * width * height + p] * 255.0) as i32;
            let got_y = rgb_to_yuv(r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8)[0];
            assert!((orig_y as i32 - got_y as i32).abs() <= 6, "luma drift at {p}");
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib convert`
Expected: FAIL — the conversion functions are not defined.

- [ ] **Step 3: Implement `src/convert.rs`**

```rust
//! YUYV 4:2:2 <-> RGB conversion.
//!
//! Uses the BT.601 limited-range matrix, which is what USB and loopback
//! webcams overwhelmingly produce. If colours look washed out or over-saturated
//! against the real camera, this is the first constant to revisit.

#[inline]
pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let y = (y as f32 - 16.0) * 1.164_383;
    let u = u as f32 - 128.0;
    let v = v as f32 - 128.0;
    let r = y + 1.596_027 * v;
    let g = y - 0.391_762 * u - 0.812_968 * v;
    let b = y + 2.017_232 * u;
    [clamp_u8(r), clamp_u8(g), clamp_u8(b)]
}

#[inline]
pub fn rgb_to_yuv(r: u8, g: u8, b: u8) -> [u8; 3] {
    let (r, g, b) = (r as f32, g as f32, b as f32);
    let y = 0.256_788 * r + 0.504_129 * g + 0.097_906 * b + 16.0;
    let u = -0.148_223 * r - 0.290_993 * g + 0.439_216 * b + 128.0;
    let v = 0.439_216 * r - 0.367_788 * g - 0.071_427 * b + 128.0;
    [clamp_u8(y), clamp_u8(u), clamp_u8(v)]
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// YUYV 4:2:2 -> planar RGB, normalized to [0,1], laid out NCHW as the model
/// expects. `out` must be `3 * width * height` long.
pub fn yuyv_to_rgb_f32_nchw(yuyv: &[u8], width: usize, height: usize, out: &mut [f32]) {
    let plane = width * height;
    debug_assert_eq!(out.len(), plane * 3);
    debug_assert_eq!(yuyv.len(), plane * 2);

    for row in 0..height {
        let src = row * width * 2;
        let dst = row * width;
        for pair in 0..width / 2 {
            let i = src + pair * 4;
            let (y0, u, y1, v) = (yuyv[i], yuyv[i + 1], yuyv[i + 2], yuyv[i + 3]);
            let p0 = yuv_to_rgb(y0, u, v);
            let p1 = yuv_to_rgb(y1, u, v);
            let o = dst + pair * 2;
            for c in 0..3 {
                out[c * plane + o] = p0[c] as f32 / 255.0;
                out[c * plane + o + 1] = p1[c] as f32 / 255.0;
            }
        }
    }
}

/// Interleaved RGB8 -> YUYV 4:2:2. Chroma is averaged across each pixel pair.
/// `out` must be `2 * width * height` long.
pub fn rgb_to_yuyv(rgb: &[u8], width: usize, height: usize, out: &mut [u8]) {
    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert_eq!(out.len(), width * height * 2);

    for row in 0..height {
        let src = row * width * 3;
        let dst = row * width * 2;
        for pair in 0..width / 2 {
            let a = src + pair * 6;
            let p0 = rgb_to_yuv(rgb[a], rgb[a + 1], rgb[a + 2]);
            let p1 = rgb_to_yuv(rgb[a + 3], rgb[a + 4], rgb[a + 5]);
            let o = dst + pair * 4;
            out[o] = p0[0];
            out[o + 1] = ((p0[1] as u16 + p1[1] as u16) / 2) as u8;
            out[o + 2] = p1[0];
            out[o + 3] = ((p0[2] as u16 + p1[2] as u16) / 2) as u8;
        }
    }
}
```

Add `pub mod convert;` to `src/lib.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib convert`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/convert.rs src/lib.rs
git commit -m "feat: add BT.601 YUYV/RGB conversion"
```

---

### Task 5: Model inference wrapper with the golden correctness test

The golden test here is the highest-value test in the suite: it proves the graph surgery from Task 2 did not change what the model computes, by comparing the frozen model against the untouched original.

**Files:**
- Create: `src/model.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `prepare::{cache_dir, frozen_model_path, set_migraphx_cache_env}`.
- Produces:
  - `model::Matting::load(frozen: &Path, width: u32, height: u32) -> Result<Matting>`
  - `model::Matting::infer(&mut self, rgb_nchw: &[f32]) -> Result<(&[f32], &[f32])>` returning `(fgr, pha)`
  - `model::Matting::reset_state(&mut self) -> Result<()>`

- [ ] **Step 1: Write the failing golden test**

Create `src/model.rs` containing only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const R50_SOURCE: &str = concat!(
        env!("HOME"),
        "/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx"
    );

    fn deterministic_frame(width: usize, height: usize) -> Vec<f32> {
        let plane = width * height;
        let mut v = vec![0f32; plane * 3];
        for i in 0..plane {
            v[i] = ((i % 255) as f32) / 255.0;
            v[plane + i] = (((i / 3) % 255) as f32) / 255.0;
            v[2 * plane + i] = (((i / 7) % 255) as f32) / 255.0;
        }
        v
    }

    /// The frozen graph must compute what the stock graph computes. Anything
    /// else means the surgery in `freeze` corrupted the model.
    #[test]
    fn frozen_model_matches_original_within_tolerance() {
        let frozen = crate::prepare::frozen_model_path(&crate::prepare::cache_dir());
        if !frozen.exists() || !std::path::Path::new(R50_SOURCE).exists() {
            eprintln!("skipping: run `matting prepare` first");
            return;
        }
        // Without this the MIGraphX EP aborts session init on an empty cache path.
        crate::prepare::set_migraphx_cache_env(&crate::prepare::cache_dir()).unwrap();
        let (w, h) = (1024usize, 576usize);
        let frame = deterministic_frame(w, h);

        let mut frozen_model = Matting::load(&frozen, w as u32, h as u32).unwrap();
        let (_, frozen_pha) = frozen_model.infer(&frame).unwrap();
        let frozen_pha = frozen_pha.to_vec();

        let reference = reference_pha(R50_SOURCE, &frame, w, h, 0.5);

        assert_eq!(frozen_pha.len(), reference.len());
        let mut worst = 0f32;
        for (a, b) in frozen_pha.iter().zip(&reference) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 0.02, "frozen model diverges from original, worst delta {worst}");
    }

    #[test]
    fn state_persists_across_frames_and_resets() {
        let frozen = crate::prepare::frozen_model_path(&crate::prepare::cache_dir());
        if !frozen.exists() {
            eprintln!("skipping: run `matting prepare` first");
            return;
        }
        crate::prepare::set_migraphx_cache_env(&crate::prepare::cache_dir()).unwrap();
        let (w, h) = (1024usize, 576usize);
        let frame = deterministic_frame(w, h);
        let mut m = Matting::load(&frozen, w as u32, h as u32).unwrap();

        let first = m.infer(&frame).unwrap().1.to_vec();
        let second = m.infer(&frame).unwrap().1.to_vec();
        assert_ne!(first, second, "recurrent state should change the second result");

        m.reset_state().unwrap();
        let after_reset = m.infer(&frame).unwrap().1.to_vec();
        assert_eq!(first, after_reset, "reset_state should reproduce the first frame");
    }
}
```

- [ ] **Step 2: Write the reference helper**

Still in `src/model.rs`'s test module, add:

```rust
    /// Runs the *unmodified* RVM export on CPU to produce ground truth.
    fn reference_pha(path: &str, frame: &[f32], w: usize, h: usize, ratio: f32) -> Vec<f32> {
        use ort::{session::Session, value::Tensor};

        let mut session = Session::builder().unwrap().commit_from_file(path).unwrap();
        let inputs = vec![
            ("src".into(), Tensor::from_array((vec![1i64, 3, h as i64, w as i64], frame.to_vec())).unwrap().into()),
            ("r1i".into(), Tensor::from_array((vec![1i64, 1, 1, 1], vec![0f32])).unwrap().into()),
            ("r2i".into(), Tensor::from_array((vec![1i64, 1, 1, 1], vec![0f32])).unwrap().into()),
            ("r3i".into(), Tensor::from_array((vec![1i64, 1, 1, 1], vec![0f32])).unwrap().into()),
            ("r4i".into(), Tensor::from_array((vec![1i64, 1, 1, 1], vec![0f32])).unwrap().into()),
            ("downsample_ratio".into(), Tensor::from_array((vec![1i64], vec![ratio])).unwrap().into()),
        ];
        let outputs = session.run(inputs).unwrap();
        let (_, data) = outputs["pha"].try_extract_tensor::<f32>().unwrap();
        data.to_vec()
    }
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --lib model`
Expected: FAIL — `Matting` is not defined.

- [ ] **Step 4: Implement `src/model.rs`**

```rust
use std::path::Path;

use anyhow::{Context, Result};
use ort::{
    ep,
    session::{Session, SessionInputValue},
    value::{DynValue, Tensor, ValueType},
};

/// Holds the ORT session plus RVM's four recurrent state tensors.
///
/// State is moved in and out by ownership rather than copied back to host
/// memory: `SessionOutputs::remove` yields an owned `DynValue` that stays on
/// the GPU, so the recurrence costs nothing per frame.
pub struct Matting {
    session: Session,
    state: Option<[DynValue; 4]>,
    state_shapes: [Vec<i64>; 4],
    width: u32,
    height: u32,
    fgr: Vec<f32>,
    pha: Vec<f32>,
}

impl Matting {
    pub fn load(frozen: &Path, width: u32, height: u32) -> Result<Matting> {
        ort::init().commit();
        let session = Session::builder()?
            .with_execution_providers([ep::MIGraphX::default().with_fp16(true).build()])?
            .commit_from_file(frozen)
            .with_context(|| format!("loading {}", frozen.display()))?;

        // The frozen graph pins every state shape; read them back so we can
        // build correctly sized zero tensors for the first frame and on reset.
        let mut state_shapes: [Vec<i64>; 4] = Default::default();
        for (i, slot) in state_shapes.iter_mut().enumerate() {
            let name = format!("r{}i", i + 1);
            let input = session
                .inputs()
                .iter()
                .find(|inp| inp.name() == name)
                .with_context(|| format!("frozen model has no input {name}"))?;
            let ValueType::Tensor { shape, .. } = input.dtype() else {
                anyhow::bail!("{name} is not a tensor");
            };
            *slot = shape.to_vec();
        }

        let plane = (width * height) as usize;
        Ok(Matting {
            session,
            state: None,
            state_shapes,
            width,
            height,
            fgr: vec![0.0; plane * 3],
            pha: vec![0.0; plane],
        })
    }

    fn zero_state(&self) -> Result<[DynValue; 4]> {
        let mut built = Vec::with_capacity(4);
        for shape in &self.state_shapes {
            let n: i64 = shape.iter().product();
            built.push(Tensor::from_array((shape.clone(), vec![0f32; n as usize]))?.into_dyn());
        }
        Ok(built.try_into().map_err(|_| anyhow::anyhow!("state arity"))?)
    }

    /// Drop the temporal memory. Call on stream reconnect so a new scene does
    /// not inherit the previous one's alpha.
    pub fn reset_state(&mut self) -> Result<()> {
        self.state = None;
        Ok(())
    }

    /// `rgb_nchw` must be `3 * width * height` normalized to [0,1].
    /// Returns `(fgr, pha)` borrowed from internal buffers.
    pub fn infer(&mut self, rgb_nchw: &[f32]) -> Result<(&[f32], &[f32])> {
        let state = match self.state.take() {
            Some(s) => s,
            None => self.zero_state()?,
        };
        let [r1, r2, r3, r4] = state;

        let src = Tensor::from_array((
            vec![1i64, 3, self.height as i64, self.width as i64],
            rgb_nchw.to_vec(),
        ))?;

        let inputs: Vec<(std::borrow::Cow<str>, SessionInputValue)> = vec![
            ("src".into(), src.into()),
            ("r1i".into(), SessionInputValue::from(r1)),
            ("r2i".into(), SessionInputValue::from(r2)),
            ("r3i".into(), SessionInputValue::from(r3)),
            ("r4i".into(), SessionInputValue::from(r4)),
        ];

        let mut outputs = self.session.run(inputs)?;

        let (_, fgr) = outputs["fgr"].try_extract_tensor::<f32>()?;
        self.fgr.copy_from_slice(fgr);
        let (_, pha) = outputs["pha"].try_extract_tensor::<f32>()?;
        self.pha.copy_from_slice(pha);

        let next = [
            outputs.remove("r1o").context("missing r1o")?,
            outputs.remove("r2o").context("missing r2o")?,
            outputs.remove("r3o").context("missing r3o")?,
            outputs.remove("r4o").context("missing r4o")?,
        ];
        drop(outputs);
        self.state = Some(next);

        Ok((&self.fgr, &self.pha))
    }
}
```

Add `pub mod model;` to `src/lib.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib model -- --test-threads=1`
Expected: PASS, 2 tests. Single-threaded because both tests build MIGraphX sessions against the same cache.

If `frozen_model_matches_original_within_tolerance` fails, the Task 2 surgery is wrong — fix `freeze.rs`, do not loosen the tolerance.

- [ ] **Step 6: Commit**

```bash
git add src/model.rs src/lib.rs
git commit -m "feat: add RVM inference wrapper with zero-copy recurrent state"
```

---

### Task 6: Compositing and background images

**Files:**
- Create: `src/composite.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `cli::{Mode, Fit}`.
- Produces:
  - `composite::Background::load(path: &str, width: u32, height: u32, fit: Fit) -> Result<Background>` with `fn rgb(&self) -> &[u8]`
  - `composite::to_bgra(fgr: &[f32], pha: &[f32], plane: usize, out: &mut [u8])`
  - `composite::over_color(fgr: &[f32], pha: &[f32], plane: usize, colour: [u8; 3], out: &mut [u8])`
  - `composite::over_image(fgr: &[f32], pha: &[f32], plane: usize, bg: &[u8], out: &mut [u8])`

- [ ] **Step 1: Write the failing tests**

Create `src/composite.rs` containing only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // One pixel, fully opaque red foreground.
    fn red_fg() -> (Vec<f32>, usize) {
        (vec![1.0, 0.0, 0.0], 1)
    }

    #[test]
    fn alpha_mode_writes_bgra_with_alpha() {
        let (fgr, plane) = red_fg();
        let mut out = vec![0u8; 4];
        to_bgra(&fgr, &[1.0], plane, &mut out);
        assert_eq!(out, vec![0, 0, 255, 255], "expected B,G,R,A");

        to_bgra(&fgr, &[0.0], plane, &mut out);
        assert_eq!(out[3], 0, "transparent pixel must have alpha 0");
    }

    #[test]
    fn greenscreen_shows_key_colour_where_transparent() {
        let (fgr, plane) = red_fg();
        let mut out = vec![0u8; 3];
        over_color(&fgr, &[0.0], plane, [0, 255, 0], &mut out);
        assert_eq!(out, vec![0, 255, 0], "fully transparent should be pure key");

        over_color(&fgr, &[1.0], plane, [0, 255, 0], &mut out);
        assert_eq!(out, vec![255, 0, 0], "fully opaque should be pure foreground");
    }

    #[test]
    fn greenscreen_blends_at_half_alpha() {
        let (fgr, plane) = red_fg();
        let mut out = vec![0u8; 3];
        over_color(&fgr, &[0.5], plane, [0, 255, 0], &mut out);
        assert!((out[0] as i32 - 128).abs() <= 1, "r={}", out[0]);
        assert!((out[1] as i32 - 128).abs() <= 1, "g={}", out[1]);
        assert_eq!(out[2], 0);
    }

    #[test]
    fn image_mode_shows_background_where_transparent() {
        let (fgr, plane) = red_fg();
        let bg = [10u8, 20, 30];
        let mut out = vec![0u8; 3];
        over_image(&fgr, &[0.0], plane, &bg, &mut out);
        assert_eq!(out, vec![10, 20, 30]);

        over_image(&fgr, &[1.0], plane, &bg, &mut out);
        assert_eq!(out, vec![255, 0, 0]);
    }

    #[test]
    fn cover_fit_matches_target_dimensions() {
        let dir = std::env::temp_dir().join(format!("matting-bg-{}.png", std::process::id()));
        image::RgbImage::from_pixel(64, 64, image::Rgb([1, 2, 3])).save(&dir).unwrap();
        let bg = Background::load(dir.to_str().unwrap(), 1024, 576, crate::cli::Fit::Cover).unwrap();
        assert_eq!(bg.rgb().len(), 1024 * 576 * 3);
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn contain_fit_matches_target_dimensions() {
        let dir = std::env::temp_dir().join(format!("matting-bg2-{}.png", std::process::id()));
        image::RgbImage::from_pixel(64, 32, image::Rgb([4, 5, 6])).save(&dir).unwrap();
        let bg = Background::load(dir.to_str().unwrap(), 1024, 576, crate::cli::Fit::Contain).unwrap();
        assert_eq!(bg.rgb().len(), 1024 * 576 * 3);
        std::fs::remove_file(&dir).ok();
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib composite`
Expected: FAIL — `to_bgra`, `over_color`, `over_image`, `Background` are not defined.

- [ ] **Step 3: Implement `src/composite.rs`**

```rust
use anyhow::{Context, Result};
use image::imageops::FilterType;

use crate::cli::Fit;

#[inline]
fn to_u8(v: f32) -> u8 {
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Planar float foreground + alpha -> interleaved BGRA8, preserving alpha.
/// This is the only mode that carries true transparency.
pub fn to_bgra(fgr: &[f32], pha: &[f32], plane: usize, out: &mut [u8]) {
    debug_assert_eq!(out.len(), plane * 4);
    for i in 0..plane {
        out[i * 4] = to_u8(fgr[2 * plane + i]);
        out[i * 4 + 1] = to_u8(fgr[plane + i]);
        out[i * 4 + 2] = to_u8(fgr[i]);
        out[i * 4 + 3] = to_u8(pha[i]);
    }
}

/// `out = fgr*pha + colour*(1-pha)`, interleaved RGB8.
pub fn over_color(fgr: &[f32], pha: &[f32], plane: usize, colour: [u8; 3], out: &mut [u8]) {
    debug_assert_eq!(out.len(), plane * 3);
    for i in 0..plane {
        let a = pha[i];
        for c in 0..3 {
            let f = fgr[c * plane + i];
            out[i * 3 + c] = to_u8(f * a + (colour[c] as f32 / 255.0) * (1.0 - a));
        }
    }
}

/// `out = fgr*pha + bg*(1-pha)`, interleaved RGB8. `bg` is interleaved RGB8.
pub fn over_image(fgr: &[f32], pha: &[f32], plane: usize, bg: &[u8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), plane * 3);
    debug_assert_eq!(bg.len(), plane * 3);
    for i in 0..plane {
        let a = pha[i];
        for c in 0..3 {
            let f = fgr[c * plane + i];
            out[i * 3 + c] = to_u8(f * a + (bg[i * 3 + c] as f32 / 255.0) * (1.0 - a));
        }
    }
}

/// A background image pre-resized to the capture resolution, so the per-frame
/// cost is only the blend.
pub struct Background {
    rgb: Vec<u8>,
}

impl Background {
    pub fn load(path: &str, width: u32, height: u32, fit: Fit) -> Result<Background> {
        let img = image::open(path).with_context(|| format!("opening background {path}"))?.to_rgb8();
        let out = match fit {
            Fit::Stretch => image::imageops::resize(&img, width, height, FilterType::CatmullRom),
            Fit::Cover => {
                let scale = (width as f32 / img.width() as f32)
                    .max(height as f32 / img.height() as f32);
                let (sw, sh) = (
                    (img.width() as f32 * scale).ceil() as u32,
                    (img.height() as f32 * scale).ceil() as u32,
                );
                let scaled = image::imageops::resize(&img, sw, sh, FilterType::CatmullRom);
                image::imageops::crop_imm(&scaled, (sw - width) / 2, (sh - height) / 2, width, height)
                    .to_image()
            }
            Fit::Contain => {
                let scale = (width as f32 / img.width() as f32)
                    .min(height as f32 / img.height() as f32);
                let (sw, sh) = (
                    ((img.width() as f32 * scale) as u32).max(1),
                    ((img.height() as f32 * scale) as u32).max(1),
                );
                let scaled = image::imageops::resize(&img, sw, sh, FilterType::CatmullRom);
                let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
                image::imageops::overlay(
                    &mut canvas,
                    &scaled,
                    ((width - sw) / 2) as i64,
                    ((height - sh) / 2) as i64,
                );
                canvas
            }
        };
        Ok(Background { rgb: out.into_raw() })
    }

    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }
}
```

Add `pub mod composite;` to `src/lib.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib composite`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/composite.rs src/lib.rs
git commit -m "feat: add alpha, greenscreen and image compositing"
```

---

### Task 7: V4L2 capture source

**Files:**
- Create: `src/capture.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `capture::Source` trait with `fn next_frame(&mut self) -> Result<&[u8]>` and `fn dimensions(&self) -> (u32, u32)`
  - `capture::V4lSource::open(path: &str, width: u32, height: u32) -> Result<V4lSource>`
  - `capture::TestSource::new(width: u32, height: u32, frames: Vec<Vec<u8>>) -> TestSource`

- [ ] **Step 1: Write the failing tests**

Create `src/capture.rs` containing only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_cycles_frames() {
        let a = vec![1u8; 8];
        let b = vec![2u8; 8];
        let mut src = TestSource::new(2, 2, vec![a.clone(), b.clone()]);
        assert_eq!(src.dimensions(), (2, 2));
        assert_eq!(src.next_frame().unwrap(), &a[..]);
        assert_eq!(src.next_frame().unwrap(), &b[..]);
        assert_eq!(src.next_frame().unwrap(), &a[..], "should wrap around");
    }

    #[test]
    fn opening_a_missing_device_is_an_error() {
        assert!(V4lSource::open("/dev/definitely-not-a-camera", 1024, 576).is_err());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib capture`
Expected: FAIL — `TestSource` and `V4lSource` are not defined.

- [ ] **Step 3: Implement `src/capture.rs`**

```rust
use anyhow::{anyhow, Context, Result};
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::FourCC;

/// A source of YUYV 4:2:2 frames. Abstracted so the pipeline can be tested
/// without a camera.
pub trait Source {
    fn next_frame(&mut self) -> Result<&[u8]>;
    fn dimensions(&self) -> (u32, u32);
}

pub struct V4lSource {
    stream: MmapStream<'static>,
    width: u32,
    height: u32,
    buffer: Vec<u8>,
}

impl V4lSource {
    pub fn open(path: &str, width: u32, height: u32) -> Result<V4lSource> {
        let dev = Device::with_path(path).with_context(|| format!("opening {path}"))?;

        let mut fmt = Capture::format(&dev).context("querying capture format")?;
        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = FourCC::new(b"YUYV");
        let fmt = Capture::set_format(&dev, &fmt).context("setting capture format")?;

        if fmt.width != width || fmt.height != height || fmt.fourcc != FourCC::new(b"YUYV") {
            return Err(anyhow!(
                "{path} would not accept YUYV {width}x{height}; it offered {} {}x{}",
                fmt.fourcc, fmt.width, fmt.height
            ));
        }

        let stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4)
            .context("allocating capture buffers")?;

        Ok(V4lSource {
            stream,
            width,
            height,
            buffer: vec![0u8; (width * height * 2) as usize],
        })
    }
}

impl Source for V4lSource {
    fn next_frame(&mut self) -> Result<&[u8]> {
        let (buf, _meta) = CaptureStream::next(&mut self.stream).context("capturing frame")?;
        let n = self.buffer.len().min(buf.len());
        self.buffer[..n].copy_from_slice(&buf[..n]);
        Ok(&self.buffer)
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Deterministic in-memory source for tests. Cycles through its frames.
pub struct TestSource {
    width: u32,
    height: u32,
    frames: Vec<Vec<u8>>,
    index: usize,
}

impl TestSource {
    pub fn new(width: u32, height: u32, frames: Vec<Vec<u8>>) -> TestSource {
        TestSource { width, height, frames, index: 0 }
    }
}

impl Source for TestSource {
    fn next_frame(&mut self) -> Result<&[u8]> {
        if self.frames.is_empty() {
            return Err(anyhow!("TestSource has no frames"));
        }
        let i = self.index % self.frames.len();
        self.index += 1;
        Ok(&self.frames[i])
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
```

Add `pub mod capture;` to `src/lib.rs`.

Note: `MmapStream::with_buffers` borrows the device. Storing both in one struct is a self-reference. Resolve it by leaking the device — a process-lifetime camera handle is acceptable here:

```rust
let dev: &'static Device = Box::leak(Box::new(dev));
```
Place that immediately after the format checks and before `MmapStream::with_buffers(dev, ...)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib capture`
Expected: PASS, 2 tests.

- [ ] **Step 5: Verify against the real camera**

Run: `v4l2-ctl -d /dev/video0 --list-formats-ext`
Expected: confirms YUYV 1024×576 is offered, matching what `V4lSource::open` requests.

- [ ] **Step 6: Commit**

```bash
git add src/capture.rs src/lib.rs
git commit -m "feat: add V4L2 capture source behind a testable trait"
```

---

### Task 8: V4L2 loopback sink

**Files:**
- Create: `src/sink.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `cli::Mode`.
- Produces:
  - `sink::Sink` trait with `fn write_frame(&mut self, data: &[u8]) -> Result<()>`
  - `sink::fourcc_for(mode: Mode) -> FourCC` and `sink::bytes_per_pixel(mode: Mode) -> usize`
  - `sink::V4lSink::open(path: &str, width: u32, height: u32, mode: Mode) -> Result<V4lSink>`
  - `sink::TestSink::new() -> TestSink` with `fn frames(&self) -> &[Vec<u8>]`

- [ ] **Step 1: Write the failing tests**

Create `src/sink.rs` containing only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Mode;

    #[test]
    fn alpha_mode_uses_bgra_fourcc() {
        assert_eq!(fourcc_for(Mode::Alpha), v4l::FourCC::new(b"AR24"));
        assert_eq!(bytes_per_pixel(Mode::Alpha), 4);
    }

    #[test]
    fn opaque_modes_use_yuyv() {
        for mode in [Mode::Greenscreen, Mode::Image] {
            assert_eq!(fourcc_for(mode), v4l::FourCC::new(b"YUYV"));
            assert_eq!(bytes_per_pixel(mode), 2);
        }
    }

    #[test]
    fn test_sink_records_frames() {
        let mut sink = TestSink::new();
        sink.write_frame(&[1, 2, 3]).unwrap();
        sink.write_frame(&[4, 5, 6]).unwrap();
        assert_eq!(sink.frames(), &[vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn opening_a_missing_device_is_an_error() {
        assert!(V4lSink::open("/dev/definitely-not-a-loopback", 1024, 576, Mode::Alpha).is_err());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib sink`
Expected: FAIL — `fourcc_for`, `bytes_per_pixel`, `TestSink`, `V4lSink` are not defined.

- [ ] **Step 3: Implement `src/sink.rs`**

```rust
use anyhow::{anyhow, Context, Result};
use v4l::buffer::Type;
use v4l::io::traits::OutputStream;
use v4l::prelude::*;
use v4l::video::Output;
use v4l::FourCC;

use crate::cli::Mode;

pub trait Sink {
    fn write_frame(&mut self, data: &[u8]) -> Result<()>;
}

/// Alpha needs a format that carries it, which only OBS consumes. Everything
/// else emits YUYV so browsers, Zoom and Discord work.
pub fn fourcc_for(mode: Mode) -> FourCC {
    match mode {
        Mode::Alpha => FourCC::new(b"AR24"),
        Mode::Greenscreen | Mode::Image => FourCC::new(b"YUYV"),
    }
}

pub fn bytes_per_pixel(mode: Mode) -> usize {
    match mode {
        Mode::Alpha => 4,
        Mode::Greenscreen | Mode::Image => 2,
    }
}

pub struct V4lSink {
    stream: MmapStream<'static>,
    frame_len: usize,
}

impl V4lSink {
    pub fn open(path: &str, width: u32, height: u32, mode: Mode) -> Result<V4lSink> {
        let dev = Device::with_path(path).with_context(|| {
            format!(
                "opening {path}. If it does not exist, create one with:\n  \
                 sudo modprobe v4l2loopback devices=1 video_nr=9 card_label=Matting exclusive_caps=1"
            )
        })?;

        let mut fmt = Output::format(&dev).context("querying output format")?;
        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = fourcc_for(mode);
        // v4l2loopback only sets its buffer length once a format is applied,
        // so this call must happen before allocating buffers.
        let fmt = Output::set_format(&dev, &fmt).context("setting output format")?;

        if fmt.width != width || fmt.height != height || fmt.fourcc != fourcc_for(mode) {
            return Err(anyhow!(
                "{path} would not accept {} {width}x{height}; it chose {} {}x{}",
                fourcc_for(mode), fmt.fourcc, fmt.width, fmt.height
            ));
        }

        let dev: &'static Device = Box::leak(Box::new(dev));
        let stream = MmapStream::with_buffers(dev, Type::VideoOutput, 4)
            .context("allocating output buffers")?;

        Ok(V4lSink {
            stream,
            frame_len: width as usize * height as usize * bytes_per_pixel(mode),
        })
    }
}

impl Sink for V4lSink {
    fn write_frame(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.frame_len {
            return Err(anyhow!("expected {} bytes, got {}", self.frame_len, data.len()));
        }
        let (buf, meta) = OutputStream::next(&mut self.stream).context("dequeuing output buffer")?;
        buf[..data.len()].copy_from_slice(data);
        meta.field = 0;
        meta.bytesused = data.len() as u32;
        Ok(())
    }
}

/// Records frames in memory so the pipeline can be tested without a device.
pub struct TestSink {
    frames: Vec<Vec<u8>>,
}

impl TestSink {
    pub fn new() -> TestSink {
        TestSink { frames: Vec::new() }
    }

    pub fn frames(&self) -> &[Vec<u8>] {
        &self.frames
    }
}

impl Default for TestSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for TestSink {
    fn write_frame(&mut self, data: &[u8]) -> Result<()> {
        self.frames.push(data.to_vec());
        Ok(())
    }
}
```

Add `pub mod sink;` to `src/lib.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib sink`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/sink.rs src/lib.rs
git commit -m "feat: add v4l2loopback sink with mode-dependent pixel format"
```

---

### Task 9: Pipeline wiring and the `run` command

**Files:**
- Create: `src/pipeline.rs`
- Modify: `src/lib.rs`, `src/main.rs`
- Create: `README.md`

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces:
  - `pipeline::Pipeline::new(model: Matting, mode: Mode, colour: [u8;3], background: Option<Background>, width: u32, height: u32) -> Pipeline`
  - `pipeline::Pipeline::process(&mut self, yuyv: &[u8]) -> Result<&[u8]>`
  - `pipeline::run(args: &RunArgs) -> Result<()>`

- [ ] **Step 1: Write the failing integration test**

Create `src/pipeline.rs` containing only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{Source, TestSource};
    use crate::cli::Mode;
    use crate::sink::{Sink, TestSink};

    fn prepared_model() -> Option<crate::model::Matting> {
        let frozen = crate::prepare::frozen_model_path(&crate::prepare::cache_dir());
        if !frozen.exists() {
            eprintln!("skipping: run `matting prepare` first");
            return None;
        }
        crate::prepare::set_migraphx_cache_env(&crate::prepare::cache_dir()).ok()?;
        crate::model::Matting::load(&frozen, 1024, 576).ok()
    }

    #[test]
    fn greenscreen_pipeline_emits_correctly_sized_yuyv() {
        let Some(model) = prepared_model() else { return };
        let (w, h) = (1024u32, 576u32);
        let frame = vec![128u8; (w * h * 2) as usize];
        let mut source = TestSource::new(w, h, vec![frame]);
        let mut sink = TestSink::new();
        let mut pipe = Pipeline::new(model, Mode::Greenscreen, [0, 255, 0], None, w, h);

        let input = source.next_frame().unwrap().to_vec();
        let out = pipe.process(&input).unwrap();
        sink.write_frame(out).unwrap();

        assert_eq!(sink.frames().len(), 1);
        assert_eq!(sink.frames()[0].len(), (w * h * 2) as usize);
    }

    #[test]
    fn alpha_pipeline_emits_correctly_sized_bgra() {
        let Some(model) = prepared_model() else { return };
        let (w, h) = (1024u32, 576u32);
        let frame = vec![128u8; (w * h * 2) as usize];
        let mut pipe = Pipeline::new(model, Mode::Alpha, [0, 255, 0], None, w, h);
        let out = pipe.process(&frame).unwrap();
        assert_eq!(out.len(), (w * h * 4) as usize);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib pipeline`
Expected: FAIL — `Pipeline` is not defined.

- [ ] **Step 3: Implement `src/pipeline.rs`**

```rust
use std::path::Path;

use anyhow::{Context, Result};

use crate::capture::{Source, V4lSource};
use crate::cli::{Mode, RunArgs};
use crate::composite::{over_color, over_image, to_bgra, Background};
use crate::convert::{rgb_to_yuyv, yuyv_to_rgb_f32_nchw};
use crate::manifest::Manifest;
use crate::model::Matting;
use crate::prepare;
use crate::sink::{bytes_per_pixel, Sink, V4lSink};

pub struct Pipeline {
    model: Matting,
    mode: Mode,
    colour: [u8; 3],
    background: Option<Background>,
    width: usize,
    height: usize,
    rgb_in: Vec<f32>,
    rgb_out: Vec<u8>,
    out: Vec<u8>,
}

impl Pipeline {
    pub fn new(
        model: Matting,
        mode: Mode,
        colour: [u8; 3],
        background: Option<Background>,
        width: u32,
        height: u32,
    ) -> Pipeline {
        let (w, h) = (width as usize, height as usize);
        let plane = w * h;
        Pipeline {
            model,
            mode,
            colour,
            background,
            width: w,
            height: h,
            rgb_in: vec![0.0; plane * 3],
            rgb_out: vec![0; plane * 3],
            out: vec![0; plane * bytes_per_pixel(mode)],
        }
    }

    /// YUYV in, mode-appropriate bytes out.
    pub fn process(&mut self, yuyv: &[u8]) -> Result<&[u8]> {
        let plane = self.width * self.height;
        yuyv_to_rgb_f32_nchw(yuyv, self.width, self.height, &mut self.rgb_in);
        let (fgr, pha) = self.model.infer(&self.rgb_in)?;

        match self.mode {
            Mode::Alpha => to_bgra(fgr, pha, plane, &mut self.out),
            Mode::Greenscreen => {
                over_color(fgr, pha, plane, self.colour, &mut self.rgb_out);
                rgb_to_yuyv(&self.rgb_out, self.width, self.height, &mut self.out);
            }
            Mode::Image => {
                let bg = self.background.as_ref().context("image mode without a background")?;
                over_image(fgr, pha, plane, bg.rgb(), &mut self.rgb_out);
                rgb_to_yuyv(&self.rgb_out, self.width, self.height, &mut self.out);
            }
        }
        Ok(&self.out)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.model.reset_state()
    }
}

pub fn run(args: &RunArgs) -> Result<()> {
    args.validate().map_err(anyhow::Error::msg)?;
    let colour = crate::cli::parse_color(&args.color).map_err(anyhow::Error::msg)?;

    let dir = prepare::cache_dir();
    let cached = Manifest::load(&dir)?;
    let needed = Manifest {
        source_sha256: cached.source_sha256.clone(),
        backbone: format!("{:?}", args.model).to_lowercase(),
        width: cached.width,
        height: cached.height,
        ratio: cached.ratio,
        gpu_arch: prepare::gpu_arch(),
        rocm_version: prepare::rocm_version(),
    };
    cached.check_matches(&needed)?;

    let (width, height) = (cached.width, cached.height);
    prepare::set_migraphx_cache_env(&dir)?;

    let frozen = prepare::frozen_model_path(&dir);
    if !Path::new(&frozen).exists() {
        anyhow::bail!("no frozen model at {}; run `matting prepare`", frozen.display());
    }
    let model = Matting::load(&frozen, width, height)?;

    let background = match args.mode {
        Mode::Image => Some(Background::load(
            args.image.as_ref().expect("validated above"),
            width,
            height,
            args.fit,
        )?),
        _ => None,
    };

    let mut source = V4lSource::open(&args.input, width, height)?;
    let mut sink = V4lSink::open(&args.output, width, height, args.mode)?;
    let mut pipeline = Pipeline::new(model, args.mode, colour, background, width, height);

    println!(
        "Streaming {}x{} {:?} from {} to {} as {}",
        width, height, args.mode, args.input, args.output,
        crate::sink::fourcc_for(args.mode)
    );
    if args.mode == Mode::Alpha {
        println!(
            "Note: alpha output is only displayed correctly by OBS. \
             Browsers, Zoom and Discord need --mode greenscreen or --mode image."
        );
    }

    let mut failures = 0u32;
    loop {
        let frame = match source.next_frame() {
            Ok(f) => {
                failures = 0;
                f.to_vec()
            }
            Err(e) => {
                failures += 1;
                if failures > 50 {
                    return Err(e).context("capture failed repeatedly; giving up");
                }
                eprintln!("capture error ({failures}/50), retrying: {e}");
                std::thread::sleep(std::time::Duration::from_millis(100));
                // A dropped stream means a new scene; do not inherit old alpha.
                pipeline.reset()?;
                continue;
            }
        };
        let out = pipeline.process(&frame)?;
        sink.write_frame(out)?;
    }
}
```

Add `pub mod pipeline;` to `src/lib.rs`, and wire `src/main.rs`:

```rust
Command::Run(args) => matting::pipeline::run(&args),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib pipeline -- --test-threads=1`
Expected: PASS, 2 tests.

- [ ] **Step 5: Run the whole suite**

Run: `cargo test -- --test-threads=1`
Expected: all tests pass.

- [ ] **Step 6: Verify end to end against real devices**

```bash
sudo modprobe v4l2loopback devices=1 video_nr=9 card_label=Matting exclusive_caps=1
cargo run --release -- run --mode greenscreen --output /dev/video9
```
Expected: prints the streaming banner and runs without error. In another terminal confirm the device carries frames:
```bash
ffplay -f v4l2 -i /dev/video9
```
Expected: the webcam feed with a green background.

Then check alpha in OBS: `cargo run --release -- run --mode alpha --output /dev/video9`, add a V4L2 source in OBS pointing at `/dev/video9`, and confirm the background is transparent with no Chroma Key filter.

- [ ] **Step 7: Write `README.md`**

```markdown
# matting

Realtime webcam background matting on AMD GPUs, using Robust Video Matting
through ONNX Runtime's MIGraphX execution provider.

## Why this exists

Stock RVM exports cannot be compiled by MIGraphX: `downsample_ratio` is a
runtime graph input, which leaves the internal `Resize` operations
non-constant. Tools that bundle RVM therefore fall back to CPU on AMD
hardware. `matting prepare` rewrites the graph so the whole thing compiles,
which is worth roughly a 10x speedup:

| Model @1024x576 | CPU | MIGraphX GPU |
|---|---|---|
| RVM MobileNetV3 | 74 ms / 13.5 fps | 7.7 ms / 130 fps |
| RVM ResNet50 | 104 ms / 9.6 fps | 10.0 ms / 100 fps |

## Requirements

- AMD GPU with ROCm and MIGraphX (developed against gfx1151 / ROCm 7.2.4)
- `onnxruntime` built with the MIGraphX execution provider at
  `/usr/lib/libonnxruntime.so`
- `v4l2loopback`
- A stock RVM ONNX export from
  <https://github.com/PeterL1n/RobustVideoMatting/releases>

## Usage

Prepare once. This compiles the model and takes about two minutes:

```sh
matting prepare --from rvm_resnet50.onnx --model resnet50
```

Create an output device:

```sh
sudo modprobe v4l2loopback devices=1 video_nr=9 card_label=Matting exclusive_caps=1
```

Then run:

```sh
matting run --mode alpha                          # transparent, OBS only
matting run --mode greenscreen --color '#00FF00'  # works everywhere
matting run --mode image --image bg.jpg --fit cover
```

`--mode alpha` emits BGRA and is only rendered correctly by OBS. Use
`greenscreen` or `image` for browsers, Zoom and Discord.
```

- [ ] **Step 8: Commit**

```bash
git add src/pipeline.rs src/lib.rs src/main.rs README.md
git commit -m "feat: wire capture, inference, compositing and sink into run command"
```

---

## Notes for the implementer

- **If `run` is slow, check the EP actually engaged.** A MIGraphX time equal to
  a CPU time means silent fallback. Compare against `--mode greenscreen` timings
  in the table above; anything near 100 ms/frame is running on CPU.
- **Do not loosen the golden test tolerance in Task 5.** A divergence there
  means `freeze.rs` corrupted the graph, and every downstream result is
  meaningless.
- **`Box::leak` on the V4L2 devices is deliberate.** The stream borrows the
  device, and both live for the process lifetime; leaking avoids a
  self-referential struct for no practical cost.
- Reference material from the spike phase is in `bench-spike/` (working `ort` +
  MIGraphX setup) and `freeze_rvm.py` (the Python original of Task 2). Both are
  throwaway, not dependencies.
