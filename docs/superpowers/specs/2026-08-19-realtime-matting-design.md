# Realtime webcam matting on AMD GPU — design

Date: 2026-08-19
Status: approved, ready for implementation planning

## Problem

Run realtime background removal on a 1024×576 webcam feed at ≥25 fps, on an AMD
Strix Halo iGPU, and publish the result to a virtual webcam.

Existing tools fail on this hardware. The OBS plugin `obs-ai-matting` links both
CUDA and MIGraphX execution providers, but MIGraphX rejects its model at load
time and it silently falls back to CPU, costing ~25% of the CPU. The
MediaPipe-based `linuxgreenscreen` runs on CPU for the same reason: its GPU
delegate needs an OpenGL ES path that is unavailable here.

## Hardware and platform

| Component | Value |
|---|---|
| GPU | AMD Radeon 8060S, gfx1151, RDNA 3.5, 40 CU |
| APU | Ryzen AI MAX+ 395 |
| VRAM carve-out | 512 MB (GTT 124 GB) |
| ROCm | 7.2.4, with MIGraphX 2.15 and MIOpen |
| ONNX Runtime | 1.28.0 (`onnxruntime-opt-rocm`), exposing `MIGraphXExecutionProvider` |
| Source device | `/dev/video0`, v4l2loopback, "Sony ILCE-6400 (PC Control)", YUYV 4:2:2, 1024×576 @ 30 fps |
| Rust | 1.97.1 |

## Findings that determine the design

All measured on the hardware above during a spike phase.

### The obvious model candidates are unusable

BRIA RMBG-2.0, PramaLLC BEN2, and BiRefNet_512x512 are all dichotomous *image*
segmentation models built on a Swin transformer backbone.

| Model | EP | ms/frame | fps |
|---|---|---|---|
| BiRefNet-512 fp16 | MIGraphX | 1757 | 0.6 |
| BiRefNet-512 fp16 | CPU | 1797 | 0.6 |
| BiRefNet-512 + onnxsim | MIGraphX | 1739 | 0.6 |
| BEN2 (1024²) | MIGraphX | 3081 | 0.3 |
| BEN2 (1024²) | CPU | 3121 | 0.3 |

MIGraphX and CPU times are identical: these models fall back to CPU entirely.
`migraphx-driver` fails to compile BiRefNet with
`MULTIBROADCAST: input dimensions should be > 0`, originating in the Swin
window-partition and relative-position-index operations. Running the graph
through `onnxsim` does not fix it. A control run of
`migraphx-driver perf --test --gpu` succeeds at 97k inferences/sec, confirming
the GPU stack itself is healthy.

RMBG-2.0 was not benchmarked (gated repo) but is the same BiRefNet SwinL
architecture at 1024², strictly heavier than the 512 variant already at 0.6 fps.

These models are also a poor fit on their merits: having no temporal memory,
their alpha edges flicker frame to frame around hair.

### Robust Video Matting works, once its graph is frozen

RVM is recurrent and purpose-built for video. Stock RVM is rejected by MIGraphX
with `PARSE_Resize: linear mode not supported for non-constant inputs`, because
`downsample_ratio` is a runtime graph input, which leaves the internal `Resize`
operations non-constant. **This is the root cause of the plugin's CPU
fallback.**

Demoting `downsample_ratio` to a constant initializer and pinning all input
shapes makes the whole graph compile:

| Model @1024×576, ratio 0.5 | CPU | MIGraphX GPU |
|---|---|---|
| RVM MobileNetV3 | 74 ms · 13.5 fps | **7.7 ms · 130 fps** |
| RVM ResNet50 | 104 ms · 9.6 fps | **10.0 ms · 100 fps** |

Roughly 10× speedup, and 4× headroom over the 25 fps target on the
higher-quality backbone.

Two operational details, both discovered the hard way:

1. **`ORT_MIGRAPHX_MODEL_CACHE_PATH` must name a real directory.** Unset, the EP
   aborts during session init trying to write `""/<hash>.mxr`. First compile
   takes ~120 s; a cached reload takes 0.8 s.
2. **Recurrent state channel widths differ per backbone** — MobileNetV3 is
   `[16,20,40,64]`, ResNet50 is `[16,32,64,128]`. Wrong values fail at load with
   an `Expand` shape-inference error.

`onnxsim` constant folding was verified **not** to be required: the unsimplified
frozen graph compiles and runs at 129.4 fps versus 129.7 fps simplified.

## Architecture

A single binary, `matting`, with two subcommands.

```
matting prepare  --from <rvm.onnx>
                 [--model resnet50|mobilenetv3] [--width 1024] [--height 576] [--ratio 0.5]
matting run      [--input /dev/video0] [--output /dev/video9]
                 [--mode alpha|greenscreen|image]
                 [--color '#00FF00'] [--image bg.png] [--fit cover|stretch|contain]
```

Defaults: `--mode alpha`, `--model resnet50`, `--ratio 0.5`, 1024×576.

### Frame path

```
/dev/video0 (YUYV 1024×576)
  → capture      mmap ring buffer, borrow frame
  → convert      YUYV 4:2:2 → RGB f32 NCHW, normalized [0,1]
  → model        RVM via ORT + MIGraphX; recurrent state persists across frames
  → composite    mode-dependent blend of fgr and pha
  → sink         v4l2loopback write
```

Single-threaded by design. The budget is 33 ms/frame at 30 fps against ~10 ms
inference and ~3–5 ms conversion. A multi-stage pipeline would buy headroom that
measurement says is not needed; add one only if profiling contradicts this.

### Modules

Each is independently testable behind a narrow interface.

- **`capture`** — opens a V4L2 device, negotiates YUYV at the requested
  geometry, manages mmap buffers, yields borrowed frames. Behind a `Source`
  trait so tests can substitute a synthetic generator.
- **`convert`** — YUYV 4:2:2 ↔ RGB, and float normalization to NCHW. The CPU hot
  path; tight loops, no allocation per frame.
- **`model`** — owns the ORT session and the four recurrent state tensors.
  Exposes `infer(&mut self, rgb: &[f32]) -> (fgr, pha)` and `reset_state()`.
  State is carried forward as ORT values, never round-tripped through
  `Vec<f32>`.
- **`prepare`** — ONNX graph surgery plus cache compilation. See below.
- **`composite`** — the three output modes.
- **`sink`** — v4l2loopback output, format chosen by mode. Behind a `Sink` trait.
- **`cli`** — `clap` argument parsing and validation.

## Output format varies by mode

The output device format is negotiated at startup from the selected mode.

| Mode | V4L2 format | Compatible with |
|---|---|---|
| `alpha` (default) | `ABGR32` (BGRA) | OBS only — true alpha, no Chroma Key filter needed |
| `greenscreen` | `YUYV` | everything |
| `image` | `YUYV` | everything |

`run` prints the negotiated format on startup and warns explicitly that `alpha`
will not display correctly in browsers, Zoom, or Discord. Without that warning
the mismatch presents as an inexplicable failure.

## Compositing

Given foreground `fgr` and alpha `pha` from the model:

- **alpha** — `RGB = fgr`, `A = pha`, passed through unmodified.
- **greenscreen** — `out = fgr·pha + color·(1−pha)`. `--color` accepts `#RRGGBB`
  or `r,g,b`; defaults to `#00FF00`.
- **image** — `out = fgr·pha + bg·(1−pha)`. The background is loaded once at
  startup, resized to the capture resolution, and cached in the output pixel
  format so per-frame cost is only the blend. `--fit` defaults to `cover` (scale
  to fill, centre-crop), with `stretch` and `contain` available.

## The `prepare` step

Implemented in Rust via `prost` against `onnx.proto`; no Python dependency.

The source model is supplied by the user via `--from`, pointing at a stock RVM
ONNX export from the upstream
[RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) releases.
`prepare` does not download anything. It infers the backbone from the model's
declared recurrent-state channel widths rather than trusting `--model`, and
errors if the two disagree. Locally available copies:
`~/Downloads/rvm_mobilenetv3_fp32.onnx` and
`~/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx`.

1. Remove `downsample_ratio` from `graph.input`; append it as an initializer
   holding the chosen ratio.
2. Pin `src` to `[1, 3, H, W]`.
3. Pin `r1i..r4i`. Channel widths are **read from the model**, never hardcoded:
   the dynamic graph declares them statically on its `r1o..r4o` outputs. Spatial
   dims are `ceil(downsampled / 2^i)` where `downsampled = (W·ratio, H·ratio)`.
4. Write the frozen `.onnx`.
5. Build a MIGraphX session against it to force compilation, populating the
   `.mxr` cache with `ORT_MIGRAPHX_MODEL_CACHE_PATH` set.
6. Write a manifest recording source model hash, resolution, ratio, ROCm
   version, and GPU architecture.

`run` validates that manifest on startup and fails with an actionable message
telling the user to run `matting prepare`. It must never silently proceed to CPU
— that is exactly the failure mode being fixed.

## Error handling

- Source device disappears mid-stream (observed: `/dev/video1` vanished during
  the spike) — detect `ENODEV`/`EIO`, attempt reconnect with backoff, reset
  recurrent state on reconnect.
- Output device missing — clear message naming the `v4l2loopback` modprobe
  invocation needed.
- Model cache missing or stale — fail fast, direct the user to `prepare`.
- Requested geometry unsupported by the source — report what the device does
  offer.

## Testing

- **Golden numerical test** — run the frozen model on a fixed input and assert
  `pha` matches a stored reference within tolerance. Graph surgery can silently
  corrupt a model and nothing else would catch it. This is the highest-value
  test in the suite.
- **`prepare` unit tests** — assert the resulting input and initializer sets and
  pinned dims; assert channel-width detection is correct for *both* backbones,
  since hardcoding them was a real bug during the spike.
- **Conversion round-trip** — YUYV → RGB → YUYV within tolerance.
- **Compositing** — each mode against known alpha values (0.0, 0.5, 1.0).
- **Pipeline integration** — synthetic `Source` and in-memory `Sink`, no
  hardware required.

## Risks

- **`v4l` crate output support.** The design assumes it handles `VIDEO_OUTPUT`
  for the loopback write. Validate first; fall back to raw ioctls via `nix`.
- **YUYV colour matrix.** BT.601 vs BT.709, limited vs full range. Getting this
  wrong shifts colours subtly. Verify against the actual A6400 feed rather than
  assuming.
- **`.mxr` cache is tied to GPU architecture and ROCm version**, hence the
  manifest check.

## Out of scope

Deliberately excluded to keep the first version focused: multi-camera support,
runtime resolution switching (the graph is frozen per-resolution by
construction), background blur, a GUI, and any model beyond the two RVM
backbones.

## Reference material retained

- `freeze_rvm.py` — the proven Python implementation of the graph surgery, kept
  as the reference for the Rust port.
- `bench-spike/` — throwaway harnesses containing working `ort` + MIGraphX setup
  code, including the cache-path handling.
- `models/rvm_{mnv3,r50}_1024x576_static.onnx` — verified frozen models, usable
  as golden-test fixtures.
