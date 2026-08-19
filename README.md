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

Measured on a Radeon 8060S (gfx1151, Strix Halo) with ROCm 7.2.4.

## Requirements

- AMD GPU with ROCm and MIGraphX (developed against gfx1151 / ROCm 7.2.4)
- `onnxruntime` built with the MIGraphX execution provider at
  `/usr/lib/libonnxruntime.so`
- `v4l2loopback`
- A stock RVM ONNX export from
  <https://github.com/PeterL1n/RobustVideoMatting/releases>

## Usage

Prepare once. This freezes the graph and compiles it, taking a few minutes:

```sh
matting prepare --from rvm_resnet50.onnx --model resnet50
```

The result is cached in `~/.cache/matting`, so later runs start in about a
second. `prepare` reads the backbone's recurrent-state widths out of the model
and refuses a mismatched `--model`.

Create an output device. If `v4l2loopback` is already loaded for another
device, add one dynamically rather than reloading the module:

```sh
pkexec v4l2loopback-ctl add -n Matting /dev/video9
```

Otherwise load the module directly:

```sh
pkexec modprobe v4l2loopback devices=1 video_nr=9 card_label=Matting exclusive_caps=1
```

Then run:

```sh
matting run --mode alpha                          # transparent, OBS only
matting run --mode greenscreen --color '#00FF00'  # works everywhere
matting run --mode image --image bg.jpg --fit cover
```

`--mode alpha` emits BGRA and is only rendered correctly by OBS, where it
removes the need for a Chroma Key filter. Use `greenscreen` or `image` for
browsers, Zoom and Discord, which only accept YUYV.

`--fit` controls how a background image is mapped onto the capture resolution:
`cover` (default, scale to fill and centre-crop), `stretch`, or `contain`.

## Development

Tests need `ORT_DYLIB_PATH`, which `.cargo/config.toml` sets automatically:

```sh
cargo test --release -- --test-threads=1
```

Single-threaded because several tests build MIGraphX sessions against the same
compile cache. Tests that need a prepared model or a stock RVM export skip
themselves when those files are absent.

The most important test is
`model::tests::frozen_model_matches_original_within_tolerance`, which runs the
unmodified RVM export on CPU and compares its alpha against the frozen graph on
GPU. If graph surgery ever corrupts the model, that test is what catches it —
do not loosen its tolerance to make it pass.
