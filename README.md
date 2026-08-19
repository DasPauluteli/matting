# matting

Realtime webcam background removal on AMD GPUs.

Reads your webcam, runs [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)
on the GPU via ONNX Runtime's MIGraphX execution provider, and publishes the
result to a virtual camera that any application can use.

Three output modes:

| Mode | Output | Use it in |
|---|---|---|
| `alpha` | Transparent background (BGRA) | OBS — no Chroma Key filter needed |
| `greenscreen` | Solid colour background (YUYV) | Anything |
| `image` | Photo background (YUYV) | Anything |

## Why this exists

RVM's published ONNX exports take `downsample_ratio` as a *runtime input*. That
leaves the model's internal `Resize` operations non-constant, and MIGraphX
refuses to compile such a graph:

```
PARSE_Resize: linear mode not supported for non-constant inputs
```

When that happens inside ONNX Runtime, the MIGraphX provider rejects the graph
and execution silently falls back to the CPU — you get a working but slow
result, with no error explaining why.

`matting prepare` rewrites the graph before compiling it: `downsample_ratio`
becomes a constant, and every input shape is pinned. The whole model then
compiles and runs on the GPU.

## Requirements

- **Linux** with `v4l2loopback`
- **An AMD GPU supported by ROCm** (see below)
- **ROCm** with **MIGraphX**
- **ONNX Runtime built with the MIGraphX execution provider**
- **Rust** — built and tested with 1.97; older toolchains are untested
- An RVM ONNX export (see [Getting a model](#3-getting-a-model))

### Supported GPUs

This needs a GPU that ROCm supports, since everything runs through MIGraphX.

**Tested:** Radeon 8060S (`gfx1151`, Strix Halo) on ROCm 7.2.4.

**Expected to work** — these are the architectures the ROCm libraries on a
current install ship kernels for. None of them have been tested here, so treat
this as a starting point rather than a guarantee:

| Architecture | gfx targets | Examples |
|---|---|---|
| RDNA 2 | `gfx1030`–`gfx1036` | RX 6800/6900 XT, RX 6700 XT, Radeon PRO W6800 |
| RDNA 3 | `gfx1100`–`gfx1103` | RX 7900 XTX/XT, RX 7800 XT, RX 7600, PRO W7900 |
| RDNA 3.5 | `gfx1150`–`gfx1153` | Radeon 890M, Radeon 8050S/8060S (Strix, Strix Halo) |
| RDNA 4 | `gfx1200`, `gfx1201` | RX 9070 XT, RX 9060 |
| CDNA | `gfx908`, `gfx90a`, `gfx942`, `gfx950` | MI100, MI210/250, MI300, MI350 |

Check what you have:

```sh
rocminfo | grep -m1 'Name:.*gfx'
```

If your GPU is not supported by ROCm, this will not work — there is no CPU
fallback path, by design.

## Installing

### 1. System dependencies

You need ROCm, MIGraphX, and an ONNX Runtime built with the MIGraphX provider.
Package names differ by distribution.

**Arch / CachyOS:**

```sh
pkexec pacman -S rocm-hip-sdk migraphx v4l2loopback-dkms v4l-utils rust
# ONNX Runtime with the MIGraphX provider, from the AUR:
paru -S onnxruntime-opt-rocm
```

**Other distributions:** install ROCm following
[AMD's instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/),
then either install a prebuilt ONNX Runtime with MIGraphX or
[build one](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html).

Verify the provider is present:

```sh
ls /usr/lib/libonnxruntime_providers_migraphx.so
```

### 2. Build

```sh
git clone https://github.com/DasPauluteli/matting.git && cd matting
cargo build --release
```

The binary lands at `target/release/matting`.

`.cargo/config.toml` points the build at `/usr/lib/libonnxruntime.so`. If your
ONNX Runtime is somewhere else, edit that file or set `ORT_DYLIB_PATH`.

### 3. Getting a model

Download a stock export from the
[RVM releases page](https://github.com/PeterL1n/RobustVideoMatting/releases):

- `rvm_resnet50_fp32.onnx` — better quality, the default
- `rvm_mobilenetv3_fp32.onnx` — faster, lighter

Models are not redistributed here; RVM has its own licence.

## Usage

### Prepare the model (once)

```sh
matting prepare --from rvm_resnet50.onnx --model resnet50
```

This rewrites the graph and compiles it for your GPU. **The first run takes
several minutes.** The result is cached in `~/.cache/matting` and reused, so
later startups take a second or two.

Match `--width`/`--height` to your webcam:

```sh
matting prepare --from rvm_resnet50.onnx --width 1280 --height 720
```

Re-run `prepare` whenever you change resolution, ratio, or model, or after a
ROCm upgrade. `run` checks a manifest and tells you when the cache is stale
rather than quietly misbehaving.

### Create a virtual camera

If `v4l2loopback` is not loaded yet:

```sh
pkexec modprobe v4l2loopback devices=1 video_nr=9 card_label=Matting exclusive_caps=1
```

If it is already loaded for something else, add a device instead of reloading
the module — reloading would disconnect whatever is using it:

```sh
pkexec v4l2loopback-ctl add -n Matting /dev/video9
```

### Run

```sh
# Transparent background, for OBS
matting run

# Green screen, works everywhere
matting run --mode greenscreen

# Custom key colour
matting run --mode greenscreen --color '#0000FF'
matting run --mode greenscreen --color 0,0,255

# Photo background
matting run --mode image --image office.jpg
matting run --mode image --image office.jpg --fit contain
```

Point your application at the virtual camera (`Matting` / `/dev/video9`).

`run` prints its throughput once a second so you can see what you are getting.
Press Ctrl-C to stop; it shuts the virtual camera down cleanly.

### Shell completions

```sh
# bash
matting completions bash | pkexec tee /etc/bash_completion.d/matting > /dev/null

# zsh  (ensure ~/.zfunc is on your fpath before compinit)
mkdir -p ~/.zfunc && matting completions zsh > ~/.zfunc/_matting

# fish
mkdir -p ~/.config/fish/completions
matting completions fish > ~/.config/fish/completions/matting.fish
```

Restart your shell afterwards.

### Choosing a mode

`alpha` is the default and gives the best result **in OBS**, where real
transparency means you can drop the Chroma Key filter entirely.

Browsers, Zoom, Discord and most other applications cannot accept an alpha
video stream — they only take YUYV. Use `greenscreen` or `image` for those.
`run` warns you when you pick `alpha`.

### Colour accuracy

V4L2 has no reliable way to tell a consumer which YCbCr matrix a stream uses.
The virtual camera declares BT.601 limited range; OBS decodes it as **BT.709
full range** regardless. So the matrix has to match whatever your consumer
assumes, and `--colorimetry` selects it:

```sh
matting run --mode greenscreen --colorimetry bt709-full     # default, correct in OBS
matting run --mode greenscreen --colorimetry bt601-limited  # if colours look off elsewhere
```

Camera pixels are unaffected by a mismatch — they are decoded and re-encoded
with the same matrix, so they reach the consumer unchanged. Only colours this
program *introduces* shift: the greenscreen key and background images. With the
wrong setting, a `#00FF00` key arrives as `#00CB08`.

If your key colour looks slightly off, try the other setting.

### Alpha mode and premultiplication

RVM's foreground prediction is only meaningful where alpha is non-zero; in the
background it contains a smeared inpainting of the scene. Compositors that treat
BGRA as *premultiplied* — OBS does — would draw that over your background.

By default the output is premultiplied, which forces transparent pixels to black
and looks correct in OBS. If your compositor expects straight alpha and edges
look too dark:

```sh
matting run --mode alpha --alpha-mode straight
```

### Fitting a background image

`--fit` controls how your image is mapped onto the webcam resolution:

- `cover` (default) — scale to fill, cropping the overflow. No distortion.
- `contain` — scale to fit, adding black bars. No distortion, no cropping.
- `stretch` — scale to exactly fill. Distorts if aspect ratios differ.

## Performance

Measured on a Radeon 8060S (`gfx1151`, Ryzen AI MAX+ 395) at 1024×576, ROCm
7.2.4. Your numbers will differ:

| | ResNet50 | MobileNetV3 |
|---|---|---|
| Model inference, GPU | 10.0 ms | 7.7 ms |
| Model inference, CPU | 104 ms | 74 ms |

End to end — inference plus colour conversion and compositing — the pipeline
costs about **17–18 ms per frame** in `alpha` mode at 1024×576, giving headroom
for roughly 55 fps. `greenscreen` and `image` add a YUYV encode and cost a few
milliseconds more. In that state it used a little under one CPU core.

Most of the non-inference time is scalar colour conversion, so there is room
left for anyone who wants to reach for SIMD.

If you need more speed: use `--model mobilenetv3`, or lower `--ratio` (0.25
instead of 0.5) when preparing, which shrinks the resolution the model works at.

## Troubleshooting

**`no prepared model at ...; run 'matting prepare' first`**
You have not run `prepare`, or `~/.cache/matting` was cleared.

**`prepared model is stale, re-run 'matting prepare'`**
Something changed — resolution, ratio, model, GPU, or ROCm version. The message
names the field. Re-run `prepare`.

**`--model X expects state channels [...] but this model declares [...]`**
`--model` does not match the file in `--from`. The two RVM backbones have
different internal widths; pass the one that matches your download.

**`/dev/videoN would not accept YUYV WxH`**
Your webcam does not offer that format or size. See what it does support:

```sh
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Then run `prepare` again with matching `--width`/`--height`.

**`opening /dev/video9 ... No such file or directory`**
The virtual camera does not exist yet. See
[Create a virtual camera](#create-a-virtual-camera).

**Output looks right but is slow**
Confirm ONNX Runtime actually has the MIGraphX provider
(`ls /usr/lib/libonnxruntime_providers_migraphx.so`) and that `rocminfo` reports
your GPU. Without them there is no GPU path.

**The image is transparent/black in a browser or Zoom**
You are in `alpha` mode. Those applications cannot display alpha — use
`--mode greenscreen` or `--mode image`.

**The key colour is slightly wrong (`#00FF00` shows up as `#00CB08`)**
The YCbCr matrix does not match what your application assumes. See
[Colour accuracy](#colour-accuracy) and try the other `--colorimetry` value.
Background images shift the same way.

**Smeared or ghostly content in the transparent areas**
Your compositor wants the opposite alpha convention. See
[Alpha mode and premultiplication](#alpha-mode-and-premultiplication).

**`/dev/videoN would not accept ...` when switching modes**
Something still has the virtual camera open, and it keeps its pixel format while
a consumer is attached. Close it (`fuser -v /dev/video9` shows what) and retry.

## How it works

```
webcam (YUYV)
  → convert    YUYV 4:2:2 → planar RGB, normalized
  → model      RVM on the GPU; recurrent state carried between frames
  → composite  alpha, colour key, or image blend
  → v4l2loopback
```

The pipeline is single-threaded: at ~21 ms per frame against a 33 ms budget at
30 fps, there is nothing for extra threads to win.

RVM is *recurrent* — it remembers previous frames — which is why edges stay
stable instead of flickering the way per-frame image segmentation models do.
That state is carried forward on the GPU without copying it back to the host.

## Development

```sh
cargo test --release -- --test-threads=1
```

Single-threaded because several tests build MIGraphX sessions against the same
compile cache.

Tests that need real model files read their paths from the environment and skip
when unset:

```sh
export MATTING_TEST_RVM_RESNET50=/path/to/rvm_resnet50.onnx
export MATTING_TEST_RVM_MOBILENETV3=/path/to/rvm_mobilenetv3_fp32.onnx
```

The most important test is
`model::tests::frozen_model_matches_original_within_tolerance`. It runs the
unmodified RVM export on CPU and compares its alpha output against the rewritten
graph on GPU. Graph surgery can corrupt a model in ways that still produce
plausible-looking output, and this test is the only thing that would catch it —
do not loosen its tolerance to make it pass.

## Credits

This project is a thin pipeline around other people's hard work. The matting
itself — the part that actually makes this useful — is entirely theirs.

**[Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)** by
Shanchuan Lin, Linjie Yang, Imran Saleemi and Soumyadip Sengupta (University of
Washington and ByteDance). RVM is the model this is built around: a recurrent
architecture that carries temporal memory between frames, which is why edges
stay stable instead of flickering the way per-frame segmentation does. This
project only reshapes their published ONNX export so an AMD compiler will accept
it — the network, the training and the quality of the result are entirely their
contribution.

> Shanchuan Lin, Linjie Yang, Imran Saleemi, Soumyadip Sengupta.
> *Robust High-Resolution Video Matting with Temporal Guidance.* WACV 2022,
> pp. 238–247. ([arXiv:2108.11515](https://arxiv.org/abs/2108.11515) ·
> [project page](https://peterl1n.github.io/RobustVideoMatting/))

RVM is itself GPL-3.0 licensed, which is why this project uses the same licence.

Also relied on:

| Project | Role | Licence |
|---|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Model execution | MIT |
| [ROCm](https://github.com/ROCm/ROCm) / [MIGraphX](https://github.com/ROCm/AMDMIGraphX) | AMD GPU compute and graph compiler | MIT |
| [v4l2loopback](https://github.com/umlaeute/v4l2loopback) | Virtual camera device | GPL-2.0-or-later |
| [ort](https://github.com/pykeio/ort) | Rust bindings for ONNX Runtime | MIT OR Apache-2.0 |
| [libv4l-rs](https://github.com/raymanfx/libv4l-rs) (`v4l`) | Rust V4L2 bindings | MIT |
| [clap](https://github.com/clap-rs/clap) | Argument parsing and completions | MIT OR Apache-2.0 |
| [image](https://github.com/image-rs/image) | Background image loading and scaling | MIT OR Apache-2.0 |
| [onnx-protobuf](https://crates.io/crates/onnx-protobuf) | Generated ONNX protobuf types | MPL-2.0 |
| [rust-protobuf](https://github.com/stepancheg/rust-protobuf) | Protobuf runtime | MIT |
| [anyhow](https://github.com/dtolnay/anyhow), [serde](https://github.com/serde-rs/serde), [sha2](https://github.com/RustCrypto/hashes) | Errors, serialisation, hashing | MIT OR Apache-2.0 |

## A note on how this was built

This project was written with substantial AI assistance — [Claude
Code](https://claude.com/claude-code) (Claude Opus 5) produced most of the
design, implementation, tests and documentation, working under human direction
and review.

That is worth stating plainly so you can calibrate your trust accordingly.
Some specifics:

- The core finding — that MIGraphX rejects RVM because `downsample_ratio` is a
  runtime input — was established by benchmarking on real hardware, not
  inferred. The measurements in [Performance](#performance) are from actual
  runs on one machine, and all three output modes were verified against a real
  webcam and virtual camera.
- The riskiest part of this project is the graph rewriting, because a corrupted
  model can still produce plausible-looking output. That is exactly what
  `frozen_model_matches_original_within_tolerance` exists to catch: it compares
  the rewritten graph against the untouched original. It is worth reading before
  trusting anything else here.
- It has been tested on exactly one GPU. The
  [supported GPU list](#supported-gpus) is derived from what ROCm ships kernels
  for, not from testing.

Bug reports are welcome, and reviewing the code before relying on it is
encouraged.

## Licence

Released under the **GNU General Public License v3.0 or later**. See
[LICENSE](LICENSE).

Note that model weights are *not* distributed with this project — you download
them yourself from the RVM releases page, and they carry
[their own licence](https://github.com/PeterL1n/RobustVideoMatting/blob/master/LICENSE).
