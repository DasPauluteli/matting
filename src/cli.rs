use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "matting",
    version,
    about = "Realtime webcam background matting on AMD GPUs",
    long_about = "Removes your webcam's background in realtime using Robust Video Matting \
                  on an AMD GPU, and publishes the result to a virtual camera.\n\n\
                  Run `matting prepare` once to compile a model, then `matting run` to stream.",
    after_help = "EXAMPLES:\n  \
        matting prepare --from rvm_resnet50.onnx --model resnet50\n  \
        matting run --mode greenscreen --color '#00FF00'\n  \
        matting run --mode image --image office.jpg --fit cover\n\n\
        Run `matting <command> --help` for the full options of a command."
)]
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
    /// Capture a background plate for BackgroundMattingV2. Step out of frame first.
    CapturePlate(CapturePlateArgs),
    /// Print a shell completion script to stdout.
    Completions(CompletionsArgs),
}

#[derive(Args, Debug)]
#[command(
    after_help = "BackgroundMattingV2 works by comparing each frame against a photo of\n\
        the empty scene. Step out of shot, run this, then keep the camera still —\n\
        moving it, or a change in lighting, invalidates the plate.\n\n\
        RVM does not need this."
)]
pub struct CapturePlateArgs {
    /// Webcam to capture from.
    #[arg(long, default_value = "/dev/video0")]
    pub input: String,
    /// Where to write the plate.
    #[arg(long, default_value = "background.png")]
    pub output: String,
    /// Seconds to wait before capturing, to get out of frame.
    #[arg(long, default_value_t = 5)]
    pub delay: u32,
}

#[derive(Args, Debug)]
#[command(
    after_help = "Write the script somewhere your shell reads completions from:\n\n  \
        bash:  matting completions bash > /etc/bash_completion.d/matting\n  \
        zsh:   matting completions zsh  > ~/.zfunc/_matting\n  \
        fish:  matting completions fish > ~/.config/fish/completions/matting.fish\n\n\
        For zsh, make sure ~/.zfunc is on your fpath before compinit runs."
)]
pub struct CompletionsArgs {
    /// Shell to generate completions for.
    #[arg(value_enum)]
    pub shell: clap_complete::Shell,
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

/// Which YCbCr convention to use when encoding YUYV.
///
/// V4L2 has no reliable way to signal this, so it must match whatever the
/// consuming application assumes. OBS assumes BT.709 full range.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colorimetry {
    /// What most webcams produce and what the V4L2 device declares.
    Bt601Limited,
    Bt601Full,
    Bt709Limited,
    /// What OBS assumes when reading a v4l2loopback device.
    Bt709Full,
}

/// How the alpha channel relates to the colour channels.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    /// Colour already multiplied by alpha. What OBS expects.
    Premultiplied,
    /// Colour independent of alpha.
    Straight,
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
#[command(
    after_help = "Compiling takes a few minutes the first time; the result is cached in\n\
        ~/.cache/matting and reused on later runs. Re-run this if you change\n\
        resolution, ratio, model, or upgrade ROCm.\n\n\
        Get a stock RVM export from:\n  \
        https://github.com/PeterL1n/RobustVideoMatting/releases"
)]
pub struct PrepareArgs {
    /// Path to a stock RVM ONNX export.
    #[arg(long)]
    pub from: String,
    /// Which backbone `--from` contains. Verified against the model.
    #[arg(long, value_enum, default_value_t = Backbone::Resnet50)]
    pub model: Backbone,

    /// Capture width. Must match your webcam.
    #[arg(long, default_value_t = 1024, help_heading = "Geometry")]
    pub width: u32,
    /// Capture height. Must match your webcam.
    #[arg(long, default_value_t = 576, help_heading = "Geometry")]
    pub height: u32,
    /// Internal downscale before inference. Lower is faster, less detailed.
    #[arg(long, default_value_t = 0.5, help_heading = "Geometry")]
    pub ratio: f32,
}

impl PrepareArgs {
    pub fn validate(&self) -> Result<(), String> {
        // YUYV 4:2:2 encodes two pixels per macropixel, so an odd width would
        // silently drop the last column rather than fail.
        if self.width == 0 || !self.width.is_multiple_of(2) {
            return Err(format!(
                "--width must be even and non-zero (YUYV pairs pixels horizontally), got {}",
                self.width
            ));
        }
        if self.height == 0 {
            return Err("--height must be non-zero".into());
        }
        if !(self.ratio > 0.0 && self.ratio <= 1.0) {
            return Err(format!(
                "--ratio must be greater than 0 and at most 1, got {}",
                self.ratio
            ));
        }
        // The model downsamples to ratio*size and then halves four more times;
        // too small and the recurrent state collapses to zero-sized tensors.
        let smallest = (self.width.min(self.height) as f32 * self.ratio) as u32 / 16;
        if smallest == 0 {
            return Err(format!(
                "--ratio {} is too small for {}x{}: the model's internal state would \
                 collapse to zero size. Try a larger ratio.",
                self.ratio, self.width, self.height
            ));
        }
        Ok(())
    }
}

#[derive(Args, Debug)]
#[command(
    after_help = "MODES:\n  \
        alpha        Transparent background (BGRA). Only OBS displays this correctly;\n               \
                     it removes the need for a Chroma Key filter.\n  \
        greenscreen  Solid colour background (YUYV). Works in every application.\n  \
        image        Photo background (YUYV). Works in every application.\n\n\
        EXAMPLES:\n  \
        matting run\n  \
        matting run --mode greenscreen --color '#0000FF'\n  \
        matting run --mode image --image beach.jpg --fit contain"
)]
pub struct RunArgs {
    /// Webcam to read from.
    #[arg(long, default_value = "/dev/video0", help_heading = "Devices")]
    pub input: String,
    /// v4l2loopback device to write to.
    #[arg(long, default_value = "/dev/video9", help_heading = "Devices")]
    pub output: String,

    /// How to replace the background.
    #[arg(long, value_enum, default_value_t = Mode::Alpha, help_heading = "Output")]
    pub mode: Mode,
    /// Which prepared backbone to load. Must match what `prepare` built.
    #[arg(long, value_enum, default_value_t = Backbone::Resnet50, help_heading = "Output")]
    pub model: Backbone,
    /// YCbCr convention for YUYV output. Must match what the consumer assumes.
    #[arg(long, value_enum, default_value_t = Colorimetry::Bt709Full, help_heading = "Output")]
    pub colorimetry: Colorimetry,
    /// Whether alpha-mode colour is premultiplied by alpha.
    #[arg(long, value_enum, default_value_t = AlphaMode::Premultiplied, help_heading = "Output")]
    pub alpha_mode: AlphaMode,

    /// Key colour as #RRGGBB or r,g,b.
    #[arg(
        long,
        default_value = "#00FF00",
        help_heading = "Greenscreen mode (--mode greenscreen)"
    )]
    pub color: String,

    /// Background plate, required when the prepared model is
    /// BackgroundMattingV2. Capture one with `matting capture-plate`.
    #[arg(long, help_heading = "BackgroundMattingV2")]
    pub plate: Option<String>,

    /// Background image file. Required for image mode.
    #[arg(long, help_heading = "Image mode (--mode image)")]
    pub image: Option<String>,
    /// How to map the image onto the webcam resolution.
    #[arg(
        long,
        value_enum,
        default_value_t = Fit::Cover,
        help_heading = "Image mode (--mode image)"
    )]
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

    fn prepare_args(width: u32, height: u32, ratio: f32) -> PrepareArgs {
        PrepareArgs {
            from: "model.onnx".into(),
            model: Backbone::Resnet50,
            width,
            height,
            ratio,
        }
    }

    #[test]
    fn accepts_sensible_geometry() {
        assert!(prepare_args(1024, 576, 0.5).validate().is_ok());
        assert!(prepare_args(1280, 720, 1.0).validate().is_ok());
    }

    #[test]
    fn rejects_odd_width_that_would_drop_a_column() {
        let err = prepare_args(1023, 576, 0.5).validate().unwrap_err();
        assert!(err.contains("even"), "{err}");
    }

    #[test]
    fn rejects_out_of_range_ratio() {
        assert!(prepare_args(1024, 576, 0.0).validate().is_err());
        assert!(prepare_args(1024, 576, -0.5).validate().is_err());
        assert!(prepare_args(1024, 576, 1.5).validate().is_err());
    }

    #[test]
    fn rejects_ratio_that_would_collapse_recurrent_state() {
        let err = prepare_args(1024, 576, 0.01).validate().unwrap_err();
        assert!(err.contains("collapse"), "{err}");
    }

    #[test]
    fn generates_completions_for_every_supported_shell() {
        use clap::CommandFactory;
        for shell in [
            clap_complete::Shell::Bash,
            clap_complete::Shell::Zsh,
            clap_complete::Shell::Fish,
        ] {
            let mut buf = Vec::new();
            let mut cmd = Cli::command();
            clap_complete::generate(shell, &mut cmd, "matting", &mut buf);
            let script = String::from_utf8(buf).expect("completion script must be UTF-8");
            assert!(script.contains("prepare"), "{shell:?} script missing subcommands");
            assert!(script.contains("greenscreen"), "{shell:?} script missing mode values");
        }
    }

    #[test]
    fn image_mode_requires_image_path() {
        let cli = Cli::parse_from(["matting", "run", "--mode", "image"]);
        let Command::Run(args) = cli.command else { panic!("expected run") };
        assert!(args.validate().is_err(), "image mode without --image must fail");
    }
}
