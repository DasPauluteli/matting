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
