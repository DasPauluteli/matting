use std::io::{IsTerminal, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::capture::{Source, V4lSource};
use crate::cli::{AlphaMode, Mode, RunArgs};
use crate::composite::{over_color, over_image, to_bgra, Background};
use crate::convert::{rgb_to_yuyv, yuyv_to_rgb_f32_nchw, Matrix};
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
    matrix: Matrix,
    premultiplied: bool,
    rgb_in: Vec<f32>,
    rgb_out: Vec<u8>,
    out: Vec<u8>,
}

/// Everything the pipeline needs besides the model itself.
pub struct Config {
    pub mode: Mode,
    pub colour: [u8; 3],
    pub background: Option<Background>,
    pub width: u32,
    pub height: u32,
    pub matrix: Matrix,
    pub premultiplied: bool,
}

impl Pipeline {
    pub fn new(model: Matting, config: Config) -> Pipeline {
        let (w, h) = (config.width as usize, config.height as usize);
        let plane = w * h;
        Pipeline {
            model,
            mode: config.mode,
            colour: config.colour,
            background: config.background,
            width: w,
            height: h,
            matrix: config.matrix,
            premultiplied: config.premultiplied,
            rgb_in: vec![0.0; plane * 3],
            rgb_out: vec![0; plane * 3],
            out: vec![0; plane * bytes_per_pixel(config.mode)],
        }
    }

    /// YUYV in, mode-appropriate bytes out.
    pub fn process(&mut self, yuyv: &[u8]) -> Result<&[u8]> {
        let plane = self.width * self.height;
        yuyv_to_rgb_f32_nchw(yuyv, self.width, self.height, self.matrix, &mut self.rgb_in);
        let (fgr, pha) = self.model.infer(&self.rgb_in)?;

        match self.mode {
            Mode::Alpha => to_bgra(fgr, pha, plane, self.premultiplied, &mut self.out),
            Mode::Greenscreen => {
                over_color(fgr, pha, plane, self.colour, &mut self.rgb_out);
                rgb_to_yuyv(&self.rgb_out, self.width, self.height, self.matrix, &mut self.out);
            }
            Mode::Image => {
                let bg = self
                    .background
                    .as_ref()
                    .context("image mode without a background")?;
                over_image(fgr, pha, plane, bg.rgb(), &mut self.rgb_out);
                rgb_to_yuyv(&self.rgb_out, self.width, self.height, self.matrix, &mut self.out);
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
        anyhow::bail!(
            "no frozen model at {}; run `matting prepare`",
            frozen.display()
        );
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
    let mut pipeline = Pipeline::new(
        model,
        Config {
            mode: args.mode,
            colour,
            background,
            width,
            height,
            matrix: Matrix::from_cli(args.colorimetry),
            premultiplied: args.alpha_mode == AlphaMode::Premultiplied,
        },
    );

    println!(
        "Streaming {}x{} {:?} from {} to {} as {}",
        width,
        height,
        args.mode,
        args.input,
        args.output,
        crate::sink::fourcc_for(args.mode)
    );
    if args.mode == Mode::Alpha {
        println!(
            "Note: alpha output is only displayed correctly by OBS. \
             Browsers, Zoom and Discord need --mode greenscreen or --mode image."
        );
    }

    // Redraw the throughput line in place on a terminal, but emit plain lines
    // when redirected so logs stay readable.
    let interactive = std::io::stdout().is_terminal();
    let mut status_drawn = false;

    // Ctrl-C sets this rather than killing the process outright, so the status
    // line is closed off and the V4L2 streams are torn down cleanly.
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    {
        let running = running.clone();
        ctrlc::set_handler(move || running.store(false, std::sync::atomic::Ordering::SeqCst))
            .context("installing the Ctrl-C handler")?;
    }

    let mut failures = 0u32;
    let mut frames = 0u64;
    let mut window_start = std::time::Instant::now();
    let mut infer_total = std::time::Duration::ZERO;
    while running.load(std::sync::atomic::Ordering::SeqCst) {
        // Borrowed, not copied: `source` and `pipeline` are separate bindings,
        // so the frame can be handed straight through.
        let frame = match source.next_frame() {
            Ok(f) => {
                failures = 0;
                f
            }
            Err(e) => {
                // Don't let a message land on top of the in-place status line.
                if std::mem::take(&mut status_drawn) {
                    println!();
                }
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
        let t0 = std::time::Instant::now();
        let out = match pipeline.process(frame) {
            Ok(out) => out,
            Err(e) => {
                // Leave the cursor on a clean line before the error surfaces.
                if status_drawn {
                    println!();
                }
                return Err(e);
            }
        };
        infer_total += t0.elapsed();
        if let Err(e) = sink.write_frame(out) {
            if status_drawn {
                println!();
            }
            return Err(e);
        }

        // Report throughput once a second: delivered fps, and how much of the
        // frame budget the model plus compositing actually consume.
        frames += 1;
        let elapsed = window_start.elapsed();
        if elapsed >= std::time::Duration::from_secs(1) {
            let fps = frames as f64 / elapsed.as_secs_f64();
            let per_frame_ms = infer_total.as_secs_f64() * 1000.0 / frames as f64;
            let line = format!("{fps:.1} fps  ({per_frame_ms:.1} ms/frame processing)");
            if interactive {
                // \x1b[K clears any leftover from a previously longer line.
                print!("\r{line}\x1b[K");
                let _ = std::io::stdout().flush();
                status_drawn = true;
            } else {
                println!("{line}");
            }
            frames = 0;
            infer_total = std::time::Duration::ZERO;
            window_start = std::time::Instant::now();
        }
    }

    if status_drawn {
        println!();
    }
    println!("Stopped.");
    Ok(())
}

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
        let mut pipe = Pipeline::new(
            model,
            Config {
                mode: Mode::Greenscreen,
                colour: [0, 255, 0],
                background: None,
                width: w,
                height: h,
                matrix: Matrix::BT709_FULL,
                premultiplied: true,
            },
        );

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
        let mut pipe = Pipeline::new(
            model,
            Config {
                mode: Mode::Alpha,
                colour: [0, 255, 0],
                background: None,
                width: w,
                height: h,
                matrix: Matrix::BT709_FULL,
                premultiplied: true,
            },
        );
        let out = pipe.process(&frame).unwrap();
        assert_eq!(out.len(), (w * h * 4) as usize);
    }
}
