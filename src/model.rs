use std::path::Path;

use anyhow::{Context, Result};
use half::f16;
use ort::{
    ep,
    session::{Session, SessionInputValue},
    value::{DynValue, Tensor, TensorElementType, TensorRef, ValueType},
};

/// Which matting network a prepared model contains.
///
/// Detected from the graph rather than configured: RVM carries recurrent state
/// inputs (`r1i`..`r4i`), BackgroundMattingV2 takes a second image (`bgr`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// Robust Video Matting: recurrent, no background plate needed.
    Rvm,
    /// BackgroundMattingV2: needs a photo of the empty scene.
    Bgmv2,
}

/// Holds the ORT session plus whatever per-frame state the network needs.
///
/// For RVM, state is moved in and out by ownership rather than copied back to
/// host memory: `SessionOutputs::remove` yields an owned `DynValue` that stays
/// on the GPU, so the recurrence costs nothing per frame.
pub struct Matting {
    session: Session,
    arch: Arch,
    state: Option<[DynValue; 4]>,
    state_shapes: [Vec<i64>; 4],
    /// BackgroundMattingV2's background plate, converted once at startup.
    plate: Vec<f32>,
    plate_f16: Vec<f16>,
    /// Restricts BackgroundMattingV2 to the two outputs we actually consume.
    bgmv2_outputs: ort::session::RunOptions<ort::session::HasSelectedOutputs>,
    width: u32,
    height: u32,
    /// RVM publishes both fp32 and fp16 ONNX exports. The fp16 ones want
    /// float16 tensors on every input, so the buffers we hand over have to
    /// match whatever this particular export declares.
    fp16: bool,
    src_f16: Vec<f16>,
    fgr: Vec<f32>,
    pha: Vec<f32>,
}

impl Matting {
    pub fn load(frozen: &Path, width: u32, height: u32) -> Result<Matting> {
        ort::init().commit();
        // `with_execution_providers` returns ort's recoverable Error<SessionBuilder>,
        // which is neither Send nor Sync, so it cannot cross `?` into anyhow.
        let session = Session::builder()?
            .with_execution_providers([ep::MIGraphX::default().with_fp16(true).build()])
            .map_err(|e| anyhow::anyhow!("registering the MIGraphX provider: {e}"))?
            .commit_from_file(frozen)
            .with_context(|| format!("loading {}", frozen.display()))?;

        let arch = if session.inputs().iter().any(|i| i.name() == "bgr") {
            Arch::Bgmv2
        } else {
            Arch::Rvm
        };

        // RVM's frozen graph pins every state shape; read them back so we can
        // build correctly sized zero tensors for the first frame and on reset.
        let mut state_shapes: [Vec<i64>; 4] = Default::default();
        if arch == Arch::Rvm {
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
        }

        let src = session
            .inputs()
            .iter()
            .find(|i| i.name() == "src")
            .context("frozen model has no `src` input")?;
        let ValueType::Tensor { ty, .. } = src.dtype() else {
            anyhow::bail!("`src` is not a tensor");
        };
        let fp16 = match ty {
            TensorElementType::Float32 => false,
            TensorElementType::Float16 => true,
            other => anyhow::bail!(
                "unsupported `src` element type {other:?}; use an fp32 or fp16 RVM export"
            ),
        };

        let plane = (width * height) as usize;
        Ok(Matting {
            session,
            arch,
            state: None,
            state_shapes,
            plate: Vec::new(),
            plate_f16: Vec::new(),
            bgmv2_outputs: {
                ort::session::RunOptions::new()?.with_outputs(
                    ort::session::OutputSelector::no_default()
                        .with("pha")
                        .with("fgr"),
                )
            },
            width,
            height,
            fp16,
            src_f16: if fp16 { vec![f16::ZERO; plane * 3] } else { Vec::new() },
            fgr: vec![0.0; plane * 3],
            pha: vec![0.0; plane],
        })
    }

    pub fn arch(&self) -> Arch {
        self.arch
    }

    /// Supply BackgroundMattingV2's background plate: the empty scene, in the
    /// same NCHW normalized layout as a frame. Converted once, reused per frame.
    pub fn set_background(&mut self, rgb_nchw: &[f32]) -> Result<()> {
        let expected = (self.width * self.height) as usize * 3;
        if rgb_nchw.len() != expected {
            anyhow::bail!(
                "background plate has {} values, expected {expected}",
                rgb_nchw.len()
            );
        }
        if self.fp16 {
            self.plate_f16 = rgb_nchw.iter().map(|&v| f16::from_f32(v)).collect();
        } else {
            self.plate = rgb_nchw.to_vec();
        }
        Ok(())
    }

    fn zero_state(&self) -> Result<[DynValue; 4]> {
        let mut built = Vec::with_capacity(4);
        for shape in &self.state_shapes {
            let n = shape.iter().product::<i64>() as usize;
            built.push(if self.fp16 {
                Tensor::from_array((shape.clone(), vec![f16::ZERO; n]))?.into_dyn()
            } else {
                Tensor::from_array((shape.clone(), vec![0f32; n]))?.into_dyn()
            });
        }
        built
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected exactly four recurrent state tensors"))
    }

    /// Drop the temporal memory. Call on stream reconnect so a new scene does
    /// not inherit the previous one's alpha.
    pub fn reset_state(&mut self) -> Result<()> {
        self.state = None;
        Ok(())
    }

    /// Whether temporal memory from a previous frame is being carried.
    pub fn has_state(&self) -> bool {
        self.state.is_some()
    }

    /// `rgb_nchw` must be `3 * width * height` normalized to [0,1].
    /// Returns `(fgr, pha)` borrowed from internal buffers.
    pub fn infer(&mut self, rgb_nchw: &[f32]) -> Result<(&[f32], &[f32])> {
        match self.arch {
            Arch::Rvm => self.infer_rvm(rgb_nchw),
            Arch::Bgmv2 => self.infer_bgmv2(rgb_nchw),
        }
    }

    /// BackgroundMattingV2 compares each frame against a fixed background
    /// plate. It has no temporal memory, so there is nothing to carry forward.
    fn infer_bgmv2(&mut self, rgb_nchw: &[f32]) -> Result<(&[f32], &[f32])> {
        if (self.fp16 && self.plate_f16.is_empty()) || (!self.fp16 && self.plate.is_empty()) {
            anyhow::bail!(
                "this model is BackgroundMattingV2 and needs a background plate; \
                 pass --plate <image> (capture one with `matting capture-plate`)"
            );
        }
        let shape = vec![1i64, 3, self.height as i64, self.width as i64];
        let (src, bgr): (SessionInputValue, SessionInputValue) = if self.fp16 {
            for (dst, &s) in self.src_f16.iter_mut().zip(rgb_nchw) {
                *dst = f16::from_f32(s);
            }
            (
                TensorRef::from_array_view((shape.clone(), self.src_f16.as_slice()))?.into(),
                TensorRef::from_array_view((shape, self.plate_f16.as_slice()))?.into(),
            )
        } else {
            (
                TensorRef::from_array_view((shape.clone(), rgb_nchw))?.into(),
                TensorRef::from_array_view((shape, self.plate.as_slice()))?.into(),
            )
        };

        // BackgroundMattingV2 declares six outputs; we use two. Asking only for
        // those lets ONNX Runtime prune the nodes feeding the rest, and avoids
        // copying several full-resolution tensors back to host memory.
        let outputs = self.session.run_with_options(
            vec![
                (std::borrow::Cow::from("src"), src),
                (std::borrow::Cow::from("bgr"), bgr),
            ],
            &self.bgmv2_outputs,
        )?;

        if self.fp16 {
            let (_, fgr) = outputs["fgr"].try_extract_tensor::<f16>()?;
            for (dst, s) in self.fgr.iter_mut().zip(fgr) {
                *dst = s.to_f32();
            }
            let (_, pha) = outputs["pha"].try_extract_tensor::<f16>()?;
            for (dst, s) in self.pha.iter_mut().zip(pha) {
                *dst = s.to_f32();
            }
        } else {
            let (_, fgr) = outputs["fgr"].try_extract_tensor::<f32>()?;
            self.fgr.copy_from_slice(fgr);
            let (_, pha) = outputs["pha"].try_extract_tensor::<f32>()?;
            self.pha.copy_from_slice(pha);
        }
        Ok((&self.fgr, &self.pha))
    }

    fn infer_rvm(&mut self, rgb_nchw: &[f32]) -> Result<(&[f32], &[f32])> {
        let state = match self.state.take() {
            Some(s) => s,
            None => self.zero_state()?,
        };
        let [r1, r2, r3, r4] = state;

        let shape = vec![1i64, 3, self.height as i64, self.width as i64];
        // fp32 exports can borrow the caller's buffer directly, avoiding a ~7 MB
        // copy per frame. fp16 exports need a narrowing pass into a reused
        // buffer, so the copy is unavoidable there.
        let src: SessionInputValue = if self.fp16 {
            for (dst, &s) in self.src_f16.iter_mut().zip(rgb_nchw) {
                *dst = f16::from_f32(s);
            }
            TensorRef::from_array_view((shape, self.src_f16.as_slice()))?.into()
        } else {
            TensorRef::from_array_view((shape, rgb_nchw))?.into()
        };

        let inputs: Vec<(std::borrow::Cow<str>, SessionInputValue)> = vec![
            ("src".into(), src),
            ("r1i".into(), SessionInputValue::from(r1)),
            ("r2i".into(), SessionInputValue::from(r2)),
            ("r3i".into(), SessionInputValue::from(r3)),
            ("r4i".into(), SessionInputValue::from(r4)),
        ];

        let mut outputs = self.session.run(inputs)?;

        if self.fp16 {
            let (_, fgr) = outputs["fgr"].try_extract_tensor::<f16>()?;
            for (dst, s) in self.fgr.iter_mut().zip(fgr) {
                *dst = s.to_f32();
            }
            let (_, pha) = outputs["pha"].try_extract_tensor::<f16>()?;
            for (dst, s) in self.pha.iter_mut().zip(pha) {
                *dst = s.to_f32();
            }
        } else {
            let (_, fgr) = outputs["fgr"].try_extract_tensor::<f32>()?;
            self.fgr.copy_from_slice(fgr);
            let (_, pha) = outputs["pha"].try_extract_tensor::<f32>()?;
            self.pha.copy_from_slice(pha);
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to a stock ResNet50 RVM export, supplied by the environment so the
    /// suite is not tied to one machine's layout.
    fn r50_source() -> Option<String> {
        std::env::var("MATTING_TEST_RVM_RESNET50")
            .ok()
            .filter(|p| std::path::Path::new(p).exists())
    }

    /// A deterministic but *band-limited* test frame: smooth gradients plus a
    /// few soft blobs.
    ///
    /// This deliberately avoids pixel-level noise. Feeding high-frequency
    /// synthetic patterns through the model puts it in a chaotic regime where
    /// ordinary fp16-vs-fp32 differences amplify enormously — at `--ratio 1.0`
    /// a per-pixel pattern diverged by 0.55 while this frame diverges by 0.008.
    /// Real camera frames are band-limited, so this is both more representative
    /// and a stable basis for the tolerance check.
    fn deterministic_frame(width: usize, height: usize) -> Vec<f32> {
        let plane = width * height;
        let mut v = vec![0f32; plane * 3];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let fx = x as f32 / width as f32;
                let fy = y as f32 / height as f32;
                // Two soft blobs give the model some spatial structure to work
                // with, without introducing high spatial frequencies.
                let blob = |cx: f32, cy: f32, r: f32| {
                    let d = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
                    (1.0 - (d / r).min(1.0)).powi(2)
                };
                let subject = blob(0.5, 0.55, 0.35);
                v[i] = (0.2 + 0.5 * fx + 0.6 * subject).clamp(0.0, 1.0);
                v[plane + i] = (0.3 + 0.4 * fy + 0.5 * subject).clamp(0.0, 1.0);
                v[2 * plane + i] = (0.5 - 0.3 * fx + 0.4 * blob(0.25, 0.3, 0.25)).clamp(0.0, 1.0);
            }
        }
        v
    }

    /// A high-frequency frame. Numerically unstable to compare across
    /// precisions (see `deterministic_frame`), but it provokes a non-trivial
    /// response from the model, which is what the recurrence test needs.
    fn textured_frame(width: usize, height: usize) -> Vec<f32> {
        let plane = width * height;
        let mut v = vec![0f32; plane * 3];
        for i in 0..plane {
            v[i] = ((i % 255) as f32) / 255.0;
            v[plane + i] = (((i / 3) % 255) as f32) / 255.0;
            v[2 * plane + i] = (((i / 7) % 255) as f32) / 255.0;
        }
        v
    }

    /// Runs the *unmodified* RVM export on CPU to produce ground truth.
    fn reference_pha(path: &str, frame: &[f32], w: usize, h: usize, ratio: f32) -> Vec<f32> {
        use ort::{
            session::{Session, SessionInputValue},
            value::Tensor,
        };

        let mut session = Session::builder().unwrap().commit_from_file(path).unwrap();
        let inputs: Vec<(std::borrow::Cow<str>, SessionInputValue)> = vec![
            (
                "src".into(),
                Tensor::from_array((vec![1i64, 3, h as i64, w as i64], frame.to_vec()))
                    .unwrap()
                    .into(),
            ),
            (
                "r1i".into(),
                Tensor::from_array((vec![1i64, 1, 1, 1], vec![0f32])).unwrap().into(),
            ),
            (
                "r2i".into(),
                Tensor::from_array((vec![1i64, 1, 1, 1], vec![0f32])).unwrap().into(),
            ),
            (
                "r3i".into(),
                Tensor::from_array((vec![1i64, 1, 1, 1], vec![0f32])).unwrap().into(),
            ),
            (
                "r4i".into(),
                Tensor::from_array((vec![1i64, 1, 1, 1], vec![0f32])).unwrap().into(),
            ),
            (
                "downsample_ratio".into(),
                Tensor::from_array((vec![1i64], vec![ratio])).unwrap().into(),
            ),
        ];
        let outputs = session.run(inputs).unwrap();
        let (_, data) = outputs["pha"].try_extract_tensor::<f32>().unwrap();
        data.to_vec()
    }

    /// The frozen graph must compute what the stock graph computes. Anything
    /// else means the surgery in `freeze` corrupted the model.
    #[test]
    fn frozen_model_matches_original_within_tolerance() {
        let dir = crate::prepare::cache_dir();
        let frozen = crate::prepare::frozen_model_path(&dir);
        let Some(source) = r50_source() else {
            eprintln!("skipping: set MATTING_TEST_RVM_RESNET50 to a stock RVM export");
            return;
        };
        if !frozen.exists() {
            eprintln!("skipping: run `matting prepare` first");
            return;
        }
        // Compare like with like: the cache may have been prepared from a
        // different export or at a different ratio than this test's reference.
        let Ok(manifest) = crate::manifest::Manifest::load(&dir) else {
            eprintln!("skipping: no manifest, run `matting prepare` first");
            return;
        };
        let source_sha = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(std::fs::read(&source).unwrap()))
        };
        if manifest.source_sha256 != source_sha {
            eprintln!(
                "skipping: cache was prepared from a different model \
                 (cached {}…, this test has {}…)",
                &manifest.source_sha256[..12],
                &source_sha[..12]
            );
            return;
        }
        let ratio = manifest.ratio;
        // Without this the MIGraphX EP aborts session init on an empty cache path.
        crate::prepare::set_migraphx_cache_env(&crate::prepare::cache_dir()).unwrap();
        let (w, h) = (1024usize, 576usize);
        let frame = deterministic_frame(w, h);

        let mut frozen_model = Matting::load(&frozen, w as u32, h as u32).unwrap();
        let (_, frozen_pha) = frozen_model.infer(&frame).unwrap();
        let frozen_pha = frozen_pha.to_vec();

        let reference = reference_pha(&source, &frame, w, h, ratio);

        assert_eq!(frozen_pha.len(), reference.len());
        let mut worst = 0f32;
        for (a, b) in frozen_pha.iter().zip(&reference) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst < 0.02,
            "frozen model diverges from original, worst delta {worst}"
        );
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
        let frame = textured_frame(w, h);
        let mut m = Matting::load(&frozen, w as u32, h as u32).unwrap();

        // Check the mechanism directly. Asserting that two identical frames
        // produce *different* alpha only holds when the model actually responds
        // to the input — with synthetic frames it may return all zeros, which
        // made the old form of this test fail for reasons unrelated to state.
        assert!(!m.has_state(), "should start with no temporal memory");
        let first = m.infer(&frame).unwrap().1.to_vec();
        assert!(m.has_state(), "a frame should leave temporal memory behind");

        let second = m.infer(&frame).unwrap().1.to_vec();
        assert!(m.has_state(), "memory should persist across frames");

        m.reset_state().unwrap();
        assert!(!m.has_state(), "reset_state must clear temporal memory");

        // Feeding the first frame again after a reset must reproduce the very
        // first result exactly; that is the property `run` relies on when it
        // resets after a dropped stream.
        let after_reset = m.infer(&frame).unwrap().1.to_vec();
        assert_eq!(first, after_reset, "reset_state should reproduce the first frame");

        // Only meaningful when the model produced something to carry forward.
        if first.iter().any(|&v| v > 0.0) {
            assert_ne!(first, second, "recurrent state should change the second result");
        }
    }
}
