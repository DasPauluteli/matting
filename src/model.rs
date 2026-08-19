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
        // `with_execution_providers` returns ort's recoverable Error<SessionBuilder>,
        // which is neither Send nor Sync, so it cannot cross `?` into anyhow.
        let session = Session::builder()?
            .with_execution_providers([ep::MIGraphX::default().with_fp16(true).build()])
            .map_err(|e| anyhow::anyhow!("registering the MIGraphX provider: {e}"))?
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
