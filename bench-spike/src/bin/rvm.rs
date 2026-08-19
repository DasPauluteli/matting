//! THROWAWAY SPIKE. Benchmarks Robust Video Matting with realistic recurrent
//! state feedback (r1o..r4o -> r1i..r4i), which is how it actually runs on video.

use std::time::Instant;

use ort::{
    ep,
    session::Session,
    value::{Tensor, TensorElementType, ValueType},
};

fn main() -> ort::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: rvm <model.onnx> [ep] [w] [h] [ratio] [iters]");
    let ep_name = args.next().unwrap_or_else(|| "migraphx".into());
    let w: i64 = args.next().unwrap_or_else(|| "1024".into()).parse().unwrap();
    let h: i64 = args.next().unwrap_or_else(|| "576".into()).parse().unwrap();
    let ratio: f32 = args.next().unwrap_or_else(|| "0.5".into()).parse().unwrap();
    let iters: usize = args.next().unwrap_or_else(|| "60".into()).parse().unwrap();

    ort::init().commit();

    let mut builder = Session::builder()?;
    builder = match ep_name.as_str() {
        "migraphx" => builder
            .with_execution_providers([ep::MIGraphX::default().with_fp16(true).build()])?,
        "cpu" => builder,
        other => panic!("unknown ep: {other}"),
    };

    println!("== {model_path}  [ep={ep_name}, {w}x{h}, downsample_ratio={ratio}]");
    let t0 = Instant::now();
    let mut session = builder.commit_from_file(&model_path)?;
    println!("   load+compile: {:.1}s", t0.elapsed().as_secs_f64());

    // Does this build want fp16 or fp32 tensors?
    let ValueType::Tensor { ty: src_ty, .. } = session.inputs()[0].dtype() else {
        panic!("unexpected src type")
    };
    let fp16 = matches!(src_ty, TensorElementType::Float16);
    println!("   src dtype: {src_ty:?}");

    // Recurrent state: on the dynamic graph RVM accepts 1x1x1x1 zeros for frame
    // one. On a frozen graph the shapes are pinned, so seed them from the model.
    let mut state: Vec<(Vec<i64>, Vec<f32>)> = (1..=4)
        .map(|n| {
            let name = format!("r{n}i");
            let inp = session
                .inputs()
                .iter()
                .find(|i| i.name() == name)
                .unwrap_or_else(|| panic!("missing input {name}"));
            let ValueType::Tensor { shape, .. } = inp.dtype() else { unreachable!() };
            if shape.iter().all(|&d| d > 0) {
                let s: Vec<i64> = shape.to_vec();
                let n: usize = s.iter().product::<i64>() as usize;
                (s, vec![0.0f32; n])
            } else {
                (vec![1, 1, 1, 1], vec![0.0f32])
            }
        })
        .collect();
    println!("   initial state: {:?}", state.iter().map(|(s, _)| s).collect::<Vec<_>>());

    let src_shape = vec![1i64, 3, h, w];
    let src_data = vec![0.5f32; (3 * h * w) as usize];

    let mut times: Vec<f64> = Vec::new();
    // 5 warmup frames (MIGraphX recompiles as state shapes settle), then measure.
    let total = iters + 5;
    for i in 0..total {
        let t = Instant::now();

        let mut inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = Vec::new();
        inputs.push((
            "src".into(),
            mk(fp16, src_shape.clone(), &src_data)?,
        ));
        for (n, (shape, data)) in state.iter().enumerate() {
            inputs.push((
                format!("r{}i", n + 1).into(),
                mk(fp16, shape.clone(), data)?,
            ));
        }
        // The frozen graph bakes downsample_ratio in as an initializer.
        if session.inputs().iter().any(|i| i.name() == "downsample_ratio") {
            inputs.push(("downsample_ratio".into(), mk(fp16, vec![1], &[ratio])?));
        }

        let outputs = session.run(inputs)?;

        // Feed recurrent state forward, exactly as a real video loop would.
        let mut next = Vec::with_capacity(4);
        for n in 0..4 {
            let name = format!("r{}o", n + 1);
            let v = &outputs[name.as_str()];
            let (shape, data) = extract(v, fp16)?;
            next.push((shape, data));
        }
        if i == 0 {
            println!("   state after frame 1: {:?}", next.iter().map(|(s, _)| s).collect::<Vec<_>>());
        }
        state = next;

        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= 5 {
            times.push(ms);
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    println!(
        "   >> median {:.1} ms  ({:.1} fps) | p95 {:.1} ms | min {:.1} | max {:.1}",
        median,
        1000.0 / median,
        times[(times.len() as f64 * 0.95) as usize],
        times[0],
        times[times.len() - 1]
    );
    Ok(())
}

fn mk<'a>(
    fp16: bool,
    shape: Vec<i64>,
    data: &[f32],
) -> ort::Result<ort::session::SessionInputValue<'a>> {
    Ok(if fp16 {
        let d: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
        Tensor::from_array((shape, d))?.into()
    } else {
        Tensor::from_array((shape, data.to_vec()))?.into()
    })
}

fn extract(v: &ort::value::DynValue, fp16: bool) -> ort::Result<(Vec<i64>, Vec<f32>)> {
    if fp16 {
        let (shape, data) = v.try_extract_tensor::<half::f16>()?;
        Ok((shape.to_vec(), data.iter().map(|x| x.to_f32()).collect()))
    } else {
        let (shape, data) = v.try_extract_tensor::<f32>()?;
        Ok((shape.to_vec(), data.to_vec()))
    }
}
