//! THROWAWAY SPIKE. Measures raw ONNX inference throughput on the AMD iGPU
//! via the MIGraphX execution provider. Not production code.

use std::time::Instant;

use half::f16;
use ort::{
    ep,
    session::Session,
    session::SessionInputValue,
    value::{TensorElementType, ValueType},
};

/// Dynamic dims get resolved to this unless the model pins them.
const DEFAULT_SPATIAL: i64 = 512;

fn main() -> ort::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: bench <model.onnx> [ep] [iters] [spatial]");
    let ep_name = args.next().unwrap_or_else(|| "migraphx".into());
    let iters: usize = args.next().unwrap_or_else(|| "20".into()).parse().unwrap();
    let spatial: i64 = args
        .next()
        .unwrap_or_else(|| DEFAULT_SPATIAL.to_string())
        .parse()
        .unwrap();

    ort::init().commit();

    let mut builder = Session::builder()?;
    builder = match ep_name.as_str() {
        "migraphx" => builder.with_execution_providers([ep::MIGraphX::default()
            .with_fp16(true)
            .with_exhaustive_tune(false)
            .build()])?,
        "cpu" => builder,
        other => panic!("unknown ep: {other}"),
    };

    println!("== {model_path}  [ep={ep_name}, spatial={spatial}]");
    let load_start = Instant::now();
    let mut session = builder.commit_from_file(&model_path)?;
    println!("   load+compile: {:.1}s", load_start.elapsed().as_secs_f64());

    // Introspect I/O so we don't hardcode per-model shapes.
    let mut specs = Vec::new();
    for inp in session.inputs() {
        let ValueType::Tensor { ty, shape, .. } = inp.dtype() else {
            panic!("non-tensor input {}", inp.name());
        };
        let resolved: Vec<i64> = shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0 {
                    d
                } else if i == 0 {
                    1 // batch
                } else {
                    spatial
                }
            })
            .collect();
        println!("   in  {:24} {:?} {:?} -> {:?}", inp.name(), ty, shape.as_ref(), resolved);
        specs.push((inp.name().to_string(), *ty, resolved));
    }
    for out in session.outputs() {
        println!("   out {:24} {:?}", out.name(), out.dtype());
    }

    let build_inputs = || -> ort::Result<Vec<(std::borrow::Cow<'static, str>, SessionInputValue<'static>)>> {
        let mut v = Vec::new();
        for (name, ty, shape) in &specs {
            let n: usize = shape.iter().product::<i64>() as usize;
            let val: SessionInputValue<'static> = match ty {
                TensorElementType::Float32 => {
                    ort::value::Tensor::from_array((shape.clone(), vec![0.5f32; n]))?.into()
                }
                TensorElementType::Float16 => {
                    ort::value::Tensor::from_array((shape.clone(), vec![f16::from_f32(0.5); n]))?.into()
                }
                other => panic!("unhandled input dtype {other:?}"),
            };
            v.push((std::borrow::Cow::Owned(name.clone()), val));
        }
        Ok(v)
    };

    // Warmup — first runs include kernel autotuning.
    for _ in 0..3 {
        let _ = session.run(build_inputs()?)?;
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let inputs = build_inputs()?;
        let t = Instant::now();
        let outputs = session.run(inputs)?;
        std::hint::black_box(&outputs);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "   >> median {:.1} ms  ({:.1} fps) | mean {:.1} ms | min {:.1} | max {:.1}",
        median,
        1000.0 / median,
        mean,
        times[0],
        times[times.len() - 1]
    );
    Ok(())
}
