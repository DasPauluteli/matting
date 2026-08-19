use anyhow::{anyhow, Context, Result};
use onnx_protobuf::{tensor_proto::DataType, ModelProto, TensorProto};

/// Concrete shapes for RVM's four recurrent state tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateShapes {
    pub channels: [i64; 4],
    /// (height, width) per state level.
    pub spatial: [(i64, i64); 4],
}

/// RVM declares its recurrent channel widths statically on the r1o..r4o
/// outputs even when everything else is dynamic. Read them from there rather
/// than assuming a backbone: MobileNetV3 and ResNet50 differ.
pub fn read_state_channels(model: &ModelProto) -> Result<[i64; 4]> {
    let graph = model.graph.as_ref().context("model has no graph")?;
    let mut channels = [0i64; 4];
    for (i, ch) in channels.iter_mut().enumerate() {
        let name = format!("r{}o", i + 1);
        let out = graph
            .output
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| anyhow!("model has no output {name}; is this an RVM export?"))?;
        let dim = &out
            .type_
            .as_ref()
            .context("output has no type")?
            .tensor_type()
            .shape
            .dim;
        if dim.len() != 4 {
            return Err(anyhow!("{name} should be rank 4, got {}", dim.len()));
        }
        if !dim[1].has_dim_value() || dim[1].dim_value() <= 0 {
            return Err(anyhow!("{name} has no static channel dim"));
        }
        *ch = dim[1].dim_value();
    }
    Ok(channels)
}

pub fn state_shapes(model: &ModelProto, width: u32, height: u32, ratio: f32) -> Result<StateShapes> {
    let channels = read_state_channels(model)?;
    let dw = (width as f32 * ratio) as i64;
    let dh = (height as f32 * ratio) as i64;
    let mut spatial = [(0i64, 0i64); 4];
    for (i, s) in spatial.iter_mut().enumerate() {
        let stride = 1i64 << (i + 1); // /2, /4, /8, /16 of the downsampled size
        *s = (div_ceil(dh, stride), div_ceil(dw, stride));
    }
    Ok(StateShapes { channels, spatial })
}

fn div_ceil(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// Rewrite a stock RVM graph into one MIGraphX can compile:
/// demote `downsample_ratio` to a constant initializer and pin every input dim.
pub fn freeze_model(mut model: ModelProto, width: u32, height: u32, ratio: f32) -> Result<ModelProto> {
    let shapes = state_shapes(&model, width, height, ratio)?;
    let graph = model.graph.as_mut().context("model has no graph")?;

    // 1. downsample_ratio: graph input -> initializer. This is what makes the
    //    internal Resize ops constant, which MIGraphX requires.
    let before = graph.input.len();
    graph.input.retain(|i| i.name != "downsample_ratio");
    if graph.input.len() == before {
        return Err(anyhow!("model has no downsample_ratio input; already frozen?"));
    }
    let mut ratio_init = TensorProto::new();
    ratio_init.name = "downsample_ratio".to_string();
    ratio_init.data_type = DataType::FLOAT as i32;
    ratio_init.dims = vec![1];
    ratio_init.float_data = vec![ratio];
    graph.initializer.push(ratio_init);

    // 2/3. Pin src and the four recurrent states.
    let mut wanted: Vec<(String, Vec<i64>)> =
        vec![("src".to_string(), vec![1, 3, height as i64, width as i64])];
    for i in 0..4 {
        let (h, w) = shapes.spatial[i];
        wanted.push((format!("r{}i", i + 1), vec![1, shapes.channels[i], h, w]));
    }

    for (name, dims) in wanted {
        let input = graph
            .input
            .iter_mut()
            .find(|i| i.name == name)
            .ok_or_else(|| anyhow!("model has no input {name}"))?;
        // `tensor_type` is a protobuf oneof, so it is reached via the generated
        // `mut_tensor_type()` accessor rather than as a plain field.
        let shape = input
            .type_
            .as_mut()
            .context("input has no type")?
            .mut_tensor_type()
            .shape
            .mut_or_insert_default();
        if shape.dim.len() != dims.len() {
            return Err(anyhow!(
                "{name} should be rank {}, got {}",
                dims.len(),
                shape.dim.len()
            ));
        }
        for (d, v) in shape.dim.iter_mut().zip(dims) {
            d.clear_dim_param();
            d.set_dim_value(v);
        }
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use protobuf::Message;

    /// Stock RVM exports live outside the repo. Skip rather than fail when absent.
    fn load(path: &str) -> Option<onnx_protobuf::ModelProto> {
        let bytes = std::fs::read(path).ok()?;
        onnx_protobuf::ModelProto::parse_from_bytes(&bytes).ok()
    }

    const MNV3: &str = concat!(env!("HOME"), "/Downloads/rvm_mobilenetv3_fp32.onnx");
    const R50: &str = concat!(
        env!("HOME"),
        "/.config/obs-studio/plugins/obs-ai-matting/models/rvm_resnet50.onnx"
    );

    #[test]
    fn reads_mobilenetv3_channels_from_model() {
        let Some(m) = load(MNV3) else { return };
        assert_eq!(read_state_channels(&m).unwrap(), [16, 20, 40, 64]);
    }

    #[test]
    fn reads_resnet50_channels_from_model() {
        let Some(m) = load(R50) else { return };
        assert_eq!(read_state_channels(&m).unwrap(), [16, 32, 64, 128]);
    }

    #[test]
    fn computes_state_spatial_dims() {
        let Some(m) = load(MNV3) else { return };
        let s = state_shapes(&m, 1024, 576, 0.5).unwrap();
        // Downsampled is 512x288; states sit at /2, /4, /8, /16 of that.
        assert_eq!(s.spatial, [(144, 256), (72, 128), (36, 64), (18, 32)]);
    }

    #[test]
    fn frozen_model_has_no_downsample_ratio_input() {
        let Some(m) = load(MNV3) else { return };
        let f = freeze_model(m, 1024, 576, 0.5).unwrap();
        let g = f.graph.as_ref().unwrap();
        assert!(
            !g.input.iter().any(|i| i.name == "downsample_ratio"),
            "downsample_ratio must not remain a graph input"
        );
        assert!(
            g.initializer.iter().any(|i| i.name == "downsample_ratio"),
            "downsample_ratio must become an initializer"
        );
    }

    #[test]
    fn frozen_model_pins_all_input_dims() {
        let Some(m) = load(MNV3) else { return };
        let f = freeze_model(m, 1024, 576, 0.5).unwrap();
        let g = f.graph.as_ref().unwrap();
        for input in &g.input {
            let dims = &input.type_.as_ref().unwrap().tensor_type().shape.dim;
            for d in dims {
                assert!(
                    d.has_dim_value() && d.dim_value() > 0,
                    "input {} has an unpinned dim",
                    input.name
                );
            }
        }
        let src = g.input.iter().find(|i| i.name == "src").unwrap();
        let dims: Vec<i64> = src
            .type_
            .as_ref()
            .unwrap()
            .tensor_type()
            .shape
            .dim
            .iter()
            .map(|d| d.dim_value())
            .collect();
        assert_eq!(dims, vec![1, 3, 576, 1024]);
    }
}
