#!/usr/bin/env python
"""THROWAWAY SPIKE. Turn RVM's dynamic ONNX into a fully static graph.

MIGraphX rejects RVM because `downsample_ratio` is a runtime input, which makes
the internal Resize ops non-constant ("linear mode not supported for
non-constant inputs"). Here we demote downsample_ratio to an initializer and
pin src/recurrent-state to concrete shapes, then constant-fold.
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxsim import simplify

src_path, dst_path, W, H, RATIO = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])

model = onnx.load(src_path)
graph = model.graph

# Which float type does this build use?
elem_type = next(i.type.tensor_type.elem_type for i in graph.input if i.name == "src")
np_dtype = np.float16 if elem_type == onnx.TensorProto.FLOAT16 else np.float32

# 1. downsample_ratio: graph input -> constant initializer.
graph.input.remove(next(i for i in graph.input if i.name == "downsample_ratio"))
graph.initializer.append(
    numpy_helper.from_array(np.array([RATIO], dtype=np_dtype), name="downsample_ratio")
)

# 2. Pin src to the exact capture resolution.
src = next(i for i in graph.input if i.name == "src")
for dim, val in zip(src.type.tensor_type.shape.dim, (1, 3, H, W)):
    dim.ClearField("dim_param")
    dim.dim_value = val

# 3. Pin the recurrent state to its steady-state shapes. RVM's four ConvGRU
#    states sit at successive /2 strides of the *downsampled* resolution,
#    with fixed channel counts.
dw, dh = int(W * RATIO), int(H * RATIO)
# MobileNetV3 and ResNet50 decoders carry different ConvGRU widths.
state_channels = [int(c) for c in sys.argv[6].split(",")] if len(sys.argv) > 6 else [16, 20, 40, 64]
overrides = {}
for idx, ch in enumerate(state_channels, start=1):
    # r1 is at stride 2 of the downsampled input, then /2 each level.
    stride = 2**idx
    sh = (1, ch, -(-dh // stride), -(-dw // stride))
    overrides[f"r{idx}i"] = sh
    inp = next(i for i in graph.input if i.name == f"r{idx}i")
    for dim, val in zip(inp.type.tensor_type.shape.dim, sh):
        dim.ClearField("dim_param")
        dim.dim_value = val

print(f"src -> (1,3,{H},{W})  ratio={RATIO}  downsampled={dw}x{dh}")
for k, v in overrides.items():
    print(f"  {k} -> {v}")

model, ok = simplify(model, overwrite_input_shapes={"src": [1, 3, H, W], **overrides})
print("simplify ok:", ok)
onnx.save(model, dst_path)
print("wrote", dst_path)
