"""Export BackgroundMattingV2 to ONNX with fully static shapes.

The upstream export_onnx.py marks batch/height/width as dynamic axes, which
MIGraphX cannot compile. Exporting at one fixed resolution removes the problem
at the source, rather than needing the graph surgery RVM required.
"""
import argparse
import sys

import torch

sys.path.insert(0, "/tmp/bgm/BackgroundMattingV2")
from model import MattingRefine  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="/tmp/bgm/pytorch_resnet50.pth")
p.add_argument("--backbone", default="resnet50")
p.add_argument("--width", type=int, default=1024)
p.add_argument("--height", type=int, default=576)
p.add_argument("--backbone-scale", type=float, default=0.25)
p.add_argument("--refine-mode", default="sampling", choices=["full", "sampling", "thresholding"])
p.add_argument("--sample-pixels", type=int, default=80_000)
p.add_argument("--crop-method", default="roi_align", choices=["unfold", "roi_align", "gather"])
p.add_argument("--replace-method", default="scatter_element", choices=["scatter_nd", "scatter_element"])
p.add_argument("--opset", type=int, default=12)
p.add_argument("--out", required=True)
a = p.parse_args()

model = MattingRefine(
    backbone=a.backbone,
    backbone_scale=a.backbone_scale,
    refine_mode=a.refine_mode,
    refine_sample_pixels=a.sample_pixels,
    refine_threshold=0.1,
    refine_kernel_size=3,
    refine_patch_crop_method=a.crop_method,
    refine_patch_replace_method=a.replace_method,
)
model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"), strict=False)
model.eval()

src = torch.randn(1, 3, a.height, a.width)
bgr = torch.randn(1, 3, a.height, a.width)

print(f"exporting {a.backbone} at {a.width}x{a.height}, scale={a.backbone_scale}, "
      f"refine={a.refine_mode}/{a.crop_method}/{a.replace_method}")

torch.onnx.export(
    model,
    (src, bgr),
    a.out,
    verbose=False,
    opset_version=a.opset,
    do_constant_folding=True,
    input_names=["src", "bgr"],
    output_names=["pha", "fgr", "pha_sm", "fgr_sm", "err_sm", "ref_sm"],
    # No dynamic_axes: every dimension is baked in.
    dynamo=False,
)
print("wrote", a.out)
