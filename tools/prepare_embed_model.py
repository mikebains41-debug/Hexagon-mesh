"""Prepares the NPU embedding model (runs on GitHub's build server).

1. Fixes all inputs of all-MiniLM-L6-v2 to a static 16 x 256 shape (the NPU needs fixed shapes).
2. Clamps huge negative constants (used to mask padding) that overflow fp16 on the NPU.
3. Test-runs the result on CPU, including a padded batch, before it goes into the app.

Usage: python tools/prepare_embed_model.py IN.onnx OUT.onnx
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

BATCH, SEQ = 16, 256
FP16_SAFE_MIN = -1.0e4


def clamp_tensor(t):
    if t.data_type not in (TensorProto.FLOAT, TensorProto.DOUBLE):
        return None
    a = numpy_helper.to_array(t)
    if a.size == 0 or not (a < -65504).any():
        return None
    return numpy_helper.from_array(np.maximum(a, FP16_SAFE_MIN).astype(a.dtype), t.name)


def main(src, dst):
    m = onnx.load(src)
    for inp in m.graph.input:
        dims = inp.type.tensor_type.shape.dim
        for i, value in enumerate((BATCH, SEQ)):
            if i < len(dims):
                dims[i].Clear()
                dims[i].dim_value = value
    for out in m.graph.output:
        for d in out.type.tensor_type.shape.dim:
            if d.HasField("dim_param"):
                d.Clear()
    del m.graph.value_info[:]

    clamped = 0
    for i, init in enumerate(m.graph.initializer):
        new = clamp_tensor(init)
        if new is not None:
            m.graph.initializer[i].CopyFrom(new)
            clamped += 1
    for node in m.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                new = clamp_tensor(attr.t)
                if new is not None:
                    attr.t.CopyFrom(new)
                    clamped += 1

    m = onnx.shape_inference.infer_shapes(m)
    onnx.save(m, dst)
    print("inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("outputs:", [o.name for o in m.graph.output])
    print("fp16-unsafe constants clamped:", clamped)

    s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    mask = np.ones((BATCH, SEQ), np.int64)
    mask[:, SEQ // 2:] = 0                       # half padding, exercises the clamped mask
    feeds = {}
    for i in s.get_inputs():
        if i.name == "attention_mask":
            feeds[i.name] = mask
        elif i.name == "input_ids":
            feeds[i.name] = np.full((BATCH, SEQ), 2000, np.int64)
        else:
            feeds[i.name] = np.zeros((BATCH, SEQ), np.int64)
    out = s.run(None, feeds)[0]
    print("test run output:", out.shape, "finite:", bool(np.isfinite(out).all()))
    if not np.isfinite(out).all():
        sys.exit("model produced non-finite values")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
