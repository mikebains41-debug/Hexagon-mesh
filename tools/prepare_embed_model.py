"""Prepares the NPU embedding models and diagnoses fp16 accuracy (runs on GitHub's build server).

Outputs in OUT_DIR:
  minilm_ln_16x256.onnx     fp32, static 16x256, LayerNorm + Gelu fused (CPU reference and NPU fp16 variant)
  minilm_a16w8_16x256.onnx  quantized for the NPU: 16-bit activations, 8-bit weights
  embed_variants.json       build-server checks shown in the app

Usage: python tools/prepare_embed_model.py MODEL.onnx TOKENIZER.json OUT_DIR
"""
import json
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper, version_converter
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from tokenizers import Tokenizer

BATCH, SEQ = 16, 256
MASK_MIN = -100.0          # replaces float32-min padding constants; safe in fp16 and 16-bit quantization
FP16_MAX = 65504.0

WORDS = ("the a an and or but of to in on for with from by at as is are was were be been "
         "network phone data model energy system report customer market service order price value "
         "growth team product design process result policy research study analysis support update "
         "security privacy device battery signal cloud server compute memory storage speed quality "
         "city region company project meeting plan budget schedule contract partner supply demand "
         "review summary document translation transcript article email ticket request response "
         "people family school health travel weather music video image story history science "
         "quickly carefully often usually recently clearly strongly slowly directly easily "
         "important new large small local global public private final early recent common "
         "increase reduce improve provide include require create develop measure deliver share").split()


def make_docs(n, seed):
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        target, words = rng.randint(120, 320), []
        while len(words) < target:
            s = [rng.choice(WORDS) for _ in range(rng.randint(8, 20))]
            s[0] = s[0].capitalize()
            words.extend(s[:-1] + [s[-1] + "."])
        docs.append(" ".join(words))
    return docs


def encode(tok, docs):
    encs = tok.encode_batch(docs)
    ids = np.array([e.ids for e in encs], np.int64)
    mask = np.array([e.attention_mask for e in encs], np.int64)
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}


def feeds_for(sess, batch):
    return {i.name: batch.get(i.name, np.zeros((BATCH, SEQ), np.int64)) for i in sess.get_inputs()}


def embed(model_path, batches):
    s = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out = []
    for b in batches:
        y = s.run(None, feeds_for(s, b))[0]
        if y.ndim == 3:
            m = b["attention_mask"][..., None].astype(np.float32)
            y = (y * m).sum(1) / np.maximum(m.sum(1), 1.0)
        out.append(y / np.linalg.norm(y, axis=1, keepdims=True))
    return np.concatenate(out)


def cosine(a, b):
    c = (a * b).sum(1)
    return float(c.mean()), float(c.min())


def clamp_tensor(t):
    if t.data_type not in (TensorProto.FLOAT, TensorProto.DOUBLE):
        return None
    a = numpy_helper.to_array(t)
    if a.size == 0 or not (a < -FP16_MAX).any():
        return None
    return numpy_helper.from_array(np.maximum(a, MASK_MIN).astype(a.dtype), t.name)


def fix_shapes_and_clamp(m):
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
    return m, clamped


def layernorm_overflow_check(model_path, batch):
    """Largest squared value inside each decomposed LayerNorm (Pow input). Above 65,504 overflows fp16."""
    m = onnx.load(model_path)
    m = onnx.shape_inference.infer_shapes(m)
    types = {vi.name: vi for vi in m.graph.value_info}
    names = [n.input[0] for n in m.graph.node if n.op_type == "Pow"]
    existing = {o.name for o in m.graph.output}
    added = [n for n in names if n in types and n not in existing]
    for n in added:
        m.graph.output.append(types[n])
    if not added:
        return None
    path = os.path.join(tempfile.gettempdir(), "diag.onnx")
    onnx.save(m, path)
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = s.run(added, feeds_for(s, batch))
    return max(float(np.max(np.square(o.astype(np.float64)))) for o in outs)


class Reader(CalibrationDataReader):
    def __init__(self, batches):
        self.it = iter(batches)

    def get_next(self):
        return next(self.it, None)


def op_counts(path):
    counts = {}
    for n in onnx.load(path).graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    return counts


def main(src, tok_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    tok = Tokenizer.from_file(tok_path)
    tok.enable_truncation(max_length=SEQ)
    tok.enable_padding(length=SEQ, pad_id=0, pad_token="[PAD]")
    test = [encode(tok, make_docs(BATCH, 1000 + i)) for i in range(4)]
    calib = [encode(tok, make_docs(BATCH, 2000 + i)) for i in range(6)]
    report = {}

    # 1. Static shapes + fp16-safe padding constant
    m, clamped = fix_shapes_and_clamp(onnx.load(src))
    fixed = os.path.join(tmp, "fixed.onnx")
    onnx.save(m, fixed)
    report["clamped_constants"] = clamped
    print("clamped constants:", clamped)

    # 2. Diagnose: does LayerNorm overflow fp16 in the original model?
    try:
        mx = layernorm_overflow_check(fixed, test[0])
        report["max_layernorm_square"] = mx
        print("largest squared value inside LayerNorm:", mx, "(fp16 limit 65504)")
    except Exception as e:
        print("overflow check skipped:", e)

    # 3. Opset 17 + fuse LayerNorm and Gelu (NPU-native ops)
    m = onnx.load(fixed)
    opset = next(x.version for x in m.opset_import if x.domain in ("", "ai.onnx"))
    report["opset_from"] = opset
    if opset < 17:
        try:
            m = version_converter.convert_version(m, 17)
        except Exception as e:
            print("version converter failed (%s); setting opset 17 directly" % e)
            for x in m.opset_import:
                if x.domain in ("", "ai.onnx"):
                    x.version = 17
    up = os.path.join(tmp, "opset17.onnx")
    onnx.save(m, up)
    ln = os.path.join(out_dir, "minilm_ln_16x256.onnx")
    if not qnn_preprocess_model(up, ln, fuse_layernorm=True):
        shutil.copy(up, ln)
    counts = op_counts(ln)
    report["fused"] = {k: counts.get(k, 0) for k in ("LayerNormalization", "Gelu", "Pow", "Erf")}
    print("ops after fusion:", report["fused"])

    ref = embed(fixed, test)
    report["ln_vs_original"] = cosine(ref, embed(ln, test))
    print("fused fp32 vs original (cosine avg, worst):", report["ln_vs_original"])

    # 4. Quantize for the NPU: 16-bit activations, 8-bit weights
    q = os.path.join(out_dir, "minilm_a16w8_16x256.onnx")
    qcfg = get_qnn_qdq_config(ln, Reader(calib), activation_type=QuantType.QUInt16, weight_type=QuantType.QUInt8)
    quantize(ln, q, qcfg)
    report["a16w8_vs_original"] = cosine(ref, embed(q, test))
    print("quantized vs original (cosine avg, worst):", report["a16w8_vs_original"])

    with open(os.path.join(out_dir, "embed_variants.json"), "w") as fh:
        json.dump(report, fh)
    for f in (ln, q):
        print(os.path.basename(f), "%.1f MB" % (os.path.getsize(f) / 1e6))
    if report["ln_vs_original"][0] < 0.999:
        sys.exit("fused model does not match the original")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
