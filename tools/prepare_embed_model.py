"""Prepares the NPU embedding models and diagnoses fp16 accuracy (runs on GitHub's build server).

Outputs in OUT_DIR:
  minilm_ln_16x256.onnx      fp32, static 16x256, LayerNorm + Gelu fused, LayerNorm inputs pre-scaled,
                             attention scaling folded into Q (CPU reference, GPU, and NPU fp16 variant)
  minilm_a16w8_16x256.onnx   quantized for the NPU: 16-bit activations, 8-bit weights
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
from onnx import TensorProto, helper, numpy_helper, version_converter
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from tokenizers import Tokenizer

BATCH, SEQ = 16, 256
MASK_MIN = -100.0          # replaces float32-min padding constants; safe in fp16 and 16-bit quantization
FP16_MAX = 65504.0
LN_PRESCALE = 1.0 / 16     # LayerNorm is scale-invariant; this keeps its squared values far below fp16's limit

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


def prescale_layernorms(src, dst, scale):
    """Multiplies each LayerNormalization input by `scale`. LayerNorm(s*x) == LayerNorm(x), so results are
    unchanged, but the squares it sums internally shrink by scale**2 and can no longer overflow fp16."""
    m = onnx.load(src)
    g = m.graph
    g.initializer.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), "ln_prescale"))
    nodes, count = [], 0
    for n in g.node:
        if n.op_type == "LayerNormalization":
            scaled = "%s_prescaled_input" % (n.name or "ln%d" % count)
            nodes.append(helper.make_node("Mul", [n.input[0], "ln_prescale"], [scaled], name=scaled + "_mul"))
            n.input[0] = scaled
            count += 1
        nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    onnx.save(m, dst)
    return count


def fold_attention_scale(src, dst):
    """Moves the attention score scaling (MatMul -> Div/Mul by a constant) in front of the MatMul, onto Q.
    Same math, but the score tensor is scaled down before it is stored, so it cannot overflow fp16 and
    16-bit quantization gets finer steps. Returns the number of folds."""
    m = onnx.load(src)
    g = m.graph
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    producer = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    folds, remove = 0, set()
    for n in list(g.node):
        if n.op_type not in ("Div", "Mul") or len(n.input) != 2 or n.input[1] not in consts:
            continue
        c = consts[n.input[1]]
        if c.size != 1:
            continue
        mm = producer.get(n.input[0])
        if mm is None or mm.op_type != "MatMul" or len(consumers.get(mm.output[0], [])) != 1:
            continue
        if not any(x.op_type in ("Add", "Softmax", "Where") for x in consumers.get(n.output[0], [])):
            continue
        factor = (1.0 / float(c)) if n.op_type == "Div" else float(c)
        scale_name = "attn_scale_%d" % folds
        g.initializer.append(numpy_helper.from_array(np.array(factor, dtype=np.float32), scale_name))
        scaled = mm.input[0] + "_attn_scaled_%d" % folds
        mul = helper.make_node("Mul", [mm.input[0], scale_name], [scaled], name=scale_name + "_mul")
        idx = list(g.node).index(mm)
        g.node.insert(idx, mul)
        mm.input[0] = scaled
        mm.output[0] = n.output[0]
        remove.add(n.name or id(n))
        folds += 1
        n.op_type = "__REMOVE__"
    keep = [n for n in g.node if n.op_type != "__REMOVE__"]
    del g.node[:]
    g.node.extend(keep)
    onnx.save(m, dst)
    return folds


def softmax_chain(path, depth=4):
    """Op types feeding the first Softmax, for the build log (e.g. Add <- Div <- MatMul)."""
    g = onnx.load(path).graph
    producer = {o: n for n in g.node for o in n.output}
    sm = next((n for n in g.node if n.op_type == "Softmax"), None)
    chain, cur = [], sm
    while cur is not None and len(chain) < depth:
        chain.append(cur.op_type)
        cur = producer.get(cur.input[0]) if cur.input else None
    return " <- ".join(chain)


def worst_quantized_tensors(float_path, q_path, batch, tmp, top=8):
    """ONNX Runtime's quantization debugger: signal-to-noise (dB) of every internal value, lowest = worst."""
    from onnxruntime.quantization.qdq_loss_debug import (
        collect_activations, compute_activation_error, create_activation_matching,
        modify_model_output_intermediate_tensors)
    fa, qa = os.path.join(tmp, "float_aug.onnx"), os.path.join(tmp, "q_aug.onnx")
    modify_model_output_intermediate_tensors(float_path, fa)
    modify_model_output_intermediate_tensors(q_path, qa)
    f_act = collect_activations(fa, Reader([batch]))
    q_act = collect_activations(qa, Reader([batch]))
    errs = compute_activation_error(create_activation_matching(q_act, f_act))
    rows = [(name, e["xmodel_err"]) for name, e in errs.items() if "xmodel_err" in e]
    rows.sort(key=lambda r: r[1])
    return [{"tensor": n[:60], "snr_db": round(float(v), 1)} for n, v in rows[:top]]


def expose_internals(src, dst):
    """Adds internal tensors as extra model outputs so the app can compare CPU vs NPU step by step:
    every LayerNormalization output, plus the first Softmax (attention) and first Gelu output."""
    m = onnx.load(src)
    g = m.graph
    existing = {o.name for o in g.output}
    picks, ln_i, layer = [], 0, 0
    seen_softmax = seen_gelu = False
    for n in g.node:
        if n.op_type == "LayerNormalization":
            label = "embeddings LN" if ln_i == 0 else "layer %d %s LN" % ((ln_i - 1) // 2, "attention" if (ln_i - 1) % 2 == 0 else "output")
            picks.append((n.output[0], label))
            ln_i += 1
        elif n.op_type == "Softmax" and not seen_softmax:
            picks.append((n.output[0], "layer 0 attention probs"))
            seen_softmax = True
        elif n.op_type == "Gelu" and not seen_gelu:
            picks.append((n.output[0], "layer 0 Gelu"))
            seen_gelu = True
    for name, _ in picks:
        if name not in existing:
            g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    onnx.save(m, dst)
    return [{"name": n, "label": l} for n, l in picks]


def nodes_of_types(path, types):
    return [n.name for n in onnx.load(path).graph.node if n.op_type in types]


def internal_cosines(float_path, q_path, batch, names):
    """Per-checkpoint cosine (real tokens only) between the float model and a quantized model on CPU."""
    fs = ort.InferenceSession(float_path, providers=["CPUExecutionProvider"])
    qs = ort.InferenceSession(q_path, providers=["CPUExecutionProvider"])
    fo = dict(zip(names, fs.run(names, feeds_for(fs, batch))))
    qo = dict(zip(names, qs.run(names, feeds_for(qs, batch))))
    mask = batch["attention_mask"].astype(bool)
    out = {}
    for n in names:
        a, b = fo[n], qo[n]
        if a.ndim == 3:
            a, b = a[mask], b[mask]
        else:
            continue
        num = (a * b).sum(1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        out[n] = float((num / den).mean())
    return out


# Node-name patterns for each part of a BERT layer (names come from the Hugging Face export).
GROUPS = {
    "attn_qkv_proj":  lambda n: n.endswith(("query/MatMul", "key/MatMul", "value/MatMul")),
    "attn_scores":    lambda n: ("attention/self/MatMul" in n) or n.endswith("Softmax") or ("attention/self/Mul" in n) or ("attention/self/Add" in n),
    "attn_out_proj":  lambda n: "attention/output/dense/MatMul" in n,
    "ffn_up":         lambda n: "intermediate/dense/MatMul" in n,
    "ffn_down":       lambda n: ("output/dense/MatMul" in n) and ("attention" not in n),
    "embeddings":     lambda n: "embeddings" in n,
}

RECIPES = [("minmax", {}, {}, 6)] + [
    ("float_" + g, {"exclude_fn": fn}, {}, 6) for g, fn in GROUPS.items()
]


def nodes_matching(path, fn):
    return [n.name for n in onnx.load(path).graph.node if fn(n.name)]


def run_recipes(ln, calib_batches, test, ref, diag_names, tmp):
    rows = []
    for name, kwargs, extra, nbatches in RECIPES:
        q = os.path.join(tmp, "q_%s.onnx" % name)
        kw = dict(kwargs)
        exclude = kw.pop("exclude_types", None)
        if exclude:
            kw["nodes_to_exclude"] = nodes_of_types(ln, exclude)
        exclude_fn = kw.pop("exclude_fn", None)
        if exclude_fn:
            kw["nodes_to_exclude"] = nodes_matching(ln, exclude_fn)
            print("  %s: %d nodes kept in float" % (name, len(kw["nodes_to_exclude"])))
            if not kw["nodes_to_exclude"]:
                print("  (no node names matched; skipping)")
                rows.append({"recipe": name, "avg": 0.0, "worst": 0.0, "worst_layer": None, "path": None, "error": "no nodes matched"})
                continue
        try:
            cfg = get_qnn_qdq_config(ln, Reader(calib_batches[:nbatches]),
                                     activation_type=QuantType.QUInt16,
                                     weight_type=kw.pop("weight_type", QuantType.QUInt8), **kw)
            cfg.extra_options.update(extra)
            quantize(ln, q, cfg)
            avg, worst = cosine(ref, embed(q, test))
            layers = internal_cosines(ln, q, test[0], diag_names)
            first = min(layers.values()) if layers else None
            rows.append({"recipe": name, "avg": round(avg, 4), "worst": round(worst, 4),
                         "worst_layer": round(first, 4) if first is not None else None, "path": q})
            print("recipe %-22s final %.4f (worst doc %.4f)  worst checkpoint %.4f" % (name, avg, worst, first or 0))
        except Exception as e:
            print("recipe %-22s FAILED: %s" % (name, str(e)[:160]))
            rows.append({"recipe": name, "avg": 0.0, "worst": 0.0, "worst_layer": None, "path": None, "error": str(e)[:160]})
    return rows


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
    fused = os.path.join(tmp, "fused.onnx")
    if not qnn_preprocess_model(up, fused, fuse_layernorm=True):
        shutil.copy(up, fused)
    counts = op_counts(fused)
    report["fused"] = {k: counts.get(k, 0) for k in ("LayerNormalization", "Gelu", "Pow", "Erf")}
    print("ops after fusion:", report["fused"])

    pre = os.path.join(tmp, "prescaled.onnx")
    report["prescaled_layernorms"] = prescale_layernorms(fused, pre, LN_PRESCALE)
    report["prescale"] = LN_PRESCALE
    ref = embed(fixed, test)

    ln = os.path.join(out_dir, "minilm_ln_16x256.onnx")
    report["softmax_chain_before"] = softmax_chain(pre)
    print("ops feeding softmax before folding:", report["softmax_chain_before"])
    folded = fold_attention_scale(pre, ln)
    report["attention_scale_folded"] = folded
    if folded:
        cos_fold = cosine(ref, embed(ln, test))
        print("attention scale folded in %d layers; folded fp32 vs original:" % folded, cos_fold)
        if cos_fold[0] < 0.999:
            print("folding changed results, reverting")
            report["attention_scale_folded"] = 0
            shutil.copy(pre, ln)
    else:
        shutil.copy(pre, ln)
    report["softmax_chain_after"] = softmax_chain(ln)
    print("ops feeding softmax after folding:", report["softmax_chain_after"])
    plain = os.path.join(tmp, "plain.onnx")
    shutil.copy(ln, plain)
    report["diag_outputs"] = expose_internals(plain, ln)
    print("internal tensors exposed:", len(report["diag_outputs"]))
    report["ln_vs_original"] = cosine(ref, embed(ln, test))
    print("final fp32 model vs original (cosine avg, worst):", report["ln_vs_original"])

    # 4. Quantization recipe search: score each on CPU, keep the best
    calib12 = calib + [encode(tok, make_docs(BATCH, 3000 + i)) for i in range(6)]
    diag_names = [d["name"] for d in report["diag_outputs"]]
    sample_names = [n.name for n in onnx.load(ln).graph.node if n.op_type == "MatMul"][:8]
    print("sample MatMul node names:", sample_names)
    rows = run_recipes(ln, calib12, test, ref, diag_names, tmp)
    best = max(rows, key=lambda r: r["avg"])
    if best["path"] is None:
        sys.exit("every quantization recipe failed")
    q = os.path.join(out_dir, "minilm_a16w8_16x256.onnx")
    shutil.copy(best["path"], q)
    report["quant_recipe"] = best["recipe"]
    report["recipes"] = [{k: v for k, v in r.items() if k != "path"} for r in rows]
    report["a16w8_vs_original"] = (best["avg"], best["worst"])
    print("best recipe:", best["recipe"], best["avg"])
    outputs = [ln, q]

    with open(os.path.join(out_dir, "embed_variants.json"), "w") as fh:
        json.dump(report, fh)
    for f in outputs:
        print(os.path.basename(f), "%.1f MB" % (os.path.getsize(f) / 1e6))
    if report["ln_vs_original"][0] < 0.999:
        sys.exit("fused/pre-scaled model does not match the original")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
