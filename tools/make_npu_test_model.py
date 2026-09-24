"""Writes the NPU smoke-test model: a 6-layer network (MatMul + Add + Relu), fixed shape 64x768.

Pure Python, no dependencies: encodes the ONNX protobuf by hand.
Usage: python tools/make_npu_test_model.py OUTPUT.onnx
"""
import array
import random
import struct
import sys

BATCH, DIM, LAYERS = 64, 768, 6


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def field_bytes(num, data):          # length-delimited field
    return varint((num << 3) | 2) + varint(len(data)) + data


def field_int(num, value):           # varint field
    return varint(num << 3) + varint(value)


def s(text):
    return text.encode()


def tensor(name, dims, floats):
    packed = b"".join(varint(d) for d in dims)
    return (field_bytes(1, packed) + field_int(2, 1)            # dims, FLOAT
            + field_bytes(8, s(name)) + field_bytes(9, floats.tobytes()))


def value_info(name, dims):
    shape = b"".join(field_bytes(1, field_int(1, d)) for d in dims)
    tensor_type = field_int(1, 1) + field_bytes(2, shape)       # elem FLOAT, shape
    return field_bytes(1, s(name)) + field_bytes(2, field_bytes(1, tensor_type))


def node(op, inputs, outputs, name):
    body = b"".join(field_bytes(1, s(i)) for i in inputs)
    body += b"".join(field_bytes(2, s(o)) for o in outputs)
    return body + field_bytes(3, s(name)) + field_bytes(4, s(op))


def build():
    rng = random.Random(0)
    scale = (1.0 / DIM) ** 0.5
    nodes, inits, x = [], [], "input"
    for i in range(LAYERS):
        w = array.array("f", (rng.gauss(0, 1) * scale for _ in range(DIM * DIM)))
        b = array.array("f", (rng.gauss(0, 0.01) for _ in range(DIM)))
        if sys.byteorder != "little":
            w.byteswap()
            b.byteswap()
        inits += [tensor("W%d" % i, [DIM, DIM], w), tensor("b%d" % i, [DIM], b)]
        last = i == LAYERS - 1
        nodes.append(node("MatMul", [x, "W%d" % i], ["mm%d" % i], "matmul%d" % i))
        nodes.append(node("Add", ["mm%d" % i, "b%d" % i], ["output" if last else "add%d" % i], "add%d" % i))
        if not last:
            nodes.append(node("Relu", ["add%d" % i], ["h%d" % i], "relu%d" % i))
            x = "h%d" % i
    graph = b"".join(field_bytes(1, n) for n in nodes)
    graph += field_bytes(2, s("hexagon_npu_test"))
    graph += b"".join(field_bytes(5, t) for t in inits)
    graph += field_bytes(11, value_info("input", [BATCH, DIM]))
    graph += field_bytes(12, value_info("output", [BATCH, DIM]))
    opset = field_bytes(1, b"") + field_int(2, 17)
    return (field_int(1, 8) + field_bytes(2, s("hexagon-mesh"))   # ir_version 8, producer
            + field_bytes(7, graph) + field_bytes(8, opset))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "npu_test.onnx"
    data = build()
    with open(out, "wb") as fh:
        fh.write(data)
    print("wrote %s (%.1f MB, %d layers, %dx%d)" % (out, len(data) / 1e6, LAYERS, BATCH, DIM))
