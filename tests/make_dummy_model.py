"""Write a tiny *placeholder* ONNX model + labels.json into app/model/ for testing the app plumbing.

It only looks at the average colour of the photo, so its predictions are meaningless.
Built with a hand-written protobuf encoder, so it needs no onnx/torch install.

  python tests/make_dummy_model.py
"""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import load_diseases  # noqa: E402


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def key(field, wt):
    return varint(field << 3 | wt)


def f_varint(field, n):
    return key(field, 0) + varint(n)


def f_bytes(field, b):
    if isinstance(b, str):
        b = b.encode()
    return key(field, 2) + varint(len(b)) + b


def tensor(name, dims, values):
    return b"".join(f_varint(1, d) for d in dims) + f_varint(2, 1) + f_bytes(8, name) + \
        f_bytes(9, struct.pack(f"<{len(values)}f", *values))


def value_info(name, dims):
    shape = b"".join(f_bytes(1, f_bytes(2, d) if isinstance(d, str) else f_varint(1, d)) for d in dims)
    ttype = f_varint(1, 1) + f_bytes(2, shape)
    return f_bytes(1, name) + f_bytes(2, f_bytes(1, ttype))


def node(op, ins, outs, attrs=b""):
    return b"".join(f_bytes(1, i) for i in ins) + b"".join(f_bytes(2, o) for o in outs) + f_bytes(4, op) + attrs


def attr_int(name, v):
    return f_bytes(5, f_bytes(1, name) + f_varint(20, 2) + f_varint(3, v))


def main():
    dmap = {d["id"]: d for d in load_diseases()}
    manifest = ROOT / "data" / "manifest.csv"
    if manifest.exists():
        import csv
        classes = sorted({r["label"] for r in csv.DictReader(open(manifest, encoding="utf-8"))})
    else:
        classes = ["tomato.healthy", "tomato.late_blight", "tomato.early_blight", "maize.common_rust", "potato.late_blight"]
    C, img = len(classes), 224
    # weights: greener mean colour → "healthy" classes score higher, browner → disease classes
    W, B = [], []
    for c in classes:
        healthy = c.endswith(".healthy")
        W += ([-1.0, 2.0, -1.0] if healthy else [1.0, -1.0, 0.2])
        B.append(0.05 * (hash(c) % 7))
    graph = f_bytes(1, node("GlobalAveragePool", ["image"], ["pooled"])) \
      + f_bytes(1, node("Flatten", ["pooled"], ["flat"], attr_int("axis", 1))) \
      + f_bytes(1, node("Gemm", ["flat", "W", "B"], ["logits"], attr_int("transB", 1))) \
      + f_bytes(2, "dummy") + f_bytes(5, tensor("W", [C, 3], W)) + f_bytes(5, tensor("B", [C], B)) \
      + f_bytes(11, value_info("image", ["batch", 3, img, img])) + f_bytes(12, value_info("logits", ["batch", C]))
    model = f_varint(1, 8) + f_bytes(2, "sebel-hakim-dummy") + f_bytes(7, graph) + f_bytes(8, f_bytes(1, "") + f_varint(2, 17))
    out = ROOT / "app" / "model"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.onnx").write_bytes(model)
    (out / "labels.json").write_text(json.dumps({
        "model": "placeholder (colour only — not a real model)", "img": img,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "temperature": 1.0,
        "classes": [{"id": c, "crop": dmap[c]["crop"], "type": dmap[c]["type"]} for c in classes],
        "files": {"fp32": "model.onnx"}, "placeholder": True}, indent=1))
    print(f"placeholder model: {C} classes, {len(model)} bytes → {out}")


if __name__ == "__main__":
    main()
