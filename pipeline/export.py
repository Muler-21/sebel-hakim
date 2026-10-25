"""Export a trained checkpoint for the app.

Writes app/model/model.onnx (+ model.int8.onnx when --quantize) and app/model/labels.json,
then checks the ONNX output matches PyTorch.

  python pipeline/export.py runs/student/best.pt [--quantize]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch

from common import ROOT, load_diseases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=str(ROOT / "app" / "model"))
    ap.add_argument("--quantize", action="store_true", help="also write an int8 model (about 4x smaller)")
    ap.add_argument("--calib", type=int, default=300, help="calibration images for int8")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = timm.create_model(ck["model"], pretrained=False, num_classes=len(ck["classes"]), exportable=True)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 3, ck["img"], ck["img"])
    onnx_path = out / "model.onnx"
    kw = dict(input_names=["image"], output_names=["logits"],
              dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, x, str(onnx_path), dynamo=False, **kw)
    except TypeError:                      # torch < 2.5 has no `dynamo` argument
        torch.onnx.export(model, x, str(onnx_path), **kw)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"onnx check: max |diff| = {diff:.2e}")
    if diff > 1e-2:
        raise SystemExit("ONNX output differs from PyTorch; not exporting")

    dmap = {d["id"]: d for d in load_diseases()}
    labels = {
        "model": ck["model"], "img": ck["img"], "mean": list(ck["mean"]), "std": list(ck["std"]),
        "temperature": ck["temperature"], "epoch": ck.get("epoch"), "val": ck.get("val"),
        "classes": [{"id": c, "crop": dmap[c]["crop"], "type": dmap[c]["type"]} for c in ck["classes"]],
        "files": {"fp32": "model.onnx"},
    }

    if a.quantize:
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process
        import csv
        from PIL import Image
        from train import eval_tf
        rows = [r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8")) if r["split"] == "train"]
        rng = np.random.default_rng(0)
        pick = [rows[i] for i in rng.choice(len(rows), min(a.calib, len(rows)), replace=False)]
        tf = eval_tf(ck["img"])

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter(pick)

            def get_next(self):
                r = next(self.it, None)
                if r is None:
                    return None
                with Image.open(r["path"]) as im:
                    return {"image": tf(im.convert("RGB")).unsqueeze(0).numpy()}

        pre = out / "model.pre.onnx"
        quant_pre_process(str(onnx_path), str(pre))
        q = out / "model.int8.onnx"
        quantize_static(str(pre), str(q), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
        pre.unlink()
        labels["files"]["int8"] = "model.int8.onnx"
        print(f"int8 model: {q.stat().st_size / 1e6:.1f} MB (check its accuracy with evaluate.py before using it)")

    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"fp32 model: {onnx_path.stat().st_size / 1e6:.1f} MB, {len(ck['classes'])} classes → {out}")


if __name__ == "__main__":
    main()
