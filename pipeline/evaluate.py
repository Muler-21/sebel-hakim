"""Evaluate a checkpoint on the held-out test split.

Reports accuracy and macro-F1 overall, on field photos only, per dataset, per crop and per class,
plus the most common confusions and how often a "not sure" warning would be shown.

  python pipeline/evaluate.py runs/student/best.pt            # PyTorch checkpoint
  python pipeline/evaluate.py app/model/model.onnx            # exported ONNX model (what the app runs)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ROOT
from train import ImgDS, eval_tf, macro_f1, read_manifest


def load_runner(path: Path):
    """Return (classes, img, temperature, fn(batch_tensor)->logits)."""
    if path.suffix == ".onnx":
        import onnxruntime as ort
        meta = json.loads((path.parent / "labels.json").read_text())
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return ([c["id"] for c in meta["classes"]], meta["img"], meta["temperature"],
                lambda x: torch.from_numpy(sess.run(None, {name: x.numpy()})[0]))
    import timm
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = timm.create_model(ck["model"], pretrained=False, num_classes=len(ck["classes"]))
    m.load_state_dict(ck["state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.eval().to(dev)

    @torch.no_grad()
    def run(x):
        return m(x.to(dev)).float().cpu()
    return ck["classes"], ck["img"], ck["temperature"], run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=0.5, help="'not sure' below this confidence")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    path = Path(a.model)
    classes, img, temp, run = load_runner(path)
    rows = [r for r in read_manifest(Path(a.manifest)) if r["split"] == a.split and r["label"] in classes]
    dl = DataLoader(ImgDS(rows, classes, eval_tf(img)), batch_size=a.bs, num_workers=a.workers)
    probs, ys = [], []
    for x, y in dl:
        probs.append(torch.softmax(run(x) / temp, -1))
        ys.append(y)
    P, y = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    p = P.argmax(1)
    conf = P.max(1)
    top3 = (np.argsort(-P, 1)[:, :3] == y[:, None]).any(1)
    n = len(classes)

    def block(mask):
        if not mask.any():
            return None
        return {"images": int(mask.sum()), "acc": float((p[mask] == y[mask]).mean()),
                "top3": float(top3[mask].mean()), "macro_f1": macro_f1(y[mask], p[mask], n)}

    field = np.array([r["field"] in ("field", "mixed") for r in rows])
    crop_of = np.array([c.split(".")[0] for c in classes])
    res = {"model": str(path), "split": a.split, "overall": block(np.ones_like(y, bool)),
           "field_only": block(field), "lab_only": block(~field),
           "crop_correct": float((crop_of[p] == crop_of[y]).mean()),
           "not_sure_rate": float((conf < a.threshold).mean()),
           "acc_when_sure": float((p[conf >= a.threshold] == y[conf >= a.threshold]).mean()) if (conf >= a.threshold).any() else None,
           "per_dataset": {}, "per_crop": {}, "per_class": {}, "top_confusions": []}
    ds = np.array([r["dataset"] for r in rows])
    for d in sorted(set(ds)):
        res["per_dataset"][d] = block(ds == d)
    ycrop = crop_of[y]
    for c in sorted(set(ycrop)):
        res["per_crop"][c] = block(ycrop == c)
    for i, c in enumerate(classes):
        m = y == i
        if m.any():
            tp = int(((p == i) & m).sum())
            res["per_class"][c] = {"images": int(m.sum()), "recall": tp / int(m.sum()),
                                   "precision": tp / max(int((p == i).sum()), 1)}
    conf_pairs = Counter((classes[a_], classes[b_]) for a_, b_ in zip(y, p) if a_ != b_)
    res["top_confusions"] = [{"true": t, "predicted": q, "count": k} for (t, q), k in conf_pairs.most_common(25)]

    out = Path(a.out) if a.out else ROOT / "reports" / f"eval_{path.parent.name}_{a.split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))

    def fmt(b):
        return "n/a" if not b else f"acc {b['acc']:.3f} · top-3 {b['top3']:.3f} · macro-F1 {b['macro_f1']:.3f} ({b['images']:,} imgs)"
    print(f"overall     {fmt(res['overall'])}")
    print(f"field only  {fmt(res['field_only'])}")
    print(f"lab only    {fmt(res['lab_only'])}")
    print(f"crop right  {res['crop_correct']:.3f}   'not sure' shown {res['not_sure_rate']:.1%}   "
          f"accuracy when sure {res['acc_when_sure'] if res['acc_when_sure'] is None else round(res['acc_when_sure'], 3)}")
    print("per crop:")
    for c, b in res["per_crop"].items():
        print(f"  {c:11} {fmt(b)}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
