"""Train the plant-disease classifier on data/manifest.csv.

Recommended two-stage recipe for the best phone model:
  1. Teacher (big, accurate):
       python pipeline/train.py --model convnextv2_tiny.fcmae_ft_in22k_in1k --img 288 --epochs 30 --out runs/teacher
  2. Student (small, fast, distilled from the teacher; this one goes into the app):
       python pipeline/train.py --model efficientnet_b0.ra_in1k --img 224 --epochs 40 \
           --teacher runs/teacher/best.pt --out runs/student

Quick CPU smoke test:
       python pipeline/train.py --model mobilenetv3_small_100 --epochs 1 --max-steps 20 --bs 16 --workers 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from common import DATA_ROOT, ROOT

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data
def read_manifest(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ImgDS(Dataset):
    def __init__(self, rows, classes, tf):
        self.rows, self.tf = rows, tf
        self.idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
        return self.tf(im), self.idx[r["label"]]


def train_tf(img):
    # Heavy, phone-realistic augmentation: lab photos must look like field photos.
    return T.Compose([
        T.RandomResizedCrop(img, scale=(0.30, 1.0), ratio=(0.75, 1.33), antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.2),
        T.RandomApply([T.RandomRotation(25)], p=0.3),
        T.TrivialAugmentWide(),
        T.RandomApply([T.ColorJitter(0.35, 0.35, 0.3, 0.04)], p=0.6),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.25),
        T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, value="random"),
    ])


def eval_tf(img):
    return T.Compose([T.Resize(int(img * 1.14), antialias=True), T.CenterCrop(img),
                      T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


def sample_weights(rows, field_boost: float, balance_power: float):
    """Balance classes (count^-power), respect dataset weights and favour field photos."""
    cnt = Counter(r["label"] for r in rows)
    w = []
    for r in rows:
        x = float(r.get("weight") or 1.0) * cnt[r["label"]] ** -balance_power
        if r["field"] in ("field", "mixed"):
            x *= field_boost
        w.append(x)
    return torch.tensor(w, dtype=torch.double)


# ---------------------------------------------------------------- metrics
def macro_f1(y, p, n):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    present = cm.sum(1) > 0
    return float(f1[present].mean()) if present.any() else 0.0


@torch.no_grad()
def predict(model, loader, dev, amp):
    model.eval()
    if len(loader.dataset) == 0:
        return torch.empty(0, 1), torch.empty(0, dtype=torch.long)
    logits, ys = [], []
    for x, y in loader:
        with torch.autocast(dev.type, enabled=amp):
            logits.append(model(x.to(dev, non_blocking=True)).float().cpu())
        ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def fit_temperature(logits, y):
    """Temperature scaling on validation logits → calibrated confidences in the app."""
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DATA_ROOT / "manifest.csv"))
    ap.add_argument("--model", default="efficientnet_b0.ra_in1k")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=1.0, help="warm-up epochs")
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.5, help="probability of mixup/cutmix per batch")
    ap.add_argument("--field-boost", type=float, default=2.0)
    ap.add_argument("--balance-power", type=float, default=0.5)
    ap.add_argument("--epoch-size", type=int, default=0, help="samples per epoch (0 = size of train set)")
    ap.add_argument("--teacher", default="", help="checkpoint of a teacher model for distillation")
    ap.add_argument("--kd-alpha", type=float, default=0.7)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "model"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=0, help="stop after this many minutes (keeps the best model)")
    ap.add_argument("--pretrained-file", default="", help="local ImageNet weights (.pth/.safetensors) for nodes without internet")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(Path(a.manifest))
    classes = sorted({r["label"] for r in rows})
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    print(f"device={dev} classes={len(classes)} train={len(tr):,} val={len(va):,}")

    sampler = WeightedRandomSampler(sample_weights(tr, a.field_boost, a.balance_power),
                                    num_samples=a.epoch_size or len(tr), replacement=True)
    dl_tr = DataLoader(ImgDS(tr, classes, train_tf(a.img)), batch_size=a.bs, sampler=sampler,
                       num_workers=a.workers, pin_memory=amp, drop_last=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(ImgDS(va, classes, eval_tf(a.img)), batch_size=a.bs * 2,
                       num_workers=a.workers, pin_memory=amp)

    kw = {"pretrained_cfg_overlay": {"file": a.pretrained_file}} if a.pretrained_file else {}
    model = timm.create_model(a.model, pretrained=True, num_classes=len(classes), drop_rate=0.2,
                              drop_path_rate=0.1, **kw).to(dev)
    ema = None
    if a.ema > 0:
        try:
            ema = timm.utils.ModelEmaV3(model, decay=a.ema, use_warmup=True)
        except (AttributeError, TypeError):           # older timm
            ema = timm.utils.ModelEmaV2(model, decay=a.ema)

    teacher = None
    if a.teacher:
        ck = torch.load(a.teacher, map_location="cpu", weights_only=False)
        if ck["classes"] != classes:
            raise SystemExit("teacher was trained on different classes; retrain it on this manifest")
        teacher = timm.create_model(ck["model"], pretrained=False, num_classes=len(classes))
        teacher.load_state_dict(ck["state_dict"])
        teacher.eval().to(dev)
        t_img = ck["img"]
        print(f"distilling from {ck['model']} (img {t_img})")

    mix = timm.data.Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=a.mixup, label_smoothing=a.smoothing,
                          num_classes=len(classes)) if a.mixup > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = len(dl_tr)
    plan = {"total": a.epochs * steps_per_epoch}       # shortened below if a time limit applies
    warm = min(int(a.warmup * steps_per_epoch), 1000)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(plan["total"] - warm, 1))))))
    scaler = torch.amp.GradScaler(enabled=amp)

    best, step, log = -1.0, 0, []
    t_start = time.time()
    out_of_time = lambda: a.time_limit and (time.time() - t_start) / 60 > a.time_limit
    va_field = np.array([r["field"] in ("field", "mixed") for r in va])
    for ep in range(a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if mix:
                x, ys = mix(x, y)
            else:
                ys = F.one_hot(y, len(classes)).float() * (1 - a.smoothing) + a.smoothing / len(classes)
            with torch.autocast(dev.type, enabled=amp):
                logits = model(x)
                loss = torch.sum(-ys * F.log_softmax(logits.float(), -1), -1).mean()
                if teacher is not None:
                    with torch.no_grad():
                        xt = x if t_img == a.img else F.interpolate(x, size=t_img, mode="bilinear", align_corners=False)
                        tl = teacher(xt).float()
                    kd = F.kl_div(F.log_softmax(logits.float() / a.kd_temp, -1),
                                  F.softmax(tl / a.kd_temp, -1), reduction="batchmean") * a.kd_temp ** 2
                    loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model, step=step)
            run += loss.item()
            step += 1
            if a.time_limit and step == 100:
                # fit the cosine schedule into the time we have (keep ~12% for validation passes)
                per_step = (time.time() - t_start) / step
                fits = int(a.time_limit * 60 * 0.88 / per_step)
                if fits < plan["total"]:
                    plan["total"] = max(fits, warm + 1)
                    print(f"  {per_step:.3f}s/step → schedule set to {plan['total']:,} steps "
                          f"(≈{plan['total'] / steps_per_epoch:.1f} epochs) to fit {a.time_limit:.0f} min", flush=True)
            if i % 50 == 0:
                print(f"  ep {ep+1} step {i}/{steps_per_epoch} loss {run / (i + 1):.4f}", flush=True)
            if (a.max_steps and step >= a.max_steps) or out_of_time() or step >= plan["total"]:
                break

        net = ema.module if ema else model
        lg, yv = predict(net, dl_va, dev, amp)
        pv = lg.argmax(1).numpy() if len(yv) else np.zeros(0, dtype=np.int64)
        yv_np = yv.numpy()
        acc = float((pv == yv_np).mean()) if len(yv_np) else 0.0
        f1 = macro_f1(yv_np, pv, len(classes))
        f1_field = macro_f1(yv_np[va_field], pv[va_field], len(classes)) if va_field.any() else f1
        score = f1_field
        log.append({"epoch": ep + 1, "loss": run / max(i + 1, 1), "val_acc": acc, "val_f1": f1,
                    "val_f1_field": f1_field, "sec": round(time.time() - t0)})
        print(f"epoch {ep+1}/{a.epochs}  acc {acc:.4f}  macroF1 {f1:.4f}  field macroF1 {f1_field:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best = score
            temp = fit_temperature(lg, yv) if len(yv) else 1.0
            torch.save({"model": a.model, "img": a.img, "mean": MEAN, "std": STD, "classes": classes,
                        "temperature": temp, "state_dict": net.state_dict(), "epoch": ep + 1,
                        "val": log[-1]}, out / "best.pt")
            print(f"  saved best (field macroF1 {best:.4f}, temperature {temp:.2f})")
        (out / "log.json").write_text(json.dumps(log, indent=1))
        if (a.max_steps and step >= a.max_steps) or out_of_time() or step >= plan["total"]:
            if out_of_time():
                print(f"time limit of {a.time_limit:.0f} min reached; stopping with the best model so far")
            break
    print(f"done. best field macro-F1 {best:.4f} → {out / 'best.pt'}")


if __name__ == "__main__":
    main()
