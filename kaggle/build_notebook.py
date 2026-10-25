"""Build kaggle/agrodoc_kaggle.ipynb: one self-contained notebook that trains AgroDoc on Kaggle's free GPU.

  python kaggle/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["pipeline/common.py", "pipeline/download.py", "pipeline/prepare.py", "pipeline/train.py",
         "pipeline/evaluate.py", "pipeline/export.py", "pipeline/build_kb.py", "pipeline/datasets.yaml",
         "kb/crops.yaml", "kb/cereals.yaml", "kb/legumes_roots.yaml", "kb/vegetables.yaml",
         "kb/fruits_cash.yaml", "tests/test_matcher.py"]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text, hidden=False):
    meta = {"_kg_hide-input": True, "jupyter": {"source_hidden": True}} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# AgroDoc — train the plant-disease model on Kaggle's free GPU

This notebook downloads the public plant-disease datasets, cleans them, trains a large **teacher** model,
distils it into a small **phone** model, tests it on held-out photos and exports it for the AgroDoc app.

**Before you press "Save Version → Save & Run All":**
1. Right panel → **Session options**: Accelerator **GPU T4 ×2** (or P100), **Internet on**.
   (Both need a phone-verified Kaggle account: Settings → Phone verification.)
2. Right panel → **Add Input** → add these (open each competition page once and click **Join / I understand and accept** first):
   - Competition `cassava-leaf-disease-classification`
   - Competition `paddy-disease-classification`
   - Competition `plant-pathology-2021-fgvc8`
   - Dataset `smaranjitghose/corn-or-maize-leaf-disease-dataset`
3. Save Version → **Save & Run All (Commit)**. It runs in the background (about 7–9 hours); you can close the browser.
4. When it finishes: **Output** tab → download **`agrodoc_model.zip`** and send it to Claude.

Inputs you could not add are skipped automatically; the other datasets still train.
""")

code("""
# ---------------- settings (the defaults fit in one 12-hour Kaggle session) ----------------
USE_TEACHER   = True                                   # big teacher → better small model
TEACHER       = "convnextv2_tiny.fcmae_ft_in22k_in1k"
TEACHER_IMG   = 256
TEACHER_MIN   = 150                                    # minutes of training
STUDENT       = "efficientnet_b0.ra_in1k"              # the model that goes into the phone app
STUDENT_IMG   = 224
STUDENT_MIN   = 170
EPOCH_SIZE    = 60000                                  # images per epoch (balanced sampling)
RESIZE        = 320                                    # shrink photos once → much faster training
SKIP          = []                                     # dataset ids to skip, e.g. ["plantwild", "plantseg"]
WORKERS       = 4

import os, pathlib
CODE = pathlib.Path("/kaggle/working/agrodoc")
DATA = pathlib.Path("/kaggle/temp/agrodoc_data") if pathlib.Path("/kaggle/temp").exists() else pathlib.Path("/tmp/agrodoc_data")
RUNS = DATA / "runs"
for p in [CODE / "pipeline", CODE / "kb", CODE / "tests", CODE / "app", CODE / "reports", DATA, RUNS]:
    p.mkdir(parents=True, exist_ok=True)
os.environ["AGRODOC_DATA"] = str(DATA)
os.chdir(CODE)
print("code:", CODE, "| data:", DATA)
""")

code("""
# ---------------- check GPU, internet and attached inputs ----------------
import subprocess, urllib.request, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — turn on GPU in Session options!")
try:
    urllib.request.urlopen("https://github.com", timeout=10); print("Internet: on")
except Exception:
    print("Internet: OFF — turn it on in Session options (needs phone verification)")
print("Attached inputs:", sorted(p.name for p in pathlib.Path("/kaggle/input").glob("*")) if pathlib.Path("/kaggle/input").exists() else "none")
subprocess.run("pip install -q onnx onnxscript onnxruntime huggingface_hub", shell=True)
import timm; print("timm", timm.__version__, "| torch", torch.__version__)
""")

md("## Code and knowledge base\nThe cells below write the AgroDoc pipeline and disease knowledge base to disk. You don't need to read them.")
for f in FILES:
    body = (ROOT / f).read_text(encoding="utf-8")
    code(f"%%writefile {f}\n{body}", hidden=True)

md("## 1. Check the label mapping")
code("!python tests/test_matcher.py")

md("## 2. Download the datasets\nFailures are listed at the end and skipped; the rest still train.")
code("""
skip = ("--skip " + " ".join(SKIP)) if SKIP else ""
!python pipeline/download.py {skip} || true
!python pipeline/download.py --list
""")

md("## 3. Clean, de-duplicate, shrink and split")
code("""
!python pipeline/prepare.py --workers {WORKERS} --resize {RESIZE}
from IPython.display import Markdown, display
display(Markdown(open("reports/prepare_report.md", encoding="utf-8").read()))
""")

md("## Smoke test (≈5 minutes)\nRuns every later step on a tiny scale, so a problem shows up now instead of after hours of training.")
code("""
S = RUNS / "smoke"
!python pipeline/train.py --model {STUDENT} --img {STUDENT_IMG} --epochs 1 --epoch-size 640 --bs 32 --max-steps 20 --workers {WORKERS} --out {S}
!python pipeline/train.py --model {STUDENT} --img {STUDENT_IMG} --epochs 1 --epoch-size 640 --bs 32 --max-steps 10 --workers {WORKERS} --teacher {S}/best.pt --out {S}/kd
!python pipeline/evaluate.py {S}/best.pt --max-images 300 --workers {WORKERS} --out {S}/eval.json
!python pipeline/export.py {S}/best.pt --quantize --calib 20 --out {S}/model
!python pipeline/evaluate.py {S}/model/model.int8.onnx --max-images 300 --workers {WORKERS} --out {S}/eval_onnx.json
assert (S / "model/model.int8.onnx").exists() and (S / "eval_onnx.json").exists(), "Smoke test failed — scroll up for the error"
print("✓ smoke test passed: training, distillation, evaluation and export all work")
""")

md("## 4. Train the teacher (large, accurate)")
code("""
if USE_TEACHER:
    !python pipeline/train.py --model {TEACHER} --img {TEACHER_IMG} --epochs 40 --epoch-size {EPOCH_SIZE} --bs 64 --lr 4e-4 --workers {WORKERS} --time-limit {TEACHER_MIN} --out {RUNS}/teacher
""")

md("## 5. Train the phone model (distilled from the teacher)")
code("""
teacher_arg = f"--teacher {RUNS}/teacher/best.pt" if USE_TEACHER and (RUNS / "teacher/best.pt").exists() else ""
!python pipeline/train.py --model {STUDENT} --img {STUDENT_IMG} --epochs 60 --epoch-size {EPOCH_SIZE} --bs 96 --lr 1e-3 --workers {WORKERS} --time-limit {STUDENT_MIN} {teacher_arg} --out {RUNS}/student
""")

md("## 6. Test on photos the models never saw")
code("""
if (RUNS / "teacher/best.pt").exists():
    !python pipeline/evaluate.py {RUNS}/teacher/best.pt --workers {WORKERS} --out reports/eval_teacher.json
!python pipeline/evaluate.py {RUNS}/student/best.pt --workers {WORKERS} --out reports/eval_student.json
""")

md("## 7. Export for the app and check the exported model")
code("""
!python pipeline/build_kb.py
!python pipeline/export.py {RUNS}/student/best.pt --quantize
!python pipeline/evaluate.py app/model/model.onnx --workers {WORKERS} --out reports/eval_onnx_fp32.json
!python pipeline/evaluate.py app/model/model.int8.onnx --workers {WORKERS} --out reports/eval_onnx_int8.json
""")

md("## 8. Package the result → Output tab → `agrodoc_model.zip`")
code("""
import json, shutil, zipfile
fp32 = json.load(open("reports/eval_onnx_fp32.json")); int8 = json.load(open("reports/eval_onnx_int8.json"))
drop = int8["overall"]["acc"] < fp32["overall"]["acc"] - 0.01
labels = json.load(open("app/model/labels.json"))
if drop:                                    # int8 lost too much accuracy → ship the full-precision model
    labels["files"].pop("int8", None); (CODE / "app/model/model.int8.onnx").unlink(missing_ok=True)
    json.dump(labels, open("app/model/labels.json", "w"), indent=1)
print(f"fp32 acc {fp32['overall']['acc']:.3f} | int8 acc {int8['overall']['acc']:.3f} → shipping {'fp32' if drop else 'int8'}")
shutil.copy(RUNS / "student/best.pt", CODE / "student_best.pt")
with zipfile.ZipFile("/kaggle/working/agrodoc_model.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in list((CODE / "app/model").glob("*")) + list((CODE / "reports").glob("*")):
        z.write(p, p.relative_to(CODE))
    for name in ["log.json"]:
        for run in ["teacher", "student"]:
            if (RUNS / run / name).exists():
                z.write(RUNS / run / name, f"runs/{run}/{name}")
print("✓ /kaggle/working/agrodoc_model.zip", round(pathlib.Path("/kaggle/working/agrodoc_model.zip").stat().st_size / 1e6, 1), "MB")
print(json.dumps({k: fp32[k] for k in ["overall", "field_only", "lab_only", "crop_correct", "not_sure_rate"]}, indent=1))
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                   "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
for c in nb["cells"]:                       # nbformat stores source as a list of lines
    c["source"] = c["source"].splitlines(keepends=True)
out = ROOT / "kaggle" / "agrodoc_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{len(cells)} cells → {out} ({out.stat().st_size / 1024:.0f} KB)")
