"""Build one clean, labelled manifest from all downloaded datasets.

Steps
  1. Read every dataset in data/raw/ (class folders, CSV labels or Hugging Face parquet).
  2. Map each raw label to a knowledge-base disease id (kb/*.yaml synonyms,
     plus per-dataset `map:` overrides in datasets.yaml).
  3. Drop unreadable images and near-duplicates (perceptual hash), so the same
     photo can never sit in both train and test.
  4. Split train / val / test per dataset and class.
  5. Write data/manifest.csv, data/classes.json and reports/prepare_report.md.

Usage
  python pipeline/prepare.py [--commercial-only] [--min-per-class 20] [--workers 16]
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import DATA_ROOT, IMG_EXT, ROOT, Matcher, load_datasets, load_diseases, norm, walk_files

RAW = DATA_ROOT / "raw"
DATA = DATA_ROOT
REPORTS = ROOT / "reports"
GENERIC_DIRS = {"train", "test", "val", "valid", "validation", "images", "image", "img", "data",
                "dataset", "color", "raw", "train_images", "test_images", "original", "augmented"}


# ---------------------------------------------------------------- collecting
def export_parquet(folder: Path) -> None:
    """Turn Hugging Face parquet files into <folder>/_hf/<label>/<n>.jpg once."""
    out = folder / "_hf"
    files = [str(p) for p in walk_files(folder) if p.suffix == ".parquet"]
    if out.exists() or not files:
        return
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=files, split="train")
    img_col = next(c for c, f in ds.features.items() if f.__class__.__name__ == "Image")
    lab_col = next((c for c, f in ds.features.items() if f.__class__.__name__ == "ClassLabel"), None)
    if lab_col is None:
        lab_col = next(c for c in ds.column_names if c.lower() in {"label", "labels", "class", "disease"})
        names = None
    else:
        names = ds.features[lab_col].names
    for i, row in enumerate(ds):
        lab = names[row[lab_col]] if names else str(row[lab_col])
        p = out / lab / f"{i:07d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        row[img_col].convert("RGB").save(p, quality=92)


def folder_label(path: Path, base: Path, depth: int | None) -> str:
    parts = path.relative_to(base).parts[:-1]
    if depth is not None:
        return parts[depth] if len(parts) > depth else ""
    for p in reversed(parts):          # nearest non-generic parent folder
        if norm(p) not in {norm(g) for g in GENERIC_DIRS}:
            return p
    return ""


def collect(d: dict) -> list[tuple[Path, str]]:
    base = RAW / d["id"]
    if d["host"] == "hf":
        export_parquet(base)
    items: list[tuple[Path, str]] = []
    if d.get("label") == "csv":
        spec = d["csv"]
        csv_path = next((p for p in walk_files(base) if p.name == spec["file"]), None)
        img_dir = next((p.parent for p in walk_files(base)
                        if p.parent.name == spec["image_dir"] and p.suffix.lower() in IMG_EXT), None)
        if not csv_path or not img_dir:
            print(f"  ! {d['id']}: {spec['file']} or {spec['image_dir']} not found")
            return []
        sep = spec.get("multi_label_sep")
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lab = str(row[spec["label_col"]])
                if sep and len(lab.split(sep)) > 1:
                    continue                       # skip multi-disease images
                items.append((img_dir / row[spec["image_col"]], lab))
        return items
    scan = base
    if d.get("label_dir"):
        scan = next((p.parent.parent for p in walk_files(base)
                     if p.parent.parent.name == d["label_dir"] and p.suffix.lower() in IMG_EXT), base)
    for p in walk_files(scan):
        if p.suffix.lower() in IMG_EXT:
            items.append((p, folder_label(p, scan, d.get("label_depth"))))
    return items


# ---------------------------------------------------------------- hashing
def hash_one(path: str, cache: str = "", size: int = 0):
    """Difference hash (64 bit) of a 9x8 grayscale thumbnail: identical for re-saved / resized copies.

    With `cache`, also writes a copy whose short side is `size` px, so training decodes small JPEGs.
    Returns (original path, hash or None, path to use for training).
    """
    try:
        import numpy as np
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            if cache:
                im.draft("RGB", (size * 2, size * 2))
                im = ImageOps.exif_transpose(im).convert("RGB")
                if min(im.size) < 32:
                    return path, None, path
                if min(im.size) > size:
                    s = size / min(im.size)
                    im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.BILINEAR)
                dst = Path(cache) / (hashlib.sha1(path.encode()).hexdigest()[:20] + ".jpg")
                im.save(dst, quality=90)
                use = str(dst)
            else:
                im.draft("L", (64, 64))             # fast JPEG decode at reduced size
                use = path
            g = im.convert("L")
            if min(g.size) < 32:
                return path, None, path
            a = np.asarray(g.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        return path, f"{int(''.join('1' if b else '0' for b in bits), 2):016x}", use
    except Exception:
        return path, None, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commercial-only", action="store_true", help="drop datasets with non-commercial licences")
    ap.add_argument("--min-per-class", type=int, default=20)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--resize", type=int, default=0,
                    help="write copies with this short side (e.g. 320) for much faster training")
    a = ap.parse_args()

    diseases = load_diseases()
    m = Matcher(diseases)
    rows, unmapped = [], defaultdict(Counter)

    for d in load_datasets():
        if not (RAW / d["id"] / ".done").exists():
            continue
        if a.commercial_only and d.get("commercial") is not True:
            print(f"[drop] {d['id']} (licence: {d.get('licence')})")
            continue
        items = collect(d)
        mapped = 0
        cache: dict[str, str | None] = {}
        for path, raw in items:
            if raw not in cache:
                cache[raw] = m.match(raw, crop=d.get("crop"), overrides=d.get("map"))
            lab = cache[raw]
            if lab is None:
                unmapped[d["id"]][raw] += 1
                continue
            rows.append({"path": str(path), "label": lab, "dataset": d["id"],
                         "field": d.get("field", "lab"), "weight": d.get("weight", 1.0), "raw": raw})
            mapped += 1
        print(f"[read] {d['id']:24} {len(items):7} images, {mapped:7} mapped, "
              f"{len(cache)} raw classes → {len({v for v in cache.values() if v})} kb classes")

    if not rows:
        raise SystemExit("No images found. Run pipeline/download.py first.")

    # ---- readability + duplicates
    print(f"[hash] {len(rows)} images with {a.workers} workers …")
    cache = ""
    if a.resize:
        cache = str(DATA / f"img{a.resize}")
        Path(cache).mkdir(parents=True, exist_ok=True)
        print(f"       also writing {a.resize}px copies to {cache}")
    fn = functools.partial(hash_one, cache=cache, size=a.resize)
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (pth, h, use) in enumerate(ex.map(fn, [r["path"] for r in rows], chunksize=64)):
            res[pth] = (h, use)
            if i % 10000 == 0 and i:
                print(f"       {i:,}/{len(rows):,}", flush=True)
    bad = sum(1 for r in rows if res[r["path"]][0] is None)
    rows = [r | {"hash": res[r["path"]][0], "path": res[r["path"]][1], "source": r["path"]}
            for r in rows if res[r["path"]][0]]
    dup_removed, conflict_removed = 0, 0
    if not a.no_dedup:
        by_hash = defaultdict(list)
        for r in rows:
            by_hash[r["hash"]].append(r)
        keep = []
        for h, group in by_hash.items():
            labels = {r["label"] for r in group}
            if len(labels) > 1:                 # same picture, different labels → unreliable
                conflict_removed += len(group)
                continue
            # prefer a field image over a lab copy
            group.sort(key=lambda r: (r["field"] == "lab", r["dataset"]))
            keep.append(group[0])
            dup_removed += len(group) - 1
        rows = keep

    # ---- rare classes
    counts = Counter(r["label"] for r in rows)
    rare = {k for k, v in counts.items() if v < a.min_per_class}
    rows = [r for r in rows if r["label"] not in rare]

    # ---- split per (dataset, label)
    rng = random.Random(a.seed)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["label"])].append(r)
    for g in groups.values():
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, round(n * a.test)) if n >= 10 else 0
        n_val = max(1, round(n * a.val)) if n >= 10 else 0
        for i, r in enumerate(g):
            r["split"] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    classes = sorted({r["label"] for r in rows})
    DATA.mkdir(exist_ok=True)
    with open(DATA / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "dataset", "field", "weight", "split", "raw", "hash", "source"])
        w.writeheader()
        w.writerows(rows)
    dmap = {d["id"]: d for d in diseases}
    (DATA / "classes.json").write_text(json.dumps(
        [{"id": c, "crop": dmap[c]["crop"], "type": dmap[c]["type"]} for c in classes], indent=1))

    # ---- report
    REPORTS.mkdir(exist_ok=True)
    final = Counter(r["label"] for r in rows)
    split_c = Counter(r["split"] for r in rows)
    by_ds = Counter(r["dataset"] for r in rows)
    kb_ids = {d["id"] for d in diseases}
    lines = [
        "# Data preparation report", "",
        f"- images kept: **{len(rows):,}** (train {split_c['train']:,} · val {split_c['val']:,} · test {split_c['test']:,})",
        f"- classes: **{len(classes)}** of {len(kb_ids)} in the knowledge base",
        f"- unreadable or tiny images dropped: {bad:,}",
        f"- duplicates removed: {dup_removed:,}; images with conflicting labels removed: {conflict_removed:,}",
        f"- classes dropped for having < {a.min_per_class} images: {', '.join(sorted(rare)) or 'none'}", "",
        "## Images per dataset", "", "| dataset | images |", "|---|---|",
        *[f"| {k} | {v:,} |" for k, v in by_ds.most_common()], "",
        "## Images per class", "", "| class | images |", "|---|---|",
        *[f"| {k} | {v:,} |" for k, v in sorted(final.items())], "",
        "## Knowledge-base diseases with no training images (app can show advice, model cannot predict them)", "",
        ", ".join(sorted(kb_ids - set(classes))) or "none", "",
        "## Unmapped raw labels", "",
        "Add a synonym in kb/*.yaml, or a `map: {\"raw label\": crop.condition}` entry for the dataset in "
        "pipeline/datasets.yaml (map to an empty string to skip on purpose).", "",
        "| dataset | raw label | images |", "|---|---|---|",
        *[f"| {ds} | {lab} | {n:,} |" for ds, c in unmapped.items() for lab, n in c.most_common()],
    ]
    (REPORTS / "prepare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {len(rows):,} images, {len(classes)} classes → {DATA / 'manifest.csv'}; see reports/prepare_report.md")
    if unmapped:
        print(f"       {sum(sum(c.values()) for c in unmapped.values()):,} images had unmapped labels (listed in the report)")


if __name__ == "__main__":
    main()
