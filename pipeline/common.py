"""Shared helpers: paths, knowledge-base loading, label matching."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # repo root
KB_DIR = ROOT / "kb"
PIPE_DIR = ROOT / "pipeline"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# where downloaded/prepared data lives (big); override with AGRODOC_DATA, e.g. on Kaggle
DATA_ROOT = Path(os.environ.get("AGRODOC_DATA", ROOT / "data"))


def walk_files(base: Path):
    """All files under base, following symlinks (Kaggle inputs are linked in)."""
    for d, dirs, files in os.walk(base, followlinks=True):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in files:
            yield Path(d) / f


def norm(s: str) -> str:
    """Lowercase, turn every run of non-alphanumerics into one space."""
    s = str(s).lower().replace("(", " ").replace(")", " ")
    return re.sub(r"[^0-9a-zሀ-፿]+", " ", s).strip()


def load_yaml(p: Path):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_crops() -> dict:
    data = load_yaml(KB_DIR / "crops.yaml")
    return {c["id"]: c for c in data["crops"]}


def load_diseases() -> list[dict]:
    out = []
    for p in sorted(KB_DIR.glob("*.yaml")):
        if p.name == "crops.yaml":
            continue
        out.extend(load_yaml(p).get("diseases", []))
    ids = [d["id"] for d in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate disease ids in kb/: {sorted(dupes)}")
    crops = load_crops()
    for d in out:
        if d["crop"] not in crops:
            raise ValueError(f"{d['id']}: unknown crop '{d['crop']}' (add it to kb/crops.yaml)")
        if not d["id"].startswith(d["crop"] + "."):
            raise ValueError(f"{d['id']}: id must start with '{d['crop']}.'")
    return out


def load_datasets() -> list[dict]:
    return load_yaml(PIPE_DIR / "datasets.yaml")["datasets"]


# Words that name a crop inside dataset folder names (e.g. "Corn_(maize)___Common_rust_").
CROP_WORDS = {
    "maize": ["maize", "corn"], "wheat": ["wheat"], "rice": ["rice", "paddy"],
    "sorghum": ["sorghum"], "bean": ["bean", "beans"], "soybean": ["soybean", "soyabean", "soya"],
    "groundnut": ["groundnut", "peanut"], "cassava": ["cassava"], "potato": ["potato"],
    "tomato": ["tomato"], "pepper": ["pepper", "bell pepper", "capsicum"], "squash": ["squash", "pumpkin"],
    "banana": ["banana"], "mango": ["mango"], "citrus": ["citrus", "orange", "lemon"],
    "apple": ["apple"], "grape": ["grape"], "peach": ["peach"], "cherry": ["cherry"],
    "strawberry": ["strawberry"], "blueberry": ["blueberry"], "raspberry": ["raspberry"],
    "coffee": ["coffee"],
}


def detect_crop(label: str) -> str | None:
    """Return the crop named in a folder label, preferring the longest match."""
    n = " " + norm(label) + " "
    best, best_len = None, 0
    for crop, words in CROP_WORDS.items():
        for w in words:
            if f" {w} " in n and len(w) > best_len:
                # "soybean" contains "bean": longest wins; "bell pepper" beats "pepper"
                best, best_len = crop, len(w)
    return best


@dataclass
class Matcher:
    """Maps raw dataset labels onto knowledge-base disease ids."""
    diseases: list[dict]
    by_syn: dict = field(default_factory=dict)       # norm(synonym) -> set(ids)

    def __post_init__(self):
        self.ids = {d["id"] for d in self.diseases}
        for d in self.diseases:
            names = [d["id"], d["id"].replace(".", " "), d.get("name", {}).get("en", "")]
            for s in names + list(d.get("synonyms", []) or []):
                if s:
                    self.by_syn.setdefault(norm(s), set()).add(d["id"])

    def match(self, label: str, crop: str | None = None, overrides: dict | None = None) -> str | None:
        """Return a disease id or None.

        Order: explicit override → exact synonym (filtered by crop) → synonym after
        removing the crop words → longest synonym contained in the label.
        """
        if overrides:
            for k, v in overrides.items():
                if norm(k) == norm(label):
                    return v or None          # empty override = deliberately skip
        crop = crop or detect_crop(label)
        n = norm(label)

        def pick(ids: set[str]) -> str | None:
            if crop:
                ids = {i for i in ids if i.startswith(crop + ".")}
            return next(iter(ids)) if len(ids) == 1 else None

        if n.isdigit() and not crop:
            return None                    # numeric class ids only mean something inside one dataset
        if n in self.by_syn and (r := pick(self.by_syn[n])):
            return r
        if crop:
            stripped = n
            for w in sorted(CROP_WORDS.get(crop, []), key=len, reverse=True):
                stripped = re.sub(rf"\b{re.escape(w)}\b", " ", stripped)
            stripped = re.sub(r"\b(leaf|leaves|plant|disease|images?)\b", " ", stripped)
            stripped = re.sub(r"\s+", " ", stripped).strip()
            if stripped in self.by_syn and (r := pick(self.by_syn[stripped])):
                return r
            if stripped in ("", "healthy", "normal", "healthy healthy", "fresh", "good"):
                # "Strawberry leaf", "Healthy", "normal" → that crop's healthy class
                return f"{crop}.healthy" if f"{crop}.healthy" in self.ids else None
        # longest synonym fully contained in the label (only when the crop is known)
        if crop:
            best, best_len = None, 0
            for syn, ids in self.by_syn.items():
                if len(syn) >= 4 and f" {syn} " in f" {n} ":
                    r = pick(ids)
                    if r and len(syn) > best_len:
                        best, best_len = r, len(syn)
            return best
        return None
