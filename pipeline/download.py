"""Download the public datasets listed in datasets.yaml into data/raw/<id>/.

Usage:
    python pipeline/download.py                 # everything that can be fetched automatically
    python pipeline/download.py --only plantvillage cassava_kaggle
    python pipeline/download.py --list          # show datasets and their status

Already-downloaded datasets are skipped (a `.done` marker is written).
Archives (.zip/.tar/.tar.gz/.7z) are extracted in place.
Datasets with `host: manual` print instructions instead.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from common import ROOT, load_datasets

RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "agrodoc-dataset-downloader/1.0"}


def http_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def http_download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {dest.name}: {done / total:6.1%} of {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def extract_all(folder: Path):
    """Extract every archive under folder (repeat once for archives inside archives)."""
    for _ in range(2):
        for a in list(folder.rglob("*")):
            name = a.name.lower()
            try:
                if name.endswith(".zip"):
                    with zipfile.ZipFile(a) as z:
                        z.extractall(a.parent / a.stem)
                elif name.endswith((".tar", ".tar.gz", ".tgz")):
                    with tarfile.open(a) as t:
                        t.extractall(a.parent / a.name.split(".tar")[0], filter="data")
                elif name.endswith(".7z"):
                    subprocess.run(["7z", "x", "-y", f"-o{a.parent / a.stem}", str(a)], check=True)
                else:
                    continue
                a.unlink()
                print(f"    extracted {a.name}")
            except Exception as e:  # keep going; prepare.py reports what is usable
                print(f"    ! could not extract {a.name}: {e}")


# ---------------------------------------------------------------- hosts
def get_github(d, out: Path):
    url = f"https://github.com/{d['ref']}.git"
    if d.get("subdir"):
        # sparse, blob-less clone: only the folder we need
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(out)], check=True)
        subprocess.run(["git", "-C", str(out), "sparse-checkout", "set", d["subdir"]], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", url, str(out)], check=True)
    shutil.rmtree(out / ".git", ignore_errors=True)


def get_zenodo(d, out: Path):
    rec = http_json(f"https://zenodo.org/api/records/{d['ref']}")
    for f in rec.get("files", []):
        key = f.get("key") or f.get("filename")
        link = f["links"].get("self") or f["links"].get("download")
        http_download(link, out / key)


def get_mendeley(d, out: Path):
    ver = d.get("version", 1)
    base = f"https://data.mendeley.com/public-api/datasets/{d['ref']}"
    files = http_json(f"{base}/files?folder_id=root&version={ver}")
    stack = list(files)
    while stack:
        f = stack.pop()
        if f.get("content_details") is None and f.get("id") and f.get("type") == "folder":
            stack.extend(http_json(f"{base}/files?folder_id={f['id']}&version={ver}"))
            continue
        cd = f.get("content_details") or {}
        link = cd.get("download_url") or f.get("download_url")
        if link:
            http_download(link, out / f["filename"])
    if not any(out.iterdir()):
        raise RuntimeError("Mendeley API returned no files; download manually from "
                           f"https://data.mendeley.com/datasets/{d['ref']}/{ver}")


def get_dataverse(d, out: Path):
    pid = urllib.parse.quote(d["ref"], safe="")
    http_download(f"https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId={pid}",
                  out / "dataset.zip")


def get_kaggle(d, out: Path, comp: bool):
    if not shutil.which("kaggle"):
        raise RuntimeError("install the Kaggle CLI (pip install kaggle) and put your token in ~/.kaggle/kaggle.json")
    cmd = (["kaggle", "competitions", "download", "-c", d["ref"]] if comp
           else ["kaggle", "datasets", "download", "-d", d["ref"]])
    subprocess.run(cmd + ["-p", str(out)], check=True)


def get_hf(d, out: Path):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError("pip install huggingface_hub") from e
    snapshot_download(repo_id=d["ref"], repo_type="dataset", local_dir=str(out))
    # parquet-based HF datasets are turned into class folders by prepare.py


HOSTS = {
    "github": get_github, "zenodo": get_zenodo, "mendeley": get_mendeley,
    "dataverse": get_dataverse, "hf": get_hf,
    "kaggle": lambda d, o: get_kaggle(d, o, comp=False),
    "kaggle_comp": lambda d, o: get_kaggle(d, o, comp=True),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="dataset ids to download")
    ap.add_argument("--skip", nargs="*", default=[], help="dataset ids to skip")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    ds = load_datasets()
    if a.list:
        for d in ds:
            status = "done" if (RAW / d["id"] / ".done").exists() else "-"
            print(f"{d['id']:24} {d['host']:12} {d.get('field',''):6} {str(d.get('commercial')):8} {status}")
        return

    failed, manual = [], []
    for d in ds:
        if (a.only and d["id"] not in a.only) or d["id"] in a.skip:
            continue
        out = RAW / d["id"]
        if (out / ".done").exists():
            print(f"[skip] {d['id']} already downloaded")
            continue
        if d["host"] == "manual":
            manual.append(d)
            continue
        print(f"[get ] {d['id']} ({d['host']}: {d.get('ref')})")
        if out.exists() and d["host"] == "github":
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        try:
            HOSTS[d["host"]](d, out)
            extract_all(out)
            (out / ".done").write_text("ok\n")
        except Exception as e:
            print(f"  ! {d['id']} failed: {e}")
            failed.append(d["id"])

    if manual:
        print("\nManual downloads (put the files in data/raw/<id>/ and create an empty .done file):")
        for d in manual:
            if not (RAW / d["id"] / ".done").exists():
                print(f"  - {d['id']}: {d.get('url')}\n      {d.get('note', '')}")
    if failed:
        print(f"\nFailed: {', '.join(failed)}  (re-run later, or download manually)")
        sys.exit(1)


if __name__ == "__main__":
    main()
