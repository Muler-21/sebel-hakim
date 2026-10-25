"""Compile kb/*.yaml into app/kb.json (what the phone app reads).

  python pipeline/build_kb.py
"""
import json

from common import KB_DIR, ROOT, load_diseases, load_yaml

TEXT_FIELDS = ["symptoms", "spread", "cultural", "biological", "chemical", "prevention", "notes"]


def main():
    crops = load_yaml(KB_DIR / "crops.yaml")
    diseases = []
    for d in load_diseases():
        e = {k: d[k] for k in ("id", "crop", "type") if k in d}
        e["severity"] = d.get("severity", "none" if d["type"] == "healthy" else "medium")
        e["name"] = d.get("name", {})
        if d.get("scientific"):
            e["scientific"] = d["scientific"]
        for k in TEXT_FIELDS:
            if d.get(k):
                e[k] = d[k]
        diseases.append(e)
    out = ROOT / "app" / "kb.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"crops": crops["crops"], "groups": crops["groups"], "diseases": diseases},
                              ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(diseases)} diseases, {len(crops['crops'])} crops → {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
