# AgroDoc — free, offline plant disease doctor

Photograph a leaf → the app names the most likely disease, how sure it is, and what to do:
**farm practices first, then natural/biological options, chemicals only as a last resort**.
The AI runs on the phone, so after the first visit it works without internet.

- **For farmers:** English, Amharic, French and Swahili interface; big buttons; works on low-cost Android phones.
- **For students and researchers:** a browsable library of 80 diseases and pests across 23 crops, with pathogen names.
- **Africa first:** cassava mosaic and brown streak, maize lethal necrosis and streak virus, fall armyworm, wheat stem and yellow rust, bean angular leaf spot, banana Xanthomonas wilt and Fusarium wilt, coffee leaf rust, groundnut rosette and more.

```
kb/            disease knowledge base (YAML) — the single source of truth for labels and advice
pipeline/      dataset download → cleaning → training → evaluation → export
hpc/           SLURM job + conda environment
app/           the phone app (static web app / PWA)
tests/         label-mapping tests and a placeholder model for testing the app
```

## 1. Train the model (on your HPC)

```bash
# on a login node with internet
conda env create -f hpc/environment.yml && conda activate agrodoc
python pipeline/download.py --list          # what will be downloaded
python pipeline/download.py                 # all automatic sources
#   Kaggle sources need ~/.kaggle/kaggle.json and accepting each competition's rules on kaggle.com
#   "manual" sources print instructions (Zindi, Roboflow, NM-AIST)
python pipeline/prepare.py --workers 16     # → data/manifest.csv + reports/prepare_report.md
python tests/test_matcher.py                # label mapping sanity check

# pre-download ImageNet weights (compute nodes often have no internet)
python -c "import timm; [timm.create_model(m, pretrained=True) for m in ['convnextv2_tiny.fcmae_ft_in22k_in1k','efficientnet_b0.ra_in1k']]"

sbatch hpc/run_all.sh                       # teacher → distilled student → evaluation → export
```

Read `reports/prepare_report.md` after preparing: it lists **unmapped dataset labels**. Fix them by adding a
synonym in `kb/*.yaml`, or a per-dataset `map:` entry in `pipeline/datasets.yaml`, then run `prepare.py` again.

### What makes it accurate

| step | why |
|---|---|
| many datasets, field photos weighted ×2, PlantVillage ×0.5 | models trained only on lab photos fail in real fields |
| duplicate removal across datasets | the same photo can't be in train and test, so scores are honest |
| heavy phone-like augmentation (crop, blur, colour, rotation, erasing) | survives bad light and shaky hands |
| big teacher (ConvNeXt-V2) → small student (EfficientNet-B0) distillation | near-teacher accuracy at phone speed |
| temperature calibration | "87% confidence" really means about 87%; low confidence shows "not sure" |
| crop picker in the app | restricting to one crop removes cross-crop mistakes |
| reports split by field/lab, dataset, crop and class | you see exactly where it is weak |

### Licences

Each dataset's licence is recorded in `pipeline/datasets.yaml`. Several are **non-commercial** (CC BY-NC, Kaggle and
Zindi competition rules). A free app is fine; if you later sell a product, run
`python pipeline/prepare.py --commercial-only` and retrain. Check every "(check)" licence on its source page.

## 2. Put the model in the app

`hpc/run_all.sh` writes `app/model/model.onnx`, `model.int8.onnx` and `labels.json`. Compare
`reports/eval_onnx_fp32.json` and `reports/eval_onnx_int8.json`: if int8 loses more than ~1 point of accuracy, delete
`model.int8.onnx` and remove `"int8"` from `labels.json` so the app uses fp32.

After changing anything in `kb/`, run `python pipeline/build_kb.py` to rebuild `app/kb.json`.

## 3. Publish

The repository root has an `index.html` that redirects to `app/`, so GitHub Pages works with the default
settings: Settings → Pages → Deploy from a branch → `main` / `(root)`. No server, no accounts, no running costs.
The app is then at `https://<username>.github.io/<repo>/`.

## Adding a crop, disease or language

- **Disease:** add an entry to the right `kb/*.yaml` (see `kb/README.md`). Include the folder names used by datasets in `synonyms`.
- **Dataset:** add it to `pipeline/datasets.yaml`, then `download.py --only <id>` and `prepare.py`.
- **Language:** add a block to `app/i18n.js`; add `xx:` names to `kb/crops.yaml` and disease `name:` fields.

## Limits

This tool supports decisions; it does not replace an expert. Diseases with few public images (for example MLN,
banana wilts, groundnut rosette) will be recognised less reliably until more field photos are collected. The best
improvement after launch is collecting labelled photos from Ethiopian and other African farms.
