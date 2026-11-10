# Licence for content and the trained model

Copyright (c) 2026 Mulusew Kassa Bitew

Sebel Hakim is made of three kinds of material, each with its own licence:

| What | Where | Licence |
|---|---|---|
| Source code | `app/*.js`, `app/*.html`, `app/*.css`, `pipeline/`, `kaggle/`, `hpc/`, `tests/` | [PolyForm Noncommercial 1.0.0](LICENSE) |
| Disease knowledge base, translations, documentation and images | `kb/`, `app/kb.json`, `app/i18n.js` (texts), `docs/`, `README.md`, `app/icon.svg` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| Trained AI model | `app/model/` (`model.onnx`, `labels.json`) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) to the extent the author holds rights in it, and subject to the terms of its training data (below) |

## Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

The knowledge base, documentation, images and trained model are licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License. To view a copy of this licence, visit
<https://creativecommons.org/licenses/by-nc/4.0/>.

In short (the full legal code linked above is what applies):

- **You may** share and adapt this material for any non-commercial purpose: research, teaching, extension work,
  NGO and government programmes, personal use.
- **You must give credit**: cite Sebel Hakim as shown below, link to the licence, and say if you changed anything.
- **You may not** use it for commercial purposes without written permission from the author.

> Bitew, M. K. (2026). *Sebel Hakim: free, offline AI diagnosis of crop diseases for African farmers* (Version 2.1.0)
> [Computer software]. https://github.com/Muler-21/sebel-hakim

## Training data

The model was trained on public image datasets made by other people. They are **not** redistributed in this
repository and keep their own licences, listed with full citations in the [README](README.md#data-sources).

- **Non-commercial only:** PlantWild (CC BY-NC-ND 4.0), PlantSeg (CC BY-NC 4.0), MangoLeafBD (CC BY-NC 3.0) and
  Cassava Leaf Disease Classification (Kaggle competition rules).
- **Share-alike:** PlantVillage (CC BY-SA 3.0). Its authors state that algorithms trained on it fall under the same
  licence.
- **Attribution only or public domain:** all other datasets (CC BY 4.0, CC0 1.0, MIT).

Whether trained model weights legally count as adaptations of their training images is not settled, and the
non-commercial and share-alike terms above cannot both be fully satisfied by one licence. The model is therefore
offered for **non-commercial use only, with credit to the author and to every dataset listed in the README**.
Anyone planning commercial use should take legal advice. A model for commercial use would have to be retrained on
commercially licensed data only (`python pipeline/prepare.py --commercial-only --no-sharealike`) and licensed
separately.

## Commercial use

For commercial licensing of the code, the knowledge base or the model, contact the author through
<https://github.com/Muler-21>.
