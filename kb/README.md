# Disease knowledge base

One YAML file per crop group. Every entry is both:

1. **A label the AI model can predict.** Its `id` (`crop.condition`) is the class name, and its
   `synonyms` are used by `pipeline/prepare.py` to map each dataset's folder names onto it.
2. **The advice the app shows.**

## Entry fields

| field | meaning |
|---|---|
| `id` | `crop.condition`, lowercase, unique |
| `crop` | crop id (see `crops.yaml`) |
| `name` | common name, per language (`en`, `am`, …) |
| `scientific` | pathogen / pest scientific name |
| `type` | fungal · bacterial · viral · oomycete · pest · nematode · abiotic · healthy |
| `severity` | low · medium · high · critical |
| `synonyms` | folder/label spellings seen in datasets (matched case- and punctuation-insensitively) |
| `symptoms` | what to look for |
| `spread` | how it spreads / favourable conditions |
| `cultural` | farm practices (always first choice) |
| `biological` | biocontrol options |
| `chemical` | last resort: active ingredients or groups, never brand names |
| `prevention` | for next season |
| `notes` | Africa-specific or important extra information |

## Rules for contributors

- Chemical advice names **active ingredients or groups only**. The app always adds: use products
  registered in your country, follow the label, wear protection, respect pre-harvest intervals.
- Viral diseases: say plainly there is **no chemical cure**; manage the vectors and remove sources.
- Notifiable or quarantine diseases (e.g. Ug99 stem rust, MLN, BXW, banana Fusarium TR4):
  `severity: critical` and tell the farmer to **report to the extension service**.
- Add a translation by adding a language key under `name` (and, later, under text fields).
