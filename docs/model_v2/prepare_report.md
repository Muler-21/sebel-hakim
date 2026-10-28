# Data preparation report

- images kept: **136,735** (train 109,383 · val 13,676 · test 13,676)
- classes: **108** of 130 in the knowledge base
- unreadable or tiny images dropped: 0
- duplicates removed: 7,227; images with conflicting labels removed: 6,784
- classes dropped for having < 20 images: none

## Images per dataset

| dataset | images |
|---|---|
| plantvillage | 54,275 |
| tanzania_maize | 17,365 |
| plantwild | 17,013 |
| makerere_beans | 15,402 |
| cassava_kaggle | 14,239 |
| mangoleafbd | 3,758 |
| plantseg | 3,726 |
| sorghum_disease | 3,278 |
| potato_uncontrolled | 3,015 |
| plantdoc | 2,400 |
| ibean | 1,457 |
| groundnut_leaf | 807 |

## Images per class

| class | images |
|---|---|
| apple.black_rot | 764 |
| apple.cedar_rust | 620 |
| apple.healthy | 2,165 |
| apple.mosaic_virus | 148 |
| apple.scab | 928 |
| banana.anthracnose | 50 |
| banana.black_sigatoka | 104 |
| banana.bunchy_top | 104 |
| banana.cigar_end_rot | 45 |
| banana.cordana_leaf_spot | 40 |
| banana.fusarium_wilt | 187 |
| banana.healthy | 242 |
| bean.angular_leaf_spot | 5,581 |
| bean.halo_blight | 93 |
| bean.healthy | 6,018 |
| bean.mosaic_virus | 100 |
| bean.rust | 5,636 |
| blueberry.healthy | 1,891 |
| blueberry.rust | 93 |
| cassava.bacterial_blight | 1,087 |
| cassava.brown_streak | 2,189 |
| cassava.green_mottle | 2,386 |
| cassava.healthy | 2,577 |
| cassava.mosaic | 6,000 |
| cherry.healthy | 1,189 |
| cherry.leaf_spot | 193 |
| cherry.powdery_mildew | 1,141 |
| citrus.canker | 396 |
| citrus.greening | 5,697 |
| coffee.cercospora | 71 |
| coffee.healthy | 137 |
| coffee.leaf_rust | 141 |
| grape.black_rot | 1,420 |
| grape.downy_mildew | 306 |
| grape.esca | 1,383 |
| grape.healthy | 689 |
| grape.leaf_blight | 1,076 |
| groundnut.healthy | 593 |
| groundnut.rosette | 95 |
| groundnut.rust | 119 |
| maize.common_rust | 1,474 |
| maize.common_smut | 243 |
| maize.gray_leaf_spot | 735 |
| maize.healthy | 7,192 |
| maize.lethal_necrosis | 5,778 |
| maize.northern_leaf_blight | 1,318 |
| maize.streak_virus | 5,709 |
| mango.anthracnose | 485 |
| mango.bacterial_canker | 500 |
| mango.cutting_weevil | 283 |
| mango.die_back | 493 |
| mango.gall_midge | 497 |
| mango.healthy | 500 |
| mango.powdery_mildew | 500 |
| mango.sooty_mould | 500 |
| other.unknown_plant | 8,206 |
| peach.bacterial_spot | 2,297 |
| peach.brown_rot | 120 |
| peach.healthy | 627 |
| peach.leaf_curl | 187 |
| peach.scab | 52 |
| pepper.bacterial_spot | 1,207 |
| pepper.blossom_end_rot | 61 |
| pepper.healthy | 1,768 |
| potato.bacterial_disease | 569 |
| potato.early_blight | 1,251 |
| potato.fungal_disease | 740 |
| potato.healthy | 598 |
| potato.late_blight | 1,583 |
| potato.nematode | 67 |
| potato.pest_damage | 601 |
| potato.virus | 530 |
| raspberry.healthy | 663 |
| rice.blast | 129 |
| rice.healthy | 250 |
| rice.sheath_blight | 215 |
| sorghum.anthracnose | 983 |
| sorghum.rust | 2,295 |
| soybean.bacterial_blight | 53 |
| soybean.brown_spot | 35 |
| soybean.downy_mildew | 95 |
| soybean.frogeye_leaf_spot | 146 |
| soybean.healthy | 5,378 |
| soybean.mosaic_virus | 76 |
| soybean.rust | 90 |
| squash.healthy | 405 |
| squash.powdery_mildew | 2,146 |
| strawberry.anthracnose | 82 |
| strawberry.healthy | 741 |
| strawberry.leaf_scorch | 1,174 |
| tomato.bacterial_spot | 2,425 |
| tomato.early_blight | 1,303 |
| tomato.healthy | 1,859 |
| tomato.late_blight | 2,224 |
| tomato.leaf_mold | 1,211 |
| tomato.mosaic_virus | 583 |
| tomato.septoria_leaf_spot | 2,043 |
| tomato.spider_mites | 1,678 |
| tomato.target_spot | 1,404 |
| tomato.yellow_leaf_curl_virus | 5,550 |
| wheat.black_chaff | 76 |
| wheat.head_scab | 184 |
| wheat.leaf_rust | 93 |
| wheat.loose_smut | 130 |
| wheat.powdery_mildew | 185 |
| wheat.septoria | 135 |
| wheat.stem_rust | 92 |
| wheat.yellow_rust | 239 |

## Knowledge-base diseases with no training images (app can show advice, model cannot predict them)

apple.powdery_mildew, banana.xanthomonas_wilt, citrus.black_spot, citrus.healthy, citrus.melanose, coffee.leaf_miner, coffee.phoma, coffee.red_spider_mite, groundnut.early_leaf_spot, groundnut.late_leaf_spot, maize.fall_armyworm, rice.bacterial_leaf_blight, rice.bacterial_leaf_streak, rice.bacterial_panicle_blight, rice.brown_spot, rice.dead_heart, rice.downy_mildew, rice.hispa, rice.tungro, sorghum.healthy, sorghum.leaf_blight, wheat.healthy

## Unmapped raw labels

Add a synonym in kb/*.yaml, or a `map: {"raw label": crop.condition}` entry for the dataset in pipeline/datasets.yaml (map to an empty string to skip on purpose).

| dataset | raw label | images |
|---|---|---|
| plantdoc |  | 1 |
| plantwild | grape leaf spot | 197 |
| plantwild | blueberry mummy berry | 47 |
| plantwild | blueberry scorch | 43 |
| plantwild | blueberry anthracnose | 42 |
| plantwild | raspberry gray mold | 40 |
| plantwild | raspberry yellow rust | 37 |
| plantwild | blueberry botrytis blight | 36 |
| plantwild | raspberry fire blight | 34 |
| plantwild | bell pepper frogeye leaf spot | 31 |
| plantwild | bell pepper powdery mildew | 28 |
| plantwild | raspberry leaf spot | 19 |
| plantwild | peach anthracnose | 13 |
| plantwild | peach rust | 8 |
| plantwild | coffee black rot | 7 |
| rocole | export | 1,560 |
| rocole | Photos | 1,560 |
| groundnut_leaf | LEAF SPOT (EARLY AND LATE) | 450 |
| groundnut_leaf | ALTERNARIA LEAF SPOT | 450 |
| sorghum_disease | loose smut | 1,777 |
| sorghum_disease | Cereal Grain molds | 1,220 |
| sorghum_disease | Head Smut | 499 |
| sorghum_disease | Covered Kernel smut | 279 |
