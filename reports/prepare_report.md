# Data preparation report

- images kept: **3,202** (train 2,558 · val 322 · test 322)
- classes: **35** of 102 in the knowledge base
- unreadable or tiny images dropped: 0
- duplicates removed: 0; images with conflicting labels removed: 0
- classes dropped for having < 20 images: apple.cedar_rust, potato.healthy

## Images per dataset

| dataset | images |
|---|---|
| plantvillage | 3,202 |

## Images per class

| class | images |
|---|---|
| apple.black_rot | 40 |
| apple.healthy | 106 |
| apple.scab | 38 |
| blueberry.healthy | 82 |
| cherry.healthy | 49 |
| cherry.powdery_mildew | 63 |
| citrus.greening | 358 |
| grape.black_rot | 70 |
| grape.esca | 80 |
| grape.healthy | 28 |
| grape.leaf_blight | 59 |
| maize.gray_leaf_spot | 30 |
| maize.healthy | 75 |
| maize.northern_leaf_blight | 66 |
| peach.bacterial_spot | 141 |
| peach.healthy | 34 |
| pepper.bacterial_spot | 58 |
| pepper.healthy | 80 |
| potato.early_blight | 60 |
| potato.late_blight | 62 |
| raspberry.healthy | 21 |
| soybean.healthy | 306 |
| squash.powdery_mildew | 93 |
| strawberry.healthy | 27 |
| strawberry.leaf_scorch | 54 |
| tomato.bacterial_spot | 127 |
| tomato.early_blight | 66 |
| tomato.healthy | 99 |
| tomato.late_blight | 141 |
| tomato.leaf_mold | 58 |
| tomato.mosaic_virus | 27 |
| tomato.septoria_leaf_spot | 90 |
| tomato.spider_mites | 102 |
| tomato.target_spot | 81 |
| tomato.yellow_leaf_curl_virus | 331 |

## Knowledge-base diseases with no training images (app can show advice, model cannot predict them)

apple.cedar_rust, apple.powdery_mildew, banana.black_sigatoka, banana.fusarium_wilt, banana.healthy, banana.xanthomonas_wilt, bean.angular_leaf_spot, bean.healthy, bean.rust, cassava.bacterial_blight, cassava.brown_streak, cassava.green_mottle, cassava.healthy, cassava.mosaic, citrus.black_spot, citrus.canker, citrus.healthy, citrus.melanose, coffee.cercospora, coffee.healthy, coffee.leaf_miner, coffee.leaf_rust, coffee.phoma, coffee.red_spider_mite, groundnut.early_leaf_spot, groundnut.healthy, groundnut.late_leaf_spot, groundnut.rosette, groundnut.rust, maize.common_rust, maize.fall_armyworm, maize.lethal_necrosis, maize.streak_virus, mango.anthracnose, mango.bacterial_canker, mango.cutting_weevil, mango.die_back, mango.gall_midge, mango.healthy, mango.powdery_mildew, mango.sooty_mould, potato.bacterial_disease, potato.fungal_disease, potato.healthy, potato.nematode, potato.pest_damage, potato.virus, rice.bacterial_leaf_blight, rice.bacterial_leaf_streak, rice.bacterial_panicle_blight, rice.blast, rice.brown_spot, rice.dead_heart, rice.downy_mildew, rice.healthy, rice.hispa, rice.tungro, sorghum.anthracnose, sorghum.healthy, sorghum.leaf_blight, sorghum.rust, wheat.healthy, wheat.leaf_rust, wheat.powdery_mildew, wheat.septoria, wheat.stem_rust, wheat.yellow_rust

## Unmapped raw labels

Add a synonym in kb/*.yaml, or a `map: {"raw label": crop.condition}` entry for the dataset in pipeline/datasets.yaml (map to an empty string to skip on purpose).

| dataset | raw label | images |
|---|---|---|
