# Data preparation report

- images kept: **114,014** (train 91,206 · val 11,404 · test 11,404)
- classes: **60** of 103 in the knowledge base
- unreadable or tiny images dropped: 0
- duplicates removed: 6,451; images with conflicting labels removed: 9,837
- classes dropped for having < 20 images: coffee.cercospora

## Images per dataset

| dataset | images |
|---|---|
| plantvillage | 54,274 |
| cassava_kaggle | 21,397 |
| plantwild | 12,635 |
| plantseg | 11,609 |
| makerere_beans | 10,304 |
| plantdoc | 2,338 |
| ibean | 1,457 |

## Images per class

| class | images |
|---|---|
| apple.black_rot | 725 |
| apple.cedar_rust | 560 |
| apple.healthy | 2,165 |
| apple.scab | 838 |
| banana.black_sigatoka | 47 |
| banana.fusarium_wilt | 157 |
| banana.healthy | 242 |
| bean.angular_leaf_spot | 483 |
| bean.healthy | 6,019 |
| bean.rust | 5,592 |
| blueberry.healthy | 1,890 |
| cassava.bacterial_blight | 1,087 |
| cassava.brown_streak | 2,189 |
| cassava.green_mottle | 2,386 |
| cassava.healthy | 2,577 |
| cassava.mosaic | 13,158 |
| cherry.healthy | 1,189 |
| cherry.powdery_mildew | 1,128 |
| citrus.canker | 205 |
| citrus.greening | 5,649 |
| coffee.healthy | 137 |
| coffee.leaf_rust | 80 |
| grape.black_rot | 1,372 |
| grape.esca | 1,383 |
| grape.healthy | 689 |
| grape.leaf_blight | 1,076 |
| maize.common_rust | 1,400 |
| maize.gray_leaf_spot | 688 |
| maize.healthy | 1,313 |
| maize.northern_leaf_blight | 1,267 |
| other.unknown_plant | 16,923 |
| peach.bacterial_spot | 2,297 |
| peach.healthy | 627 |
| pepper.bacterial_spot | 1,188 |
| pepper.healthy | 1,767 |
| potato.early_blight | 1,218 |
| potato.healthy | 399 |
| potato.late_blight | 1,232 |
| raspberry.healthy | 663 |
| rice.blast | 97 |
| rice.healthy | 250 |
| soybean.healthy | 5,379 |
| squash.powdery_mildew | 2,064 |
| strawberry.healthy | 741 |
| strawberry.leaf_scorch | 1,163 |
| tomato.bacterial_spot | 2,376 |
| tomato.early_blight | 1,225 |
| tomato.healthy | 1,859 |
| tomato.late_blight | 2,147 |
| tomato.leaf_mold | 1,159 |
| tomato.mosaic_virus | 562 |
| tomato.septoria_leaf_spot | 1,994 |
| tomato.spider_mites | 1,678 |
| tomato.target_spot | 1,404 |
| tomato.yellow_leaf_curl_virus | 5,527 |
| wheat.leaf_rust | 54 |
| wheat.powdery_mildew | 87 |
| wheat.septoria | 79 |
| wheat.stem_rust | 52 |
| wheat.yellow_rust | 112 |

## Knowledge-base diseases with no training images (app can show advice, model cannot predict them)

apple.powdery_mildew, banana.xanthomonas_wilt, citrus.black_spot, citrus.healthy, citrus.melanose, coffee.cercospora, coffee.leaf_miner, coffee.phoma, coffee.red_spider_mite, groundnut.early_leaf_spot, groundnut.healthy, groundnut.late_leaf_spot, groundnut.rosette, groundnut.rust, maize.fall_armyworm, maize.lethal_necrosis, maize.streak_virus, mango.anthracnose, mango.bacterial_canker, mango.cutting_weevil, mango.die_back, mango.gall_midge, mango.healthy, mango.powdery_mildew, mango.sooty_mould, potato.bacterial_disease, potato.fungal_disease, potato.nematode, potato.pest_damage, potato.virus, rice.bacterial_leaf_blight, rice.bacterial_leaf_streak, rice.bacterial_panicle_blight, rice.brown_spot, rice.dead_heart, rice.downy_mildew, rice.hispa, rice.tungro, sorghum.anthracnose, sorghum.healthy, sorghum.leaf_blight, sorghum.rust, wheat.healthy

## Unmapped raw labels

Add a synonym in kb/*.yaml, or a `map: {"raw label": crop.condition}` entry for the dataset in pipeline/datasets.yaml (map to an empty string to skip on purpose).

| dataset | raw label | images |
|---|---|---|
| plantdoc |  | 1 |
| plantwild | grape downy mildew | 676 |
| plantwild | corn smut | 495 |
| plantwild | peach leaf curl | 417 |
| plantwild | squash leaf | 410 |
| plantwild | cherry leaf spot | 337 |
| plantwild | rice sheath blight | 326 |
| plantwild | wheat head scab | 319 |
| plantwild | apple mosaic virus | 270 |
| plantwild | soybean frog eye leaf spot | 238 |
| plantwild | wheat loose smut | 215 |
| plantwild | grape leaf spot | 197 |
| plantwild | bean mosaic virus | 183 |
| plantwild | peach brown rot | 173 |
| plantwild | bean halo blight | 171 |
| plantwild | blueberry rust | 161 |
| plantwild | banana bunchy top | 157 |
| plantwild | strawberry anthracnose | 156 |
| plantwild | soybean downy mildew | 153 |
| plantwild | coffee berry blotch | 118 |
| plantwild | soybean rust | 118 |
| plantwild | wheat bacterial leaf streak (black chaff) | 117 |
| plantwild | soybean mosaic | 117 |
| plantwild | bell pepper blossom end rot | 112 |
| plantwild | soybean bacterial blight | 91 |
| plantwild | peach scab | 78 |
| plantwild | soybean brown spot | 71 |
| plantwild | banana anthracnose | 71 |
| plantwild | banana cigar end rot | 60 |
| plantwild | banana cordana leaf spot | 55 |
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
| makerere_beans | als_2 | 2,597 |
| makerere_beans | als_1 | 2,501 |
| tanzania_maize | mln4 | 11,628 |
| tanzania_maize | mln1 | 11,628 |
| tanzania_maize | mln7 | 11,628 |
| tanzania_maize | mln3 | 11,628 |
| tanzania_maize | mln5 | 11,628 |
| tanzania_maize | mln2 | 11,628 |
| tanzania_maize | mln6 | 11,628 |
| tanzania_maize | healthy1 | 11,000 |
| tanzania_maize | healthy7 | 11,000 |
| tanzania_maize | healthy3 | 11,000 |
| tanzania_maize | healthy6 | 11,000 |
| tanzania_maize | healthy5 | 11,000 |
| tanzania_maize | healthy2 | 11,000 |
| tanzania_maize | healthy4 | 11,000 |
| tanzania_maize | healthy8 | 10,999 |
| tanzania_maize | msv2 | 10,300 |
| tanzania_maize | msv3 | 10,300 |
| tanzania_maize | msv4 | 10,300 |
| tanzania_maize | mln8 | 5,821 |
| tanzania_maize | msv6 | 5,487 |
| tanzania_maize | healthy9 | 5,164 |
| tanzania_maize | healthy10 | 5,163 |
| tanzania_maize | msv5 | 2,744 |
| tanzania_maize | msv7 | 2,740 |
