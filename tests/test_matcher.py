"""Label-mapping tests: python3 tests/test_matcher.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from common import Matcher, load_diseases

CASES = [
 # PlantDoc folder names (crop detected from the name)
 ("Apple Scab Leaf", None, "apple.scab"), ("Apple leaf", None, "apple.healthy"), ("Apple rust leaf", None, "apple.cedar_rust"),
 ("Bell_pepper leaf spot", None, "pepper.bacterial_spot"), ("Bell_pepper leaf", None, "pepper.healthy"),
 ("Corn Gray leaf spot", None, "maize.gray_leaf_spot"), ("Corn leaf blight", None, "maize.northern_leaf_blight"),
 ("Corn rust leaf", None, "maize.common_rust"), ("Potato leaf early blight", None, "potato.early_blight"),
 ("Potato leaf late blight", None, "potato.late_blight"), ("Soyabean leaf", None, "soybean.healthy"),
 ("Squash Powdery mildew leaf", None, "squash.powdery_mildew"), ("Tomato Early blight leaf", None, "tomato.early_blight"),
 ("Tomato leaf yellow virus", None, "tomato.yellow_leaf_curl_virus"), ("Tomato leaf mosaic virus", None, "tomato.mosaic_virus"),
 ("Tomato Septoria leaf spot", None, "tomato.septoria_leaf_spot"), ("Tomato leaf bacterial spot", None, "tomato.bacterial_spot"),
 ("Tomato mold leaf", None, "tomato.leaf_mold"), ("Tomato two spotted spider mites leaf", None, "tomato.spider_mites"),
 ("grape leaf black rot", None, "grape.black_rot"), ("Strawberry leaf", None, "strawberry.healthy"), ("Peach leaf", None, "peach.healthy"),
 ("Tomato leaf", None, "tomato.healthy"), ("grape leaf", None, "grape.healthy"),
 # PlantVillage
 ("Corn_(maize)___Common_rust_", None, "maize.common_rust"), ("Pepper,_bell___healthy", None, "pepper.healthy"),
 ("Orange___Haunglongbing_(Citrus_greening)", None, "citrus.greening"),
 # single-crop datasets (crop given in datasets.yaml)
 ("0", "cassava", "cassava.bacterial_blight"), ("3", "cassava", "cassava.mosaic"), ("4", "cassava", "cassava.healthy"),
 ("angular_leaf_spot", "bean", "bean.angular_leaf_spot"), ("bean_rust", "bean", "bean.rust"), ("healthy", "bean", "bean.healthy"),
 ("bacterial_leaf_blight", "rice", "rice.bacterial_leaf_blight"), ("blast", "rice", "rice.blast"), ("normal", "rice", "rice.healthy"),
 ("dead_heart", "rice", "rice.dead_heart"), ("Anthracnose", "mango", "mango.anthracnose"), ("Sooty Mould", "mango", "mango.sooty_mould"),
 ("Die Back", "mango", "mango.die_back"), ("Bacterial Canker", "mango", "mango.bacterial_canker"),
 ("Blight", "maize", "maize.northern_leaf_blight"), ("Common_Rust", "maize", "maize.common_rust"),
 ("Gray_Leaf_Spot", "maize", "maize.gray_leaf_spot"), ("Healthy", "maize", "maize.healthy"),
 ("Phytopthora", "potato", "potato.late_blight"), ("Virus", "potato", "potato.virus"), ("Nematode", "potato", "potato.nematode"),
 ("Fungi", "potato", "potato.fungal_disease"), ("stem_rust", "wheat", "wheat.stem_rust"), ("leaf_rust", "wheat", "wheat.leaf_rust"),
 ("healthy_wheat", "wheat", "wheat.healthy"), ("miner", "coffee", "coffee.leaf_miner"), ("rust_level_2", "coffee", "coffee.leaf_rust"),
 ("red_spider_mite", "coffee", "coffee.red_spider_mite"), ("cassava mosaic disease cmd", "cassava", "cassava.mosaic"),
 ("MSV", "maize", "maize.streak_virus"), ("MLN", "maize", "maize.lethal_necrosis"),
 # numbered folders (split archives)
 ("mln4", "maize", "maize.lethal_necrosis"), ("msv2", "maize", "maize.streak_virus"), ("healthy10", "maize", "maize.healthy"),
 ("als_2", "bean", "bean.angular_leaf_spot"), ("rust_level_3", "coffee", "coffee.leaf_rust"),
 # round 3: mango fruit, guava, papaya, sorghum, groundnut, multi-crop ABCGMP (label_join)
 ("Stem End Rot", "mango", "mango.stem_end_rot"), ("Scab", "mango", "mango.scab"), ("Anthracnose", "mango", "mango.anthracnose"),
 ("Mango Healthy Fruit", None, "mango.healthy"), ("Mango Bacterial Canker", None, "mango.bacterial_canker"),
 ("Banana Cordana", None, "banana.cordana_leaf_spot"), ("Citrus Cancker Leaf", None, "citrus.canker"),
 ("Guava Red Rust", None, "guava.red_rust"), ("Papaya Ring Spot", None, "papaya.ringspot_virus"),
 ("Fruit Flies", "guava", "guava.fruit_fly"), ("Styler End Rot", "guava", "guava.styler_end_rot"), ("Canker", "guava", "guava.scab"),
 ("Rust", "guava", "guava.red_rust"), ("Healthy fruits", "guava", "guava.healthy"),
 ("Curl", "papaya", "papaya.leaf_curl"), ("Mite Disease", "papaya", "papaya.mites"), ("Mosaic", "papaya", "papaya.ringspot_virus"),
 ("Cereal Grain molds", "sorghum", "sorghum.grain_mold"), ("Covered Kernel smut", "sorghum", "sorghum.covered_smut"),
 ("loose smut", "sorghum", "sorghum.loose_smut"), ("Head Smut", "sorghum", "sorghum.head_smut"),
 ("ALTERNARIA LEAF SPOT", "groundnut", "groundnut.alternaria_leaf_spot"),
 # must NOT map without a crop
 ("healthy", None, None), ("rust", None, None), ("3", None, None),
]

if __name__ == "__main__":
    m = Matcher(load_diseases())
    bad = 0
    for raw, crop, exp in CASES:
        got = m.match(raw, crop)
        if got != exp:
            bad += 1
            print(f"FAIL {raw!r} ({crop}) -> {got}, expected {exp}")
    print(f"{len(CASES) - bad}/{len(CASES)} correct")
    sys.exit(1 if bad else 0)
