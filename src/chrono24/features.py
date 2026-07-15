import re
from datetime import datetime

import numpy as np
import pandas as pd

CURRENT_YEAR = datetime.now().year

CONDITION_ORDER = {
    "Incomplete": 0,
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
    "Very good": 4,
    "New": 5,
    "Unworn": 6,
}

COLOR_HUE = {
    "red": 0,
    "orange": 30,
    "salmon": 10,
    "brown": 25,
    "copper": 20,
    "bronze": 30,
    "yellow": 55,
    "gold": 50,
    "champagne": 45,
    "khaki": 55,
    "olive": 80,
    "green": 120,
    "teal": 175,
    "blue": 210,
    "navy": 220,
    "purple": 280,
    "pink": 330,
    "burgundy": 350,
}
COLOR_LIGHTNESS = {
    "black": 0.05,
    "anthracite": 0.2,
    "grey": 0.5,
    "gray": 0.5,
    "brown": 0.3,
    "burgundy": 0.25,
    "navy": 0.2,
    "bronze": 0.4,
    "copper": 0.45,
    "olive": 0.35,
    "red": 0.45,
    "purple": 0.35,
    "blue": 0.45,
    "green": 0.4,
    "teal": 0.45,
    "gold": 0.6,
    "orange": 0.55,
    "yellow": 0.7,
    "khaki": 0.65,
    "pink": 0.75,
    "salmon": 0.75,
    "champagne": 0.85,
    "beige": 0.85,
    "ivory": 0.92,
    "silver": 0.85,
    "white": 0.98,
}
ACHROMATIC = {"black", "white", "grey", "gray", "silver", "anthracite", "ivory", "beige"}
COLOR_LIST = list(set(COLOR_HUE) | set(COLOR_LIGHTNESS))

SIZE_BINS = [0, 34, 38, 40, 42, 44, 46, np.inf]
SIZE_LABELS = ["<34", "34-38", "38-40", "40-42", "42-44", "44-46", "46+"]
NAME_LEN_BINS = [-1, 20, 40, 60, 80, 100, np.inf]
NAME_LEN_LABELS = ["0-20", "21-40", "41-60", "61-80", "81-100", "100+"]

RARE_BRAND_THRESHOLD = 100
MIN_MODEL_COUNT = 3
MIN_PREFIX_COUNT = 5

RAW_COLUMNS = [
    "name",
    "price",
    "brand",
    "model",
    "ref",
    "mvmt",
    "casem",
    "bracem",
    "yop",
    "sex",
    "condition",
    "size_mm",
    "log_price",
]
DROP_COLUMNS = [
    "name",
    "model",
    "ref",
    "price",
    "color",
    "yop",
    "condition",
    "brand_model",
    "brand_model_prefix",
    "model_median_price",
]
CATEGORY_COLUMNS = ["brand", "mvmt", "casem", "bracem", "sex", "ref_prefix"]
TARGET = "log_price"


def _first_n_alnum(ref: str, n: int) -> str:
    clean = re.sub(r"[^A-Z0-9]", "", ref)
    return clean[:n] if clean else "Unknown"


def _first_segments(ref: str, n: int) -> str:
    parts = [p for p in re.split(r"[.\s/\-]+", ref) if p]
    return ".".join(parts[:n]) if parts else "Unknown"


# Each brand encodes its reference numbers differently, so the "family" prefix
# that groups similar watches is a different length / format per brand.
BRAND_REF_RULE = {
    "Rolex": lambda r: _first_n_alnum(r, 6),
    "Tudor": lambda r: _first_n_alnum(r, 4),
    "Audemars Piguet": lambda r: _first_n_alnum(r, 5),
    "Patek Philippe": lambda r: _first_n_alnum(r, 4),
    "Vacheron Constantin": lambda r: _first_n_alnum(r, 5),
    "Cartier": lambda r: _first_n_alnum(r, 4),
    "Seiko": lambda r: _first_n_alnum(r, 4),
    "IWC": lambda r: _first_n_alnum(r, 5),
    "Jaeger-LeCoultre": lambda r: _first_n_alnum(r, 4),
    "Longines": lambda r: _first_n_alnum(r, 4),
    "Panerai": lambda r: _first_n_alnum(r, 6),
    "Breitling": lambda r: _first_n_alnum(r, 4),
    "TAG Heuer": lambda r: _first_n_alnum(r, 4),
    "Omega": lambda r: _first_segments(r, 1),
    "Hublot": lambda r: _first_segments(r, 1),
    "A. Lange & Söhne": lambda r: _first_segments(r, 1),
    "Zenith": lambda r: _first_segments(r, 2),
    "Oris": lambda r: _first_segments(r, 3),
}


def extract_ref_key(ref, brand) -> str:
    if not isinstance(ref, str) or not ref.strip():
        return "Unknown"
    ref = ref.strip().upper()
    rule = BRAND_REF_RULE.get(brand, lambda r: _first_n_alnum(r, 4))
    return rule(ref)


def extract_color(name):
    if not isinstance(name, str):
        return "unknown"
    text = name.lower()
    for color in COLOR_LIST:
        if re.search(rf"\b{color}\b", text):
            return color
    return "unknown"


def add_numeric_features(df):
    df["watch_age"] = CURRENT_YEAR - df["yop"]
    df["size_mm"] = df["size_mm"].fillna(df.groupby("sex")["size_mm"].transform("median"))
    df["size_mm"] = df["size_mm"].fillna(df["size_mm"].median())

    df["size_mm"] = pd.cut(df["size_mm"], bins=SIZE_BINS, labels=SIZE_LABELS)

    df["yop"] = df["yop"].fillna(df["yop"].median())
    df["watch_age"] = df["watch_age"].fillna(df["watch_age"].median())
    return df


def add_condition_features(df):
    df["condition_score"] = df["condition"].map(CONDITION_ORDER)
    df["condition_score"] = df["condition_score"].fillna(df["condition_score"].median())
    return df


def add_material_features(df):
    df["same_material"] = (df["casem"] == df["bracem"]).astype(int)
    gold_pattern = "gold|platinum"
    df["precious_case"] = df["casem"].str.contains(gold_pattern, case=False, na=False).astype(int)
    df["precious_brace"] = df["bracem"].str.contains(gold_pattern, case=False, na=False).astype(int)
    return df


def add_name_features(df):
    name_filled = df["name"].fillna("")
    df["name_missing"] = df["name"].isna().astype(int)
    df["has_limited"] = name_filled.str.contains(
        "limited|anniversary|edition|remaster|re-edition", case=False
    ).astype(int)
    df["has_papers"] = name_filled.str.contains("papers|box|full set", case=False).astype(int)
    df["has_wealthy_words"] = name_filled.str.contains(
        "king|royal|luxury|exclusive|elite|prestige", case=False
    ).astype(int)

    name_len = name_filled.str.len()
    df["name_len_bucket"] = pd.cut(name_len, bins=NAME_LEN_BINS, labels=NAME_LEN_LABELS)
    return df


def add_color_features(df):
    df["color"] = df["name"].apply(extract_color)
    df["color_has_hue"] = df["color"].apply(lambda c: 0 if c in ACHROMATIC or c == "unknown" else 1)
    hue_deg = df["color"].map(COLOR_HUE).fillna(0)
    df["color_hue_sin"] = np.where(df["color_has_hue"] == 1, np.sin(np.radians(hue_deg)), 0)
    df["color_hue_cos"] = np.where(df["color_has_hue"] == 1, np.cos(np.radians(hue_deg)), 0)
    df["color_lightness"] = df["color"].map(COLOR_LIGHTNESS).fillna(0.5)
    return df


def add_price_history_features(
    df,
    rare_brand_threshold=RARE_BRAND_THRESHOLD,
    min_model_count=MIN_MODEL_COUNT,
    min_prefix_count=MIN_PREFIX_COUNT,
):
    rare_brands = (
        df["brand"].value_counts()[df["brand"].value_counts() < rare_brand_threshold].index
    )
    df["brand"] = df["brand"].replace(rare_brands, "Other")

    prefix = df.apply(lambda r: extract_ref_key(r["ref"], r["brand"]), axis=1)
    prefix_counts = prefix.value_counts()
    rare_prefixes = prefix_counts[prefix_counts < min_prefix_count].index
    df["ref_prefix"] = prefix.where(~prefix.isin(rare_prefixes), "Other")

    global_median = df["log_price"].median()

    df["brand_model"] = df["brand"] + "_" + df["model"]
    model_stats = df.groupby("brand_model")["log_price"].agg(["median", "count"])
    reliable_models = model_stats[model_stats["count"] >= min_model_count]["median"]
    df["model_median_price"] = df["brand_model"].map(reliable_models).fillna(global_median)

    df["brand_model_prefix"] = df["brand_model"] + "_" + df["ref_prefix"]
    prefix_stats = df.groupby("brand_model_prefix")["log_price"].agg(["median", "count"])
    reliable_prefixes = prefix_stats[prefix_stats["count"] >= min_model_count]["median"]
    df["model_prefix_median_price"] = (
        df["brand_model_prefix"].map(reliable_prefixes).fillna(df["model_median_price"])
    )

    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df[RAW_COLUMNS].copy()

    df = add_numeric_features(df)
    df = add_condition_features(df)
    df = add_material_features(df)
    df = add_name_features(df)
    df = add_color_features(df)
    df = add_price_history_features(df)

    X = df.drop(columns=DROP_COLUMNS + [TARGET])
    y = df[TARGET]

    for col in CATEGORY_COLUMNS:
        X[col] = X[col].astype("category")

    return X, y
