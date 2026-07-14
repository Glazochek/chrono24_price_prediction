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
DROP_COLUMNS = ["name", "model", "ref", "price", "color", "yop", "condition"]
CATEGORY_COLUMNS = ["brand", "mvmt", "casem", "bracem", "sex"]
TARGET = "log_price"


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


def add_frequency_features(df, rare_brand_threshold=RARE_BRAND_THRESHOLD):
    brand_counts = df["brand"].value_counts()
    rare_brands = brand_counts[brand_counts < rare_brand_threshold].index
    df["brand"] = df["brand"].replace(rare_brands, "Other")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df[RAW_COLUMNS].copy()

    df = add_numeric_features(df)
    df = add_condition_features(df)
    df = add_material_features(df)
    df = add_name_features(df)
    df = add_color_features(df)
    df = add_frequency_features(df)

    X = df.drop(columns=DROP_COLUMNS + [TARGET])
    y = df[TARGET]

    for col in CATEGORY_COLUMNS:
        X[col] = X[col].astype("category")

    return X, y
