import numpy as np
import pandas as pd

SIZE_MM_RANGE = (15, 60)
YOP_RANGE = (1900, 2026)
PRICE_QUANTILES = (0.005, 0.995)
IQR_COLUMNS = ["log_price", "yop", "size_mm"]
MIN_BRAND_COUNT = 25
MIN_CONDITION_COUNT = 50


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[(df["size_mm"] < SIZE_MM_RANGE[0]) | (df["size_mm"] > SIZE_MM_RANGE[1]), "size_mm"] = (
        np.nan
    )
    df.loc[(df["yop"] < YOP_RANGE[0]) | (df["yop"] > YOP_RANGE[1]), "yop"] = np.nan

    price_low, price_high = df["price"].quantile(list(PRICE_QUANTILES))
    df = df[(df["price"] >= price_low) & (df["price"] <= price_high)]

    df["log_price"] = np.log1p(df["price"])

    # bounds are computed once, then applied — rows with NaN in these columns drop out too
    bounds = {col: iqr_bounds(df[col].dropna()) for col in IQR_COLUMNS}
    for col, (low, high) in bounds.items():
        df = df[(df[col] >= low) & (df[col] <= high)]

    brand_counts = df["brand"].value_counts()
    df = df[~df["brand"].isin(brand_counts[brand_counts < MIN_BRAND_COUNT].index)]

    condition_counts = df["condition"].value_counts()
    df = df[~df["condition"].isin(condition_counts[condition_counts < MIN_CONDITION_COUNT].index)]

    return df
