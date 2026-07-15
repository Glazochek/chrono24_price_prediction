import numpy as np
import pandas as pd

from chrono24.features import (
    CATEGORY_COLUMNS,
    build_features,
    extract_color,
    extract_ref_key,
)


def make_outlier_free_frame(n=60):
    rng = np.random.default_rng(1)
    price = rng.uniform(1000, 30000, n)
    half = n // 2
    return pd.DataFrame(
        {
            "name": ["Rolex Submariner blue limited edition"] * half
            + ["Omega Speedmaster full set"] * (n - half),
            "price": price,
            "brand": ["Rolex"] * half + ["Omega"] * (n - half),
            "model": ["Submariner"] * half + ["Speedmaster"] * (n - half),
            "ref": ["116610LN"] * half + ["311.30.42.30.01.005"] * (n - half),
            "mvmt": ["Automatic"] * n,
            "casem": ["Steel"] * (n - 1) + ["Yellow gold"],
            "bracem": ["Steel"] * n,
            "yop": rng.integers(1990, 2024, n).astype(float),
            "sex": ["Men"] * n,
            "condition": ["Good"] * n,
            "size_mm": rng.uniform(36, 44, n),
            "log_price": np.log1p(price),
        }
    )


def test_extract_ref_key_brand_rules():
    assert extract_ref_key("116610LN", "Rolex") == "116610"  # first 6 alnum
    assert extract_ref_key("311.30.42.30", "Omega") == "311"  # first segment
    assert extract_ref_key("ab-cd12", "UnknownBrand") == "ABCD"  # default: 4 alnum, upper
    assert extract_ref_key(None, "Rolex") == "Unknown"


def test_extract_color():
    assert extract_color("Submariner blue dial") == "blue"
    assert extract_color("Plain name") == "unknown"
    assert extract_color(None) == "unknown"


def test_build_features_shape_and_target():
    df = make_outlier_free_frame()
    X, y = build_features(df)
    assert len(X) == len(y) == len(df)
    assert X.shape[1] == 22
    assert (y == df["log_price"]).all()


def test_no_leaky_or_dropped_columns_in_x():
    X, _ = build_features(make_outlier_free_frame())
    for col in ["price", "log_price", "name", "model", "ref", "brand_model", "model_median_price"]:
        assert col not in X.columns


def test_price_history_and_category_features():
    X, _ = build_features(make_outlier_free_frame())
    assert "ref_prefix" in X.columns
    assert "model_prefix_median_price" in X.columns
    assert X["model_prefix_median_price"].notna().all()
    for col in CATEGORY_COLUMNS:
        assert str(X[col].dtype) == "category"


def test_engineered_flags():
    X, _ = build_features(make_outlier_free_frame())
    assert X["has_limited"].iloc[0] == 1
    assert X["has_papers"].iloc[-1] == 1
    assert X["precious_case"].iloc[-1] == 1
