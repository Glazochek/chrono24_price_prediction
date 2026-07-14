import numpy as np
import pandas as pd

from chrono24.features import (
    CATEGORY_COLUMNS,
    build_features,
    extract_color,
)


def make_outlier_free_frame(n=200):
    rng = np.random.default_rng(1)
    price = rng.uniform(1000, 20000, n)
    return pd.DataFrame(
        {
            "name": ["Rolex Submariner blue limited edition"] * (n // 2)
            + ["Omega Speedmaster full set"] * (n - n // 2),
            "price": price,
            "brand": ["Rolex"] * (n // 2) + ["Omega"] * (n - n // 2),
            "model": ["Submariner"] * (n // 2) + ["Speedmaster"] * (n - n // 2),
            "ref": ["114060"] * n,
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


def test_extract_color():
    assert extract_color("Submariner blue dial") == "blue"
    assert extract_color("Plain name") == "unknown"
    assert extract_color(None) == "unknown"


def test_build_features_shapes_and_target():
    df = make_outlier_free_frame()
    X, y = build_features(df)
    assert len(X) == len(y) == len(df)
    assert (y == df["log_price"]).all()


def test_no_leaky_or_dropped_columns_in_x():
    X, _ = build_features(make_outlier_free_frame())
    for col in ["price", "log_price", "name", "model", "ref", "color", "yop", "condition"]:
        assert col not in X.columns


def test_category_dtypes_and_no_numeric_nans():
    X, _ = build_features(make_outlier_free_frame())
    for col in CATEGORY_COLUMNS:
        assert str(X[col].dtype) == "category"
    numeric = X.select_dtypes(include=np.number)
    assert numeric.isnull().sum().sum() == 0


def test_engineered_flags():
    X, _ = build_features(make_outlier_free_frame())
    assert X["has_limited"].iloc[0] == 1
    assert X["has_papers"].iloc[-1] == 1
    assert X["same_material"].iloc[0] == 1
    assert X["precious_case"].iloc[-1] == 1
