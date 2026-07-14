import numpy as np
import pandas as pd

from chrono24.outliers import iqr_bounds, remove_outliers


def make_cleaned_frame(n=300):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "price": rng.uniform(1000, 20000, n),
            "brand": ["Rolex"] * (n - 3) + ["RareBrand"] * 3,
            "condition": ["Good"] * (n - 3) + ["Odd"] * 3,
            "yop": rng.integers(1990, 2024, n).astype(float),
            "size_mm": rng.uniform(36, 44, n),
        }
    )
    return df


def test_iqr_bounds_symmetric():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    low, high = iqr_bounds(s)
    assert low < s.min()
    assert high > s.max()


def test_log_price_added():
    out = remove_outliers(make_cleaned_frame())
    assert "log_price" in out.columns
    assert np.allclose(out["log_price"], np.log1p(out["price"]))


def test_out_of_range_values_removed():
    df = make_cleaned_frame()
    df.loc[0, "size_mm"] = 80.0
    df.loc[1, "yop"] = 1850.0
    out = remove_outliers(df)
    assert 0 not in out.index
    assert 1 not in out.index
    assert out["size_mm"].between(15, 60).all()
    assert out["yop"].between(1900, 2026).all()


def test_rare_categories_removed():
    out = remove_outliers(make_cleaned_frame())
    assert "RareBrand" not in out["brand"].values
    assert "Odd" not in out["condition"].values
