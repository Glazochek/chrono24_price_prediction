import pandas as pd

from chrono24.cleaning import clean_watches


def make_raw_frame():
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3, 4, 5],
            "name": ["Rolex Sub", "Omega Speed", "Rolex Sub", "Seiko 5", "No price", "Cheap"],
            "price": ["$12,500", "$4,200", "$12,500", "$350", None, "$0"],
            "brand": ["Rolex", "Omega", "Rolex", None, "Casio", "Casio"],
            "model": ["Submariner", None, "Submariner", "5", "F91", "F91"],
            "ref": ["114060", "311", "114060", "SNK", "F91W", "F91W"],
            "mvmt": ["Automatic", "Manual", "Automatic", "Automatic", "Quartz", "Quartz"],
            "casem": ["Steel", "Steel", "Steel", "Steel", "Resin", "Resin"],
            "bracem": ["Steel", "Leather", "Steel", "Steel", "Resin", "Resin"],
            "yop": ["2020", "Approximation 1998", "2020", None, "2015", "2015"],
            "sex": ["Men", None, "Men", "Men", "Men", "Men"],
            "condition": ["New", None, "New", "Good", "Good", "Good"],
            "cond": [None, "Very good", None, None, None, None],
            "size": ["40 mm", "42.5 mm", "40 mm", None, "35 mm", "35 mm"],
        }
    )


def test_price_parsed_and_invalid_rows_dropped():
    out = clean_watches(make_raw_frame())
    assert out["price"].dtype.kind == "f"
    assert (out["price"] > 0).all()
    assert "No price" not in out["name"].values
    assert "Cheap" not in out["name"].values


def test_duplicates_removed():
    out = clean_watches(make_raw_frame())
    assert (out["name"] == "Rolex Sub").sum() == 1


def test_condition_filled_from_cond():
    out = clean_watches(make_raw_frame())
    omega = out[out["name"] == "Omega Speed"]
    assert omega["condition"].iloc[0] == "Very good"
    assert "cond" not in out.columns


def test_yop_and_size_extracted():
    out = clean_watches(make_raw_frame())
    omega = out[out["name"] == "Omega Speed"].iloc[0]
    assert omega["yop"] == 1998
    assert omega["yop_approx"] == 1
    assert omega["size_mm"] == 42.5


def test_missing_categoricals_filled_with_unknown():
    out = clean_watches(make_raw_frame())
    seiko = out[out["name"] == "Seiko 5"]
    assert seiko["brand"].iloc[0] == "Unknown"
    assert seiko["sex"].iloc[0] == "Men"
