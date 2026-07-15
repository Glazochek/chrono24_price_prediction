import pandas as pd

from chrono24.cleaning import clean_watches


def make_raw_frame():
    return pd.DataFrame(
        {
            "Unnamed: 0": range(6),
            "name": ["Rolex Sub", "Omega Speed", "Rolex Sub", "No price", "Gap casem", "Gap ref"],
            "price": ["$12,500", "$4,200", "$12,500", None, "$8,000", "$3,000"],
            "brand": ["Rolex", "Omega", "Rolex", "Casio", "Seiko", "Tudor"],
            "model": ["Submariner", None, "Submariner", "F91", "5", "BB"],
            "ref": ["114060", "311", "114060", "F91W", "SNK", None],
            "mvmt": ["Automatic", None, "Automatic", "Quartz", "Automatic", "Automatic"],
            "casem": ["Steel", "Steel", "Steel", "Resin", None, "Steel"],
            "bracem": ["Steel", "Leather", "Steel", "Resin", "Steel", "Steel"],
            "yop": ["2020", "Approximation 1998", "2020", "2015", "2019", "2019"],
            "sex": ["Men", None, "Men", "Men", "Men", "Men"],
            "condition": ["New", None, "New", "Good", "Good", "Good"],
            "cond": [None, "Very good", None, None, None, None],
            "size": ["40 mm", "42.5 mm", "40 mm", "35 mm", "40 mm", "40 mm"],
        }
    )


def test_price_parsed_and_invalid_rows_dropped():
    out = clean_watches(make_raw_frame())
    assert out["price"].dtype.kind == "f"
    assert (out["price"] > 0).all()
    assert "No price" not in out["name"].values


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


def test_missing_mvmt_filled_with_unknown():
    out = clean_watches(make_raw_frame())
    omega = out[out["name"] == "Omega Speed"]
    assert omega["mvmt"].iloc[0] == "Unknown"


def test_rows_missing_key_columns_dropped():
    out = clean_watches(make_raw_frame())
    assert "Gap casem" not in out["name"].values  # dropped: no casem
    assert "Gap ref" not in out["name"].values  # dropped: no ref
