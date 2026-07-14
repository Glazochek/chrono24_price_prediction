import pandas as pd

KEY_COLUMNS = ["price", "brand", "mvmt", "casem", "bracem"]


def clean_watches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    df["price"] = df["price"].str.replace(r"[\$,]", "", regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    df["condition"] = df["condition"].fillna(df["cond"])
    df = df.drop(columns=["cond"])

    df["yop_approx"] = df["yop"].str.contains("Approximation", na=False).astype(int)
    df["yop"] = pd.to_numeric(df["yop"].str.extract(r"(\d{4})")[0], errors="coerce")

    df["size_mm"] = pd.to_numeric(df["size"].str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    df = df.drop(columns=["size"])

    for col in ["sex", "condition", "model", "brand"]:
        df[col] = df[col].fillna("Unknown")

    df = df.drop_duplicates(subset=["name", "ref", "price", "brand"], keep="first")

    df = df.dropna(subset=KEY_COLUMNS)
    df = df[(df["casem"] != "Unknown") & (df["bracem"] != "Unknown") & (df["mvmt"] != "Unknown")]
    return df.copy()
