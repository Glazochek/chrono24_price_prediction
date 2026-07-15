from pathlib import Path

import pandas as pd

from chrono24.cleaning import clean_watches
from chrono24.data import load_watches
from chrono24.features import build_features
from chrono24.outliers import remove_outliers
from chrono24.plots import save_all_plots
from chrono24.training import (
    evaluate_model,
    save_metrics,
    save_model,
    split_data,
    train_model,
)

RAW_PATH = Path("data/raw/Watches.csv")
PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/model.txt")
METRICS_PATH = Path("models/metrics.json")
FIGURES_DIR = Path("reports/figures")


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"   -> {path}  ({len(df):,} rows x {df.shape[1]} cols)")


def main() -> None:
    print("[1/6] Load raw data")
    df = load_watches()
    _save(df, RAW_PATH)

    print("[2/6] Clean")
    df = clean_watches(df)
    _save(df, PROCESSED_DIR / "01_cleaned.csv")

    print("[3/6] Remove outliers")
    df = remove_outliers(df)
    _save(df, PROCESSED_DIR / "02_no_outliers.csv")

    print("[4/6] Build features")
    X, y = build_features(df)
    _save(X.assign(log_price=y), PROCESSED_DIR / "03_features.csv")

    print("[5/6] Train model")
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, X_test, y_test)

    print("[6/6] Evaluate + save artifacts")
    metrics = evaluate_model(model, X_test, y_test)
    for name, value in metrics.items():
        print(f"   {name}: {value:,.4f}")
    save_model(model, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)
    save_all_plots(model, X, y_test, model.predict(X_test), FIGURES_DIR)

    print("\nDone. Everything the pipeline produced:")
    print(f"   raw data       -> {RAW_PATH}")
    print(f"   processed data -> {PROCESSED_DIR}/  (01_cleaned, 02_no_outliers, 03_features)")
    print(f"   model          -> {MODEL_PATH}")
    print(f"   metrics        -> {METRICS_PATH}")
    print(f"   figures        -> {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
