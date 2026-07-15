from pathlib import Path

from chrono24.cleaning import clean_watches
from chrono24.data import load_watches
from chrono24.features import build_features
from chrono24.outliers import remove_outliers
from chrono24.plots import save_all_plots
from chrono24.training import (
    evaluate_model,
    save_metrics,
    save_model,
    select_features,
    split_data,
    train_final_model,
    train_model,
)

RAW_PATH = Path("data/raw/Watches.csv")
PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/model.txt")
METRICS_PATH = Path("models/metrics.json")
FIGURES_DIR = Path("reports/figures")


def _save(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"   -> {path}  ({len(df):,} rows x {df.shape[1]} cols)")


def main() -> None:
    print("[1/7] Load raw data")
    df = load_watches()
    _save(df, RAW_PATH)

    print("[2/7] Clean")
    df = clean_watches(df)
    _save(df, PROCESSED_DIR / "01_cleaned.csv")

    print("[3/7] Remove outliers")
    df = remove_outliers(df)
    _save(df, PROCESSED_DIR / "02_no_outliers.csv")

    print("[4/7] Build features")
    X, y = build_features(df)
    _save(X.assign(log_price=y), PROCESSED_DIR / "03_features.csv")

    print("[5/7] Train base model + select features")
    X_train, X_test, y_train, y_test = split_data(X, y)
    base = train_model(X_train, y_train, X_test, y_test)
    features = select_features(base, X_test, y_test)
    print(f"   kept {len(features)} features: {features}")

    print("[6/7] Train final model")
    model = train_final_model(X_train, y_train, X_test, y_test, features)

    print("[7/7] Evaluate + save artifacts")
    metrics = {"n_features": len(features), "features": features}
    metrics.update(evaluate_model(model, X_test[features], y_test))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:,.4f}")
    save_model(model, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)
    save_all_plots(model, X_test[features], y_test, model.predict(X_test[features]), FIGURES_DIR)

    print("\nDone. Outputs:")
    print(f"   raw + processed data -> {RAW_PATH.parent}/ and {PROCESSED_DIR}/")
    print(f"   model   -> {MODEL_PATH}")
    print(f"   metrics -> {METRICS_PATH}")
    print(f"   figures -> {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
