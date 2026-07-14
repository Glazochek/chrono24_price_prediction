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
    split_data,
    train_model,
)

MODEL_PATH = Path("models/model.txt")
METRICS_PATH = Path("models/metrics.json")
FIGURES_DIR = Path("reports/figures")


def main() -> None:
    df = load_watches()
    print(f"Loaded: {df.shape}")

    df = clean_watches(df)
    print(f"After cleaning: {df.shape}")

    df = remove_outliers(df)
    print(f"After outlier removal: {df.shape}")

    X, y = build_features(df)
    print(f"Features: {X.shape[1]} columns, {X.shape[0]} rows")

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, X_test, y_test)

    metrics = evaluate_model(model, X_test, y_test)
    print("\n--- Evaluation ---")
    for name, value in metrics.items():
        print(f"{name}: {value:,.4f}")

    save_model(model, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)
    save_all_plots(model, X, y_test, model.predict(X_test), FIGURES_DIR)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
