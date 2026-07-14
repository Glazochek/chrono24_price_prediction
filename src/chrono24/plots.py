from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_loss_curve(model, out_dir: Path) -> None:
    eval_results = model.evals_result_
    plt.figure(figsize=(8, 5))
    plt.plot(eval_results["valid_0"]["l2"], label="Validation L2 loss")
    plt.xlabel("Boosting Round")
    plt.ylabel("L2 Loss (log_price)")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()


def save_residual_distribution(y_test_actual, y_pred_actual, out_dir: Path) -> None:
    residuals = y_pred_actual - y_test_actual
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=60)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual (Predicted - Actual, $)")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_distribution.png", dpi=150)
    plt.close()


def save_predicted_vs_actual(y_test_actual, y_pred_actual, out_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.2, s=10)
    max_val = y_test_actual.max()
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Predicted vs Actual Price")
    plt.tight_layout()
    plt.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
    plt.close()


def save_residuals_vs_predicted(y_test_actual, y_pred_actual, out_dir: Path) -> None:
    residuals = y_pred_actual - y_test_actual
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred_actual, y=residuals, alpha=0.2, s=10)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residual ($)")
    plt.title("Residuals vs Predicted Price")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_vs_predicted.png", dpi=150)
    plt.close()


def save_feature_importance(model, feature_names, out_dir: Path, top_n: int = 25) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n).sort_values()
    plt.figure(figsize=(10, 8))
    importances.plot(kind="barh")
    plt.title(f"Top {top_n} Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()


def save_all_plots(model, X, y_test, y_pred_log, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)

    save_loss_curve(model, out_dir)
    save_residual_distribution(y_test_actual, y_pred_actual, out_dir)
    save_predicted_vs_actual(y_test_actual, y_pred_actual, out_dir)
    save_residuals_vs_predicted(y_test_actual, y_pred_actual, out_dir)
    save_feature_importance(model, X.columns, out_dir)
