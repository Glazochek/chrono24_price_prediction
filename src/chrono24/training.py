import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from chrono24.features import CATEGORY_COLUMNS

TEST_SIZE = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
PERM_THRESHOLD = 0.001
BASE_ESTIMATORS = 500
FINAL_ESTIMATORS = 1500


def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def _fit(X_train, y_train, X_test, y_test, features, n_estimators, verbose):
    categorical = [c for c in CATEGORY_COLUMNS if c in features]
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        X_train[features],
        y_train,
        eval_set=[(X_test[features], y_test)],
        categorical_feature=categorical,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(verbose)],
    )
    return model


def train_model(X_train, y_train, X_test, y_test) -> lgb.LGBMRegressor:
    """Train the first model on all features (used to rank feature importance)."""
    return _fit(X_train, y_train, X_test, y_test, list(X_train.columns), BASE_ESTIMATORS, 0)


def select_features(model, X_test, y_test, threshold=PERM_THRESHOLD) -> list[str]:
    """Keep features whose permutation importance clears the threshold."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = pd.Series(result.importances_mean, index=X_test.columns)
    return importance[importance > threshold].sort_values(ascending=False).index.tolist()


def train_final_model(X_train, y_train, X_test, y_test, features) -> lgb.LGBMRegressor:
    """Retrain a larger model on only the selected features — the final model."""
    return _fit(X_train, y_train, X_test, y_test, features, FINAL_ESTIMATORS, 50)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred_log = model.predict(X_test)
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)

    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_test, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_test, y_pred_log)),
        "r2": float(r2_score(y_test, y_pred_log)),
        "rmse_usd": float(np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))),
        "mae_usd": float(mean_absolute_error(y_test_actual, y_pred_actual)),
        "mape_usd": float(np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100),
    }


def save_model(model: lgb.LGBMRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def load_model(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def predict_price(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Predict prices in dollars. Selects the columns the model was trained on."""
    features = model.feature_name()
    return np.expm1(model.predict(X[features]))
