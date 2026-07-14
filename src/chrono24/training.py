import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from chrono24.features import CATEGORY_COLUMNS

TEST_SIZE = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50


def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_model(X_train, y_train, X_test, y_test) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        categorical_feature=CATEGORY_COLUMNS,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50)],
    )
    return model


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
    return np.expm1(model.predict(X))
