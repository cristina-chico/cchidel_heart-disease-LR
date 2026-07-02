"""
train.py
========
Train the final Random Forest model and save it to models/final_model.pkl.

Run from the project root:
    python train.py

The script re-creates the exact pipeline used in the notebook so the saved
artefact is reproducible from a clean environment.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.processing import run_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
MODEL_PATH   = os.path.join("models", "final_model.pkl")


def train() -> None:
    print("Loading and processing data …")
    X, y = run_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(random_state=RANDOM_STATE)),
    ])

    print("Training Random Forest …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n── Evaluation on test set ──────────────────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print("────────────────────────────────────────────\n")

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
