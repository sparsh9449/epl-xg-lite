import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


GOLD_PATH = Path("data/gold/shots_features_2015_16.parquet")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(GOLD_PATH)

    # Simple feature set (fast + interpretable)
    feature_cols = ["distance", "angle", "is_header", "is_penalty"]
    X = df[feature_cols].astype(float)
    y = df["is_goal"].astype(int)

    # Train/valid split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Probabilities
    p_val = model.predict_proba(X_val)[:, 1]

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_val": int(len(X_val)),
        "goal_rate": float(y.mean()),
        "log_loss": float(log_loss(y_val, p_val)),
        "brier_score": float(brier_score_loss(y_val, p_val)),
        "roc_auc": float(roc_auc_score(y_val, p_val)),
        "features": feature_cols,
        "coefficients": dict(zip(feature_cols, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0]),
    }

    # Save artifacts
    model_path = MODELS_DIR / "xg_lite_logreg.joblib"
    joblib.dump(model, model_path)

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Saved model: {model_path}")
    print(f"✅ Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

