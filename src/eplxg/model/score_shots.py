from pathlib import Path
import joblib
import pandas as pd

GOLD_PATH = Path("data/gold/shots_features_2015_16.parquet")
MODEL_PATH = Path("models/xg_lite_logreg.joblib")
OUT_PATH = Path("data/gold/shots_scored_2015_16.parquet")

def main():
    df = pd.read_parquet(GOLD_PATH)
    model = joblib.load(MODEL_PATH)

    feature_cols = ["distance", "angle", "is_header", "is_penalty"]
    X = df[feature_cols].astype(float)

    df["xg"] = model.predict_proba(X)[:, 1]

    df.to_parquet(OUT_PATH, index=False)

    print(f"✅ Scored shots saved to {OUT_PATH}")
    print("Example rows:")
    print(df[["player", "xg", "is_goal"]].head())

if __name__ == "__main__":
    main()

