import math
from pathlib import Path

import numpy as np
import pandas as pd

SILVER_PATH = Path("data/silver/shots_2015_16.parquet")
GOLD_DIR = Path("data/gold")

# StatsBomb pitch is 120 x 80
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Goal center on StatsBomb coordinate system
GOAL_X = PITCH_LENGTH
GOAL_Y = PITCH_WIDTH / 2.0
GOAL_WIDTH = 8.0  # yards, but this is relative; we use as 8 in SB units approx for geometry
# We'll compute angle using goal posts at y = 40 +/- 4 (since goal width ~8)
LEFT_POST_Y = GOAL_Y - 4.0
RIGHT_POST_Y = GOAL_Y + 4.0


def shot_distance(x, y):
    if pd.isna(x) or pd.isna(y):
        return np.nan
    return math.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def shot_angle(x, y):
    """
    Angle between lines from shot location to left and right goal posts.
    """
    if pd.isna(x) or pd.isna(y):
        return np.nan

    a = math.sqrt((GOAL_X - x) ** 2 + (LEFT_POST_Y - y) ** 2)
    b = math.sqrt((GOAL_X - x) ** 2 + (RIGHT_POST_Y - y) ** 2)
    c = abs(RIGHT_POST_Y - LEFT_POST_Y)  # distance between posts

    # Law of cosines: angle at shot point
    # cos(theta) = (a^2 + b^2 - c^2) / (2ab)
    denom = 2 * a * b
    if denom == 0:
        return 0.0
    cos_theta = (a * a + b * b - c * c) / denom
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def main():
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SILVER_PATH)

    # Label: goal or not
    df["is_goal"] = (df["outcome"].fillna("").str.lower() == "goal").astype(int)

    # Features
    df["distance"] = df.apply(lambda r: shot_distance(r["x"], r["y"]), axis=1)
    df["angle"] = df.apply(lambda r: shot_angle(r["x"], r["y"]), axis=1)

    df["is_header"] = (df["body_part"].fillna("").str.lower() == "head").astype(int)
    df["is_penalty"] = (df["technique"].fillna("").str.lower() == "penalty").astype(int)

    # Keep only rows with geometry
    df = df.dropna(subset=["distance", "angle"])

    out_path = GOLD_DIR / "shots_features_2015_16.parquet"
    df.to_parquet(out_path, index=False)

    print(f"✅ Gold features saved: {out_path}")
    print("Rows:", len(df))
    print("Goals:", int(df['is_goal'].sum()))

if __name__ == "__main__":
    main()

