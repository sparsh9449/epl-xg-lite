from pathlib import Path
import pandas as pd

SCORED_PATH = Path("data/gold/shots_scored_2015_16.parquet")
OUT_DIR = Path("data/gold")

def main():
    df = pd.read_parquet(SCORED_PATH)

    # Team aggregation
    team_df = (
        df.groupby("team")
        .agg(
            shots=("xg", "count"),
            goals=("is_goal", "sum"),
            xg=("xg", "sum"),
        )
        .reset_index()
    )
    team_df["goal_minus_xg"] = team_df["goals"] - team_df["xg"]

    # Player aggregation
    player_df = (
        df.groupby("player")
        .agg(
            shots=("xg", "count"),
            goals=("is_goal", "sum"),
            xg=("xg", "sum"),
        )
        .reset_index()
    )
    player_df["goal_minus_xg"] = player_df["goals"] - player_df["xg"]

    team_df.to_parquet(OUT_DIR / "team_metrics_2015_16.parquet", index=False)
    player_df.to_parquet(OUT_DIR / "player_metrics_2015_16.parquet", index=False)

    print("✅ Aggregated team and player metrics saved.")
    print("Top 5 overperforming players:")
    print(player_df.sort_values("goal_minus_xg", ascending=False).head())

if __name__ == "__main__":
    main()

