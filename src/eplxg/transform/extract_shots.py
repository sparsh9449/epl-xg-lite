import json
from pathlib import Path
import pandas as pd

BRONZE_EVENTS_DIR = Path("data/bronze/statsbomb/events")
SILVER_DIR = Path("data/silver")

def main():
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for file in BRONZE_EVENTS_DIR.glob("*.json"):
        with open(file) as f:
            events = json.load(f)

        for event in events:
            if event.get("type", {}).get("name") == "Shot":
                location = event.get("location", [None, None])
                shot_data = event.get("shot", {})

                rows.append({
                    "match_id": event.get("match_id"),
                    "team": event.get("team", {}).get("name"),
                    "player": event.get("player", {}).get("name"),
                    "minute": event.get("minute"),
                    "second": event.get("second"),
                    "x": location[0] if len(location) > 0 else None,
                    "y": location[1] if len(location) > 1 else None,
                    "outcome": shot_data.get("outcome", {}).get("name"),
                    "body_part": shot_data.get("body_part", {}).get("name"),
                    "technique": shot_data.get("technique", {}).get("name"),
                    "play_pattern": event.get("play_pattern", {}).get("name"),
                })

    df = pd.DataFrame(rows)

    out_path = SILVER_DIR / "shots_2015_16.parquet"
    df.to_parquet(out_path, index=False)

    print(f"✅ Extracted {len(df)} shots")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()

