import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BRONZE_DIR = Path("data/bronze")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="shots", help="Base name for the output file")
    parser.add_argument("--season", default="2023-2024", help="Label for season (used in filename only)")
    parser.add_argument("--url", required=True, help="Direct CSV URL (e.g., raw.githubusercontent.com/...)")
    args = parser.parse_args()

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CSV from: {args.url}")
    df = pd.read_csv(args.url)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = BRONZE_DIR / f"{args.name}_{args.season}_{ts}.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Saved: {out_path}")
    print("Rows:", len(df))
    print("Columns:", list(df.columns)[:25], ("..." if len(df.columns) > 25 else ""))


if __name__ == "__main__":
    main()

