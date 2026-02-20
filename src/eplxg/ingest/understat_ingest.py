import argparse
import re
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
import requests


BRONZE_DIR = Path("data/bronze")


def get_json(url: str) -> dict:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


def fetch_match_ids(league: str, season: int, max_matches: int) -> list[str]:
    # Example: https://understat.com/league/EPL/2023
    url = f"https://understat.com/league/{league}/{season}"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    # Match pages are linked as /match/<id>
    match_ids = re.findall(r"/match/(\d+)", r.text)
    # de-dupe preserving order
    seen = set()
    unique = []
    for mid in match_ids:
        if mid not in seen:
            seen.add(mid)
            unique.append(mid)

    return unique[:max_matches]


def fetch_match_shots(match_id: str) -> pd.DataFrame:
    url = f"https://understat.com/match/{match_id}"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    # shotsData = JSON.parse("....")
    m = re.search(r"shotsData\s*=\s*JSON\.parse\((['\"])(.*?)\1\)", r.text, flags=re.DOTALL)
    if not m:
        return pd.DataFrame()

    encoded = m.group(2)
    decoded = bytes(encoded, "utf-8").decode("unicode_escape")
    data = json.loads(decoded)

    rows = []
    for side in ("h", "a"):
        for s in data.get(side, []):
            s = dict(s)
            s["match_id"] = match_id
            s["side"] = side
            rows.append(s)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="EPL", help="Use EPL for Premier League")
    parser.add_argument("--season", type=int, default=2023, help="Season start year (e.g., 2023 for 23/24)")
    parser.add_argument("--max_matches", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.2, help="Polite delay between match requests")
    args = parser.parse_args()

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching match IDs for {args.league} {args.season}...")
    match_ids = fetch_match_ids(args.league, args.season, args.max_matches)
    print(f"Found {len(match_ids)} matches (using max_matches={args.max_matches})")

    all_shots = []
    for i, mid in enumerate(match_ids, 1):
        df = fetch_match_shots(mid)
        if not df.empty:
            all_shots.append(df)
        if args.sleep:
            time.sleep(args.sleep)
        if i % 10 == 0:
            print(f"  downloaded {i}/{len(match_ids)} matches...")

    shots = pd.concat(all_shots, ignore_index=True) if all_shots else pd.DataFrame()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = BRONZE_DIR / f"understat_shots_{args.league}_{args.season}_{ts}.csv"
    shots.to_csv(out_path, index=False)

    print(f"\n✅ Saved: {out_path}")
    print("Rows:", len(shots))
    print("Columns:", list(shots.columns)[:20], ("..." if len(shots.columns) > 20 else ""))


if __name__ == "__main__":
    main()

