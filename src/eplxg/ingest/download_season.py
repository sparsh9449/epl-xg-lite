import argparse
import json
from pathlib import Path
import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

def download_json(url, out_path):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(r.text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition_id", type=int, required=True)
    parser.add_argument("--season_id", type=int, required=True)
    args = parser.parse_args()

    comp_id = args.competition_id
    season_id = args.season_id

    bronze_dir = Path("data/bronze/statsbomb")
    matches_path = bronze_dir / f"matches_{comp_id}_{season_id}.json"

    # Download matches list
    matches_url = f"{BASE_URL}/matches/{comp_id}/{season_id}.json"
    print(f"Downloading matches from: {matches_url}")
    download_json(matches_url, matches_path)

    with open(matches_path) as f:
        matches = json.load(f)

    match_ids = [m["match_id"] for m in matches]
    print(f"Found {len(match_ids)} matches")

    # Download events for each match
    events_dir = bronze_dir / "events"

    for i, match_id in enumerate(match_ids, 1):
        events_url = f"{BASE_URL}/events/{match_id}.json"
        out_path = events_dir / f"{match_id}.json"
        download_json(events_url, out_path)

        if i % 20 == 0:
            print(f"Downloaded {i}/{len(match_ids)} matches")

    print("✅ Season download complete")

if __name__ == "__main__":
    main()

