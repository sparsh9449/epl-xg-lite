import argparse
import re
import pandas as pd

COMPETITIONS_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default="", help="Regex filter for competition/season names")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_json(COMPETITIONS_URL)

    # Useful columns (exist in StatsBomb open data competitions.json)
    cols = [c for c in [
        "competition_id", "season_id", "competition_name", "season_name",
        "country_name", "competition_gender", "competition_youth"
    ] if c in df.columns]
    df = df[cols]

    if args.filter:
        pat = re.compile(args.filter, flags=re.IGNORECASE)
        mask = df.apply(lambda r: bool(pat.search(" ".join([str(x) for x in r.values]))), axis=1)
        df = df[mask]

    df = df.sort_values(["competition_name", "season_name"]).reset_index(drop=True)
    print(df.head(args.top).to_string(index=False))

if __name__ == "__main__":
    main()

