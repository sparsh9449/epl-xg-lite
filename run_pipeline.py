import subprocess

steps = [
    ["python", "src/eplxg/ingest/download_season.py", "--competition_id", "2", "--season_id", "27"],
    ["python", "src/eplxg/transform/extract_shots.py"],
    ["python", "src/eplxg/transform/features_shots.py"],
    ["python", "src/eplxg/model/train_xg.py"],
    ["python", "src/eplxg/model/score_shots.py"],
    ["python", "src/eplxg/transform/aggregate_metrics.py"],
]

for step in steps:
    print(f"\nRunning: {' '.join(step)}\n")
    subprocess.run(step, check=True)

print("\nPipeline complete.")
print("Now run: streamlit run app/app.py")
