import streamlit as st
import pandas as pd
import json
from pathlib import Path
import altair as alt

st.set_page_config(layout="wide")

TEAM_PATH = Path("data/gold/team_metrics_2015_16.parquet")
PLAYER_PATH = Path("data/gold/player_metrics_2015_16.parquet")
METRICS_PATH = Path("reports/metrics.json")

st.title("⚽ EPL 2015/16 xG-lite Dashboard")

# Load data
team_df = pd.read_parquet(TEAM_PATH)
player_df = pd.read_parquet(PLAYER_PATH)

with open(METRICS_PATH) as f:
    metrics = json.load(f)

# ---------- Model Metrics ----------
st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Log Loss", round(metrics["log_loss"], 3))
col2.metric("Brier Score", round(metrics["brier_score"], 3))
col3.metric("ROC-AUC", round(metrics["roc_auc"], 3))

with st.expander("About the Model"):
    st.write(
        """
        **xG-lite** estimates the probability a shot becomes a goal.

        **Features used**
        - Distance to goal
        - Shot angle
        - Header indicator
        - Penalty indicator

        **Model**
        - Logistic Regression

        **Metrics**
        - *Log Loss*: lower is better (probability quality)
        - *Brier Score*: lower is better (calibration / squared error)
        - *ROC-AUC*: higher is better (ranking ability)
        """
    )

st.divider()

# ---------- Team Performance ----------
st.subheader("Team Performance")

sort_metric = st.selectbox(
    "Sort teams by",
    ["goal_minus_xg", "xg", "goals", "shots"],
    index=0
)

team_sorted = team_df.sort_values(sort_metric, ascending=False).reset_index(drop=True)
st.dataframe(team_sorted, use_container_width=True)

st.subheader("Goals vs Expected Goals (Teams)")

scatter = (
    alt.Chart(team_df)
    .mark_circle(size=120)
    .encode(
        x=alt.X("xg:Q", title="Total xG"),
        y=alt.Y("goals:Q", title="Total Goals"),
        tooltip=["team:N", "shots:Q", "xg:Q", "goals:Q", "goal_minus_xg:Q"],
    )
    .interactive()
)
st.altair_chart(scatter, use_container_width=True)

st.divider()

# ---------- Player Performance ----------
st.subheader("Player Performance")

min_shots = st.slider("Minimum Shots", 10, 120, 30)

filtered_players = player_df[player_df["shots"] >= min_shots].copy()

player_sort_metric = st.selectbox(
    "Sort players by",
    ["goal_minus_xg", "xg", "goals", "shots"],
    index=0
)

players_sorted = filtered_players.sort_values(player_sort_metric, ascending=False).reset_index(drop=True)
st.dataframe(players_sorted, use_container_width=True)

st.subheader("Top 10 Overperforming Players (Goals − xG)")
top10 = (
    filtered_players.sort_values("goal_minus_xg", ascending=False)
    .head(10)
    .set_index("player")
)

st.bar_chart(top10["goal_minus_xg"])

st.caption("xG-lite model using distance, angle, header, and penalty features.")

