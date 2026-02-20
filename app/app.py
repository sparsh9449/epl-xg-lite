import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EPL xG-lite", layout="wide")

TEAM_PATH = Path("data/gold/team_metrics_2015_16.parquet")
PLAYER_PATH = Path("data/gold/player_metrics_2015_16.parquet")
PLAYER_TEAM_PATH = Path("data/gold/player_team_metrics_2015_16.parquet")
METRICS_PATH = Path("reports/metrics.json")


@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_tables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["shots", "goals"]:
        if c in out.columns:
            out[c] = out[c].astype(int)
    for c in ["xg", "goal_minus_xg"]:
        if c in out.columns:
            out[c] = out[c].round(2)
    return out


st.markdown("# ⚽ EPL 2015/16 xG-lite Dashboard")
st.markdown("Interactive expected goals analysis built with StatsBomb data.")
st.markdown("---")

# ---- Load data (with friendly errors) ----
try:
    team_df = load_parquet(TEAM_PATH)
    player_df = load_parquet(PLAYER_PATH)
except FileNotFoundError:
    st.error("Missing aggregated parquet files. Run the pipeline (or at least aggregate_metrics.py) first.")
    st.stop()

player_team_df = None
try:
    player_team_df = load_parquet(PLAYER_TEAM_PATH)
except FileNotFoundError:
    # We'll show a note in the UI; the dashboard still works without it.
    pass

try:
    metrics = load_json(METRICS_PATH)
except FileNotFoundError:
    metrics = None

# ---- Sidebar controls ----
st.sidebar.markdown("### 🏟 Team Settings")
team_sort_metric = st.sidebar.selectbox(
    "Sort teams by",
    ["goal_minus_xg", "xg", "goals", "shots"],
    index=0,
)

st.sidebar.divider()

st.sidebar.markdown("### 👤 Player Settings")
min_shots = st.sidebar.slider("Minimum shots (players)", 10, 150, 30)

view_mode = st.sidebar.radio(
    "Player view",
    ["Overperformers (Goals − xG)", "Underperformers (Goals − xG)"],
    index=0,
)

player_sort_metric = st.sidebar.selectbox(
    "Sort players by",
    ["goal_minus_xg", "xg", "goals", "shots"],
    index=0,
)

player_search = st.sidebar.text_input("Search player (optional)", "")

if player_team_df is not None:
    team_pick = st.sidebar.selectbox(
        "Filter players by team",
        ["All teams"] + sorted(player_team_df["team"].unique().tolist()),
        index=0,
    )
else:
    team_pick = "All teams"

ascending_gmxg = (view_mode.startswith("Underperformers"))

# ---- Tabs ----
tab_overview, tab_teams, tab_players = st.tabs(["Overview", "Teams", "Players"])

with tab_overview:

    st.markdown("## 📦 Season Summary")
    
    total_shots = int(shots_df.shape[0])
    total_goals = int(shots_df["is_goal"].sum())
    total_xg = float(shots_df["xg"].sum())
    avg_xg = float(shots_df["xg"].mean())
    
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Total Shots", f"{total_shots:,}")
    c2.metric("Total Goals", f"{total_goals:,}")
    c3.metric("Total xG", f"{total_xg:.1f}")
    c4.metric("Avg xG per Shot", f"{avg_xg:.3f}")
    
    st.divider()
    
    # -----------------------
    # 1️⃣ Season Highlights (Top of page)
    # -----------------------
    st.markdown("## 🎯 Season Highlights")

    top_team = team_df.sort_values("goal_minus_xg", ascending=False).iloc[0]
    bottom_team = team_df.sort_values("goal_minus_xg", ascending=True).iloc[0]

    top_player = player_df.sort_values("goal_minus_xg", ascending=False).iloc[0]
    top_xg_player = player_df.sort_values("xg", ascending=False).iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Top Overperforming Team",
            top_team["team"],
            f"{top_team['goal_minus_xg']:.2f} Goals − xG",
        )

        st.metric(
            "Top Overperforming Player",
            top_player["player"],
            f"{top_player['goal_minus_xg']:.2f} Goals − xG",
        )

    with col2:
        st.metric(
            "Top Underperforming Team",
            bottom_team["team"],
            f"{bottom_team['goal_minus_xg']:.2f} Goals − xG",
        )

        st.metric(
            "Most xG Player",
            top_xg_player["player"],
            f"{top_xg_player['xg']:.2f} xG",
        )
    
    # -----------------------
    # 2️⃣ Model Performance (Bottom of page)
    # -----------------------
    st.divider()
    st.markdown("## 📈 Model Performance")

    if metrics is None:
        st.info("Model metrics not found (reports/metrics.json). Run training to generate it.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Log Loss", round(metrics["log_loss"], 3))
        c2.metric("Brier Score", round(metrics["brier_score"], 3))
        c3.metric("ROC-AUC", round(metrics["roc_auc"], 3))

    # -----------------------
    # 3️⃣ About the Model (Very Bottom)
    # -----------------------
    with st.expander("About the Model"):
        st.write(
            """
            **xG-lite** estimates the probability that a shot becomes a goal.

            **Features**
            - Distance to goal
            - Shot angle
            - Header indicator
            - Penalty indicator

            **Model**
            - Logistic Regression

            **Metrics**
            - **Log Loss**: lower is better (probability quality)
            - **Brier Score**: lower is better (calibration / squared error)
            - **ROC-AUC**: higher is better (ranking ability)
            """
        )

with tab_teams:
    st.markdown("## 📊 Team Performance")
    team_sorted = team_df.sort_values(team_sort_metric, ascending=False).reset_index(drop=True)
    st.dataframe(fmt_tables(team_sorted), use_container_width=True, hide_index=True)

    st.markdown("### ⚖️ Goals vs Expected Goals")
    
    chart_df = team_df.copy()
    chart_df["goal_minus_xg"] = chart_df["goal_minus_xg"].round(2)
    chart_df["xg"] = chart_df["xg"].round(2)
    chart_df["goals"] = chart_df["goals"].astype(int)
    
    min_v = float(min(chart_df["xg"].min(), chart_df["goals"].min()))
    max_v = float(max(chart_df["xg"].max(), chart_df["goals"].max()))
    
    scatter = (
        alt.Chart(chart_df)
        .mark_circle(size=120)
        .encode(
            x=alt.X("xg:Q", title="Total xG"),
            y=alt.Y("goals:Q", title="Total Goals"),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("xg:Q", title="xG"),
                alt.Tooltip("goals:Q", title="Goals"),
                alt.Tooltip("goal_minus_xg:Q", title="Goals − xG"),
            ],
        )
        .interactive()
    )
    
    ref = (
        alt.Chart(pd.DataFrame({"x": [min_v, max_v], "y": [min_v, max_v]}))
        .mark_line(strokeDash=[6, 6])
        .encode(x="x:Q", y="y:Q")
    )
    
    st.altair_chart(scatter + ref, use_container_width=True)

with tab_players:
    st.markdown("## 🧍 Player Performance")

    # Choose player dataset
    if player_team_df is not None and team_pick != "All teams":
        base_players = player_team_df[player_team_df["team"] == team_pick].copy()
    else:
        base_players = player_df.copy()

    # Apply filters
    base_players = base_players[base_players["shots"] >= min_shots].copy()

    if player_search.strip():
        base_players = base_players[
            base_players["player"].str.contains(player_search.strip(), case=False, na=False)
        ].copy()

    players_sorted = base_players.sort_values(player_sort_metric, ascending=False).reset_index(drop=True)

    # Show over/under view ordering by goal_minus_xg too
    ranked = base_players.sort_values("goal_minus_xg", ascending=ascending_gmxg).reset_index(drop=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(fmt_tables(players_sorted), use_container_width=True, hide_index=True)

    with c2:
        st.write("### Top 10 (Goals − xG)")
        
        chart_df = ranked.head(10).copy()
        chart_df["goal_minus_xg"] = chart_df["goal_minus_xg"].round(2)
        chart_df["xg"] = chart_df["xg"].round(2)
        
        bar = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("goal_minus_xg:Q", title="Goals − xG"),
                y=alt.Y("player:N", sort="-x", title=""),
                tooltip=[
                    alt.Tooltip("player:N", title="Player"),
                    alt.Tooltip("shots:Q", title="Shots"),
                    alt.Tooltip("goals:Q", title="Goals"),
                    alt.Tooltip("xg:Q", title="xG"),
                    alt.Tooltip("goal_minus_xg:Q", title="Goals − xG"),
                ],
            )
        )
        
        st.altair_chart(bar, use_container_width=True)

    if player_team_df is None:
        st.info("Team filter requires player_team_metrics_2015_16.parquet. Run aggregate_metrics.py after updating it.")
