# ⚽ EPL 2015/16 xG-lite Pipeline

End-to-end Expected Goals (xG) modeling pipeline built using StatsBomb Open Data.

This project combines structured data engineering and applied machine learning to build a reproducible xG workflow from raw event data to an interactive analytics dashboard.

---

## 📊 Problem

Expected Goals (xG) models estimate the probability that a shot results in a goal.

This project builds a clean "xG-lite" baseline model using:

- Distance to goal  
- Shot angle  
- Header indicator  
- Penalty indicator  

The objective was not just to train a model — but to design a modular, production-style data pipeline.

---

## 🏗 Architecture

Bronze → Silver → Gold → Model → Scoring → Aggregation → Dashboard

```
data/
  bronze/   # Raw StatsBomb JSON
  silver/   # Flattened shot-level dataset
  gold/     # Feature-engineered modeling dataset

src/eplxg/
  ingest/
  transform/
  model/

app/
  app.py

run_pipeline.py
```

---

## ⚙️ Data Engineering Workflow

### Bronze
- Downloads 2015/16 EPL season from StatsBomb Open Data  
- Stores raw event JSON files  

### Silver
- Extracts shot events  
- Flattens nested JSON into structured tabular format  

### Gold
Engineers modeling features:
- Shot distance  
- Shot angle (goal-post geometry)  
- Header flag  
- Penalty flag  

Creates final ML-ready dataset.

---

## 🤖 Model

Model: Logistic Regression  
Train/validation split: 80/20 (stratified)

### Features
- distance  
- angle  
- is_header  
- is_penalty  

### Validation Performance

| Metric       | Value |
|-------------|-------|
| Log Loss    | ~0.28 |
| Brier Score | ~0.081 |
| ROC-AUC     | ~0.77 |

Goal rate: ~10%

The model achieves strong baseline predictive performance using minimal geometric features.

---

## 📈 Interactive Dashboard

The Streamlit dashboard provides:

- Model performance metrics display  
- Team sorting controls  
- Goals vs xG scatter plot (with reference line)  
- Player filtering by team  
- Over/underperformance toggle (Goals − xG)  
- Top 10 player bar chart  
- Minimum shots filter  
- Player name search  

This allows dynamic exploration of finishing performance across teams and players.

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```
git clone https://github.com/sparsh9449/epl-xg-lite.git
cd epl-xg-lite
```

### 2️⃣ Create virtual environment

```
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run full pipeline (one command)

```
python run_pipeline.py
```

### 5️⃣ Launch dashboard

```
streamlit run app/app.py
```

---

**Note:** Raw data and model artifacts are not stored in the repository.  
Running the pipeline generates them locally.

---

## 🎯 What This Project Demonstrates

- Structured multi-layer data pipeline (Bronze/Silver/Gold)
- Feature engineering from spatial event data
- Logistic regression modeling & evaluation
- Proper probabilistic metrics (log loss, Brier, ROC-AUC)
- Aggregation & analytics layer
- Interactive dashboard for stakeholder exploration
- Reproducibility and modular code design

---

## 🧠 Future Improvements

- Shot location heatmaps  
- Interaction features (distance × angle)  
- Calibration curve visualization  
- Multi-season training  
- Cloud deployment  

---

## 📚 Data Source

StatsBomb Open Data — English Premier League 2015/16
