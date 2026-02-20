# ⚽ EPL 2015/16 xG-lite Pipeline

End-to-end Expected Goals (xG) modeling pipeline built using StatsBomb Open Data.

This project implements a structured data engineering + machine learning workflow to:

- Ingest raw event data  
- Extract shot-level records  
- Engineer geometric features  
- Train a logistic regression xG model  
- Evaluate predictive performance  
- Score all shots  
- Aggregate team & player metrics  
- Serve an interactive Streamlit dashboard  

---

## 📊 Problem

Expected Goals (xG) models estimate the probability that a shot results in a goal.

This project builds a clean “xG-lite” baseline model using:

- Distance to goal  
- Shot angle  
- Header indicator  
- Penalty indicator  

The focus was to build a reproducible, production-style ML pipeline — not just a notebook experiment.

---

## 🏗 Architecture

Bronze → Silver → Gold → Model → Scoring → Aggregation → Dashboard

```
data/
  bronze/   # Raw StatsBomb JSON
  silver/   # Flattened shot-level data
  gold/     # Feature-engineered dataset

src/eplxg/
  ingest/
  transform/
  model/

app/
  app.py

run_pipeline.py
```

---

## ⚙️ Data Pipeline

### Bronze
- Downloads 2015/16 EPL season from StatsBomb Open Data  
- Stores raw event JSON files  

### Silver
- Extracts shot events  
- Flattens nested JSON into tabular format  

### Gold
Computes:
- Shot distance  
- Shot angle (goal-post geometry)  
- Header flag  
- Penalty flag  

Creates final modeling dataset.

---

## 🤖 Model

Model: Logistic Regression  
Train/validation split: 80/20 stratified  

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

The model demonstrates strong baseline predictive performance using minimal features.

---

## 📈 Dashboard Features

The interactive Streamlit dashboard includes:

- Model performance metrics  
- Team sorting controls  
- Goals vs xG scatter plot  
- Player filtering by team  
- Over/underperformance toggle  
- Top 10 bar chart  
- Model explanation section  

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
Running the pipeline will generate them locally.

---

## 🧠 Future Improvements

- Shot map visualizations  
- Feature interaction terms (distance × angle)  
- Calibration curve analysis  
- Multi-season training  
- Cloud deployment  

---

## 📚 Data Source

StatsBomb Open Data — English Premier League 2015/16

---

## 🎯 Project Goals

This project emphasizes:

- Clean data engineering structure  
- Modular ML workflow  
- Reproducibility  
- Clear evaluation metrics  
- Practical analytics application  
- Production-style organization  
