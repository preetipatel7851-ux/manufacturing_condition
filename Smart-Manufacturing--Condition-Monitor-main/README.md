# Smart Manufacturing — Condition Monitor

A machine-learning system that predicts a machine's operating **condition** from
live sensor readings and production KPIs, helping manufacturers catch early
warning signals of failures, downtime, and quality loss — the cases where
traditional fixed-threshold rules fall short.

## Problem

Manufacturing organizations struggle with:

- Unexpected machine breakdowns
- High maintenance and downtime costs
- Subtle, multidimensional anomalies buried in noisy sensor data

Fixed rule-based thresholds fail because normal behaviour varies by machine and
mode, and early anomalies are rarely visible on a single gauge. This project
replaces thresholds with a learned classifier.

## What it does

Predicts one of five condition classes from the available record:

| Class              | Meaning                                              |
| ------------------ | ---------------------------------------------------- |
| `Optimal_Operation`| Running within normal bounds                        |
| `Quality_Degraded` | Quality/reject KPIs drifting out of spec             |
| `High_Downtime`    | Elevated downtime trend                              |
| `Critical_Failure` | Imminent/active critical fault (overheat, power, …) |
| `Energy_Inefficient`| Abnormal energy consumption                         |

Inputs: `Temperature_C`, `Vibration_mm_s`, `Pressure_bar`, `Humidity_%`,
`Energy_Consumption_kWh`, `Machine_Speed_RPM`, `Downtime_Minutes`,
`Throughput_units_hr`, `Product_Quality_Score`, `Reject_Rate_%`,
`Operation_Mode`, `Machine_ID`, `Fault_Type`.

## Model

- Trains and compares **LogisticRegression**, **RandomForest**, and
  **HistGradientBoosting** on a stratified 80/20 split.
- Selection metric: test accuracy (with macro-F1 reported for the rare class).
- **Best model: HistGradientBoosting → ~99.9% test accuracy, 0.977 macro-F1.**
- The full preprocessing + classifier is saved as a single scikit-learn
  `Pipeline`, so the app applies identical transforms at inference time.

> **Note on `Fault_Type`:** it is a recorded diagnostic label. In a strict
> early-warning setting it would be leakage (Overheat/Power_Failure map
> deterministically to `Critical_Failure`). It is included here because the task
> is condition classification and the field is part of the available record,
> which is what makes critical failures detectable. For a pure sensor-only
> early-warning variant, drop the column in `model/train.py`.

## Project structure

```
app.py                      Streamlit frontend (predict + dashboard)
model/train.py              Training / model-selection script
model/model/                Saved pipeline (condition_model.joblib) + metrics.json
data/smart_manufacturing_dataset.csv
requirements.txt            Runtime deps
uv.lock                     Locked deps (uv)
```

## Run it

Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -r requirements.txt
streamlit run app.py
```

Or with plain pip + venv:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

### App pages

1. **Single Prediction** — enter sensor/KPI values, get the predicted condition
   and class probabilities.
2. **Batch Prediction** — upload a CSV (same columns), download predictions.
3. **Model Performance** — accuracy, macro-F1, per-class report, confusion matrix.

## Retrain

```bash
python -m model.train
```

This regenerates `model/model/condition_model.joblib` and `metrics.json`.
