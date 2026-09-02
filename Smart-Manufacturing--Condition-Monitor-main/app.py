import json
import os

import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "model", "condition_model.joblib")
METRICS_PATH = os.path.join(HERE, "model", "model", "metrics.json")
DATA_PATH = os.path.join(HERE, "data", "smart_manufacturing_dataset.csv")

NUMERIC_FEATURES = [
    "Temperature_C",
    "Vibration_mm_s",
    "Pressure_bar",
    "Humidity_%",
    "Energy_Consumption_kWh",
    "Machine_Speed_RPM",
    "Downtime_Minutes",
    "Throughput_units_hr",
    "Product_Quality_Score",
    "Reject_Rate_%",
]
CATEGORICAL_FEATURES = ["Operation_Mode", "Machine_ID", "Fault_Type"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


def main():
    st.set_page_config(page_title="Smart Manufacturing Monitor", layout="wide")
    model = load_model()
    metrics = load_metrics()
    ref = load_reference_data()

    st.title("🏭 Smart Manufacturing — Condition Monitor")
    st.caption(
        "Predict machine condition (Optimal / Quality Degraded / High Downtime / "
        "Critical Failure / Energy Inefficient) from live sensor & KPI readings."
    )

    page = st.sidebar.radio(
        "Navigation", ["Single Prediction", "Batch Prediction", "Model Performance"]
    )

    if page == "Single Prediction":
        single_prediction(model, ref)
    elif page == "Batch Prediction":
        batch_prediction(model, ref)
    else:
        model_performance(metrics)


def _feature_inputs(ref):
    stats = {c: (float(ref[c].min()), float(ref[c].max()), float(ref[c].mean())) for c in NUMERIC_FEATURES}
    cols = st.columns(2)
    values = {}
    halves = (NUMERIC_FEATURES[:5], NUMERIC_FEATURES[5:])
    for col, group in zip(cols, halves):
        with col:
            for feat in group:
                lo, hi, mean = stats[feat]
                step = round((hi - lo) / 100, 3) if hi > lo else 0.1
                values[feat] = st.number_input(
                    feat, min_value=float(lo), max_value=float(hi),
                    value=round(mean, 3), step=step, format="%.3f",
                )
    op_mode = st.selectbox("Operation_Mode", sorted(ref["Operation_Mode"].unique()))
    machine_id = st.selectbox("Machine_ID", sorted(ref["Machine_ID"].unique()))
    fault_type = st.selectbox(
        "Fault_Type", ["None"] + sorted([x for x in ref["Fault_Type"].dropna().unique()])
    )
    return values, op_mode, machine_id, fault_type


def single_prediction(model, ref):
    st.header("🔎 Single Record Prediction")
    st.write("Enter the current readings for a machine and get its predicted condition.")
    values, op_mode, machine_id, fault_type = _feature_inputs(ref)

    if st.button("Predict Condition", type="primary"):
        row = {**values, "Operation_Mode": op_mode, "Machine_ID": machine_id, "Fault_Type": fault_type}
        X = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = model.classes_

        st.success(f"Predicted condition: **{pred}**")
        proba_df = pd.DataFrame({"Condition": classes, "Probability": proba}).sort_values(
            "Probability", ascending=False
        )
        st.subheader("Class probabilities")
        st.bar_chart(proba_df.set_index("Condition"))
        st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}), width="stretch")


def batch_prediction(model, ref):
    st.header("📦 Batch Prediction")
    st.write("Upload a CSV with the same sensor/KPI columns (Target_Class optional).")
    st.code(", ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES))

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        df["Predicted_Condition"] = model.predict(X)
        df["Predicted_Probability"] = model.predict_proba(X).max(axis=1)
        st.write(f"Predicted **{len(df)}** records.")
        st.dataframe(df.head(200), width="stretch")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions", csv, "predictions.csv", "text/csv")


def model_performance(metrics):
    st.header("📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    c3.metric("Best Model", metrics["best_model"])

    st.subheader("Per-class report")
    report = metrics["classification_report"]
    rows = []
    for cls, v in report.items():
        if isinstance(v, dict):
            rows.append({
                "Class": cls, "Precision": v["precision"], "Recall": v["Recall"],
                "F1": v["f1-score"], "Support": v["support"],
            })
    st.dataframe(pd.DataFrame(rows).style.format({
        "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"
    }), width="stretch")

    st.subheader("Confusion matrix")
    classes = metrics["classes"]
    cm = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
    st.dataframe(cm.style.background_gradient(cmap="Blues"), width="stretch")


if __name__ == "__main__":
    main()
