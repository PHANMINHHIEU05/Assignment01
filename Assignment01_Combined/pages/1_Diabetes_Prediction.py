from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.diabetes.neo4j_service import (  # noqa: E402
    get_knowledge_graph_summary,
    get_recent_predictions,
    save_prediction,
    verify_connection,
)


MODULE_ROOT = PROJECT_ROOT / "modules" / "diabetes"
MODELS_DIR = MODULE_ROOT / "models"
FEATURES_PATH = MODELS_DIR / "selected_features.json"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

ZERO_AS_MISSING = ["Glucose", "BMI", "Insulin", "BloodPressure"]


st.set_page_config(page_title="Diabetes Prediction", page_icon="DI", layout="wide")


def load_json_file(path: Path, label: str):
    if not path.exists():
        st.error(f"{label} was not found: {path}")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(ttl=60)
def check_neo4j_status():
    try:
        return bool(verify_connection())
    except Exception:
        return False


def format_metric(value):
    return f"{float(value):.4f}"


features_payload = load_json_file(FEATURES_PATH, "selected_features.json")
selected_features = features_payload["selected_features"]
registry = load_json_file(REGISTRY_PATH, "model_registry.json")

st.title("Diabetes Prediction Intelligent System")
st.caption("Traditional Machine Learning Classification Demo")
st.info("Educational demonstration only. This is not a medical diagnosis tool.")

model_names = list(registry["models"].keys())
selected_model_name = st.sidebar.selectbox("Choose classifier", model_names)
model_info = registry["models"][selected_model_name]

if selected_model_name == registry["final_selected_model"]:
    st.sidebar.success("Scientific Final Model")
else:
    st.sidebar.info("Deployment Comparison Model")

st.sidebar.subheader("5-fold CV Performance")
metric_col_1, metric_col_2 = st.sidebar.columns(2)
metric_col_1.metric("Accuracy", format_metric(model_info["accuracy_cv"]))
metric_col_2.metric("Precision", format_metric(model_info["precision_cv"]))
metric_col_1.metric("Recall", format_metric(model_info["recall_cv"]))
metric_col_2.metric("F1-score", format_metric(model_info["f1_cv"]))

if selected_model_name == registry["final_selected_model"]:
    with st.sidebar.expander("Scientific Final Evaluation"):
        final_metrics = registry["final_test_metrics"]
        st.write(f"Accuracy: {format_metric(final_metrics['accuracy'])}")
        st.write(f"Precision: {format_metric(final_metrics['precision'])}")
        st.write(f"Recall: {format_metric(final_metrics['recall'])}")
        st.write(f"F1-score: {format_metric(final_metrics['f1'])}")

st.sidebar.divider()
st.sidebar.subheader("Knowledge Graph")
neo4j_available = check_neo4j_status()
st.sidebar.success("Neo4j AuraDB Connected") if neo4j_available else st.sidebar.warning("Neo4j AuraDB Unavailable")

st.subheader("Six Input Features")
input_col_1, input_col_2 = st.columns(2)
with input_col_1:
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
with input_col_2:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0)

input_values = {
    "Glucose": glucose,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age,
    "Insulin": insulin,
    "BloodPressure": blood_pressure,
}
sample = pd.DataFrame([{feature: input_values[feature] for feature in selected_features}])
sample[[feature for feature in ZERO_AS_MISSING if feature in sample.columns]] = sample[
    [feature for feature in ZERO_AS_MISSING if feature in sample.columns]
].replace(0, np.nan)

model_path = MODELS_DIR / model_info["file"]
model = load_model(str(model_path))

if st.button("Predict Diabetes", type="primary", use_container_width=True):
    prediction = int(model.predict(sample)[0])
    label = "Diabetes" if prediction == 1 else "No Diabetes"
    st.write(f"Selected Model: **{selected_model_name}**")
    st.warning(f"Model Prediction: {label}") if prediction == 1 else st.success(f"Model Prediction: {label}")
    probability = None
    decision_score = None
    if bool(model_info.get("supports_probability")) and hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(sample)[0][1])
        st.metric("Probability for Diabetes class", f"{probability * 100:.2f}%")
    elif hasattr(model, "decision_function"):
        decision_score = float(np.ravel(model.decision_function(sample))[0])
        st.metric("Decision Score", f"{decision_score:.4f}")
    try:
        graph_input_values = {
            feature: None if pd.isna(sample.iloc[0][feature]) else float(sample.iloc[0][feature])
            for feature in selected_features
        }
        save_prediction(selected_model_name, graph_input_values, prediction, probability, decision_score)
        st.success("Anonymous prediction saved to the Diabetes Knowledge Graph.")
    except Exception:
        st.warning("Prediction succeeded, but the Knowledge Graph record could not be saved.")

st.subheader("Knowledge Graph")
if neo4j_available:
    try:
        summary = get_knowledge_graph_summary()
        st.json(summary)
    except Exception:
        st.warning("Knowledge Graph summary is currently unavailable.")
    try:
        recent = get_recent_predictions(limit=10)
        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True)
    except Exception:
        st.warning("Recent prediction history is currently unavailable.")
else:
    st.info("Knowledge Graph features are unavailable. Model prediction remains available.")
