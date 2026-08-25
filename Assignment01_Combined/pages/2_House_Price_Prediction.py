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

from modules.house_price.neo4j_service import (  # noqa: E402
    get_knowledge_graph_summary,
    get_recent_predictions,
    save_prediction,
    verify_connection,
)


MODULE_ROOT = PROJECT_ROOT / "modules" / "house_price"
MODELS_DIR = MODULE_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
METADATA_PATH = MODELS_DIR / "ui_metadata.json"


st.set_page_config(page_title="Vietnam House Price Prediction", page_icon="HP", layout="centered")


def load_json(path: Path, label: str):
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


registry = load_json(REGISTRY_PATH, "model_registry.json")
metadata = load_json(METADATA_PATH, "ui_metadata.json")
selected_features = registry["selected_features"]

st.title("Vietnam House Price Prediction System")
st.caption("Vietnam Housing Dataset 2024")
st.info(
    "Educational machine-learning demonstration. Predictions are model estimates "
    "and are not official property valuations."
)

selected_model_name = st.selectbox("Choose regression model", list(registry["models"].keys()))
model_info = registry["models"][selected_model_name]
if selected_model_name == registry["scientific_final_model"]:
    st.success("Scientific Final Model")
else:
    st.info("Deployment Comparison Model")

province = st.selectbox("Province", metadata["provinces"])
district = st.selectbox("District", metadata["province_districts"].get(province, []))
ranges = metadata["numeric_ranges"]
area = st.number_input("Area", min_value=0.0, max_value=max(1000.0, float(ranges["Area"]["max"])), value=float(ranges["Area"]["median"]), step=1.0)
floors = st.number_input("Floors", min_value=0.0, max_value=50.0, value=float(ranges["Floors"]["median"]), step=1.0)
bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=50.0, value=float(ranges["Bedrooms"]["median"]), step=1.0)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=50.0, value=float(ranges["Bathrooms"]["median"]), step=1.0)

sample = pd.DataFrame([{
    "Area": np.nan if area == 0 else area,
    "Floors": np.nan if floors == 0 else floors,
    "Bedrooms": np.nan if bedrooms == 0 else bedrooms,
    "Bathrooms": np.nan if bathrooms == 0 else bathrooms,
    "Province": province,
    "District": district,
}], columns=selected_features)

model = load_model(str(MODELS_DIR / model_info["file"]))
if st.button("Predict House Price", type="primary"):
    predicted_billion = float(model.predict(sample)[0])
    predicted_vnd = predicted_billion * 1_000_000_000
    st.subheader("Model Estimate")
    st.write(f"Selected Model: **{selected_model_name}**")
    st.metric("Estimated House Price", f"{predicted_billion:,.2f} billion VND")
    st.write(f"Approximate VND: {predicted_vnd:,.0f} VND")
    try:
        save_prediction(selected_model_name, sample.iloc[0].to_dict(), predicted_billion)
        st.success("Anonymous prediction saved to the House Price Knowledge Graph.")
    except Exception as error:
        st.warning(f"Prediction completed, but Neo4j save is unavailable: {error}")

st.subheader("Selected Model Metrics")
metrics = model_info["cv_metrics"]
cols = st.columns(2)
cols[0].metric("CV MAE", format_metric(metrics["MAE"]))
cols[1].metric("CV MSE", format_metric(metrics["MSE"]))
cols[0].metric("CV RMSE", format_metric(metrics["RMSE"]))
cols[1].metric("CV R2", format_metric(metrics["R2"]))
st.metric("CV MAPE", format_metric(metrics["MAPE"]))

if selected_model_name == registry["scientific_final_model"]:
    with st.expander("Scientific Held-out Test Metrics"):
        final_metrics = registry["final_test_metrics"]
        st.write(f"MAE: {format_metric(final_metrics['MAE'])}")
        st.write(f"MSE: {format_metric(final_metrics['MSE'])}")
        st.write(f"RMSE: {format_metric(final_metrics['RMSE'])}")
        st.write(f"R2: {format_metric(final_metrics['R2'])}")
        st.write(f"MAPE: {format_metric(final_metrics['MAPE'])}")

st.divider()
st.subheader("Neo4j AuraDB")
if check_neo4j_status():
    st.success("Connected")
else:
    st.warning("Unavailable")

try:
    st.write("Knowledge Graph Summary")
    st.json(get_knowledge_graph_summary())
except Exception:
    st.caption("Knowledge Graph summary is unavailable.")

try:
    recent = get_recent_predictions(limit=5)
    if recent:
        st.write("Recent Anonymous Price Predictions")
        st.dataframe(pd.DataFrame(recent), use_container_width=True)
except Exception:
    st.caption("Recent predictions are unavailable.")
