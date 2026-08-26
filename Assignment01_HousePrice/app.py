from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from neo4j_service import (
    get_knowledge_graph_summary,
    get_recent_predictions,
    save_prediction,
    verify_connection,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
METADATA_PATH = MODELS_DIR / "ui_metadata.json"


st.set_page_config(
    page_title="Vietnam House Price Prediction",
    page_icon="HP",
    layout="centered",
)


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


def optional_numeric(label: str, feature: str, ranges: dict):
    feature_range = ranges[feature]
    value = st.number_input(
        label,
        min_value=0.0,
        max_value=max(1000.0, float(feature_range["max"])),
        value=float(feature_range["median"]),
        step=1.0,
        help="Enter 0 if this value is not available in the listing.",
    )
    return np.nan if value == 0 else value


def optional_choice(label: str, feature: str, metadata: dict):
    choices = metadata["categorical_choices"].get(feature, [])
    missing_label = metadata.get("missing_label", "Not provided")
    options = [missing_label] + [choice for choice in choices if choice != missing_label]
    value = st.selectbox(label, options)
    return np.nan if value == missing_label else value


registry = load_json(REGISTRY_PATH, "model_registry.json")
metadata = load_json(METADATA_PATH, "ui_metadata.json")
selected_features = registry["selected_features"]

st.title("Vietnam House Price Prediction System")
st.caption("Vietnam Housing Dataset 2024 · 11 model features")
st.info(
    "Educational machine-learning demonstration. Predictions are model estimates "
    "and are not official property valuations."
)

model_names = list(registry["models"].keys())
selected_model_name = st.selectbox("Choose regression model", model_names)
model_info = registry["models"][selected_model_name]
model_path = MODELS_DIR / model_info["file"]
model = load_model(str(model_path))

if selected_model_name == registry["scientific_final_model"]:
    st.success("Scientific Final Model")
else:
    st.info("Deployment Comparison Model")

ranges = metadata["numeric_ranges"]
location = st.selectbox("Location", metadata["location_values"])
location_context = metadata.get("location_mapping", {}).get(location, {})

st.subheader("Physical Attributes")
area = optional_numeric("Area (m²)", "Area", ranges)
frontage = optional_numeric("Frontage (m)", "Frontage", ranges)
access_road = optional_numeric("Access Road (m)", "Access Road", ranges)
floors = optional_numeric("Floors", "Floors", ranges)
bedrooms = optional_numeric("Bedrooms", "Bedrooms", ranges)
bathrooms = optional_numeric("Bathrooms", "Bathrooms", ranges)

st.subheader("Categorical Attributes")
house_direction = optional_choice("House direction", "House direction", metadata)
balcony_direction = optional_choice("Balcony direction", "Balcony direction", metadata)
legal_status = optional_choice("Legal status", "Legal status", metadata)
furniture_state = optional_choice("Furniture state", "Furniture state", metadata)

sample = pd.DataFrame([{
    "Area": area,
    "Frontage": frontage,
    "Access Road": access_road,
    "House direction": house_direction,
    "Balcony direction": balcony_direction,
    "Floors": floors,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Legal status": legal_status,
    "Furniture state": furniture_state,
    "Location": location,
}], columns=selected_features)

if st.button("Predict House Price", type="primary"):
    predicted_billion = float(model.predict(sample)[0])
    predicted_vnd = predicted_billion * 1_000_000_000
    st.subheader("Model Estimate")
    st.metric("Estimated House Price", f"{predicted_billion:,.2f} billion VND")
    st.write(f"Approximate VND: {predicted_vnd:,.0f} VND")
    try:
        save_prediction(
            model_name=selected_model_name,
            input_values=sample.iloc[0].to_dict(),
            predicted_price_billion=predicted_billion,
            metadata=metadata,
        )
        st.success("Anonymous prediction saved to Neo4j Knowledge Graph.")
    except Exception as error:
        st.warning(f"Prediction completed, but Neo4j save is unavailable: {error}")

st.subheader("Location Context")
st.write({
    "Location model input": location,
    "District context": location_context.get("district"),
    "Province context": location_context.get("province"),
})

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
    summary = get_knowledge_graph_summary()
    st.write("Knowledge Graph Summary")
    st.json(summary)
except Exception:
    st.caption("Knowledge Graph summary is unavailable.")

try:
    recent = get_recent_predictions(limit=5)
    if recent:
        st.write("Recent Anonymous Price Predictions")
        st.dataframe(pd.DataFrame(recent), use_container_width=True)
except Exception:
    st.caption("Recent predictions are unavailable.")
