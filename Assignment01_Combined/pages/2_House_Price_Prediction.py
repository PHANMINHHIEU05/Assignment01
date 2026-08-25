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
    get_house_graph_summary,
    get_house_system_graph_data,
    get_knowledge_graph_summary,
    get_recent_predictions,
    get_latest_house_prediction_graph_data,
    initialize_house_domain_graph,
    save_prediction,
    verify_connection,
)
from ui.components import (  # noqa: E402
    format_decimal,
    format_percent,
    format_time,
    inject_global_styles,
    render_card,
    render_empty_state,
    render_footer,
    render_graph_legend,
    render_info_banner,
    render_metric_grid,
    render_model_badge,
    render_page_header,
    render_prediction_card,
    render_recent_prediction_card,
    render_schema_graph,
    render_section_header,
    render_status_badge,
)
from ui.knowledge_graph import render_network_graph  # noqa: E402


MODULE_ROOT = PROJECT_ROOT / "modules" / "house_price"
MODELS_DIR = MODULE_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
METADATA_PATH = MODELS_DIR / "ui_metadata.json"


st.set_page_config(
    page_title="Vietnam House Price Prediction",
    page_icon="HP",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()


def load_json(path: Path, label: str):
    if not path.exists():
        st.error(f"{label} was not found.")
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


@st.cache_data(ttl=3600)
def ensure_domain_graph():
    return initialize_house_domain_graph()


def detail_table(summary: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Metric": key.replace("_", " ").title(), "Value": value} for key, value in summary.items()]
    )


registry = load_json(REGISTRY_PATH, "model_registry.json")
metadata = load_json(METADATA_PATH, "ui_metadata.json")
selected_features = registry["selected_features"]
model_names = list(registry["models"].keys())
neo4j_available = check_neo4j_status()
if neo4j_available:
    try:
        ensure_domain_graph()
    except Exception:
        pass

with st.sidebar:
    st.markdown("### INTELLIGENT SYSTEM")
    st.caption("Assignment 01")
    st.divider()
    st.markdown("**CURRENT MODEL**")
    selected_model_name = st.selectbox("Choose regressor", model_names, label_visibility="collapsed")
    model_info = registry["models"][selected_model_name]
    render_model_badge(selected_model_name == registry["scientific_final_model"])
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    render_status_badge("Neo4j", "Connected" if neo4j_available else "Unavailable", "success" if neo4j_available else "warning")

render_page_header(
    "Vietnam House Price",
    "Vietnam House Price Prediction",
    "Machine-learning regression for estimating residential property prices from structured property information.",
    ["Regression", "6 Features", "Vietnam Housing Dataset 2024"],
)
render_info_banner("Educational model estimate. Not an official property valuation.")

overview_cols = st.columns(4)
with overview_cols[0]:
    render_card("Selected Model", selected_model_name, "Scientific final model" if selected_model_name == registry["scientific_final_model"] else "Deployment comparison model")
with overview_cols[1]:
    render_card("Problem", "Regression", "Predicts continuous price")
with overview_cols[2]:
    render_card("Representation", "6 Features", "Area, structure and location")
with overview_cols[3]:
    render_card("Target", "House Price", "billion VND")

render_section_header("Property Features", "Select location and enter the physical property attributes used by the saved model.")
ranges = metadata["numeric_ranges"]
with st.container(border=True):
    loc_1, loc_2 = st.columns(2)
    with loc_1:
        province = st.selectbox("Province", metadata["provinces"], help="Broad location feature extracted from Address.")
    with loc_2:
        district = st.selectbox("District", metadata["province_districts"].get(province, []), help="District list is filtered by Province.")

    num_1, num_2 = st.columns(2)
    with num_1:
        area = st.number_input("Area (m²)", min_value=0.0, max_value=max(1000.0, float(ranges["Area"]["max"])), value=float(ranges["Area"]["median"]), step=1.0, help="Property area in square meters. Enter 0 if unavailable.")
        bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=50.0, value=float(ranges["Bedrooms"]["median"]), step=1.0, help="Number of bedrooms. Enter 0 if unavailable.")
    with num_2:
        floors = st.number_input("Floors", min_value=0.0, max_value=50.0, value=float(ranges["Floors"]["median"]), step=1.0, help="Number of floors. Enter 0 if unavailable.")
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=50.0, value=float(ranges["Bathrooms"]["median"]), step=1.0, help="Number of bathrooms. Enter 0 if unavailable.")
    submit = st.button("Estimate House Price", type="primary", use_container_width=True)

sample = pd.DataFrame(
    [
        {
            "Area": np.nan if area == 0 else area,
            "Floors": np.nan if floors == 0 else floors,
            "Bedrooms": np.nan if bedrooms == 0 else bedrooms,
            "Bathrooms": np.nan if bathrooms == 0 else bathrooms,
            "Province": province,
            "District": district,
        }
    ],
    columns=selected_features,
)

model_info = registry["models"][selected_model_name]
model = load_model(str(MODELS_DIR / model_info["file"]))
if submit:
    predicted_billion = float(model.predict(sample)[0])
    predicted_vnd = predicted_billion * 1_000_000_000
    sub_parts = [
        f"≈ {predicted_vnd:,.0f} VND",
        f"Model: {selected_model_name}",
    ]
    if selected_model_name == registry["scientific_final_model"]:
        sub_parts.append("Scientific Final Model")
    render_prediction_card(
        "Estimated House Price",
        f"{predicted_billion:,.2f} B VND",
        " · ".join(sub_parts),
        "success",
    )
    try:
        save_prediction(selected_model_name, sample.iloc[0].to_dict(), predicted_billion)
        render_info_banner("Anonymous prediction saved to the House Price Knowledge Graph.")
    except Exception:
        st.warning("Unable to save prediction to Knowledge Graph. The model prediction is still available.")

render_section_header("Model Performance", "5-fold cross-validation metrics from the selected saved model registry.")
metrics = model_info["cv_metrics"]
render_metric_grid(
    [
        ("MAE", f"{format_decimal(metrics['MAE'])} B VND", "Mean absolute error"),
        ("MSE", format_decimal(metrics["MSE"]), "Mean squared error"),
        ("RMSE", f"{format_decimal(metrics['RMSE'])} B VND", "Primary selection metric"),
        ("R2", format_decimal(metrics["R2"]), "Explained variance quality"),
        ("MAPE", format_percent(metrics["MAPE"]), "Average percentage-type error"),
    ]
)

if selected_model_name == registry["scientific_final_model"]:
    with st.expander("Scientific Held-out Test Metrics"):
        final_metrics = registry["final_test_metrics"]
        render_metric_grid(
            [
                ("MAE", f"{format_decimal(final_metrics['MAE'])} B VND", "Held-out test"),
                ("MSE", format_decimal(final_metrics["MSE"]), "Held-out test"),
                ("RMSE", f"{format_decimal(final_metrics['RMSE'])} B VND", "Held-out test"),
                ("R2", format_decimal(final_metrics["R2"]), "Held-out test"),
                ("MAPE", format_percent(final_metrics["MAPE"]), "Held-out test"),
            ]
        )

graph_tab, context_tab, recent_tab, graph_details_tab, details_tab = st.tabs(
    ["Interactive Graph", "Property Context", "Recent Predictions", "Graph Details", "Model Details"]
)

with graph_tab:
    render_section_header("Interactive Knowledge Graph", "Drag, zoom, pan and hover nodes to inspect model, target and location context.")
    if neo4j_available:
        try:
            graph_mode = st.radio("Graph view", ["System Graph", "Latest Prediction Graph"], horizontal=True)
            graph_data = get_house_system_graph_data() if graph_mode == "System Graph" else get_latest_house_prediction_graph_data()
            if graph_data.get("nodes"):
                render_network_graph(graph_data["nodes"], graph_data["edges"], key=f"house-{graph_mode}", height=520)
                render_graph_legend(
                    [
                        ("Model Input", ""),
                        ("Model", ""),
                        ("Prediction", ""),
                        ("Target", "success"),
                        ("Location", "warning"),
                        ("Source", ""),
                    ]
                )
                st.caption(
                    "This graph connects the house-price prediction with the producing model, six deployment features, "
                    "and District to Province hierarchy. Context nodes are not additional model inputs."
                )
            else:
                render_empty_state("No latest prediction graph yet", "Run a prediction to create an anonymous house observation.")
        except Exception:
            render_empty_state("Interactive graph unavailable", "Prediction remains operational.")
    else:
        render_empty_state("Knowledge Graph temporarily unavailable", "Prediction system is still operational.")

with context_tab:
    render_section_header("Property Context", "Context connected to the current six-feature representation.")
    render_metric_grid(
        [
            ("Province", str(province), "Broad location"),
            ("District", str(district), "District-level location"),
            ("Area", f"{area:,.1f} m²", "Model input"),
            ("Floors", f"{floors:,.0f}", "Model input"),
            ("Bedrooms", f"{bedrooms:,.0f}", "Model input"),
            ("Bathrooms", f"{bathrooms:,.0f}", "Model input"),
        ]
    )
    render_info_banner("Valuation factors shown here are factors represented in the model, not official market classifications.")

with recent_tab:
    render_section_header("Recent House Price Predictions", "Anonymous predictions without exact street address.")
    if neo4j_available:
        try:
            recent_predictions = get_recent_predictions(limit=10)
            if recent_predictions:
                for record in recent_predictions:
                    price = float(record.get("predicted_price_billion", 0.0))
                    location = " · ".join(
                        item
                        for item in [record.get("province"), record.get("district")]
                        if item
                    )
                    render_recent_prediction_card(
                        format_time(record.get("created_at")),
                        str(record.get("model", "Model")),
                        f"{price:,.2f} B VND",
                        location or "Anonymous house price estimate",
                    )
            else:
                render_empty_state("No predictions recorded yet", "Run a prediction to create the first anonymous graph observation.")
        except Exception:
            render_empty_state("Recent predictions unavailable", "The model prediction system remains operational.")
    else:
        render_empty_state("Recent predictions unavailable", "Neo4j is currently unavailable.")

with graph_details_tab:
    render_section_header("Graph Details", "Technical graph metadata for teacher/demo inspection.")
    if neo4j_available:
        try:
            summary = get_house_graph_summary()
            render_metric_grid(
                [
                    ("Models", str(summary.get("HouseModel", 0)), "Regressor nodes"),
                    ("Features", str(summary.get("HouseFeature", 0)), "Input feature nodes"),
                    ("Predictions", str(summary.get("HousePrediction", 0)), "Anonymous predictions"),
                    ("Districts", str(summary.get("HouseDistrict", 0)), "Location nodes"),
                    ("Domain Factors", str(summary.get("HouseValuationFactor", 0)), "Context-only factor nodes"),
                    ("Sources", str(summary.get("HouseKnowledgeSource", 0)), "Traceability nodes"),
                ]
            )
            st.table(detail_table(summary))
        except Exception:
            render_empty_state("Graph details unavailable", "Neo4j metadata could not be loaded.")
    else:
        render_empty_state("Graph details unavailable", "Neo4j is currently unavailable.")

with details_tab:
    render_section_header("Model Details", "A compact explanation of the selected deployment artifact.")
    render_card(
        "Selected Algorithm",
        selected_model_name,
        "Selected by training CV RMSE" if selected_model_name == registry["scientific_final_model"] else "Comparison model available for deployment demonstration",
    )
    st.markdown("**Input features**")
    st.table(pd.DataFrame({"Feature": selected_features}))
    st.caption("The saved pipeline handles imputation, scaling, and one-hot encoding internally. No model retraining occurs in Streamlit.")

render_footer()
