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
    get_diabetes_domain_context,
    get_knowledge_graph_summary,
    get_recent_predictions,
    get_latest_prediction_graph_data,
    get_system_graph_data,
    initialize_diabetes_domain_graph,
    save_prediction,
    verify_connection,
)
from ui.components import (  # noqa: E402
    format_time,
    format_percent,
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


MODULE_ROOT = PROJECT_ROOT / "modules" / "diabetes"
MODELS_DIR = MODULE_ROOT / "models"
FEATURES_PATH = MODELS_DIR / "selected_features.json"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
ZERO_AS_MISSING = ["Glucose", "BMI", "Insulin", "BloodPressure"]


st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="DI",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()


def load_json_file(path: Path, label: str):
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
    return initialize_diabetes_domain_graph()


def detail_table(summary: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Metric": key.replace("_", " ").title(), "Value": value} for key, value in summary.items()]
    )


features_payload = load_json_file(FEATURES_PATH, "selected_features.json")
selected_features = features_payload["selected_features"]
registry = load_json_file(REGISTRY_PATH, "model_registry.json")
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
    selected_model_name = st.selectbox("Choose classifier", model_names, label_visibility="collapsed")
    model_info = registry["models"][selected_model_name]
    render_model_badge(selected_model_name == registry["final_selected_model"])
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    render_status_badge("Neo4j", "Connected" if neo4j_available else "Unavailable", "success" if neo4j_available else "warning")

render_page_header(
    "Diabetes Prediction",
    "Diabetes Prediction",
    "Traditional machine-learning classification for diabetes risk prediction.",
    ["Binary Classification", "6 Features", "5 Classifiers"],
)
render_info_banner("Educational demonstration only. This is not a medical diagnosis tool.")

overview_cols = st.columns(4)
with overview_cols[0]:
    render_card("Selected Model", selected_model_name, "Scientific final model" if selected_model_name == registry["final_selected_model"] else "Deployment comparison model")
with overview_cols[1]:
    render_card("Model Type", "Binary Classification", "Predicts class 0 or 1")
with overview_cols[2]:
    render_card("Representation", "6 Features", "Saved sklearn pipeline input")
with overview_cols[3]:
    render_card("Knowledge Graph", "Connected" if neo4j_available else "Unavailable", "Neo4j AuraDB")

render_section_header("Patient Features", "Enter the six selected features used by the saved classifier pipeline.")
with st.container(border=True):
    input_col_1, input_col_2 = st.columns(2)
    with input_col_1:
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0, help="Plasma glucose concentration. Enter 0 if unavailable.")
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1, help="Body Mass Index. Enter 0 if unavailable.")
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, help="Family-history related diabetes pedigree score.")
    with input_col_2:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1, help="Patient age in years.")
        insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0, step=1.0, help="Enter 0 if unavailable.")
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0, help="Diastolic blood pressure. Enter 0 if unavailable.")
    submit = st.button("Predict Diabetes Risk", type="primary", use_container_width=True)

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

model_info = registry["models"][selected_model_name]
model = load_model(str(MODELS_DIR / model_info["file"]))
if submit:
    prediction = int(model.predict(sample)[0])
    label = "Diabetes" if prediction == 1 else "No Diabetes"
    probability = None
    decision_score = None
    sub_parts = [f"Model: {selected_model_name}"]
    progress = None
    if bool(model_info.get("supports_probability")) and hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(sample)[0][1])
        progress = probability
        sub_parts.append(f"Estimated diabetes probability: {probability * 100:.2f}%")
    elif hasattr(model, "decision_function"):
        decision_score = float(np.ravel(model.decision_function(sample))[0])
        sub_parts.append(f"Decision Score: {decision_score:.4f}")
    if selected_model_name == registry["final_selected_model"]:
        sub_parts.append("Scientific Final Model")
    render_prediction_card(
        "Model Prediction",
        label.upper(),
        " · ".join(sub_parts),
        "danger" if prediction == 1 else "success",
        progress,
    )
    try:
        graph_input_values = {
            feature: None if pd.isna(sample.iloc[0][feature]) else float(sample.iloc[0][feature])
            for feature in selected_features
        }
        save_prediction(selected_model_name, graph_input_values, prediction, probability, decision_score)
        render_info_banner("Anonymous prediction saved to the Diabetes Knowledge Graph.")
    except Exception:
        st.warning("Unable to save prediction to Knowledge Graph. The model prediction is still available.")

render_section_header("Model Performance", "5-fold cross-validation metrics from the selected saved model registry.")
render_metric_grid(
    [
        ("Accuracy", format_percent(model_info["accuracy_cv"]), "Correct classification rate"),
        ("Precision", format_percent(model_info["precision_cv"]), "Positive prediction reliability"),
        ("Recall", format_percent(model_info["recall_cv"]), "Positive class coverage"),
        ("F1 Score", format_percent(model_info["f1_cv"]), "Precision-recall balance"),
    ]
)

if selected_model_name == registry["final_selected_model"]:
    with st.expander("Scientific Final Evaluation"):
        final_metrics = registry["final_test_metrics"]
        render_metric_grid(
            [
                ("Accuracy", format_percent(final_metrics["accuracy"]), "Held-out test"),
                ("Precision", format_percent(final_metrics["precision"]), "Held-out test"),
                ("Recall", format_percent(final_metrics["recall"]), "Held-out test"),
                ("F1 Score", format_percent(final_metrics["f1"]), "Held-out test"),
            ]
        )

graph_tab, knowledge_tab, recent_tab, graph_details_tab, details_tab = st.tabs(
    ["Interactive Graph", "Related Knowledge", "Recent Predictions", "Graph Details", "Model Details"]
)

with graph_tab:
    render_section_header("Interactive Knowledge Graph", "Drag, zoom, pan and hover nodes to inspect ML and domain context.")
    if neo4j_available:
        try:
            graph_mode = st.radio("Graph view", ["System Graph", "Latest Prediction Context"], horizontal=True)
            graph_data = get_system_graph_data() if graph_mode == "System Graph" else get_latest_prediction_graph_data()
            if graph_data.get("nodes"):
                render_network_graph(graph_data["nodes"], graph_data["edges"], key=f"diabetes-{graph_mode}", height=520)
                render_graph_legend(
                    [
                        ("Model Input", ""),
                        ("Model", ""),
                        ("Prediction", ""),
                        ("Outcome / Target", "warning"),
                        ("Domain Knowledge", "danger"),
                        ("Source", ""),
                    ]
                )
                st.caption(
                    "This graph connects the machine-learning layer with educational diabetes-domain knowledge. "
                    "Context nodes are not model inputs."
                )
            else:
                render_empty_state("No latest prediction graph yet", "Run a prediction to create an anonymous graph observation.")
        except Exception:
            render_empty_state("Interactive graph unavailable", "Prediction remains operational.")
    else:
        render_empty_state("Knowledge Graph temporarily unavailable", "Prediction system is still operational.")

with knowledge_tab:
    render_section_header("Related Knowledge", "General educational information connected to the Diabetes outcome.")
    if neo4j_available:
        try:
            context = get_diabetes_domain_context()
            render_info_banner("This information is educational and does not replace professional medical diagnosis or treatment.")
            columns = st.columns(2)
            groups = [
                ("Risk Factors", context.get("risk_factors", [])),
                ("Clinical Context", context.get("clinical_metrics", [])),
                ("General Guidance", context.get("general_guidance", [])),
                ("Relevant Specialties", context.get("specialties", [])),
            ]
            for index, (title, rows) in enumerate(groups):
                with columns[index % 2]:
                    render_section_header(title)
                    for row in rows[:5]:
                        render_card(row.get("name", "Knowledge"), row.get("name", "Knowledge"), row.get("description", ""))
            with st.expander("Possible Complications"):
                for row in context.get("complications", []):
                    render_card(row.get("name", "Complication"), row.get("name", "Complication"), row.get("description", ""))
            with st.expander("Sources"):
                for source in context.get("sources", []):
                    st.markdown(f"- [{source.get('name')} · {source.get('title')}]({source.get('url')})")
        except Exception:
            render_empty_state("Related knowledge unavailable", "The ML prediction workflow remains operational.")
    else:
        render_empty_state("Related knowledge unavailable", "Neo4j is currently unavailable.")

with recent_tab:
    render_section_header("Recent Predictions", "Anonymous graph observations without personal identifiers.")
    if neo4j_available:
        try:
            recent_predictions = get_recent_predictions(limit=10)
            if recent_predictions:
                for record in recent_predictions:
                    if record.get("probability") is not None:
                        detail = f"{float(record['probability']) * 100:.2f}% probability"
                    elif record.get("decision_score") is not None:
                        detail = f"Decision Score {float(record['decision_score']):.4f}"
                    else:
                        detail = "Classifier output"
                    render_recent_prediction_card(
                        format_time(record.get("created_at")),
                        str(record.get("model_name", "Model")),
                        str(record.get("outcome_label", "Prediction")),
                        detail,
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
            summary = get_knowledge_graph_summary()
            relationship_total = sum(
                value
                for key, value in summary.items()
                if key not in {"models", "features", "conditions", "representations", "outcomes", "metrics", "observations", "predictions", "Disease", "RiskFactor", "ClinicalMetric", "GeneralGuidance", "Complication", "MedicalSpecialty", "KnowledgeSource"}
                and isinstance(value, int)
            )
            render_metric_grid(
                [
                    ("Models", str(summary.get("models", 0)), "Classifier nodes"),
                    ("Features", str(summary.get("features", 0)), "Input feature nodes"),
                    ("Predictions", str(summary.get("predictions", 0)), "Anonymous predictions"),
                    ("Observations", str(summary.get("observations", 0)), "Anonymous observations"),
                    ("Metrics", str(summary.get("metrics", 0)), "Metric nodes"),
                    ("Domain Nodes", str(sum(summary.get(label, 0) for label in ["Disease", "RiskFactor", "ClinicalMetric", "GeneralGuidance", "Complication", "MedicalSpecialty"])), "Context-only knowledge"),
                    ("Sources", str(summary.get("KnowledgeSource", 0)), "Traceability nodes"),
                    ("Relationships", str(relationship_total), "Stored links"),
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
        "Scientific final model selected in the notebook" if selected_model_name == registry["final_selected_model"] else "Comparison model available for deployment demonstration",
    )
    st.markdown("**Input features**")
    st.table(pd.DataFrame({"Feature": selected_features}))
    st.caption("The saved pipeline handles missing values and scaling internally. No model retraining occurs in Streamlit.")

render_footer()
