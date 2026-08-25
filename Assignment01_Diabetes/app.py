from pathlib import Path
import json

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
FEATURES_PATH = MODELS_DIR / "selected_features.json"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

ZERO_AS_MISSING = [
    "Glucose",
    "BMI",
    "Insulin",
    "BloodPressure",
]


st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
)


def load_json_file(path: Path, label: str) -> dict:
    if not path.exists():
        st.error(f"{label} was not found: {path}")
        st.stop()

    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        st.error(f"{label} is not valid JSON: {error}")
        st.stop()


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


@st.cache_data(ttl=60)
def check_neo4j_status() -> bool:
    try:
        return bool(verify_connection())
    except Exception:
        return False


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def build_input_dataframe(input_values: dict, selected_features: list[str]) -> pd.DataFrame:
    sample = pd.DataFrame(
        [
            {
                feature: input_values[feature]
                for feature in selected_features
            }
        ]
    )

    if sample.columns.tolist() != selected_features:
        st.error("Input feature order does not match the selected feature configuration.")
        st.stop()

    missing_columns = [
        feature
        for feature in ZERO_AS_MISSING
        if feature in sample.columns
    ]
    sample[missing_columns] = sample[missing_columns].replace(0, np.nan)

    return sample


def build_graph_input_values(sample: pd.DataFrame, selected_features: list[str]) -> dict:
    graph_input_values = {}
    for feature in selected_features:
        value = sample.iloc[0][feature]
        if pd.isna(value):
            graph_input_values[feature] = None
        else:
            graph_input_values[feature] = float(value)
    return graph_input_values


features_payload = load_json_file(FEATURES_PATH, "selected_features.json")

if "selected_features" not in features_payload:
    st.error("selected_features.json does not contain the key 'selected_features'.")
    st.stop()

selected_features = features_payload["selected_features"]

if not isinstance(selected_features, list) or len(selected_features) != 6:
    st.error("selected_features.json must contain exactly 6 selected features.")
    st.stop()

registry = load_json_file(REGISTRY_PATH, "model_registry.json")

if "models" not in registry or not isinstance(registry["models"], dict):
    st.error("model_registry.json does not contain a valid 'models' section.")
    st.stop()

if len(registry["models"]) != 5:
    st.error("model_registry.json must define exactly 5 deployment models.")
    st.stop()

if "final_selected_model" not in registry:
    st.error("model_registry.json does not contain 'final_selected_model'.")
    st.stop()

if registry.get("selected_features") != selected_features:
    st.error(
        "Feature configuration mismatch between selected_features.json and model_registry.json"
    )
    st.stop()


st.title("Diabetes Prediction Intelligent System")
st.caption("Traditional Machine Learning Classification Demo")
st.info(
    "This application is an educational machine-learning demonstration. "
    "It is not a medical diagnosis tool."
)


model_names = list(registry["models"].keys())

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a machine learning model",
    model_names,
)
st.sidebar.write("Selected Model:")
st.sidebar.write(f"**{selected_model_name}**")

if selected_model_name == registry["final_selected_model"]:
    st.sidebar.success("Scientific Final Model")
else:
    st.sidebar.info("Deployment comparison model")

model_info = registry["models"][selected_model_name]

st.sidebar.subheader("5-fold CV Performance")
metric_col_1, metric_col_2 = st.sidebar.columns(2)
metric_col_1.metric("CV Accuracy", format_metric(model_info["accuracy_cv"]))
metric_col_2.metric("CV Precision", format_metric(model_info["precision_cv"]))
metric_col_1.metric("CV Recall", format_metric(model_info["recall_cv"]))
metric_col_2.metric("CV F1-score", format_metric(model_info["f1_cv"]))

if selected_model_name == registry["final_selected_model"]:
    with st.sidebar.expander("Scientific Final Evaluation"):
        final_metrics = registry.get("final_test_metrics", {})
        st.write(f"Accuracy: {format_metric(final_metrics.get('accuracy', 0.0))}")
        st.write(f"Precision: {format_metric(final_metrics.get('precision', 0.0))}")
        st.write(f"Recall: {format_metric(final_metrics.get('recall', 0.0))}")
        st.write(f"F1-score: {format_metric(final_metrics.get('f1', 0.0))}")

st.sidebar.divider()
st.sidebar.subheader("Knowledge Graph")
neo4j_available = check_neo4j_status()

if neo4j_available:
    st.sidebar.success("Neo4j AuraDB Connected")
else:
    st.sidebar.warning("Neo4j AuraDB Unavailable")

st.sidebar.caption(
    "Predictions continue to work even if the Knowledge Graph is unavailable."
)


st.subheader("Input Features")

input_col_1, input_col_2 = st.columns(2)

with input_col_1:
    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0,
        help="Plasma glucose concentration. Enter 0 if the value is unavailable.",
    )
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index. Enter 0 if the value is unavailable.",
    )
    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
    )

with input_col_2:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1,
    )
    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0,
        help="Enter 0 if the value is unavailable.",
    )
    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0,
        help="Enter 0 if the value is unavailable.",
    )

input_values = {
    "Glucose": glucose,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age,
    "Insulin": insulin,
    "BloodPressure": blood_pressure,
}

sample = build_input_dataframe(input_values, selected_features)

model_file = model_info.get("file")
model_path = MODELS_DIR / model_file if model_file else None

if model_path is None or not model_path.exists():
    st.error(f"Model file was not found for {selected_model_name}.")
    st.stop()

model = load_model(model_path)

if st.button("Predict", type="primary", use_container_width=True):
    prediction = model.predict(sample)[0]
    prediction_label = "Diabetes" if prediction == 1 else "No Diabetes"
    probability_for_graph = None
    decision_score_for_graph = None

    result_left, result_right = st.columns([1, 1])
    with result_left:
        st.write("Selected Model")
        st.write(f"**{selected_model_name}**")

    with result_right:
        st.write("Prediction")
        if prediction == 1:
            st.warning(f"Model Prediction: {prediction_label}")
        else:
            st.success(f"Model Prediction: {prediction_label}")

    supports_probability = bool(model_info.get("supports_probability"))

    if supports_probability and hasattr(model, "predict_proba"):
        probability = model.predict_proba(sample)[0][1]
        probability_for_graph = float(probability)
        st.metric(
            "Model probability for Diabetes class",
            f"{probability * 100:.2f}%",
        )
    elif hasattr(model, "decision_function"):
        decision_score = model.decision_function(sample)
        decision_score_for_graph = float(np.ravel(decision_score)[0])
        st.metric("Decision Score", f"{decision_score_for_graph:.4f}")
        st.caption(
            "Positive values generally favor the positive class; this score is not a probability."
        )
    else:
        st.caption("This model does not expose probability or decision score output.")

    with st.expander("Input Representation"):
        st.dataframe(sample.replace({np.nan: "Missing"}), use_container_width=True)

    graph_input_values = build_graph_input_values(sample, selected_features)
    predicted_class_for_graph = int(prediction)

    try:
        graph_record = save_prediction(
            model_name=selected_model_name,
            input_values=graph_input_values,
            predicted_class=predicted_class_for_graph,
            probability=probability_for_graph,
            decision_score=decision_score_for_graph,
        )
        st.success("Anonymous prediction saved to the Knowledge Graph.")
        with st.expander("Knowledge Graph Record"):
            st.write(f"Observation ID: `{graph_record['observation_id']}`")
            st.write(f"Prediction ID: `{graph_record['prediction_id']}`")
    except Exception:
        st.warning(
            "Prediction succeeded, but the Knowledge Graph record could not be saved."
        )


st.markdown("### System Pipeline")
st.markdown(
    """
User Input
→ Six-Feature Representation
→ Missing-Value Handling
→ Standardization
→ Selected ML Model
→ Prediction
"""
)
st.caption("SimpleImputer and StandardScaler are included inside the saved sklearn Pipeline.")

with st.expander("Model Information"):
    st.write(f"Selected model: **{selected_model_name}**")
    st.write(f"Model file: `{model_file}`")
    st.write(f"Supports probability: `{bool(model_info.get('supports_probability'))}`")
    st.write("Six selected features:")
    st.write(selected_features)
    st.write("5-fold CV metrics:")
    st.json(
        {
            "accuracy_cv": model_info["accuracy_cv"],
            "precision_cv": model_info["precision_cv"],
            "recall_cv": model_info["recall_cv"],
            "f1_cv": model_info["f1_cv"],
        }
    )
    if selected_model_name == registry["final_selected_model"]:
        st.success("This is the Scientific Final Model selected in the notebook.")
        st.caption(
            "This model is also represented in the Knowledge Graph as the "
            "Scientific Final Model for the Diabetes prediction task."
        )

st.subheader("Knowledge Graph")
st.write(
    "Neo4j represents relationships between Machine Learning Models, "
    "Features, the Six-Feature Representation, the Diabetes Condition, "
    "Evaluation Metrics, Anonymous Observations, Predictions, and Outcomes."
)
st.code(
    """User Input
→ Observation
→ Prediction
→ Model
→ Features
→ Diabetes Condition

Prediction
→ Outcome""",
    language="text",
)
st.caption(
    "Knowledge Graph records are anonymous and contain only the six model "
    "input measurements, selected model, prediction output, and timestamp."
)

if neo4j_available:
    with st.expander("Knowledge Graph Summary"):
        try:
            graph_summary = get_knowledge_graph_summary()
            summary_cols = st.columns(3)
            summary_cols[0].metric("Models", graph_summary["models"])
            summary_cols[1].metric("Features", graph_summary["features"])
            summary_cols[2].metric("Conditions", graph_summary["conditions"])
            summary_cols[0].metric("Representations", graph_summary["representations"])
            summary_cols[1].metric("Outcomes", graph_summary["outcomes"])
            summary_cols[2].metric("Metrics", graph_summary["metrics"])

            relationship_summary = pd.DataFrame(
                [
                    {
                        "Relationship": "USES_FEATURE",
                        "Count": graph_summary["uses_feature"],
                    },
                    {
                        "Relationship": "INPUT_FOR",
                        "Count": graph_summary["input_for"],
                    },
                    {
                        "Relationship": "HAS_CV_METRIC",
                        "Count": graph_summary["cv_metric_relationships"],
                    },
                    {
                        "Relationship": "HAS_FINAL_TEST_METRIC",
                        "Count": graph_summary[
                            "final_test_metric_relationships"
                        ],
                    },
                ]
            )
            st.dataframe(relationship_summary, use_container_width=True)
        except Exception:
            st.warning("Knowledge Graph summary is currently unavailable.")

    with st.expander("Recent Anonymous Predictions"):
        try:
            recent_predictions = get_recent_predictions(limit=10)
            if recent_predictions:
                recent_predictions_df = pd.DataFrame(recent_predictions)
                display_columns = [
                    "created_at",
                    "model_name",
                    "outcome_label",
                    "probability",
                    "decision_score",
                ]
                st.dataframe(
                    recent_predictions_df[display_columns],
                    use_container_width=True,
                )
            else:
                st.info("No prediction history is available yet.")
        except Exception:
            st.warning("Recent prediction history is currently unavailable.")
else:
    st.info(
        "Knowledge Graph features are unavailable right now. "
        "Model prediction remains available."
    )

with st.expander("Why Neo4j?"):
    st.write("Machine Learning:")
    st.markdown(
        """
- learns predictive patterns;
- produces predictions.
"""
    )
    st.write("Neo4j Knowledge Graph:")
    st.markdown(
        """
- represents relationships between system entities;
- stores anonymous prediction relationships;
- connects models, features, metrics, outcomes and the Diabetes target.
"""
    )
    st.caption("Neo4j does not replace the ML classifier.")

st.divider()
st.caption(
    "Educational machine-learning demonstration only. "
    "The output should not be interpreted as a medical diagnosis "
    "or treatment recommendation."
)
