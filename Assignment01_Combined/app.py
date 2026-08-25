from __future__ import annotations

import streamlit as st

from ui.components import (
    inject_global_styles,
    render_card,
    render_footer,
    render_page_header,
    render_section_header,
    render_status_badge,
)


st.set_page_config(
    page_title="Intelligent System Assignment 01",
    page_icon="IS",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()

with st.sidebar:
    st.markdown("### INTELLIGENT SYSTEM")
    st.caption("Assignment 01")
    st.divider()
    st.markdown("**Navigation**")
    st.caption("Use the page list above to open each system.")
    st.divider()
    render_status_badge("Cloud Deployment", "Live", "success")
    render_status_badge("Neo4j AuraDB", "Integrated", "success")

render_page_header(
    "Intelligent System",
    "Assignment 01",
    "Two machine-learning systems for classification and regression, integrated with Neo4j Knowledge Graph and deployed on the cloud.",
    ["Streamlit", "scikit-learn", "Neo4j AuraDB", "Render"],
)

left, right = st.columns(2)
with left:
    render_card(
        "Diabetes Prediction",
        "Binary Classification",
        "5 classifiers · Logistic Regression final model · 6 selected features · Neo4j Knowledge Graph",
    )
    st.page_link("pages/1_Diabetes_Prediction.py", label="Open Diabetes System", use_container_width=True)

with right:
    render_card(
        "Vietnam House Price Prediction",
        "Regression",
        "5 regressors · Random Forest final model · 6 selected features · Vietnam Housing Dataset 2024",
    )
    st.page_link("pages/2_House_Price_Prediction.py", label="Open House Price System", use_container_width=True)

render_section_header(
    "Production Architecture",
    "A clean path from structured inputs to model prediction, graph persistence, and cloud delivery.",
)

steps = st.columns(6)
architecture = [
    ("Data", "Structured clinical and housing inputs"),
    ("Preprocessing", "Imputation, scaling and encoding inside pipelines"),
    ("ML Models", "Traditional supervised learning artifacts"),
    ("Prediction", "Classification and regression outputs"),
    ("Neo4j KG", "Anonymous graph records and schema knowledge"),
    ("Render", "Public Streamlit web deployment"),
]
for column, (title, caption) in zip(steps, architecture):
    with column:
        render_card(title, title, caption)

render_footer()
