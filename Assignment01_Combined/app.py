from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Intelligent System Assignment 01",
    page_icon="IS",
    layout="centered",
)

st.title("Intelligent System Assignment 01")
st.caption("Two traditional machine-learning systems in one Streamlit deployment")

st.subheader("1. Diabetes Prediction")
st.write(
    "Binary classification system using six clinical input features, five "
    "traditional classifiers, cross-validation metrics, final-test metrics for "
    "the scientific final model, and anonymous Neo4j Knowledge Graph records."
)

st.subheader("2. Vietnam House Price Prediction")
st.write(
    "Regression system using six property input features, five traditional "
    "regressors, CV metrics, held-out metrics for the scientific final model, "
    "and a separate House-prefixed Neo4j Knowledge Graph."
)

st.divider()
st.write("Deployment architecture:")
st.code(
    """                     Neo4j AuraDB
                      /        \\
                     /          \\
              Diabetes KG     House Price KG
                   ^               ^
                   |               |
                   +-------+-------+
                           |
                      Streamlit
                        Render
                      /        \\
                     /          \\
          Diabetes Page      House Price Page
               |                  |
          5 classifiers       5 regressors

                           |
                    Public HTTPS URL
                         /      \\
                        /        \\
                    Desktop    Mobile Browser""",
    language="text",
)

st.info(
    "Use the sidebar navigation to open either prediction system. Both systems "
    "are educational demonstrations, not medical diagnosis or official property valuation tools."
)
