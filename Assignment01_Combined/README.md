# Intelligent System Assignment 01

This Streamlit project combines two Assignment 01 intelligent systems:

1. Diabetes Prediction - binary classification.
2. Vietnam House Price Prediction - supervised regression.

The project is self-contained for cloud deployment. Runtime model artifacts are copied into `modules/diabetes/models/` and `modules/house_price/models/`. It does not retrain models at runtime.

## Run Locally

```bash
conda activate houseprice_ml
streamlit run app.py
```

## Render Deployment

`render.yaml` defines a Python Streamlit web service. Configure these environment variables in Render:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

Do not commit `.env` or Streamlit secrets.

## Architecture

```text
                     Neo4j AuraDB
                      /        \
                     /          \
              Diabetes KG     House Price KG
                   ^               ^
                   |               |
                   +-------+-------+
                           |
                      Streamlit
                        Render
                      /        \
                     /          \
          Diabetes Page      House Price Page
               |                  |
          5 classifiers       5 regressors

                           |
                    Public HTTPS URL
                         /      \
                        /        \
                    Desktop    Mobile Browser
```

Both domains may use one AuraDB instance. Diabetes labels and House-prefixed labels are kept separate.
