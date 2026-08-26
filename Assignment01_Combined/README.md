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

## Hybrid Knowledge Graph

The deployed application uses a hybrid Neo4j Knowledge Graph with three layers.

Layer 1: Machine Learning System Knowledge

- models
- selected features
- metrics
- representations
- targets/outcomes
- saved prediction artifacts

Layer 2: Domain Knowledge

Diabetes:

- Type 2 Diabetes domain concept
- risk factors
- clinical concepts
- general educational guidance
- complications
- medical specialties
- source attribution

House Price:

- geographic hierarchy: District -> Province
- 11 active model features, including Location parsed as District, Province
- valuation feature concepts represented in the model
- house price target context
- dataset source attribution

Layer 3: Dynamic Prediction Knowledge

- anonymous observation
- prediction node
- producing model
- predicted outcome or target
- location context for house price predictions

```text
                         NEO4J AURADB
                              |
            +-----------------+----------------+
            |                                  |
     ML SYSTEM GRAPH                    DOMAIN GRAPH
            |                                  |
 Models -> Features                       Diabetes
 Metrics -> Models                       /   |    \
 Representations                  RiskFactor Guidance Source
            |                                  |
            +-------- Prediction --------------+

House:

Model -> Features
   |
Prediction
   |
House Target
   |
District -> Province
```

The Streamlit UI renders graph data as interactive PyVis node-edge networks. The graph can be dragged, zoomed, panned and inspected with hover tooltips. Domain context is educational only and does not replace professional medical or property valuation advice.
