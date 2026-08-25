# Assignment 01 - Vietnam House Price Prediction Intelligent System

Dataset: House Price Prediction Dataset Vietnam - 2024  
Source: https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024

This project is a supervised regression system that estimates house listing price in billion VND. It is an educational machine-learning demonstration and not an official property valuation.

## Features

Full engineered features: Area, Frontage, Access Road, House direction, Balcony direction, Floors, Bedrooms, Bathrooms, Legal status, Furniture state, Province, District.

Final deployed six features: Area, Floors, Bedrooms, Bathrooms, Province, District.

## Models

Linear Regression, Decision Tree Regressor, Random Forest Regressor, Extra Trees Regressor, Gradient Boosting Regressor.

Metrics: MAE, MSE, RMSE, R2, MAPE. Primary selection metric: RMSE.

Controlled experiments:

1. Five-model comparison using six features.
2. Random Forest max_depth investigation.
3. Full 12 engineered features vs six selected features.

## Run Locally

```bash
conda activate houseprice_ml
streamlit run app.py
```

To initialize Neo4j locally, create `.env` from `.env.example` and run:

```bash
python neo4j/init_graph.py
```

Neo4j stores only anonymous house observations and predictions. No exact street address, name, email, phone, account, or identity is stored.

## Deployment

`render.yaml` defines a Streamlit web service. Configure Neo4j secrets in Render environment variables, not in source code.

The Streamlit layout is browser based and usable from desktop and mobile browsers.

## Limitations

The dataset contains listings, not necessarily transaction prices. The data represents 2024 market conditions. Province/District extraction is simplified. Several raw attributes contain substantial missing data. The six-feature deployed representation is a simplification.
