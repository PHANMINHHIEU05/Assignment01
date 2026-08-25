# Assignment 01 - Diabetes Prediction

Project for Intelligent System Development.

## Project Structure

- `data/`: contains Diabetes CSV dataset
- `notebooks/`: main Jupyter Notebook for the assignment
- `models/`: stores trained `.joblib` models later
- `figures/`: stores exported analysis charts later
- `neo4j/`: contains Knowledge Graph related scripts later
- `app.py`: Streamlit app entry point later
- `neo4j_service.py`: Neo4j connection service later
- `requirements.txt`: project dependencies

## Environment

Conda environment:

```bash
conda activate diabetes_ml
```

Jupyter kernel:

```text
Python (diabetes_ml)
```

## Run the Streamlit Application

```bash
conda activate diabetes_ml
cd /home/hiubeo/Documents/TKHTTM/Assignment01_Diabetes
python -m streamlit run app.py
```

## Neo4j Knowledge Graph Integration

Neo4j AuraDB stores structured graph relationships for the intelligent
system. Predictions are stored anonymously and connect observations,
measurements, selected models, prediction outputs, outcomes, features,
metrics, and the Diabetes target.

Local configuration uses environment variables. Do not commit `.env`.

Required environment variables:

```text
NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
NEO4J_DATABASE
```
