from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT house_model_name_unique IF NOT EXISTS FOR (m:HouseModel) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT house_feature_name_unique IF NOT EXISTS FOR (f:HouseFeature) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT house_representation_name_unique IF NOT EXISTS FOR (r:HouseRepresentation) REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT house_target_name_unique IF NOT EXISTS FOR (t:HouseTarget) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT house_metric_name_unique IF NOT EXISTS FOR (m:HouseMetric) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT house_province_name_unique IF NOT EXISTS FOR (p:HouseProvince) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT house_district_key_unique IF NOT EXISTS FOR (d:HouseDistrict) REQUIRE d.key IS UNIQUE",
    "CREATE CONSTRAINT house_observation_id_unique IF NOT EXISTS FOR (o:HouseObservation) REQUIRE o.observation_id IS UNIQUE",
    "CREATE CONSTRAINT house_prediction_id_unique IF NOT EXISTS FOR (p:HousePrediction) REQUIRE p.prediction_id IS UNIQUE",
]


def get_neo4j_config():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    missing = [name for name, value in {
        "NEO4J_URI": uri,
        "NEO4J_USER": user,
        "NEO4J_PASSWORD": password,
    }.items() if not value]
    if missing:
        raise RuntimeError("Missing Neo4j environment variables: " + ", ".join(missing))
    return uri, user, password, database


def create_driver():
    uri, user, password, _database = get_neo4j_config()
    return GraphDatabase.driver(uri, auth=(user, password))


def verify_connection():
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            return session.run("RETURN 1 AS ok").single()["ok"] == 1


def initialize_graph(registry, metadata):
    _uri, _user, _password, database = get_neo4j_config()
    model_rows = []
    for name, info in registry["models"].items():
        model_rows.append({
            "name": name,
            "artifact_file": info["file"],
            "scientific_final_model": name == registry["scientific_final_model"],
        })
    feature_rows = [{"name": feature} for feature in registry["selected_features"]]
    metric_names = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
    cv_metric_rows = []
    for model_name, info in registry["models"].items():
        for metric_name, value in info["cv_metrics"].items():
            cv_metric_rows.append({"model_name": model_name, "metric_name": metric_name, "value": float(value)})
    final_metric_rows = []
    final_metrics = registry["final_test_metrics"]
    for metric_name in metric_names:
        final_metric_rows.append({
            "model_name": final_metrics["model"],
            "metric_name": metric_name,
            "value": float(final_metrics[metric_name]),
        })
    district_rows = []
    for province, districts in metadata["province_districts"].items():
        for district in districts:
            district_rows.append({"province": province, "district": district, "key": f"{province}|{district}"})

    with create_driver() as driver:
        with driver.session(database=database) as session:
            for query in CONSTRAINT_QUERIES:
                session.run(query).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (m:HouseModel {name: row.name})
                SET m.artifact_file = row.artifact_file,
                    m.scientific_final_model = row.scientific_final_model
                """,
                rows=model_rows,
            ).consume()
            session.run("UNWIND $rows AS row MERGE (:HouseFeature {name: row.name})", rows=feature_rows).consume()
            session.run(
                """
                MERGE (r:HouseRepresentation {name: 'Six Selected Features'})
                SET r.feature_count = 6
                MERGE (t:HouseTarget {name: 'House Price'})
                SET t.unit = 'billion VND'
                MERGE (r)-[:HOUSE_TARGETS]->(t)
                """,
            ).consume()
            session.run("UNWIND $names AS name MERGE (:HouseMetric {name: name})", names=metric_names).consume()
            session.run(
                """
                UNWIND $features AS feature
                MATCH (f:HouseFeature {name: feature})
                MATCH (r:HouseRepresentation {name: 'Six Selected Features'})
                MERGE (f)-[:HOUSE_PART_OF_REPRESENTATION]->(r)
                WITH f
                MATCH (m:HouseModel)
                MERGE (m)-[:HOUSE_USES_FEATURE]->(f)
                """,
                features=registry["selected_features"],
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (p:HouseProvince {name: row.province})
                MERGE (d:HouseDistrict {key: row.key})
                SET d.name = row.district
                MERGE (d)-[:HOUSE_IN_PROVINCE]->(p)
                """,
                rows=district_rows,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (m:HouseModel {name: row.model_name})
                MATCH (metric:HouseMetric {name: row.metric_name})
                MERGE (m)-[rel:HOUSE_HAS_CV_METRIC]->(metric)
                SET rel.value = row.value,
                    rel.evaluation = '5-fold cross-validation'
                """,
                rows=cv_metric_rows,
            ).consume()
            session.run(
                """
                MATCH (m:HouseModel {name: $model_name})
                MATCH (t:HouseTarget {name: 'House Price'})
                MERGE (m)-[:HOUSE_SCIENTIFIC_FINAL_MODEL_FOR]->(t)
                """,
                model_name=registry["scientific_final_model"],
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (m:HouseModel {name: row.model_name})
                MATCH (metric:HouseMetric {name: row.metric_name})
                MERGE (m)-[rel:HOUSE_HAS_FINAL_TEST_METRIC]->(metric)
                SET rel.value = row.value,
                    rel.evaluation = 'held-out test'
                """,
                rows=final_metric_rows,
            ).consume()


def save_prediction(model_name, input_values, predicted_price_billion):
    _uri, _user, _password, database = get_neo4j_config()
    prediction_id = str(uuid4())
    observation_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    province = input_values.get("Province")
    district = input_values.get("District")
    measurements = [
        {"feature": key, "value": None if value is None else float(value)}
        for key, value in input_values.items()
        if key in {"Area", "Floors", "Bedrooms", "Bathrooms"}
    ]
    with create_driver() as driver:
        with driver.session(database=database) as session:
            session.run(
                """
                MERGE (target:HouseTarget {name: 'House Price'})
                MATCH (model:HouseModel {name: $model_name})
                MERGE (province:HouseProvince {name: $province})
                MERGE (district:HouseDistrict {key: $district_key})
                SET district.name = $district
                MERGE (district)-[:HOUSE_IN_PROVINCE]->(province)
                CREATE (obs:HouseObservation {
                    observation_id: $observation_id,
                    created_at: $created_at,
                    province: $province,
                    district: $district
                })
                CREATE (pred:HousePrediction {
                    prediction_id: $prediction_id,
                    predicted_price_billion: $predicted_price_billion,
                    created_at: $created_at
                })
                MERGE (obs)-[:HOUSE_LOCATED_IN]->(district)
                MERGE (obs)-[:HOUSE_HAS_PREDICTION]->(pred)
                MERGE (pred)-[:HOUSE_PRODUCED_BY]->(model)
                MERGE (pred)-[:HOUSE_PREDICTS]->(target)
                WITH obs
                UNWIND $measurements AS measurement
                MATCH (feature:HouseFeature {name: measurement.feature})
                MERGE (obs)-[rel:HOUSE_HAS_MEASUREMENT]->(feature)
                SET rel.value = measurement.value
                """,
                model_name=model_name,
                province=province,
                district=district,
                district_key=f"{province}|{district}",
                observation_id=observation_id,
                prediction_id=prediction_id,
                predicted_price_billion=float(predicted_price_billion),
                created_at=created_at,
                measurements=measurements,
            ).consume()
    return prediction_id


def get_recent_predictions(limit=5):
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (obs:HouseObservation)-[:HOUSE_HAS_PREDICTION]->(pred:HousePrediction)
                MATCH (pred)-[:HOUSE_PRODUCED_BY]->(model:HouseModel)
                RETURN pred.prediction_id AS prediction_id,
                       pred.predicted_price_billion AS predicted_price_billion,
                       pred.created_at AS created_at,
                       model.name AS model,
                       obs.province AS province,
                       obs.district AS district
                ORDER BY pred.created_at DESC
                LIMIT $limit
                """,
                limit=int(limit),
            )
            return [dict(record) for record in result]


def get_knowledge_graph_summary():
    _uri, _user, _password, database = get_neo4j_config()
    labels = [
        "HouseModel", "HouseFeature", "HouseRepresentation", "HouseTarget",
        "HouseMetric", "HouseProvince", "HouseDistrict", "HouseObservation",
        "HousePrediction",
    ]
    with create_driver() as driver:
        with driver.session(database=database) as session:
            summary = {}
            for label in labels:
                count = session.run(f"MATCH (n:{label}) RETURN count(n) AS count").single()["count"]
                summary[label] = int(count)
            return summary


HOUSE_DOMAIN_CONSTRAINTS = [
    "CREATE CONSTRAINT house_valuation_factor_id_unique IF NOT EXISTS FOR (f:HouseValuationFactor) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT house_knowledge_source_id_unique IF NOT EXISTS FOR (s:HouseKnowledgeSource) REQUIRE s.id IS UNIQUE",
]


HOUSE_SOURCES = [
    {
        "id": "kaggle_vietnam_housing_2024",
        "name": "Kaggle",
        "title": "House Price Prediction Dataset Vietnam - 2024",
        "url": "https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024",
        "retrieved_at": "2026-08-26",
        "source_type": "dataset_source",
    }
]


HOUSE_FACTORS = [
    {"id": "area_factor", "name": "Area", "description": "Model input representing property area.", "feature": "Area"},
    {"id": "floors_factor", "name": "Floors", "description": "Model input representing structural floor count.", "feature": "Floors"},
    {"id": "bedrooms_factor", "name": "Bedrooms", "description": "Model input representing bedroom count.", "feature": "Bedrooms"},
    {"id": "bathrooms_factor", "name": "Bathrooms", "description": "Model input representing bathroom count.", "feature": "Bathrooms"},
    {"id": "province_factor", "name": "Province", "description": "Model input representing broad location.", "feature": "Province"},
    {"id": "district_factor", "name": "District", "description": "Model input representing district-level location.", "feature": "District"},
]


def initialize_house_domain_graph():
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        with driver.session(database=database) as session:
            for query in HOUSE_DOMAIN_CONSTRAINTS:
                session.run(query).consume()
            session.run(
                """
                UNWIND $sources AS source
                MERGE (s:HouseKnowledgeSource {id: source.id})
                SET s.name = source.name,
                    s.title = source.title,
                    s.url = source.url,
                    s.retrieved_at = source.retrieved_at,
                    s.source_type = source.source_type
                """,
                sources=HOUSE_SOURCES,
            ).consume()
            session.run(
                """
                UNWIND $factors AS factor
                MERGE (vf:HouseValuationFactor {id: factor.id})
                SET vf.name = factor.name,
                    vf.description = factor.description,
                    vf.feature = factor.feature
                WITH vf, factor
                MATCH (target:HouseTarget {name: 'House Price'})
                MERGE (target)-[:HOUSE_RELATED_TO_FACTOR]->(vf)
                WITH vf, factor
                MATCH (feature:HouseFeature {name: factor.feature})
                MERGE (feature)-[:HOUSE_REPRESENTS_FACTOR]->(vf)
                WITH vf
                MATCH (source:HouseKnowledgeSource {id: 'kaggle_vietnam_housing_2024'})
                MERGE (vf)-[:HOUSE_SUPPORTED_BY_SOURCE]->(source)
                """,
                factors=HOUSE_FACTORS,
            ).consume()
    return get_knowledge_graph_summary()


def get_house_graph_summary():
    summary = get_knowledge_graph_summary()
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        with driver.session(database=database) as session:
            extra = session.run(
                """
                MATCH (f:HouseValuationFactor)
                WITH count(f) AS factors
                MATCH (s:HouseKnowledgeSource)
                RETURN factors, count(s) AS sources
                """
            ).single()
    summary["HouseValuationFactor"] = int(extra["factors"]) if extra else 0
    summary["HouseKnowledgeSource"] = int(extra["sources"]) if extra else 0
    return summary


def get_house_system_graph_data():
    nodes = [{"id": "house_target", "label": "House Price", "group": "target", "title": "Regression target"}]
    edges = []
    models = ["Linear Regression", "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting"]
    features = ["Area", "Floors", "Bedrooms", "Bathrooms", "Province", "District"]
    nodes.append({"id": "house_representation", "label": "Six Feature Representation", "group": "representation", "title": "Deployment input representation"})
    edges.append({"source": "house_representation", "target": "house_target", "label": "TARGETS"})
    for model in models:
        node_id = f"house_model_{model}"
        nodes.append({"id": node_id, "label": model, "group": "model", "title": f"Regressor: {model}"})
        edges.append({"source": node_id, "target": "house_target", "label": "PREDICTS"})
    for feature in features:
        node_id = f"house_feature_{feature}"
        nodes.append({"id": node_id, "label": feature, "group": "feature", "title": "Model input feature"})
        edges.append({"source": "house_representation", "target": node_id, "label": "INCLUDES"})
        for model in models:
            edges.append({"source": f"house_model_{model}", "target": node_id, "label": "USES_FEATURE"})
    for factor in HOUSE_FACTORS:
        node_id = f"house_factor_{factor['id']}"
        nodes.append({"id": node_id, "label": factor["name"], "group": "factor", "title": factor["description"]})
        edges.append({"source": "house_target", "target": node_id, "label": "RELATED_TO_FACTOR"})
    nodes.append({"id": "house_source_kaggle", "label": "Kaggle Dataset", "group": "source", "title": HOUSE_SOURCES[0]["url"]})
    edges.append({"source": "house_target", "target": "house_source_kaggle", "label": "SUPPORTED_BY_SOURCE"})
    return {"nodes": nodes, "edges": edges}


def get_latest_house_prediction_graph_data():
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        with driver.session(database=database) as session:
            record = session.run(
                """
                MATCH (obs:HouseObservation)-[:HOUSE_HAS_PREDICTION]->(pred:HousePrediction)
                MATCH (pred)-[:HOUSE_PRODUCED_BY]->(model:HouseModel)
                MATCH (pred)-[:HOUSE_PREDICTS]->(target:HouseTarget)
                OPTIONAL MATCH (obs)-[:HOUSE_LOCATED_IN]->(district:HouseDistrict)-[:HOUSE_IN_PROVINCE]->(province:HouseProvince)
                RETURN obs.observation_id AS observation_id,
                       pred.prediction_id AS prediction_id,
                       pred.predicted_price_billion AS predicted_price_billion,
                       pred.created_at AS created_at,
                       model.name AS model_name,
                       target.name AS target_name,
                       district.name AS district,
                       province.name AS province
                ORDER BY pred.created_at DESC
                LIMIT 1
                """
            ).single()
    nodes = []
    edges = []
    if not record:
        return {"nodes": nodes, "edges": edges, "latest": None}
    latest = dict(record)
    nodes.extend([
        {"id": "house_latest_observation", "label": "HouseObservation", "group": "observation", "title": "Latest anonymous house input"},
        {"id": "house_latest_prediction", "label": f"{latest['predicted_price_billion']:.2f} B VND", "group": "prediction", "title": f"Created at {latest.get('created_at')}"},
        {"id": "house_latest_model", "label": latest["model_name"], "group": "model", "title": "Producing regressor"},
        {"id": "house_latest_target", "label": latest["target_name"], "group": "target", "title": "Predicted target"},
    ])
    edges.extend([
        {"source": "house_latest_observation", "target": "house_latest_prediction", "label": "HAS_PREDICTION"},
        {"source": "house_latest_prediction", "target": "house_latest_model", "label": "PRODUCED_BY"},
        {"source": "house_latest_prediction", "target": "house_latest_target", "label": "PREDICTS"},
    ])
    if latest.get("district"):
        nodes.append({"id": "house_latest_district", "label": latest["district"], "group": "location", "title": "District"})
        edges.append({"source": "house_latest_observation", "target": "house_latest_district", "label": "LOCATED_IN"})
    if latest.get("province"):
        nodes.append({"id": "house_latest_province", "label": latest["province"], "group": "location", "title": "Province"})
        if latest.get("district"):
            edges.append({"source": "house_latest_district", "target": "house_latest_province", "label": "IN_PROVINCE"})
    for factor in HOUSE_FACTORS[:6]:
        node_id = f"latest_factor_{factor['id']}"
        nodes.append({"id": node_id, "label": factor["name"], "group": "feature", "title": factor["description"]})
        edges.append({"source": "house_latest_model", "target": node_id, "label": "USES_FEATURE"})
    return {"nodes": nodes, "edges": edges, "latest": latest}
