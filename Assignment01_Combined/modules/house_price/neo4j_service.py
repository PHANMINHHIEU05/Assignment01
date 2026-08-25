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
