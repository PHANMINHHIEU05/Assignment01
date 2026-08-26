from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


REPRESENTATION_ID = "house_11_feature_v2"
REPRESENTATION_NAME = "Eleven Feature House Representation"
LEGACY_REPRESENTATION_NAME = "Six Selected Features"
HOUSE_FEATURES_11 = [
    "Area",
    "Frontage",
    "Access Road",
    "House direction",
    "Balcony direction",
    "Floors",
    "Bedrooms",
    "Bathrooms",
    "Legal status",
    "Furniture state",
    "Location",
]
NUMERIC_FEATURES = {"Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"}
LEGACY_FEATURES = {"Province", "District"}
MISSING_LABEL = "Not provided"


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


def _clean_value(value):
    if value in (None, "", MISSING_LABEL):
        return None
    return value


def _split_location(location: str | None, metadata: dict | None = None) -> tuple[str | None, str | None]:
    location = _clean_value(location)
    if not location:
        return None, None
    mapping = (metadata or {}).get("location_mapping", {})
    if location in mapping:
        return mapping[location].get("province"), mapping[location].get("district")
    parts = [part.strip() for part in str(location).split(",")]
    if len(parts) >= 2:
        return parts[-1], ", ".join(parts[:-1])
    return None, str(location)


def initialize_graph(registry, metadata):
    _uri, _user, _password, database = get_neo4j_config()
    selected_features = registry.get("selected_features", HOUSE_FEATURES_11)
    model_rows = [
        {
            "name": name,
            "artifact_file": info["file"],
            "scientific_final_model": name == registry["scientific_final_model"],
        }
        for name, info in registry["models"].items()
    ]
    feature_rows = [{"name": feature, "position": index + 1} for index, feature in enumerate(selected_features)]
    metric_names = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
    cv_metric_rows = [
        {"model_name": model_name, "metric_name": metric_name, "value": float(value)}
        for model_name, info in registry["models"].items()
        for metric_name, value in info["cv_metrics"].items()
    ]
    final_metrics = registry["final_test_metrics"]
    final_metric_rows = [
        {"model_name": final_metrics["model"], "metric_name": metric_name, "value": float(final_metrics[metric_name])}
        for metric_name in metric_names
    ]
    district_rows = []
    for location, item in metadata.get("location_mapping", {}).items():
        province = item.get("province")
        district = item.get("district")
        if province and district:
            district_rows.append({"province": province, "district": district, "location": location, "key": f"{province}|{district}"})

    with create_driver() as driver:
        with driver.session(database=database) as session:
            for query in CONSTRAINT_QUERIES:
                session.run(query).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (m:HouseModel {name: row.name})
                SET m.artifact_file = row.artifact_file,
                    m.scientific_final_model = row.scientific_final_model,
                    m.representation_version = $representation_id
                """,
                rows=model_rows,
                representation_id=REPRESENTATION_ID,
            ).consume()
            session.run(
                """
                MATCH (old:HouseRepresentation {name: $legacy_name})
                SET old.active = false,
                    old.representation_version = 'legacy_6_feature'
                """,
                legacy_name=LEGACY_REPRESENTATION_NAME,
            ).consume()
            session.run(
                """
                MERGE (r:HouseRepresentation {name: $name})
                SET r.id = $id,
                    r.feature_count = $feature_count,
                    r.active = true,
                    r.representation_version = $id
                MERGE (t:HouseTarget {name: 'House Price'})
                SET t.unit = 'billion VND'
                MERGE (r)-[:HOUSE_TARGETS]->(t)
                """,
                id=REPRESENTATION_ID,
                name=REPRESENTATION_NAME,
                feature_count=len(selected_features),
            ).consume()
            session.run(
                """
                UNWIND $legacy_features AS name
                MERGE (f:HouseFeature {name: name})
                SET f.active = false,
                    f.representation_version = 'legacy_6_feature'
                """,
                legacy_features=sorted(LEGACY_FEATURES),
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (f:HouseFeature {name: row.name})
                SET f.active = true,
                    f.position = row.position,
                    f.representation_version = $representation_id
                WITH f
                MATCH (r:HouseRepresentation {name: $representation_name})
                MERGE (f)-[:HOUSE_PART_OF_REPRESENTATION]->(r)
                WITH f
                MATCH (m:HouseModel)
                MERGE (m)-[:HOUSE_USES_FEATURE]->(f)
                """,
                rows=feature_rows,
                representation_id=REPRESENTATION_ID,
                representation_name=REPRESENTATION_NAME,
            ).consume()
            session.run("UNWIND $names AS name MERGE (:HouseMetric {name: name})", names=metric_names).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (p:HouseProvince {name: row.province})
                MERGE (d:HouseDistrict {key: row.key})
                SET d.name = row.district,
                    d.location = row.location
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
                    rel.evaluation = '5-fold cross-validation',
                    rel.representation_version = $representation_id
                """,
                rows=cv_metric_rows,
                representation_id=REPRESENTATION_ID,
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
                    rel.evaluation = 'held-out test',
                    rel.representation_version = $representation_id
                """,
                rows=final_metric_rows,
                representation_id=REPRESENTATION_ID,
            ).consume()


def save_prediction(model_name, input_values, predicted_price_billion, metadata=None):
    _uri, _user, _password, database = get_neo4j_config()
    prediction_id = str(uuid4())
    observation_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    location = _clean_value(input_values.get("Location"))
    province, district = _split_location(location, metadata)
    measurements = []
    for feature in HOUSE_FEATURES_11:
        value = _clean_value(input_values.get(feature))
        if feature in NUMERIC_FEATURES and value is not None:
            value = float(value)
        measurements.append({"feature": feature, "value": value})

    with create_driver() as driver:
        with driver.session(database=database) as session:
            session.run(
                """
                MERGE (target:HouseTarget {name: 'House Price'})
                MATCH (model:HouseModel {name: $model_name})
                CREATE (obs:HouseObservation {
                    observation_id: $observation_id,
                    created_at: $created_at,
                    location: $location,
                    province: $province,
                    district: $district,
                    representation_version: $representation_id
                })
                CREATE (pred:HousePrediction {
                    prediction_id: $prediction_id,
                    predicted_price_billion: $predicted_price_billion,
                    created_at: $created_at,
                    representation_version: $representation_id
                })
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
                observation_id=observation_id,
                prediction_id=prediction_id,
                predicted_price_billion=float(predicted_price_billion),
                created_at=created_at,
                location=location,
                province=province,
                district=district,
                measurements=measurements,
                representation_id=REPRESENTATION_ID,
            ).consume()
            if province and district:
                session.run(
                    """
                    MATCH (obs:HouseObservation {observation_id: $observation_id})
                    MERGE (province:HouseProvince {name: $province})
                    MERGE (district:HouseDistrict {key: $district_key})
                    SET district.name = $district,
                        district.location = $location
                    MERGE (district)-[:HOUSE_IN_PROVINCE]->(province)
                    MERGE (obs)-[:HOUSE_LOCATED_IN]->(district)
                    """,
                    observation_id=observation_id,
                    province=province,
                    district=district,
                    district_key=f"{province}|{district}",
                    location=location,
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
                       obs.location AS location,
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
            active = session.run("MATCH (f:HouseFeature {active: true}) RETURN count(f) AS count").single()["count"]
            summary["ActiveHouseFeature"] = int(active)
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
    {"id": "area_factor", "name": "Area", "description": "Property area in square meters.", "feature": "Area"},
    {"id": "frontage_factor", "name": "Frontage", "description": "Street-facing width of the property.", "feature": "Frontage"},
    {"id": "access_road_factor", "name": "Access Road", "description": "Road width/accessibility near the property.", "feature": "Access Road"},
    {"id": "house_direction_factor", "name": "House direction", "description": "Direction/orientation of the house.", "feature": "House direction"},
    {"id": "balcony_direction_factor", "name": "Balcony direction", "description": "Direction/orientation of the balcony.", "feature": "Balcony direction"},
    {"id": "floors_factor", "name": "Floors", "description": "Number of floors.", "feature": "Floors"},
    {"id": "bedrooms_factor", "name": "Bedrooms", "description": "Number of bedrooms.", "feature": "Bedrooms"},
    {"id": "bathrooms_factor", "name": "Bathrooms", "description": "Number of bathrooms.", "feature": "Bathrooms"},
    {"id": "legal_status_factor", "name": "Legal status", "description": "Document/legal availability in listing data.", "feature": "Legal status"},
    {"id": "furniture_state_factor", "name": "Furniture state", "description": "Furniture condition or furnishing status.", "feature": "Furniture state"},
    {"id": "location_factor", "name": "Location", "description": "Combined District, Province model feature parsed from Address.", "feature": "Location"},
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
                MATCH (vf:HouseValuationFactor)
                SET vf.active = false
                """,
            ).consume()
            session.run(
                """
                UNWIND $factors AS factor
                MERGE (vf:HouseValuationFactor {id: factor.id})
                SET vf.name = factor.name,
                    vf.description = factor.description,
                    vf.feature = factor.feature,
                    vf.representation_version = $representation_id,
                    vf.active = true
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
                representation_id=REPRESENTATION_ID,
            ).consume()
    return get_knowledge_graph_summary()


def get_house_graph_summary():
    summary = get_knowledge_graph_summary()
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        with driver.session(database=database) as session:
            extra = session.run(
                """
                MATCH (f:HouseValuationFactor {active: true})
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
    nodes.append({"id": "house_representation", "label": "Eleven Feature Representation", "group": "representation", "title": "Active 11-feature deployment input"})
    edges.append({"source": "house_representation", "target": "house_target", "label": "TARGETS"})
    for model in models:
        node_id = f"house_model_{model}"
        nodes.append({"id": node_id, "label": model, "group": "model", "title": f"Regressor: {model}"})
        edges.append({"source": node_id, "target": "house_target", "label": "PREDICTS"})
    for feature in HOUSE_FEATURES_11:
        node_id = f"house_feature_{feature}"
        nodes.append({"id": node_id, "label": feature, "group": "feature", "title": "Active model input feature"})
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
                OPTIONAL MATCH (obs)-[measurement:HOUSE_HAS_MEASUREMENT]->(feature:HouseFeature)
                RETURN obs.observation_id AS observation_id,
                       pred.prediction_id AS prediction_id,
                       pred.predicted_price_billion AS predicted_price_billion,
                       pred.created_at AS created_at,
                       model.name AS model_name,
                       target.name AS target_name,
                       coalesce(obs.location, district.location) AS location,
                       coalesce(obs.district, district.name) AS district,
                       coalesce(obs.province, province.name) AS province,
                       collect({feature: feature.name, value: measurement.value}) AS measurements
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
        nodes.append({"id": "house_latest_district", "label": latest["district"], "group": "location", "title": "District context parsed from Location"})
        edges.append({"source": "house_latest_observation", "target": "house_latest_district", "label": "LOCATED_IN"})
    if latest.get("province"):
        nodes.append({"id": "house_latest_province", "label": latest["province"], "group": "location", "title": "Province context parsed from Location"})
        if latest.get("district"):
            edges.append({"source": "house_latest_district", "target": "house_latest_province", "label": "IN_PROVINCE"})
    measurement_lookup = {
        item.get("feature"): item.get("value")
        for item in latest.get("measurements", [])
        if item.get("feature")
    }
    for feature in HOUSE_FEATURES_11:
        value = measurement_lookup.get(feature)
        title = f"Model input: {feature}" + (f" = {value}" if value is not None else "")
        node_id = f"latest_feature_{feature}"
        nodes.append({"id": node_id, "label": feature, "group": "feature", "title": title})
        edges.append({"source": "house_latest_observation", "target": node_id, "label": "HAS_MEASUREMENT"})
        edges.append({"source": "house_latest_model", "target": node_id, "label": "USES_FEATURE"})
    return {"nodes": nodes, "edges": edges, "latest": latest}
