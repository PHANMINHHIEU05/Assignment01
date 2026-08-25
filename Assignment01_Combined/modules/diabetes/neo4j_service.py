import os
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


CONSTRAINT_QUERIES = [
    """
    CREATE CONSTRAINT model_name_unique IF NOT EXISTS
    FOR (m:Model)
    REQUIRE m.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT feature_name_unique IF NOT EXISTS
    FOR (f:Feature)
    REQUIRE f.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT outcome_value_unique IF NOT EXISTS
    FOR (o:Outcome)
    REQUIRE o.value IS UNIQUE
    """,
    """
    CREATE CONSTRAINT observation_id_unique IF NOT EXISTS
    FOR (o:Observation)
    REQUIRE o.observation_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT prediction_id_unique IF NOT EXISTS
    FOR (p:Prediction)
    REQUIRE p.prediction_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT condition_name_unique IF NOT EXISTS
    FOR (c:Condition)
    REQUIRE c.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT representation_name_unique IF NOT EXISTS
    FOR (r:Representation)
    REQUIRE r.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT metric_name_unique IF NOT EXISTS
    FOR (m:Metric)
    REQUIRE m.name IS UNIQUE
    """,
]


METRIC_NAME_MAP = {
    "accuracy_cv": "Accuracy",
    "precision_cv": "Precision",
    "recall_cv": "Recall",
    "f1_cv": "F1-score",
}

FINAL_TEST_METRIC_NAME_MAP = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
}


def get_neo4j_config():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    missing = []
    if not uri:
        missing.append("NEO4J_URI")
    if not user:
        missing.append("NEO4J_USER")
    if not password:
        missing.append("NEO4J_PASSWORD")

    if missing:
        raise RuntimeError(
            "Missing Neo4j environment variables: " + ", ".join(missing)
        )

    return uri, user, password, database


def create_driver():
    uri, user, password, _database = get_neo4j_config()
    return GraphDatabase.driver(
        uri,
        auth=(user, password),
    )


def verify_connection():
    _uri, _user, _password, database = get_neo4j_config()

    with create_driver() as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            result = session.run("RETURN 1 AS ok")
            return result.single()["ok"] == 1


def initialize_graph(selected_features, model_registry):
    _uri, _user, _password, database = get_neo4j_config()
    model_names = list(model_registry["models"].keys())
    final_selected_model = model_registry["final_selected_model"]
    condition_name = "Diabetes"
    representation_name = "Six Selected Features"

    model_rows = [
        {
            "name": name,
            "scientific_final_model": name == final_selected_model,
            "artifact_file": model_registry["models"][name].get("file"),
        }
        for name in model_names
    ]

    outcome_rows = [
        {"value": 0, "label": "No Diabetes"},
        {"value": 1, "label": "Diabetes"},
    ]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    cv_metric_rows = []
    for model_name, model_info in model_registry["models"].items():
        for registry_key, metric_name in METRIC_NAME_MAP.items():
            cv_metric_rows.append(
                {
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "value": float(model_info[registry_key]),
                    "evaluation": "5-fold cross-validation",
                }
            )

    final_test_metric_rows = []
    final_test_metrics = model_registry.get("final_test_metrics", {})
    final_test_model = final_test_metrics.get("model", final_selected_model)
    for registry_key, metric_name in FINAL_TEST_METRIC_NAME_MAP.items():
        if registry_key in final_test_metrics:
            final_test_metric_rows.append(
                {
                    "model_name": final_test_model,
                    "metric_name": metric_name,
                    "value": float(final_test_metrics[registry_key]),
                    "evaluation": "held-out test",
                }
            )

    with create_driver() as driver:
        with driver.session(database=database) as session:
            for query in CONSTRAINT_QUERIES:
                session.run(query).consume()

            session.run(
                """
                UNWIND $models AS model
                MERGE (m:Model {name: model.name})
                SET m.scientific_final_model = model.scientific_final_model,
                    m.artifact_file = model.artifact_file
                """,
                models=model_rows,
            ).consume()

            session.run(
                """
                UNWIND $features AS feature_name
                MERGE (:Feature {name: feature_name})
                """,
                features=selected_features,
            ).consume()

            session.run(
                """
                UNWIND $outcomes AS outcome
                MERGE (o:Outcome {value: outcome.value})
                SET o.label = outcome.label
                """,
                outcomes=outcome_rows,
            ).consume()

            session.run(
                """
                MERGE (:Condition {name: $condition_name})
                """,
                condition_name=condition_name,
            ).consume()

            session.run(
                """
                MERGE (r:Representation {name: $representation_name})
                SET r.feature_count = $feature_count
                """,
                representation_name=representation_name,
                feature_count=len(selected_features),
            ).consume()

            session.run(
                """
                UNWIND $metric_names AS metric_name
                MERGE (:Metric {name: metric_name})
                """,
                metric_names=metric_names,
            ).consume()

            session.run(
                """
                UNWIND $models AS model_name
                MATCH (m:Model {name: model_name})
                WITH m
                UNWIND $features AS feature_name
                MATCH (f:Feature {name: feature_name})
                MERGE (m)-[:USES_FEATURE]->(f)
                """,
                models=model_names,
                features=selected_features,
            ).consume()

            session.run(
                """
                MATCH (r:Representation {name: $representation_name})
                MATCH (c:Condition {name: $condition_name})
                MERGE (r)-[:TARGETS]->(c)
                """,
                representation_name=representation_name,
                condition_name=condition_name,
            ).consume()

            session.run(
                """
                UNWIND $features AS feature_name
                MATCH (f:Feature {name: feature_name})
                MATCH (r:Representation {name: $representation_name})
                MATCH (c:Condition {name: $condition_name})
                MERGE (f)-[:PART_OF_REPRESENTATION]->(r)
                MERGE (f)-[:INPUT_FOR]->(c)
                """,
                features=selected_features,
                representation_name=representation_name,
                condition_name=condition_name,
            ).consume()

            session.run(
                """
                MATCH (c:Condition {name: $condition_name})
                MATCH (o:Outcome)
                MERGE (o)-[:OUTCOME_OF]->(c)
                """,
                condition_name=condition_name,
            ).consume()

            session.run(
                """
                UNWIND $models AS model_name
                MATCH (m:Model {name: model_name})
                MATCH (r:Representation {name: $representation_name})
                MERGE (m)-[:USES_REPRESENTATION]->(r)
                """,
                models=model_names,
                representation_name=representation_name,
            ).consume()

            session.run(
                """
                MATCH (m:Model {name: $final_selected_model})
                MATCH (c:Condition {name: $condition_name})
                MERGE (m)-[:SCIENTIFIC_FINAL_MODEL_FOR]->(c)
                """,
                final_selected_model=final_selected_model,
                condition_name=condition_name,
            ).consume()

            session.run(
                """
                UNWIND $metric_rows AS row
                MATCH (m:Model {name: row.model_name})
                MATCH (metric:Metric {name: row.metric_name})
                MERGE (m)-[rel:HAS_CV_METRIC]->(metric)
                SET rel.value = row.value,
                    rel.evaluation = row.evaluation
                """,
                metric_rows=cv_metric_rows,
            ).consume()

            session.run(
                """
                UNWIND $metric_rows AS row
                MATCH (m:Model {name: row.model_name})
                MATCH (metric:Metric {name: row.metric_name})
                MERGE (m)-[rel:HAS_FINAL_TEST_METRIC]->(metric)
                SET rel.value = row.value,
                    rel.evaluation = row.evaluation
                """,
                metric_rows=final_test_metric_rows,
            ).consume()

            counts = _get_knowledge_graph_summary_in_session(session)

    return counts


def _get_knowledge_graph_summary_in_session(session):
    counts = session.run(
        """
        MATCH (m:Model)
        WITH count(m) AS models
        MATCH (f:Feature)
        WITH models, count(f) AS features
        MATCH (c:Condition)
        WITH models, features, count(c) AS conditions
        MATCH (r:Representation)
        WITH models, features, conditions, count(r) AS representations
        MATCH (o:Outcome)
        WITH models, features, conditions, representations, count(o) AS outcomes
        MATCH (metric:Metric)
        RETURN models,
               features,
               conditions,
               representations,
               outcomes,
               count(metric) AS metrics
        """
    ).single()

    relationships = session.run(
        """
        MATCH (:Model)-[uses_feature:USES_FEATURE]->(:Feature)
        WITH count(uses_feature) AS uses_feature
        MATCH (:Feature)-[input_for:INPUT_FOR]->(:Condition)
        WITH uses_feature, count(input_for) AS input_for
        MATCH (:Feature)-[part_of_representation:PART_OF_REPRESENTATION]->(:Representation)
        WITH uses_feature, input_for, count(part_of_representation) AS part_of_representation
        MATCH (:Representation)-[targets:TARGETS]->(:Condition)
        WITH uses_feature, input_for, part_of_representation, count(targets) AS targets
        MATCH (:Outcome)-[outcome_of:OUTCOME_OF]->(:Condition)
        WITH uses_feature, input_for, part_of_representation, targets, count(outcome_of) AS outcome_of
        MATCH (:Model)-[uses_representation:USES_REPRESENTATION]->(:Representation)
        WITH uses_feature, input_for, part_of_representation, targets, outcome_of,
             count(uses_representation) AS uses_representation
        MATCH (:Model)-[cv_metric:HAS_CV_METRIC]->(:Metric)
        WITH uses_feature, input_for, part_of_representation, targets, outcome_of,
             uses_representation, count(cv_metric) AS cv_metric_relationships
        MATCH (:Model)-[final_test_metric:HAS_FINAL_TEST_METRIC]->(:Metric)
        WITH uses_feature, input_for, part_of_representation, targets, outcome_of,
             uses_representation, cv_metric_relationships,
             count(final_test_metric) AS final_test_metric_relationships
        MATCH (:Model)-[scientific_final_model_for:SCIENTIFIC_FINAL_MODEL_FOR]->(:Condition)
        RETURN uses_feature,
               input_for,
               part_of_representation,
               targets,
               outcome_of,
               uses_representation,
               cv_metric_relationships,
               final_test_metric_relationships,
               count(scientific_final_model_for) AS scientific_final_model_for
        """
    ).single()

    return {
        "models": counts["models"],
        "features": counts["features"],
        "conditions": counts["conditions"],
        "representations": counts["representations"],
        "outcomes": counts["outcomes"],
        "metrics": counts["metrics"],
        "uses_feature": relationships["uses_feature"],
        "input_for": relationships["input_for"],
        "part_of_representation": relationships["part_of_representation"],
        "targets": relationships["targets"],
        "outcome_of": relationships["outcome_of"],
        "uses_representation": relationships["uses_representation"],
        "cv_metric_relationships": relationships["cv_metric_relationships"],
        "final_test_metric_relationships": relationships[
            "final_test_metric_relationships"
        ],
        "scientific_final_model_for": relationships[
            "scientific_final_model_for"
        ],
    }


def get_knowledge_graph_summary():
    _uri, _user, _password, database = get_neo4j_config()

    with create_driver() as driver:
        with driver.session(database=database) as session:
            return _get_knowledge_graph_summary_in_session(session)


def _is_missing(value):
    return pd.isna(value)


def _to_graph_value(value):
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def save_prediction(
    model_name,
    input_values,
    predicted_class,
    probability=None,
    decision_score=None,
):
    _uri, _user, _password, database = get_neo4j_config()
    observation_id = str(uuid4())
    prediction_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    measurements = [
        {
            "feature": feature,
            "value": _to_graph_value(value),
            "missing": bool(_is_missing(value)),
        }
        for feature, value in input_values.items()
    ]

    prediction_props = {
        "prediction_id": prediction_id,
        "predicted_class": int(predicted_class),
        "created_at": created_at,
    }
    if probability is not None:
        prediction_props["probability"] = float(probability)
    if decision_score is not None:
        prediction_props["decision_score"] = float(decision_score)

    with create_driver() as driver:
        with driver.session(database=database) as session:
            session.run(
                """
                MERGE (obs:Observation {observation_id: $observation_id})
                SET obs.created_at = $created_at

                CREATE (pred:Prediction)
                SET pred = $prediction_props

                MERGE (obs)-[:HAS_PREDICTION]->(pred)

                WITH obs, pred
                MATCH (model:Model {name: $model_name})
                MERGE (pred)-[:PRODUCED_BY]->(model)

                WITH obs, pred
                MATCH (outcome:Outcome {value: $predicted_class})
                MERGE (pred)-[:PREDICTED_AS]->(outcome)

                WITH obs
                UNWIND $measurements AS measurement
                MATCH (feature:Feature {name: measurement.feature})
                MERGE (obs)-[rel:HAS_MEASUREMENT]->(feature)
                SET rel.value = measurement.value,
                    rel.missing = measurement.missing
                """,
                observation_id=observation_id,
                created_at=created_at,
                prediction_props=prediction_props,
                model_name=model_name,
                predicted_class=int(predicted_class),
                measurements=measurements,
            ).consume()

    return {
        "observation_id": observation_id,
        "prediction_id": prediction_id,
    }


def get_recent_predictions(limit=10):
    _uri, _user, _password, database = get_neo4j_config()

    with create_driver() as driver:
        with driver.session(database=database) as session:
            records = session.run(
                """
                MATCH (prediction:Prediction)-[:PRODUCED_BY]->(model:Model)
                MATCH (prediction)-[:PREDICTED_AS]->(outcome:Outcome)
                WITH prediction,
                     properties(prediction) AS prediction_props,
                     model,
                     outcome
                RETURN prediction.prediction_id AS prediction_id,
                       prediction.created_at AS created_at,
                       model.name AS model_name,
                       prediction.predicted_class AS predicted_class,
                       outcome.label AS outcome_label,
                       prediction_props.probability AS probability,
                       prediction_props.decision_score AS decision_score
                ORDER BY prediction.created_at DESC
                LIMIT $limit
                """,
                limit=int(limit),
            )

            return [dict(record) for record in records]
