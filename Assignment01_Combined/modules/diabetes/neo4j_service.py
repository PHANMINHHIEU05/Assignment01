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
        WITH models,
             features,
             conditions,
             representations,
             outcomes,
             count(metric) AS metrics
        OPTIONAL MATCH (obs:Observation)
        WITH models, features, conditions, representations, outcomes, metrics, count(obs) AS observations
        OPTIONAL MATCH (prediction:Prediction)
        RETURN models,
               features,
               conditions,
               representations,
               outcomes,
               metrics,
               observations,
               count(prediction) AS predictions
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
        "observations": counts["observations"],
        "predictions": counts["predictions"],
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
            summary = _get_knowledge_graph_summary_in_session(session)
            for label in [
                "Disease",
                "RiskFactor",
                "ClinicalMetric",
                "GeneralGuidance",
                "Complication",
                "MedicalSpecialty",
                "KnowledgeSource",
            ]:
                summary[label] = int(
                    session.run(f"MATCH (n:{label}) RETURN count(n) AS count").single()["count"]
                )
            return summary


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


DIABETES_DOMAIN_CONSTRAINTS = [
    "CREATE CONSTRAINT disease_id_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT risk_factor_id_unique IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT clinical_metric_id_unique IF NOT EXISTS FOR (m:ClinicalMetric) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT general_guidance_id_unique IF NOT EXISTS FOR (g:GeneralGuidance) REQUIRE g.id IS UNIQUE",
    "CREATE CONSTRAINT complication_id_unique IF NOT EXISTS FOR (c:Complication) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT medical_specialty_id_unique IF NOT EXISTS FOR (s:MedicalSpecialty) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT knowledge_source_id_unique IF NOT EXISTS FOR (s:KnowledgeSource) REQUIRE s.id IS UNIQUE",
]


DIABETES_SOURCES = [
    {
        "id": "cdc_diabetes_risk_factors",
        "name": "CDC",
        "title": "Diabetes Risk Factors",
        "url": "https://www.cdc.gov/diabetes/risk-factors/",
        "retrieved_at": "2026-08-26",
        "source_type": "official_public_health",
    },
    {
        "id": "cdc_diabetes_complications",
        "name": "CDC",
        "title": "Diabetes Complications",
        "url": "https://www.cdc.gov/diabetes/complications/index.html",
        "retrieved_at": "2026-08-26",
        "source_type": "official_public_health",
    },
    {
        "id": "who_diabetes_fact_sheet",
        "name": "WHO",
        "title": "Diabetes fact sheet",
        "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "retrieved_at": "2026-08-26",
        "source_type": "official_public_health",
    },
    {
        "id": "long_chau_diabetes_lifestyle",
        "name": "Long Chau",
        "title": "Nhung thoi xau tiep tay cho benh tieu duong",
        "url": "https://nhathuoclongchau.com.vn/bai-viet/nhung-thoi-xau-tiep-tay-cho-benh-tieu-duong.html",
        "retrieved_at": "2026-08-26",
        "source_type": "medical_education",
    },
]


DIABETES_DOMAIN_NODES = {
    "risk_factors": [
        {"id": "elevated_glucose", "name": "Elevated glucose", "description": "Higher blood glucose is central to diabetes screening and monitoring.", "source_id": "who_diabetes_fact_sheet"},
        {"id": "higher_bmi", "name": "Higher BMI / overweight", "description": "Overweight and obesity are associated with higher type 2 diabetes risk.", "source_id": "cdc_diabetes_risk_factors"},
        {"id": "increasing_age", "name": "Increasing age", "description": "Age 45 or older is listed as a risk factor for prediabetes and type 2 diabetes.", "source_id": "cdc_diabetes_risk_factors"},
        {"id": "family_history", "name": "Family/genetic risk", "description": "Having a parent or sibling with type 2 diabetes is a risk factor.", "source_id": "cdc_diabetes_risk_factors"},
        {"id": "physical_inactivity", "name": "Physical inactivity", "description": "Low physical activity is associated with higher risk; activity can reduce risk.", "source_id": "cdc_diabetes_risk_factors"},
    ],
    "clinical_metrics": [
        {"id": "glucose_metric", "name": "Glucose", "description": "Blood glucose testing is used for diagnosis and monitoring context.", "source_id": "who_diabetes_fact_sheet"},
        {"id": "bmi_metric", "name": "BMI", "description": "Body weight context is relevant to type 2 diabetes risk.", "source_id": "cdc_diabetes_risk_factors"},
        {"id": "blood_pressure_metric", "name": "Blood Pressure", "description": "Blood pressure management is relevant to diabetes complication risk.", "source_id": "cdc_diabetes_complications"},
    ],
    "guidance": [
        {"id": "regular_activity", "name": "Regular physical activity", "description": "General prevention guidance includes regular physical activity.", "source_id": "who_diabetes_fact_sheet"},
        {"id": "balanced_diet", "name": "Balanced diet", "description": "General guidance includes healthy eating and limiting sugar/saturated fat.", "source_id": "who_diabetes_fact_sheet"},
        {"id": "weight_management", "name": "Weight management", "description": "Maintaining healthy body weight can help prevent or delay type 2 diabetes.", "source_id": "who_diabetes_fact_sheet"},
        {"id": "clinical_screening", "name": "Clinical screening when appropriate", "description": "Early detection is supported by regular check-ups and blood tests with a healthcare provider.", "source_id": "who_diabetes_fact_sheet"},
    ],
    "complications": [
        {"id": "cardiovascular_complications", "name": "Cardiovascular complications", "description": "Diabetes is associated with heart attack and stroke risk.", "source_id": "cdc_diabetes_complications"},
        {"id": "kidney_complications", "name": "Kidney complications", "description": "High blood sugar can damage kidneys and is linked to chronic kidney disease.", "source_id": "cdc_diabetes_complications"},
        {"id": "eye_complications", "name": "Eye complications", "description": "Diabetes can damage retinal blood vessels and affect vision.", "source_id": "cdc_diabetes_complications"},
        {"id": "nerve_complications", "name": "Nerve complications", "description": "Nerve damage is a common diabetes complication.", "source_id": "cdc_diabetes_complications"},
    ],
    "specialties": [
        {"id": "endocrinology", "name": "Endocrinology", "description": "Medical specialty related to hormone and metabolic disorders.", "source_id": "who_diabetes_fact_sheet"},
        {"id": "nutrition", "name": "Nutrition", "description": "Nutrition guidance is relevant to general diabetes education.", "source_id": "long_chau_diabetes_lifestyle"},
    ],
}


def initialize_diabetes_domain_graph():
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        with driver.session(database=database) as session:
            for query in DIABETES_DOMAIN_CONSTRAINTS:
                session.run(query).consume()
            session.run(
                """
                MERGE (d:Disease {id: 'diabetes_type_2'})
                SET d.name = 'Type 2 Diabetes',
                    d.description = 'Educational domain concept connected to the diabetes prediction outcome.'
                """,
            ).consume()
            session.run(
                """
                UNWIND $sources AS source
                MERGE (s:KnowledgeSource {id: source.id})
                SET s.name = source.name,
                    s.title = source.title,
                    s.url = source.url,
                    s.retrieved_at = source.retrieved_at,
                    s.source_type = source.source_type
                """,
                sources=DIABETES_SOURCES,
            ).consume()
            _merge_domain_group(session, "RiskFactor", "HAS_RISK_FACTOR", DIABETES_DOMAIN_NODES["risk_factors"])
            _merge_domain_group(session, "ClinicalMetric", "HAS_CLINICAL_METRIC", DIABETES_DOMAIN_NODES["clinical_metrics"])
            _merge_domain_group(session, "GeneralGuidance", "HAS_GENERAL_GUIDANCE", DIABETES_DOMAIN_NODES["guidance"])
            _merge_domain_group(session, "Complication", "MAY_BE_ASSOCIATED_WITH", DIABETES_DOMAIN_NODES["complications"])
            _merge_domain_group(session, "MedicalSpecialty", "RELEVANT_SPECIALTY", DIABETES_DOMAIN_NODES["specialties"])
            session.run(
                """
                MATCH (o:Outcome {label: 'Diabetes'})
                MATCH (d:Disease {id: 'diabetes_type_2'})
                MERGE (o)-[:RELATED_TO_DISEASE]->(d)
                """,
            ).consume()
            session.run(
                """
                MATCH (d:Disease {id: 'diabetes_type_2'})
                MATCH (s:KnowledgeSource)
                WHERE s.id IN $source_ids
                MERGE (d)-[:SUPPORTED_BY_SOURCE]->(s)
                """,
                source_ids=[source["id"] for source in DIABETES_SOURCES],
            ).consume()
    return get_knowledge_graph_summary()


def _merge_domain_group(session, label, relationship, rows):
    query = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{id: row.id}})
    SET n.name = row.name,
        n.description = row.description,
        n.source_id = row.source_id
    WITH n, row
    MATCH (d:Disease {{id: 'diabetes_type_2'}})
    MERGE (d)-[:{relationship}]->(n)
    WITH n, row
    MATCH (s:KnowledgeSource {{id: row.source_id}})
    MERGE (n)-[:SUPPORTED_BY_SOURCE]->(s)
    """
    session.run(query, rows=rows).consume()


def get_diabetes_domain_context():
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        with driver.session(database=database) as session:
            disease = session.run(
                "MATCH (d:Disease {id: 'diabetes_type_2'}) RETURN d.name AS name, d.description AS description"
            ).single()
            context = {
                "disease": dict(disease) if disease else {},
                "risk_factors": _context_rows(session, "RiskFactor"),
                "clinical_metrics": _context_rows(session, "ClinicalMetric"),
                "general_guidance": _context_rows(session, "GeneralGuidance"),
                "complications": _context_rows(session, "Complication"),
                "specialties": _context_rows(session, "MedicalSpecialty"),
                "sources": _source_rows(session, "KnowledgeSource"),
            }
            return context


def _context_rows(session, label):
    result = session.run(
        f"""
        MATCH (:Disease {{id: 'diabetes_type_2'}})-->(n:{label})
        OPTIONAL MATCH (n)-[:SUPPORTED_BY_SOURCE]->(s:KnowledgeSource)
        RETURN n.name AS name, n.description AS description, s.name AS source_name, s.url AS source_url
        ORDER BY n.name
        LIMIT 12
        """
    )
    return [dict(record) for record in result]


def _source_rows(session, label):
    result = session.run(
        f"""
        MATCH (s:{label})
        RETURN s.name AS name, s.title AS title, s.url AS url, s.source_type AS source_type
        ORDER BY s.name, s.title
        LIMIT 10
        """
    )
    return [dict(record) for record in result]


def get_system_graph_data():
    summary = get_knowledge_graph_summary()
    nodes = [{"id": "condition_diabetes", "label": "Diabetes", "group": "target", "title": "Prediction condition"}]
    edges = []
    for model in ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]:
        node_id = f"model_{model}"
        nodes.append({"id": node_id, "label": model, "group": "model", "title": f"Classifier: {model}"})
        edges.append({"source": node_id, "target": "condition_diabetes", "label": "PREDICTS"})
    for feature in ["Glucose", "BMI", "DiabetesPedigreeFunction", "Age", "Insulin", "BloodPressure"]:
        node_id = f"feature_{feature}"
        nodes.append({"id": node_id, "label": feature, "group": "feature", "title": "Model input feature"})
        for model in ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]:
            edges.append({"source": f"model_{model}", "target": node_id, "label": "USES_FEATURE"})
    nodes.append({"id": "representation_six", "label": "Six Feature Representation", "group": "representation", "title": "Deployment representation"})
    edges.append({"source": "representation_six", "target": "condition_diabetes", "label": "TARGETS"})
    nodes.append({"id": "metric_summary", "label": f"{summary.get('metrics', 4)} Metrics", "group": "metric", "title": "CV metrics stored in Neo4j"})
    edges.append({"source": "metric_summary", "target": "model_Logistic Regression", "label": "EVALUATES"})
    return {"nodes": nodes, "edges": edges}


def get_latest_prediction_graph_data():
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        with driver.session(database=database) as session:
            record = session.run(
                """
                MATCH (obs:Observation)-[:HAS_PREDICTION]->(pred:Prediction)
                MATCH (pred)-[:PRODUCED_BY]->(model:Model)
                MATCH (pred)-[:PREDICTED_AS]->(outcome:Outcome)
                RETURN obs.observation_id AS observation_id,
                       pred.prediction_id AS prediction_id,
                       pred.created_at AS created_at,
                       pred.predicted_class AS predicted_class,
                       pred.probability AS probability,
                       pred.decision_score AS decision_score,
                       model.name AS model_name,
                       outcome.label AS outcome_label
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
        {"id": "latest_observation", "label": "Observation", "group": "observation", "title": "Latest anonymous observation"},
        {"id": "latest_prediction", "label": "Prediction", "group": "prediction", "title": f"Created at {latest.get('created_at')}"},
        {"id": "latest_model", "label": latest["model_name"], "group": "model", "title": "Producing classifier"},
        {"id": "latest_outcome", "label": latest["outcome_label"], "group": "outcome", "title": "Predicted outcome"},
    ])
    edges.extend([
        {"source": "latest_observation", "target": "latest_prediction", "label": "HAS_PREDICTION"},
        {"source": "latest_prediction", "target": "latest_model", "label": "PRODUCED_BY"},
        {"source": "latest_prediction", "target": "latest_outcome", "label": "PREDICTED_AS"},
    ])
    if latest["outcome_label"] == "Diabetes":
        nodes.append({"id": "domain_disease", "label": "Type 2 Diabetes", "group": "disease", "title": "Educational domain concept"})
        edges.append({"source": "latest_outcome", "target": "domain_disease", "label": "RELATED_TO_DISEASE"})
        for item in DIABETES_DOMAIN_NODES["risk_factors"][:3]:
            node_id = f"risk_{item['id']}"
            nodes.append({"id": node_id, "label": item["name"], "group": "risk", "title": item["description"]})
            edges.append({"source": "domain_disease", "target": node_id, "label": "HAS_RISK_FACTOR"})
        for item in DIABETES_DOMAIN_NODES["guidance"][:2]:
            node_id = f"guidance_{item['id']}"
            nodes.append({"id": node_id, "label": item["name"], "group": "guidance", "title": item["description"]})
            edges.append({"source": "domain_disease", "target": node_id, "label": "HAS_GUIDANCE"})
        nodes.append({"id": "source_cdc", "label": "CDC / WHO Sources", "group": "source", "title": "Authoritative educational sources"})
        edges.append({"source": "domain_disease", "target": "source_cdc", "label": "SUPPORTED_BY_SOURCE"})
    return {"nodes": nodes, "edges": edges, "latest": latest}
