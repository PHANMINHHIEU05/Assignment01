// Diabetes Prediction Knowledge Graph schema.
// This file documents constraints and useful inspection queries.
// It does not contain credentials.

// Neo4j 5 constraints.
CREATE CONSTRAINT model_name_unique IF NOT EXISTS
FOR (m:Model)
REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT feature_name_unique IF NOT EXISTS
FOR (f:Feature)
REQUIRE f.name IS UNIQUE;

CREATE CONSTRAINT outcome_value_unique IF NOT EXISTS
FOR (o:Outcome)
REQUIRE o.value IS UNIQUE;

CREATE CONSTRAINT observation_id_unique IF NOT EXISTS
FOR (o:Observation)
REQUIRE o.observation_id IS UNIQUE;

CREATE CONSTRAINT prediction_id_unique IF NOT EXISTS
FOR (p:Prediction)
REQUIRE p.prediction_id IS UNIQUE;

CREATE CONSTRAINT condition_name_unique IF NOT EXISTS
FOR (c:Condition)
REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT representation_name_unique IF NOT EXISTS
FOR (r:Representation)
REQUIRE r.name IS UNIQUE;

CREATE CONSTRAINT metric_name_unique IF NOT EXISTS
FOR (m:Metric)
REQUIRE m.name IS UNIQUE;

// Visualize model-feature relationships.
MATCH (m:Model)-[:USES_FEATURE]->(f:Feature)
RETURN m, f;

// Demo query: Model -> Feature -> Diabetes condition.
MATCH p =
    (m:Model)
    -[:USES_FEATURE]->
    (f:Feature)
    -[:INPUT_FOR]->
    (c:Condition)
RETURN p;

// Demo query: Model -> Six-feature representation -> Diabetes condition.
MATCH p =
    (m:Model)
    -[:USES_REPRESENTATION]->
    (r:Representation)
    -[:TARGETS]->
    (c:Condition)
RETURN p;

// Demo query: Feature -> Six-feature representation -> Diabetes condition.
MATCH p =
    (f:Feature)
    -[:PART_OF_REPRESENTATION]->
    (r:Representation)
    -[:TARGETS]->
    (c:Condition)
RETURN p;

// Cross-validation metrics for all deployment models.
MATCH (m:Model)-[r:HAS_CV_METRIC]->(metric:Metric)
RETURN
    m.name AS Model,
    metric.name AS Metric,
    r.value AS Value
ORDER BY Model, Metric;

// Held-out final test metrics for the scientific final model only.
MATCH (m:Model)-[r:HAS_FINAL_TEST_METRIC]->(metric:Metric)
RETURN
    m.name AS Model,
    metric.name AS Metric,
    r.value AS Value
ORDER BY Metric;

// Full recent prediction knowledge path.
MATCH
(o:Observation)-[:HAS_PREDICTION]->(p:Prediction)
-[:PRODUCED_BY]->(m:Model),
(p)-[:PREDICTED_AS]->(outcome:Outcome),
(m)-[:USES_REPRESENTATION]->(r:Representation)
-[:TARGETS]->(c:Condition)

RETURN o, p, m, outcome, r, c
ORDER BY p.created_at DESC
LIMIT 5;

// Complete static knowledge graph view.
MATCH p1 =
    (m:Model)-[:USES_FEATURE]->(f:Feature)-[:INPUT_FOR]->(c:Condition)

OPTIONAL MATCH p2 =
    (m)-[:USES_REPRESENTATION]->(r:Representation)-[:TARGETS]->(c)

OPTIONAL MATCH p3 =
    (o:Outcome)-[:OUTCOME_OF]->(c)

RETURN p1, p2, p3;

// Count static graph nodes and relationships.
MATCH (m:Model)
RETURN count(m) AS models;

MATCH (f:Feature)
RETURN count(f) AS features;

MATCH (:Model)-[r:USES_FEATURE]->(:Feature)
RETURN count(r) AS uses_feature_relationships;

MATCH (c:Condition)
RETURN count(c) AS conditions;

MATCH (r:Representation)
RETURN count(r) AS representations;

MATCH (metric:Metric)
RETURN count(metric) AS metrics;

MATCH (:Feature)-[r:INPUT_FOR]->(:Condition)
RETURN count(r) AS input_for_relationships;

MATCH (:Feature)-[r:PART_OF_REPRESENTATION]->(:Representation)
RETURN count(r) AS part_of_representation_relationships;

MATCH (:Representation)-[r:TARGETS]->(:Condition)
RETURN count(r) AS targets_relationships;

MATCH (:Outcome)-[r:OUTCOME_OF]->(:Condition)
RETURN count(r) AS outcome_of_relationships;

MATCH (:Model)-[r:USES_REPRESENTATION]->(:Representation)
RETURN count(r) AS uses_representation_relationships;

MATCH (:Model)-[r:HAS_CV_METRIC]->(:Metric)
RETURN count(r) AS cv_metric_relationships;

MATCH (:Model)-[r:HAS_FINAL_TEST_METRIC]->(:Metric)
RETURN count(r) AS final_test_metric_relationships;

MATCH (:Model)-[r:SCIENTIFIC_FINAL_MODEL_FOR]->(:Condition)
RETURN count(r) AS scientific_final_model_for_relationships;

// Expected after initialization:
// models = 5
// features = 6
// uses_feature_relationships = 30
// conditions = 1
// representations = 1
// metrics = 4
// input_for_relationships = 6
// part_of_representation_relationships = 6
// targets_relationships = 1
// outcome_of_relationships = 2
// uses_representation_relationships = 5
// cv_metric_relationships = 20
// final_test_metric_relationships = 4
// scientific_final_model_for_relationships = 1

// Check outcome nodes.
MATCH (o:Outcome)
RETURN o.value, o.label
ORDER BY o.value;

// Expected:
// 0 No Diabetes
// 1 Diabetes
