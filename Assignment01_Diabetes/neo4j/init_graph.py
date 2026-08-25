from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo4j_service import (  # noqa: E402
    get_knowledge_graph_summary,
    initialize_graph,
    verify_connection,
)


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main():
    features_path = PROJECT_ROOT / "models" / "selected_features.json"
    registry_path = PROJECT_ROOT / "models" / "model_registry.json"

    selected_features_payload = load_json(features_path)
    model_registry = load_json(registry_path)
    selected_features = selected_features_payload["selected_features"]

    print("Verifying Neo4j connection...")
    if verify_connection():
        print("Connection success.")

    print("Initializing graph...")
    counts = initialize_graph(selected_features, model_registry)

    print("Graph initialized.")
    print("Static knowledge graph summary:")
    for key, value in counts.items():
        print(f"- {key}: {value}")

    print("Re-reading graph summary...")
    summary = get_knowledge_graph_summary()
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Neo4j configuration error: {error}")
        print("Please configure:")
        print("- NEO4J_URI")
        print("- NEO4J_USER")
        print("- NEO4J_PASSWORD")
        print("- NEO4J_DATABASE (optional, defaults to neo4j)")
        raise SystemExit(1)
