from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j_service import initialize_graph, verify_connection


def main():
    registry = json.loads((PROJECT_ROOT / "models" / "model_registry.json").read_text(encoding="utf-8"))
    metadata = json.loads((PROJECT_ROOT / "models" / "ui_metadata.json").read_text(encoding="utf-8"))
    if verify_connection():
        initialize_graph(registry, metadata)
        print("House Price Neo4j graph initialized successfully.")


if __name__ == "__main__":
    main()
