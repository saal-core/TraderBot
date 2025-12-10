# train_vanna.py

import os
import logging
import json
from pathlib import Path
from config import get_vanna, get_db_params, VANNA_PERSIST_DIR

# --- 1. Vanna Setup ---
logging.basicConfig(level=logging.INFO)
vn = get_vanna()

# --- 2. Database Connection ---
try:
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT = get_db_params()
    if all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
        vn.connect_to_postgres(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        logging.info("Connected to the database")
    else:
        logging.warning("Database environment variables not fully set; proceeding without DB connection")
except Exception as e:
    logging.error(f"Database connection failed: {e}")

# ==============================================================================
# --- 3. VANNA TRAINING DATA ---
# ==============================================================================
logging.info("Starting Vanna training")
training_dir = Path(__file__).parent / "data" / "vanna_training"
schema_path = training_dir / "schema_docs.json"
table_path = training_dir / "table_docs.json"
column_path = training_dir / "column_docs.json"
qa_path = training_dir / "question_sql_pairs.json"

def _load_json_list(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def vanna_add_doc(vn_instance, item: dict):
    if "column_name" in item:
        vn_instance.add_documentation(
            table_name=item["table_name"],
            schema_name=item["schema_name"],
            column_name=item["column_name"],
            documentation=item["documentation"],
        )
    else:
        vn_instance.add_documentation(
            table_name=item["table_name"],
            schema_name=item["schema_name"],
            documentation=item["documentation"],
        )

def vanna_add_docs_from_files(vn_instance, schema_path: Path, table_path: Path, column_path: Path):
    docs = []
    if schema_path.exists():
        docs = _load_json_list(schema_path)
    elif table_path.exists():
        docs = _load_json_list(table_path)
    for d in docs:
        vanna_add_doc(vn_instance, d)
    if column_path.exists():
        for d in _load_json_list(column_path):
            vanna_add_doc(vn_instance, d)

def vanna_add_qa_from_file(vn_instance, qa_path: Path):
    if qa_path.exists():
        for q in _load_json_list(qa_path):
            vn_instance.add_question_sql(question=q["question"], sql=q["sql"])
if schema_path.exists() or table_path.exists() or column_path.exists():
    vanna_add_docs_from_files(vn, schema_path, table_path, column_path)
    vanna_add_qa_from_file(vn, qa_path)
    logging.info("Loaded Vanna training data from JSON")
    logging.info(f"Vanna training complete; index in '{VANNA_PERSIST_DIR}'")
    raise SystemExit(0)

# ==============================================================================
# If JSON files don't exist, you need to create them first.
# ==============================================================================
logging.error("Training JSON files not found!")
logging.error(f"Expected files in: {training_dir}/")
logging.error("- schema_docs.json (or table_docs.json)")
logging.error("- column_docs.json")
logging.error("- question_sql_pairs.json")

if __name__ == "__main__":
    pass
