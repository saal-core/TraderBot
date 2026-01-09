"""Extract Vanna training data from vanna_train.py to JSON files."""
import json
import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def extract_training_data():
    """Extract table/column documentation and Q&A pairs from vanna_train.py."""

    # Read the original training file
    training_file = Path(__file__).parent.parent / "vanna_train.py"
    with open(training_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract table-level documentation (multi-line format)
    table_docs_pattern = r"vn\.add_documentation\(\s*table_name='([^']+)',\s*schema_name='([^']+)',\s*documentation='([^']+)'\s*\)"
    table_matches = re.findall(table_docs_pattern, content, re.DOTALL)

    # Also handle double-quote format
    table_docs_pattern2 = r'vn\.add_documentation\(\s*table_name="([^"]+)",\s*schema_name="([^"]+)",\s*documentation="([^"]+)"\s*\)'
    table_matches2 = re.findall(table_docs_pattern2, content, re.DOTALL)

    table_docs = []
    for table_name, schema_name, documentation in table_matches + table_matches2:
        # Skip if it has column_name (that's column-level doc)
        table_docs.append({
            "table_name": table_name,
            "schema_name": schema_name,
            "documentation": documentation.strip()
        })

    # Extract column-level documentation
    column_docs_pattern = r"vn\.add_documentation\(table_name='([^']+)',\s*schema_name='([^']+)',\s*column_name='([^']+)',\s*documentation='([^']+)'\)"
    column_matches_single = re.findall(column_docs_pattern, content)

    # Also handle triple-quoted column documentation
    column_docs_pattern_multi = r"vn\.add_documentation\(\s*table_name='([^']+)',\s*schema_name='([^']+)',\s*column_name='([^']+)',\s*documentation=\"\"\"([^\"]+)\"\"\"\s*\)"
    column_matches_multi = re.findall(column_docs_pattern_multi, content, re.DOTALL)

    column_docs = []
    for table_name, schema_name, column_name, documentation in column_matches_single:
        column_docs.append({
            "table_name": table_name,
            "schema_name": schema_name,
            "column_name": column_name,
            "documentation": documentation.strip()
        })

    for table_name, schema_name, column_name, documentation in column_matches_multi:
        column_docs.append({
            "table_name": table_name,
            "schema_name": schema_name,
            "column_name": column_name,
            "documentation": documentation.strip()
        })

    # Extract Q&A pairs (single-line SQL)
    qa_pattern_single = r"vn\.add_question_sql\(question=\"([^\"]+)\",\s*sql=\"([^\"]+)\"\)"
    qa_matches_single = re.findall(qa_pattern_single, content)

    # Extract Q&A pairs (multi-line SQL)
    qa_pattern_multi = r'vn\.add_question_sql\(\s*question="([^"]+)",\s*sql="""(.*?)"""\s*\)'
    qa_matches_multi = re.findall(qa_pattern_multi, content, re.DOTALL)

    qa_pairs = []
    for question, sql in qa_matches_single:
        qa_pairs.append({
            "question": question,
            "sql": sql.strip()
        })

    for question, sql in qa_matches_multi:
        qa_pairs.append({
            "question": question,
            "sql": sql.strip()
        })

    return table_docs, column_docs, qa_pairs


def save_to_json():
    """Save extracted training data to JSON files."""
    print("Extracting training data...")
    table_docs, column_docs, qa_pairs = extract_training_data()

    output_dir = Path(__file__).parent.parent / "data" / "vanna_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "table_docs.json", 'w', encoding='utf-8') as f:
        json.dump(table_docs, f, indent=2, ensure_ascii=False)
    with open(output_dir / "schema_docs.json", 'w', encoding='utf-8') as f:
        json.dump(table_docs, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(table_docs)} schema docs to schema_docs.json and table_docs.json")

    # Save column documentation
    with open(output_dir / "column_docs.json", 'w', encoding='utf-8') as f:
        json.dump(column_docs, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(column_docs)} column documentation entries to column_docs.json")

    # Save Q&A pairs
    with open(output_dir / "question_sql_pairs.json", 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(qa_pairs)} question-SQL pairs to question_sql_pairs.json")

    print(f"\nTraining data successfully externalized to {output_dir}/")


if __name__ == "__main__":
    save_to_json()
