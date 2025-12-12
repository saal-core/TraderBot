#!/usr/bin/env python3
"""
csv_to_sql.py

Read a benchmark CSV (default: `benchmark_results.csv`) and create a SQL file
containing the extracted SQL queries. Each query is preceded by comments with
- the original natural-language question, and
- the LLM/model name used.

Usage:
    python scripts/csv_to_sql.py                    # reads benchmark_results.csv -> generated_queries.sql
    /bin/python3 scripts/csv_to_sql.py -i benchmark_results.csv -o out.sql --include-non-sql

The script will skip rows that don't look like SQL by default. Use
`--include-non-sql` to include everything.
"""

from __future__ import annotations
import argparse
import csv
import datetime
import os
import sys
from typing import Optional


def looks_like_sql(s: Optional[str]) -> bool:
    if not s:
        return False
    s_clean = s.strip().lower()
    # Basic checks: starts with select/with or contains known SQL keywords at start
    return s_clean.startswith("select") or s_clean.startswith("with") or "select" in s_clean[:50]


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark CSV to SQL file with comments.")
    parser.add_argument("-i", "--input", default="benchmark_results.csv", help="Input CSV file path")
    parser.add_argument("-o", "--output", default="generated_queries.sql", help="Output SQL file path")
    parser.add_argument("--include-non-sql", action="store_true", help="Include rows that do not look like SQL")
    parser.add_argument("--encoding", default="utf-8", help="File encoding to use when reading/writing")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    with open(args.input, newline="", encoding=args.encoding) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        print(f"No rows found in {args.input}")
        sys.exit(0)

    header = f"-- SQL file generated from {args.input} on {datetime.datetime.utcnow().isoformat()}Z\n"
    header += "-- Each query is preceded by comments with the question and model name.\n\n"

    written = 0
    with open(args.output, "w", encoding=args.encoding) as out:
        out.write(header)

        for idx, row in enumerate(rows, start=1):
            # Common column names used in benchmark output
            sql = (row.get("sql_query") or row.get("sql") or row.get("sql_query ") or "").strip()
            model = (row.get("llm_type") or row.get("model") or row.get("llm") or "").strip()
            question = (row.get("query") or row.get("question") or "").strip()
            success = row.get("success")

            if not sql:
                # nothing to write
                if args.include_non_sql:
                    out.write(f"-- ==== Entry {idx} (no SQL provided) ====\n")
                    out.write(f"-- Question: {question}\n")
                    out.write(f"-- Model: {model}\n")
                    out.write("-- No SQL found for this row.\n\n")
                continue

            if not args.include_non_sql and not looks_like_sql(sql):
                # skip chat responses / error messages that are not SQL
                continue

            # Write comment block
            out.write(f"-- ==== Entry {idx} ====\n")
            out.write(f"-- Question: {question}\n")
            out.write(f"-- Model: {model}\n")
            if success is not None:
                out.write(f"-- Success: {success}\n")
            out.write("-- Generated SQL:\n")

            # Write SQL. If the sql contains multiple queries, keep it as-is.
            out.write(sql)
            if not sql.rstrip().endswith(";"):
                out.write(";")
            out.write("\n\n")
            written += 1

    print(f"Wrote {written} SQL queries to {args.output}")


if __name__ == '__main__':
    main()
