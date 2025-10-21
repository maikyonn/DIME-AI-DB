#!/usr/bin/env python3
"""
Quickly introspect a LanceDB table: print schema, row counts, sample rows.
"""
import argparse
import lancedb
import pprint
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-uri", type=str, default="data/lancedb")
    ap.add_argument("--table", type=str, default="influencer_facets")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    db = lancedb.connect(args.db_uri)
    tbl = db.open_table(args.table)

    print("=== TABLE ===")
    print(f"Name: {args.table}")
    try:
        print("Rows:", tbl.count_rows())
    except Exception:
        pass

    print("\n=== SCHEMA ===")
    try:
        print(tbl.schema)
    except Exception as e:
        print("Schema unavailable:", e)

    print("\n=== SAMPLE ROWS ===")
    try:
        df = tbl.to_pandas()
    except TypeError:
        df = tbl.to_arrow().to_pandas()
    if args.limit:
        df = df.head(args.limit)
    with pd.option_context("display.max_colwidth", 200):
        print(df)


if __name__ == "__main__":
    main()
