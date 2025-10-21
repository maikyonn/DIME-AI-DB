#!/usr/bin/env python3
"""
Setup LanceDB indexes on an existing table:

- Vector ANN index on `embedding` (IVF_PQ by default).
- Native BM25 full-text index on `text` and `biography` columns.

Examples:
  python scripts/setup_indexes.py --db-uri data/lancedb --table influencer_facets \
    --index-type IVF_PQ --metric cosine --num-partitions 256 --num-sub-vectors 64 \
    --fts

Note: Exact keyword args for `create_index` can vary slightly between LanceDB versions.
This script handles common variants with try/except fallbacks.
"""
import argparse
import logging
from datetime import timedelta

import lancedb

LOGGER = logging.getLogger("setup_indexes")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-uri", type=str, default="data/lancedb")
    ap.add_argument("--table", type=str, default="influencer_facets")
    ap.add_argument("--index-type", type=str, default="IVF_PQ", choices=["IVF_PQ", "IVF_HNSW_SQ", "HNSW"])
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2", "ip"])
    ap.add_argument("--num-partitions", type=int, default=256, help="IVF partitions (aka nlist)")
    ap.add_argument("--num-sub-vectors", type=int, default=64, help="PQ codebooks (m)")
    ap.add_argument("--wait-seconds", type=int, default=600)
    ap.add_argument("--fts", action="store_true", help="Create native BM25 full-text index on text+biography")
    ap.add_argument("--tokenizer", type=str, default="en_stem", help="FTS tokenizer name")
    args = ap.parse_args()

    db = lancedb.connect(args.db_uri)
    tbl = db.open_table(args.table)

    LOGGER.info("Creating vector index on 'embedding' (type=%s metric=%s)", args.index_type, args.metric)
    # Try modern signature
    try:
        tbl.create_index(
            vector_column_name="embedding",
            index_type=args.index_type,
            metric=args.metric,
            num_partitions=args.num_partitions,
            num_sub_vectors=args.num_sub_vectors,
            wait_timeout=timedelta(seconds=args.wait_seconds),
        )
    except TypeError:
        tbl.create_index(
            metric=args.metric,
            num_partitions=args.num_partitions,
            num_sub_vectors=args.num_sub_vectors,
            vector_column_name="embedding",
            index_type=args.index_type,
        )

    if args.fts:
        LOGGER.info("Creating native BM25 FTS on ['text','biography'] with tokenizer=%s", args.tokenizer)
        try:
            tbl.create_fts_index(["text", "biography"], use_tantivy=False, tokenizer_name=args.tokenizer)
        except (TypeError, ValueError):
            for column in ("text", "biography"):
                tbl.create_fts_index(column, use_tantivy=False, tokenizer_name=args.tokenizer)

    LOGGER.info("Done creating indexes.")


if __name__ == "__main__":
    main()
