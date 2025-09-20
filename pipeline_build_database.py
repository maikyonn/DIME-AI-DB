#!/usr/bin/env python3
"""
Build LanceDB (and optional vector database) from processed influencer datasets.

Assumes `pipeline_batch_process.py` has already produced
`<dataset>/organized_results/database_ready_results.csv`.

Example usage (single dataset):
    python pipeline_build_database.py data/instagram/insta100kemail1 \
        --vectors --model-name sentence-transformers/all-mpnet-base-v2

Example usage (root with many datasets):
    python pipeline_build_database.py data --vectors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from src.data.unified_data_loader import UnifiedDataLoader
from src.data.vector_database_builder import VectorDatabaseBuilder


def iter_dataset_dirs(root: Path) -> Iterable[Path]:
    """Yield dataset directories beneath `root`.

    If `root` contains CSVs directly, treat it as a single dataset directory.
    Otherwise yield immediate subdirectories that contain CSV inputs.
    """

    csvs = list(root.glob("*.csv"))
    if csvs:
        yield root
        return

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.csv")):
            yield child


def resolve_csv_path(dataset_dir: Path, csv_argument: Optional[str]) -> Path:
    if csv_argument:
        candidate = Path(csv_argument)
        if not candidate.is_absolute():
            candidate = dataset_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"CSV file not found: {candidate}")
        return candidate

    candidates = sorted(dataset_dir.glob("*with_lance_id*.csv"))
    if not candidates:
        candidates = sorted(dataset_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {dataset_dir}. Provide one with --csv."
        )
    return max(candidates, key=lambda path: path.stat().st_size)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LanceDB dataset")
    parser.add_argument(
        "dataset_dir",
        help="Path to the dataset directory (contains CSV and organized_results)",
    )
    parser.add_argument(
        "--csv",
        help="Specific CSV filename or path to use (defaults to largest *_with_lance_id.csv)",
    )
    parser.add_argument(
        "--db-path",
        help="Directory where LanceDB should be created (default: <dataset>/influencers_lancedb)",
    )
    parser.add_argument(
        "--table-name",
        default="influencer_profiles",
        help="LanceDB table name (default: influencer_profiles)",
    )
    parser.add_argument(
        "--vectors",
        action="store_true",
        help="Also build the vector database with embeddings",
    )
    parser.add_argument(
        "--vector-db-path",
        help="Directory for the vector LanceDB (default: <dataset>/influencers_vectordb)",
    )
    parser.add_argument(
        "--vector-table-name",
        default="influencer_profiles",
        help="Table name for the vector database",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model to use for embeddings",
    )
    return parser.parse_args(argv)


def process_dataset(args: argparse.Namespace, dataset_dir: Path) -> bool:
    try:
        csv_path = resolve_csv_path(dataset_dir, args.csv)
    except Exception as exc:
        print(f"❌ {exc}")
        return False

    db_path = Path(args.db_path).resolve() if args.db_path else dataset_dir / "influencers_lancedb"
    vector_db_path = (
        Path(args.vector_db_path).resolve()
        if args.vector_db_path
        else dataset_dir / "influencers_vectordb"
    )

    print(f"\n📁 Dataset directory: {dataset_dir}")
    print(f"📄 Source CSV: {csv_path.relative_to(dataset_dir)}")
    print(f"💾 LanceDB path: {db_path}")

    loader = UnifiedDataLoader(str(dataset_dir))
    table = loader.load_and_process_all(
        csv_filename=csv_path.name,
        jsonl_filename=None,
        db_path=str(db_path),
        table_name=args.table_name,
    )

    print(f"✅ LanceDB table `{args.table_name}` ready with {table.count_rows():,} rows")

    if args.vectors:
        print("\n🚀 Building vector database (this may take a while)…")
        builder = VectorDatabaseBuilder(
            data_dir=str(dataset_dir), model_name=args.model_name
        )
        vector_table = builder.create_vector_database(
            csv_filename=csv_path.name,
            jsonl_filename=None,
            db_path=str(vector_db_path),
            table_name=args.vector_table_name,
        )
        print(
            f"✅ Vector table `{args.vector_table_name}` ready with {vector_table.count_rows():,} rows"
        )

    return True


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    root_dir = Path(args.dataset_dir).resolve()
    if not root_dir.exists():
        print(f"❌ Dataset directory not found: {root_dir}")
        return 1

    dataset_dirs = list(iter_dataset_dirs(root_dir))
    if not dataset_dirs:
        print(f"❌ No CSV datasets found under {root_dir}")
        return 1

    successes = 0
    for dataset_dir in tqdm(dataset_dirs, desc="Datasets", unit="dataset"):
        if process_dataset(args, dataset_dir):
            successes += 1

    if successes == 0:
        print("❌ No datasets were processed successfully")
        return 1

    print(f"\n🎉 Database build complete for {successes} dataset(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
