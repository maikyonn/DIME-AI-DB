#!/usr/bin/env python3
"""Combine Instagram and TikTok datasets into a unified CSV with LLM enrichment."""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings(
    "ignore",
    message="A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy",
)
warnings.filterwarnings(
    "ignore",
    message="[Errno 13] Permission denied.  joblib will operate in serial mode",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib\\._multiprocessing_helpers",
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pipeline_build_parquet import (  # noqa: E402
    COMBINED_FILENAME,
    COMBINED_HEADERS,
    COMBINED_SUBDIR,
    assign_combined_lance_ids,
    gather_csv_paths,
    merge_dataset_dataframe,
)


def iter_roots(base: Path) -> Iterable[Path]:
    instagram_dir = base / "instagram"
    tiktok_dir = base / "tiktok"
    if instagram_dir.is_dir() and tiktok_dir.is_dir():
        if list(instagram_dir.glob("*.csv")) and list(tiktok_dir.glob("*.csv")):
            yield base
            return

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        instagram_dir = subdir / "instagram"
        tiktok_dir = subdir / "tiktok"
        if instagram_dir.is_dir() and tiktok_dir.is_dir():
            if list(instagram_dir.glob("*.csv")) and list(tiktok_dir.glob("*.csv")):
                yield subdir


def combine_root(root: Path) -> Path:
    instagram_dir = root / "instagram"
    tiktok_dir = root / "tiktok"

    print(f"\n🔗 Combining platform datasets under {root}")

    instagram_csvs = gather_csv_paths(instagram_dir, None)
    instagram_df, insta_successes, insta_total = merge_dataset_dataframe(
        instagram_dir,
        instagram_csvs,
        max_workers=max(1, os.cpu_count() or 1),
    )
    print(
        f"   ✅ Instagram merge complete "
        f"({insta_successes}/{insta_total} CSVs → {len(instagram_df):,} rows)"
    )

    tiktok_csvs = gather_csv_paths(tiktok_dir, None)
    tiktok_df, tiktok_successes, tiktok_total = merge_dataset_dataframe(
        tiktok_dir,
        tiktok_csvs,
        max_workers=max(1, os.cpu_count() or 1),
    )
    print(
        f"   ✅ TikTok merge complete   "
        f"({tiktok_successes}/{tiktok_total} CSVs → {len(tiktok_df):,} rows)"
    )

    if instagram_df.empty and tiktok_df.empty:
        raise RuntimeError("Both Instagram and TikTok merges produced no rows")

    assign_combined_lance_ids(instagram_df, tiktok_df)

    combined_df = pd.concat([instagram_df, tiktok_df], ignore_index=True, sort=False)

    combined_dir = root / COMBINED_SUBDIR
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = combined_dir / COMBINED_FILENAME

    ordered_columns = COMBINED_HEADERS + [
        col for col in combined_df.columns if col not in COMBINED_HEADERS
    ]
    combined_df.to_csv(combined_csv, index=False, columns=ordered_columns)

    print(
        f"   💾 Wrote {len(combined_df):,} combined rows to {combined_csv}"
    )
    return combined_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine instagram/ and tiktok/ datasets into a unified CSV"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="data",
        help="Root directory containing instagram/ and tiktok/ subfolders (default: data)",
    )
    args = parser.parse_args()

    root_path = Path(args.root).expanduser().resolve()
    if not root_path.exists():
        print(f"❌ Root directory not found: {root_path}")
        return 1

    targets = list(iter_roots(root_path))
    if not targets:
        print(
            "❌ No directories found with both instagram/ and tiktok/ subfolders under"
            f" {root_path}"
        )
        return 1

    print(f"🔗 Found {len(targets)} dataset root(s) to combine")

    for dataset_root in targets:
        try:
            output = combine_root(dataset_root)
            print(f"✅ Output written to {output}")
        except Exception as exc:
            print(f"❌ Failed to combine dataset at {dataset_root}: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
