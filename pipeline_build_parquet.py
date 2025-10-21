#!/usr/bin/env python3
"""Build normalized parquet datasets from CSV + LLM outputs.

This utility replaces the previous LanceDB build stage. It merges language-
filtered CSV exports with LLM batch results and writes `normalized_profiles.parquet`
files that downstream tools (e.g., LanceDB loaders, embedding builders) can
consume. When both Instagram and TikTok datasets are present under the root
path, a combined dataset is produced in the configured combined subdirectory.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable

from pipeline_batch_process import (
    COMBINED_FILENAME,
    COMBINED_HEADERS,
    COMBINED_SUBDIR,
    _compute_post_statistics,
    _count_instagram_media,
    _merge_tiktok_posts,
    _normalize_posts,
)
from src.data.dataset_builder import UnifiedDataLoader

SCRIPT_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_ROOT / "pipeline"
LANGUAGE_FILTER_DIR = PIPELINE_DIR / "step0_language_filter"
LLM_RESULTS_DIR = PIPELINE_DIR / "step2_batch_results"

NORMALIZED_PROFILE_COLUMNS = COMBINED_HEADERS + [
    "individual_vs_org_score",
    "generational_appeal_score",
    "professionalization_score",
    "relationship_status_score",
    "location",
    "ethnicity",
    "age",
    "occupation",
    "keyword1",
    "keyword2",
    "keyword3",
    "keyword4",
    "keyword5",
    "keyword6",
    "keyword7",
    "keyword8",
    "keyword9",
    "keyword10",
    "prompt_file",
    "raw_response",
    "processing_error",
    "source_batch",
    "llm_processed",
    "source_csv",
]


@dataclass
class CSVContext:
    dataset_dir: Path
    requested_csv: Path
    effective_csv: Path
    namespace: str
    stage0_csv: Optional[Path]


def _first_non_empty(*values: Optional[str]) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_flag(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return "true" if value else "false"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return "true"
        if normalized in {"false", "0", "no", "n"}:
            return "false"
    return ""


def _to_string(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _normalize_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _normalize_tiktok_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    length = len(df)

    df["platform"] = "tiktok"
    df["platform_id"] = df.get("id", pd.Series([""] * length)).apply(_to_string)
    df["username"] = df.get("account_id", pd.Series([""] * length)).apply(_to_string)
    df["display_name"] = df.apply(
        lambda row: _first_non_empty(
            _to_string(row.get("profile_name")),
            _to_string(row.get("nickname")),
            _to_string(row.get("account_id")),
        ),
        axis=1,
    )

    df["likes_total"] = df.get("likes", pd.Series([""] * length)).apply(_to_string)
    df["engagement_rate"] = df.get("awg_engagement_rate", pd.Series([""] * length)).apply(
        _to_string
    )
    df["posts_count"] = df.get("videos_count", pd.Series([""] * length)).apply(
        _to_string
    )
    df["external_url"] = df.get("bio_link", pd.Series([""] * length)).apply(_to_string)
    df["profile_url"] = df.get("url", pd.Series([""] * length)).apply(_to_string)
    df["profile_image_url"] = df.apply(
        lambda row: _first_non_empty(
            _to_string(row.get("profile_pic_url_hd")),
            _to_string(row.get("profile_pic_url")),
        ),
        axis=1,
    )
    df["is_verified"] = df.get("is_verified", pd.Series([""] * length)).apply(
        _normalize_flag
    )
    df["is_private"] = df.get("is_private", pd.Series([""] * length)).apply(
        _normalize_flag
    )
    df["is_commerce_user"] = df.get("is_commerce_user", pd.Series([""] * length)).apply(
        _normalize_flag
    )

    posts_json: List[str] = []
    reel_ratio: List[str] = []
    median_view: List[str] = []
    median_like: List[str] = []
    median_comment: List[str] = []

    for row in df.to_dict("records"):
        merged_posts = _merge_tiktok_posts(row)
        normalized_posts = _normalize_posts(merged_posts, "tiktok")
        stats = _compute_post_statistics(normalized_posts)
        posts_json.append(normalized_posts)
        reel_ratio.append(stats[0])
        median_view.append(stats[1])
        median_like.append(stats[2])
        median_comment.append(stats[3])

    df["posts"] = posts_json
    df["reel_post_ratio_last10"] = reel_ratio
    df["median_view_count_last10"] = median_view
    df["median_like_count_last10"] = median_like
    df["median_comment_count_last10"] = median_comment
    df["total_img_posts_ig"] = [""] * length
    df["total_reels_ig"] = [""] * length

    return df


def _normalize_instagram_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    length = len(df)

    df["platform"] = "instagram"
    df["platform_id"] = df.apply(
        lambda row: _first_non_empty(
            _to_string(row.get("fbid")),
            _to_string(row.get("id")),
        ),
        axis=1,
    )
    df["username"] = df.get("account", pd.Series([""] * length)).apply(_to_string)
    df["display_name"] = df.apply(
        lambda row: _first_non_empty(
            _to_string(row.get("profile_name")),
            _to_string(row.get("full_name")),
            _to_string(row.get("account")),
        ),
        axis=1,
    )
    df["likes_total"] = df.get("likes_total", pd.Series([""] * length)).apply(
        _to_string
    )
    df["engagement_rate"] = df.get("avg_engagement", pd.Series([""] * length)).apply(
        _to_string
    )
    df["posts_count"] = df.get("posts_count", pd.Series([""] * length)).apply(
        _to_string
    )
    df["profile_url"] = df.get("profile_url", pd.Series([""] * length)).apply(
        _to_string
    )
    df["profile_image_url"] = df.get(
        "profile_image_link", pd.Series([""] * length)
    ).apply(_to_string)
    df["is_verified"] = df.get("is_verified", pd.Series([""] * length)).apply(
        _normalize_flag
    )
    df["is_private"] = df.get("is_private", pd.Series([""] * length)).apply(
        _normalize_flag
    )
    df["is_commerce_user"] = ["false"] * length

    def _normalize_external(value: object) -> str:
        text = _to_string(value)
        if not text:
            return ""
        if text.strip().startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list) and parsed:
                    return _to_string(parsed[0])
            except Exception:
                return text
        return text

    df["external_url"] = df.get("external_url", pd.Series([""] * length)).apply(
        _normalize_external
    )

    posts_json: List[str] = []
    reel_ratio: List[str] = []
    median_view: List[str] = []
    median_like: List[str] = []
    median_comment: List[str] = []
    total_images: List[str] = []
    total_reels: List[str] = []

    for row in df.to_dict("records"):
        raw_posts = row.get("posts", "")
        normalized_posts = _normalize_posts(raw_posts, "instagram")
        stats = _compute_post_statistics(normalized_posts)
        posts_json.append(normalized_posts)
        reel_ratio.append(stats[0])
        median_view.append(stats[1])
        median_like.append(stats[2])
        median_comment.append(stats[3])

        try:
            decoded_posts = json.loads(normalized_posts) if normalized_posts else []
        except json.JSONDecodeError:
            decoded_posts = []
        img_count, reel_count = _count_instagram_media(decoded_posts)
        total_images.append(img_count)
        total_reels.append(reel_count)

    df["posts"] = posts_json
    df["reel_post_ratio_last10"] = reel_ratio
    df["median_view_count_last10"] = median_view
    df["median_like_count_last10"] = median_like
    df["median_comment_count_last10"] = median_comment
    df["total_img_posts_ig"] = total_images
    df["total_reels_ig"] = total_reels

    return df


def normalize_dataset_records(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset_name.lower()
    if dataset == "tiktok":
        return _normalize_tiktok_dataframe(df)
    if dataset == "instagram":
        return _normalize_instagram_dataframe(df)
    return df


def gather_csv_paths(dataset_dir: Path, csv_argument: Optional[str]) -> List[Path]:
    dataset_dir = dataset_dir.resolve()

    if dataset_dir.is_file():
        if dataset_dir.suffix.lower() == ".csv":
            return [dataset_dir]
        raise FileNotFoundError(f"Dataset path is a file but not a CSV: {dataset_dir}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if csv_argument:
        candidate = Path(csv_argument)
        if not candidate.is_absolute():
            candidate = dataset_dir / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"CSV file not found: {candidate}")
        if candidate.is_dir():
            raise IsADirectoryError(f"CSV argument points to a directory: {candidate}")
        if candidate.suffix.lower() != ".csv":
            raise ValueError(f"CSV argument must be a .csv file: {candidate}")
        return [candidate]

    candidates: List[Path] = []
    seen: set[Path] = set()
    for pattern in ("*with_lance_id*.csv", "*.csv"):
        for path in sorted(dataset_dir.glob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved not in seen:
                candidates.append(resolved)
                seen.add(resolved)

    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}. Provide one with --csv.")

    return candidates


def iter_dataset_dirs(root: Path) -> Iterable[Path]:
    csvs = list(root.glob("*.csv"))
    if csvs:
        yield root.resolve()
        return

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.csv")):
            yield child.resolve()


def build_csv_context(dataset_dir: Path, csv_path: Path) -> CSVContext:
    csv_path = csv_path.resolve()
    namespace = f"{dataset_dir.name}_{csv_path.stem}_csv"
    stage0_candidate = LANGUAGE_FILTER_DIR / f"{namespace}_english_profiles_with_lance_id.csv"
    effective_csv = stage0_candidate if stage0_candidate.exists() else csv_path

    return CSVContext(
        dataset_dir=dataset_dir.resolve(),
        requested_csv=csv_path,
        effective_csv=effective_csv,
        namespace=namespace,
        stage0_csv=stage0_candidate if stage0_candidate.exists() else None,
    )


def create_loader(csv_context: CSVContext) -> UnifiedDataLoader:
    loader = UnifiedDataLoader(str(csv_context.dataset_dir))
    loader.csv_dir = csv_context.effective_csv.parent
    loader.llm_dir = LLM_RESULTS_DIR
    loader.llm_namespace = csv_context.namespace
    return loader


def print_csv_overview(csv_context: CSVContext) -> None:
    def describe_path(path: Path) -> str:
        try:
            return str(path.relative_to(SCRIPT_ROOT))
        except ValueError:
            return str(path)

    print(f"Dataset directory : {describe_path(csv_context.dataset_dir)}")
    print(f"Requested CSV     : {describe_path(csv_context.requested_csv)}")
    if csv_context.stage0_csv is not None:
        print(
            "Language-filtered : "
            f"{describe_path(csv_context.stage0_csv)} (used for merge)"
        )
    else:
        print(f"Loading CSV       : {describe_path(csv_context.effective_csv)}")


def load_merged_dataframe(loader: UnifiedDataLoader, csv_context: CSVContext) -> pd.DataFrame:
    csv_filename = csv_context.effective_csv.name
    csv_df = loader.load_csv_data(csv_filename)
    llm_data = loader.parse_batch_output(filename=None)
    merged_df = loader.merge_data(csv_df, llm_data)
    merged_df = normalize_dataset_records(csv_context.dataset_dir.name.lower(), merged_df)
    merged_df = merged_df.copy()
    merged_df["source_csv"] = describe_path(csv_context.effective_csv)
    return merged_df


def describe_path(path: Path) -> str:
    try:
        return str(path.relative_to(SCRIPT_ROOT))
    except ValueError:
        return str(path)


def merge_dataset_dataframe(
    dataset_dir: Path,
    csv_paths: List[Path],
    max_workers: int,
) -> Tuple[pd.DataFrame, int, int]:
    csv_contexts = [build_csv_context(dataset_dir, path) for path in csv_paths]
    total_csvs = len(csv_contexts)
    worker_count = min(max(1, max_workers), total_csvs)

    merged_frames: List[pd.DataFrame] = []
    csv_successes = 0

    def merge_worker(csv_context: CSVContext) -> pd.DataFrame:
        loader = create_loader(csv_context)
        return load_merged_dataframe(loader, csv_context)

    print(f"→ Merging {total_csvs} CSV file(s) using {worker_count} worker(s)…")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(merge_worker, csv_context): csv_context
            for csv_context in csv_contexts
        }

        iterator = as_completed(futures)
        iterator = tqdm(iterator, total=total_csvs, desc="CSV merges", unit="csv")

        for future in iterator:
            csv_context = futures[future]
            print_csv_overview(csv_context)

            try:
                merged_df = future.result()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"❌ Failed to merge {describe_path(csv_context.effective_csv)}: {exc}"
                )
                continue

            print(
                f"✅ Merged {len(merged_df):,} rows from "
                f"{describe_path(csv_context.effective_csv)}"
            )
            merged_frames.append(merged_df)
            csv_successes += 1

    combined_df = (
        pd.concat(merged_frames, ignore_index=True)
        if merged_frames
        else pd.DataFrame()
    )

    return combined_df, csv_successes, total_csvs


def write_normalized_parquet(dataset_dir: Path, combined_df: pd.DataFrame) -> Path:
    normalized_columns = [
        column for column in NORMALIZED_PROFILE_COLUMNS if column in combined_df.columns
    ]
    output_path = dataset_dir / "normalized_profiles.parquet"
    if combined_df.empty:
        print(f"⚠️ Skipping parquet export for {describe_path(dataset_dir)} (no rows)")
        return output_path

    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ Exporting normalized profiles to {describe_path(output_path)}…")
    normalized_df = combined_df.loc[:, normalized_columns].copy()
    normalized_df.to_parquet(output_path, index=False)
    print(f"✅ Normalized parquet ready at {describe_path(output_path)}")
    return output_path


def assign_combined_lance_ids(
    instagram_df: pd.DataFrame, tiktok_df: pd.DataFrame
) -> None:
    for frame in (instagram_df, tiktok_df):
        if "lance_db_id" not in frame.columns:
            frame["lance_db_id"] = ""

    used_ids: set[str] = set()
    next_numeric = 1

    def _allocate(value: str) -> str:
        nonlocal next_numeric
        base = value or ""
        if not base:
            base = str(next_numeric)
            next_numeric += 1
        candidate = base
        suffix = 1
        while candidate in used_ids:
            if base.isdigit():
                candidate = str(next_numeric)
                next_numeric += 1
            else:
                candidate = f"{base}_{suffix}"
                suffix += 1
        used_ids.add(candidate)
        if candidate.isdigit():
            try:
                next_numeric = max(next_numeric, int(candidate) + 1)
            except ValueError:
                pass
        return candidate

    insta_col = instagram_df.columns.get_loc("lance_db_id")
    for idx in range(len(instagram_df)):
        current = _normalize_id(instagram_df.iat[idx, insta_col])
        instagram_df.iat[idx, insta_col] = _allocate(current)

    tiktok_col = tiktok_df.columns.get_loc("lance_db_id")
    for idx in range(len(tiktok_df)):
        current = _normalize_id(tiktok_df.iat[idx, tiktok_col])
        tiktok_df.iat[idx, tiktok_col] = _allocate(current)


def process_dataset(
    dataset_dir: Path,
    csv_paths: List[Path],
    max_workers: int,
) -> Tuple[int, int, int]:
    combined_df, csv_successes, total_csvs = merge_dataset_dataframe(
        dataset_dir,
        csv_paths,
        max_workers,
    )

    if combined_df.empty:
        print(
            f"❌ No CSVs were merged successfully for {describe_path(dataset_dir)}"
        )
        return 0, csv_successes, total_csvs

    write_normalized_parquet(dataset_dir, combined_df)
    success_flag = 1 if not combined_df.empty else 0
    return success_flag, csv_successes, total_csvs


def process_combined_platforms(args: argparse.Namespace, root_dir: Path) -> int:
    instagram_dir = root_dir / "instagram"
    tiktok_dir = root_dir / "tiktok"

    instagram_df, insta_successes, insta_total = merge_dataset_dataframe(
        instagram_dir,
        gather_csv_paths(instagram_dir, None),
        max_workers=args.csv_workers,
    )

    tiktok_df, tiktok_successes, tiktok_total = merge_dataset_dataframe(
        tiktok_dir,
        gather_csv_paths(tiktok_dir, None),
        max_workers=args.csv_workers,
    )

    if instagram_df.empty and tiktok_df.empty:
        print("❌ Failed to merge both Instagram and TikTok datasets")
        return 1

    assign_combined_lance_ids(instagram_df, tiktok_df)

    combined_df = pd.concat(
        [instagram_df, tiktok_df], ignore_index=True, sort=False
    )

    combined_dir = root_dir / COMBINED_SUBDIR
    combined_dir.mkdir(parents=True, exist_ok=True)

    ordered_columns = COMBINED_HEADERS + [
        col for col in combined_df.columns if col not in COMBINED_HEADERS
    ]

    combined_csv_path = combined_dir / COMBINED_FILENAME
    print(f"→ Writing combined CSV to {describe_path(combined_csv_path)}")
    combined_df.to_csv(combined_csv_path, index=False, columns=ordered_columns)

    write_normalized_parquet(combined_dir, combined_df)

    print(
        f"✅ Combined {len(instagram_df):,} Instagram rows"
        f" ({insta_successes}/{insta_total} CSVs)"
    )
    print(
        f"✅ Combined {len(tiktok_df):,} TikTok rows"
        f" ({tiktok_successes}/{tiktok_total} CSVs)"
    )
    print(
        "🎉 Combined platform dataset complete"
        f" (rows={len(combined_df):,})"
    )
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized parquet dataset(s)")
    parser.add_argument(
        "dataset_dir",
        help="Path to the dataset directory (contains one or more CSV files)",
    )
    parser.add_argument(
        "--csv",
        help="Specific CSV filename or path to use (skip auto-discovery)",
    )
    parser.add_argument(
        "--csv-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Maximum number of CSV merge workers (default: CPU count)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    root_dir = Path(args.dataset_dir).expanduser().resolve()
    if not root_dir.exists():
        print(f"❌ Dataset directory not found: {root_dir}")
        return 1

    instagram_dir = root_dir / "instagram"
    tiktok_dir = root_dir / "tiktok"
    if instagram_dir.is_dir() and tiktok_dir.is_dir():
        return process_combined_platforms(args, root_dir)

    dataset_dirs = list(iter_dataset_dirs(root_dir))
    if not dataset_dirs:
        print(f"❌ No CSV datasets found under {root_dir}")
        return 1

    dataset_successes = 0
    csv_successes_total = 0
    csv_attempts_total = 0

    for dataset_dir in tqdm(dataset_dirs, desc="Datasets", unit="dataset"):
        csv_paths = gather_csv_paths(dataset_dir, args.csv)
        success, csv_successes, csv_attempts = process_dataset(
            dataset_dir,
            csv_paths,
            max_workers=args.csv_workers,
        )
        dataset_successes += success
        csv_successes_total += csv_successes
        csv_attempts_total += csv_attempts

    if dataset_successes == 0:
        print("❌ No datasets were processed successfully")
        return 1

    print(
        f"\n🎉 Parquet build complete for {dataset_successes} dataset(s)"
        f" ({csv_successes_total}/{csv_attempts_total} CSVs merged)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
