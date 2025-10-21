#!/usr/bin/env python3
"""Verify that the combined social profiles CSV is complete."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

from dataclasses import dataclass

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            class _Dummy:
                def update(self, *_args, **_kwargs):
                    return None

                def close(self):
                    return None

            return _Dummy()
        return iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - parquet export requires pandas
    pd = None

# Columns that should always be present after LLM enrichment
LLM_SCORE_OPTIONS: Sequence[Tuple[str, Sequence[str]]] = (
    ("individual_vs_org", ("individual_vs_org", "individual_vs_org_score")),
    ("generational_appeal", ("generational_appeal", "generational_appeal_score")),
    ("professionalization", ("professionalization", "professionalization_score")),
    ("relationship_status", ("relationship_status", "relationship_status_score")),
)
LLM_REQUIRED_COLUMNS: Sequence[str] = [
    "location",
    "ethnicity",
    "age",
    "occupation",
    *[f"keyword{i}" for i in range(1, 11)],
    "prompt_file",
    "raw_response",
    "processing_error",
    "llm_processed",
    "source_batch",
    "source_csv",
]

INSTAGRAM_KEY_FIELDS: Dict[str, str] = {
    "account": "account",
    "username": "account",
    "id": "id",
    "fbid": "fbid",
    "profile_url": "profile_url",
}
TIKTOK_KEY_FIELDS: Dict[str, str] = {
    "account_id": "account_id",
    "username": "account_id",
    "id": "id",
    "url": "url",
    "profile_url": "url",
}

ACCOUNT_KEYS = {"account", "username", "account_id"}
URL_KEYS = {"url", "profile_url", "external_url", "bio_link"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instagram-dir",
        type=Path,
        default=Path("data/instagram"),
        help="Directory containing Instagram source CSV files (default: data/instagram)",
    )
    parser.add_argument(
        "--tiktok-file",
        type=Path,
        default=Path("data/tiktok/tiktok.csv"),
        help="TikTok source CSV file (default: data/tiktok/tiktok.csv)",
    )
    parser.add_argument(
        "--combined-file",
        type=Path,
        default=Path("data/combined/social_profiles.csv"),
        help="Combined social profiles CSV file to verify (default: data/combined/social_profiles.csv)",
    )
    parser.add_argument(
        "--language-dir",
        type=Path,
        default=Path("pipeline/step0_language_filter"),
        help="Directory containing language-filtered CSV outputs (default: pipeline/step0_language_filter)",
    )
    parser.add_argument(
        "--max-missing-preview",
        type=int,
        default=10,
        help="Maximum number of missing-row examples to display (default: 10)",
    )
    parser.add_argument(
        "--parquet-out",
        type=Path,
        default=Path("data/combined/social_profiles.parquet"),
        help="Write a Parquet copy of the combined CSV to this path when verification succeeds",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip writing a Parquet copy even if verification succeeds",
    )
    return parser.parse_args(argv)


def set_csv_field_size_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2_147_483_647)


def normalize_value(key: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    trimmed = value.strip()
    if not trimmed:
        return None
    if key in ACCOUNT_KEYS:
        return trimmed.lower()
    if key in URL_KEYS:
        return trimmed.lower()
    return trimmed


def _add(index: DefaultDict[str, set], key: str, value: str | None) -> None:
    normalized = normalize_value(key, value)
    if normalized:
        index[key].add(normalized)


def build_combined_index(
    combined_csv: Path,
) -> Tuple[List[str], Dict[str, DefaultDict[str, set]], Dict[str, int]]:
    indexes: Dict[str, DefaultDict[str, set]] = {
        "instagram": defaultdict(set),
        "tiktok": defaultdict(set),
    }
    counts: Dict[str, int] = defaultdict(int)

    with combined_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        for row in tqdm(reader, desc="Indexing combined CSV", unit="rows"):
            platform = (row.get("platform") or "").strip().lower()
            counts[platform] += 1
            if platform == "instagram":
                _add(indexes["instagram"], "account", row.get("account"))
                _add(indexes["instagram"], "account", row.get("username"))
                _add(indexes["instagram"], "id", row.get("id"))
                _add(indexes["instagram"], "fbid", row.get("fbid"))
                _add(indexes["instagram"], "fbid", row.get("platform_id"))
                _add(indexes["instagram"], "profile_url", row.get("profile_url"))
            elif platform == "tiktok":
                _add(indexes["tiktok"], "account_id", row.get("account_id"))
                _add(indexes["tiktok"], "account_id", row.get("username"))
                _add(indexes["tiktok"], "id", row.get("id"))
                _add(indexes["tiktok"], "id", row.get("platform_id"))
                _add(indexes["tiktok"], "url", row.get("url"))
                _add(indexes["tiktok"], "url", row.get("profile_url"))
    return header, indexes, counts


def read_csv_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if row:
                return [col.strip() for col in row]
    return []


def _row_in_index(
    row: Dict[str, str],
    field_map: Dict[str, str],
    index: DefaultDict[str, set],
) -> bool:
    for source_field, index_key in field_map.items():
        value = row.get(source_field)
        normalized = normalize_value(index_key, value)
        if normalized and normalized in index[index_key]:
            return True
    return False


def _row_identity(row: Dict[str, str], field_map: Dict[str, str]) -> List[Tuple[str, str]]:
    identity: List[Tuple[str, str]] = []
    for source_field in field_map:
        value = row.get(source_field)
        if value:
            identity.append((source_field, value))
    if identity:
        return identity
    for fallback in ("platform_id", "lance_db_id", "username", "account_id", "id"):
        value = row.get(fallback)
        if value:
            identity.append((fallback, value))
    return identity


@dataclass
class FileCoverage:
    total: int = 0
    matched: int = 0


def verify_source_rows(
    csv_paths: Sequence[Path],
    field_map: Dict[str, str],
    index: DefaultDict[str, set],
) -> Tuple[int, int, List[Tuple[str, int, List[Tuple[str, str]]]], Dict[str, FileCoverage]]:
    missing: List[Tuple[str, int, List[Tuple[str, str]]]] = []
    total = 0
    matched = 0
    coverage: Dict[str, FileCoverage] = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            iterator = tqdm(reader, desc=f"Checking {csv_path.name}", unit="rows")
            file_stats = FileCoverage()
            for row_number, row in enumerate(iterator, start=2):
                total += 1
                file_stats.total += 1
                if not row:
                    continue
                if _row_in_index(row, field_map, index):
                    matched += 1
                    file_stats.matched += 1
                    continue
                missing.append((csv_path.name, row_number, _row_identity(row, field_map)))
            if hasattr(iterator, "close"):
                iterator.close()
            coverage[csv_path.name] = file_stats
    return total, matched, missing, coverage


def describe_missing(
    label: str,
    missing: Sequence[Tuple[str, int, List[Tuple[str, str]]]],
    max_examples: int,
) -> None:
    print(f"❌ Missing {len(missing)} {label} rows in combined CSV")
    for entry in missing[:max_examples]:
        file_name, row_number, identity = entry
        identity_str = ", ".join(f"{key}={value}" for key, value in identity) or "<no identifiers>"
        print(f"   - {file_name}:{row_number} → {identity_str}")
    if len(missing) > max_examples:
        print(f"     ... {len(missing) - max_examples} more not shown")


def verify_columns(
    combined_header: Sequence[str],
    instagram_columns: Sequence[str],
    tiktok_columns: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    hdr_set = set(combined_header)
    instagram_missing = sorted(col for col in set(instagram_columns) if col and col not in hdr_set)
    tiktok_missing = sorted(col for col in set(tiktok_columns) if col and col not in hdr_set)

    llm_missing: List[str] = []
    for logical_name, options in LLM_SCORE_OPTIONS:
        if not any(option in hdr_set for option in options):
            llm_missing.append(logical_name)
    for column in LLM_REQUIRED_COLUMNS:
        if column not in hdr_set:
            llm_missing.append(column)
    return instagram_missing, tiktok_missing, llm_missing


def resolve_language_csv(source_csv: Path, platform: str, language_dir: Path) -> Optional[Path]:
    candidates = [
        language_dir / f"{platform}_{source_csv.stem}_csv_english_profiles_with_lance_id.csv",
        language_dir / f"{platform}_{source_csv.stem}_csv_english_profiles.csv",
        language_dir / f"{platform}_{source_csv.stem}_english_profiles.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_language_csvs(
    source_csvs: Sequence[Path], platform: str, language_dir: Path
) -> Tuple[List[Path], List[str]]:
    results: List[Path] = []
    missing: List[str] = []
    for csv_path in source_csvs:
        resolved = resolve_language_csv(csv_path, platform, language_dir)
        if resolved is None:
            missing.append(csv_path.name)
        else:
            results.append(resolved)
    return results, missing


def print_coverage(
    label: str, coverage: Dict[str, FileCoverage], combined_total: Optional[int] = None
) -> None:
    if not coverage:
        print(f"{label}: no files evaluated")
        return
    print(label)
    for name in sorted(coverage):
        stats = coverage[name]
        if stats.total == 0:
            pct = 0.0
        else:
            pct = 100.0 * stats.matched / stats.total
        share_text = ""
        if combined_total and combined_total > 0:
            share = 100.0 * stats.matched / combined_total
            share_text = f", {share:.2f}% of combined"
        print(
            f"   - {name}: {stats.matched}/{stats.total} rows ({pct:.2f}% coverage{share_text})"
        )


def export_parquet(csv_path: Path, parquet_path: Path) -> bool:
    if pd is None:
        print("⚠️ pandas is not available; skipping Parquet export")
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"❌ Failed to read combined CSV for Parquet export: {exc}")
        return False
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        print(f"❌ Failed to write Parquet file: {exc}")
        return False
    print(f"💾 Wrote Parquet dataset to {parquet_path}")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    set_csv_field_size_limit()

    instagram_dir = args.instagram_dir.resolve()
    tiktok_file = args.tiktok_file.resolve()
    combined_file = args.combined_file.resolve()

    if not instagram_dir.is_dir():
        print(f"❌ Instagram directory not found: {instagram_dir}", file=sys.stderr)
        return 2
    instagram_csvs = sorted(instagram_dir.glob("*.csv"))
    if not instagram_csvs:
        print(f"❌ No Instagram CSV files found under {instagram_dir}", file=sys.stderr)
        return 2
    if not tiktok_file.is_file():
        print(f"❌ TikTok CSV not found: {tiktok_file}", file=sys.stderr)
        return 2
    if not combined_file.is_file():
        print(f"❌ Combined CSV not found: {combined_file}", file=sys.stderr)
        return 2

    combined_header, combined_index, platform_counts = build_combined_index(combined_file)

    instagram_columns: List[str] = []
    for csv_path in instagram_csvs:
        instagram_columns.extend(read_csv_header(csv_path))
    tiktok_columns = read_csv_header(tiktok_file)

    instagram_missing_cols, tiktok_missing_cols, llm_missing_cols = verify_columns(
        combined_header,
        instagram_columns,
        tiktok_columns,
    )

    instagram_total, instagram_matched, instagram_missing_rows, instagram_coverage = verify_source_rows(
        instagram_csvs,
        INSTAGRAM_KEY_FIELDS,
        combined_index["instagram"],
    )
    tiktok_total, tiktok_matched, tiktok_missing_rows, tiktok_coverage = verify_source_rows(
        (tiktok_file,),
        TIKTOK_KEY_FIELDS,
        combined_index["tiktok"],
    )

    passed = True

    instagram_combined_total = platform_counts.get("instagram", 0)
    instagram_pct = 100.0 * instagram_matched / instagram_total if instagram_total else 0.0
    print(
        f"Instagram rows: source={instagram_total} matched={instagram_matched} "
        f"combined={instagram_combined_total} ({instagram_pct:.2f}% coverage)"
    )
    if instagram_combined_total != instagram_total:
        passed = False
        print("❌ Instagram row count mismatch")

    tiktok_combined_total = platform_counts.get("tiktok", 0)
    tiktok_pct = 100.0 * tiktok_matched / tiktok_total if tiktok_total else 0.0
    print(
        f"TikTok rows: source={tiktok_total} matched={tiktok_matched} "
        f"combined={tiktok_combined_total} ({tiktok_pct:.2f}% coverage)"
    )
    if tiktok_combined_total != tiktok_total:
        passed = False
        print("❌ TikTok row count mismatch")

    if instagram_missing_cols:
        passed = False
        print("❌ Combined CSV is missing Instagram columns:")
        for column in instagram_missing_cols:
            print(f"   - {column}")

    if tiktok_missing_cols:
        passed = False
        print("❌ Combined CSV is missing TikTok columns:")
        for column in tiktok_missing_cols:
            print(f"   - {column}")

    if llm_missing_cols:
        passed = False
        print("❌ Combined CSV is missing LLM columns:")
        for column in llm_missing_cols:
            print(f"   - {column}")

    if instagram_missing_rows:
        passed = False
        describe_missing("Instagram", instagram_missing_rows, args.max_missing_preview)

    if tiktok_missing_rows:
        passed = False
        describe_missing("TikTok", tiktok_missing_rows, args.max_missing_preview)

    print_coverage("Instagram source coverage", instagram_coverage, instagram_combined_total)
    print_coverage("TikTok source coverage", tiktok_coverage, tiktok_combined_total)

    language_dir = args.language_dir.resolve()
    if language_dir.is_dir():
        instagram_language_csvs, instagram_language_missing = collect_language_csvs(
            instagram_csvs, "instagram", language_dir
        )
        tiktok_language_csvs, tiktok_language_missing = collect_language_csvs(
            (tiktok_file,), "tiktok", language_dir
        )

        if instagram_language_missing:
            print("⚠️ Missing Instagram language-filtered files:")
            for name in instagram_language_missing:
                print(f"   - {name}")
        if tiktok_language_missing:
            print("⚠️ Missing TikTok language-filtered files:")
            for name in tiktok_language_missing:
                print(f"   - {name}")

        if instagram_language_csvs:
            (
                _,
                _,
                instagram_lang_missing,
                instagram_lang_coverage,
            ) = verify_source_rows(
                instagram_language_csvs,
                INSTAGRAM_KEY_FIELDS,
                combined_index["instagram"],
            )
            if instagram_lang_missing:
                passed = False
                describe_missing(
                    "Instagram language-filtered",
                    instagram_lang_missing,
                    args.max_missing_preview,
                )
            print_coverage(
                "Instagram language-filtered coverage",
                instagram_lang_coverage,
                instagram_combined_total,
            )
        else:
            print("⚠️ No Instagram language-filtered CSVs to evaluate")

        if tiktok_language_csvs:
            (
                _,
                _,
                tiktok_lang_missing,
                tiktok_lang_coverage,
            ) = verify_source_rows(
                tiktok_language_csvs,
                TIKTOK_KEY_FIELDS,
                combined_index["tiktok"],
            )
            if tiktok_lang_missing:
                passed = False
                describe_missing(
                    "TikTok language-filtered",
                    tiktok_lang_missing,
                    args.max_missing_preview,
                )
            print_coverage(
                "TikTok language-filtered coverage",
                tiktok_lang_coverage,
                tiktok_combined_total,
            )
        else:
            print("⚠️ No TikTok language-filtered CSVs to evaluate")
    else:
        print(f"⚠️ Language-filter directory not found: {language_dir}")

    if not args.skip_parquet and args.parquet_out:
        export_parquet(combined_file, args.parquet_out.resolve())

    if passed:
        print("✅ Combined CSV contains all source data and required LLM columns")
        return 0

    print("❌ Verification failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
