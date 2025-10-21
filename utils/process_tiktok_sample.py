#!/usr/bin/env python3
"""Quick sanity check for TikTok pipeline transforms on a single row."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from pipeline_batch_process import (
    _compute_post_statistics,
    _merge_tiktok_posts,
    _normalize_posts,
    _safe_json_loads,
)
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


def _extract_keywords_from_row(row: Dict[str, Any]) -> List[str]:
    keywords: List[str] = []
    for idx in range(1, 11):
        value = row.get(f"keyword{idx}")
        if isinstance(value, str):
            value = value.strip()
        if value:
            keywords.append(str(value))
    return keywords


def _build_keyword_text(row: Dict[str, Any]) -> str:
    return " ".join(_extract_keywords_from_row(row))


def _build_profile_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("biography", "occupation", "location"):
        value = row.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value:
            parts.append(str(value))
    keywords = _extract_keywords_from_row(row)
    if keywords:
        parts.append(" ".join(keywords))
    return " ".join(parts)


def _extract_captions(row: Dict[str, Any], limit: int = 10) -> List[str]:
    raw_posts = row.get("posts")
    if not raw_posts:
        return []
    try:
        posts = json.loads(raw_posts) if isinstance(raw_posts, str) else raw_posts
    except json.JSONDecodeError:
        return []
    captions: List[str] = []
    if isinstance(posts, list):
        for post in posts[:limit]:
            if isinstance(post, dict):
                caption = post.get("caption")
                if isinstance(caption, str):
                    caption = caption.strip()
                    if caption:
                        captions.append(caption)
    return captions


def _build_content_text(row: Dict[str, Any]) -> str:
    captions = _extract_captions(row)
    return " ".join(captions)


def _load_row(csv_path: Path, index: int) -> Tuple[Dict[str, Any], List[str]]:
    if index < 0:
        raise ValueError("row index must be non-negative")
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if idx == index:
                fieldnames = reader.fieldnames or list(row.keys())
                return row, fieldnames
    raise IndexError(f"CSV has fewer than {index + 1} rows")


def _summarize_posts(posts_json: str) -> Tuple[int, List[Dict[str, Any]]]:
    posts = _safe_json_loads(posts_json)
    if not isinstance(posts, list):
        return 0, []
    preview = []
    for item in posts[:3]:
        if not isinstance(item, dict):
            continue
        preview.append(
            {
                "id": item.get("id") or item.get("video_id") or item.get("post_id"),
                "caption": item.get("caption", "")[:120],
                "like_count": item.get("like_count"),
                "comment_count": item.get("comment_count"),
                "share_count": item.get("share_count"),
                "view_count": item.get("view_count"),
                "media_type": item.get("media_type"),
            }
        )
    return len(posts), preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a TikTok row through the pipeline helpers")
    parser.add_argument(
        "--dataset",
        default="data/tiktok/tiktok.csv",
        help="Path to the TikTok CSV (default: data/tiktok/tiktok.csv)",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Zero-based index of the row to inspect (default: 0)",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Generate embedding vectors for the sample texts",
    )
    args = parser.parse_args()

    csv_path = Path(args.dataset).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"TikTok CSV not found: {csv_path}")

    source_row, fieldnames = _load_row(csv_path, args.row)

    merged_posts = _merge_tiktok_posts(source_row)
    normalized_posts = _normalize_posts(merged_posts, "tiktok")
    stats = _compute_post_statistics(normalized_posts)

    row_for_text = dict(source_row)
    row_for_text["posts"] = normalized_posts

    keyword_text = _build_keyword_text(row_for_text)
    profile_text = _build_profile_text(row_for_text)
    content_text = _build_content_text(row_for_text)
    captions = _extract_captions(row_for_text)

    post_count, post_preview = _summarize_posts(normalized_posts)

    numbered_columns = [
        {
            "index": idx,
            "name": name,
            "value": source_row.get(name),
        }
        for idx, name in enumerate(fieldnames)
    ]

    result: Dict[str, Any] = {
        "row_index": args.row,
        "lance_db_id": source_row.get("lance_db_id"),
        "platform_id": source_row.get("id"),
        "username": source_row.get("account_id"),
        "display_name": source_row.get("profile_name") or source_row.get("nickname"),
        "followers": source_row.get("followers"),
        "following": source_row.get("following"),
        "videos_count": source_row.get("videos_count"),
        "likes": source_row.get("likes"),
        "post_stats": {
            "reel_post_ratio_last10": stats[0],
            "median_view_count_last10": stats[1],
            "median_like_count_last10": stats[2],
            "median_comment_count_last10": stats[3],
            "post_count": post_count,
        },
        "texts": {
            "keyword_text": keyword_text,
            "profile_text": profile_text,
            "content_text_preview": content_text[:200],
            "content_caption_count": len(captions),
        },
        "posts_preview": post_preview,
        "source_columns": numbered_columns,
    }

    if args.with_embeddings:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for --with-embeddings; install the package first."
            )

        model = SentenceTransformer("google/embeddinggemma-300m")
        text_blocks: List[Tuple[str, str]] = []
        if profile_text.strip():
            text_blocks.append(("profile_embedding", profile_text))
        if keyword_text.strip():
            text_blocks.append(("keyword_embedding", keyword_text))
        if content_text.strip():
            text_blocks.append(("content_embedding", content_text))

        embeddings: Dict[str, Any] = {}
        if text_blocks:
            labels, payloads = zip(*text_blocks)
            vectors = model.encode(
                list(payloads),
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for label, vector in zip(labels, vectors):
                preview = vector[:5].tolist()
                embeddings[label] = {
                    "dimension": len(vector),
                    "preview": preview,
                }
        result["embeddings"] = embeddings

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
