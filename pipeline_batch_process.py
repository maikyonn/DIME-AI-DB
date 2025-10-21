#!/usr/bin/env python3
"""
Sequential OpenAI Batch pipeline with language filtering.

Step 0 performs language filtering so that only English profiles (or profiles
with very little text) proceed. Subsequent steps build batch input files and
submit them to the OpenAI Batch API one-at-a-time, downloading the results for
each chunk. Every source CSV keeps its own outputs under
`<dataset>/pipeline_outputs/<csv-relative-path>/stepX_*`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import hashlib
import statistics
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
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
    from lingua_language_detector import Language, LanguageDetectorBuilder
except ImportError:
    try:
        from lingua import Language, LanguageDetectorBuilder  # type: ignore
    except ImportError:
        from enum import Enum

        class _FallbackLanguage(Enum):
            ENGLISH = "ENGLISH"

        class _FallbackDetector:
            def detect_language_of(self, text: str):
                return _FallbackLanguage.ENGLISH

        class _FallbackBuilder:
            @staticmethod
            def from_all_languages():
                return _FallbackBuilder()

            def with_preloaded_language_models(self):
                return self

            def build(self):
                return _FallbackDetector()

        Language = _FallbackLanguage
        LanguageDetectorBuilder = _FallbackBuilder
from openai import OpenAI

# Text processing constants
CAPTION_SNIPPET_CHARS = 50
# Number of post captions sampled for language detection (biography + first 9 posts)
CAPTIONS_TO_INSPECT = 9
DEFAULT_MIN_TEXT_CHARS = 60

DEFAULT_LANGUAGE_BATCH_SIZE = 1500
DEFAULT_CSV_WORKERS = max(1, min(8, os.cpu_count() or 4))
LANGUAGE_FILTER_VERSION = "normalized-batching-v4"

# Combined dataset constants
COMBINED_SUBDIR = "combined"
COMBINED_FILENAME = "social_profiles.csv"

COMBINED_HEADERS = [
    "lance_db_id",
    "platform",
    "platform_id",
    "username",
    "display_name",
    "biography",
    "followers",
    "following",
    "posts_count",
    "likes_total",
    "engagement_rate",
    "external_url",
    "profile_url",
    "profile_image_url",
    "is_verified",
    "is_private",
    "is_commerce_user",
    "posts",
    "reel_post_ratio_last10",
    "median_view_count_last10",
    "median_like_count_last10",
    "median_comment_count_last10",
    "total_img_posts_ig",
    "total_reels_ig",
]

# Pipeline step choices
STEP_LANGUAGE = "language"
STEP_PREPARE = "prepare"
STEP_PROCESS = "process"
STEP_CHOICES = [STEP_LANGUAGE, STEP_PREPARE, STEP_PROCESS]
STEP_ORDER = {name: idx for idx, name in enumerate(STEP_CHOICES)}

CSV_FIELD_ORDER = [
    "lance_db_id",
    "individual_vs_org",
    "generational_appeal",
    "professionalization",
    "relationship_status",
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
]


def _parse_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "1", "yes"}:
            return "true"
        if stripped in {"false", "0", "no"}:
            return "false"
        return ""
    return ""


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return []


def _decode_text(value: str) -> str:
    if not value:
        return value
    try:
        escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        decoded = json.loads(f'"{escaped}"')
    except Exception:
        return value.strip()
    return decoded.strip()


def _normalize_posts(entries: Any, platform: str) -> str:
    posts: List[Dict[str, Any]] = []
    if isinstance(entries, str):
        entries = _safe_json_loads(entries)
    if not isinstance(entries, list):
        entries = []

    def _first(item: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in item and item[key] not in (None, "", []):
                return item[key]
        return None

    def _to_int(value: Any) -> Optional[int]:
        if value in (None, "", []):
            return None
        try:
            return int(float(value))
        except Exception:
            return None

    def _to_list(value: Any) -> Optional[List[Any]]:
        if value in (None, ""):
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # comma separated or JSON string fallbacks
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                parts = [part.strip() for part in value.split(",") if part.strip()]
                return parts or None
        return None

    for item in entries:
        if not isinstance(item, dict):
            continue

        post_id = _first(item, "id", "post_id", "aweme_id", "video_id")
        caption = _first(item, "caption", "desc", "title", "text", "description") or ""
        if isinstance(caption, str):
            caption = _decode_text(caption)
        like_count = _to_int(
            _first(item, "likes", "like_count", "diggCount", "diggcount", "collectCount")
        )
        favorite_count = _to_int(
            _first(item, "favorites_count", "favoriteCount", "collectCount")
        )
        comment_count = _to_int(
            _first(item, "comments", "comment_count", "commentCount", "commentcount")
        )
        share_count = _to_int(
            _first(item, "share_count", "shareCount", "forwardCount")
        )
        view_count = _to_int(
            _first(item, "view_count", "viewCount", "playCount", "playcount")
        )
        url = _first(
            item,
            "url",
            "videoUrl",
            "video_url",
            "share_url",
            "permalink",
            "post_url",
        )
        media_type = _first(
            item,
            "content_type",
            "media_type",
            "type",
            "post_type",
        ) or ("video" if platform == "tiktok" else "image")
        timestamp = _first(
            item,
            "datetime",
            "createTime",
            "create_time",
            "create_date",
            "published_at",
        )
        duration = _first(item, "duration", "videoDuration", "video_duration")
        hashtags_raw = _to_list(_first(item, "hashtags", "post_hashtags"))
        hashtags: List[str] = []
        if hashtags_raw:
            for tag in hashtags_raw:
                if isinstance(tag, str) and tag.strip():
                    clean_tag = tag.strip()
                    if clean_tag.startswith("#"):
                        clean_tag = clean_tag[1:]
                    if clean_tag:
                        hashtags.append(clean_tag)
        thumbnail_url = _first(
            item,
            "image_url",
            "thumbnail_url",
            "thumb_url",
            "cover_image",
        )

        location_name = ""
        if platform == "instagram":
            location_value = (
                item.get("location")
                or item.get("place")
                or item.get("location_name")
            )
            if isinstance(location_value, dict):
                location_name = str(
                    location_value.get("name")
                    or location_value.get("title")
                    or location_value.get("short_name")
                    or ""
                )
                location_name = _decode_text(location_name)
            elif isinstance(location_value, str):
                location_name = _decode_text(location_value)
            elif isinstance(location_value, list) and location_value:
                first_loc = location_value[0]
                if isinstance(first_loc, dict):
                    location_name = str(
                        first_loc.get("name")
                        or first_loc.get("title")
                        or first_loc.get("short_name")
                        or ""
                    )
                    location_name = _decode_text(location_name)

        if caption and hashtags:
            cleaned_caption = caption
            for tag in hashtags:
                pattern = re.compile(rf"(?i)(?<!\\w)#\s*{re.escape(tag)}\b")
                cleaned_caption = pattern.sub("", cleaned_caption)
            cleaned_caption = re.sub(r"\s+", " ", cleaned_caption).strip()
            caption = cleaned_caption
        elif caption:
            caption = re.sub(r"\s+", " ", caption).strip()

        mapped = {
            "platform": platform,
            "id": post_id,
            "caption": caption,
            "hashtags": hashtags or [],
            "like_count": like_count,
            "comment_count": comment_count,
            "share_count": share_count,
            "view_count": view_count,
            "favorite_count": favorite_count,
            "url": url,
            "media_type": media_type,
            "timestamp": timestamp,
            "duration": duration,
            "thumbnail_url": thumbnail_url,
            "location_name": location_name,
        }

        extra = {
            key: value
            for key, value in item.items()
            if key not in {
                "id",
                "post_id",
                "aweme_id",
                "video_id",
                "caption",
                "desc",
                "title",
                "text",
                "description",
                "likes",
                "like_count",
                "diggCount",
                "diggcount",
                "collectCount",
                "favorites_count",
                "favoriteCount",
                "comments",
                "comment_count",
                "commentCount",
                "commentcount",
                "share_count",
                "shareCount",
                "forwardCount",
                "view_count",
                "viewCount",
                "playCount",
                "playcount",
                "url",
                "videoUrl",
                "video_url",
                "share_url",
                "permalink",
                "post_url",
                "content_type",
                "media_type",
                "type",
                "post_type",
                "datetime",
                "createTime",
                "create_time",
                "create_date",
                "published_at",
                "duration",
                "videoDuration",
                "video_duration",
                "hashtags",
                "post_hashtags",
                "image_url",
                "thumbnail_url",
                "thumb_url",
                "cover_image",
            }
        }
        if extra:
            mapped["extra"] = extra

        posts.append(mapped)

    return json.dumps(posts, ensure_ascii=False)


def _merge_tiktok_posts(raw: Dict[str, str]) -> List[Dict[str, Any]]:
    videos = _safe_json_loads(raw.get("top_videos"))
    posts = _safe_json_loads(raw.get("top_posts_data"))

    def _as_id(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        return str(value)

    combined: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def _ensure_entry(identifier: str) -> Dict[str, Any]:
        if identifier not in combined:
            combined[identifier] = {"video_id": identifier}
            order.append(identifier)
        return combined[identifier]

    def _merge_items(items: Any, id_keys: Sequence[str]) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            identifier: Optional[str] = None
            for key in id_keys:
                identifier = _as_id(item.get(key))
                if identifier:
                    break
            if not identifier:
                continue
            entry = _ensure_entry(identifier)
            for key, value in item.items():
                if key not in entry or entry[key] in (None, "", []):
                    entry[key] = value

    _merge_items(posts, ("post_id", "video_id", "aweme_id"))
    _merge_items(videos, ("video_id", "post_id", "aweme_id"))

    return [combined[key] for key in order]


def _compute_post_statistics(posts_json: str) -> Tuple[str, str, str, str]:
    if not posts_json:
        return "", "", ""
    try:
        posts = json.loads(posts_json)
    except Exception:
        return "", "", ""
    if not isinstance(posts, list):
        return "", "", ""

    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        raw = value.strip()
        try:
            if raw.endswith("Z"):
                return datetime.fromisoformat(raw[:-1] + "+00:00")
            return datetime.fromisoformat(raw)
        except Exception:
            # best-effort fallback for common ISO formats without timezone
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(raw, fmt)
                except Exception:
                    continue
        return None

    enumerated: List[Tuple[int, Dict[str, Any], Optional[datetime]]] = []
    for idx, post in enumerate(posts):
        if isinstance(post, dict):
            enumerated.append((idx, post, _parse_timestamp(post.get("timestamp"))))

    with_timestamp = [item for item in enumerated if item[2] is not None]
    without_timestamp = [item for item in enumerated if item[2] is None]
    with_timestamp.sort(key=lambda item: item[2], reverse=True)
    without_timestamp.sort(key=lambda item: item[0])

    ordered_posts = [item[1] for item in with_timestamp + without_timestamp][:10]

    total = 0
    reel_like = 0
    view_values: List[int] = []
    like_values: List[int] = []
    comment_values: List[int] = []

    def _is_reel(media_type: Optional[str]) -> bool:
        if not media_type:
            return False
        lowered = media_type.lower()
        if "reel" in lowered:
            return True
        if "video" in lowered:
            return True
        if lowered in {"igtv", "graphvideo"}:
            return True
        return False

    for post in ordered_posts:
        if not isinstance(post, dict):
            continue
        total += 1
        media_type = post.get("media_type")
        if _is_reel(media_type if isinstance(media_type, str) else None):
            reel_like += 1

        like = post.get("like_count")
        if isinstance(like, (int, float)):
            like_values.append(int(like))
        view = post.get("view_count")
        if isinstance(view, (int, float)):
            view_values.append(int(view))
        comments = post.get("comment_count")
        if isinstance(comments, (int, float)):
            comment_values.append(int(comments))

    def _format_ratio(num: int, denom: int) -> str:
        if denom <= 0:
            return ""
        ratio = num / denom
        return f"{ratio:.3f}"

    def _format_median(values: List[int]) -> str:
        if not values:
            return ""
        med = statistics.median(values)
        if isinstance(med, float) and med.is_integer():
            return str(int(med))
        return f"{med:.3f}" if isinstance(med, float) else str(med)

    ratio = _format_ratio(reel_like, total)
    median_view = _format_median(view_values)
    median_like = _format_median(like_values)
    median_comment = _format_median(comment_values)

    return ratio, median_view, median_like, median_comment


def _count_instagram_media(posts_json: str) -> Tuple[str, str]:
    if not posts_json:
        return "", ""
    try:
        posts = json.loads(posts_json)
    except Exception:
        return "", ""
    if not isinstance(posts, list):
        return "", ""

    image_types = {"graphimage", "image", "photo", "graphsidecar"}
    reel_types = {"reel", "video", "graphvideo", "igtv"}

    total_images = 0
    total_reels = 0
    for post in posts:
        if not isinstance(post, dict):
            continue
        media_type = post.get("media_type")
        if isinstance(media_type, str):
            lowered = media_type.lower()
            if lowered in image_types:
                total_images += 1
            elif lowered in reel_types:
                total_reels += 1
            elif "video" in lowered or "reel" in lowered:
                total_reels += 1
            elif "image" in lowered or "photo" in lowered:
                total_images += 1

    return (str(total_images) if total_images else "", str(total_reels) if total_reels else "")


def combine_platform_datasets(root_dir: Path) -> Path:
    print(f"\n🔗 Combining platform datasets under {root_dir}")
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2_147_483_647)
    instagram_dir = root_dir / "instagram"
    tiktok_dir = root_dir / "tiktok"
    tiktok_file = tiktok_dir / "tiktok.csv"

    if not instagram_dir.exists() or not tiktok_file.exists():
        raise FileNotFoundError(
            "Expected 'instagram' directory and 'tiktok/tiktok.csv' under the provided path"
        )

    output_dir = root_dir / COMBINED_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / COMBINED_FILENAME

    used_ids: set[str] = set()
    max_numeric_id = 0
    instagram_count = 0
    tiktok_count = 0

    print(f"   📝 Streaming combined dataset to {output_file}")
    with output_file.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=COMBINED_HEADERS)
        writer.writeheader()

        for csv_file in sorted(instagram_dir.glob("*.csv")):
            print(f"   📥 Processing Instagram file: {csv_file.name}")
            with csv_file.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for raw in tqdm(reader, desc=f"Instagram {csv_file.name}", unit="rows"):
                    posts_json = _normalize_posts(_safe_json_loads(raw.get("posts")), "instagram")
                    reel_ratio, median_view, median_like, median_comment = _compute_post_statistics(posts_json)
                    total_images, total_reels = _count_instagram_media(posts_json)

                    external_raw = raw.get("external_url", "")
                    if isinstance(external_raw, str) and external_raw.strip().startswith("["):
                        try:
                            first_url = json.loads(external_raw)[0]
                        except Exception:
                            first_url = external_raw
                    else:
                        first_url = external_raw

                    lance_id = (raw.get("lance_db_id") or "").strip()
                    if not lance_id:
                        lance_id = str(instagram_count + 1)
                    if lance_id in used_ids:
                        candidate = instagram_count + 1
                        while str(candidate) in used_ids:
                            candidate += 1
                        lance_id = str(candidate)
                    used_ids.add(lance_id)
                    try:
                        max_numeric_id = max(max_numeric_id, int(lance_id))
                    except Exception:
                        pass

                    writer.writerow(
                        {
                            "lance_db_id": lance_id,
                            "platform": "instagram",
                            "platform_id": raw.get("fbid", ""),
                            "username": raw.get("account", ""),
                            "display_name": raw.get("profile_name") or raw.get("full_name", ""),
                            "biography": raw.get("biography", ""),
                            "followers": raw.get("followers", ""),
                            "following": raw.get("following", ""),
                            "posts_count": raw.get("posts_count", ""),
                            "likes_total": "",
                            "engagement_rate": raw.get("avg_engagement", ""),
                            "external_url": first_url,
                            "profile_url": raw.get("profile_url", ""),
                            "profile_image_url": raw.get("profile_image_link", ""),
                            "is_verified": _parse_bool(raw.get("is_verified")),
                            "is_private": _parse_bool(raw.get("is_private")),
                            "is_commerce_user": "false",
                            "posts": posts_json,
                            "reel_post_ratio_last10": reel_ratio,
                            "median_view_count_last10": median_view,
                            "median_like_count_last10": median_like,
                            "median_comment_count_last10": median_comment,
                            "total_img_posts_ig": total_images,
                            "total_reels_ig": total_reels,
                        }
                    )
                    instagram_count += 1

        starting_lance_id = max_numeric_id + 1 if max_numeric_id > 0 else instagram_count + 1
        print("   📥 Processing TikTok file: tiktok.csv")
        with tiktok_file.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for offset, raw in enumerate(tqdm(reader, desc="TikTok", unit="rows")):
                lance_id = str(starting_lance_id + offset)
                while lance_id in used_ids:
                    starting_lance_id += 1
                    lance_id = str(starting_lance_id + offset)
                used_ids.add(lance_id)

                merged_posts = _merge_tiktok_posts(raw)
                posts_json = _normalize_posts(merged_posts, "tiktok")
                reel_ratio, median_view, median_like, median_comment = _compute_post_statistics(posts_json)

                writer.writerow(
                    {
                        "lance_db_id": lance_id,
                        "platform": "tiktok",
                        "platform_id": raw.get("id", ""),
                        "username": raw.get("account_id", ""),
                        "display_name": raw.get("profile_name") or raw.get("nickname", ""),
                        "biography": raw.get("biography") or raw.get("signature", ""),
                        "followers": raw.get("followers", ""),
                        "following": raw.get("following", ""),
                        "posts_count": raw.get("videos_count", ""),
                        "likes_total": raw.get("likes", ""),
                        "engagement_rate": raw.get("awg_engagement_rate", ""),
                        "external_url": raw.get("bio_link", ""),
                        "profile_url": raw.get("url", ""),
                        "profile_image_url": raw.get("profile_pic_url_hd", raw.get("profile_pic_url", "")),
                        "is_verified": _parse_bool(raw.get("is_verified")),
                        "is_private": _parse_bool(raw.get("is_private")),
                        "is_commerce_user": _parse_bool(raw.get("is_commerce_user")),
                        "posts": posts_json,
                        "reel_post_ratio_last10": reel_ratio,
                        "median_view_count_last10": median_view,
                        "median_like_count_last10": median_like,
                        "median_comment_count_last10": median_comment,
                        "total_img_posts_ig": "",
                        "total_reels_ig": "",
                    }
                )
                tiktok_count += 1

    print(
        f"Combined {instagram_count} Instagram rows and {tiktok_count} TikTok rows → {output_file}"
    )
    return output_file


@dataclass
class ChunkInfo:
    index: int
    jsonl_path: Path
    row_count: int


@dataclass
class BatchJobRecord:
    chunk_number: int
    batch_id: str
    file_id: str
    profile_count: int
    submitted_at: float
    completed_at: Optional[float] = None
    output_file_id: Optional[str] = None
    status: str = "created"
    result_csv: Optional[str] = None


class SequentialBatchPipeline:
    @staticmethod
    def _make_namespace(name: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip())
        cleaned = cleaned.strip("_")
        return cleaned or "dataset"

    def __init__(
        self,
        csv_path: Path,
        chunk_size: int,
        language_batch_size: int,
        poll_interval: int,
        max_attempts: int,
        stop_after: Optional[str],
        resume_from: str,
        test_mode: bool,
        min_text_chars: int,
        prompt_file: Path,
        force: bool,
        dataset_namespace: Optional[str] = None,
    ) -> None:
        self.original_csv_path = csv_path.resolve()
        if not self.original_csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.dataset_dir = self.original_csv_path.parent
        self.dataset_namespace = self._make_namespace(dataset_namespace) if dataset_namespace else None
        if language_batch_size <= 0:
            raise ValueError("language_batch_size must be greater than 0")

        self.chunk_size = chunk_size
        self.language_batch_size = language_batch_size
        self.poll_interval = poll_interval
        self.max_attempts = max_attempts
        self.stop_after = stop_after
        self.resume_from = resume_from
        self.test_mode = test_mode
        self.min_text_chars = min_text_chars
        self.max_rows = 500 if test_mode else None
        self.force = force

        # Shared pipeline directories under project root
        self.project_root = Path(__file__).resolve().parent
        load_dotenv(self.project_root / ".env")
        load_dotenv()
        self.pipeline_root_dir = self.project_root / "pipeline"
        namespace = self.dataset_namespace or self._make_namespace(self.original_csv_path.stem)
        self.dataset_namespace = namespace
        self.namespace = namespace
        self.pipeline_dir = self.pipeline_root_dir
        self.language_dir = self.pipeline_dir / "step0_language_filter"
        self.batch_input_dir = self.pipeline_dir / "step1_batch_inputs"
        self.batch_results_dir = self.pipeline_dir / "step2_batch_results"
        for path in (
            self.pipeline_root_dir,
            self.pipeline_dir,
            self.language_dir,
            self.batch_input_dir,
            self.batch_results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.job_state_path = self.pipeline_dir / f"{self.namespace}_batch_jobs_state.json"
        if self.job_state_path.exists():
            try:
                with self.job_state_path.open("r", encoding="utf-8") as fh:
                    raw_state = json.load(fh)
            except json.JSONDecodeError:
                raw_state = {}
        else:
            raw_state = {}
        self.job_state: Dict[str, Dict[str, Any]] = {str(k): v for k, v in raw_state.items()}
        if force and self.job_state:
            self.job_state = {}
            self._persist_job_state()

        self.processed_files_path = self.pipeline_dir / f"{self.namespace}_processed_files.json"
        if self.processed_files_path.exists():
            try:
                with self.processed_files_path.open("r", encoding="utf-8") as fh:
                    self.processed_files = json.load(fh)
            except json.JSONDecodeError:
                self.processed_files = {}
        else:
            self.processed_files = {}

        self.source_csv_key = str(self.original_csv_path)
        self.file_hash = self._hash_file(self.original_csv_path)
        print("🧠 Initializing language detector")

        self.detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        )
        print("   ✅ Language detector ready (Lingua, preloaded models)")

        self.prompt_file_path = prompt_file.resolve()
        if not self.prompt_file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file_path}")
        self.prompt_file_name = self.prompt_file_path.name
        self.prompt_template = self.prompt_file_path.read_text(encoding="utf-8")

        self.client: Optional[OpenAI] = None
        self.language_pass_count: int = 0
        self.pending_batch_submissions: bool = False
        self.filtered_csv_path: Optional[Path] = None
        self.filtered_csv_with_ids: Optional[Path] = None
        self.chunk_infos: List[ChunkInfo] = []
        self.jobs: List[BatchJobRecord] = []

    # ------------------------------------------------------------------ helpers
    def _get_client(self) -> OpenAI:
        if self.client is None:
            print("⚙️  Initializing OpenAI client for batch processing")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Please add it to your environment or .env file."
                )
            self.client = OpenAI(api_key=api_key)
        return self.client

    def _build_language_sample(self, row: Dict[str, str]) -> str:
        biography = row.get("biography", "") or ""
        posts_raw = row.get("posts", "") or ""
        snippets: List[str] = []

        if posts_raw:
            try:
                posts = json.loads(posts_raw)
                if isinstance(posts, list):
                    for post in posts[:CAPTIONS_TO_INSPECT]:
                        caption = ""
                        if isinstance(post, dict):
                            caption = str(post.get("caption", ""))
                        elif isinstance(post, str):
                            caption = post
                        caption = caption.strip()
                        if caption:
                            snippets.append(caption[:CAPTION_SNIPPET_CHARS])
            except json.JSONDecodeError:
                pass

        text_parts = [biography.strip()] if biography.strip() else []
        text_parts.extend(snippets)
        return " ".join(text_parts).strip()

    def _normalize_row(self, row: Dict[str, str]) -> Dict[str, str]:
        normalized = dict(row)

        biography = normalized.get("biography")
        if isinstance(biography, str):
            normalized["biography"] = _decode_text(biography)

        platform_hint_candidates = (
            normalized.get("platform"),
            normalized.get("platform_type"),
            normalized.get("source_platform"),
            normalized.get("platform_name"),
        )
        platform_hint = next(
            (str(value).strip().lower() for value in platform_hint_candidates if value),
            "",
        )

        posts_raw = normalized.get("posts")
        if posts_raw not in (None, ""):
            try:
                normalized_posts = _normalize_posts(posts_raw, platform_hint or "generic")
            except Exception:
                normalized_posts = json.dumps(_safe_json_loads(posts_raw), ensure_ascii=False)
            normalized["posts"] = normalized_posts

            reel_ratio, median_view, median_like, median_comment = _compute_post_statistics(normalized_posts)
            normalized["reel_post_ratio_last10"] = reel_ratio
            normalized["median_view_count_last10"] = median_view
            normalized["median_like_count_last10"] = median_like
            normalized["median_comment_count_last10"] = median_comment

            if platform_hint == "instagram":
                total_images, total_reels = _count_instagram_media(normalized_posts)
            else:
                total_images, total_reels = "", ""
            normalized["total_img_posts_ig"] = total_images
            normalized["total_reels_ig"] = total_reels
        else:
            normalized["reel_post_ratio_last10"] = ""
            normalized["median_view_count_last10"] = ""
            normalized["median_like_count_last10"] = ""
            normalized["median_comment_count_last10"] = ""
            normalized["total_img_posts_ig"] = ""
            normalized["total_reels_ig"] = ""

        text_fields = [
            "profile_name",
            "full_name",
            "account",
            "business_category_name",
            "category_name",
            "external_url",
            "bio_hashtags",
            "business_email",
            "email_address",
            "location",
        ]
        for field in text_fields:
            value = normalized.get(field)
            if isinstance(value, str):
                normalized[field] = _decode_text(value)

        return normalized

    def _needs_language_detection(self, sample_text: str) -> bool:
        return bool(sample_text) and len(sample_text) >= self.min_text_chars

    def _should_keep_row(self, sample_text: str, detected_language: Optional[Language] = None) -> bool:
        if not self._needs_language_detection(sample_text):
            return True
        return detected_language == Language.ENGLISH

    def _write_csv(self, path: Path, header: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _language_output_paths(self) -> Dict[str, Path]:
        suffix = "_sample" if self.test_mode else ""
        prefix = self.namespace
        english_path = self.language_dir / f"{prefix}_english_profiles{suffix}.csv"
        rejected_path = self.language_dir / f"{prefix}_excluded_profiles{suffix}.csv"
        return {"english": english_path, "rejected": rejected_path}

    def _rel(self, path: Path) -> str:
        for base in (self.dataset_dir, self.pipeline_root_dir, self.project_root):
            try:
                return str(path.relative_to(base))
            except ValueError:
                continue
        return str(path)

    def _load_existing_filtered_csv(self) -> Path:
        paths = self._language_output_paths()
        english_path = paths["english"]
        if not english_path.exists():
            raise FileNotFoundError(
                "Cannot resume from language filtering: English profiles CSV not found."
            )
        rejected_path = paths["rejected"]
        if not rejected_path.exists():
            print(
                "⚠️  English CSV found but excluded CSV missing; continuing with English-only data."
            )
        self.filtered_csv_path = english_path
        self.filtered_csv_with_ids = self._ensure_lance_ids(english_path)
        self.language_pass_count = self._count_rows(self.filtered_csv_with_ids)
        return self.filtered_csv_with_ids

    @staticmethod
    def _count_rows(path: Path) -> int:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # skip header
            return sum(1 for _ in reader)

    def _record_job(self, job: BatchJobRecord) -> None:
        state = {
            "chunk_number": job.chunk_number,
            "batch_id": job.batch_id,
            "file_id": job.file_id,
            "profile_count": job.profile_count,
            "submitted_at": job.submitted_at,
            "completed_at": job.completed_at,
            "output_file_id": job.output_file_id,
            "status": job.status,
            "result_csv": job.result_csv,
            "prompt_file": self.prompt_file_name,
        }
        self.job_state[str(job.chunk_number)] = state
        self._persist_job_state()

    def _persist_job_state(self) -> None:
        with self.job_state_path.open("w", encoding="utf-8") as fh:
            json.dump(self.job_state, fh, indent=2, default=str)

    def _load_job(self, chunk_number: int) -> Optional[BatchJobRecord]:
        entry = self.job_state.get(str(chunk_number))
        if not entry:
            return None
        if entry.get("prompt_file") and entry.get("prompt_file") != self.prompt_file_name:
            return None
        return BatchJobRecord(
            chunk_number=entry.get("chunk_number", chunk_number),
            batch_id=entry.get("batch_id", ""),
            file_id=entry.get("file_id", ""),
            profile_count=entry.get("profile_count", 0),
            submitted_at=entry.get("submitted_at", 0.0),
            completed_at=entry.get("completed_at"),
            output_file_id=entry.get("output_file_id"),
            status=entry.get("status", "created"),
            result_csv=entry.get("result_csv"),
        )

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _save_processed_files(self) -> None:
        with self.processed_files_path.open("w", encoding="utf-8") as fh:
            json.dump(self.processed_files, fh, indent=2)

    def _mark_processed(self, stage: str, output_path: Optional[Path], total_rows: int) -> None:
        if self.test_mode:
            return
        entry = {
            "hash": self.file_hash,
            "prompt_file": self.prompt_file_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": stage,
            "output_path": str(output_path) if output_path else "",
            "rows": total_rows,
        }
        records = self.processed_files.setdefault(self.source_csv_key, [])
        records = [
            r for r in records
            if r.get("hash") != self.file_hash
            or r.get("prompt_file") != self.prompt_file_name
            or r.get("stage") != "process"
        ]
        records.append(entry)
        self.processed_files[self.source_csv_key] = records
        self._save_processed_files()

    def _already_processed(self) -> bool:
        records = self.processed_files.get(self.source_csv_key, [])
        for record in records:
            if (
                record.get("hash") == self.file_hash
                and record.get("prompt_file") == self.prompt_file_name
                and record.get("stage") == "process"
            ):
                print(
                    f"⚠️  {self.original_csv_path.name} already processed with prompt '{self.prompt_file_name}'. "
                    "Use --force or select a different prompt file to reprocess."
                )
                return True
        return False

    def _ensure_lance_ids(self, csv_path: Path) -> Path:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        needs_rewrite = False
        existing_ids: set[str] = set()
        if "lance_db_id" not in fieldnames:
            needs_rewrite = True
        else:
            for row in rows:
                current_id = str(row.get("lance_db_id", "")).strip()
                if not current_id or current_id in existing_ids:
                    needs_rewrite = True
                    break
                existing_ids.add(current_id)

        if not needs_rewrite:
            return csv_path

        base_name = csv_path.stem
        if base_name.endswith("_with_lance_id"):
            target_path = csv_path
        else:
            target_path = csv_path.with_name(base_name + "_with_lance_id.csv")

        new_fieldnames = list(fieldnames)
        if "lance_db_id" not in new_fieldnames:
            new_fieldnames.insert(0, "lance_db_id")

        with target_path.open("w", encoding="utf-8", newline="") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=new_fieldnames)
            writer.writeheader()
            for idx, row in enumerate(rows, start=1):
                updated = {key: row.get(key, "") for key in fieldnames}
                lance_id = f"{self.namespace}_{idx:06d}"
                updated["lance_db_id"] = lance_id
                writer.writerow(updated)

        return target_path

    def _iter_filtered_rows(self) -> Iterable[Dict[str, str]]:
        assert self.filtered_csv_with_ids is not None
        with self.filtered_csv_with_ids.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield row

    # ------------------------------------------------------------------ step 0
    def perform_language_filter(self) -> Path:
        print("\n🧪 Step 0: Language filtering")
        paths = self._language_output_paths()
        english_path = paths["english"]
        rejected_path = paths["rejected"]
        metadata_path = self.language_dir / "metadata.json"

        reuse_cached = False
        if not self.force and english_path.exists() and rejected_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                metadata = {}
            if (
                metadata.get("hash") == self.file_hash
                and metadata.get("version") == LANGUAGE_FILTER_VERSION
                and metadata.get("language_batch_size") == self.language_batch_size
            ):
                reuse_cached = True

        if reuse_cached:
            print("   ♻️  Using cached language filter outputs")
            english_rows = self._count_rows(english_path)
            rejected_rows = self._count_rows(rejected_path)
        else:
            print(
                "   🧹 Normalizing profiles and scanning in batches of "
                f"{self.language_batch_size} rows"
            )
            english_rows = 0
            rejected_rows = 0

            with (
                self.original_csv_path.open("r", encoding="utf-8", newline="") as fh,
                english_path.open("w", encoding="utf-8", newline="") as english_fh,
                rejected_path.open("w", encoding="utf-8", newline="") as rejected_fh,
            ):

                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    raise ValueError("Input CSV has no header row")
                header = list(reader.fieldnames)
                enrichment_fields = [
                    "reel_post_ratio_last10",
                    "median_view_count_last10",
                    "median_like_count_last10",
                    "median_comment_count_last10",
                    "total_img_posts_ig",
                    "total_reels_ig",
                ]
                for field in enrichment_fields:
                    if field not in header:
                        header.append(field)
                english_writer = csv.DictWriter(english_fh, fieldnames=header)
                rejected_writer = csv.DictWriter(rejected_fh, fieldnames=header)
                english_writer.writeheader()
                rejected_writer.writeheader()

                batch_rows: List[Dict[str, str]] = []
                batch_samples: List[str] = []
                batch_detection_inputs: List[str] = []
                batch_detection_positions: List[int] = []

                def _process_batch() -> None:
                    nonlocal english_rows, rejected_rows
                    if not batch_rows:
                        return
                    batch_total = len(batch_rows)
                    detected: Dict[int, Optional[Language]] = {}
                    if batch_detection_inputs:
                        print(
                            "   🔄 Language detection for "
                            f"{len(batch_detection_inputs)} of {batch_total} normalized rows"
                        )
                        languages = self.detector.detect_languages_in_parallel_of(batch_detection_inputs)
                        for idx_position, language in zip(batch_detection_positions, languages):
                            detected[idx_position] = language

                    for position, row in enumerate(batch_rows):
                        sample_text = batch_samples[position]
                        language = detected.get(position)
                        if self._should_keep_row(sample_text, language):
                            english_writer.writerow(row)
                            english_rows += 1
                        else:
                            rejected_writer.writerow(row)
                            rejected_rows += 1

                    batch_rows.clear()
                    batch_samples.clear()
                    batch_detection_inputs.clear()
                    batch_detection_positions.clear()

                progress_total = self.max_rows if self.max_rows else None
                scan_progress = tqdm(total=progress_total, desc="Language scan", unit="rows")

                try:
                    for idx, row in enumerate(reader, start=1):
                        if self.max_rows and idx > self.max_rows:
                            break

                        normalized_row = self._normalize_row(dict(row))
                        sample_text = self._build_language_sample(normalized_row)
                        batch_rows.append(normalized_row)
                        batch_samples.append(sample_text)
                        if self._needs_language_detection(sample_text):
                            batch_detection_positions.append(len(batch_rows) - 1)
                            batch_detection_inputs.append(sample_text)

                        scan_progress.update(1)
                        if len(batch_rows) >= self.language_batch_size:
                            current_batch_size = len(batch_rows)
                            print(
                                "   … Normalizing + detecting batch of "
                                f"{current_batch_size} rows (processed up to row {idx})"
                            )
                            _process_batch()

                    if batch_rows:
                        print(f"   … Final batch of {len(batch_rows)} rows")
                        _process_batch()
                finally:
                    if hasattr(scan_progress, "close"):
                        scan_progress.close()

            metadata = {
                "hash": self.file_hash,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "rows": english_rows,
                "version": LANGUAGE_FILTER_VERSION,
                "language_batch_size": self.language_batch_size,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))

        print(f"   ✅ English / low-text rows: {english_rows} saved to {self._rel(english_path)}")
        print(f"   🚫 Excluded rows: {rejected_rows} saved to {self._rel(rejected_path)}")

        if english_rows == 0:
            raise RuntimeError("No rows passed the language filter")

        self.language_pass_count = english_rows
        self.filtered_csv_path = english_path
        self.filtered_csv_with_ids = self._ensure_lance_ids(english_path)
        if self.filtered_csv_with_ids != english_path:
            print(f"   ⚙️  Added lance_db_id → {self._rel(self.filtered_csv_with_ids)}")
        return self.filtered_csv_with_ids

    # ------------------------------------------------------------------ step 1
    def prepare_batches(self) -> List[ChunkInfo]:
        assert self.filtered_csv_with_ids is not None
        print("\n📦 Step 1: Preparing batch input files")
        chunk_rows: List[Dict[str, str]] = []
        chunk_infos: List[ChunkInfo] = []

        with self.filtered_csv_with_ids.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            current_chunk = 1
            for row in reader:
                chunk_rows.append(row)
                if len(chunk_rows) == self.chunk_size:
                    info = self._write_chunk_jsonl(current_chunk, chunk_rows)
                    chunk_infos.append(info)
                    chunk_rows = []
                    current_chunk += 1

            if chunk_rows:
                info = self._write_chunk_jsonl(current_chunk, chunk_rows)
                chunk_infos.append(info)

        print(f"   ✅ Prepared {len(chunk_infos)} batch input file(s) in {self._rel(self.batch_input_dir)}")
        self.chunk_infos = chunk_infos
        return chunk_infos

    def _write_chunk_jsonl(self, chunk_index: int, rows: Sequence[Dict[str, str]]) -> ChunkInfo:
        jsonl_path = self.batch_input_dir / f"{self.namespace}_batch_{chunk_index:03d}.jsonl"
        metadata_path = self.batch_input_dir / f"{self.namespace}_batch_{chunk_index:03d}.metadata.json"

        if not self.force and jsonl_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                saved_rows = metadata.get("row_count")
                saved_prompt = metadata.get("prompt_file")
            except json.JSONDecodeError:
                saved_rows = None
                saved_prompt = None
            if saved_rows == len(rows) and saved_prompt == self.prompt_file_name:
                return ChunkInfo(index=chunk_index, jsonl_path=jsonl_path, row_count=len(rows))

        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                lance_db_id = str(row.get("lance_db_id", "")).strip()
                prompt = self._build_prompt(row)
                request = {
                    "custom_id": f"profile-{lance_db_id}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": "gpt-5-nano",
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        "text": {"format": {"type": "text"}, "verbosity": "medium"},
                        "reasoning": {"effort": "medium"},
                        "store": True,
                    },
                }
                fh.write(json.dumps(request) + "\n")

        metadata = {
            "row_count": len(rows),
            "source_csv": self.filtered_csv_with_ids.name if self.filtered_csv_with_ids else "",
            "prompt_file": self.prompt_file_name,
            "source_hash": self.file_hash,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return ChunkInfo(index=chunk_index, jsonl_path=jsonl_path, row_count=len(rows))

    def _build_prompt(self, row: Dict[str, str]) -> str:
        posts_raw = row.get("posts", "") or ""
        caption_location_pairs: List[str] = []
        location_summary: List[str] = []
        if posts_raw:
            try:
                posts = json.loads(posts_raw)
                if isinstance(posts, list):
                    for post in posts[:CAPTIONS_TO_INSPECT]:
                        caption = ""
                        location_name = ""
                        if isinstance(post, dict):
                            caption = str(post.get("caption", ""))
                            raw_location = post.get("location_name")
                            if not raw_location and isinstance(post.get("location"), dict):
                                raw_location = post.get("location", {}).get("name")
                            if isinstance(raw_location, str):
                                location_name = _decode_text(raw_location).strip()
                                if location_name and location_name not in location_summary:
                                    location_summary.append(location_name)
                        elif isinstance(post, str):
                            caption = post

                        caption = caption.strip()
                        if caption:
                            if location_name:
                                caption_location_pairs.append(
                                    f"Post: {caption} (Location: {location_name})"
                                )
                            else:
                                caption_location_pairs.append(
                                    f"Post: {caption} (Location: Unknown)"
                                )
            except json.JSONDecodeError:
                trimmed = posts_raw[:200]
                caption_location_pairs.append(f"Post: {trimmed} (Location: Unknown)")

        context = {
            "account": row.get("account", ""),
            "full_name": row.get("full_name", ""),
            "biography": row.get("biography", ""),
            "captions": " | ".join(caption_location_pairs),
            "post_locations": " | ".join(location_summary) if location_summary else "Unknown",
        }

        try:
            prompt = self.prompt_template.format(**context)
        except KeyError:
            prompt = self.prompt_template
        return prompt


    # ------------------------------------------------------------------ step 2
    def process_batches(self, chunk_infos: Sequence[ChunkInfo]) -> None:
        print("\n🚀 Step 2: Submitting batches sequentially and downloading results")

        any_submitted = False
        for info in chunk_infos:
            chunk_csv = self.batch_results_dir / f"{self.namespace}_batch_{info.index:03d}_chunk.csv"
            results_jsonl = self.batch_results_dir / f"{self.namespace}_batch_{info.index:03d}_results.jsonl"
            existing_job = None if self.force else self._load_job(info.index)

            if not self.force and chunk_csv.exists():
                print(f"   ⏭️  Batch {info.index:03d} already processed, skipping")
                if existing_job and existing_job.status != "completed":
                    existing_job.status = "completed"
                    existing_job.result_csv = str(chunk_csv)
                    self._record_job(existing_job)
                continue

            if existing_job and existing_job.status == "completed" and chunk_csv.exists():
                print(f"   ⏭️  Batch {info.index:03d} already completed")
                continue

            if existing_job and existing_job.status != "completed":
                print(f"   ♻️  Resuming batch {info.index:03d} (batch id {existing_job.batch_id})")
                existing_job.profile_count = existing_job.profile_count or info.row_count
                try:
                    job = self._resume_job(existing_job, info, results_jsonl, chunk_csv)
                except RuntimeError as exc:
                    print(f"      ⚠️ Previous batch failed ({exc}). Submitting a new batch.")
                    self.job_state.pop(str(info.index), None)
                    self._persist_job_state()
                else:
                    self.jobs.append(job)
                    continue

            jsonl_path = info.jsonl_path
            print(f"   📤 Uploading batch {info.index:03d} ({info.row_count} profiles)")
            job = self._upload_and_create_batch(jsonl_path, info.index)
            job.profile_count = info.row_count
            job.status = job.status or "submitted"
            self._record_job(job)
            self.jobs.append(job)
            any_submitted = True
            print(f"      📨 Submitted batch {info.index:03d} (batch id {job.batch_id}, status {job.status})")

        self.pending_batch_submissions = any_submitted
        if any_submitted:
            print("\n⏳ All batch jobs have been submitted. Re-run the pipeline later with --resume-from process to wait for completion and download results.")

    def _upload_and_create_batch(self, jsonl_path: Path, chunk_number: int) -> BatchJobRecord:
        client = self._get_client()
        with jsonl_path.open("rb") as fh:
            file_obj = client.files.create(file=fh, purpose="batch")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": f"Dataset chunk {chunk_number:03d}"},
        )

        print(f"      → Batch ID: {batch.id}")
        return BatchJobRecord(
            chunk_number=chunk_number,
            batch_id=batch.id,
            file_id=file_obj.id,
            profile_count=0,
            submitted_at=time.time(),
            status=batch.status,
        )

    def _wait_for_batch(self, job: BatchJobRecord) -> BatchJobRecord:
        client = self._get_client()
        attempts = 0
        while True:
            batch = client.batches.retrieve(job.batch_id)
            job.status = batch.status
            if batch.request_counts:
                try:
                    total = batch.request_counts.total
                    completed = batch.request_counts.completed
                    failed = batch.request_counts.failed
                except AttributeError:
                    counts = batch.request_counts
                    total = counts.get("total")
                    completed = counts.get("completed")
                    failed = counts.get("failed")
                print(f"      status={batch.status} total={total} completed={completed} failed={failed}")
            else:
                print(f"      status={batch.status}")

            if batch.status == "completed":
                job.completed_at = time.time()
                job.output_file_id = batch.output_file_id
                print("      ✅ Batch completed")
                return job

            if batch.status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"Batch {job.batch_id} ended with status {batch.status}")

            attempts += 1
            if attempts >= self.max_attempts:
                raise RuntimeError("Maximum polling attempts exceeded")

            wait_for = max(self.poll_interval, 30)
            print(f"      ⏳ Waiting {wait_for}s...")
            time.sleep(wait_for)

    def _download_results(self, job: BatchJobRecord, target_path: Path) -> None:
        if not job.output_file_id:
            raise RuntimeError("No output file id available for download")
        client = self._get_client()
        response = client.files.content(job.output_file_id)
        content = getattr(response, "text", None)
        if content is None:
            content = response.read().decode("utf-8")  # type: ignore[attr-defined]
        with target_path.open("w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"      💾 Results downloaded to {self._rel(target_path)}")

    def _resume_job(
        self,
        job: BatchJobRecord,
        info: ChunkInfo,
        results_jsonl: Path,
        chunk_csv: Path,
    ) -> BatchJobRecord:
        if not job.batch_id:
            raise RuntimeError("Missing batch identifier for resume")

        job.profile_count = job.profile_count or info.row_count

        try:
            job = self._wait_for_batch(job)
        finally:
            self._record_job(job)

        if not job.output_file_id:
            raise RuntimeError("Batch completed without output file id")

        if not results_jsonl.exists():
            self._download_results(job, results_jsonl)

        chunk_csv_path = self._process_results(info.index, results_jsonl, chunk_csv)
        job.result_csv = str(chunk_csv_path)
        job.status = "completed"
        self._record_job(job)
        print(f"      ✅ Saved parsed results to {self._rel(chunk_csv_path)}")
        return job

    def _process_results(self, chunk_index: int, results_path: Path, csv_target: Path) -> Path:
        output_rows: List[Dict[str, Optional[str]]] = []
        with results_path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    output_rows.append(
                        {
                            "lance_db_id": "",
                            "raw_response": line.strip(),
                            "processing_error": f"json_decode_error_line_{line_num}",
                            "source_batch": f"batch_{chunk_index:03d}",
                            "prompt_file": self.prompt_file_name,
                        }
                    )
                    continue

                custom_id = payload.get("custom_id", "")
                lance_db_id = custom_id.replace("profile-", "") if custom_id.startswith("profile-") else ""

                parsed: Dict[str, Optional[str]] = {
                    "lance_db_id": lance_db_id,
                    "raw_response": "",
                    "processing_error": "",
                    "source_batch": f"batch_{chunk_index:03d}",
                    "prompt_file": self.prompt_file_name,
                }

                if payload.get("response") and payload["response"].get("status_code") == 200:
                    response_body = payload["response"].get("body", {})
                    text_content = ""
                    for output in response_body.get("output", []):
                        if output.get("type") == "message":
                            for part in output.get("content", []):
                                if part.get("type") == "output_text":
                                    text_content = part.get("text", "")
                                    break
                    parsed.update(self._parse_response_text(text_content))
                    parsed["raw_response"] = text_content
                else:
                    error = payload.get("error") or payload.get("response", {}).get("status_code")
                    parsed["processing_error"] = f"api_error:{error}"
                    parsed["raw_response"] = json.dumps(payload)[:500]

                output_rows.append(parsed)

        with csv_target.open("w", encoding="utf-8", newline="") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=CSV_FIELD_ORDER)
            writer.writeheader()
            for row in output_rows:
                writer.writerow({key: row.get(key, "") for key in CSV_FIELD_ORDER})
        return csv_target

    def _parse_response_text(self, text: str) -> Dict[str, Optional[str]]:
        if not text:
            return {
                "individual_vs_org": None,
                "generational_appeal": None,
                "professionalization": None,
                "relationship_status": None,
                "location": "",
                "ethnicity": "",
                "age": "",
                "occupation": "",
                **{f"keyword{i}": "" for i in range(1, 11)},
                "processing_error": "empty_response",
            }

        candidate_line = None
        for line in text.strip().splitlines():
            if "," in line:
                candidate_line = line.strip()
                break
        if candidate_line is None:
            candidate_line = text.strip()

        reader = csv.reader([candidate_line])
        try:
            values = next(reader)
        except Exception:
            return {
                "individual_vs_org": None,
                "generational_appeal": None,
                "professionalization": None,
                "relationship_status": None,
                "location": "",
                "ethnicity": "",
                "age": "",
                **{f"keyword{i}": "" for i in range(1, 11)},
                "processing_error": "csv_parse_error",
            }

        if len(values) < 18:
            return {
                "individual_vs_org": None,
                "generational_appeal": None,
                "professionalization": None,
                "relationship_status": None,
                "location": "",
                "ethnicity": "",
                "age": "",
                "occupation": "",
                **{f"keyword{i}": "" for i in range(1, 11)},
                "processing_error": f"unexpected_value_count:{len(values)}",
            }

        def parse_score(raw: str) -> Optional[int]:
            raw = raw.strip()
            if not raw:
                return None
            try:
                val = int(round(float(raw)))
                return max(0, min(10, val))
            except ValueError:
                return None

        scores = [parse_score(values[i]) for i in range(4)]
        location = values[4].strip() or ""
        ethnicity = values[5].strip() or ""
        age = values[6].strip() or ""
        occupation = values[7].strip() or ""
        keywords = [v.strip() for v in values[8:18]]

        result = {
            "individual_vs_org": scores[0],
            "generational_appeal": scores[1],
            "professionalization": scores[2],
            "relationship_status": scores[3],
            "location": location,
            "ethnicity": ethnicity,
            "age": age,
            "occupation": occupation,
            "processing_error": "" if all(score is not None for score in scores) else "missing_scores",
        }
        for idx in range(10):
            result[f"keyword{idx + 1}"] = keywords[idx] if idx < len(keywords) else ""
        return result

    # ------------------------------------------------------------------ public
    def run(self) -> None:
        print(f"📁 Dataset directory: {self.dataset_dir}")
        print(f"📄 Input CSV: {self.original_csv_path}")
        if self.test_mode:
            print("⚠️ Test mode enabled: processing first 500 rows only")

        if not self.force and not self.test_mode and self._already_processed():
            return

        if STEP_ORDER[self.resume_from] <= STEP_ORDER[STEP_LANGUAGE]:
            filtered_csv = self.perform_language_filter()
        else:
            print("⏭️  Resuming from existing language-filter outputs")
            filtered_csv = self._load_existing_filtered_csv()

        language_pass_rows = getattr(self, "language_pass_count", 0)
        if self._should_stop(STEP_LANGUAGE):
            print("⏹️  Stopping after language filtering as requested")
            self._mark_processed("language", filtered_csv, language_pass_rows)
            return

        chunk_infos = self.prepare_batches()
        if self._should_stop(STEP_PREPARE):
            print("⏹️  Stopping after batch preparation as requested")
            rows_prepared = sum(info.row_count for info in chunk_infos)
            self._mark_processed("prepare", None, rows_prepared)
            return

        self.process_batches(chunk_infos)
        if self.pending_batch_submissions:
            return
        if self._should_stop(STEP_PROCESS):
            print("⏹️  Stopping after batch processing as requested")
            rows_processed = sum(info.row_count for info in chunk_infos)
            self._mark_processed("process", None, rows_processed)
            return

        rows_processed = sum(info.row_count for info in chunk_infos)
        self._mark_processed("process", None, rows_processed)
        print("\n🎉 Language filtering and batch processing completed")

    def _should_stop(self, step: str) -> bool:
        if self.stop_after is None:
            return False
        return STEP_ORDER[step] >= STEP_ORDER[self.stop_after]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential OpenAI batch pipeline with language filtering")
    parser.add_argument("csv", help="Path to the raw influencer CSV")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help="Profiles per batch chunk (default: 20000)",
    )
    parser.add_argument(
        "--language-batch-size",
        type=int,
        default=DEFAULT_LANGUAGE_BATCH_SIZE,
        help="Rows per language-detection batch (default: 1500)",
    )
    parser.add_argument(
        "--csv-workers",
        type=int,
        default=DEFAULT_CSV_WORKERS,
        help=(
            "Maximum number of CSV files to process in parallel when the input "
            "path is a directory (default: matches local CPU count)."
        ),
    )
    parser.add_argument("--poll-interval", type=int, default=300, help="Seconds between status checks (default: 300)")
    parser.add_argument("--max-attempts", type=int, default=1000, help="Maximum polling iterations (default: 1000)")
    parser.add_argument(
        "--stop-after",
        choices=STEP_CHOICES,
        help="Stop after the given step (language, prepare, process)",
    )
    parser.add_argument(
        "--resume-from",
        choices=STEP_CHOICES,
        default=STEP_LANGUAGE,
        help="Resume processing at the given step (default: language)",
    )
    parser.add_argument("--test", action="store_true", help="Process only the first 500 rows for a test run")
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=DEFAULT_MIN_TEXT_CHARS,
        help="Minimum characters required before language detection applies (default: 60)",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompts/current_prompt.txt",
        help="Path to the prompt template file (default: prompts/current_prompt.txt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if this CSV has already been handled with the same prompt",
    )
    parser.add_argument(
        "--combine-platforms",
        action="store_true",
        help="Treat the provided path as a dataset root and combine instagram/tiktok CSVs before processing",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    print("\n🚀 Starting pipeline_batch_process")
    print(
        f"   • chunk_size={args.chunk_size}\n"
        f"   • language_batch_size={args.language_batch_size}\n"
        f"   • csv_workers={args.csv_workers}\n"
        f"   • poll_interval={args.poll_interval}s\n"
        f"   • max_attempts={args.max_attempts}\n"
        f"   • resume_from={args.resume_from}\n"
        f"   • stop_after={args.stop_after or 'auto'}\n"
        f"   • test_mode={'on' if args.test else 'off'}\n"
        f"   • min_text_chars={args.min_text_chars}\n"
        f"   • force={'on' if args.force else 'off'}\n"
        f"   • combine_platforms={'on' if args.combine_platforms else 'off'}"
    )
    csv_input = Path(args.csv)
    input_csvs: List[Path] = []
    base_dir: Optional[Path] = None

    if args.combine_platforms:
        print("   🔀 Combining Instagram and TikTok datasets before processing")
        if not csv_input.exists() or not csv_input.is_dir():
            print(
                "❌ When using --combine-platforms, the provided path must be a directory containing instagram/ and tiktok/ subfolders"
            )
            return 1
        try:
            combined_path = combine_platform_datasets(csv_input)
        except Exception as exc:
            print(f"❌ Failed to combine platform datasets: {exc}")
            return 1
        resolved = combined_path.resolve()
        input_csvs = [resolved]
        base_dir = resolved.parent
    else:
        if not csv_input.exists():
            print(f"❌ CSV path not found: {csv_input}")
            return 1
        if csv_input.is_dir():
            base_dir = csv_input.resolve()
            input_csvs = sorted(
                path.resolve()
                for path in base_dir.rglob("*.csv")
                if path.is_file()
            )
            if not input_csvs:
                print(f"❌ No CSV files found under directory: {csv_input}")
                return 1
            print(
                f"   📦 Detected {len(input_csvs)} CSV file(s) under {csv_input} (recursive). "
                f"Processing with up to {args.csv_workers} worker(s)."
            )
        else:
            resolved = csv_input.resolve()
            input_csvs = [resolved]
            base_dir = resolved.parent
            print("   📄 Using provided CSV without combining")

    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_absolute():
        candidate = Path(__file__).resolve().parent / prompt_path
        if candidate.exists():
            prompt_path = candidate

    if len(input_csvs) == 1:
        print(f"   📂 Resolved CSV path: {input_csvs[0]}")
    else:
        print(f"   📂 CSV directory: {csv_input.resolve()}")
    print(f"   🗒️  Prompt file: {prompt_path}")

    target_resume = args.resume_from
    target_stop = args.stop_after or STEP_PREPARE
    if STEP_ORDER[target_stop] < STEP_ORDER[target_resume]:
        target_stop = target_resume

    def _derive_namespace(csv_path: Path) -> Optional[str]:
        if len(input_csvs) == 1 and not args.combine_platforms:
            return None
        if base_dir:
            try:
                relative = csv_path.relative_to(base_dir)
            except ValueError:
                relative = Path(csv_path.name)
        else:
            relative = Path(csv_path.name)
        relative_str = str(relative).replace(os.sep, "_")
        return SequentialBatchPipeline._make_namespace(relative_str)

    def _run_pipeline_for_csv(csv_path: Path) -> None:
        namespace = _derive_namespace(csv_path)
        pipeline = SequentialBatchPipeline(
            csv_path=csv_path,
            chunk_size=args.chunk_size,
            language_batch_size=args.language_batch_size,
            poll_interval=args.poll_interval,
            max_attempts=args.max_attempts,
            stop_after=target_stop,
            resume_from=target_resume,
            test_mode=args.test,
            min_text_chars=args.min_text_chars,
            prompt_file=prompt_path,
            force=args.force,
            dataset_namespace=namespace,
        )
        print(f"\n▶️  Executing pipeline for {csv_path.name}...")
        pipeline.run()

    try:
        if len(input_csvs) > 1:
            errors: List[Tuple[Path, Exception]] = []
            max_workers = max(1, args.csv_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_run_pipeline_for_csv, path): path for path in input_csvs}
                for future in as_completed(future_map):
                    csv_path = future_map[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"\n❌ Pipeline failed for {csv_path}: {exc}")
                        errors.append((csv_path, exc))
            if errors:
                return 1
        else:
            _run_pipeline_for_csv(input_csvs[0])
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user. Re-run the same command to resume.")
        return 130
    except Exception as exc:
        print(f"\n❌ Pipeline failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
