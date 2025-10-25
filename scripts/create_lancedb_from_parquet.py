#!/usr/bin/env python3
"""
Create a LanceDB table from `normalized_profiles.parquet` via a staged pipeline.

Stages:
  1. Load parquet into a Polars DataFrame.
  2. Build normalized facet records (profile/posts) from each profile row.
  3. Fit a TF-IDF vectorizer.
  4. Persist the vectorizer artifact and add per-record sparse TF-IDF features.
  5. Fetch embeddings from DeepInfra asynchronously (batch size 256, <=190 concurrent requests) using aiometer with retries.
  6. Build an Arrow schema from the enriched records.
  7. Write records into LanceDB.

This script only uses DeepInfra for embeddings. Set `DEEPINFRA_API_KEY` or pass
`--deepinfra-api-key`. The embedding batch size is fixed at 256 per API request.
"""
import argparse
import asyncio
import functools
import gc
import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiometer
import joblib
import lancedb
import numpy as np
import pyarrow as pa
import polars as pl
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

try:
    from cuml.feature_extraction.text import TfidfVectorizer as CuMLTfidfVectorizer
    import cupy as cp
    import cudf
    _HAS_CUML = True
except Exception:  # pragma: no cover - optional dependency guard
    CuMLTfidfVectorizer = None  # type: ignore
    cp = None  # type: ignore
    cudf = None  # type: ignore
    _HAS_CUML = False

LOGGER = logging.getLogger("create_lancedb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

EMBED_BATCH_SIZE = 256
MAX_CONCURRENT_REQUESTS = 190
MAX_RETRY_ATTEMPTS = 5

# Load environment variables from .env if present (e.g., DEEPINFRA_API_KEY).
load_dotenv()


@dataclass(frozen=True)
class PipelineConfig:
    parquet_path: str
    db_uri: str
    table: str
    recreate: bool
    sample_rows: Optional[int]
    text_trunc: int
    posts_max: int
    tfidf_max_features: int
    tfidf_min_df: int
    ngram_range: Tuple[int, int]
    tfidf_backend: str
    tfidf_workers: int
    vectorizer_path: str
    embed_model: str
    embed_concurrency: int
    deepinfra_api_key: str
    deepinfra_endpoint: str


class DeepInfraEmbedder:
    """Async client for DeepInfra's OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        endpoint: str,
        concurrency: int,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise ValueError("DeepInfra API key is required; set DEEPINFRA_API_KEY or use --deepinfra-api-key")
        self.endpoint = endpoint.rstrip("/")
        self.concurrency = max(1, min(concurrency, MAX_CONCURRENT_REQUESTS))
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.endpoint)

    async def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        if not texts:
            return []

        batches: List[Tuple[int, List[str]]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = [str(t) for t in texts[start:start + EMBED_BATCH_SIZE]]
            batches.append((start, batch))

        results: List[Optional[np.ndarray]] = [None] * len(texts)

        async def run_batch(start: int, batch_texts: List[str]) -> None:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
                wait=wait_exponential_jitter(initial=1, max=10),
                retry=retry_if_exception_type(
                    (
                        APIError,
                        APITimeoutError,
                        APIConnectionError,
                        RateLimitError,
                        RuntimeError,
                    )
                ),
            ):
                with attempt:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts,
                        encoding_format="float",
                    )

            items = response.data or []
            if len(items) != len(batch_texts):
                raise RuntimeError(
                    f"Embedding count mismatch (expected {len(batch_texts)}, got {len(items)})"
                )
            for offset, item in enumerate(items):
                embedding = item.embedding
                if embedding is None:
                    raise RuntimeError("Missing embedding in DeepInfra response")
                results[start + offset] = np.asarray(embedding, dtype=np.float32)

        max_concurrency = min(self.concurrency, len(batches)) if batches else 1
        await aiometer.run_all(
            [functools.partial(run_batch, start, batch) for start, batch in batches],
            max_at_once=max_concurrency,
        )

        missing = [idx for idx, vector in enumerate(results) if vector is None]
        if missing:
            raise RuntimeError(f"Missing embeddings for indices: {missing[:5]} ...")
        return [vector for vector in results if vector is not None]


def booly(x: Any) -> Optional[bool]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return None


def inty(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None


def floaty(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None


def coalesce(*vals, default="") -> str:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return default


def clean_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def normalize_field_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return value


def build_profile_text(row: Dict[str, Any], text_trunc: int, include_keywords: bool = True) -> str:
    parts: List[str] = []
    parts.append(coalesce(row.get("display_name"), row.get("username")))
    occ = clean_text(row.get("occupation"))
    if occ:
        parts.append(occ)
    bio = clean_text(row.get("biography"))
    if bio:
        parts.append(bio)
    if include_keywords:
        for i in range(1, 11):
            ki = clean_text(row.get(f"keyword{i}"))
            if ki:
                parts.append(ki)
    txt = " • ".join([p for p in parts if p])
    return txt[:text_trunc] if text_trunc and text_trunc > 0 else txt


def _normalize_hashtags(raw: Any) -> List[str]:
    tags: List[str] = []

    def add_token(token: str) -> None:
        cleaned = clean_text(token).replace("#", " ")
        for part in cleaned.split():
            part = part.strip("# ")
            if part:
                tags.append(part)

    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return tags

    if isinstance(raw, str):
        add_token(raw)
    elif isinstance(raw, (list, tuple, set)):
        for entry in raw:
            if isinstance(entry, (list, tuple, set)):
                tags.extend(_normalize_hashtags(entry))
            else:
                add_token(str(entry))
    elif isinstance(raw, dict):
        for value in raw.values():
            tags.extend(_normalize_hashtags(value))
    else:
        add_token(str(raw))

    return tags


def extract_posts_chunks(posts_field: Any, posts_max: int = 5, snippet_max_len: Optional[int] = None) -> List[str]:
    if posts_field is None or (isinstance(posts_field, float) and math.isnan(posts_field)):
        return []

    parsed: Any = None
    if isinstance(posts_field, (list, tuple, dict)):
        parsed = posts_field
    else:
        s = str(posts_field).strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except Exception:
            text = clean_text(s)
            if snippet_max_len is not None and snippet_max_len > 0:
                text = text[:snippet_max_len]
            return [text]

    def add_caption(base_text: Optional[str], hashtags: List[str]) -> Optional[str]:
        snippet = clean_text(base_text) if base_text else ""
        if snippet_max_len is not None and snippet_max_len > 0 and snippet:
            snippet = snippet[:snippet_max_len]
        if hashtags:
            snippet = f"{snippet} {' '.join(hashtags)}".strip()
        return snippet or (" ".join(hashtags) if hashtags else None)

    captions: List[str] = []

    if isinstance(parsed, list):
        for item in parsed[:posts_max]:
            hashtags: List[str] = []
            text_part: Optional[str] = None
            if isinstance(item, dict):
                text_part = item.get("caption") or item.get("text") or item.get("title")
                extra = item.get("extra")
                if not text_part and isinstance(extra, dict):
                    text_part = extra.get("caption")
                for key in ("hashtags", "post_hashtags", "tags"):
                    hashtags.extend(_normalize_hashtags(item.get(key)))
            else:
                text_part = str(item)
            snippet = add_caption(text_part, hashtags)
            if snippet:
                captions.append(snippet)
    elif isinstance(parsed, dict):
        if "captions" in parsed and isinstance(parsed["captions"], list):
            for candidate in parsed["captions"][:posts_max]:
                hashtags = []
                text_part: Optional[str] = None
                if isinstance(candidate, dict):
                    text_part = candidate.get("text") or candidate.get("caption")
                    for key in ("hashtags", "post_hashtags", "tags"):
                        hashtags.extend(_normalize_hashtags(candidate.get(key)))
                else:
                    text_part = str(candidate)
                snippet = add_caption(text_part, hashtags)
                if snippet:
                    captions.append(snippet)
        else:
            for key, value in parsed.items():
                if "caption" not in key.lower() or not value:
                    continue
                snippet = add_caption(str(value), [])
                if snippet:
                    captions.append(snippet)

    return [c for c in captions if c]


def batched(iterable: Iterable[Any], n: int) -> Iterable[List[Any]]:
    buf: List[Any] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def load_dataframe(parquet_path: str, sample_rows: Optional[int] = None) -> pl.DataFrame:
    df = pl.read_parquet(parquet_path)
    if sample_rows and sample_rows > 0 and sample_rows < df.height:
        df = df.sample(n=sample_rows, shuffle=True, seed=42)
    return df


def make_rows(df: pl.DataFrame, text_trunc: int, posts_max: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    columns = df.columns
    for row in df.iter_rows(named=True):
        profile_text = build_profile_text(row, text_trunc=text_trunc)
        posts_chunks = extract_posts_chunks(row.get("posts"), posts_max=posts_max) if "posts" in row else []
        posts_text = " \n ".join(posts_chunks)

        lance_db_id = coalesce(row.get("lance_db_id"), row.get("platform_id"), row.get("username"))
        row_values = {col: normalize_field_value(row.get(col)) for col in columns}

        meta_common = dict(
            lance_db_id=lance_db_id,
            platform=coalesce(row.get("platform")),
            platform_id=coalesce(row.get("platform_id")),
            username=coalesce(row.get("username")),
            display_name=coalesce(row.get("display_name")),
            biography=coalesce(row.get("biography")),
            external_url=coalesce(row.get("external_url")),
            profile_url=coalesce(row.get("profile_url")),
            profile_image_url=coalesce(row.get("profile_image_url")),
            followers=inty(row.get("followers")),
            following=inty(row.get("following")),
            likes_total=inty(row.get("likes_total")),
            posts_count=inty(row.get("posts_count")),
            engagement_rate=floaty(row.get("engagement_rate")),
            median_view_count_last10=floaty(row.get("median_view_count_last10")),
            median_like_count_last10=floaty(row.get("median_like_count_last10")),
            median_comment_count_last10=floaty(row.get("median_comment_count_last10")),
            reel_post_ratio_last10=floaty(row.get("reel_post_ratio_last10")),
            total_img_posts_ig=inty(row.get("total_img_posts_ig")),
            total_reels_ig=inty(row.get("total_reels_ig")),
            individual_vs_org_score=floaty(row.get("individual_vs_org_score")),
            generational_appeal_score=floaty(row.get("generational_appeal_score")),
            professionalization_score=floaty(row.get("professionalization_score")),
            relationship_status_score=floaty(row.get("relationship_status_score")),
            occupation=coalesce(row.get("occupation")),
            is_verified=booly(row.get("is_verified")),
            is_private=booly(row.get("is_private")),
            is_commerce_user=booly(row.get("is_commerce_user")),
            source_batch=coalesce(row.get("source_batch")),
            llm_processed=booly(row.get("llm_processed")),
            source_csv=coalesce(row.get("source_csv")),
            prompt_file=coalesce(row.get("prompt_file")),
            raw_response=coalesce(row.get("raw_response")),
            processing_error=coalesce(row.get("processing_error")),
        )

        for col, value in row_values.items():
            if col not in meta_common and col not in {"vector_id", "content_type", "text"}:
                meta_common[col] = value

        if profile_text:
            rows.append(
                {
                    "vector_id": f"{lance_db_id}::profile",
                    "content_type": "profile",
                    "text": profile_text,
                    **meta_common,
                }
            )
        if posts_text:
            rows.append(
                {
                    "vector_id": f"{lance_db_id}::posts",
                    "content_type": "posts",
                    "text": posts_text,
                    **meta_common,
                    "_post_chunks": posts_chunks,
                }
            )
    return rows


def fit_tfidf(
    texts: List[str],
    max_features: int,
    min_df: int,
    ngram_range: Tuple[int, int],
    backend: str,
):
    if backend == "cuml":
        if not _HAS_CUML:
            raise RuntimeError("cuML requested but unavailable")
        vec = CuMLTfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
        )
        texts = cudf.Series(texts, dtype="str")
    else:
        vec = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
        )
        texts = [str(t) for t in texts]
    vec.fit(texts)
    return vec


def add_sparse_fields(
    records: List[Dict[str, Any]],
    vectorizer,
    batch_size: int = 4096,
    workers: int = 1,
    backend: str = "sklearn",
) -> None:
    use_cuml = (
        backend == "cuml"
        and _HAS_CUML
        and CuMLTfidfVectorizer is not None
        and isinstance(vectorizer, CuMLTfidfVectorizer)
    )
    if use_cuml:
        workers = 1

    def ranges() -> Iterable[Tuple[int, int]]:
        for start in range(0, len(records), batch_size):
            end = min(start + batch_size, len(records))
            yield start, end

    def process_batch(bounds: Tuple[int, int]):
        start, end = bounds
        batch_texts = [records[i]["text"] for i in range(start, end)]
        if use_cuml:
            batch_series = cudf.Series(batch_texts, dtype="str")
            X = vectorizer.transform(batch_series)
            X = X.get()
        else:
            batch_texts = [str(t) for t in batch_texts]
            X = vectorizer.transform(batch_texts)
        if use_cuml:
            cp.get_default_memory_pool().free_all_blocks()
        return start, end, X

    if workers is None or workers < 1:
        workers = 1

    if workers == 1:
        iterator = map(process_batch, ranges())
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        iterator = executor.map(process_batch, ranges())

    try:
        for start, end, X in iterator:
            for i, row in enumerate(range(start, end)):
                csr = X[i]
                indices = csr.indices.astype(np.int32)
                values = csr.data.astype(np.float32)
                records[row]["sparse_indices"] = indices.tolist()
                records[row]["sparse_values"] = values.tolist()
            del X
            gc.collect()
    finally:
        if workers > 1 and "executor" in locals():
            executor.shutdown(wait=True)


def infer_arrow_type(value) -> pa.DataType:
    if value is None:
        return pa.null()
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float32()
    if isinstance(value, list):
        if not value:
            return pa.list_(pa.float32())
        if all(isinstance(x, (float, np.floating)) for x in value):
            return pa.list_(pa.float32())
        if all(isinstance(x, (int, np.integer)) for x in value):
            return pa.list_(pa.int32())
        return pa.list_(pa.string())
    return pa.string()


def infer_arrow_type_from_records(records: List[Dict[str, Any]], key: str) -> pa.DataType:
    for record in records:
        value = record.get(key)
        dtype = infer_arrow_type(value)
        if not pa.types.is_null(dtype):
            return dtype
    return pa.null()


def build_schema_from_records(records: List[Dict[str, Any]]) -> pa.Schema:
    if not records:
        raise ValueError("No records available to infer schema")

    fields = []
    vector_dim = len(records[0].get("embedding", []))
    core_types = {
        "vector_id": pa.string(),
        "content_type": pa.string(),
        "text": pa.string(),
        "biography": pa.string(),
        "embedding": pa.list_(pa.float32(), vector_dim) if vector_dim else pa.list_(pa.float32()),
        "sparse_indices": pa.list_(pa.int32()),
        "sparse_values": pa.list_(pa.float32()),
    }

    first = records[0]
    for key in first.keys():
        if key in core_types:
            fields.append(pa.field(key, core_types[key]))
        else:
            dtype = infer_arrow_type_from_records(records, key)
            fields.append(pa.field(key, dtype))
    return pa.schema(fields)


def _resolve_tfidf_backend(requested: str) -> str:
    if requested == "auto":
        return "cuml" if _HAS_CUML else "sklearn"
    if requested == "cuml" and not _HAS_CUML:
        LOGGER.warning("cuML TF-IDF requested but unavailable; falling back to sklearn")
        return "sklearn"
    return requested


def stage_load_dataframe(config: PipelineConfig) -> pl.DataFrame:
    LOGGER.info("Stage 1: loading parquet -> Polars DataFrame (%s)", config.parquet_path)
    df = load_dataframe(config.parquet_path, config.sample_rows)
    LOGGER.info("Loaded %d profile rows", df.height)
    return df


def stage_build_records(config: PipelineConfig, df: pl.DataFrame) -> List[Dict[str, Any]]:
    LOGGER.info("Stage 2: assembling facet records")
    records = make_rows(df, text_trunc=config.text_trunc, posts_max=config.posts_max)
    LOGGER.info("Prepared %d records", len(records))
    if not records:
        raise RuntimeError("No non-empty texts found. Nothing to write.")
    return records


def stage_fit_tfidf(config: PipelineConfig, records: List[Dict[str, Any]]):
    backend = _resolve_tfidf_backend(config.tfidf_backend)
    LOGGER.info(
        "Stage 3: fitting TF-IDF (backend=%s, max_features=%d, min_df=%d, ngram=%d-%d)",
        backend,
        config.tfidf_max_features,
        config.tfidf_min_df,
        config.ngram_range[0],
        config.ngram_range[1],
    )
    vectorizer = fit_tfidf(
        [r["text"] for r in records],
        max_features=config.tfidf_max_features,
        min_df=config.tfidf_min_df,
        ngram_range=config.ngram_range,
        backend=backend,
    )
    return vectorizer, backend


def stage_save_vectorizer(config: PipelineConfig, vectorizer) -> None:
    path = config.vectorizer_path
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        joblib.dump(vectorizer, path)
        LOGGER.info("Stage 4: saved TF-IDF vectorizer to %s", path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to persist TF-IDF vectorizer: %s", exc)


def stage_add_sparse_fields(
    config: PipelineConfig,
    records: List[Dict[str, Any]],
    vectorizer,
    backend: str,
) -> None:
    LOGGER.info("Stage 5: adding sparse TF-IDF fields (workers=%d)", config.tfidf_workers)
    add_sparse_fields(records, vectorizer, workers=config.tfidf_workers, backend=backend)
    gc.collect()


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


async def stage_embed_records_async(
    config: PipelineConfig,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    LOGGER.info(
        "Stage 6: requesting embeddings from DeepInfra (batch_size=%d, concurrency<=%d)",
        EMBED_BATCH_SIZE,
        config.embed_concurrency,
    )
    embedder = DeepInfraEmbedder(
        model=config.embed_model,
        api_key=config.deepinfra_api_key,
        endpoint=config.deepinfra_endpoint,
        concurrency=config.embed_concurrency,
    )

    flat_texts: List[str] = []
    spans: List[Tuple[int, int]] = []
    for record in records:
        chunks = record.get("_post_chunks")
        if chunks:
            start = len(flat_texts)
            flat_texts.extend(chunks)
            spans.append((start, len(chunks)))
        else:
            start = len(flat_texts)
            flat_texts.append(record["text"])
            spans.append((start, 1))

    LOGGER.info("Embedding %d text fragments across %d records", len(flat_texts), len(records))
    embeddings = await embedder.embed(flat_texts)

    enriched: List[Dict[str, Any]] = []
    for record, (start, length) in zip(records, spans):
        enriched_record = dict(record)
        chunk_vecs = embeddings[start:start + length]
        if length == 1 and not record.get("_post_chunks"):
            vec = chunk_vecs[0]
        else:
            vec = np.vstack(chunk_vecs).mean(axis=0)
        vec = _normalize_vector(vec.astype(np.float32))
        enriched_record["embedding"] = vec.tolist()
        if "_post_chunks" in enriched_record:
            del enriched_record["_post_chunks"]
        enriched.append(enriched_record)
    return enriched


def stage_build_schema(records: List[Dict[str, Any]]) -> pa.Schema:
    LOGGER.info("Stage 7: building Arrow schema")
    schema = build_schema_from_records(records)
    return schema


def stage_write_lancedb(
    config: PipelineConfig,
    records: List[Dict[str, Any]],
    schema: pa.Schema,
) -> None:
    LOGGER.info("Stage 8: writing records to LanceDB (%s/%s)", config.db_uri, config.table)
    db = lancedb.connect(config.db_uri)

    if config.recreate:
        try:
            tbl = db.open_table(config.table)
            tbl.delete("")
            LOGGER.info("Cleared existing table %s", config.table)
        except Exception:
            LOGGER.info("Table %s does not exist, creating fresh", config.table)

    try:
        tbl = db.create_table(config.table, schema=schema, mode="overwrite")
        LOGGER.info("Created table %s", config.table)
    except Exception:
        tbl = db.open_table(config.table)
        LOGGER.info("Opened existing table %s", config.table)

    batch_size = 10_000
    total = 0
    for chunk in batched(records, batch_size):
        tbl.add(chunk)
        total += len(chunk)
        LOGGER.info("Inserted %d rows (running total=%d)", len(chunk), total)
    LOGGER.info("Completed LanceDB load (%d total rows)", total)


def run_pipeline(config: PipelineConfig) -> None:
    df = stage_load_dataframe(config)
    records = stage_build_records(config, df)
    vectorizer, backend = stage_fit_tfidf(config, records)
    stage_save_vectorizer(config, vectorizer)
    stage_add_sparse_fields(config, records, vectorizer, backend)
    records = asyncio.run(stage_embed_records_async(config, records))
    schema = stage_build_schema(records)
    stage_write_lancedb(config, records, schema)


def parse_args() -> PipelineConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default="normalized_profiles.parquet")
    ap.add_argument("--db-uri", type=str, default="data/lancedb")
    ap.add_argument("--table", type=str, default="influencer_facets")
    ap.add_argument("--recreate", action="store_true", help="Drop and recreate the table")
    ap.add_argument("--sample-rows", type=int, default=None)
    ap.add_argument("--text-trunc", type=int, default=2000)
    ap.add_argument("--posts-max", type=int, default=5)
    ap.add_argument("--tfidf-max-features", type=int, default=100_000)
    ap.add_argument("--tfidf-min-df", type=int, default=2)
    ap.add_argument("--ngram-min", type=int, default=1)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument(
        "--tfidf-backend",
        type=str,
        choices=["auto", "sklearn", "cuml"],
        default="auto",
        help="Backend for TF-IDF (auto selects cuML when available)",
    )
    default_workers = os.cpu_count() or 1
    ap.add_argument(
        "--tfidf-workers",
        type=int,
        default=default_workers,
        help=f"Worker threads for TF-IDF transform (default: {default_workers})",
    )
    ap.add_argument("--save-vectorizer", type=str, default="artifacts/tfidf_vectorizer.pkl")
    ap.add_argument("--embed-model", type=str, default=os.environ.get("EMBED_MODEL", "google/embeddinggemma-300m"))
    ap.add_argument(
        "--embed-concurrency",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help="Maximum concurrent DeepInfra embedding requests (capped at 190)",
    )
    ap.add_argument(
        "--deepinfra-api-key",
        type=str,
        default=None,
        help="DeepInfra API key (falls back to DEEPINFRA_API_KEY env)",
    )
    ap.add_argument(
        "--deepinfra-endpoint",
        type=str,
        default=os.environ.get("DEEPINFRA_ENDPOINT", "https://api.deepinfra.com/v1/openai"),
        help="DeepInfra endpoint base URL",
    )
    args = ap.parse_args()

    api_key = args.deepinfra_api_key or os.environ.get("DEEPINFRA_API_KEY", "")
    if not api_key:
        raise SystemExit("DeepInfra API key required. Set DEEPINFRA_API_KEY or pass --deepinfra-api-key.")

    concurrency = min(args.embed_concurrency, MAX_CONCURRENT_REQUESTS)
    if args.embed_concurrency > MAX_CONCURRENT_REQUESTS:
        LOGGER.warning(
            "embed_concurrency capped at %d (requested %d)",
            MAX_CONCURRENT_REQUESTS,
            args.embed_concurrency,
        )

    config = PipelineConfig(
        parquet_path=args.parquet,
        db_uri=args.db_uri,
        table=args.table,
        recreate=args.recreate,
        sample_rows=args.sample_rows,
        text_trunc=args.text_trunc,
        posts_max=args.posts_max,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        ngram_range=(args.ngram_min, args.ngram_max),
        tfidf_backend=args.tfidf_backend,
        tfidf_workers=args.tfidf_workers,
        vectorizer_path=args.save_vectorizer,
        embed_model=args.embed_model,
        embed_concurrency=concurrency,
        deepinfra_api_key=api_key,
        deepinfra_endpoint=args.deepinfra_endpoint,
    )
    return config


def main() -> None:
    config = parse_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
