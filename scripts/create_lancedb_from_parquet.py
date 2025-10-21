#!/usr/bin/env python3
"""
Create a LanceDB table from `normalized_profiles.parquet`.

- Builds two text facets per profile (profile_text, posts_text) without using unreliable location.
- Computes dense embeddings (default: google/embeddinggemma-300m via sentence-transformers; overridable).
- Fits a global TF-IDF (1–2 grams, min_df=2, max_features configurable) on all row `text`, with optional cuML acceleration.
- Stores per-row sparse TF-IDF as (indices, values) lists.
- Emits two rows per profile when both texts are present:
    vector_id = "<lance_db_id>::profile" | content_type="profile"
    vector_id = "<lance_db_id>::posts"   | content_type="posts"
- Creates/opens a LanceDB table and appends rows in batches.
- Does NOT attempt to parse or store locations (as requested).

Usage (typical):
  python scripts/create_lancedb_from_parquet.py \
      --parquet data/normalized_profiles.parquet \
      --db-uri data/lancedb \
      --table influencer_facets \
      --embed-model google/embeddinggemma-300m \
      --batch-size 512 --recreate \
      --save-vectorizer artifacts/tfidf_vectorizer.pkl

Requirements:
  pip install lancedb pyarrow pandas numpy scikit-learn sentence-transformers torch joblib ujson
"""
import os
import re
import gc
import json
import math
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# IO & ML deps
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Vector backend
import lancedb

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

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore
    _HAS_TORCH = False

# Embedding (we prefer sentence-transformers for simplicity & speed)
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_ST = False

# If you really want to use HF models directly, you can add a simple
# AutoModel/AutoTokenizer mean-pooling encoder here as a fallback.
# For now, we keep the script minimal and recommend a SentenceTransformers model.
# e.g., --embed-model BAAI/bge-m3 or intfloat/multilingual-e5-large-instruct

LOGGER = logging.getLogger("create_lancedb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return value


def build_profile_text(row: pd.Series, text_trunc: int, include_keywords: bool=True) -> str:
    # Avoid unreliable location entirely, per user instruction
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
            # Treat as free-form text when the JSON parse fails
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
                if not text_part and isinstance(item.get("extra"), dict):
                    text_part = item["extra"].get("caption")
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
                hashtags: List[str] = []
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
    buf = []
    for it in iterable:
        buf.append(it)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def load_dataframe(parquet_path: str, sample_rows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if sample_rows and sample_rows > 0 and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=42)
    df = df.reset_index(drop=True)
    return df


def make_rows(df: pd.DataFrame, text_trunc: int, posts_max: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        lance_db_id = coalesce(r.get("lance_db_id"), r.get("platform_id"), r.get("username"))
        # Build profile text
        profile_text = build_profile_text(r, text_trunc=text_trunc)
        posts_chunks = extract_posts_chunks(r.get("posts"), posts_max=posts_max) if "posts" in r else []
        posts_text = " \n ".join(posts_chunks)

        row_values = {col: normalize_field_value(r.get(col)) for col in df.columns}

        meta_common = dict(
            lance_db_id=lance_db_id,
            platform=coalesce(r.get("platform")),
            platform_id=coalesce(r.get("platform_id")),
            username=coalesce(r.get("username")),
            display_name=coalesce(r.get("display_name")),
            biography=coalesce(r.get("biography")),
            external_url=coalesce(r.get("external_url")),
            profile_url=coalesce(r.get("profile_url")),
            profile_image_url=coalesce(r.get("profile_image_url")),
            followers=inty(r.get("followers")),
            following=inty(r.get("following")),
            likes_total=inty(r.get("likes_total")),
            posts_count=inty(r.get("posts_count")),
            engagement_rate=floaty(r.get("engagement_rate")),
            median_view_count_last10=floaty(r.get("median_view_count_last10")),
            median_like_count_last10=floaty(r.get("median_like_count_last10")),
            median_comment_count_last10=floaty(r.get("median_comment_count_last10")),
            reel_post_ratio_last10=floaty(r.get("reel_post_ratio_last10")),
            total_img_posts_ig=inty(r.get("total_img_posts_ig")),
            total_reels_ig=inty(r.get("total_reels_ig")),
            individual_vs_org_score=floaty(r.get("individual_vs_org_score")),
            generational_appeal_score=floaty(r.get("generational_appeal_score")),
            professionalization_score=floaty(r.get("professionalization_score")),
            relationship_status_score=floaty(r.get("relationship_status_score")),
            occupation=coalesce(r.get("occupation")),
            is_verified=booly(r.get("is_verified")),
            is_private=booly(r.get("is_private")),
            is_commerce_user=booly(r.get("is_commerce_user")),
            source_batch=coalesce(r.get("source_batch")),
            llm_processed=booly(r.get("llm_processed")),
            source_csv=coalesce(r.get("source_csv")),
            prompt_file=coalesce(r.get("prompt_file")),
            raw_response=coalesce(r.get("raw_response")),
            processing_error=coalesce(r.get("processing_error")),
        )

        for col, value in row_values.items():
            if col not in meta_common and col not in {"vector_id", "content_type", "text"}:
                meta_common[col] = value

        if profile_text:
            rows.append({
                "vector_id": f"{lance_db_id}::profile",
                "content_type": "profile",
                "text": profile_text,
                **meta_common,
            })
        if posts_text:
            rows.append({
                "vector_id": f"{lance_db_id}::posts",
                "content_type": "posts",
                "text": posts_text,
                **meta_common,
                "_post_chunks": posts_chunks,
            })
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
            raise RuntimeError("cuML is not available but --tfidf-backend=cuml was requested")
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
    # Transform in batches to avoid memory spikes; optionally parallelize across threads.
    use_cuml = (
        backend == "cuml"
        and _HAS_CUML
        and CuMLTfidfVectorizer is not None
        and isinstance(vectorizer, CuMLTfidfVectorizer)
    )
    if use_cuml:
        workers = 1  # GPU transform runs on-device; threading offers no benefit

    def ranges() -> Iterable[Tuple[int, int]]:
        for start in range(0, len(records), batch_size):
            end = min(start + batch_size, len(records))
            yield start, end

    def process_batch(bounds: Tuple[int, int]):
        start, end = bounds
        batch_texts = [records[i]["text"] for i in range(start, end)]
        if use_cuml:
            batch_series = cudf.Series(batch_texts, dtype="str")
            X = vectorizer.transform(batch_series)  # returns GPU CSR
            X = X.get()  # move to host as scipy CSR
        else:
            batch_texts = [str(t) for t in batch_texts]
            X = vectorizer.transform(batch_texts)  # CSR
        if use_cuml:
            # ensure garbage collected promptly on device
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
        if workers > 1 and 'executor' in locals():
            executor.shutdown(wait=True)


class STEmbedder:
    def __init__(self, model_name: str, device: Optional[str] = None, normalize: bool = True):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers is not installed. pip install sentence-transformers")
        resolved_device = device
        if resolved_device is None and _HAS_TORCH and torch.cuda.is_available():  # pragma: no branch
            resolved_device = "cuda"
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_API_KEY")
        )
        st_kwargs = {"device": resolved_device or "cpu"}
        if token:
            # SentenceTransformer switched from use_auth_token -> token in newer releases; support both.
            try:
                st_kwargs["token"] = token
                self.model = SentenceTransformer(model_name, **st_kwargs)
            except TypeError:
                st_kwargs.pop("token", None)
                st_kwargs["use_auth_token"] = token
                self.model = SentenceTransformer(model_name, **st_kwargs)
        else:
            self.model = SentenceTransformer(model_name, **st_kwargs)
        self.device = resolved_device or "cpu"
        self.normalize = normalize

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        # Many ST models can normalize automatically; set normalize_embeddings=True for some models.
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=self.normalize)
        if isinstance(embs, list):
            embs = np.array(embs, dtype=np.float32)
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32)
        return embs


def append_embeddings(records: List[Dict[str, Any]], embedder: STEmbedder, batch_size: int) -> None:
    for batch_idx, batch in enumerate(batched(records, batch_size)):
        flat_texts: List[str] = []
        spans: List[Tuple[int, int]] = []

        for record in batch:
            chunks = record.get("_post_chunks")
            if chunks:
                start = len(flat_texts)
                flat_texts.extend(chunks)
                spans.append((start, len(chunks)))
            else:
                start = len(flat_texts)
                flat_texts.append(record["text"])
                spans.append((start, 1))

        embs = embedder.encode(flat_texts, batch_size=batch_size)

        for record, (start, length) in zip(batch, spans):
            if length == 1 and not record.get("_post_chunks"):
                vec = embs[start]
            else:
                chunk_vecs = embs[start:start + length]
                vec = chunk_vecs.mean(axis=0)
                norm = np.linalg.norm(vec)
                if embedder.normalize and norm > 0:
                    vec = vec / norm
            record["embedding"] = vec.astype(np.float32).tolist()
            if "_post_chunks" in record:
                del record["_post_chunks"]


def infer_arrow_type(value) -> pa.DataType:
    # Helper for schema inference where we want stable types
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
        # assume list of floats or ints
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default="normalized_profiles.parquet")
    ap.add_argument("--db-uri", type=str, default="data/lancedb")
    ap.add_argument("--table", type=str, default="influencer_facets")
    ap.add_argument("--recreate", action="store_true", help="Drop and recreate the table")
    ap.add_argument("--embed-model", type=str, default=os.environ.get("EMBED_MODEL", "google/embeddinggemma-300m"))
    ap.add_argument(
        "--device",
        type=str,
        default=os.environ.get("EMBED_DEVICE"),
        help="Force embedding device (e.g. cuda, cpu)",
    )
    ap.add_argument("--batch-size", type=int, default=512)
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
    args = ap.parse_args()

    LOGGER.info("Loading parquet: %s", args.parquet)
    df = load_dataframe(args.parquet, args.sample_rows)
    LOGGER.info("Loaded %d profile rows", len(df))

    LOGGER.info("Assembling facet texts (profile/posts) without location...")
    records = make_rows(df, text_trunc=args.text_trunc, posts_max=args.posts_max)
    LOGGER.info("Emittable facet records: %d", len(records))
    if not records:
        LOGGER.error("No non-empty texts found. Nothing to write.")
        return

    tfidf_backend = args.tfidf_backend
    if tfidf_backend == "auto":
        tfidf_backend = "cuml" if _HAS_CUML else "sklearn"
    elif tfidf_backend == "cuml" and not _HAS_CUML:
        LOGGER.warning("cuML TF-IDF requested but unavailable; falling back to sklearn")
        tfidf_backend = "sklearn"

    LOGGER.info("Fitting TF-IDF on %d texts (max_features=%d, min_df=%d, ngram=%d-%d, backend=%s)",
                len(records), args.tfidf_max_features, args.tfidf_min_df, args.ngram_min, args.ngram_max, tfidf_backend)
    vectorizer = fit_tfidf(
        [r["text"] for r in records],
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
        ngram_range=(args.ngram_min, args.ngram_max),
        backend=tfidf_backend,
    )
    os.makedirs(os.path.dirname(args.save_vectorizer) or ".", exist_ok=True)
    try:
        joblib.dump(vectorizer, args.save_vectorizer)
        LOGGER.info("Saved TF-IDF vectorizer to %s", args.save_vectorizer)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to persist TF-IDF vectorizer (backend=%s): %s", tfidf_backend, exc)

    LOGGER.info("Transforming TF-IDF to sparse (indices/values) fields...")
    add_sparse_fields(records, vectorizer, workers=args.tfidf_workers, backend=tfidf_backend)

    LOGGER.info("Loading embedding model: %s", args.embed_model)
    embedder = STEmbedder(args.embed_model, device=args.device, normalize=True)
    LOGGER.info("Embedding device: %s", embedder.device)

    LOGGER.info("Encoding dense embeddings in batches of %d...", args.batch_size)
    append_embeddings(records, embedder, batch_size=args.batch_size)

    # Prepare LanceDB
    LOGGER.info("Connecting LanceDB: %s", args.db_uri)
    db = lancedb.connect(args.db_uri)

    schema = build_schema_from_records(records)

    if args.recreate:
        try:
            tbl = db.open_table(args.table)
            LOGGER.info("Dropping existing table: %s", args.table)
            tbl.delete("")  # delete all rows
            # lancedb doesn't expose explicit drop_table universally; overwrite on create instead.
        except Exception:
            pass

    # Create or open table
    try:
        tbl = db.create_table(args.table, schema=schema, mode="overwrite")
        LOGGER.info("Created table %s", args.table)
    except Exception:
        tbl = db.open_table(args.table)
        LOGGER.info("Opened existing table %s", args.table)

    # Append in chunks
    batch = 10_000
    for chunk in batched(records, batch):
        tbl.add(chunk)
        LOGGER.info("Added %d rows (total so far unknown; LanceDB append ok)", len(chunk))

    LOGGER.info("Done. Table: %s", args.table)


if __name__ == "__main__":
    main()
