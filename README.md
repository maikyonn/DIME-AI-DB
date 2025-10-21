# DIME AI Database Pipeline

Modern pipeline for turning raw Instagram/TikTok profile exports into LanceDB datasets enriched with LLM scoring and optional embeddings.

## Pipeline Overview

1. **Language filtering & batching** – `pipeline_batch_process.py`
   - Normalises the raw CSV, keeps only English (or low-text) profiles, and generates new `language_filter` CSVs.
   - Builds JSONL batch payloads (default 20k rows per file).
   - Submits all batches to the OpenAI Batch API (no polling by default) and later downloads the completed results.

2. **Parquet build** – `pipeline_build_parquet.py`
   - Reads the Stage‑0 CSV plus the Stage‑2 chunk results and merges the LLM output back onto each profile.
   - Writes `normalized_profiles.parquet` per dataset (and combined parquet when both Instagram/TikTok are present).

3. **LanceDB build & exploration**
   - `scripts/create_lancedb_from_parquet.py` ingests the combined parquet, carries forward every source column, builds `profile` + `posts` facets, fits TF‑IDF features, and generates dense embeddings (auto-selecting CUDA when available).
   - `scripts/setup_indexes.py` provisions IVF_PQ vector indexes and BM25 FTS indexes with compatibility fallbacks.
   - `scripts/inspect_lancedb.py` quickly prints schema, counts, and sample rows.
   - `streamlit_app.py` provides an interactive dashboard (filters, search, column picker, schema/stats panels).

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: install the RAPIDS CUDA stack (includes cuML TF-IDF, cuDF, cuGraph) 
# if you want GPU-accelerated preprocessing on CUDA 12 hardware.
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.8.*" "dask-cudf-cu12==25.8.*" "cuml-cu12==25.8.*" \
    "cugraph-cu12==25.8.*" "nx-cugraph-cu12==25.8.*" "cuxfilter-cu12==25.8.*" \
    "cucim-cu12==25.8.*" "pylibraft-cu12==25.8.*" "raft-dask-cu12==25.8.*" \
    "cuvs-cu12==25.8.*" "nx-cugraph-cu12==25.8.*"

# IMPORTANT: export or add to .env before Stage 2
export OPENAI_API_KEY=sk-...
```

### Stage 0 & 1 – Normalise and prepare JSONL batches

```bash
python pipeline_batch_process.py data/instagram/insta100kemail1 \
  --language-batch-size 1500 \
  --chunk-size 20000
```

- Language-filtered CSVs land in `pipeline/step0_language_filter/`.
- JSONL batches plus metadata appear in `pipeline/step1_batch_inputs/`.

### Stage 2 – Submit batches (no polling)

```bash
python pipeline_batch_process.py data/instagram/insta100kemail1 \
  --resume-from process --stop-after process
```

When the batches have completed (check your OpenAI dashboard), rerun without `--stop-after` to download and parse results:

```bash
python pipeline_batch_process.py data/instagram/insta100kemail1 \
  --resume-from process
```

Parsed chunk CSVs are written to `pipeline/step2_batch_results/` alongside the raw JSONL responses.

### Stage 3 – Produce normalized parquet

```bash
python pipeline_build_parquet.py data
```

Each dataset gains a `normalized_profiles.parquet` with the merged LLM annotations alongside a combined CSV when both platforms exist under the root.

### Stage 4 – Create / inspect / explore LanceDB artifacts

Create or refresh the LanceDB table directly from the combined parquet:

```bash
HUGGINGFACE_HUB_TOKEN=hf_xxx python scripts/create_lancedb_from_parquet.py \
  --parquet data/combined/normalized_profiles.parquet \
  --db-uri data/lancedb \
      --table influencer_facets \
  --embed-model google/embeddinggemma-300m \
  --batch-size 512 \
  --tfidf-backend cuml \
  --tfidf-workers 8
```

- Serialises all parquet columns (scalars, JSON/text) and produces both `profile` and `posts` facets when captions exist.
- Fits a TF‑IDF vectorizer (`artifacts/tfidf_vectorizer.pkl`) plus sparse indices/values alongside dense embeddings (disable via `--no-embeddings`).
- Auto-detects CUDA for embeddings; override with `--device {cpu|cuda}`.
- Default encoder: `google/embeddinggemma-300m` (override with `--embed-model` or `EMBED_MODEL`).

Create ANN/FTS indexes on an existing table:

```bash
python scripts/setup_indexes.py --db-uri data/lancedb --table influencer_facets --fts
```

Preview a table quickly:

```bash
python scripts/inspect_lancedb.py --db-uri data/lancedb --table influencer_facets --limit 5
```

Launch the Streamlit explorer (random sampling with refresh, filters, BM25 search, schema/stats panels):

```bash
streamlit run streamlit_app.py
```

### Useful flags

- `--resume-from {language|prepare|process}` – jump straight into a stage.
- `--stop-after {language|prepare|process}` – run only up to a stage.
- `--prompt-file prompts/custom.txt` – swap in another instruction template.
- `--force` – ignore cached language-filter/Batched state.

## Data Reference

Reference CSVs under `data-reference/` list every column from the normalised profiles and the LLM outputs, with sample values for quick lookup.

## Notes

- The legacy LanceDB loaders have been removed. Use the new `scripts/create_lancedb_from_parquet.py`, `scripts/setup_indexes.py`, and `scripts/inspect_lancedb.py` utilities instead.
- All data files remain gitignored; drop raw CSVs under `data/`, run the pipeline, and collect results from `pipeline/` and the LanceDB folders.

## Next Steps

- **Streamlit search & export** – expose larger result caps with pagination / download options so users can browse big BM25 result sets safely.
- **Posts parsing QA** – keep validating new post formats (extra nested JSON, platform-specific fields) and extend `extract_posts_chunks` with additional fallbacks as new shapes appear.
- **TF-IDF backend** – pass `--tfidf-backend cuml` to harness a CUDA GPU via cuML, or leave as `auto` (the default) to pick the best available implementation.
- **Analytics dashboards** – layer on optional charts/alerts (engagement vs followers, platform coverage, missing posts) directly in the Streamlit app.
