# DIME AI Database Pipeline

Modern pipeline for turning raw Instagram/TikTok profile exports into LanceDB datasets enriched with LLM scoring and optional embeddings.

## Pipeline Overview

1. **Language filtering & batching** – `pipeline_batch_process.py`
   - Normalises the raw CSV, keeps only English (or low-text) profiles, and generates new `language_filter` CSVs.
   - Builds JSONL batch payloads (default 20k rows per file).
   - Submits all batches to the OpenAI Batch API (no polling by default) and later downloads the completed results.

2. **Database build** – `pipeline_build_database.py`
   - Reads the Stage‑0 CSV plus the Stage‑2 chunk results and merges the LLM output back onto each profile.
   - Writes a LanceDB table containing both the original columns and LLM scores/demographics/keywords.
   - (Optional) Builds the companion vector LanceDB (keyword/profile/content embeddings).

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

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

### Stage 3 – Build LanceDB (and optional vectors)

The database builder now iterates through every dataset directory beneath the path you provide and shows a progress bar.

```bash
python pipeline_build_database.py data \
  --vectors --model-name sentence-transformers/all-mpnet-base-v2
```

Output layout per dataset:

```
<dataset>/
├── influencers_lancedb/
└── influencers_vectordb/    (only when --vectors was supplied)
```

Each LanceDB record contains:

- Normalised profile fields (followers, biography, cleaned posts JSON, reel/median stats, etc.)
- LLM scores/demographics (`individual_vs_org`, `location`, `age`, `occupation`, ...)
- Prompt metadata and raw model response for auditing

### Useful flags

- `--resume-from {language|prepare|process}` – jump straight into a stage.
- `--stop-after {language|prepare|process}` – run only up to a stage.
- `--prompt-file prompts/custom.txt` – swap in another instruction template.
- `--force` – ignore cached language-filter/Batched state.

## Data Reference

Reference CSVs under `data-reference/` list every column from the normalised profiles and the LLM outputs, with sample values for quick lookup.

## Notes

- The legacy scripts (dashboards, manual rebuild utilities, old batch runners) have been removed. The pipeline now revolves around the two scripts listed above.
- All data files remain gitignored; drop raw CSVs under `data/`, run the pipeline, and collect results from `pipeline/` and the LanceDB folders.
