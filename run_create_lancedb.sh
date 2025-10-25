#!/usr/bin/env bash
set -euo pipefail

# Helper to run the LanceDB creation pipeline.
# Usage:
#   ./scripts/run_create_lancedb.sh \
#       --parquet normalized_profiles.parquet \
#       --db-uri data/lancedb_sample \
#       --table influencer_facets_sample \
#       --sample-rows 1000 \
#       --recreate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

cd "${PROJECT_ROOT}"

# Load environment variables from .env if it exists (e.g., DEEPINFRA_API_KEY).
if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source ".env"
  set +a
fi

ARGS=("$@")
DB_URI=""
TABLE_NAME=""

for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --db-uri)
      if (( i + 1 < ${#ARGS[@]} )); then
        DB_URI="${ARGS[i+1]}"
      fi
      ;;
    --table)
      if (( i + 1 < ${#ARGS[@]} )); then
        TABLE_NAME="${ARGS[i+1]}"
      fi
      ;;
  esac
done

print_sample_rows() {
  local when_label="$1"
  if [[ -z "${DB_URI}" || -z "${TABLE_NAME}" ]]; then
    return
  fi

  python - <<'PY' "${DB_URI}" "${TABLE_NAME}" "${when_label}"
import json
import sys

import lancedb

db_uri, table_name, label = sys.argv[1:4]

print(f"--- {label} ({table_name} @ {db_uri}) ---")
try:
    db = lancedb.connect(db_uri)
    table = db.open_table(table_name)
except Exception as exc:  # noqa: BLE001
    print(f"[info] unable to open table: {exc}")
    sys.exit(0)

try:
    batch = table.head(3)
    if hasattr(batch, "to_pylist"):
        batch = batch.to_pylist()
except Exception as exc:  # noqa: BLE001
    print(f"[error] failed to read table: {exc}")
    sys.exit(1)

if not batch:
    print("[info] table contains no rows.")
    sys.exit(0)

def prepare(row):
    safe = {}
    if not isinstance(row, dict):
        row = dict(row.items())

    for key in ("vector_id", "content_type", "platform", "username", "lance_db_id", "followers"):
        if key in row:
            safe[key] = row[key]

    text = row.get("text")
    if isinstance(text, str):
        safe["text"] = text[:160] + ("…" if len(text) > 160 else "")

    posts = row.get("posts")
    if isinstance(posts, str):
        safe["posts_preview"] = posts[:120] + ("…" if len(posts) > 120 else "")

    if isinstance(row.get("embedding"), list):
        safe["embedding"] = f"<float32[{len(row['embedding'])}]>"
    if isinstance(row.get("sparse_indices"), list):
        safe["sparse_indices"] = f"<len={len(row['sparse_indices'])}>"
    if isinstance(row.get("sparse_values"), list):
        safe["sparse_values"] = f"<len={len(row['sparse_values'])}>"

    remaining_keys = sorted(set(row.keys()) - set(safe.keys()))
    if remaining_keys:
        display = ", ".join(remaining_keys[:8])
        if len(remaining_keys) > 8:
            display += f", … (+{len(remaining_keys) - 8} more)"
        safe["other_keys"] = display

    return safe

for idx, row in enumerate(batch, start=1):
    print(f"Row {idx}:")
    print(json.dumps(prepare(row), ensure_ascii=False, indent=2))
PY
}

print_sample_rows "Before run"
python scripts/create_lancedb_from_parquet.py "${ARGS[@]}"
print_sample_rows "After run"
