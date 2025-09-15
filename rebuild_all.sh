#!/bin/bash

# DIME-AI-DB Rebuild Script - Clean all databases and rebuild from scratch
# Usage: ./rebuild_all.sh [base_folder] [--yes]

set -e

# Parse arguments
BASE_FOLDER=""
CONFIRM_YES=false

for arg in "$@"; do
    case $arg in
        --yes|-y) CONFIRM_YES=true ;;
        --*) echo "Unknown option: $arg"; exit 1 ;;
        *) [[ -z "$BASE_FOLDER" ]] && BASE_FOLDER="$arg" ;;
    esac
done

# Check directory
[[ ! -f "scripts/rebuild_database_simple.py" ]] && { echo "❌ Run from DIME-AI-DB directory"; exit 1; }

# Confirmation
echo "🚀 DIME-AI-DB Rebuild"
[[ -n "$BASE_FOLDER" ]] && echo "Base folder: $BASE_FOLDER"
echo "Will clean databases → rebuild LanceDB → create vectors → validate"

if [[ "$CONFIRM_YES" != true ]]; then
    read -p "Continue? (y/N): " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && { echo "Cancelled"; exit 0; }
fi

start_time=$(date +%s)

# Clean old databases
echo "🧹 Cleaning databases..."
rm -rf influencers_lancedb influencers_vectordb test_vectordb snap_data_lancedb 2>/dev/null || true

# Verify data files
echo "📋 Verifying data..."
if [[ -n "$BASE_FOLDER" ]]; then
    [[ ! -d "$BASE_FOLDER" ]] && { echo "❌ Base folder not found: $BASE_FOLDER"; exit 1; }
    DATA_DIR="$BASE_FOLDER"
else
    DATA_DIR="/Users/maikyon/Documents/Programming/100k-filter/DIME-AI-DB/data/dataset-3-100k-email"
fi

CSV_FILE=$(find "$DATA_DIR" -name "*.csv" -type f | head -1)
[[ -z "$CSV_FILE" ]] && { echo "❌ No CSV file found in $DATA_DIR"; exit 1; }

BATCH_DIRS=($(find "$DATA_DIR" -name "batch_*" -type d | sort))
echo "Found CSV: $(basename "$CSV_FILE")"
echo "Found ${#BATCH_DIRS[@]} batch directories"

# Rebuild databases
echo "🏗️ Rebuilding LanceDB..."
python scripts/rebuild_database_simple.py "$DATA_DIR"

echo "📊 Database stats:"
python -c "
import lancedb
db = lancedb.connect('influencers_lancedb')
table = db.open_table('influencer_profiles')
print(f'Records: {table.count_rows():,}')
" 2>/dev/null || echo "Could not get stats"

# Check dependencies
echo "🐍 Checking dependencies..."
python -c "
required = ['sentence_transformers', 'lancedb', 'pandas', 'numpy', 'pyarrow', 'tqdm']
missing = []
for pkg in required:
    try: __import__(pkg)
    except ImportError: missing.append(pkg)
if missing:
    print(f'Missing: {missing}')
    exit(1)
print('All dependencies available')
"

# Build vector database
echo "🤖 Building vector database..."
python scripts/rebuild_database_simple.py "$DATA_DIR" --vectors --test

# Final stats
python -c "
import sys
sys.path.append('src')
from src.search.vector_search import VectorSearchEngine
try:
    engine = VectorSearchEngine()
    stats = engine.get_database_stats()
    print(f\"📊 Final Stats:\")
    print(f\"   Records: {stats['total_records']:,}\")
    print(f\"   LLM Processed: {stats['llm_processed']:,}\")
    print(f\"   Vector Dims: {stats['vector_dimensions']}\")
    results = engine.search('tech influencer', limit=10)
    print(f\"   Search: {'✅ Working' if not results.empty else '❌ Failed'}\")
except Exception as e:
    print(f'Validation error: {e}')
" 2>/dev/null || echo "Could not validate"

# Summary
duration=$(($(date +%s) - start_time))
echo ""
echo "✅ Rebuild complete in ${duration}s"
echo "🚀 Ready: python scripts/build_vector_database.py --interactive"