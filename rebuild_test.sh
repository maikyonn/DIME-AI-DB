#!/bin/bash

# Quick Test Rebuild Script - Uses 1000-row sample for faster testing
# Perfect for development and validation

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_header() { echo -e "\n${PURPLE}$1${NC}\n${PURPLE}$(printf '=%.0s' $(seq 1 ${#1}))${NC}"; }

start_time=$(date +%s)

print_header "🧪 DIME-AI-DB Quick Test Rebuild"
echo "Using 1000-row sample for fast testing and validation"

# Clean test databases
print_status "Cleaning test databases..."
rm -rf test_vectordb test_lancedb 2>/dev/null || true

# Create sample CSV if needed
CSV_SAMPLE="data/dataset-2/Snap Data_1000rows.csv"
if [[ ! -f "$CSV_SAMPLE" ]]; then
    print_status "Creating 1000-row sample..."
    head -n 1001 "data/dataset-2/Snap Data (1).csv" > "$CSV_SAMPLE"
fi

# Build test unified database
print_status "Building test unified database..."
python -c "
import sys
sys.path.append('src')
sys.path.append('src/data')
from src.data.unified_data_loader import UnifiedDataLoader

loader = UnifiedDataLoader()
csv_df = loader.load_csv_data('Snap Data_1000rows.csv')
llm_data = loader.parse_all_batch_outputs()  # Use all batches
merged_df = loader.merge_data(csv_df, llm_data)

import lancedb
db = lancedb.connect('test_lancedb')
try:
    db.drop_table('influencer_profiles')
except:
    pass
table = db.create_table('influencer_profiles', merged_df)
print(f'✅ Test unified database: {table.count_rows()} records')
"

# Build test vector database
print_status "Building test vector database..."
python -c "
import sys
sys.path.append('src')
sys.path.append('src/data')
from src.data.vector_database_builder import VectorDatabaseBuilder

builder = VectorDatabaseBuilder()
table = builder.create_vector_database(
    csv_filename='Snap Data_1000rows.csv',
    jsonl_filename=None,  # Use all batches
    db_path='test_vectordb',
    table_name='influencer_profiles_test'
)
print(f'✅ Test vector database: {table.count_rows()} records')
" 2>/dev/null

# Test search functionality
print_status "Testing search functionality..."
python -c "
import sys
sys.path.append('src/search')
from src.search.vector_search import VectorSearchEngine

engine = VectorSearchEngine(
    db_path='test_vectordb',
    table_name='influencer_profiles_test'
)

stats = engine.get_database_stats()
print(f'📊 Records: {stats[\"total_records\"]:,}, LLM: {stats[\"llm_processed\"]:,}, Dimensions: {stats[\"vector_dimensions\"]}')

# Test queries
queries = ['home decor', 'fashion influencers', 'travel bloggers']
for query in queries:
    results = engine.search(query, limit=2)
    print(f'🔍 \"{query}\": {len(results)} results')

print('✅ Search functionality working!')
" 2>/dev/null

duration=$(($(date +%s) - start_time))

print_header "🎉 Test Build Complete!"
print_success "Time: ${duration}s • Databases: test_lancedb/ & test_vectordb/"

echo ""
echo "🧪 Test the system:"
echo "   python -c \"
import sys; sys.path.append('src/search')
from src.search.vector_search import VectorSearchEngine
engine = VectorSearchEngine('test_vectordb', 'influencer_profiles_test')
results = engine.search('home decor influencers', limit=3)
for i, row in results.iterrows():
    print(f'{i+1}. @{row[\"account\"]} - {row[\"profile_name\"]}')
\""

echo ""
echo "🚀 Ready for full build: ./rebuild_all.sh"