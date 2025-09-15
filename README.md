# DIME AI Database - Instagram Influencer Vector Search System

**Advanced vector database system for Instagram profile analysis combining BrightData scraping with OpenAI GPT-5 LLM analysis and semantic search capabilities.**

## 🏗️ Architecture Overview

This repository contains a comprehensive database and search system for processing Instagram profile data with three main components:

- **BrightData CSV**: Raw Instagram profile data (followers, posts, bio, etc.)
- **OpenAI GPT-5 Analysis**: AI-generated scores and keywords via batch API
- **Three-Vector Database**: Semantic search with keyword, profile, and content vectors
- **Interactive Dashboard**: Data exploration and visualization tools

### Three-Vector Search Architecture

The system implements a sophisticated **three-vector architecture** for semantic search:

1. **Keyword Vector** (45% weight) - LLM-extracted keywords (highest priority)
2. **Profile Vector** (35% weight) - Pure natural content: biography + business category
3. **Content Vector** (20% weight) - Aggregated post captions

**LLM Analysis Scores** are stored as metadata for precise filtering (not embedded as text).

This enables natural language queries like:
- "couple influencers in new york"
- "home decor micro influencers" 
- "professional fashion brands"

## 📁 Project Structure

```
DIME-AI-DB/
├── src/
│   ├── data/
│   │   ├── unified_data_loader.py         # Core data combination logic
│   │   └── vector_database_builder.py     # Three-vector database builder
│   ├── search/
│   │   └── vector_search.py              # Weighted multi-vector search
│   ├── services/                         # Future: API services
│   └── api/                             # Future: REST endpoints
├── scripts/                             # Python scripts and utilities
│   ├── rebuild_database_simple.py      # Complete database rebuild
│   ├── show_row.py                     # Single row CSV export utility
│   ├── visualize_data.py               # Streamlit dashboard
│   ├── run_visualization.py           # Interactive runner
│   ├── display_single_row.py          # Text display utility
│   └── test_display_row.py            # Display testing script
├── utils/                              # Helper utilities
├── data/
│   ├── brightdata-csv-dataset/
│   │   ├── Snap Data.csv                # Main dataset
│   │   └── Snap Data_1000rows.csv       # Sample dataset
│   └── llm-analysis/
│       └── Batch Output *.jsonl         # LLM analysis files (auto-detected)
├── rebuild_all.sh                      # Complete rebuild pipeline
├── rebuild_test.sh                     # Quick test rebuild
└── requirements.txt                    # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/maikyonn/DIME-AI-DB.git
cd DIME-AI-DB

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build Vector Database

#### Full Dataset (20-30 minutes)
```bash
# Complete rebuild pipeline with validation
./rebuild_all.sh

# Or manual build
python scripts/rebuild_database_simple.py --vectors
```

#### Quick Test (1000 rows)
```bash
# Test with sample data
python -c "
from src.data.vector_database_builder import VectorDatabaseBuilder
builder = VectorDatabaseBuilder()
table = builder.create_vector_database(
    csv_filename='Snap Data_1000rows.csv',
    jsonl_filename=None,
    db_path='test_vectordb',
    table_name='influencer_profiles_test'
)"
```

### 3. Search Interface

#### Interactive Search
```bash
python scripts/rebuild_database_simple.py --interactive
```

#### Programmatic Search
```python
from src.search.vector_search import VectorSearchEngine, SearchWeights

# Initialize search engine
engine = VectorSearchEngine(
    db_path="influencers_vectordb",
    table_name="influencer_profiles"
)

# Basic search
results = engine.search("home decor influencers", limit=10)

# Search with filters
results = engine.search(
    query="fashion micro influencers",
    limit=5,
    filters={
        "followers": (10000, 100000),
        "engagement_rate": (0.03, 1.0),
        "professionalization_score": (7, 10)
    }
)

# Custom search weights
weights = SearchWeights(keyword=0.6, profile=0.3, content=0.1)
results = engine.search("sustainable fashion", weights=weights)
```

### 4. Data Export and Visualization

```bash
# Export single row for LLM analysis
python scripts/show_row.py 1000

# Launch dashboard
python scripts/run_visualization.py
```

## 🎯 Search Capabilities

### Natural Language Queries
- **"couple influencers in new york"** → Finds profiles with relationship indicators + NYC location
- **"home decor DIY micro influencers"** → Home improvement creators with 10K-100K followers
- **"professional fashion brands"** → High professionalization scores + fashion keywords
- **"gen z travel bloggers"** → Young demographic + travel content
- **"sustainable lifestyle creators"** → Eco-conscious content and keywords

### Query Type Auto-Detection
- **Keyword-focused**: "hashtag fashion trends" → 70% keyword weight
- **Profile-focused**: "micro influencers in texas" → 60% profile weight
- **Content-focused**: "posts about home renovation" → 50% content weight
- **Balanced**: General queries → 45% keyword, 35% profile, 20% content

### Advanced Filtering
```python
filters = {
    "followers": (50000, 500000),        # Range filter
    "is_verified": True,                 # Boolean filter
    "business_category_name": ["Blogger", "Influencer"],  # List filter
    "individual_vs_org_score": (7, 10)  # High individual score
}

results = engine.search("fashion creators", filters=filters)
```

## 📊 Database Schema

### Vector Database Schema
The vector database contains all original fields plus:

#### Three Vector Columns
- **keyword_vector**: 768-dim embeddings from LLM keywords
- **profile_vector**: 768-dim embeddings from biography + metadata (NO names)
- **content_vector**: 768-dim embeddings from post captions

#### Text Representations
- **keyword_text**: Space-separated LLM keywords for analysis
- **profile_text**: Biography + category + tier + location hints
- **content_text_sample**: First 200 chars of aggregated captions

### Original BrightData Fields
- Profile info: `account`, `full_name`, `biography`
- Metrics: `followers`, `following`, `avg_engagement`, `posts_count`
- Flags: `is_business_account`, `is_verified`, `is_private`
- Content: `posts`, `highlights`, `business_category_name`

### LLM Analysis Fields
- **Individual vs Org Score** (0-10): Individual vs Organization classification
- **Generational Appeal Score** (0-10): Gen Z appeal assessment
- **Professionalization Score** (0-10): Professional/brand level
- **Keywords 1-5**: Distinctive profile descriptors extracted by LLM
- **LLM Processed**: Boolean flag for analysis availability
- **Source Batch**: Which batch file provided the LLM analysis

## 🎮 Usage Examples

### Testing Search Functionality
```bash
# Run comprehensive tests
python scripts/rebuild_database_simple.py --test

# Build and test in one command
python scripts/rebuild_database_simple.py --vectors --test
```

### Database Statistics
```python
from src.search.vector_search import VectorSearchEngine

engine = VectorSearchEngine()
stats = engine.get_database_stats()

# Example output:
# Total Records: 101,409
# LLM Processed: 20,932
# Vector Dimensions: 768
# Avg Followers: 89,432
# Verified Accounts: 1,248
# Business Accounts: 15,632
```

### Similar Profile Search
```python
# Find profiles similar to a specific account
similar = engine.search_similar_profiles("homedecorbydollfacefefe", limit=5)
print(similar[['account', 'profile_name', 'keyword_text', 'combined_score']])
```

## 🔧 Technical Details

### Vector Database Technology
- **LanceDB**: High-performance vector database with PyArrow backend
- **Sentence Transformers**: all-mpnet-base-v2 model (768 dimensions)
- **IVF_PQ Indices**: Optimized vector search with <200ms query times
- **Hybrid Search**: Combines semantic similarity with metadata filtering
- **Multi-Batch Support**: Automatically loads all Batch Output *.jsonl files

### Performance Features
- **Smart Search Weights**: Auto-detects query type and adjusts vector weights
- **Optimized Indices**: IVF_PQ vector indices + scalar indices for metadata
- **Adaptive Partitioning**: Dynamic partitioning based on dataset size
- **Multi-Vector Scoring**: Weighted combination of three vector similarities

### Data Processing Pipeline
```
CSV (profiles) + Multiple JSONL (LLM analysis)
    ↓ unified_data_loader.py (multi-batch processing)
    ↓ Data cleaning, duplicate handling, type conversion
    ↓ vector_database_builder.py (three-vector generation)
    ↓ LanceDB table with vector indices
    ↓ vector_search.py (weighted semantic search)
```

## 📈 Performance Benchmarks

### Search Performance
- **Query Time**: <200ms for semantic search
- **Index Creation**: ~20-30 minutes for full dataset
- **Storage**: ~3GB for 100K profiles with vectors
- **Memory Usage**: Efficient streaming for large datasets

### Score Ranges (Search Results)
- **0.4-1.0**: Excellent semantic match
- **0.2-0.4**: Good relevance
- **0.1-0.2**: Moderate match
- **<0.1**: Low relevance

## 🔍 Data Exploration Dashboard

The Streamlit dashboard provides comprehensive analysis tools:

### Analytics Tabs
- **🧠 LLM Analysis**: Score distributions and keyword frequency analysis
- **👥 Followers**: Follower count analysis and tier distributions
- **📈 Engagement**: Engagement vs followers scatter plots and correlations
- **🏢 Business**: Business vs personal account breakdowns
- **🔍 Data Explorer**: Advanced filtering with real-time statistics
- **🎯 Vector Search**: Interactive semantic search interface

### Advanced Filtering
- **Text Search**: Account names, biography content, keywords
- **Engagement Metrics**: Follower ranges, engagement rates, post counts
- **Account Types**: Business vs personal, verified status, privacy settings
- **LLM Scores**: All three analysis dimensions with range filters
- **Vector Similarity**: Find similar profiles based on content vectors

## 🛠️ Development and Customization

### Adding New Analysis
1. Process new data through OpenAI Batch API
2. Place JSONL files in `data/llm-analysis/` (auto-detected)
3. Run `./rebuild_all.sh` to rebuild with new data
4. New batch files automatically take precedence for duplicates

### Extending Search Functionality
```python
# Custom search weights for specific use cases
weights = SearchWeights(
    keyword=0.7,    # High keyword importance
    profile=0.2,    # Lower profile weight
    content=0.1     # Minimal content weight
)

# Specialized filters for your domain
domain_filters = {
    "professionalization_score": (8, 10),
    "followers": (100000, 1000000),
    "business_category_name": ["Fashion", "Beauty"]
}
```

### Multi-Batch Processing
The system automatically:
- Detects all `Batch Output *.jsonl` files in `data/llm-analysis/`
- Processes files in order with duplicate handling
- Later batch files take precedence for the same profile
- Provides detailed statistics on batch distribution

## 🔗 Data Linking and Processing

### Multi-Batch JSONL Processing
- **Auto-Detection**: Finds all `Batch Output *.jsonl` files automatically
- **Batch-to-CSV Mapping**: Each batch processes different CSV row ranges (~21K rows each)
  - Batch 001: CSV rows 1-21,000 (`profile-1` to `profile-21000` → CSV rows 1-21,000)
  - Batch 002: CSV rows 21,001-42,000 (`profile-1` to `profile-21000` → CSV rows 21,001-42,000)
  - And so on...
- **Statistics Tracking**: Shows distribution and row ranges across batch files
- **No Duplicates**: Each batch processes unique CSV row ranges

### Vector Generation Strategy
- **No Unique Identifiers**: Profile names excluded for better clustering
- **Location Extraction**: Smart location detection from bio and posts
- **Follower Tiers**: Semantic tier descriptions (nano/micro/macro/mega)
- **Content Cleaning**: Hashtag filtering and length optimization for embeddings

## 🚨 Troubleshooting

### Common Issues
1. **"Table not found"**: Run `--build` first to create the database
2. **Slow queries**: Check if vector indices are created properly  
3. **Poor results**: Try adjusting search weights or adding filters
4. **Memory issues**: Use smaller batch sizes in embedding generation
5. **Empty results**: Verify database has been built with `--build`

### Performance Tips
- Use filters to reduce search space before vector similarity
- Limit results to reasonable numbers (10-50) for faster response
- Cache frequent queries if building an application
- Use the test dataset for development and debugging

### Build Troubleshooting
```bash
# Check data files exist
ls -la data/brightdata-csv-dataset/
ls -la data/llm-analysis/

# Verify database creation
python -c "import lancedb; db=lancedb.connect('influencers_vectordb'); print(db.table_names())"

# Test search functionality
python scripts/rebuild_database_simple.py --test
```

## 📚 Dependencies

### Core Libraries
- **pandas, numpy**: Data processing and manipulation
- **lancedb, pyarrow**: Vector database with columnar storage
- **sentence-transformers**: Text embedding generation (all-mpnet-base-v2)
- **tqdm**: Progress bars for long operations

### Visualization
- **streamlit, plotly**: Interactive dashboard and charts
- **matplotlib, seaborn**: Statistical visualizations

### Optional
- **openai**: For batch processing (if generating new analysis)
- **flask**: For future API development

## 📄 Additional Documentation

- **VECTOR_DATABASE_GUIDE.md**: Comprehensive vector search documentation
- **build_info.txt**: Generated after successful builds with system information
- **Individual script docstrings**: Detailed function and class documentation

## 🔒 Privacy & Security

- **No Data in Repo**: All actual data files are gitignored for privacy
- **Local Processing**: All analysis and search runs locally
- **No External Calls**: Search works offline once database is built
- **User Control**: Complete control over data storage and access
- **No Name Vectors**: Profile names excluded from similarity calculations

## 🎯 Future Enhancements

- **API Endpoints**: REST API for programmatic access
- **Real-time Updates**: Streaming updates for new profile data
- **Advanced Analytics**: Trend analysis and cohort studies
- **Export Tools**: Advanced export formats and integrations
- **Clustering**: Profile clustering based on vector similarities

---

## 🚀 Ready to Use Commands

```bash
# Complete rebuild and test
./rebuild_all.sh

# Interactive search
python scripts/rebuild_database_simple.py --interactive

# Export profile for analysis
python scripts/show_row.py 1000

# Launch dashboard
streamlit run scripts/visualize_data.py
```

*This repository provides a complete solution for Instagram influencer discovery and analysis using cutting-edge vector search technology. All data files must be provided separately and are not tracked in version control.*