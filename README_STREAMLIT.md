# Streamlit Dashboard for LanceDB Pipeline

Interactive dashboard to monitor pipeline execution, verify embeddings, and explore your influencer dataset.

## Features

### 📊 Overview Tab
- **Pipeline Status**: Visual status of all pipeline stages (Extract → TF-IDF → Embeddings → Combine → Load)
- **Dataset Metrics**: Total rows, sample size, platform distribution
- **Data Distributions**: Interactive histograms and bar charts

### 🔍 Embeddings Tab
- **Embedding Verification**: Checks if embeddings exist and are valid
- **Dimension Analysis**: Shows embedding dimensions for each vector column
- **Coverage Metrics**: Null counts and coverage percentages
- **Sample Values**: Preview first 5 dimensions of embeddings

### ✅ Quality Checks Tab
- **Required Columns**: Validates presence of vector_id, text, content_type, platform
- **Data Completeness**: Checks for null values and empty fields
- **Duplicate Detection**: Identifies duplicate vector_ids
- **Platform Distribution**: Shows record counts per platform
- **Column Statistics**: Descriptive statistics for numeric columns

### 🔎 Search Tab
- **Full-Text Search**: BM25 search across your dataset
- **Top-K Results**: Configurable number of results (5-50)
- **Relevance Scoring**: Shows search scores and matched content

### 📋 Raw Data Tab
- **Data Browser**: Explore sample rows with column selection
- **Filtering**: Filter by content type, platform, followers, verification status
- **Schema Inspector**: View table schema and statistics

## Quick Start

### 1. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### 2. Configure Connection

In the sidebar:
- **Database URI**: Path to LanceDB (default: `data/lancedb`)
- **Table**: Select from available tables
- **Sample Size**: Number of rows to load (100-5000)
- **Pipeline Output**: Path to pipeline output directory (default: `pipeline/output`)

### 3. Explore Tabs

Navigate through the 5 tabs to:
1. Check pipeline status and metrics
2. Verify embeddings are working
3. Run quality checks on data
4. Test semantic search
5. Browse raw data

## Dashboard Sections

### Pipeline Status Display

Shows completion status for each stage:

```
1️⃣ Extract Facets  ✅
   12,450 records

2️⃣ TF-IDF  ✅
   Completed

3️⃣ Embeddings  ✅
   Completed

4️⃣ Combine  ✅
   Completed

5️⃣ Load to DB  ✅
   12,450 loaded
```

### Embedding Verification

Automatically detects and validates embedding columns:

```
✅ Found 1 embedding column(s)

📈 vector
   Dimension: 384
   Null Count: 0 (0.0%)
   Coverage: 100.0%
   Sample values: [0.123, -0.456, 0.789, ...]
```

### Data Quality Checks

| Check | Status | Details |
|-------|--------|---------|
| vector_id present | ✅ Pass | 0.0% null |
| text present | ✅ Pass | 0.5% null |
| content_type present | ✅ Pass | 0.0% null |
| platform present | ✅ Pass | 0.0% null |
| Unique vector_ids | ✅ Pass | 0 duplicates |
| Text populated | ✅ Pass | 0.5% empty |
| Platform distribution | ✅ Pass | instagram: 8000, tiktok: 4450 |

## Filtering Options

Sidebar filters (applied to all tabs):

- **Content Types**: profile, posts
- **Platforms**: instagram, tiktok
- **Follower Range**: Min to max slider
- **Verified Only**: Checkbox to filter verified profiles

## Usage Examples

### Verify Pipeline Completion

1. Open **Overview** tab
2. Check all 5 stages show ✅ green
3. Review record counts match expectations

### Check Embeddings Working

1. Open **Embeddings** tab
2. Verify "✅ Found N embedding column(s)"
3. Check coverage is 100% (or close)
4. Review sample embedding values look valid

### Test Semantic Search

1. Open **Search** tab
2. Enter query: "fitness influencer"
3. Adjust top-k results slider
4. Review matched profiles and scores

### Identify Data Issues

1. Open **Quality Checks** tab
2. Look for ⚠️ Warning or ❌ Fail statuses
3. Review details column for specifics
4. Filter data in sidebar to isolate issues

### Export Filtered Data

1. Apply filters in sidebar
2. Go to **Raw Data** tab
3. Select columns to display
4. Use browser tools to export (Streamlit has export button)

## Troubleshooting

### "No tables found at that URI"

- Check `Database URI` path is correct
- Ensure LanceDB directory exists
- Verify pipeline has run successfully

### "No embedding columns found"

- Pipeline may not have completed embedding stage
- Check **Overview** tab for pipeline status
- Verify `.embed.done` marker exists in pipeline output

### "Failed to list tables"

- Database path may be incorrect
- LanceDB directory may not exist
- Check file permissions

### Slow loading

- Reduce **Sample Size** slider (try 500 or 1000)
- Clear cache with **🔄 Refresh Data** button
- Close unused tabs in browser

## Performance Tips

1. **Sample Size**: Start with 1000 rows for fast loading
2. **Caching**: Data is cached for 5 minutes - use Refresh button to reload
3. **Filtering**: Apply filters to reduce data before exploring
4. **Column Selection**: Select fewer columns in Raw Data tab for faster rendering

## Architecture

The dashboard uses:
- **Streamlit**: Web framework
- **LanceDB**: Vector database connection
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical operations

Data flow:
```
LanceDB → Arrow Table → Pandas DataFrame → Streamlit UI
                ↓
         Pipeline Metadata (JSON markers)
```

## Customization

### Add Custom Metrics

Edit `render_metrics()` function to add new metric cards:

```python
def render_metrics(metadata: Dict[str, object], df: pd.DataFrame) -> None:
    # Add your custom metric
    col5.metric("Your Metric", calculate_your_metric(df))
```

### Add Custom Visualizations

Add new plots in `render_distribution_plots()`:

```python
def render_distribution_plots(df: pd.DataFrame) -> None:
    # Add your custom plot
    fig = px.scatter(df, x="followers", y="engagement_rate")
    st.plotly_chart(fig, use_container_width=True)
```

### Modify Quality Checks

Edit `render_data_quality_checks()` to add custom validation:

```python
# Add custom check
checks.append({
    "Check": "Your custom check",
    "Status": "✅ Pass" if condition else "❌ Fail",
    "Details": "Check details"
})
```

## Configuration

Dashboard configuration in `streamlit_app.py`:

```python
# Cache TTL (time to live)
@st.cache_data(ttl=300)  # 5 minutes

# Default sample size
sample_size = st.slider(..., value=1000)

# Default paths
db_uri = "data/lancedb"
pipeline_dir = "pipeline/output"
```

## API Reference

### Key Functions

- `list_tables(db_uri)`: Get available tables in database
- `load_table_metadata(db_uri, table_name)`: Load table schema and stats
- `load_sample_frame(db_uri, table_name, sample_size, seed)`: Load sample data
- `load_pipeline_metadata(output_dir)`: Load pipeline completion markers
- `check_embedding_columns(df)`: Verify embedding columns exist and are valid
- `render_pipeline_status(pipeline_meta)`: Display pipeline stage status
- `render_embedding_verification(df)`: Show embedding verification results
- `render_data_quality_checks(df)`: Run and display quality checks

## Next Steps

After verifying your pipeline:

1. ✅ Check all pipeline stages complete in Overview tab
2. ✅ Verify embeddings in Embeddings tab
3. ✅ Review quality checks pass in Quality Checks tab
4. ✅ Test search functionality in Search tab
5. 🚀 Use data in your application via LanceDB API

## Support

For issues with the dashboard:
- Check browser console for errors (F12)
- Review Streamlit terminal output
- Verify LanceDB connection works: `python scripts/inspect_lancedb.py`
