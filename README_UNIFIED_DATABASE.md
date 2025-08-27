# Unified Database Approach

This document describes the new unified approach for combining BrightData CSV with LLM analysis into a single LanceDB table.

## 📁 Data Organization

```
data/
├── brightdata-csv-dataset/
│   ├── Snap Data.csv              # Full dataset (~21K profiles)
│   └── Snap Data_1000rows.csv     # Sample dataset (1K profiles)
└── llm-analysis/
    ├── Batch Input 004.jsonl      # GPT-5 analysis prompts
    └── Batch Output 004.jsonl     # GPT-5 analysis results
```

## 🔗 Data Linkage

The files are linked by profile numbers:
- `custom_id: "profile-1"` in JSONL files → Row 1 in CSV
- `custom_id: "profile-2"` in JSONL files → Row 2 in CSV
- And so on...

## 🚀 Quick Start

1. **Rebuild database with unified approach:**
   ```bash
   python3 rebuild_database_simple.py
   ```

2. **Launch interactive visualization dashboard:**
   ```bash
   python3 run_visualization.py
   ```
   Or directly:
   ```bash
   streamlit run visualize_data.py
   ```

## 📊 Database Schema

The unified LanceDB table contains:

### Original CSV Fields
- `account`, `full_name`, `biography`
- `followers`, `following`, `posts_count`
- `is_business_account`, `is_verified`
- And all other original BrightData fields (~30 columns)

### LLM Analysis Fields
- `individual_vs_org_score` (0-10): Individual vs Organization
- `generational_appeal_score` (0-10): Gen Z appeal
- `professionalization_score` (0-10): Professional/brand level
- `keyword1` through `keyword5`: Descriptive keywords
- `llm_processed` (boolean): Whether LLM analysis is available

## 🎯 Visualization Features

The Streamlit dashboard provides:

1. **LLM Analysis Tab:**
   - Score distributions for all three metrics
   - Keyword frequency analysis
   - Top 20 most common keywords

2. **Followers Tab:**
   - Follower count distribution
   - Log-scale histogram for better visualization

3. **Engagement Tab:**
   - Engagement vs followers scatter plot
   - Color-coded by LLM scores when available

4. **Business Tab:**
   - Business vs personal account breakdown
   - Correlation with individual vs org scores

5. **Data Tab:**
   - Raw data sample view
   - CSV export functionality

## 🔧 Technical Details

### Files Structure
- `src/data/unified_data_loader.py`: Core data combination logic
- `rebuild_database_simple.py`: Simple database rebuild (no vectors)
- `visualize_data.py`: Streamlit dashboard
- `run_visualization.py`: Interactive runner script

### Key Features
- **No Vector Embeddings**: Simplified approach focusing on structured data
- **Automatic Data Linking**: Smart matching of JSONL custom_id to CSV rows
- **Error Handling**: Graceful handling of missing or malformed data
- **Performance**: Efficient data loading and caching
- **Extensible**: Easy to add new analysis fields or visualization

## 🎨 Customization

### Adding New Visualizations
Edit `visualize_data.py` and add new functions to create additional charts.

### Modifying Database Schema
Edit `unified_data_loader.py` to change how data is processed and combined.

### Different Data Files
Update file paths in the loader to use different CSV or JSONL sources.

## 📈 Performance

- **Database Creation**: ~30-60 seconds for 21K profiles
- **Dashboard Loading**: ~2-3 seconds for initial data load
- **Visualization Rendering**: <1 second for most charts
- **Memory Usage**: ~200-500MB for full dataset

## 🔄 Migration from Old Approach

The new approach is designed to coexist with existing code. Old vector-based functionality is preserved in the original files, while this provides a simpler, more direct approach to data analysis.

## 💡 Best Practices

1. **Always rebuild** the database after changing data files
2. **Use the sample dataset** for testing new features
3. **Export data** from the dashboard for external analysis
4. **Check data coverage** in the sidebar stats before drawing conclusions