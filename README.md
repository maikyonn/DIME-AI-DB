# DIME AI Database

**Unified database system for Instagram profile analysis combining BrightData scraping with OpenAI GPT-5 LLM analysis.**

## 🏗️ Architecture

This repository contains the database layer and analysis tools for processing Instagram profile data. The system combines:

- **BrightData CSV**: Raw Instagram profile data (followers, posts, bio, etc.)
- **OpenAI GPT-5 Analysis**: AI-generated scores and keywords via batch API
- **LanceDB Storage**: Unified columnar database for fast analytics
- **Streamlit Dashboard**: Interactive data exploration and visualization

## 📁 Project Structure

```
DIME-AI-DB/
├── src/
│   ├── data/
│   │   └── unified_data_loader.py    # Core data combination logic
│   ├── services/                     # Future: API services
│   └── api/                         # Future: REST endpoints
├── scripts/                         # Utility scripts
├── utils/                          # Helper utilities
├── rebuild_database_simple.py       # Database rebuild script
├── visualize_data.py               # Streamlit dashboard
├── run_visualization.py            # Interactive runner
├── batch_process_snap_data.py      # GPT-5 batch processing
├── combine_batch_results.py        # Batch result combination
└── requirements.txt                # Python dependencies
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

### 2. Prepare Data Files

**⚠️ Data files are NOT included in this repository for privacy/size reasons.**

Create the following data structure in your project directory:

```
data/
├── brightdata-csv-dataset/
│   ├── Snap Data.csv              # Your BrightData CSV export
│   └── Snap Data_1000rows.csv     # Sample dataset (optional)
└── llm-analysis/
    ├── Batch Input 004.jsonl      # GPT-5 batch input
    └── Batch Output 004.jsonl     # GPT-5 batch output
```

### 3. Build Database

```bash
# Rebuild unified database
python3 rebuild_database_simple.py
```

### 4. Launch Dashboard

```bash
# Interactive launcher
python3 run_visualization.py

# Or directly
streamlit run visualize_data.py
```

## 📊 Database Schema

The unified LanceDB table combines:

### Original BrightData Fields
- Profile info: `account`, `full_name`, `biography`
- Metrics: `followers`, `following`, `avg_engagement`, `posts_count`
- Flags: `is_business_account`, `is_verified`, `is_private`
- Content: `posts`, `highlights`, `business_category_name`

### LLM Analysis Fields
- **Individual vs Org Score** (0-10): Individual vs Organization classification
- **Generational Appeal Score** (0-10): Gen Z appeal assessment  
- **Professionalization Score** (0-10): Professional/brand level
- **Keywords 1-5**: Distinctive profile descriptors
- **LLM Processed**: Boolean flag for analysis availability

## 🔍 Data Exploration

The Streamlit dashboard provides:

### **Analytics Tabs**
- **🧠 LLM Analysis**: Score distributions and keyword frequency
- **👥 Followers**: Follower count analysis and distributions
- **📈 Engagement**: Engagement vs followers scatter plots
- **🏢 Business**: Business vs personal account breakdown
- **🔍 Data Explorer**: Advanced filtering and custom views

### **Advanced Filtering**
- Text search: Account name, biography content
- Engagement metrics: Follower count, engagement rate, posts count
- Account types: Business vs personal, verified status
- LLM scores: All three analysis dimensions
- Keywords: Search across all AI-generated keywords

### **Export Options**
- Filtered data export with timestamps
- Full dataset CSV download
- Real-time statistics for filtered results

## 🔧 Technical Details

### **Database Technology**
- **LanceDB**: Columnar vector database built on Apache Arrow
- **Local Storage**: File-based, no server required
- **ACID Transactions**: Atomic operations with rollback capability
- **Schema Evolution**: Easy to add new analysis columns

### **Data Flow**
```
CSV (profiles) + JSONL (LLM scores) 
    ↓ unified_data_loader.py
    ↓ Data cleaning & type conversion
    ↓ LanceDB table (columnar storage)
    ↓ Streamlit dashboard (interactive analysis)
```

### **Performance**
- **Storage**: Columnar compression (~2.8GB for 21K profiles)
- **Queries**: Optimized for filtering and aggregation
- **Memory**: Efficient loading of required columns only
- **Caching**: Streamlit data caching for responsive UI

## 🔗 Data Linking

The system links data files via:
- **JSONL custom_id**: `"profile-1"`, `"profile-2"`, etc.
- **CSV row mapping**: custom_id number corresponds to CSV row
- **Example**: `profile-1` → Row 1 in CSV, `profile-2` → Row 2, etc.

## 📈 Use Cases

### **Influencer Discovery**
- High engagement micro-influencers (1K-50K followers, >5% engagement)
- Gen Z focused creators (high generational appeal scores)
- Professional brand accounts (high professionalization scores)

### **Market Analysis**  
- Business vs personal account trends
- Engagement rate distributions by follower count
- Keyword clustering and content themes
- Verification status correlation with metrics

### **Data Export**
- Custom filtered datasets for external analysis
- Campaign targeting lists
- Competitor analysis exports

## 🛠️ Development

### **Adding New LLM Analysis**
1. Process new data through OpenAI Batch API
2. Update `unified_data_loader.py` to handle new score columns
3. Add visualization to `visualize_data.py`
4. Rebuild database with `rebuild_database_simple.py`

### **Extending Visualizations**
Edit `visualize_data.py` to add new charts or analysis views. The system is designed to be easily extensible.

## 📚 Dependencies

- **Core**: pandas, numpy, lancedb, pyarrow
- **Visualization**: streamlit, plotly
- **LLM Integration**: openai (for batch processing)
- **Web Framework**: flask (for future API development)

## 🔒 Privacy & Security

- **No Data in Repo**: All actual data files are gitignored
- **Local Processing**: All analysis runs locally
- **No External Calls**: Dashboard works offline once data is loaded
- **User Control**: Complete control over data storage and access

---

*This repository contains only the code and tools for database management. Data files must be provided separately and are not tracked in version control.*