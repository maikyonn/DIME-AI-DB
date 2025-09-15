#!/usr/bin/env python3
"""
Manual database building script for custom CSV files
"""
import sys
from pathlib import Path
import os

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Change to project root for relative paths
os.chdir(project_root)

from src.data.vector_database_builder import VectorDatabaseBuilder

def build_with_custom_csv():
    """Build vector database with custom CSV file"""
    print("🚀 Building Vector Database with Custom CSV")
    print("=" * 60)
    
    # Your CSV file path
    csv_file = "data/dataset-3-100k-email/combined_snap_data_20250903_143736.english_20250903_151630_with_lance_id.csv"
    
    # Initialize builder
    builder = VectorDatabaseBuilder(
        data_dir="data/dataset-3-100k-email",  # Point to your data directory
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    try:
        # Build database
        table = builder.create_vector_database(
            csv_filename="combined_snap_data_20250903_143736.english_20250903_151630_with_lance_id.csv",
            jsonl_filename=None,  # No LLM data yet, will use CSV only
            db_path="english_influencers_vectordb",
            table_name="english_profiles"
        )
        
        print("\n🎉 Vector database created successfully!")
        print(f"   • Records: {table.count_rows():,}")
        print(f"   • Database: english_influencers_vectordb")
        print(f"   • Table: english_profiles")
        
        return table
        
    except Exception as e:
        print(f"\n❌ Error building database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_with_custom_csv()