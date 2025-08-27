#!/usr/bin/env python3
"""
Simplified database rebuild script that combines CSV and JSONL data
into a single LanceDB table without vector embeddings.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.unified_data_loader import UnifiedDataLoader


def main():
    """
    Main function to rebuild the database with combined data
    """
    print("🔄 Rebuilding database with unified CSV + LLM data...")
    print("💡 This version skips vector embeddings for simplicity")
    print("📁 Using organized data structure: data/brightdata-csv-dataset/ and data/llm-analysis/")
    print()
    
    try:
        # Initialize the unified data loader
        loader = UnifiedDataLoader()
        
        # Load and process all data
        table = loader.load_and_process_all(
            csv_filename="Snap Data.csv",  # Use full dataset
            jsonl_filename="Batch Output 004.jsonl",
            db_path="snap_data_lancedb",
            table_name="influencer_profiles"
        )
        
        print("\n📊 Database Summary:")
        print(f"   • Table: influencer_profiles")
        print(f"   • Location: snap_data_lancedb/")
        print(f"   • Records: {table.count_rows():,}")
        print(f"   • Columns: {len(table.schema)}")
        
        print("\n🔍 Sample columns:")
        schema_names = [field.name for field in table.schema]
        for i, col in enumerate(schema_names[:10]):  # Show first 10 columns
            print(f"   • {col}")
        if len(schema_names) > 10:
            print(f"   ... and {len(schema_names) - 10} more columns")
            
        print("\n🧠 LLM Analysis columns:")
        llm_cols = ['individual_vs_org_score', 'generational_appeal_score', 'professionalization_score', 
                   'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']
        for col in llm_cols:
            if col in schema_names:
                print(f"   ✅ {col}")
        
        print("\n✅ Database rebuild complete!")
        print("💡 Use visualize_data.py to explore the data")
        
    except Exception as e:
        print(f"\n❌ Error during database rebuild: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()