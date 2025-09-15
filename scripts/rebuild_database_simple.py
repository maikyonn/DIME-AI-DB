#!/usr/bin/env python3
"""
Complete database rebuild script - LanceDB + Vector Database
Usage: python rebuild_database_simple.py <dataset_directory> [--vectors] [--test] [--interactive]
"""
import sys
import argparse
from pathlib import Path
import os

# Add src to path for imports - works from any directory
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Change to project root directory for relative paths to work
os.chdir(project_root)

from src.data.unified_data_loader import UnifiedDataLoader
from src.data.vector_database_builder import VectorDatabaseBuilder
from src.search.vector_search import VectorSearchEngine


def find_csv_file(dataset_dir):
    """Find the first CSV file in the dataset directory"""
    dataset_path = Path(dataset_dir)
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    
    # Return the largest CSV file (likely the main dataset)
    csv_file = max(csv_files, key=lambda f: f.stat().st_size)
    return csv_file.name


def build_vector_database(dataset_dir, csv_filename):
    """Build vector database with embeddings"""
    print("🤖 Building vector database with embeddings...")
    
    builder = VectorDatabaseBuilder(data_dir=dataset_dir)
    table = builder.create_vector_database(
        csv_filename=csv_filename,
        jsonl_filename=None,
        db_path="influencers_vectordb", 
        table_name="influencer_profiles"
    )
    
    print(f"✅ Vector database: {table.count_rows():,} records")
    return table

def validate_database():
    """Test search functionality"""
    print("🧪 Testing search...")
    try:
        engine = VectorSearchEngine()
        stats = engine.get_database_stats()
        results = engine.search("test", limit=1)
        print(f"✅ Search working: {stats['total_records']:,} records, {stats['vector_dimensions']} dims")
        return True
    except Exception as e:
        print(f"⚠️ Search test failed: {e}")
        return False

def interactive_search():
    """Interactive search mode"""
    print("\n🚀 Interactive search - type queries or 'quit'")
    engine = VectorSearchEngine()
    
    while True:
        try:
            query = input("\nSearch: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
                
            results = engine.search(query, limit=3)
            if results.empty:
                print("No results")
                continue
                
            for i, row in results.iterrows():
                print(f"{i+1}. @{row.get('account', 'N/A')} - {row.get('followers', 0):,} followers")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Search ended")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Rebuild DIME-AI-DB")
    parser.add_argument("dataset_dir", nargs="?", 
                       default="/Users/maikyon/Documents/Programming/100k-filter/DIME-AI-DB/data/dataset-3-100k-email",
                       help="Dataset directory path")
    parser.add_argument("csv_filename", nargs="?", help="CSV filename (auto-detected if not provided)")
    parser.add_argument("--vectors", action="store_true", help="Also build vector database with embeddings")
    parser.add_argument("--test", action="store_true", help="Test search functionality")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive search")
    
    args = parser.parse_args()
    
    print("🔄 DIME-AI-DB Rebuild")
    
    try:
        # Auto-detect CSV file if not provided
        if not args.csv_filename:
            args.csv_filename = find_csv_file(args.dataset_dir)
        
        print(f"📁 Dataset: {args.dataset_dir}")
        print(f"📄 CSV: {args.csv_filename}")
        
        # Build LanceDB
        print("\n🏗️ Building LanceDB...")
        loader = UnifiedDataLoader(dataset_dir=args.dataset_dir)
        table = loader.load_and_process_all(
            csv_filename=args.csv_filename,
            jsonl_filename=None,
            db_path="influencers_lancedb",
            table_name="influencer_profiles"
        )
        
        print(f"✅ LanceDB: {table.count_rows():,} records")
        
        # Build vector database if requested
        if args.vectors:
            build_vector_database(args.dataset_dir, args.csv_filename)
        
        # Test functionality
        if args.test or args.vectors:
            validate_database()
        
        # Interactive mode
        if args.interactive:
            interactive_search()
        
        print(f"\n✅ Build complete!")
        if not args.vectors:
            print("💡 Add --vectors for search functionality")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())