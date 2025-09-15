#!/usr/bin/env python3
"""
Quick runner script to rebuild database and launch visualization dashboard.
"""
import subprocess
import sys
from pathlib import Path
import time


def check_data_files():
    """Check if required data files exist"""
    data_dir = Path("data")
    csv_file = data_dir / "brightdata-csv-dataset" / "Snap Data.csv"
    jsonl_file = data_dir / "llm-analysis" / "Batch Output 004.jsonl"
    
    if not csv_file.exists():
        print(f"❌ Missing CSV file: {csv_file}")
        return False
    
    if not jsonl_file.exists():
        print(f"❌ Missing JSONL file: {jsonl_file}")
        return False
    
    print("✅ All required data files found")
    return True


def rebuild_database():
    """Run the database rebuild script"""
    print("🔄 Rebuilding database...")
    try:
        result = subprocess.run([sys.executable, "rebuild_database_simple.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Database rebuild failed:")
        print(e.stdout)
        print(e.stderr)
        return False


def launch_streamlit():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Streamlit dashboard...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "visualize_data.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed by user")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")


def main():
    print("🎯 Instagram Profile Analytics Runner")
    print("=" * 50)
    
    # Check data files
    if not check_data_files():
        print("\n❌ Please ensure your data files are in the correct locations:")
        print("   • data/brightdata-csv-dataset/Snap Data.csv")
        print("   • data/llm-analysis/Batch Output 004.jsonl")
        sys.exit(1)
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Rebuild database and launch dashboard")
    print("2. Just rebuild database")
    print("3. Just launch dashboard (database must exist)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        if rebuild_database():
            print("\n" + "="*50)
            time.sleep(2)  # Brief pause
            launch_streamlit()
    
    elif choice == "2":
        rebuild_database()
    
    elif choice == "3":
        # Check if database exists
        db_path = Path("influencers_lancedb")
        if not db_path.exists():
            print(f"❌ Database not found at {db_path}")
            print("Run option 1 or 2 first to create the database")
            sys.exit(1)
        launch_streamlit()
    
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()