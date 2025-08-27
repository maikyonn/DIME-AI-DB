#!/usr/bin/env python3
"""
Setup script for DIME AI Database
"""
import subprocess
import sys
from pathlib import Path


def setup_environment():
    """Setup the development environment"""
    print("🔧 Setting up DIME AI Database environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check for data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("📁 Creating data directory structure...")
        data_dir.mkdir(exist_ok=True)
        (data_dir / "brightdata-csv-dataset").mkdir(exist_ok=True)
        (data_dir / "llm-analysis").mkdir(exist_ok=True)
        
        print("✅ Data directories created")
        print("💡 Please add your data files:")
        print("   • data/brightdata-csv-dataset/Snap Data.csv")
        print("   • data/llm-analysis/Batch Output 004.jsonl")
    else:
        print("✅ Data directory exists")
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Add your data files to data/ directories")
    print("2. Run: python3 rebuild_database_simple.py")  
    print("3. Run: python3 run_visualization.py")


if __name__ == "__main__":
    setup_environment()