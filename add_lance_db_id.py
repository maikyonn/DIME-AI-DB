#!/usr/bin/env python3
"""
Add lance_db_id column to CSV file for batch processing
"""
import pandas as pd
import sys

def add_lance_db_id(input_csv, output_csv=None):
    """Add lance_db_id column to CSV file"""
    print(f"📊 Loading CSV: {input_csv}")
    
    # Read CSV with low_memory=False to handle large files
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"✅ Loaded {len(df):,} rows")
    
    # Add lance_db_id column (1-based indexing)
    df['lance_db_id'] = range(1, len(df) + 1)
    print(f"✅ Added lance_db_id column (1 to {len(df)})")
    
    # Set output filename
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_with_lance_id.csv')
    
    # Save updated CSV
    print(f"💾 Saving to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df):,} rows with lance_db_id column")
    
    return output_csv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_lance_db_id.py input.csv [output.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    add_lance_db_id(input_file, output_file)