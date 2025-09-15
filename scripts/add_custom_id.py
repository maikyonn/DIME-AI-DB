#!/usr/bin/env python3
"""
Add lance_db_id column to CSV files
Adds a unique lance_db_id column as the first column in each CSV file
"""

import csv
import sys
import os
import argparse
from typing import List


def add_lance_db_id_to_csv(csv_file: str, start_id: int, overwrite: bool = False) -> int:
    """
    Add lance_db_id column to a CSV file as the first column
    
    Args:
        csv_file: Path to the CSV file
        start_id: Starting ID number for this file
        overwrite: Whether to overwrite existing lance_db_id column
        
    Returns:
        Next available ID number (last_id + 1)
    """
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return start_id
    
    print(f"📝 Processing: {csv_file}")
    
    # Read the existing CSV
    rows = []
    fieldnames = []
    
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                print(f"❌ No fieldnames found in {csv_file}")
                return start_id
            
            # Check if lance_db_id already exists
            if 'lance_db_id' in fieldnames and not overwrite:
                print(f"⚠️  lance_db_id column already exists in {csv_file}, skipping")
                # Count rows to return the correct next ID
                row_count = sum(1 for _ in reader)
                return start_id + row_count
            
            rows = list(reader)
    
    except Exception as e:
        print(f"❌ Error reading {csv_file}: {e}")
        return start_id
    
    if not rows:
        print(f"⚠️  No data rows found in {csv_file}")
        return start_id
    
    # Create new fieldnames with lance_db_id as first column
    if 'lance_db_id' not in fieldnames:
        new_fieldnames = ['lance_db_id'] + fieldnames
    else:
        # If overwriting, keep existing column order but ensure lance_db_id is first
        new_fieldnames = ['lance_db_id'] + [f for f in fieldnames if f != 'lance_db_id']
        if overwrite:
            print(f"🔄 Overwriting existing lance_db_id column in {csv_file}")
    
    # Add lance_db_id to each row
    current_id = start_id
    for row in rows:
        row['lance_db_id'] = current_id
        current_id += 1
    
    # Write the updated CSV
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✅ Added lance_db_id column to {csv_file} (IDs {start_id} to {current_id-1})")
        return current_id
    
    except Exception as e:
        print(f"❌ Error writing {csv_file}: {e}")
        return start_id


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Add lance_db_id column to CSV files")
    parser.add_argument("csv_files", nargs='+', help="CSV files to process")
    parser.add_argument("--start-id", type=int, default=1, help="Starting ID number (default: 1)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing lance_db_id column if it exists")
    
    args = parser.parse_args()
    
    print("🚀 Adding lance_db_id column to CSV files")
    print("=" * 50)
    
    current_id = args.start_id
    total_rows = 0
    
    for csv_file in args.csv_files:
        start_id_for_file = current_id
        current_id = add_lance_db_id_to_csv(csv_file, current_id, args.overwrite)
        rows_added = current_id - start_id_for_file
        total_rows += rows_added
    
    print("=" * 50)
    print(f"✅ Completed processing {len(args.csv_files)} CSV files")
    print(f"📊 Total rows processed: {total_rows}")
    print(f"🆔 ID range: {args.start_id} to {current_id-1}")


if __name__ == "__main__":
    main()
