#!/usr/bin/env python3
"""
Quick utility to show a single database row in CSV format.
Usage examples:
  python show_row.py                    # Show first row
  python show_row.py 5                  # Show row index 5
  python show_row.py homedecor           # Find account containing 'homedecor'
"""
import sys
import pandas as pd
import lancedb
from pathlib import Path
import os

# Add src to path for imports - works from any directory
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Change to project root directory for relative paths to work
os.chdir(project_root)


def get_single_row_csv(row_index=None, account_name=None):
    """Get a single row from the LanceDB table in CSV format"""
    
    # Connect to the database
    db_path = "influencers_lancedb"
    table_name = "influencer_profiles"
    
    try:
        db = lancedb.connect(db_path)
        table = db.open_table(table_name)
    except Exception as e:
        raise Exception(f"Could not connect to database: {e}")
    
    # Get the row based on criteria
    if account_name:
        # Search by account name
        results = table.search().where(f"account LIKE '%{account_name}%'").limit(1).to_pandas()
        if results.empty:
            raise Exception(f"No account found containing '{account_name}'")
        row = results.iloc[0]
    else:
        # Get by index (default 0)
        if row_index is None:
            row_index = 0
        
        # Convert table to pandas and get the row
        df = table.to_pandas()
        if row_index >= len(df):
            raise Exception(f"Row index {row_index} out of range (max: {len(df)-1})")
        
        row = df.iloc[row_index]
    
    # Convert to CSV format
    df_single = pd.DataFrame([row])
    
    # Remove vector columns for cleaner output
    vector_columns = ['keyword_vector', 'profile_vector', 'content_vector']
    for col in vector_columns:
        if col in df_single.columns:
            df_single = df_single.drop(columns=[col])
    
    # Convert to CSV string
    csv_output = df_single.to_csv(index=False)
    return csv_output


def main():
    """Simple command line interface"""
    
    if len(sys.argv) > 2:
        print("Usage: python show_row.py [row_index_or_account_name]")
        print("Examples:")
        print("  python show_row.py           # Show first row")
        print("  python show_row.py 5         # Show row at index 5")
        print("  python show_row.py homedecor # Find account containing 'homedecor'")
        sys.exit(1)
    
    try:
        if len(sys.argv) == 1:
            # No arguments - show first row
            csv_output = get_single_row_csv(row_index=0)
            
        else:
            arg = sys.argv[1]
            
            # Try to parse as integer (row index)
            try:
                row_index = int(arg)
                csv_output = get_single_row_csv(row_index=row_index)
            except ValueError:
                # Treat as account name
                csv_output = get_single_row_csv(account_name=arg)
        
        # Print the CSV output
        print(csv_output, end='')
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()