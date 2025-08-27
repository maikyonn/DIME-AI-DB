"""
Unified data loader that combines BrightData CSV with LLM analysis JSONL files
into a single LanceDB table.
"""
import json
import pandas as pd
import lancedb
from pathlib import Path
from typing import Dict, List, Optional
import pyarrow as pa


class UnifiedDataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.csv_dir = self.data_dir / "brightdata-csv-dataset"
        self.llm_dir = self.data_dir / "llm-analysis"
        
    def load_csv_data(self, filename: str = "Snap Data.csv") -> pd.DataFrame:
        """Load the BrightData CSV file"""
        csv_path = self.csv_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        print(f"📊 Loading CSV data from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df):,} rows from CSV")
        return df
    
    def parse_batch_output(self, filename: str = "Batch Output 004.jsonl") -> Dict[int, Dict]:
        """Parse the LLM batch output JSONL file"""
        jsonl_path = self.llm_dir / filename
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
            
        print(f"🧠 Loading LLM analysis from {jsonl_path}")
        llm_data = {}
        processed_count = 0
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        custom_id = data.get('custom_id')
                        
                        if custom_id and custom_id.startswith('profile-'):
                            # Extract profile number (profile-1 -> 1, profile-2 -> 2, etc.)
                            profile_num = int(custom_id.replace('profile-', ''))
                            
                            # Extract the LLM response text (CSV format)
                            response_body = data.get('response', {}).get('body', {})
                            output = response_body.get('output', [])
                            
                            if len(output) > 1 and output[1].get('type') == 'message':
                                content = output[1].get('content', [])
                                if content and content[0].get('type') == 'output_text':
                                    csv_text = content[0].get('text', '').strip()
                                    
                                    # Parse CSV text: "7,5,2,furry,sketch,Berlin,artists,furryart"
                                    parts = csv_text.split(',')
                                    if len(parts) >= 8:
                                        llm_data[profile_num] = {
                                            'individual_vs_org_score': int(parts[0]),
                                            'generational_appeal_score': int(parts[1]),
                                            'professionalization_score': int(parts[2]),
                                            'keyword1': parts[3],
                                            'keyword2': parts[4],
                                            'keyword3': parts[5],
                                            'keyword4': parts[6],
                                            'keyword5': parts[7],
                                            'llm_processed': True
                                        }
                                        processed_count += 1
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        print(f"⚠️ Error processing line: {e}")
                        continue
        
        print(f"✅ Processed {processed_count:,} LLM analysis records")
        return llm_data
    
    def merge_data(self, csv_df: pd.DataFrame, llm_data: Dict[int, Dict]) -> pd.DataFrame:
        """Merge CSV data with LLM analysis data"""
        print("🔗 Merging CSV data with LLM analysis...")
        
        # Create empty columns for LLM data
        llm_columns = [
            'individual_vs_org_score', 'generational_appeal_score', 'professionalization_score',
            'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'llm_processed'
        ]
        
        for col in llm_columns:
            csv_df[col] = None
            
        csv_df['llm_processed'] = False
        
        # Merge LLM data by row index (profile-1 = row 0, profile-2 = row 1, etc.)
        merged_count = 0
        for profile_num, llm_record in llm_data.items():
            # Convert profile number to DataFrame index (profile-1 -> index 0)
            df_index = profile_num - 1
            
            if 0 <= df_index < len(csv_df):
                for key, value in llm_record.items():
                    csv_df.iloc[df_index, csv_df.columns.get_loc(key)] = value
                merged_count += 1
        
        print(f"✅ Merged {merged_count:,} records with LLM analysis")
        print(f"📊 Total records: {len(csv_df):,}")
        print(f"🧠 Records with LLM data: {csv_df['llm_processed'].sum():,}")
        
        return csv_df
    
    def create_lancedb_table(self, df: pd.DataFrame, 
                           db_path: str = "snap_data_lancedb", 
                           table_name: str = "influencer_profiles") -> None:
        """Create LanceDB table from merged DataFrame"""
        print(f"💾 Creating LanceDB table at {db_path}")
        
        # Connect to LanceDB
        db = lancedb.connect(db_path)
        
        # Drop table if it exists
        try:
            db.drop_table(table_name)
            print(f"🗑️ Dropped existing table: {table_name}")
        except:
            pass
        
        # Clean data for LanceDB
        df_clean = df.copy()
        
        # Convert numeric columns
        numeric_columns = ['individual_vs_org_score', 'generational_appeal_score', 'professionalization_score']
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle boolean columns - convert empty strings and various formats to proper booleans
        boolean_columns = [
            'is_private', 'is_verified', 'is_business_account', 
            'is_professional_account', 'is_joined_recently', 'has_channel'
        ]
        
        for col in boolean_columns:
            if col in df_clean.columns:
                # Convert to string first, then handle different boolean representations
                df_clean[col] = df_clean[col].astype(str).str.lower()
                df_clean[col] = df_clean[col].map({
                    'true': True, '1': True, 'yes': True, 'y': True,
                    'false': False, '0': False, 'no': False, 'n': False,
                    '': False, 'nan': False, 'none': False
                })
                # Fill any remaining NaN with False
                df_clean[col] = df_clean[col].fillna(False)
        
        # Convert other numeric columns
        numeric_columns_csv = ['followers', 'following', 'avg_engagement', 'posts_count', 'highlights_count']
        for col in numeric_columns_csv:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Fill remaining NaN values with appropriate defaults
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # String columns
                df_clean[col] = df_clean[col].fillna('')
            elif df_clean[col].dtype in ['int64', 'float64']:  # Numeric columns
                df_clean[col] = df_clean[col].fillna(0)
        
        # Debug: Show data types before creating table
        print("🔍 Data types after cleaning:")
        dtype_summary = df_clean.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            print(f"   {dtype}: {count} columns")
        
        # Show any remaining issues
        problematic_cols = []
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                problematic_cols.append(col)
        
        if problematic_cols:
            print(f"⚠️ Columns with NaN values: {problematic_cols}")
        
        # Create table
        table = db.create_table(table_name, df_clean)
        
        print(f"✅ Created LanceDB table '{table_name}' with {len(df_clean):,} records")
        print(f"📋 Schema: {len(df_clean.columns)} columns")
        
        return table
    
    def load_and_process_all(self, csv_filename: str = "Snap Data.csv",
                           jsonl_filename: str = "Batch Output 004.jsonl",
                           db_path: str = "snap_data_lancedb",
                           table_name: str = "influencer_profiles") -> None:
        """Main function to load, merge, and create LanceDB table"""
        print("🚀 Starting unified data loading process...")
        
        # Load CSV data
        csv_df = self.load_csv_data(csv_filename)
        
        # Parse LLM analysis
        llm_data = self.parse_batch_output(jsonl_filename)
        
        # Merge data
        merged_df = self.merge_data(csv_df, llm_data)
        
        # Create LanceDB table
        table = self.create_lancedb_table(merged_df, db_path, table_name)
        
        print("🎉 Unified data loading complete!")
        return table


def main():
    """Main function for standalone execution"""
    loader = UnifiedDataLoader()
    loader.load_and_process_all()


if __name__ == "__main__":
    main()