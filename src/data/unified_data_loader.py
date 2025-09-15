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
    def __init__(self, dataset_dir: str):
        """
        Initialize with a specific dataset directory path
        Args:
            dataset_dir: Path to dataset directory containing CSV and batch folders
        """
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        self.csv_dir = self.dataset_dir
        self.llm_dir = self.dataset_dir
        
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """Load the CSV file from the dataset directory"""
        csv_path = self.csv_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        print(f"📊 Loading CSV data from {csv_path}")
        
        # Try different parsing strategies for large/problematic CSV files
        try:
            # First try with low_memory=False to handle mixed types
            df = pd.read_csv(csv_path, low_memory=False)
        except pd.errors.ParserError as e:
            print(f"⚠️ Standard parsing failed: {e}")
            print("🔧 Trying alternative parsing with Python engine...")
            try:
                df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip', low_memory=False)
                print("⚠️ Some malformed lines were skipped")
            except Exception as e2:
                print(f"⚠️ Python engine failed: {e2}")
                print("🔧 Trying with chunked reading...")
                # Read in chunks and combine
                chunk_list = []
                chunk_size = 10000
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size, engine='python', 
                                       on_bad_lines='skip', low_memory=False):
                    chunk_list.append(chunk)
                    print(f"   📦 Loaded chunk: {len(chunk):,} rows")
                df = pd.concat(chunk_list, ignore_index=True)
                print(f"📦 Combined {len(chunk_list)} chunks")
        
        print(f"✅ Loaded {len(df):,} rows from CSV")
        return df
    
    def load_organized_results(self) -> Dict[int, Dict]:
        """Load the pre-organized batch results CSV file (preferred method)"""
        organized_path = Path("organized_results/database_ready_results.csv")
        if not organized_path.exists():
            return None
            
        print(f"🎯 Loading organized batch results from {organized_path}")
        df = pd.read_csv(organized_path)
        print(f"✅ Loaded {len(df):,} processed profiles from organized results")
        
        llm_data = {}
        for _, row in df.iterrows():
            lance_db_id = int(row['lance_db_id'])
            llm_data[lance_db_id] = {
                'individual_vs_org_score': float(row['individual_vs_org']) if pd.notna(row['individual_vs_org']) else None,
                'generational_appeal_score': float(row['generational_appeal']) if pd.notna(row['generational_appeal']) else None,
                'professionalization_score': float(row['professionalization']) if pd.notna(row['professionalization']) else None,
                'relationship_status_score': float(row['relationship_status']) if pd.notna(row['relationship_status']) else None,
                'keyword1': str(row['keyword1']) if pd.notna(row['keyword1']) else '',
                'keyword2': str(row['keyword2']) if pd.notna(row['keyword2']) else '',
                'keyword3': str(row['keyword3']) if pd.notna(row['keyword3']) else '',
                'keyword4': str(row['keyword4']) if pd.notna(row['keyword4']) else '',
                'keyword5': str(row['keyword5']) if pd.notna(row['keyword5']) else '',
                'keyword6': str(row['keyword6']) if pd.notna(row['keyword6']) else '',
                'keyword7': str(row['keyword7']) if pd.notna(row['keyword7']) else '',
                'keyword8': str(row['keyword8']) if pd.notna(row['keyword8']) else '',
                'keyword9': str(row['keyword9']) if pd.notna(row['keyword9']) else '',
                'keyword10': str(row['keyword10']) if pd.notna(row['keyword10']) else '',
                'llm_processed': True
            }
        
        return llm_data

    def parse_batch_output(self, filename: Optional[str] = None) -> Dict[int, Dict]:
        """Parse LLM batch output JSONL files - if filename is None, loads all batch output files"""
        # First try to load organized results
        organized_data = self.load_organized_results()
        if organized_data is not None:
            return organized_data
            
        # Fallback to original method
        if filename is None:
            return self.parse_all_batch_outputs()
        
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
                            # Extract lance_db_id (profile-1 -> 1, profile-2 -> 2, etc.)
                            lance_db_id = int(custom_id.replace('profile-', ''))
                            
                            # Extract the LLM response text (CSV format)
                            response_body = data.get('response', {}).get('body', {})
                            output = response_body.get('output', [])
                            
                            if len(output) > 1 and output[1].get('type') == 'message':
                                content = output[1].get('content', [])
                                if content and content[0].get('type') == 'output_text':
                                    csv_text = content[0].get('text', '').strip()
                                    
                                    # Parse CSV text: "7,5,2,3,furry,sketch,Berlin,artists,furryart"
                                    if self._is_valid_csv_response(csv_text):
                                        parts = csv_text.split(',')
                                        # Clean and convert scores with error handling
                                        scores = self._parse_scores(parts[:4])
                                        keywords = parts[4:14] if len(parts) > 4 else [''] * 10
                                        
                                        llm_data[lance_db_id] = {
                                            'individual_vs_org_score': scores[0],
                                            'generational_appeal_score': scores[1],
                                            'professionalization_score': scores[2],
                                            'relationship_status_score': scores[3],
                                            'keyword1': keywords[0].strip(),
                                            'keyword2': keywords[1].strip() if len(keywords) > 1 else '',
                                            'keyword3': keywords[2].strip() if len(keywords) > 2 else '',
                                            'keyword4': keywords[3].strip() if len(keywords) > 3 else '',
                                            'keyword5': keywords[4].strip() if len(keywords) > 4 else '',
                                            'keyword6': keywords[5].strip() if len(keywords) > 5 else '',
                                            'keyword7': keywords[6].strip() if len(keywords) > 6 else '',
                                            'keyword8': keywords[7].strip() if len(keywords) > 7 else '',
                                            'keyword9': keywords[8].strip() if len(keywords) > 8 else '',
                                            'keyword10': keywords[9].strip() if len(keywords) > 9 else '',
                                            'llm_processed': True
                                        }
                                        processed_count += 1
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        print(f"⚠️ Error processing line: {e}")
                        continue
        
        print(f"✅ Processed {processed_count:,} LLM analysis records")
        return llm_data
    
    def parse_all_batch_outputs(self) -> Dict[int, Dict]:
        """Parse all batch output JSONL files with correct lance_db_id mapping"""
        # Find all batch result files in batch_XXX folders
        batch_files = list(self.llm_dir.glob("batch_*/*_results.jsonl"))
        
        if not batch_files:
            raise FileNotFoundError(f"No batch result files found in {self.llm_dir}")
        
        # Sort by batch number (extract number from folder/filename)
        def extract_batch_number(filepath):
            import re
            match = re.search(r'batch_(\d+)', str(filepath))
            return int(match.group(1)) if match else 0
        
        batch_files.sort(key=extract_batch_number)
        
        print(f"🧠 Loading LLM analysis from {len(batch_files)} batch files:")
        for file in batch_files:
            print(f"   📄 {file.name}")
        
        combined_llm_data = {}
        total_processed = 0
        
        for jsonl_path in batch_files:
            batch_number = extract_batch_number(jsonl_path)
            print(f"\n📖 Processing {jsonl_path.name} (Batch {batch_number})...")
            
            file_processed = 0
            
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            custom_id = data.get('custom_id')
                            
                            if custom_id and custom_id.startswith('profile-'):
                                # Extract lance_db_id from custom_id (profile-1 -> 1, profile-2 -> 2, etc.)
                                lance_db_id = int(custom_id.replace('profile-', ''))
                                
                                # Extract the LLM response text (CSV format)
                                response_body = data.get('response', {}).get('body', {})
                                output = response_body.get('output', [])
                                
                                if len(output) > 1 and output[1].get('type') == 'message':
                                    content = output[1].get('content', [])
                                    if content and content[0].get('type') == 'output_text':
                                        csv_text = content[0].get('text', '').strip()
                                        
                                        # Parse CSV text: "7,5,2,3,furry,sketch,Berlin,artists,furryart"
                                        if self._is_valid_csv_response(csv_text):
                                            parts = csv_text.split(',')
                                            # Clean and convert scores with error handling
                                            scores = self._parse_scores(parts[:4])
                                            keywords = parts[4:14] if len(parts) > 4 else [''] * 10
                                            
                                            combined_llm_data[lance_db_id] = {
                                                'individual_vs_org_score': scores[0],
                                                'generational_appeal_score': scores[1],
                                                'professionalization_score': scores[2],
                                                'relationship_status_score': scores[3],
                                                'keyword1': keywords[0].strip(),
                                                'keyword2': keywords[1].strip() if len(keywords) > 1 else '',
                                                'keyword3': keywords[2].strip() if len(keywords) > 2 else '',
                                                'keyword4': keywords[3].strip() if len(keywords) > 3 else '',
                                                'keyword5': keywords[4].strip() if len(keywords) > 4 else '',
                                                'keyword6': keywords[5].strip() if len(keywords) > 5 else '',
                                                'keyword7': keywords[6].strip() if len(keywords) > 6 else '',
                                                'keyword8': keywords[7].strip() if len(keywords) > 7 else '',
                                                'keyword9': keywords[8].strip() if len(keywords) > 8 else '',
                                                'keyword10': keywords[9].strip() if len(keywords) > 9 else '',
                                                'llm_processed': True,
                                                'source_batch': jsonl_path.name
                                            }
                                            file_processed += 1
                                            total_processed += 1
                        except (json.JSONDecodeError, ValueError, KeyError) as e:
                            print(f"   ⚠️ Error processing line in {jsonl_path.name}: {e}")
                            continue
            
            print(f"   ✅ {jsonl_path.name}: {file_processed:,} records processed")
        
        print(f"\n🎯 Combined Results:")
        print(f"   Total LLM records: {total_processed:,}")
        print(f"   lance_db_id range: {min(combined_llm_data.keys()) if combined_llm_data else 'N/A'} - {max(combined_llm_data.keys()) if combined_llm_data else 'N/A'}")
        
        # Show batch distribution
        batch_counts = {}
        for record in combined_llm_data.values():
            batch = record.get('source_batch', 'unknown')
            batch_counts[batch] = batch_counts.get(batch, 0) + 1
        
        print(f"   Distribution by batch:")
        for batch, count in sorted(batch_counts.items()):
            print(f"     📊 {batch}: {count:,} records")
        
        return combined_llm_data
    
    def _is_valid_csv_response(self, text: str) -> bool:
        """Check if the LLM response looks like valid CSV data"""
        if not text or len(text.strip()) == 0:
            return False
        
        # Skip responses that are clearly explanatory text
        if any(keyword in text.lower() for keyword in [
            'analysis:', 'individual vs', 'organization', 'please provide', 
            'bio text', 'missing', 'fetch', 'access'
        ]):
            return False
        
        # Check if it starts with numbers or has comma-separated structure
        parts = text.split(',')
        if len(parts) < 4:  # Need at least 4 scores
            return False
            
        # First 4 parts should be convertible to numbers
        for i in range(min(4, len(parts))):
            part = parts[i].strip()
            if not part or not self._can_convert_to_number(part):
                return False
                
        return True
    
    def _can_convert_to_number(self, text: str) -> bool:
        """Check if text can be converted to a number"""
        try:
            float(text)
            return True
        except (ValueError, TypeError):
            return False
    
    def _parse_scores(self, score_parts: list) -> list:
        """Parse score parts with error handling, returns [int, int, int, int]"""
        scores = []
        for i, part in enumerate(score_parts):
            try:
                # Handle both int and float strings
                score = int(float(part.strip())) if part.strip() else 0
                # Clamp to valid range (0-10)
                score = max(0, min(10, score))
                scores.append(score)
            except (ValueError, TypeError):
                scores.append(0)  # Default to 0 for invalid scores
        
        # Ensure we always return 4 scores
        while len(scores) < 4:
            scores.append(0)
            
        return scores[:4]  # Only return first 4
    
    def merge_data(self, csv_df: pd.DataFrame, llm_data: Dict[int, Dict]) -> pd.DataFrame:
        """Merge CSV data with LLM analysis data"""
        print("🔗 Merging CSV data with LLM analysis...")
        
        # Create empty columns for LLM data
        llm_columns = [
            'individual_vs_org_score', 'generational_appeal_score', 'professionalization_score', 'relationship_status_score',
            'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'keyword6', 'keyword7', 'keyword8', 'keyword9', 'keyword10',
            'llm_processed', 'source_batch'
        ]
        
        for col in llm_columns:
            csv_df[col] = None
            
        csv_df['llm_processed'] = False
        
        # Merge LLM data by lance_db_id (direct matching)
        merged_count = 0
        for lance_db_id, llm_record in llm_data.items():
            # Find CSV rows where lance_db_id matches
            matching_rows = csv_df[csv_df['lance_db_id'] == lance_db_id]
            
            if not matching_rows.empty:
                # Update the matching row(s)
                row_index = matching_rows.index[0]  # Get the first matching row index
                for key, value in llm_record.items():
                    csv_df.loc[row_index, key] = value
                merged_count += 1
        
        print(f"✅ Merged {merged_count:,} records with LLM analysis")
        print(f"📊 Total records: {len(csv_df):,}")
        print(f"🧠 Records with LLM data: {csv_df['llm_processed'].sum():,}")
        
        return csv_df
    
    def create_lancedb_table(self, df: pd.DataFrame, 
                           db_path: str = "influencers_lancedb", 
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
        numeric_columns = ['individual_vs_org_score', 'generational_appeal_score', 'professionalization_score', 'relationship_status_score']
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
                           jsonl_filename: Optional[str] = None,  # None loads all batch files
                           db_path: str = "influencers_lancedb",
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