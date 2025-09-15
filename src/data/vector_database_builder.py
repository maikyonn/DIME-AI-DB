"""
Enhanced Vector Database Builder for Instagram Influencer Data
Implements three-vector architecture: keywords, profile, content
"""
import json
import pandas as pd
import lancedb
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import re
from tqdm import tqdm
from src.data.unified_data_loader import UnifiedDataLoader

# Removed language detection - assuming all users are English


class VectorDatabaseBuilder(UnifiedDataLoader):
    """Enhanced vector database builder with three-vector architecture"""
    
    def __init__(self, data_dir: str = "data", model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__(data_dir)
        self.model_name = model_name
        self.model = None
        
    def load_embedding_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print(f"🤖 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded with {self.model.get_sentence_embedding_dimension()} dimensions")
        return self.model
    
    
    
    
    def build_keyword_text(self, row) -> str:
        """Build text from LLM-extracted keywords"""
        keywords = []
        for i in range(1, 11):  # keyword1 through keyword10
            keyword = row.get(f'keyword{i}', '')
            if keyword and isinstance(keyword, str) and len(keyword.strip()) > 0:
                keywords.append(keyword.strip())
        
        return " ".join(keywords) if keywords else ""
    
    def build_profile_text(self, row) -> str:
        """Build profile vector text from pure natural content only"""
        text_parts = []
        
        # Biography (main content, no names) - primary semantic signal
        biography = row.get('biography', '')
        if biography and isinstance(biography, str):
            text_parts.append(biography.strip())
        
        # Business category - semantic category information
        category = row.get('business_category_name', '')
        if category and isinstance(category, str):
            text_parts.append(f"Category: {category}")
        
        # Note: All numerical data (followers, scores) stored as metadata for precise filtering
        # Profile vector now contains only genuine natural language content
        
        return " ".join(text_parts)
    
    def build_content_text(self, row) -> str:
        """Build content vector text from post captions"""
        posts_data = row.get('posts', '')
        if not posts_data or posts_data == '' or pd.isna(posts_data):
            return ""
        
        try:
            posts = json.loads(posts_data) if isinstance(posts_data, str) else posts_data
            if not isinstance(posts, list):
                return ""
            
            captions = []
            for post in posts[:10]:  # Use top 10 posts
                if isinstance(post, dict) and 'caption' in post:
                    caption = post.get('caption', '')
                    if caption and isinstance(caption, str):
                        # Clean caption (remove excessive hashtags, keep core content)
                        clean_caption = self.clean_caption(caption)
                        if clean_caption:
                            captions.append(clean_caption)
            
            return " ".join(captions)
            
        except (json.JSONDecodeError, TypeError, AttributeError):
            return ""
    
    def clean_caption(self, caption: str) -> str:
        """Clean post caption for better embedding quality"""
        if not caption:
            return ""
        
        # Remove excessive hashtags (keep first few)
        lines = caption.split('\n')
        clean_lines = []
        hashtag_count = 0
        
        for line in lines:
            if line.strip().startswith('#'):
                hashtag_count += 1
                if hashtag_count <= 5:  # Keep first 5 hashtag lines
                    clean_lines.append(line)
            else:
                clean_lines.append(line)
        
        clean_text = '\n'.join(clean_lines)
        
        # Remove excessive emojis and special characters
        clean_text = re.sub(r'[^\w\s#@\.\,\!\?\-\n]', ' ', clean_text)
        
        # Limit length (embeddings work better with reasonable length)
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "..."
        
        return clean_text.strip()
    
    def generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for clean three-vector architecture (natural content only)"""
        print("🚀 Generating embeddings for improved three-vector architecture...")
        print("📊 Architecture: Keyword (LLM) + Profile (natural) + Content (captions)")
        
        # Load model
        model = self.load_embedding_model()
        
        # Build text for each vector type
        print("📝 Building text representations (scores stored as metadata)...")
        keyword_texts = []
        profile_texts = []
        content_texts = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building natural text"):
            keyword_texts.append(self.build_keyword_text(row))
            profile_texts.append(self.build_profile_text(row))
            content_texts.append(self.build_content_text(row))
        
        # Generate embeddings in batches (increased batch size for efficiency)
        batch_size = 128
        
        print("🔤 Generating keyword embeddings (LLM-extracted)...")
        keyword_embeddings = model.encode(
            keyword_texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print("👤 Generating profile embeddings (natural content)...")
        profile_embeddings = model.encode(
            profile_texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print("📱 Generating content embeddings (post captions)...")
        content_embeddings = model.encode(
            content_texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings to dataframe
        df['keyword_vector'] = keyword_embeddings.tolist()
        df['profile_vector'] = profile_embeddings.tolist()
        df['content_vector'] = content_embeddings.tolist()
        
        # Add text for debugging/analysis
        df['keyword_text'] = keyword_texts
        df['profile_text'] = profile_texts
        df['content_text_sample'] = [text[:200] + "..." if len(text) > 200 else text for text in content_texts]
        
        # Mark all as English (preprocessing already filtered)
        df['is_english'] = True
        df['detected_language'] = 'en'
        df['language_confidence'] = 1.0
        
        # Calculate text quality metrics
        non_empty_keywords = sum(1 for text in keyword_texts if text.strip())
        non_empty_profiles = sum(1 for text in profile_texts if text.strip())
        non_empty_content = sum(1 for text in content_texts if text.strip())
        
        print(f"✅ Generated embeddings: {len(df)} records x 3 vectors x {model.get_sentence_embedding_dimension()} dimensions")
        print(f"📊 Text quality: Keywords: {non_empty_keywords}/{len(df)} | Profiles: {non_empty_profiles}/{len(df)} | Content: {non_empty_content}/{len(df)}")
        print(f"🎯 LLM scores preserved as metadata for precise filtering")
        
        return df
    
    def create_vector_database(self, 
                              csv_filename: str = "Snap Data.csv",
                              jsonl_filename: Optional[str] = None,  # None means load all batch files
                              db_path: str = "influencers_vectordb",
                              table_name: str = "influencer_profiles"):
        """Create complete vector database with three-vector architecture"""
        
        print("🏗️ Building Vector Database with Three-Vector Architecture")
        print("=" * 70)
        
        # Load and merge data
        csv_df = self.load_csv_data(csv_filename)
        llm_data = self.parse_batch_output(jsonl_filename)
        merged_df = self.merge_data(csv_df, llm_data)
        
        print(f"📊 Merged dataset: {len(merged_df):,} total records")
        llm_processed = merged_df[merged_df['llm_processed'] == True]
        print(f"🧠 LLM processed records: {len(llm_processed):,}")
        
        # Generate embeddings
        vector_df = self.generate_embeddings(merged_df)
        
        # Create LanceDB table
        print(f"💾 Creating LanceDB table: {db_path}/{table_name}")
        db = lancedb.connect(db_path)
        
        # Convert to PyArrow table for LanceDB
        table = db.create_table(table_name, vector_df, mode="overwrite")
        
        print(f"✅ Vector database created successfully!")
        print(f"   • Database: {db_path}")
        print(f"   • Table: {table_name}")
        print(f"   • Records: {table.count_rows():,}")
        print(f"   • Schema: {len(table.schema)} columns")
        
        # Create indices for vector search
        self.create_vector_indices(table)
        
        return table
    
    def create_vector_indices(self, table):
        """Create optimized indices for vector search"""
        print("🔧 Creating vector indices for optimized search...")
        
        try:
            # Create IVF indices for each vector column
            index_configs = [
                ("keyword_vector", "Keywords"),
                ("profile_vector", "Profile"), 
                ("content_vector", "Content")
            ]
            
            for vector_column, description in index_configs:
                print(f"   Creating index for {description} vector...")
                table.create_index(
                    vector_column_name=vector_column,
                    metric="cosine",
                    num_partitions=min(256, max(1, table.count_rows() // 1000)),  # Adaptive partitions
                    num_sub_vectors=32
                )
                print(f"   ✅ {description} vector index created")
            
            # Create scalar indices for common filters
            scalar_indices = ['followers', 'engagement_rate', 'is_verified', 'is_business_account']
            for column in scalar_indices:
                if column in table.schema.names:
                    try:
                        table.create_scalar_index(column)
                        print(f"   ✅ Scalar index created for {column}")
                    except:
                        print(f"   ⚠️ Could not create scalar index for {column}")
            
            print("🚀 All indices created successfully!")
            
        except Exception as e:
            print(f"⚠️ Index creation warning: {e}")
            print("   Database will still work, just slower for large queries")


def main():
    """Example usage"""
    builder = VectorDatabaseBuilder()
    
    # Create vector database
    table = builder.create_vector_database(
        csv_filename="Snap Data.csv",
        jsonl_filename=None,  # Load all batch output files
        db_path="influencers_vectordb",
        table_name="influencer_profiles"
    )
    
    print("\n🎯 Database ready for semantic search!")
    print("Next: implement search interface with weighted multi-vector queries")


if __name__ == "__main__":
    main()