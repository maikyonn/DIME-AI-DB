"""
Multi-Vector Search Interface for Instagram Influencer Database
Implements weighted search across keyword, profile, and content vectors
"""
import lancedb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass


@dataclass
class SearchWeights:
    """Search weight configuration for different query types"""
    keyword: float = 0.45
    profile: float = 0.35
    content: float = 0.20
    
    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = self.keyword + self.profile + self.content
        if total > 0:
            self.keyword /= total
            self.profile /= total
            self.content /= total


class VectorSearchEngine:
    """Multi-vector search engine with weighted combination"""
    
    def __init__(self, db_path: str = "influencers_vectordb", 
                 table_name: str = "influencer_profiles",
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        
        self.db_path = db_path
        self.table_name = table_name
        self.model_name = model_name
        self.model = None
        self.table = None
        
        # Default search weights (balanced distribution)
        self.default_weights = SearchWeights(keyword=0.33, profile=0.33, content=0.34)
    
    def connect(self):
        """Connect to LanceDB and load model"""
        if self.table is None:
            print(f"🔌 Connecting to vector database: {self.db_path}")
            db = lancedb.connect(self.db_path)
            self.table = db.open_table(self.table_name)
            print(f"✅ Connected to table with {self.table.count_rows():,} records")
        
        if self.model is None:
            print(f"🤖 Loading search model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Search model loaded")
    
    
    def search(self, 
               query: str,
               limit: int = 20,
               weights: Optional[SearchWeights] = None,
               filters: Optional[Dict[str, Any]] = None,
               return_vectors: bool = False) -> pd.DataFrame:
        """
        Perform weighted multi-vector search
        
        Args:
            query: Natural language search query
            limit: Number of results to return
            weights: Custom search weights, or None for auto-detection
            filters: Optional metadata filters
            return_vectors: Whether to include vector data in results
            
        Returns:
            DataFrame with search results and combined scores
        """
        self.connect()
        
        # Use default weights if not provided
        if weights is None:
            weights = self.default_weights
        
        weights.normalize()
        print(f"⚖️ Search weights - Keyword: {weights.keyword:.2f}, Profile: {weights.profile:.2f}, Content: {weights.content:.2f}")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0].tolist()
        
        # Store filters for later application (after search)
        apply_filters_later = filters is not None
        
        # Search each vector type
        search_limit = min(limit * 5, 1000)  # Get more results for better combination
        
        try:
            # Keyword vector search
            keyword_results = self.table.search(
                query_embedding,
                vector_column_name="keyword_vector"
            ).limit(search_limit).to_pandas()
            
            # Profile vector search  
            profile_results = self.table.search(
                query_embedding,
                vector_column_name="profile_vector"
            ).limit(search_limit).to_pandas()
            
            # Content vector search
            content_results = self.table.search(
                query_embedding,
                vector_column_name="content_vector"
            ).limit(search_limit).to_pandas()
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            print("💡 Tip: Make sure vector indices are created")
            return pd.DataFrame()
        
        # Combine results with weighted scores
        combined_results = self._combine_search_results(
            keyword_results, profile_results, content_results,
            weights, query
        )
        
        # Apply filters after combining results
        if filters and not combined_results.empty:
            combined_results = self._apply_dataframe_filters(combined_results, filters)
            print(f"📊 Filtered to {len(combined_results)} records after search")
        
        # Sort by combined score and limit
        final_results = combined_results.nlargest(limit, 'combined_score') if not combined_results.empty else combined_results
        
        # Clean up results for display
        if not return_vectors:
            vector_columns = ['keyword_vector', 'profile_vector', 'content_vector']
            final_results = final_results.drop(columns=[col for col in vector_columns if col in final_results.columns])
        
        print(f"🎯 Returning {len(final_results)} results")
        return final_results.reset_index(drop=True)
    
    def _apply_filters(self, table, filters: Dict[str, Any]):
        """Apply metadata filters to search"""
        filter_conditions = []
        
        for key, value in filters.items():
            if isinstance(value, tuple) and len(value) == 2:
                # Range filter (min, max)
                min_val, max_val = value
                filter_conditions.append(f"{key} BETWEEN {min_val} AND {max_val}")
            elif isinstance(value, (list, tuple)):
                # IN filter
                values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                filter_conditions.append(f"{key} IN ({values_str})")
            elif isinstance(value, str):
                # String match
                filter_conditions.append(f"{key} = '{value}'")
            else:
                # Exact match
                filter_conditions.append(f"{key} = {value}")
        
        if filter_conditions:
            where_clause = " AND ".join(filter_conditions)
            try:
                return table.where(where_clause)
            except AttributeError:
                # Fallback for newer LanceDB versions
                import lancedb
                return table.search().where(where_clause)
        
        return table
    
    def _apply_dataframe_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to a pandas DataFrame"""
        filtered_df = df.copy()
        
        for key, value in filters.items():
            if key not in filtered_df.columns:
                continue
                
            if isinstance(value, tuple) and len(value) == 2:
                # Range filter (min, max)
                min_val, max_val = value
                filtered_df = filtered_df[
                    (filtered_df[key] >= min_val) & (filtered_df[key] <= max_val)
                ]
            elif isinstance(value, (list, tuple)):
                # IN filter
                filtered_df = filtered_df[filtered_df[key].isin(value)]
            else:
                # Exact match
                filtered_df = filtered_df[filtered_df[key] == value]
        
        return filtered_df
    
    def _combine_search_results(self, 
                               keyword_results: pd.DataFrame,
                               profile_results: pd.DataFrame, 
                               content_results: pd.DataFrame,
                               weights: SearchWeights,
                               query: str) -> pd.DataFrame:
        """Combine search results from all three vectors with weights"""
        
        # Create result dictionaries for faster lookup
        def create_result_dict(df, score_name):
            return {
                row['id']: {'score': 1 / (1 + row['_distance']), 'data': row} 
                for _, row in df.iterrows()
            }
        
        keyword_dict = create_result_dict(keyword_results, 'keyword_score')
        profile_dict = create_result_dict(profile_results, 'profile_score') 
        content_dict = create_result_dict(content_results, 'content_score')
        
        # Get all unique IDs
        all_ids = set(keyword_dict.keys()) | set(profile_dict.keys()) | set(content_dict.keys())
        
        combined_results = []
        for record_id in all_ids:
            # Get scores for each vector (0 if not found)
            keyword_score = keyword_dict.get(record_id, {}).get('score', 0)
            profile_score = profile_dict.get(record_id, {}).get('score', 0)  
            content_score = content_dict.get(record_id, {}).get('score', 0)
            
            # Calculate weighted combined score
            combined_score = (
                weights.keyword * keyword_score +
                weights.profile * profile_score + 
                weights.content * content_score
            )
            
            # Get the record data (prefer keyword > profile > content)
            record_data = None
            for result_dict in [keyword_dict, profile_dict, content_dict]:
                if record_id in result_dict:
                    record_data = result_dict[record_id]['data'].copy()
                    break
            
            if record_data is not None:
                record_data['keyword_score'] = keyword_score
                record_data['profile_score'] = profile_score
                record_data['content_score'] = content_score
                record_data['combined_score'] = combined_score
                record_data['query'] = query
                
                combined_results.append(record_data)
        
        return pd.DataFrame(combined_results)
    
    def search_similar_profiles(self, 
                              account_name: str,
                              limit: int = 10,
                              weights: Optional[SearchWeights] = None) -> pd.DataFrame:
        """Find profiles similar to a given account using text-based search (legacy method)"""
        self.connect()
        
        # Get the target profile
        try:
            target_profile = self.table.where(f"account = '{account_name}'").to_pandas()
        except AttributeError:
            # Fallback for newer LanceDB versions - search through all data
            full_table = self.table.to_pandas()
            target_profile = full_table[full_table['account'] == account_name]
        
        if target_profile.empty:
            print(f"❌ Account '{account_name}' not found")
            return pd.DataFrame()
        
        target_row = target_profile.iloc[0]
        
        # Use the target profile's vectors for similarity search
        if weights is None:
            weights = SearchWeights(keyword=0.5, profile=0.4, content=0.1)  # Focus on keywords and profile
        
        # Search using target profile's characteristics
        query_text = f"{target_row.get('keyword_text', '')} {target_row.get('business_category_name', '')}"
        
        print(f"🔍 Finding profiles similar to @{account_name}")
        print(f"📝 Using query: {query_text}")
        
        results = self.search(
            query=query_text,
            limit=limit + 1,  # +1 to account for the target profile itself
            weights=weights
        )
        
        # Remove the target profile from results
        results = results[results['account'] != account_name]
        
        return results.head(limit)
    
    def search_similar_by_vectors(self,
                                 account_name: str,
                                 limit: int = 10,
                                 weights: Optional[SearchWeights] = None,
                                 similarity_threshold: float = 0.1,
                                 include_similarity_scores: bool = True,
                                 filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Find profiles similar to a given account using direct vector similarity comparison.
        
        This method provides true vector similarity by comparing embeddings directly,
        rather than using text-based search reconstruction.
        
        Args:
            account_name: Target account to find similar profiles for
            limit: Number of similar profiles to return
            weights: Similarity weights for different vector types
            similarity_threshold: Minimum similarity score to include in results
            include_similarity_scores: Whether to include detailed similarity metrics
            filters: Optional metadata filters (followers, engagement, etc.)
            
        Returns:
            DataFrame with similar profiles ranked by composite similarity score
        """
        self.connect()
        
        # Get the target profile with vectors
        try:
            target_profile = self.table.where(f"account = '{account_name}'").to_pandas()
        except AttributeError:
            # Fallback for newer LanceDB versions
            full_table = self.table.to_pandas()
            target_profile = full_table[full_table['account'] == account_name]
        
        if target_profile.empty:
            print(f"❌ Account '{account_name}' not found")
            return pd.DataFrame()
        
        target_row = target_profile.iloc[0]
        
        # Set default weights for similarity search
        if weights is None:
            weights = SearchWeights(keyword=0.4, profile=0.4, content=0.2)
        weights.normalize()
        
        print(f"🎯 Finding vector-similar profiles to @{account_name}")
        print(f"⚖️ Similarity weights - Keyword: {weights.keyword:.2f}, Profile: {weights.profile:.2f}, Content: {weights.content:.2f}")
        
        # Extract target vectors
        try:
            target_keyword_vector = np.array(target_row['keyword_vector'])
            target_profile_vector = np.array(target_row['profile_vector']) 
            target_content_vector = np.array(target_row['content_vector'])
        except (KeyError, TypeError) as e:
            print(f"❌ Error extracting vectors: {e}")
            return pd.DataFrame()
        
        # Performance optimization: Load profiles in batches for large databases
        batch_size = 10000  # Process 10K profiles at a time
        all_similarities = []
        
        try:
            # Get total count first
            total_rows = self.table.count_rows()
            print(f"🔄 Processing {total_rows:,} profiles in batches of {batch_size:,}...")
            
            # Process in batches for memory efficiency
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                
                # Load batch
                try:
                    batch_profiles = self.table.to_pandas().iloc[batch_start:batch_end]
                except Exception as e:
                    print(f"⚠️ Error loading batch {batch_start}-{batch_end}: {e}")
                    continue
                
                # Remove target account from batch
                batch_profiles = batch_profiles[batch_profiles['account'] != account_name].copy()
                
                if batch_profiles.empty:
                    continue
                
                # Compute similarities for this batch
                batch_similarities = self._compute_vector_similarities_optimized(
                    batch_profiles,
                    target_keyword_vector,
                    target_profile_vector, 
                    target_content_vector,
                    weights
                )
                
                if not batch_similarities.empty:
                    all_similarities.append(batch_similarities)
                
                print(f"✓ Processed batch {batch_start//batch_size + 1}/{(total_rows-1)//batch_size + 1}")
            
            # Combine all batch results
            if all_similarities:
                similarities = pd.concat(all_similarities, ignore_index=True)
                print(f"✅ Combined {len(similarities):,} similarity scores")
            else:
                print("❌ No similarities computed from any batch")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            print("🔄 Falling back to single-batch processing...")
            
            # Fallback to loading all data at once
            try:
                all_profiles = self.table.to_pandas()
            except Exception as e:
                print(f"❌ Error loading profiles: {e}")
                return pd.DataFrame()
            
            if all_profiles.empty:
                print("❌ No profiles found in database")
                return pd.DataFrame()
            
            # Remove target account from comparison set
            comparison_profiles = all_profiles[all_profiles['account'] != account_name].copy()
            
            if comparison_profiles.empty:
                print("❌ No other profiles found for comparison")
                return pd.DataFrame()
            
            print(f"🔄 Computing similarities against {len(comparison_profiles):,} profiles...")
            
            # Compute vector similarities using optimized operations
            similarities = self._compute_vector_similarities_optimized(
                comparison_profiles,
                target_keyword_vector,
                target_profile_vector, 
                target_content_vector,
                weights
            )
        
        if similarities.empty:
            print("❌ No similarities computed")
            return pd.DataFrame()
        
        # Apply similarity threshold
        similarities = similarities[similarities['vector_similarity_score'] >= similarity_threshold]
        
        # Apply optional filters
        if filters and not similarities.empty:
            similarities = self._apply_dataframe_filters(similarities, filters)
            print(f"📊 Filtered to {len(similarities)} profiles after applying filters")
        
        # Sort by similarity score and limit results
        final_results = similarities.nlargest(limit, 'vector_similarity_score')
        
        # Add similarity explanations if requested
        if include_similarity_scores and not final_results.empty:
            final_results = self._add_similarity_explanations(final_results, target_row)
        
        # Clean up vector columns for output unless specifically requested
        vector_columns = ['keyword_vector', 'profile_vector', 'content_vector']
        final_results = final_results.drop(columns=[col for col in vector_columns if col in final_results.columns])
        
        print(f"✅ Found {len(final_results)} similar profiles (similarity ≥ {similarity_threshold:.3f})")
        
        return final_results.reset_index(drop=True)
    
    def _compute_vector_similarities_optimized(self, 
                                             profiles_df: pd.DataFrame,
                                             target_keyword_vector: np.ndarray,
                                             target_profile_vector: np.ndarray,
                                             target_content_vector: np.ndarray,
                                             weights: SearchWeights) -> pd.DataFrame:
        """
        Optimized vector similarity computation using vectorized operations.
        
        This method is significantly faster than the row-by-row approach for large datasets.
        """
        
        # Extract all vectors at once for vectorized computation
        try:
            # Convert vector lists to numpy arrays for batch processing
            keyword_vectors = []
            profile_vectors = []
            content_vectors = []
            valid_indices = []
            
            for idx, row in profiles_df.iterrows():
                try:
                    kw_vec = np.array(row['keyword_vector']) if 'keyword_vector' in row and row['keyword_vector'] is not None else None
                    pr_vec = np.array(row['profile_vector']) if 'profile_vector' in row and row['profile_vector'] is not None else None
                    ct_vec = np.array(row['content_vector']) if 'content_vector' in row and row['content_vector'] is not None else None
                    
                    if kw_vec is not None and pr_vec is not None and ct_vec is not None:
                        keyword_vectors.append(kw_vec)
                        profile_vectors.append(pr_vec)
                        content_vectors.append(ct_vec)
                        valid_indices.append(idx)
                except:
                    continue
            
            if not valid_indices:
                return pd.DataFrame()
            
            # Convert to numpy matrices for vectorized operations
            keyword_matrix = np.vstack(keyword_vectors)  # Shape: (n_profiles, vector_dim)
            profile_matrix = np.vstack(profile_vectors)
            content_matrix = np.vstack(content_vectors)
            
            # Compute cosine similarities using vectorized operations
            keyword_similarities = self._batch_cosine_similarity(target_keyword_vector, keyword_matrix)
            profile_similarities = self._batch_cosine_similarity(target_profile_vector, profile_matrix)
            content_similarities = self._batch_cosine_similarity(target_content_vector, content_matrix)
            
            # Create results DataFrame with valid profiles only
            results_df = profiles_df.loc[valid_indices].copy()
            
            # Add similarity scores
            results_df['keyword_similarity'] = keyword_similarities
            results_df['profile_similarity'] = profile_similarities
            results_df['content_similarity'] = content_similarities
            
            # Compute weighted composite similarity score
            results_df['vector_similarity_score'] = (
                weights.keyword * results_df['keyword_similarity'] +
                weights.profile * results_df['profile_similarity'] + 
                weights.content * results_df['content_similarity']
            )
            
            return results_df
            
        except Exception as e:
            print(f"⚠️ Error in optimized similarity computation: {e}")
            print("🔄 Falling back to row-by-row computation...")
            
            # Fallback to the original method
            return self._compute_vector_similarities(
                profiles_df, target_keyword_vector, target_profile_vector, target_content_vector, weights
            )
    
    def _batch_cosine_similarity(self, target_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a target vector and a matrix of vectors.
        
        Args:
            target_vector: 1D numpy array
            matrix: 2D numpy array where each row is a vector to compare
            
        Returns:
            1D numpy array of similarity scores
        """
        try:
            # Normalize target vector
            target_norm = np.linalg.norm(target_vector)
            if target_norm == 0:
                return np.zeros(matrix.shape[0])
            
            target_normalized = target_vector / target_norm
            
            # Normalize matrix rows
            matrix_norms = np.linalg.norm(matrix, axis=1)
            
            # Handle zero norms
            zero_mask = matrix_norms == 0
            matrix_norms[zero_mask] = 1  # Avoid division by zero
            
            matrix_normalized = matrix / matrix_norms[:, np.newaxis]
            
            # Compute dot products (cosine similarities)
            similarities = np.dot(matrix_normalized, target_normalized)
            
            # Set similarities to 0 where original norms were 0
            similarities[zero_mask] = 0
            
            # Ensure results are in [0, 1] range
            similarities = np.maximum(0, similarities)
            
            return similarities
            
        except Exception as e:
            print(f"⚠️ Error in batch cosine similarity: {e}")
            # Fallback to individual computations
            similarities = []
            for i, row_vector in enumerate(matrix):
                sim = self._cosine_similarity(target_vector, row_vector)
                similarities.append(sim)
            return np.array(similarities)
    
    def _compute_vector_similarities(self, 
                                   profiles_df: pd.DataFrame,
                                   target_keyword_vector: np.ndarray,
                                   target_profile_vector: np.ndarray,
                                   target_content_vector: np.ndarray,
                                   weights: SearchWeights) -> pd.DataFrame:
        """Compute cosine similarities between target vectors and all profile vectors"""
        
        # Initialize similarity scores
        keyword_similarities = []
        profile_similarities = []
        content_similarities = []
        valid_indices = []
        
        for idx, row in profiles_df.iterrows():
            try:
                # Extract profile vectors
                profile_keyword_vec = np.array(row['keyword_vector']) if 'keyword_vector' in row and row['keyword_vector'] is not None else None
                profile_profile_vec = np.array(row['profile_vector']) if 'profile_vector' in row and row['profile_vector'] is not None else None
                profile_content_vec = np.array(row['content_vector']) if 'content_vector' in row and row['content_vector'] is not None else None
                
                # Compute cosine similarities
                keyword_sim = self._cosine_similarity(target_keyword_vector, profile_keyword_vec) if profile_keyword_vec is not None else 0.0
                profile_sim = self._cosine_similarity(target_profile_vector, profile_profile_vec) if profile_profile_vec is not None else 0.0
                content_sim = self._cosine_similarity(target_content_vector, profile_content_vec) if profile_content_vec is not None else 0.0
                
                keyword_similarities.append(keyword_sim)
                profile_similarities.append(profile_sim)
                content_similarities.append(content_sim)
                valid_indices.append(idx)
                
            except Exception as e:
                # Skip profiles with invalid vectors
                continue
        
        if not valid_indices:
            print("❌ No valid vectors found for similarity computation")
            return pd.DataFrame()
        
        # Create results DataFrame with valid profiles only
        results_df = profiles_df.loc[valid_indices].copy()
        
        # Add similarity scores
        results_df['keyword_similarity'] = keyword_similarities
        results_df['profile_similarity'] = profile_similarities
        results_df['content_similarity'] = content_similarities
        
        # Compute weighted composite similarity score
        results_df['vector_similarity_score'] = (
            weights.keyword * results_df['keyword_similarity'] +
            weights.profile * results_df['profile_similarity'] + 
            weights.content * results_df['content_similarity']
        )
        
        return results_df
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            if vec1 is None or vec2 is None:
                return 0.0
            
            # Ensure vectors are numpy arrays and have same shape
            vec1 = np.array(vec1).flatten()
            vec2 = np.array(vec2).flatten()
            
            if len(vec1) != len(vec2) or len(vec1) == 0:
                return 0.0
            
            # Compute dot product and norms
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Avoid division by zero
            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0
            
            # Compute cosine similarity
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range (cosine similarity can be negative)
            return max(0.0, float(similarity))
            
        except Exception as e:
            print(f"⚠️ Error computing cosine similarity: {e}")
            return 0.0
    
    def _add_similarity_explanations(self, results_df: pd.DataFrame, target_row) -> pd.DataFrame:
        """Add human-readable explanations for why profiles are similar"""
        
        explanations = []
        
        for _, row in results_df.iterrows():
            explanation_parts = []
            
            # Analyze keyword similarity
            if row.get('keyword_similarity', 0) > 0.3:
                explanation_parts.append(f"shared content themes (keyword similarity: {row['keyword_similarity']:.2f})")
            
            # Analyze profile similarity  
            if row.get('profile_similarity', 0) > 0.3:
                explanation_parts.append(f"similar profile characteristics (profile similarity: {row['profile_similarity']:.2f})")
            
            # Analyze content similarity
            if row.get('content_similarity', 0) > 0.3:
                explanation_parts.append(f"similar posting style (content similarity: {row['content_similarity']:.2f})")
            
            # Check LLM score similarities
            llm_similarities = []
            
            for score_field in ['individual_vs_org_score', 'generational_appeal_score', 'professionalization_score', 'relationship_status_score']:
                target_score = target_row.get(score_field, 0)
                candidate_score = row.get(score_field, 0)
                
                if abs(target_score - candidate_score) <= 1 and target_score > 0:  # Similar scores
                    field_name = score_field.replace('_score', '').replace('_', ' ')
                    llm_similarities.append(field_name)
            
            if llm_similarities:
                explanation_parts.append(f"similar {', '.join(llm_similarities[:2])}")
            
            # Create final explanation
            if explanation_parts:
                explanation = f"Similar due to: {'; '.join(explanation_parts[:3])}"
            else:
                explanation = f"General similarity (score: {row.get('vector_similarity_score', 0):.3f})"
            
            explanations.append(explanation)
        
        results_df = results_df.copy()
        results_df['similarity_explanation'] = explanations
        
        return results_df
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        self.connect()
        
        df = self.table.to_pandas()
        
        stats = {
            'total_records': len(df),
            'llm_processed': len(df[df['llm_processed'] == True]) if 'llm_processed' in df.columns else 0,
            'vector_dimensions': len(df.iloc[0]['keyword_vector']) if len(df) > 0 and 'keyword_vector' in df.columns else 0,
            'avg_followers': df['followers'].mean() if 'followers' in df.columns else 0,
            'verified_accounts': len(df[df['is_verified'] == True]) if 'is_verified' in df.columns else 0,
            'business_accounts': len(df[df['is_business_account'] == True]) if 'is_business_account' in df.columns else 0
        }
        
        return stats


# Example usage and testing functions
def test_search_examples():
    """Test various search queries"""
    engine = VectorSearchEngine()
    
    test_queries = [
        ("home decor influencers", None),
        ("couple influencers new york", None),  
        ("professional fashion brands", SearchWeights(keyword=0.3, profile=0.5, content=0.2)),
        ("micro influencers with diy content", SearchWeights(keyword=0.6, profile=0.3, content=0.1)),
        ("travel bloggers", None)
    ]
    
    for query, weights in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("=" * 50)
        
        results = engine.search(query, limit=5, weights=weights)
        
        if not results.empty:
            for idx, row in results.head(3).iterrows():
                print(f"🎯 Result {idx+1}:")
                print(f"   Account: @{row.get('account', 'N/A')}")
                print(f"   Name: {row.get('profile_name', 'N/A')}")
                print(f"   Category: {row.get('business_category_name', 'N/A')}")
                print(f"   Followers: {row.get('followers', 'N/A'):,}")
                print(f"   Score: {row.get('combined_score', 0):.3f}")
                if 'keyword_text' in row:
                    print(f"   Keywords: {row['keyword_text']}")
                print()
        else:
            print("❌ No results found")


def main():
    """Example usage"""
    engine = VectorSearchEngine()
    
    # Show database stats
    stats = engine.get_database_stats()
    print("📊 Database Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Run test searches
    test_search_examples()


if __name__ == "__main__":
    main()