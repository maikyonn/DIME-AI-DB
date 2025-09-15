#!/usr/bin/env python3
"""
Test script for the new vector-based similarity search functionality.

This script demonstrates the improved vector similarity search and compares it
with the legacy text-based approach.
"""

import sys
import os
import time

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)

from search.vector_search import VectorSearchEngine, SearchWeights


def test_vector_similarity_search():
    """Test the new vector similarity search functionality"""
    
    print("🚀 Testing Vector-Based Similarity Search")
    print("=" * 50)
    
    # Initialize the search engine
    engine = VectorSearchEngine(
        db_path="influencers_vectordb",
        table_name="influencer_profiles"
    )
    
    # Test account (you can change this to any account in your database)
    test_account = "dailybutie"  # Using the account from the schema example
    
    print(f"🎯 Finding similar creators to @{test_account}")
    print()
    
    # Test 1: New Vector-Based Similarity Search
    print("📊 Test 1: Vector-Based Similarity Search")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        vector_results = engine.search_similar_by_vectors(
            account_name=test_account,
            limit=5,
            similarity_threshold=0.1,
            include_similarity_scores=True
        )
        
        vector_time = time.time() - start_time
        
        if not vector_results.empty:
            print(f"✅ Found {len(vector_results)} similar profiles in {vector_time:.2f}s")
            print()
            
            for idx, row in vector_results.head(3).iterrows():
                print(f"🎯 Result {idx+1}:")
                print(f"   Account: @{row.get('account', 'N/A')}")
                print(f"   Name: {row.get('profile_name', 'N/A')}")
                print(f"   Followers: {row.get('followers', 'N/A'):,}")
                print(f"   Vector Similarity: {row.get('vector_similarity_score', 0):.3f}")
                
                # Show individual similarity scores
                if 'keyword_similarity' in row:
                    print(f"   - Keyword: {row['keyword_similarity']:.3f}")
                    print(f"   - Profile: {row['profile_similarity']:.3f}")  
                    print(f"   - Content: {row['content_similarity']:.3f}")
                
                # Show explanation if available
                if 'similarity_explanation' in row and row['similarity_explanation']:
                    print(f"   Explanation: {row['similarity_explanation']}")
                
                print()
        else:
            print(f"❌ No similar profiles found in {vector_time:.2f}s")
    
    except Exception as e:
        print(f"❌ Vector similarity search failed: {e}")
        return
    
    # Test 2: Legacy Text-Based Search (for comparison)
    print("📊 Test 2: Legacy Text-Based Search")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        legacy_results = engine.search_similar_profiles(
            account_name=test_account,
            limit=5
        )
        
        legacy_time = time.time() - start_time
        
        if not legacy_results.empty:
            print(f"✅ Found {len(legacy_results)} similar profiles in {legacy_time:.2f}s")
            print()
            
            for idx, row in legacy_results.head(3).iterrows():
                print(f"🎯 Result {idx+1}:")
                print(f"   Account: @{row.get('account', 'N/A')}")
                print(f"   Name: {row.get('profile_name', 'N/A')}")
                print(f"   Followers: {row.get('followers', 'N/A'):,}")
                print(f"   Combined Score: {row.get('combined_score', 0):.3f}")
                print()
        else:
            print(f"❌ No similar profiles found in {legacy_time:.2f}s")
    
    except Exception as e:
        print(f"❌ Legacy similarity search failed: {e}")
    
    # Performance comparison
    print("⚡ Performance Comparison")
    print("-" * 40)
    try:
        if 'vector_time' in locals() and 'legacy_time' in locals():
            if vector_time < legacy_time:
                improvement = ((legacy_time - vector_time) / legacy_time) * 100
                print(f"✅ Vector search is {improvement:.1f}% faster")
            else:
                slowdown = ((vector_time - legacy_time) / legacy_time) * 100
                print(f"⚠️ Vector search is {slowdown:.1f}% slower (expected for small datasets)")
            
            print(f"📊 Vector search: {vector_time:.2f}s")
            print(f"📊 Legacy search: {legacy_time:.2f}s")
    except:
        print("📊 Performance comparison not available")
    
    print()
    print("🎉 Vector similarity search test completed!")


def test_different_similarity_weights():
    """Test different similarity weight configurations"""
    
    print("\n🎛️ Testing Different Similarity Weight Configurations")
    print("=" * 55)
    
    engine = VectorSearchEngine()
    test_account = "dailybutie"
    
    # Different weight configurations
    weight_configs = [
        ("Keyword-focused", SearchWeights(keyword=0.7, profile=0.2, content=0.1)),
        ("Profile-focused", SearchWeights(keyword=0.2, profile=0.6, content=0.2)), 
        ("Content-focused", SearchWeights(keyword=0.2, profile=0.2, content=0.6)),
        ("Balanced", SearchWeights(keyword=0.33, profile=0.33, content=0.34))
    ]
    
    for config_name, weights in weight_configs:
        print(f"\n📊 {config_name} Configuration")
        print(f"   Weights - Keyword: {weights.keyword:.2f}, Profile: {weights.profile:.2f}, Content: {weights.content:.2f}")
        
        try:
            results = engine.search_similar_by_vectors(
                account_name=test_account,
                limit=3,
                weights=weights,
                similarity_threshold=0.05
            )
            
            if not results.empty:
                print(f"   ✅ Found {len(results)} results")
                
                for idx, row in results.iterrows():
                    sim_score = row.get('vector_similarity_score', 0)
                    account = row.get('account', 'N/A')
                    print(f"      {idx+1}. @{account} (similarity: {sim_score:.3f})")
            else:
                print(f"   ❌ No results found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    try:
        test_vector_similarity_search()
        test_different_similarity_weights()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()