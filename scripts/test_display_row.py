#!/usr/bin/env python3
"""
Test script for the database row display functionality.
This script demonstrates how to use the DatabaseRowViewer class.
"""
import pandas as pd
from display_single_row import DatabaseRowViewer


def test_with_csv_fallback():
    """Test the display functionality with CSV data if LanceDB is not available"""
    print("🧪 Testing Database Row Display Functionality")
    print("=" * 60)
    
    # Initialize viewer
    viewer = DatabaseRowViewer()
    
    # Try to connect to existing database
    if viewer.connect_database():
        print("\n✅ Using existing LanceDB database")
        
        # Show database info
        info = viewer.get_table_info()
        print("\n📊 Database Information:")
        for key, value in info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Display first row
        print("\n" + "="*80)
        print("DISPLAYING FIRST ROW FROM LANCEDB:")
        success = viewer.display_row_by_index(0)
        
        if success:
            print("\n✅ LanceDB row display test successful!")
        else:
            print("\n❌ LanceDB row display test failed!")
        
    else:
        print("\n⚠️ LanceDB not available, testing with CSV data...")
        
        # Fallback to CSV testing
        try:
            df = pd.read_csv('data/brightdata-csv-dataset/Snap Data_1000rows.csv')
            print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Get first row
            first_row = df.iloc[0]
            
            # Create a mock viewer for testing
            formatted_output = viewer.format_single_row(first_row, 0)
            print("\n" + "="*80)
            print("DISPLAYING FIRST ROW FROM CSV:")
            print(formatted_output)
            
            print("\n✅ CSV row display test successful!")
            
        except FileNotFoundError:
            print("❌ CSV file not found. Please ensure data file exists.")
        except Exception as e:
            print(f"❌ Error testing with CSV: {e}")


def demonstrate_features():
    """Demonstrate different features of the row viewer"""
    viewer = DatabaseRowViewer()
    
    if not viewer.connect_database():
        print("❌ Cannot demonstrate features without database connection")
        return
    
    print("\n🎯 FEATURE DEMONSTRATIONS")
    print("=" * 60)
    
    # 1. Show database info
    print("\n1. Database Information:")
    info = viewer.get_table_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # 2. Display by index
    print("\n2. Display by Index (Row 0):")
    print("-" * 30)
    viewer.display_row_by_index(0)
    
    # 3. Try search functionality
    print("\n3. Search Functionality:")
    print("-" * 30)
    search_terms = ["home", "fashion", "food", "travel"]
    
    for term in search_terms:
        print(f"\nSearching for '{term}':")
        success = viewer.search_and_display(term, limit=1)
        if success:
            break
    
    print("\n✅ Feature demonstration complete!")


if __name__ == "__main__":
    # Run basic test
    test_with_csv_fallback()
    
    # Run feature demonstration if database is available
    print("\n" + "="*80)
    demonstrate_features()