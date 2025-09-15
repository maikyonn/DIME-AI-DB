#!/usr/bin/env python3
"""
Combine all batch results into a single CSV file
"""

import os
import sys
import csv
import json
import glob
from typing import List, Dict

def find_batch_directories() -> List[str]:
    """Find all batch_XXX directories"""
    return sorted(glob.glob("batch_*"))

def combine_csv_files(batch_dirs: List[str], output_file: str = "combined_analyzed_profiles.csv"):
    """Combine all CSV files from batch directories"""
    all_rows = []
    fieldnames = None
    total_processed = 0
    
    print(f"🔄 Combining results from {len(batch_dirs)} batch directories...")
    
    for batch_dir in batch_dirs:
        # Find CSV files in this directory
        csv_files = glob.glob(os.path.join(batch_dir, "analyzed_profiles_*.csv"))
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if fieldnames is None:
                        fieldnames = reader.fieldnames
                    
                    rows = list(reader)
                    all_rows.extend(rows)
                    total_processed += len(rows)
                    print(f"   📄 {csv_file}: {len(rows)} profiles")
            except Exception as e:
                print(f"   ❌ Error reading {csv_file}: {e}")
    
    if not all_rows:
        print("❌ No data found to combine")
        return False
    
    # Write combined CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"✅ Combined {total_processed} profiles into: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error writing combined file: {e}")
        return False

def create_summary_report(batch_dirs: List[str], output_file: str = "batch_summary_report.txt"):
    """Create a summary report of all batch processing"""
    try:
        with open(output_file, 'w') as f:
            f.write("Instagram Profile Analysis - Batch Processing Summary\n")
            f.write("=" * 60 + "\n\n")
            
            total_profiles = 0
            total_batches = len(batch_dirs)
            
            for batch_dir in batch_dirs:
                csv_files = glob.glob(os.path.join(batch_dir, "analyzed_profiles_*.csv"))
                batch_profiles = 0
                
                for csv_file in csv_files:
                    try:
                        with open(csv_file, 'r', encoding='utf-8') as csvf:
                            reader = csv.DictReader(csvf)
                            batch_profiles += len(list(reader))
                    except:
                        pass
                
                f.write(f"Batch Directory: {batch_dir}\n")
                f.write(f"Profiles Processed: {batch_profiles}\n")
                f.write(f"CSV Files: {len(csv_files)}\n")
                f.write("-" * 30 + "\n")
                
                total_profiles += batch_profiles
            
            f.write(f"\nTOTAL SUMMARY:\n")
            f.write(f"Total Batch Directories: {total_batches}\n")
            f.write(f"Total Profiles Processed: {total_profiles}\n")
            f.write(f"Average Profiles per Batch: {total_profiles // total_batches if total_batches > 0 else 0}\n")
        
        print(f"📄 Summary report created: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating summary report: {e}")
        return False

def analyze_keywords():
    """Analyze most common keywords across all profiles"""
    keyword_counts = {}
    total_profiles = 0
    
    # Find combined file
    combined_file = "combined_analyzed_profiles.csv"
    if not os.path.exists(combined_file):
        print("❌ Combined file not found. Run combination first.")
        return
    
    try:
        with open(combined_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_profiles += 1
                # Count keywords
                for i in range(1, 6):  # keyword1 through keyword5
                    keyword_col = f'keyword{i}'
                    if keyword_col in row and row[keyword_col].strip():
                        keyword = row[keyword_col].strip().lower()
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Sort keywords by frequency
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 Top 20 Keywords (from {total_profiles} profiles):")
        print("-" * 50)
        for keyword, count in sorted_keywords[:20]:
            percentage = (count / total_profiles) * 100
            print(f"{keyword:30s}: {count:5d} ({percentage:5.1f}%)")
        
        # Save to file
        with open("keyword_analysis.txt", 'w') as f:
            f.write(f"Keyword Analysis - {total_profiles} Profiles\n")
            f.write("=" * 50 + "\n\n")
            for keyword, count in sorted_keywords:
                percentage = (count / total_profiles) * 100
                f.write(f"{keyword:30s}: {count:5d} ({percentage:5.1f}%)\n")
        
        print(f"\n📄 Full keyword analysis saved to: keyword_analysis.txt")
        
    except Exception as e:
        print(f"❌ Error analyzing keywords: {e}")

def main():
    """Main function"""
    print("🔄 Combining batch processing results...")
    
    # Find batch directories
    batch_dirs = find_batch_directories()
    if not batch_dirs:
        print("❌ No batch directories found")
        print("Make sure you've run the batch processor and have batch_XXX directories")
        return 1
    
    print(f"📁 Found {len(batch_dirs)} batch directories: {', '.join(batch_dirs)}")
    
    # Combine CSV files
    if not combine_csv_files(batch_dirs):
        return 1
    
    # Create summary report
    create_summary_report(batch_dirs)
    
    # Analyze keywords
    analyze_keywords()
    
    print("\n🎉 Batch results combination completed!")
    print("📄 Files created:")
    print("   - combined_analyzed_profiles.csv")
    print("   - batch_summary_report.txt")
    print("   - keyword_analysis.txt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())