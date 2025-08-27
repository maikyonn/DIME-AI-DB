#!/usr/bin/env python3
"""
Helper script to check batch status and download results
"""

import os
import sys
import dotenv
from batch_process_snap_data import BatchProcessor

# Load environment variables
dotenv.load_dotenv()

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_batch_status.py <batch_id>")
        return 1
    
    batch_id = sys.argv[1]
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return 1
    
    processor = BatchProcessor()
    
    # Check status
    print(f"🔍 Checking status for batch: {batch_id}")
    status_info = processor.check_batch_status(batch_id)
    
    if not status_info:
        return 1
    
    print(f"\n📊 Batch Status: {status_info['status']}")
    if 'request_counts' in status_info and status_info['request_counts']:
        counts = status_info['request_counts']
        print(f"📈 Progress: {counts.get('completed', 0)}/{counts.get('total', 0)} completed, {counts.get('failed', 0)} failed")
    
    # If completed, offer to download results
    if status_info['status'] == 'completed' and status_info.get('output_file_id'):
        download = input("\n💾 Batch is complete! Download and process results? (y/n): ").lower().strip()
        if download == 'y':
            if processor.download_results(status_info['output_file_id']):
                if processor.process_results():
                    print("🎉 Results processed successfully! Check 'analyzed_profiles.csv'")
    elif status_info['status'] in ['failed', 'expired', 'cancelled']:
        if status_info.get('error_file_id'):
            download_errors = input("\n❌ Batch failed. Download error file? (y/n): ").lower().strip()
            if download_errors == 'y':
                processor.download_results(status_info['error_file_id'], "batch_errors.jsonl")
    else:
        print(f"⏳ Batch is still {status_info['status']}. Check again later.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())