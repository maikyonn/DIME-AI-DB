#!/usr/bin/env python3
"""
Helper script to check status of all batch jobs and download results when ready
"""

import os
import sys
import json
import dotenv
from batch_process_snap_data import BatchProcessor

# Load environment variables
dotenv.load_dotenv()

def main():
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return 1
    
    # Load batch job info
    batch_info_file = "batch_jobs_info.json"
    if not os.path.exists(batch_info_file):
        print(f"❌ Batch info file not found: {batch_info_file}")
        print("Make sure you've run the batch processor first")
        return 1
    
    with open(batch_info_file, 'r') as f:
        batch_jobs = json.load(f)
    
    processor = BatchProcessor()
    
    print(f"🔍 Checking status of {len(batch_jobs)} batch jobs...")
    
    completed_batches = 0
    failed_batches = 0
    pending_batches = 0
    
    for job in batch_jobs:
        chunk_number = job['chunk_number']
        batch_id = job['batch_id']
        output_dir = job['output_dir']
        
        print(f"\n📦 Chunk {chunk_number} (Batch ID: {batch_id})")
        
        # Check status
        status_info = processor.check_batch_status(batch_id)
        if not status_info:
            print(f"   ❌ Failed to get status")
            failed_batches += 1
            continue
        
        status = status_info['status']
        print(f"   📊 Status: {status}")
        
        if 'request_counts' in status_info and status_info['request_counts']:
            counts = status_info['request_counts']
            print(f"   📈 Progress: {counts.get('completed', 0)}/{counts.get('total', 0)} completed, {counts.get('failed', 0)} failed")
        
        # If completed, download results
        if status == 'completed' and status_info.get('output_file_id'):
            output_file = os.path.join(output_dir, f"batch_results_{chunk_number:03d}.jsonl")
            csv_file = os.path.join(output_dir, f"analyzed_profiles_{chunk_number:03d}.csv")
            
            # Check if we already downloaded results
            if os.path.exists(csv_file):
                print(f"   ✅ Already processed: {csv_file}")
                completed_batches += 1
            else:
                print(f"   💾 Downloading results...")
                if processor.download_results(status_info['output_file_id'], output_file):
                    if processor.process_results(output_file, csv_file):
                        print(f"   ✅ Results processed: {csv_file}")
                        completed_batches += 1
                    else:
                        print(f"   ❌ Failed to process results")
                        failed_batches += 1
                else:
                    print(f"   ❌ Failed to download results")
                    failed_batches += 1
        elif status in ['failed', 'expired', 'cancelled']:
            print(f"   ❌ Batch {status}")
            if status_info.get('error_file_id'):
                error_file = os.path.join(output_dir, f"batch_errors_{chunk_number:03d}.jsonl")
                processor.download_results(status_info['error_file_id'], error_file)
                print(f"   📄 Error file saved: {error_file}")
            failed_batches += 1
        else:
            print(f"   ⏳ Still {status}")
            pending_batches += 1
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Completed: {completed_batches}")
    print(f"   ⏳ Pending: {pending_batches}")
    print(f"   ❌ Failed: {failed_batches}")
    
    if completed_batches > 0:
        print(f"\n📁 Results are organized in batch_XXX directories")
        print(f"💡 To combine all results into one file, run: python combine_batch_results.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())