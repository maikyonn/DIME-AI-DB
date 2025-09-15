#!/usr/bin/env python3
"""
Download OpenAI Batch Results
Downloads all completed batch jobs from remaining_batch_jobs_info.json
"""

import os
import sys
import json
import time
import dotenv
import argparse
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Load environment variables
dotenv.load_dotenv()

class BatchDownloader:
    """OpenAI Batch results downloader"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY")) if api_key or os.getenv("OPENAI_API_KEY") else None
        if not self.client:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Check the status of a batch"""
        try:
            batch = self.client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "failed_at": batch.failed_at,
                "expired_at": batch.expired_at,
                "request_counts": batch.request_counts,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id
            }
        except Exception as e:
            print(f"❌ Error checking batch status for {batch_id}: {e}")
            return {}
    
    def download_results(self, file_id: str, output_path: str) -> bool:
        """Download batch results from OpenAI"""
        try:
            file_response = self.client.files.content(file_id)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(file_response.text)
            print(f"💾 Downloaded results to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error downloading results: {e}")
            return False
    
    def download_error_file(self, file_id: str, output_path: str) -> bool:
        """Download batch error file from OpenAI"""
        try:
            file_response = self.client.files.content(file_id)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(file_response.text)
            print(f"⚠️ Downloaded error file to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error downloading error file: {e}")
            return False
    
    def process_batch_results(self, results_file: str, output_csv: str, chunk_number: int) -> bool:
        """Process batch results and create CSV for a specific chunk"""
        try:
            if not os.path.exists(results_file):
                print(f"❌ Results file not found: {results_file}")
                return False
            
            results = []
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line.strip())
                        results.append(result)
            
            if not results:
                print(f"❌ No results found in {results_file}")
                return False
            
            # Process results and extract scores
            processed_data = []
            for result in results:
                custom_id = result.get('custom_id', '')
                lance_db_id = custom_id.replace('profile-', '') if custom_id.startswith('profile-') else ''
                
                if result.get('response') and result['response'].get('status_code') == 200:
                    response_body = result['response']['body']
                    
                    # Extract text from response output
                    text_content = ""
                    if 'output' in response_body:
                        for output_item in response_body['output']:
                            if output_item.get('type') == 'message' and 'content' in output_item:
                                for content_item in output_item['content']:
                                    if content_item.get('type') == 'output_text':
                                        text_content = content_item.get('text', '')
                                        break
                    
                    # Parse CSV scores and keywords (expecting 14 values: 4 scores + 10 keywords)
                    if ',' in text_content and text_content.count(',') == 13:
                        try:
                            values = text_content.strip().split(',')
                            individual_vs_org = float(values[0])
                            generational_appeal = float(values[1])
                            professionalization = float(values[2])
                            relationship_status = float(values[3])
                            keywords = [keyword.strip() for keyword in values[4:14]]
                            
                            processed_data.append({
                                'lance_db_id': lance_db_id,
                                'custom_id': custom_id,
                                'individual_vs_org': individual_vs_org,
                                'generational_appeal': generational_appeal,
                                'professionalization': professionalization,
                                'relationship_status': relationship_status,
                                'keyword1': keywords[0] if len(keywords) > 0 else '',
                                'keyword2': keywords[1] if len(keywords) > 1 else '',
                                'keyword3': keywords[2] if len(keywords) > 2 else '',
                                'keyword4': keywords[3] if len(keywords) > 3 else '',
                                'keyword5': keywords[4] if len(keywords) > 4 else '',
                                'keyword6': keywords[5] if len(keywords) > 5 else '',
                                'keyword7': keywords[6] if len(keywords) > 6 else '',
                                'keyword8': keywords[7] if len(keywords) > 7 else '',
                                'keyword9': keywords[8] if len(keywords) > 8 else '',
                                'keyword10': keywords[9] if len(keywords) > 9 else '',
                                'raw_response': text_content
                            })
                        except ValueError as ve:
                            print(f"⚠️ Invalid scores for {custom_id}: {text_content} - {ve}")
                    else:
                        expected_commas = 13
                        actual_commas = text_content.count(',')
                        print(f"⚠️ Invalid CSV format for {custom_id} (expected {expected_commas + 1} values, got {actual_commas + 1}): {text_content[:100]}...")
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"❌ Failed request for {custom_id}: {error}")
            
            # Write to CSV
            if processed_data:
                import csv
                with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['lance_db_id', 'custom_id', 'individual_vs_org', 'generational_appeal', 
                                'professionalization', 'relationship_status', 'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 
                                'keyword6', 'keyword7', 'keyword8', 'keyword9', 'keyword10', 'raw_response']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(processed_data)
                
                print(f"✅ Processed {len(processed_data)} profiles for chunk {chunk_number} and saved to: {output_csv}")
                return True
            else:
                print(f"❌ No valid results to process for chunk {chunk_number}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing results for chunk {chunk_number}: {e}")
            return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Download OpenAI batch results")
    parser.add_argument("--batch-info", default="remaining_batch_jobs_info.json", 
                       help="Path to the JSON file containing batch job information (default: remaining_batch_jobs_info.json)")
    parser.add_argument("--status-only", action="store_true", 
                       help="Only check batch status without downloading")
    parser.add_argument("--force-download", action="store_true", 
                       help="Download results even if status is not completed")
    parser.add_argument("--process-results", action="store_true", 
                       help="Process downloaded results into CSV files")
    
    args = parser.parse_args()
    
    print("🚀 OpenAI Batch Results Downloader")
    print(f"📄 Batch info file: {args.batch_info}")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return 1
    
    # Load batch job info
    if not os.path.exists(args.batch_info):
        print(f"❌ Batch info file not found: {args.batch_info}")
        return 1
    
    try:
        with open(args.batch_info, 'r') as f:
            batch_jobs = json.load(f)
    except Exception as e:
        print(f"❌ Error reading batch info file: {e}")
        return 1
    
    if not batch_jobs:
        print("❌ No batch jobs found in info file")
        return 1
    
    # Initialize downloader
    try:
        downloader = BatchDownloader()
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    print(f"📊 Found {len(batch_jobs)} batch jobs")
    
    # Process each batch
    completed_count = 0
    failed_count = 0
    pending_count = 0
    downloaded_count = 0
    
    for i, job in enumerate(batch_jobs, 1):
        batch_id = job.get('batch_id')
        chunk_number = job.get('chunk_number', i)
        output_dir = job.get('output_dir', f'batch_{chunk_number:03d}')
        
        if not batch_id:
            print(f"⚠️ Chunk {chunk_number}: No batch_id found, skipping")
            continue
        
        print(f"\n📦 Processing Chunk {chunk_number} (Batch ID: {batch_id})")
        
        # Check batch status
        status_info = downloader.check_batch_status(batch_id)
        if not status_info:
            failed_count += 1
            continue
        
        status = status_info['status']
        print(f"📊 Status: {status}")
        
        # Show request counts if available
        if 'request_counts' in status_info and status_info['request_counts']:
            counts = status_info['request_counts']
            total = getattr(counts, 'total', 0)
            completed = getattr(counts, 'completed', 0)
            failed = getattr(counts, 'failed', 0)
            print(f"   Requests - Total: {total}, Completed: {completed}, Failed: {failed}")
        
        # If only checking status, continue to next batch
        if args.status_only:
            if status == 'completed':
                completed_count += 1
            elif status in ['failed', 'expired', 'cancelled']:
                failed_count += 1
            else:
                pending_count += 1
            continue
        
        # Download results if batch is completed or force download is enabled
        if status == 'completed' or args.force_download:
            if status != 'completed':
                print(f"⚠️ Batch status is '{status}' but force download is enabled")
            
            output_file_id = status_info.get('output_file_id')
            error_file_id = status_info.get('error_file_id')
            
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"📁 Created directory: {output_dir}")
            
            results_downloaded = False
            errors_downloaded = False
            
            # Download results file
            if output_file_id:
                results_file = os.path.join(output_dir, f"batch_results_{chunk_number:03d}.jsonl")
                if downloader.download_results(output_file_id, results_file):
                    results_downloaded = True
                    downloaded_count += 1
                    
                    # Process results if requested
                    if args.process_results:
                        csv_file = os.path.join(output_dir, f"analyzed_profiles_{chunk_number:03d}.csv")
                        downloader.process_batch_results(results_file, csv_file, chunk_number)
            else:
                print("⚠️ No output file ID available")
            
            # Download error file if it exists
            if error_file_id:
                error_file = os.path.join(output_dir, f"batch_errors_{chunk_number:03d}.jsonl")
                if downloader.download_error_file(error_file_id, error_file):
                    errors_downloaded = True
            
            if results_downloaded:
                completed_count += 1
                print(f"✅ Chunk {chunk_number} completed successfully")
                if errors_downloaded:
                    print(f"⚠️ Error file also downloaded for chunk {chunk_number}")
            else:
                failed_count += 1
                print(f"❌ Failed to download results for chunk {chunk_number}")
        
        elif status in ['failed', 'expired', 'cancelled']:
            failed_count += 1
            print(f"❌ Batch failed with status: {status}")
            
            # Try to download error file if available
            error_file_id = status_info.get('error_file_id')
            if error_file_id:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                error_file = os.path.join(output_dir, f"batch_errors_{chunk_number:03d}.jsonl")
                downloader.download_error_file(error_file_id, error_file)
        
        else:
            pending_count += 1
            print(f"⏳ Batch still processing (status: {status})")
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Completed: {completed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   ⏳ Pending: {pending_count}")
    
    if not args.status_only:
        print(f"   💾 Downloaded: {downloaded_count}")
        if args.process_results:
            print(f"   📄 CSV files created for completed batches")
    
    if pending_count > 0:
        print(f"\n💡 {pending_count} batch(es) still processing. Run this script again later to download their results.")
    
    if failed_count > 0:
        print(f"\n⚠️ {failed_count} batch(es) failed. Check error files in respective output directories.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())