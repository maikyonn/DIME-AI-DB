#!/usr/bin/env python3
"""
OpenAI Batch API processor for Instagram Profile Analysis
Processes all rows from Snap Data CSV using OpenAI Batch API for 50% cost savings
"""

import os
import sys
import csv
import json
import time
import math
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

class BatchProcessor:
    """OpenAI Batch API processor for Instagram profile analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY")) if api_key or os.getenv("OPENAI_API_KEY") else None
        self.analysis_prompt_template = """Analyze the provided Instagram bio and recent post captions to assign four scores and generate 10 unique keywords as described below, relying on thorough reasoning from bio and content features before concluding with a single CSV row:

- "Individual vs. Organization" Score (0–10):  
  Consider use of pronouns, family mentions, first-person narrative, and company keywords ("LLC," "shop," "brand"). Examine frequency of self versus third-person language. A score of 0 indicates a clear individual account, 2 is an individual content creator, 10 is a company/studio.  
- "Generational Appeal" Score ("Gen Z Fit" 0–10):  
  Detect slang, short-form language, heavy emoji use, popular hashtags, orientation toward audio/video content, and trending cultural references.  
- "Professionalization Index" (0–10):  
  Evaluate for presence of sponsorships, disclaimers ("AD:"), affiliate links, organized highlights, partner tags, or clear evidence the user works with brands.
- "Relationship Status" Score (0–10):
  Determine the apparent relationship status based on bio mentions, partner tags, couple photos, wedding references, or family content. A score of 0 indicates clearly single, 3-4 indicates dating/in relationship, 6-7 indicates engaged/married, 8-10 indicates family/couples account with children or joint account management.
- "Profile Keywords" (10 unique keywords):
  Extract 10 distinctive keywords that best characterize this profile's content, interests, industry, style, and personality. Focus on unique descriptors, not generic terms. Include content themes, professional areas, lifestyle elements, and distinctive characteristics.

Reasoning Order:  
First, reason and note feature evidence from the profile regarding each scoring dimension and identify distinctive keywords; afterwards, assign a single, final CSV line with the four scores and 10 keywords.

Persist until you have fully analyzed all required features and evidence before proceeding to generate the final CSV output. Always output exactly one CSV row—do not include explanations, text, or formatting beyond the CSV line.

**Output format:**  
- Only a single CSV row with values in this exact order: individual_vs_org,generational_appeal,professionalization,relationship_status,keyword1,keyword2,keyword3,keyword4,keyword5,keyword6,keyword7,keyword8,keyword9,keyword10
- Keywords should be single words or short phrases (max 2 words each)
- No additional text, punctuation, or formatting beyond the CSV line

Profile Data:
Account: @{account}
Full Name: {full_name}
Biography: {biography}
Recent Post Captions (First 10 Posts): {posts}

_Reminder: Carefully analyze profile content in detail before assigning numeric scores and generating distinctive keywords in the required CSV output._"""
    
    def read_snap_data(self, csv_path: str) -> List[Dict[str, str]]:
        """Read all rows from CSV file and ensure lance_db_id uniqueness"""
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            # Check for lance_db_id column
            if 'lance_db_id' not in rows[0].keys() if rows else True:
                print(f"❌ CSV file must contain 'lance_db_id' column")
                return []
            
            # Check for unique lance_db_ids
            lance_db_ids = [row['lance_db_id'] for row in rows if row['lance_db_id']]
            unique_lance_db_ids = set(lance_db_ids)
            
            if len(lance_db_ids) != len(unique_lance_db_ids):
                duplicate_count = len(lance_db_ids) - len(unique_lance_db_ids)
                print(f"❌ Found {duplicate_count} duplicate lance_db_id(s) in CSV. All lance_db_ids must be unique.")
                
                # Show some duplicate lance_db_ids for debugging
                seen = set()
                duplicates = set()
                for lance_db_id in lance_db_ids:
                    if lance_db_id in seen:
                        duplicates.add(lance_db_id)
                    seen.add(lance_db_id)
                
                print(f"   Example duplicate lance_db_ids: {list(duplicates)[:5]}")
                return []
            
            # Filter out rows with empty lance_db_ids
            valid_rows = [row for row in rows if row['lance_db_id'].strip()]
            empty_lance_db_id_count = len(rows) - len(valid_rows)
            
            if empty_lance_db_id_count > 0:
                print(f"⚠️ Skipped {empty_lance_db_id_count} rows with empty lance_db_id")
            
            print(f"📊 Loaded {len(valid_rows)} profiles with unique lance_db_ids from {csv_path}")
            return valid_rows
            
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return []
    
    def extract_captions_from_posts(self, posts_json_str: str, max_posts: int = 10) -> str:
        """Extract only captions from the first N posts"""
        try:
            import json
            posts = json.loads(posts_json_str)
            captions = []
            
            for i, post in enumerate(posts[:max_posts]):
                if isinstance(post, dict) and 'caption' in post:
                    caption = post['caption'].strip()
                    if caption:
                        captions.append(f"Post {i+1}: {caption}")
            
            return "\n\n".join(captions)
        except:
            # Fallback to original posts if parsing fails
            return posts_json_str[:2000]
    
    def split_profiles_into_chunks(self, profiles: List[Dict[str, str]], chunk_size: int = 21000) -> List[List[Dict[str, str]]]:
        """Split profiles into chunks of specified size"""
        chunks = []
        for i in range(0, len(profiles), chunk_size):
            chunk = profiles[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
    
    def create_output_directory(self, chunk_number: int, base_dir: str = None) -> str:
        """Create output directory for a specific chunk"""
        if base_dir:
            dir_name = os.path.join(base_dir, f"batch_{chunk_number:03d}")
        else:
            dir_name = f"batch_{chunk_number:03d}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
    
    def create_batch_input_file(self, profiles: List[Dict[str, str]], output_path: str = "batch_input.jsonl", start_row_index: int = 0) -> str:
        """Create JSONL batch input file for OpenAI Batch API"""
        batch_requests = []
        
        for i, profile in enumerate(profiles):
            # Use the lance_db_id from the CSV data
            lance_db_id = profile.get('lance_db_id', f'unknown-{i+1}')
            
            account = profile.get('account', '')
            full_name = profile.get('full_name', '')
            biography = profile.get('biography', '')
            posts_raw = profile.get('posts', '')
            
            # Extract captions from first 10 posts
            captions = self.extract_captions_from_posts(posts_raw, max_posts=10)
            
            # Create the analysis prompt for this profile
            prompt = self.analysis_prompt_template.format(
                account=account,
                full_name=full_name,
                biography=biography,
                posts=captions
            )
            
            # Create batch request with lance_db_id as custom_id
            batch_request = {
                "custom_id": f"profile-{lance_db_id}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-5-nano",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "text": {
                        "format": {"type": "text"},
                        "verbosity": "medium"
                    },
                    "reasoning": {
                        "effort": "medium"
                    },
                    "tools": [],
                    "store": True
                }
            }
            batch_requests.append(batch_request)
        
        # Write to JSONL file
        with open(output_path, 'w', encoding='utf-8') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
        
        print(f"📝 Created batch input file: {output_path} with {len(batch_requests)} requests (CSV rows {start_row_index+1}-{start_row_index+len(batch_requests)})")
        return output_path
    
    def upload_batch_file(self, file_path: str) -> str:
        """Upload batch input file to OpenAI"""
        try:
            with open(file_path, 'rb') as f:
                file_obj = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            print(f"📤 Uploaded batch file. File ID: {file_obj.id}")
            return file_obj.id
        except Exception as e:
            print(f"❌ Error uploading file: {e}")
            return ""
    
    def create_batch(self, input_file_id: str, description: str = "Instagram profile analysis batch") -> str:
        """Create batch processing job"""
        try:
            batch = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/responses",
                completion_window="24h",
                metadata={"description": description}
            )
            print(f"🚀 Created batch job. Batch ID: {batch.id}")
            print(f"📊 Status: {batch.status}")
            return batch.id
        except Exception as e:
            print(f"❌ Error creating batch: {e}")
            return ""
    
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
            print(f"❌ Error checking batch status: {e}")
            return {}
    
    def download_results(self, file_id: str, output_path: str = "batch_results.jsonl") -> bool:
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
    
    def process_results(self, results_file: str = "batch_results.jsonl", output_csv: str = "analyzed_profiles.csv") -> bool:
        """Process batch results and create final CSV"""
        try:
            results = []
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line.strip())
                    results.append(result)
            
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
                        except ValueError:
                            print(f"⚠️ Invalid scores for {custom_id}: {text_content}")
                    else:
                        print(f"⚠️ Invalid CSV format for {custom_id} (expected 14 values, got {text_content.count(',') + 1}): {text_content}")
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"❌ Failed request for {custom_id}: {error}")
            
            # Write to CSV
            if processed_data:
                with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['lance_db_id', 'custom_id', 'individual_vs_org', 'generational_appeal', 
                                'professionalization', 'relationship_status', 'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 
                                'keyword6', 'keyword7', 'keyword8', 'keyword9', 'keyword10', 'raw_response']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(processed_data)
                
                print(f"✅ Processed {len(processed_data)} profiles and saved to: {output_csv}")
                return True
            else:
                print("❌ No valid results to process")
                return False
                
        except Exception as e:
            print(f"❌ Error processing results: {e}")
            return False
    
    def monitor_batch(self, batch_id: str, check_interval: int = 300) -> bool:
        """Monitor batch progress until completion"""
        print(f"🔄 Monitoring batch {batch_id}...")
        
        while True:
            status_info = self.check_batch_status(batch_id)
            if not status_info:
                return False
            
            status = status_info['status']
            print(f"📊 Batch Status: {status}")
            
            if 'request_counts' in status_info and status_info['request_counts']:
                counts = status_info['request_counts']
                print(f"   Total: {counts.get('total', 0)}, Completed: {counts.get('completed', 0)}, Failed: {counts.get('failed', 0)}")
            
            if status == 'completed':
                print("✅ Batch completed successfully!")
                return True
            elif status in ['failed', 'expired', 'cancelled']:
                print(f"❌ Batch ended with status: {status}")
                return False
            elif status in ['validating', 'in_progress', 'finalizing']:
                print(f"⏳ Waiting {check_interval} seconds before next check...")
                time.sleep(check_interval)
            else:
                print(f"❓ Unknown status: {status}")
                time.sleep(check_interval)

    def process_chunk_batch(self, chunk: List[Dict[str, str]], chunk_number: int, total_chunks: int, json_only: bool = False, base_dir: str = None) -> Dict[str, str]:
        """Process a single chunk as a batch"""
        print(f"\n📦 Processing Chunk {chunk_number}/{total_chunks} ({len(chunk)} profiles)")
        
        # Create output directory for this chunk
        output_dir = self.create_output_directory(chunk_number, base_dir)
        
        # Create batch input file in the chunk directory
        batch_file = os.path.join(output_dir, f"batch_input_{chunk_number:03d}.jsonl")
        if not self.create_batch_input_file(chunk, batch_file):
            return {"status": "failed", "error": "Failed to create batch input file"}
        
        # If json_only mode, return success without submitting to OpenAI
        if json_only:
            return {
                "status": "json_created",
                "batch_id": None,
                "file_id": None,
                "chunk_number": chunk_number,
                "output_dir": output_dir,
                "profile_count": len(chunk),
                "batch_file": batch_file
            }
        
        # Upload file
        file_id = self.upload_batch_file(batch_file)
        if not file_id:
            return {"status": "failed", "error": "Failed to upload batch file"}
        
        # Create batch
        batch_id = self.create_batch(file_id, f"Instagram profile analysis - Chunk {chunk_number}")
        if not batch_id:
            return {"status": "failed", "error": "Failed to create batch"}
        
        return {
            "status": "created",
            "batch_id": batch_id,
            "file_id": file_id,
            "chunk_number": chunk_number,
            "output_dir": output_dir,
            "profile_count": len(chunk)
        }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Create and submit OpenAI batch jobs for Instagram profile analysis")
    parser.add_argument("csv_file", help="Path to the CSV file containing profile data")
    parser.add_argument("--chunk-size", type=int, default=21000, help="Number of profiles per batch chunk (default: 21000)")
    parser.add_argument("--test-single-batch", action="store_true", help="Only submit the first batch as a test")
    parser.add_argument("--skip-first-batch", action="store_true", help="Skip the first batch and submit all remaining batches")
    parser.add_argument("--json-only", action="store_true", help="Only create JSON files without submitting to OpenAI")
    parser.add_argument("--delay", type=int, default=0, help="Delay in hours between submitting each batch (default: 0, submit all immediately)")
    
    args = parser.parse_args()
    
    # Check for conflicting arguments
    if args.test_single_batch and args.skip_first_batch:
        print("❌ Cannot use both --test-single-batch and --skip-first-batch at the same time")
        return 1
    
    # Validate delay argument
    if args.delay < 0:
        print("❌ Delay must be 0 or positive")
        return 1
    
    print("🚀 Creating and Submitting OpenAI Batch Jobs for Instagram Profile Analysis")
    print(f"📄 Input CSV: {args.csv_file}")
    
    if args.json_only:
        print("📝 JSON-only mode: Creating batch files without submitting to OpenAI")
    elif args.test_single_batch:
        print("🧪 Test mode: Only submitting the first batch")
    elif args.skip_first_batch:
        print("⏭️ Skip mode: Submitting all batches except the first one")
    
    if args.delay > 0:
        print(f"⏰ Delay mode: Will wait {args.delay} hour(s) between each batch submission")
    
    # Check API key only if not in json-only mode
    if not args.json_only:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return 1
    
    # Initialize processor
    processor = BatchProcessor()
    
    # Read CSV data
    profiles = processor.read_snap_data(args.csv_file)
    if not profiles:
        return 1
    
    # Split into chunks
    chunks = processor.split_profiles_into_chunks(profiles, args.chunk_size)
    total_chunks = len(chunks)
    
    print(f"📊 Total profiles: {len(profiles)}")
    print(f"📦 Split into {total_chunks} chunks of max {args.chunk_size} profiles each")
    
    # Determine which chunks to process
    if args.test_single_batch:
        chunks_to_process = chunks[:1]
        print(f"🧪 Testing with first chunk only ({len(chunks_to_process[0])} profiles)")
    elif args.skip_first_batch:
        chunks_to_process = chunks[1:]
        print(f"⏭️ Skipping first batch, processing {len(chunks_to_process)} remaining chunks")
    else:
        chunks_to_process = chunks
        print(f"📦 Processing all {len(chunks_to_process)} chunks")
    
    if not chunks_to_process:
        print("❌ No chunks to process")
        return 0
    
    # Show estimated total time if delay is used
    if args.delay > 0 and not args.json_only and len(chunks_to_process) > 1:
        total_delay_hours = (len(chunks_to_process) - 1) * args.delay
        print(f"⏰ Total estimated time: {total_delay_hours} hours for all batch submissions")
    
    # Confirm before proceeding
    if args.json_only:
        confirm_message = f"\n📝 This will create JSON files for {len(chunks_to_process)} chunk(s). Continue? (y/n): "
    elif args.delay > 0 and len(chunks_to_process) > 1:
        total_delay_hours = (len(chunks_to_process) - 1) * args.delay
        confirm_message = f"\n⚠️  This will create {len(chunks_to_process)} batch job(s) with {args.delay}h delay between submissions (total time: {total_delay_hours}h). Continue? (y/n): "
    else:
        confirm_message = f"\n⚠️  This will create {len(chunks_to_process)} batch job(s). Continue? (y/n): "
    
    confirm = input(confirm_message).lower().strip()
    if confirm != 'y':
        print("❌ Operation cancelled")
        return 0
    
    # Process each chunk
    batch_jobs = []
    start_chunk_number = 2 if args.skip_first_batch else 1
    
    for i, chunk in enumerate(chunks_to_process):
        chunk_number = start_chunk_number + i if args.skip_first_batch else i + 1
        
        # Add delay before submitting (except for the first batch)
        if args.delay > 0 and i > 0 and not args.json_only:
            delay_seconds = args.delay * 3600  # Convert hours to seconds
            print(f"\n⏰ Waiting {args.delay} hour(s) before submitting next batch...")
            print(f"   Time until next submission: {delay_seconds} seconds")
            
            # Show countdown for the first minute, then check every 5 minutes
            countdown_start = time.time()
            while time.time() - countdown_start < delay_seconds:
                remaining = delay_seconds - (time.time() - countdown_start)
                remaining_hours = int(remaining // 3600)
                remaining_minutes = int((remaining % 3600) // 60)
                remaining_seconds = int(remaining % 60)
                
                if remaining <= 60:
                    # Show seconds countdown for last minute
                    print(f"\r   ⏱️  {remaining_seconds:02d}s remaining", end="", flush=True)
                    time.sleep(1)
                elif remaining <= 300:
                    # Show every 10 seconds for last 5 minutes
                    print(f"\r   ⏱️  {remaining_minutes:02d}m {remaining_seconds:02d}s remaining", end="", flush=True)
                    time.sleep(10)
                else:
                    # Show every 5 minutes for longer delays
                    print(f"\r   ⏱️  {remaining_hours:02d}h {remaining_minutes:02d}m remaining", end="", flush=True)
                    time.sleep(300)  # 5 minutes
            
            print(f"\n✅ Delay completed, proceeding with chunk {chunk_number}")
        
        # Get directory of CSV file to create batch folders in the same location
        csv_dir = os.path.dirname(os.path.abspath(args.csv_file))
        result = processor.process_chunk_batch(chunk, chunk_number, total_chunks, args.json_only, csv_dir)
        if result["status"] in ["created", "json_created"]:
            batch_jobs.append(result)
            if args.json_only:
                print(f"✅ Chunk {chunk_number} JSON file created: {result['batch_file']}")
            else:
                print(f"✅ Chunk {chunk_number} batch created: {result['batch_id']}")
                # Add timestamp for batch submission
                result["submitted_at"] = time.time()
        else:
            print(f"❌ Failed to create batch for chunk {chunk_number}: {result['error']}")
    
    # Save batch job info
    if batch_jobs:
        if args.json_only:
            batch_info_file = "json_files_info.json"
        elif args.test_single_batch:
            batch_info_file = "test_batch_job_info.json"
        elif args.skip_first_batch:
            batch_info_file = "remaining_batch_jobs_info.json"
        else:
            batch_info_file = "batch_jobs_info.json"
            
        with open(batch_info_file, 'w') as f:
            json.dump(batch_jobs, f, indent=2)
        
        if args.json_only:
            print(f"\n📝 Successfully created JSON files for {len(batch_jobs)} chunk(s)!")
            print(f"📄 JSON file info saved to: {batch_info_file}")
            print(f"\n📊 JSON Files Summary:")
            for job in batch_jobs:
                print(f"   Chunk {job['chunk_number']}: {job['batch_file']} ({job['profile_count']} profiles)")
            print(f"\n💡 To submit these to OpenAI later, remove the --json-only flag and re-run")
        else:
            print(f"\n🎯 Successfully created {len(batch_jobs)} batch job(s)!")
            print(f"💰 Total cost savings: 50% compared to synchronous API")
            print(f"⏱️ Completion window: 24 hours per batch")
            print(f"📄 Batch job info saved to: {batch_info_file}")
            
            if args.delay > 0 and len(batch_jobs) > 1:
                total_submission_time = (len(batch_jobs) - 1) * args.delay
                print(f"⏰ Total submission time: {total_submission_time} hours")
            
            print(f"\n📊 Batch Job Summary:")
            for job in batch_jobs:
                submitted_time = ""
                if "submitted_at" in job:
                    import datetime
                    submitted_time = f" (submitted at {datetime.datetime.fromtimestamp(job['submitted_at']).strftime('%Y-%m-%d %H:%M:%S')})"
                print(f"   Chunk {job['chunk_number']}: {job['batch_id']} ({job['profile_count']} profiles){submitted_time}")
            
            if args.test_single_batch:
                print(f"\n💡 Test batch submitted. Once you verify the results, run with --skip-first-batch to process the remaining {total_chunks - 1} chunks")
            
            print(f"\n💡 To check batch status later, use the OpenAI API:")
            print(f"   Check individual batch status or use batch management tools")
            print(f"   Batch details are saved in: {batch_info_file}")
    else:
        if args.json_only:
            print("❌ No JSON files were created")
        else:
            print("❌ No batch jobs were created")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())