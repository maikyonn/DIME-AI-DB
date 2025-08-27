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
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.analysis_prompt_template = """Analyze the provided Instagram bio and recent post captions to assign three scores and generate 10 unique keywords as described below, relying on thorough reasoning from bio and content features before concluding with a single CSV row:

- "Individual vs. Organization" Score (0–10):  
  Consider use of pronouns, family mentions, first-person narrative, and company keywords ("LLC," "shop," "brand"). Examine frequency of self versus third-person language. A score of 0 indicates a clear individual account, 2 is an individual content creator, 10 is a company/studio.  
- "Generational Appeal" Score ("Gen Z Fit" 0–10):  
  Detect slang, short-form language, heavy emoji use, popular hashtags, orientation toward audio/video content, and trending cultural references.  
- "Professionalization Index" (0–10):  
  Evaluate for presence of sponsorships, disclaimers ("AD:"), affiliate links, organized highlights, partner tags, or clear evidence the user works with brands.
- "Profile Keywords" (5 unique keywords):
  Extract 5 distinctive keywords that best characterize this profile's content, interests, industry, style, and personality. Focus on unique descriptors, not generic terms. Include content themes, professional areas, lifestyle elements, and distinctive characteristics.

Reasoning Order:  
First, reason and note feature evidence from the profile regarding each scoring dimension and identify distinctive keywords; afterwards, assign a single, final CSV line with the three scores and 5 keywords.

Persist until you have fully analyzed all required features and evidence before proceeding to generate the final CSV output. Always output exactly one CSV row—do not include explanations, text, or formatting beyond the CSV line.

**Output format:**  
- Only a single CSV row with values in this exact order: individual_vs_org,generational_appeal,professionalization,keyword1,keyword2,keyword3,keyword4,keyword5
- Keywords should be single words or short phrases (max 2 words each)
- No additional text, punctuation, or formatting beyond the CSV line

Profile Data:
Account: @{account}
Full Name: {full_name}
Biography: {biography}
Recent Post Captions (First 10 Posts): {posts}

_Reminder: Carefully analyze profile content in detail before assigning numeric scores and generating distinctive keywords in the required CSV output._"""
    
    def read_snap_data(self, csv_path: str = "Snap Data.csv") -> List[Dict[str, str]]:
        """Read all rows from Snap Data CSV"""
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                print(f"📊 Loaded {len(rows)} profiles from {csv_path}")
                return rows
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
    
    def create_output_directory(self, chunk_number: int) -> str:
        """Create output directory for a specific chunk"""
        dir_name = f"batch_{chunk_number:03d}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
    
    def create_batch_input_file(self, profiles: List[Dict[str, str]], output_path: str = "batch_input.jsonl") -> str:
        """Create JSONL batch input file for OpenAI Batch API"""
        batch_requests = []
        
        for i, profile in enumerate(profiles):
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
            
            # Create batch request in required format
            batch_request = {
                "custom_id": f"profile-{i+1}",
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
        
        print(f"📝 Created batch input file: {output_path} with {len(batch_requests)} requests")
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
                profile_index = custom_id.replace('profile-', '') if custom_id.startswith('profile-') else ''
                
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
                    
                    # Parse CSV scores and keywords (expecting 8 values: 3 scores + 5 keywords)
                    if ',' in text_content and text_content.count(',') == 7:
                        try:
                            values = text_content.strip().split(',')
                            individual_vs_org = float(values[0])
                            generational_appeal = float(values[1])
                            professionalization = float(values[2])
                            keywords = [keyword.strip() for keyword in values[3:8]]
                            
                            processed_data.append({
                                'profile_index': profile_index,
                                'custom_id': custom_id,
                                'individual_vs_org': individual_vs_org,
                                'generational_appeal': generational_appeal,
                                'professionalization': professionalization,
                                'keyword1': keywords[0] if len(keywords) > 0 else '',
                                'keyword2': keywords[1] if len(keywords) > 1 else '',
                                'keyword3': keywords[2] if len(keywords) > 2 else '',
                                'keyword4': keywords[3] if len(keywords) > 3 else '',
                                'keyword5': keywords[4] if len(keywords) > 4 else '',
                                'raw_response': text_content
                            })
                        except ValueError:
                            print(f"⚠️ Invalid scores for {custom_id}: {text_content}")
                    else:
                        print(f"⚠️ Invalid CSV format for {custom_id} (expected 8 values, got {text_content.count(',') + 1}): {text_content}")
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"❌ Failed request for {custom_id}: {error}")
            
            # Write to CSV
            if processed_data:
                with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['profile_index', 'custom_id', 'individual_vs_org', 'generational_appeal', 
                                'professionalization', 'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'raw_response']
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

    def process_chunk_batch(self, chunk: List[Dict[str, str]], chunk_number: int, total_chunks: int) -> Dict[str, str]:
        """Process a single chunk as a batch"""
        print(f"\n📦 Processing Chunk {chunk_number}/{total_chunks} ({len(chunk)} profiles)")
        
        # Create output directory for this chunk
        output_dir = self.create_output_directory(chunk_number)
        
        # Create batch input file in the chunk directory
        batch_file = os.path.join(output_dir, f"batch_input_{chunk_number:03d}.jsonl")
        if not self.create_batch_input_file(chunk, batch_file):
            return {"status": "failed", "error": "Failed to create batch input file"}
        
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
    print("🚀 Starting Large-Scale OpenAI Batch Processing for Instagram Profile Analysis")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return 1
    
    # Initialize processor
    processor = BatchProcessor()
    
    # Read CSV data
    profiles = processor.read_snap_data()
    if not profiles:
        return 1
    
    # Split into chunks of 21k
    chunk_size = 21000
    chunks = processor.split_profiles_into_chunks(profiles, chunk_size)
    total_chunks = len(chunks)
    
    print(f"📊 Total profiles: {len(profiles)}")
    print(f"📦 Split into {total_chunks} chunks of max {chunk_size} profiles each")
    
    # Confirm before proceeding
    confirm = input(f"\n⚠️  This will create {total_chunks} separate batch jobs. Continue? (y/n): ").lower().strip()
    if confirm != 'y':
        print("❌ Operation cancelled")
        return 0
    
    # Process each chunk
    batch_jobs = []
    for i, chunk in enumerate(chunks, 1):
        result = processor.process_chunk_batch(chunk, i, total_chunks)
        if result["status"] == "created":
            batch_jobs.append(result)
            print(f"✅ Chunk {i} batch created: {result['batch_id']}")
        else:
            print(f"❌ Failed to create batch for chunk {i}: {result['error']}")
    
    print(f"\n🎯 Successfully created {len(batch_jobs)} batch jobs!")
    print(f"💰 Total cost savings: 50% compared to synchronous API")
    print(f"⏱️ Completion window: 24 hours per batch")
    
    # Save batch job info to file
    batch_info_file = "batch_jobs_info.json"
    with open(batch_info_file, 'w') as f:
        json.dump(batch_jobs, f, indent=2)
    print(f"📄 Batch job info saved to: {batch_info_file}")
    
    # Ask user if they want to monitor
    response = input("\n🔄 Do you want to monitor all batch progress? (y/n): ").lower().strip()
    
    if response == 'y':
        print("🔄 Monitoring all batches...")
        for job in batch_jobs:
            print(f"\n📦 Monitoring Chunk {job['chunk_number']} (Batch ID: {job['batch_id']})")
            success = processor.monitor_batch(job['batch_id'])
            if success:
                # Get final status and download results
                final_status = processor.check_batch_status(job['batch_id'])
                if final_status and final_status.get('output_file_id'):
                    output_file = os.path.join(job['output_dir'], f"batch_results_{job['chunk_number']:03d}.jsonl")
                    csv_file = os.path.join(job['output_dir'], f"analyzed_profiles_{job['chunk_number']:03d}.csv")
                    
                    if processor.download_results(final_status['output_file_id'], output_file):
                        processor.process_results(output_file, csv_file)
                        print(f"✅ Chunk {job['chunk_number']} completed: {csv_file}")
        
        print("\n🎉 All batch processing completed!")
        print("📊 Check individual batch_XXX directories for results")
    else:
        print(f"\n📝 To check status of all batches later, use:")
        print(f"   python check_all_batches.py")
        print(f"\n📄 Batch job details saved in: {batch_info_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())