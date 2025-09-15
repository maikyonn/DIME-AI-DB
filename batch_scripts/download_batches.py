#!/usr/bin/env python3
"""
Batch download script for OpenAI batch jobs.
Downloads all batch results from batch_jobs_info.json using OpenAI Python library.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI library not found. Install with: pip install openai")
    sys.exit(1)

def load_batch_info(file_path="batch_jobs_info.json"):
    """Load batch job information from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        sys.exit(1)

def download_batch(client, batch_id, output_dir):
    """Download a single batch using OpenAI Python library."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{batch_id}_results.jsonl")
    
    print(f"Downloading {batch_id} to {output_file}")
    
    try:
        # Check batch status first
        batch = client.batches.retrieve(batch_id)
        print(f"  Status: {batch.status}")
        
        if batch.status != "completed":
            print(f"  ⚠️ Batch not completed (status: {batch.status})")
            return False
            
        if not batch.output_file_id:
            print(f"  ❌ No output file available")
            return False
        
        # Download the results
        file_response = client.files.content(batch.output_file_id)
        
        # Save to file
        with open(output_file, 'wb') as f:
            f.write(file_response.content)
            
        print(f"  ✅ Successfully downloaded {batch_id}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to download {batch_id}: {str(e)}")
        return False

def main():
    """Main function to download all batches."""
    print("Loading batch information...")
    batches = load_batch_info()
    
    print(f"Found {len(batches)} batches to download")
    
    # Initialize OpenAI client
    try:
        client = OpenAI()  # Uses OPENAI_API_KEY environment variable
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {str(e)}")
        print("Make sure OPENAI_API_KEY environment variable is set")
        sys.exit(1)
    
    success_count = 0
    failed_count = 0
    
    for batch in batches:
        batch_id = batch["batch_id"]
        output_dir = batch["output_dir"]
        chunk_number = batch["chunk_number"]
        
        print(f"\nProcessing chunk {chunk_number}: {batch_id}")
        
        if download_batch(client, batch_id, output_dir):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Total: {len(batches)}")

if __name__ == "__main__":
    main()