#!/usr/bin/env python3
"""
Organize Batch Results for Database
Downloads and processes all batch results into a single unified CSV file
Handles malformed responses and creates database-ready output
"""

import os
import sys
import json
import csv
import re
import time
import dotenv
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Load environment variables
dotenv.load_dotenv()

class BatchOrganizer:
    """Organizes batch results for database ingestion"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY")) if api_key or os.getenv("OPENAI_API_KEY") else None
        if not self.client:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    def download_batch_results(self, batch_id: str, output_file: str) -> bool:
        """Download batch results from OpenAI"""
        try:
            # Get batch status
            batch = self.client.batches.retrieve(batch_id)
            if batch.status != 'completed':
                print(f"⚠️ Batch {batch_id} status: {batch.status}")
                return False
            
            if not batch.output_file_id:
                print(f"❌ No output file for batch {batch_id}")
                return False
            
            # Download results
            file_response = self.client.files.content(batch.output_file_id)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(file_response.text)
            
            print(f"✅ Downloaded batch {batch_id} to {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error downloading batch {batch_id}: {e}")
            return False
    
    def parse_csv_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse CSV response with error handling for malformed data"""
        if not response_text or not response_text.strip():
            return None
        
        # Clean the response text
        response_text = response_text.strip()
        
        # Handle various malformed responses
        if response_text.startswith("Please provide") or response_text.startswith("I'm missing"):
            return {
                'individual_vs_org': None,
                'generational_appeal': None,
                'professionalization': None,
                'relationship_status': None,
                'keywords': ['missing_data'] * 10,
                'error': 'missing_input_data'
            }
        
        # Try to extract CSV line from longer responses
        lines = response_text.split('\n')
        csv_line = None
        
        for line in lines:
            line = line.strip()
            if ',' in line and not line.startswith('Please') and not line.startswith('I'):
                # Count commas to see if this looks like our CSV format
                comma_count = line.count(',')
                if comma_count >= 10:  # At least some keywords
                    csv_line = line
                    break
        
        if not csv_line:
            # If no clear CSV line found, try the entire response as CSV
            csv_line = response_text
        
        # Parse the CSV line
        try:
            # Handle quoted values and complex parsing
            values = []
            current_value = ""
            in_quotes = False
            
            for char in csv_line:
                if char == '"' and not in_quotes:
                    in_quotes = True
                elif char == '"' and in_quotes:
                    in_quotes = False
                elif char == ',' and not in_quotes:
                    values.append(current_value.strip().strip('"'))
                    current_value = ""
                else:
                    current_value += char
            
            # Add the last value
            values.append(current_value.strip().strip('"'))
            
            # Clean up values
            cleaned_values = []
            for val in values:
                # Remove extra whitespace and clean up
                val = val.strip()
                # Remove any remaining quotes
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                cleaned_values.append(val)
            
            values = cleaned_values
            
            # Ensure we have at least 4 numeric values
            if len(values) < 4:
                return {
                    'individual_vs_org': None,
                    'generational_appeal': None,
                    'professionalization': None,
                    'relationship_status': None,
                    'keywords': ['parse_error'] * 10,
                    'error': f'insufficient_values_{len(values)}'
                }
            
            # Parse numeric scores
            try:
                individual_vs_org = float(values[0]) if values[0] and values[0] != '' else None
                generational_appeal = float(values[1]) if values[1] and values[1] != '' else None
                professionalization = float(values[2]) if values[2] and values[2] != '' else None
                relationship_status = float(values[3]) if values[3] and values[3] != '' else None
            except ValueError as e:
                return {
                    'individual_vs_org': None,
                    'generational_appeal': None,
                    'professionalization': None,
                    'relationship_status': None,
                    'keywords': ['numeric_error'] * 10,
                    'error': f'numeric_parse_error: {e}'
                }
            
            # Extract keywords (up to 10)
            keywords = []
            for i in range(4, min(len(values), 14)):
                keyword = values[i].strip()
                if keyword:
                    # Clean keyword - remove special characters, spaces
                    keyword = re.sub(r'[^a-zA-Z0-9_\u0080-\uFFFF]', '', keyword)
                    if keyword:
                        keywords.append(keyword)
            
            # Pad keywords to 10
            while len(keywords) < 10:
                keywords.append('')
            
            # Truncate to 10
            keywords = keywords[:10]
            
            return {
                'individual_vs_org': individual_vs_org,
                'generational_appeal': generational_appeal,
                'professionalization': professionalization,
                'relationship_status': relationship_status,
                'keywords': keywords,
                'error': None
            }
            
        except Exception as e:
            return {
                'individual_vs_org': None,
                'generational_appeal': None,
                'professionalization': None,
                'relationship_status': None,
                'keywords': ['exception_error'] * 10,
                'error': f'parse_exception: {str(e)}'
            }
    
    def process_batch_file(self, results_file: str) -> List[Dict[str, Any]]:
        """Process a batch results file"""
        processed_data = []
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        result = json.loads(line.strip())
                        custom_id = result.get('custom_id', '')
                        lance_db_id = custom_id.replace('profile-', '') if custom_id.startswith('profile-') else ''
                        
                        # Initialize data structure
                        processed_item = {
                            'lance_db_id': lance_db_id,
                            'custom_id': custom_id,
                            'individual_vs_org': None,
                            'generational_appeal': None,
                            'professionalization': None,
                            'relationship_status': None,
                            'keyword1': '',
                            'keyword2': '',
                            'keyword3': '',
                            'keyword4': '',
                            'keyword5': '',
                            'keyword6': '',
                            'keyword7': '',
                            'keyword8': '',
                            'keyword9': '',
                            'keyword10': '',
                            'raw_response': '',
                            'processing_error': ''
                        }
                        
                        # Check if request was successful
                        if result.get('response') and result['response'].get('status_code') == 200:
                            response_body = result['response']['body']
                            
                            # Extract text from response
                            text_content = ""
                            if 'output' in response_body:
                                for output_item in response_body['output']:
                                    if output_item.get('type') == 'message' and 'content' in output_item:
                                        for content_item in output_item['content']:
                                            if content_item.get('type') == 'output_text':
                                                text_content = content_item.get('text', '')
                                                break
                            
                            processed_item['raw_response'] = text_content
                            
                            # Parse the response
                            parsed_data = self.parse_csv_response(text_content)
                            if parsed_data:
                                processed_item.update({
                                    'individual_vs_org': parsed_data['individual_vs_org'],
                                    'generational_appeal': parsed_data['generational_appeal'],
                                    'professionalization': parsed_data['professionalization'],
                                    'relationship_status': parsed_data['relationship_status'],
                                    'keyword1': parsed_data['keywords'][0],
                                    'keyword2': parsed_data['keywords'][1],
                                    'keyword3': parsed_data['keywords'][2],
                                    'keyword4': parsed_data['keywords'][3],
                                    'keyword5': parsed_data['keywords'][4],
                                    'keyword6': parsed_data['keywords'][5],
                                    'keyword7': parsed_data['keywords'][6],
                                    'keyword8': parsed_data['keywords'][7],
                                    'keyword9': parsed_data['keywords'][8],
                                    'keyword10': parsed_data['keywords'][9],
                                    'processing_error': parsed_data['error'] or ''
                                })
                        else:
                            # Handle failed requests
                            error = result.get('error', 'Unknown error')
                            processed_item['processing_error'] = f"API_error: {error}"
                            processed_item['raw_response'] = f"Error: {error}"
                        
                        processed_data.append(processed_item)
                        
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON decode error on line {line_num}: {e}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Processing error on line {line_num}: {e}")
                        continue
        
        except Exception as e:
            print(f"❌ Error processing batch file {results_file}: {e}")
            return []
        
        return processed_data

def main():
    """Main execution function"""
    print("🚀 Batch Results Organizer")
    print("📊 Organizing all batch results for database ingestion")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return 1
    
    # Load batch job info
    batch_info_file = "remaining_batch_jobs_info.json"
    if not os.path.exists(batch_info_file):
        print(f"❌ Batch info file not found: {batch_info_file}")
        return 1
    
    try:
        with open(batch_info_file, 'r') as f:
            batch_jobs = json.load(f)
    except Exception as e:
        print(f"❌ Error reading batch info file: {e}")
        return 1
    
    if not batch_jobs:
        print("❌ No batch jobs found in info file")
        return 1
    
    # Initialize organizer
    try:
        organizer = BatchOrganizer()
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    # Create results directory
    results_dir = "organized_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"📁 Created directory: {results_dir}")
    
    print(f"📊 Processing {len(batch_jobs)} batch jobs...")
    
    # Download and process each batch
    all_results = []
    successful_batches = 0
    
    for i, job in enumerate(batch_jobs, 1):
        batch_id = job.get('batch_id')
        chunk_number = job.get('chunk_number', i)
        
        if not batch_id:
            print(f"⚠️ Chunk {chunk_number}: No batch_id found, skipping")
            continue
        
        print(f"\n📦 Processing Chunk {chunk_number} (Batch ID: {batch_id})")
        
        # Download batch results
        results_file = os.path.join(results_dir, f"batch_results_{chunk_number:03d}.jsonl")
        if not organizer.download_batch_results(batch_id, results_file):
            continue
        
        # Process the results
        chunk_results = organizer.process_batch_file(results_file)
        if chunk_results:
            all_results.extend(chunk_results)
            successful_batches += 1
            print(f"✅ Processed {len(chunk_results)} profiles from chunk {chunk_number}")
        else:
            print(f"❌ Failed to process chunk {chunk_number}")
    
    if not all_results:
        print("❌ No results to save")
        return 1
    
    # Create unified CSV
    unified_csv = os.path.join(results_dir, "unified_batch_results.csv")
    try:
        with open(unified_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'lance_db_id', 'custom_id', 'individual_vs_org', 'generational_appeal',
                'professionalization', 'relationship_status',
                'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5',
                'keyword6', 'keyword7', 'keyword8', 'keyword9', 'keyword10',
                'raw_response', 'processing_error'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✅ Created unified CSV: {unified_csv}")
        print(f"📊 Total profiles: {len(all_results)}")
        
        # Statistics
        valid_results = sum(1 for r in all_results if r['individual_vs_org'] is not None)
        error_results = sum(1 for r in all_results if r['processing_error'])
        
        print(f"📈 Statistics:")
        print(f"   ✅ Valid results: {valid_results}")
        print(f"   ⚠️ Results with errors: {error_results}")
        print(f"   📊 Success rate: {valid_results/len(all_results)*100:.1f}%")
        
        # Create database-ready CSV (only valid results)
        if valid_results > 0:
            db_ready_csv = os.path.join(results_dir, "database_ready_results.csv")
            with open(db_ready_csv, 'w', newline='', encoding='utf-8') as f:
                fieldnames_db = [
                    'lance_db_id', 'individual_vs_org', 'generational_appeal',
                    'professionalization', 'relationship_status',
                    'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5',
                    'keyword6', 'keyword7', 'keyword8', 'keyword9', 'keyword10'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames_db)
                writer.writeheader()
                
                for result in all_results:
                    if result['individual_vs_org'] is not None:
                        db_row = {k: result[k] for k in fieldnames_db}
                        writer.writerow(db_row)
            
            print(f"✅ Created database-ready CSV: {db_ready_csv}")
        
        print(f"\n🎯 Results organized successfully!")
        print(f"📁 All files saved in: {results_dir}/")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating unified CSV: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())