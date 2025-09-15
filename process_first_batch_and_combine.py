#!/usr/bin/env python3
"""
Process First Batch and Combine with Organized Results
Processes batch_001 and combines it with existing organized results
"""

import os
import sys
import json
import csv
import pandas as pd
import re
from typing import List, Dict, Any, Optional

class BatchCombiner:
    """Combines the first batch with existing organized results"""
    
    def __init__(self):
        pass
    
    def parse_csv_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse CSV response with error handling - same logic as organizer"""
        if not response_text or not response_text.strip():
            return None
        
        response_text = response_text.strip()
        
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
                comma_count = line.count(',')
                if comma_count >= 10:  # At least some keywords
                    csv_line = line
                    break
        
        if not csv_line:
            csv_line = response_text
        
        # Parse the CSV line
        try:
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
            
            values.append(current_value.strip().strip('"'))
            
            # Clean up values
            cleaned_values = []
            for val in values:
                val = val.strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                cleaned_values.append(val)
            
            values = cleaned_values
            
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
                    keyword = re.sub(r'[^a-zA-Z0-9_\u0080-\uFFFF]', '', keyword)
                    if keyword:
                        keywords.append(keyword)
            
            # Pad keywords to 10
            while len(keywords) < 10:
                keywords.append('')
            
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
    
    def process_first_batch(self, results_file: str) -> List[Dict[str, Any]]:
        """Process the first batch results file"""
        processed_data = []
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        result = json.loads(line.strip())
                        custom_id = result.get('custom_id', '')
                        
                        # Extract lance_db_id from custom_id (profile-1 -> 1, profile-2 -> 2, etc.)
                        lance_db_id = custom_id.replace('profile-', '') if custom_id.startswith('profile-') else ''
                        
                        # Initialize data structure
                        processed_item = {
                            'lance_db_id': lance_db_id,
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
                            'keyword10': ''
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
                            
                            # Parse the response
                            parsed_data = self.parse_csv_response(text_content)
                            if parsed_data and parsed_data.get('individual_vs_org') is not None:
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
                                })
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
    print("🚀 Processing First Batch and Combining with Organized Results")
    
    # File paths
    first_batch_file = "batch_001/batch_results_001.jsonl"
    existing_results_file = "organized_results/database_ready_results.csv"
    combined_output_file = "organized_results/complete_database_ready_results.csv"
    
    # Check if files exist
    if not os.path.exists(first_batch_file):
        print(f"❌ First batch file not found: {first_batch_file}")
        return 1
    
    if not os.path.exists(existing_results_file):
        print(f"❌ Existing results file not found: {existing_results_file}")
        return 1
    
    # Initialize processor
    combiner = BatchCombiner()
    
    # Process the first batch
    print(f"📦 Processing first batch: {first_batch_file}")
    first_batch_results = combiner.process_first_batch(first_batch_file)
    
    if not first_batch_results:
        print("❌ No valid results from first batch")
        return 1
    
    print(f"✅ Processed {len(first_batch_results)} valid results from first batch")
    
    # Load existing results
    print(f"📊 Loading existing results: {existing_results_file}")
    existing_df = pd.read_csv(existing_results_file)
    print(f"✅ Loaded {len(existing_df)} existing results")
    
    # Convert first batch results to DataFrame
    first_batch_df = pd.DataFrame(first_batch_results)
    
    # Combine the results
    print("🔗 Combining results...")
    combined_df = pd.concat([first_batch_df, existing_df], ignore_index=True)
    
    # Sort by lance_db_id
    combined_df['lance_db_id'] = combined_df['lance_db_id'].astype(int)
    combined_df = combined_df.sort_values('lance_db_id')
    
    # Save combined results
    combined_df.to_csv(combined_output_file, index=False)
    
    print(f"✅ Created combined results: {combined_output_file}")
    print(f"📊 Total profiles: {len(combined_df)}")
    
    # Statistics
    valid_first_batch = len(first_batch_results)
    valid_existing = len(existing_df)
    total_valid = len(combined_df)
    
    print(f"📈 Final Statistics:")
    print(f"   First batch (001): {valid_first_batch:,}")
    print(f"   Existing batches (002-005): {valid_existing:,}")
    print(f"   Total combined: {total_valid:,}")
    
    # Update the original file to be the complete version
    print("🔄 Updating original database_ready_results.csv...")
    combined_df.to_csv(existing_results_file, index=False)
    print("✅ Original file updated with complete results")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())