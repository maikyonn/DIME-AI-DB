#!/usr/bin/env python3
"""
Enhanced database visualization script with text output for single row display.
Displays database rows in a human-readable format for analysis.
"""
import pandas as pd
import lancedb
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class DatabaseRowViewer:
    def __init__(self, db_path: str = "snap_data_lancedb"):
        """Initialize database connection"""
        self.db_path = db_path
        self.db = None
        self.table = None
        
    def connect_database(self, table_name: str = "snap_data") -> bool:
        """Connect to LanceDB and open table"""
        try:
            self.db = lancedb.connect(self.db_path)
            
            # Check if table exists
            available_tables = self.db.table_names()
            if table_name not in available_tables:
                print(f"❌ Table '{table_name}' not found.")
                print(f"Available tables: {available_tables}")
                return False
                
            self.table = self.db.open_table(table_name)
            print(f"✅ Connected to database: {self.db_path}")
            print(f"✅ Opened table: {table_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return False
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get basic table information"""
        if not self.table:
            return {}
        
        try:
            df = self.table.to_pandas()
            info = {
                'total_rows': len(df),
                'columns': list(df.columns),
                'column_count': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
            }
            
            # Check for LLM data
            if 'llm_processed' in df.columns:
                llm_count = df['llm_processed'].sum() if 'llm_processed' in df.columns else 0
                info['llm_analyzed'] = int(llm_count)
                info['llm_coverage'] = f"{llm_count/len(df)*100:.1f}%"
            
            return info
        except Exception as e:
            print(f"❌ Error getting table info: {e}")
            return {}
    
    def format_json_content(self, json_str: str, field_name: str, max_items: int = 3) -> str:
        """Format JSON content for readable display"""
        if not json_str or str(json_str) in ['nan', '', 'None']:
            return f"  {field_name}: No data"
        
        try:
            data = json.loads(json_str)
            
            if isinstance(data, list):
                if len(data) == 0:
                    return f"  {field_name}: Empty list"
                
                output = [f"  {field_name}: {len(data)} items"]
                
                # Show first few items
                for i, item in enumerate(data[:max_items]):
                    if isinstance(item, dict):
                        # Extract key information from dict items
                        if field_name == 'posts':
                            caption = item.get('caption', '')
                            likes = item.get('likes', 0)
                            date = item.get('datetime', '')
                            hashtags = item.get('post_hashtags', [])
                            
                            output.append(f"    Post {i+1}:")
                            output.append(f"      Caption: {caption[:100]}..." if len(caption) > 100 else f"      Caption: {caption}")
                            output.append(f"      Likes: {likes:,}")
                            if hashtags:
                                hashtag_str = ', '.join(hashtags[:5])
                                if len(hashtags) > 5:
                                    hashtag_str += f" (+{len(hashtags)-5} more)"
                                output.append(f"      Hashtags: {hashtag_str}")
                            output.append("")
                            
                        elif field_name == 'highlights':
                            title = item.get('title', item.get('text', ''))
                            output.append(f"    Highlight {i+1}: {title}")
                            
                        elif field_name == 'related_accounts':
                            username = item.get('username', item.get('profile_name', ''))
                            category = item.get('category', '')
                            output.append(f"    Account {i+1}: @{username} ({category})")
                        else:
                            # Generic dict display
                            key_info = []
                            for key in ['title', 'name', 'text', 'username', 'caption'][:3]:
                                if key in item and item[key]:
                                    key_info.append(f"{key}: {str(item[key])[:50]}")
                            output.append(f"    Item {i+1}: {', '.join(key_info) if key_info else str(item)[:100]}")
                    else:
                        output.append(f"    Item {i+1}: {str(item)[:100]}")
                
                if len(data) > max_items:
                    output.append(f"    ... and {len(data) - max_items} more items")
                
                return '\n'.join(output)
            
            elif isinstance(data, dict):
                # Single dict item
                output = [f"  {field_name}: Dictionary with {len(data)} keys"]
                for key, value in list(data.items())[:3]:
                    output.append(f"    {key}: {str(value)[:100]}")
                if len(data) > 3:
                    output.append(f"    ... and {len(data) - 3} more keys")
                return '\n'.join(output)
            
            else:
                return f"  {field_name}: {str(data)[:200]}"
                
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON, treat as plain text
            text = str(json_str)[:200]
            return f"  {field_name}: {text}{'...' if len(str(json_str)) > 200 else ''}"
    
    def format_single_row(self, row: pd.Series, row_index: int = 0) -> str:
        """Format a single database row for readable text display"""
        output = []
        
        # Header
        output.append("=" * 80)
        output.append(f"DATABASE ROW #{row_index + 1}")
        output.append("=" * 80)
        output.append("")
        
        # Basic Profile Information
        output.append("👤 BASIC PROFILE INFORMATION")
        output.append("-" * 40)
        
        basic_fields = [
            ('id', 'Profile ID'),
            ('account', 'Account Handle'),
            ('profile_name', 'Profile Name'),
            ('full_name', 'Full Name'),
            ('profile_url', 'Profile URL'),
            ('biography', 'Biography')
        ]
        
        for field, label in basic_fields:
            if field in row.index and pd.notna(row[field]):
                value = str(row[field])
                if field == 'biography' and len(value) > 150:
                    value = value[:150] + "..."
                output.append(f"  {label}: {value}")
        
        output.append("")
        
        # Account Status & Verification
        output.append("✓ ACCOUNT STATUS")
        output.append("-" * 40)
        
        status_fields = [
            ('is_private', 'Private Account'),
            ('is_verified', 'Verified'),
            ('is_business_account', 'Business Account'),
            ('is_professional_account', 'Professional Account'),
            ('is_joined_recently', 'Recently Joined'),
            ('has_channel', 'Has Channel')
        ]
        
        for field, label in status_fields:
            if field in row.index:
                value = row[field]
                if pd.isna(value):
                    status = "Unknown"
                else:
                    status = "Yes" if value in [True, 'True', 'true', 1, '1'] else "No"
                output.append(f"  {label}: {status}")
        
        output.append("")
        
        # Metrics & Engagement
        output.append("📊 METRICS & ENGAGEMENT")
        output.append("-" * 40)
        
        metric_fields = [
            ('followers', 'Followers'),
            ('following', 'Following'),
            ('posts_count', 'Posts Count'),
            ('avg_engagement', 'Avg Engagement Rate'),
            ('highlights_count', 'Highlights Count')
        ]
        
        for field, label in metric_fields:
            if field in row.index and pd.notna(row[field]):
                value = row[field]
                if field == 'followers' or field == 'following':
                    output.append(f"  {label}: {int(float(value)):,}")
                elif field == 'avg_engagement':
                    output.append(f"  {label}: {float(value):.3f}%")
                else:
                    output.append(f"  {label}: {int(float(value)) if str(value).replace('.', '').isdigit() else value}")
        
        output.append("")
        
        # Business Information
        if any(field in row.index for field in ['business_category_name', 'category_name', 'business_address', 'external_url']):
            output.append("🏢 BUSINESS INFORMATION")
            output.append("-" * 40)
            
            business_fields = [
                ('business_category_name', 'Business Category'),
                ('category_name', 'Category'),
                ('business_address', 'Business Address'),
                ('external_url', 'External URL')
            ]
            
            for field, label in business_fields:
                if field in row.index and pd.notna(row[field]) and str(row[field]) != '':
                    output.append(f"  {label}: {row[field]}")
            
            output.append("")
        
        # LLM Analysis Results
        if 'llm_processed' in row.index and row['llm_processed']:
            output.append("🧠 LLM ANALYSIS RESULTS")
            output.append("-" * 40)
            
            llm_fields = [
                ('individual_vs_org_score', 'Individual vs Organization Score', '0=Individual, 10=Organization'),
                ('generational_appeal_score', 'Generational Appeal Score', '0=Low Gen Z appeal, 10=High Gen Z appeal'),
                ('professionalization_score', 'Professionalization Score', '0=Casual, 10=Highly professional')
            ]
            
            for field, label, description in llm_fields:
                if field in row.index and pd.notna(row[field]):
                    score = int(float(row[field]))
                    output.append(f"  {label}: {score}/10 ({description})")
            
            # Keywords
            keywords = []
            for i in range(1, 6):
                keyword_field = f'keyword{i}'
                if keyword_field in row.index and pd.notna(row[keyword_field]) and str(row[keyword_field]) != '':
                    keywords.append(str(row[keyword_field]))
            
            if keywords:
                output.append(f"  AI Generated Keywords: {', '.join(keywords)}")
            
            output.append("")
        
        # Content Analysis (JSON fields)
        output.append("📱 CONTENT ANALYSIS")
        output.append("-" * 40)
        
        # Posts
        if 'posts' in row.index:
            posts_text = self.format_json_content(row['posts'], 'Posts', max_items=2)
            output.append(posts_text)
            output.append("")
        
        # Highlights
        if 'highlights' in row.index:
            highlights_text = self.format_json_content(row['highlights'], 'Highlights', max_items=3)
            output.append(highlights_text)
            output.append("")
        
        # Related Accounts
        if 'related_accounts' in row.index:
            related_text = self.format_json_content(row['related_accounts'], 'Related Accounts', max_items=5)
            output.append(related_text)
            output.append("")
        
        # Hashtags
        if 'post_hashtags' in row.index and pd.notna(row['post_hashtags']) and str(row['post_hashtags']) != '':
            hashtags = str(row['post_hashtags'])
            if hashtags.startswith('['):
                try:
                    hashtag_list = json.loads(hashtags)
                    if hashtag_list:
                        output.append(f"  Post Hashtags: {', '.join(hashtag_list[:10])}")
                        if len(hashtag_list) > 10:
                            output.append(f"    ... and {len(hashtag_list) - 10} more hashtags")
                except:
                    output.append(f"  Post Hashtags: {hashtags[:100]}")
            else:
                output.append(f"  Post Hashtags: {hashtags[:100]}")
            output.append("")
        
        # Metadata
        output.append("📋 METADATA")
        output.append("-" * 40)
        
        metadata_fields = [
            ('partner_id', 'Partner ID'),
            ('email_address', 'Email Address'),
            ('timestamp', 'Data Timestamp')
        ]
        
        for field, label in metadata_fields:
            if field in row.index and pd.notna(row[field]) and str(row[field]) != '':
                value = str(row[field])
                if field == 'email_address' and '@' in value:
                    # Partially mask email for privacy
                    parts = value.split('@')
                    if len(parts) == 2:
                        masked = parts[0][:3] + '*' * max(0, len(parts[0]) - 3) + '@' + parts[1]
                        value = masked
                output.append(f"  {label}: {value}")
        
        output.append("")
        output.append("=" * 80)
        
        return '\n'.join(output)
    
    def display_row_by_index(self, row_index: int = 0) -> bool:
        """Display a specific row by index"""
        if not self.table:
            print("❌ Database not connected. Call connect_database() first.")
            return False
        
        try:
            df = self.table.to_pandas()
            
            if row_index >= len(df) or row_index < 0:
                print(f"❌ Row index {row_index} out of range. Database has {len(df)} rows (0-{len(df)-1})")
                return False
            
            row = df.iloc[row_index]
            formatted_output = self.format_single_row(row, row_index)
            print(formatted_output)
            return True
            
        except Exception as e:
            print(f"❌ Error displaying row: {e}")
            return False
    
    def display_row_by_account(self, account_name: str) -> bool:
        """Display row for specific account handle"""
        if not self.table:
            print("❌ Database not connected. Call connect_database() first.")
            return False
        
        try:
            df = self.table.to_pandas()
            
            if 'account' not in df.columns:
                print("❌ 'account' column not found in database")
                return False
            
            # Find matching account (case insensitive)
            matches = df[df['account'].str.contains(account_name, case=False, na=False)]
            
            if len(matches) == 0:
                print(f"❌ No account found matching '{account_name}'")
                return False
            elif len(matches) > 1:
                print(f"⚠️ Multiple accounts found matching '{account_name}':")
                for idx, row in matches.iterrows():
                    print(f"  {idx}: @{row['account']} - {row.get('profile_name', 'N/A')}")
                print(f"\nShowing first match:")
            
            # Display first match
            first_match = matches.iloc[0]
            row_index = matches.index[0]
            formatted_output = self.format_single_row(first_match, row_index)
            print(formatted_output)
            return True
            
        except Exception as e:
            print(f"❌ Error finding account: {e}")
            return False
    
    def search_and_display(self, query: str, limit: int = 1) -> bool:
        """Search database and display matching rows"""
        if not self.table:
            print("❌ Database not connected. Call connect_database() first.")
            return False
        
        try:
            # Try vector search if available
            try:
                results = self.table.search(query).limit(limit).to_pandas()
                print(f"🔍 Vector search results for: '{query}'")
            except:
                # Fallback to pandas filtering
                df = self.table.to_pandas()
                text_columns = ['account', 'profile_name', 'full_name', 'biography', 'category_name']
                available_text_cols = [col for col in text_columns if col in df.columns]
                
                mask = pd.Series([False] * len(df))
                for col in available_text_cols:
                    mask |= df[col].astype(str).str.contains(query, case=False, na=False)
                
                results = df[mask].head(limit)
                print(f"🔍 Text search results for: '{query}'")
            
            if len(results) == 0:
                print(f"❌ No results found for: '{query}'")
                return False
            
            print(f"Found {len(results)} result(s)\n")
            
            for idx, (_, row) in enumerate(results.iterrows()):
                if idx > 0:
                    print("\n" + "="*80 + "\n")
                formatted_output = self.format_single_row(row, idx)
                print(formatted_output)
            
            return True
            
        except Exception as e:
            print(f"❌ Error searching database: {e}")
            return False


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Display database rows in human-readable format")
    parser.add_argument("--db", default="snap_data_lancedb", help="Database path")
    parser.add_argument("--table", default="snap_data", help="Table name")
    parser.add_argument("--index", type=int, help="Display row by index")
    parser.add_argument("--account", help="Display row by account handle")
    parser.add_argument("--search", help="Search and display matching rows")
    parser.add_argument("--limit", type=int, default=1, help="Limit search results")
    parser.add_argument("--info", action="store_true", help="Show database info only")
    
    args = parser.parse_args()
    
    # Initialize viewer
    viewer = DatabaseRowViewer(args.db)
    
    # Connect to database
    if not viewer.connect_database(args.table):
        sys.exit(1)
    
    # Show database info
    if args.info:
        info = viewer.get_table_info()
        print("\n📊 DATABASE INFORMATION")
        print("=" * 40)
        for key, value in info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print()
        return
    
    # Handle different display modes
    success = False
    
    if args.account:
        success = viewer.display_row_by_account(args.account)
    elif args.search:
        success = viewer.search_and_display(args.search, args.limit)
    elif args.index is not None:
        success = viewer.display_row_by_index(args.index)
    else:
        # Default: show first row
        print("📋 Displaying first row (use --help for more options)")
        success = viewer.display_row_by_index(0)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()