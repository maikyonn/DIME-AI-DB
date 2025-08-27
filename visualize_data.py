#!/usr/bin/env python3
"""
Streamlit dashboard for visualizing the unified LanceDB data
combining BrightData CSV with LLM analysis results.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lancedb
from pathlib import Path
from collections import Counter
import numpy as np


@st.cache_data
def load_data():
    """Load data from LanceDB table"""
    try:
        db = lancedb.connect("snap_data_lancedb")
        table = db.open_table("influencer_profiles")
        df = table.to_pandas()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def create_score_distributions(df):
    """Create distribution plots for LLM scores"""
    # Filter records with LLM data
    llm_df = df[df['llm_processed'] == True].copy()
    
    if len(llm_df) == 0:
        st.warning("No records with LLM analysis found")
        return
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            'Individual vs Organization', 
            'Generational Appeal (Gen Z Fit)', 
            'Professionalization Index'
        ]
    )
    
    # Individual vs Org Score
    fig.add_trace(
        go.Histogram(x=llm_df['individual_vs_org_score'], name='Individual vs Org', 
                    marker_color='lightblue', showlegend=False),
        row=1, col=1
    )
    
    # Generational Appeal Score
    fig.add_trace(
        go.Histogram(x=llm_df['generational_appeal_score'], name='Gen Z Fit',
                    marker_color='lightgreen', showlegend=False),
        row=1, col=2
    )
    
    # Professionalization Score
    fig.add_trace(
        go.Histogram(x=llm_df['professionalization_score'], name='Professionalization',
                    marker_color='lightcoral', showlegend=False),
        row=1, col=3
    )
    
    fig.update_layout(
        title_text="Distribution of LLM Analysis Scores",
        height=400
    )
    
    fig.update_xaxes(title_text="Score (0-10)", range=[0, 10])
    fig.update_yaxes(title_text="Count")
    
    st.plotly_chart(fig, use_container_width=True)


def create_keyword_analysis(df):
    """Create keyword frequency analysis"""
    llm_df = df[df['llm_processed'] == True].copy()
    
    if len(llm_df) == 0:
        return
    
    # Collect all keywords
    all_keywords = []
    for col in ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']:
        keywords = llm_df[col].dropna().str.strip()
        all_keywords.extend(keywords[keywords != ''].tolist())
    
    # Count keywords
    keyword_counts = Counter(all_keywords)
    top_keywords = dict(keyword_counts.most_common(20))
    
    if top_keywords:
        fig = px.bar(
            x=list(top_keywords.values()),
            y=list(top_keywords.keys()),
            orientation='h',
            title="Top 20 Keywords from LLM Analysis",
            labels={'x': 'Frequency', 'y': 'Keywords'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def create_follower_analysis(df):
    """Create follower count analysis"""
    # Convert followers to numeric, handle text values
    df_clean = df.copy()
    df_clean['followers_numeric'] = pd.to_numeric(df_clean['followers'], errors='coerce')
    df_clean = df_clean.dropna(subset=['followers_numeric'])
    
    if len(df_clean) == 0:
        st.warning("No valid follower data found")
        return
    
    # Create follower distribution
    fig = px.histogram(
        df_clean, 
        x='followers_numeric',
        nbins=50,
        title="Distribution of Follower Counts",
        labels={'followers_numeric': 'Followers', 'count': 'Number of Profiles'}
    )
    fig.update_xaxes(type="log")  # Log scale for better visualization
    st.plotly_chart(fig, use_container_width=True)


def create_engagement_scatter(df):
    """Create engagement vs followers scatter plot"""
    df_clean = df.copy()
    df_clean['followers_numeric'] = pd.to_numeric(df_clean['followers'], errors='coerce')
    df_clean['engagement_numeric'] = pd.to_numeric(df_clean['avg_engagement'], errors='coerce')
    
    # Filter valid data
    df_clean = df_clean.dropna(subset=['followers_numeric', 'engagement_numeric'])
    df_clean = df_clean[
        (df_clean['followers_numeric'] > 0) & 
        (df_clean['engagement_numeric'] >= 0)
    ]
    
    if len(df_clean) == 0:
        st.warning("No valid engagement data found")
        return
    
    # Add LLM score info if available
    color_col = 'llm_processed'
    if df_clean['llm_processed'].any():
        color_col = 'individual_vs_org_score'
        df_clean[color_col] = df_clean[color_col].fillna(-1)
    
    fig = px.scatter(
        df_clean.sample(min(5000, len(df_clean))),  # Sample for performance
        x='followers_numeric',
        y='engagement_numeric',
        color=color_col,
        title="Engagement vs Followers",
        labels={
            'followers_numeric': 'Followers',
            'engagement_numeric': 'Average Engagement',
            'individual_vs_org_score': 'Individual vs Org Score'
        },
        hover_data=['account', 'full_name'] if 'account' in df_clean.columns else None
    )
    
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)


def create_business_analysis(df):
    """Analyze business vs personal accounts"""
    if 'is_business_account' not in df.columns:
        return
        
    # Business account distribution
    business_counts = df['is_business_account'].value_counts()
    
    fig = px.pie(
        values=business_counts.values,
        names=business_counts.index,
        title="Business vs Personal Accounts"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # If we have LLM data, show correlation with individual vs org score
    llm_df = df[df['llm_processed'] == True].copy()
    if len(llm_df) > 0 and 'individual_vs_org_score' in llm_df.columns:
        fig2 = px.box(
            llm_df,
            x='is_business_account',
            y='individual_vs_org_score',
            title="Individual vs Org Score by Business Account Status"
        )
        st.plotly_chart(fig2, use_container_width=True)


def create_advanced_data_panel(df):
    """Create an advanced data panel with comprehensive filtering"""
    st.subheader("🔍 Advanced Data Explorer")
    
    # Create filter sidebar
    with st.expander("🎛️ Filter Controls", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Account Filters**")
            
            # Account name filter
            account_filter = st.text_input("Account contains:", placeholder="e.g., fashion, tech")
            
            # Biography filter  
            bio_filter = st.text_input("Biography contains:", placeholder="e.g., blogger, artist")
            
            # Business account filter
            if 'is_business_account' in df.columns:
                business_filter = st.selectbox(
                    "Account Type:",
                    ['All', 'Business Only', 'Personal Only']
                )
            else:
                business_filter = 'All'
            
            # Verification filter
            if 'is_verified' in df.columns:
                verified_filter = st.selectbox(
                    "Verification:",
                    ['All', 'Verified Only', 'Unverified Only']
                )
            else:
                verified_filter = 'All'
        
        with col2:
            st.markdown("**Engagement Metrics**")
            
            # Follower range
            if 'followers' in df.columns:
                followers_numeric = pd.to_numeric(df['followers'], errors='coerce').fillna(0)
                max_followers = int(followers_numeric.max()) if not followers_numeric.empty else 1000000
                min_followers = int(followers_numeric.min()) if not followers_numeric.empty else 0
                
                follower_range = st.slider(
                    "Followers Range:",
                    min_value=min_followers,
                    max_value=max_followers,
                    value=(min_followers, max_followers),
                    format="%d"
                )
            else:
                follower_range = (0, 1000000)
            
            # Posts count range
            if 'posts_count' in df.columns:
                posts_numeric = pd.to_numeric(df['posts_count'], errors='coerce').fillna(0)
                max_posts = int(posts_numeric.max()) if not posts_numeric.empty else 1000
                min_posts = int(posts_numeric.min()) if not posts_numeric.empty else 0
                
                posts_range = st.slider(
                    "Posts Count Range:",
                    min_value=min_posts,
                    max_value=min(max_posts, 10000),  # Cap for performance
                    value=(min_posts, min_posts + 1000),
                    format="%d"
                )
            else:
                posts_range = (0, 1000)
            
            # Engagement rate range
            if 'avg_engagement' in df.columns:
                engagement_numeric = pd.to_numeric(df['avg_engagement'], errors='coerce').fillna(0)
                max_engagement = float(engagement_numeric.max()) if not engagement_numeric.empty else 100.0
                min_engagement = float(engagement_numeric.min()) if not engagement_numeric.empty else 0.0
                
                engagement_range = st.slider(
                    "Engagement Rate Range:",
                    min_value=min_engagement,
                    max_value=min(max_engagement, 50.0),  # Cap for reasonable UI
                    value=(min_engagement, min(max_engagement, 10.0)),
                    step=0.1,
                    format="%.1f%%",
                    help="Average engagement rate as percentage"
                )
            else:
                engagement_range = (0.0, 100.0)
        
        with col3:
            st.markdown("**LLM Analysis Filters**")
            
            # LLM processed filter
            if 'llm_processed' in df.columns:
                llm_filter = st.selectbox(
                    "LLM Analysis:",
                    ['All', 'With LLM Analysis', 'Without LLM Analysis']
                )
            else:
                llm_filter = 'All'
            
            # Individual vs Org score range
            if 'individual_vs_org_score' in df.columns:
                org_score_range = st.slider(
                    "Individual vs Org Score:",
                    min_value=0,
                    max_value=10,
                    value=(0, 10),
                    help="0=Individual, 10=Organization"
                )
            else:
                org_score_range = (0, 10)
            
            # Generational appeal range
            if 'generational_appeal_score' in df.columns:
                gen_score_range = st.slider(
                    "Gen Z Appeal Score:",
                    min_value=0,
                    max_value=10,
                    value=(0, 10),
                    help="0=Low Gen Z appeal, 10=High Gen Z appeal"
                )
            else:
                gen_score_range = (0, 10)
            
            # Professionalization range
            if 'professionalization_score' in df.columns:
                prof_score_range = st.slider(
                    "Professionalization Score:",
                    min_value=0,
                    max_value=10,
                    value=(0, 10),
                    help="0=Casual, 10=Highly professional"
                )
            else:
                prof_score_range = (0, 10)
            
            # Keyword filter
            keyword_filter = st.text_input(
                "Contains Keyword:", 
                placeholder="e.g., fashion, food, tech",
                help="Searches across all 5 LLM keywords"
            )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Text filters
    if account_filter:
        if 'account' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['account'].str.contains(account_filter, case=False, na=False)
            ]
    
    if bio_filter:
        if 'biography' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['biography'].str.contains(bio_filter, case=False, na=False)
            ]
    
    # Boolean filters
    if business_filter != 'All' and 'is_business_account' in filtered_df.columns:
        if business_filter == 'Business Only':
            filtered_df = filtered_df[filtered_df['is_business_account'] == True]
        elif business_filter == 'Personal Only':
            filtered_df = filtered_df[filtered_df['is_business_account'] == False]
    
    if verified_filter != 'All' and 'is_verified' in filtered_df.columns:
        if verified_filter == 'Verified Only':
            filtered_df = filtered_df[filtered_df['is_verified'] == True]
        elif verified_filter == 'Unverified Only':
            filtered_df = filtered_df[filtered_df['is_verified'] == False]
    
    # Numeric range filters
    if 'followers' in filtered_df.columns:
        followers_numeric = pd.to_numeric(filtered_df['followers'], errors='coerce').fillna(0)
        filtered_df = filtered_df[
            (followers_numeric >= follower_range[0]) & 
            (followers_numeric <= follower_range[1])
        ]
    
    if 'posts_count' in filtered_df.columns:
        posts_numeric = pd.to_numeric(filtered_df['posts_count'], errors='coerce').fillna(0)
        filtered_df = filtered_df[
            (posts_numeric >= posts_range[0]) & 
            (posts_numeric <= posts_range[1])
        ]
    
    # Engagement rate filter
    if 'avg_engagement' in filtered_df.columns:
        engagement_numeric = pd.to_numeric(filtered_df['avg_engagement'], errors='coerce').fillna(0)
        filtered_df = filtered_df[
            (engagement_numeric >= engagement_range[0]) & 
            (engagement_numeric <= engagement_range[1])
        ]
    
    # LLM filters
    if llm_filter != 'All' and 'llm_processed' in filtered_df.columns:
        if llm_filter == 'With LLM Analysis':
            filtered_df = filtered_df[filtered_df['llm_processed'] == True]
        elif llm_filter == 'Without LLM Analysis':
            filtered_df = filtered_df[filtered_df['llm_processed'] != True]
    
    # LLM score range filters
    if 'individual_vs_org_score' in filtered_df.columns:
        score_mask = (
            (filtered_df['individual_vs_org_score'].isna()) |
            ((filtered_df['individual_vs_org_score'] >= org_score_range[0]) & 
             (filtered_df['individual_vs_org_score'] <= org_score_range[1]))
        )
        filtered_df = filtered_df[score_mask]
    
    if 'generational_appeal_score' in filtered_df.columns:
        score_mask = (
            (filtered_df['generational_appeal_score'].isna()) |
            ((filtered_df['generational_appeal_score'] >= gen_score_range[0]) & 
             (filtered_df['generational_appeal_score'] <= gen_score_range[1]))
        )
        filtered_df = filtered_df[score_mask]
    
    if 'professionalization_score' in filtered_df.columns:
        score_mask = (
            (filtered_df['professionalization_score'].isna()) |
            ((filtered_df['professionalization_score'] >= prof_score_range[0]) & 
             (filtered_df['professionalization_score'] <= prof_score_range[1]))
        )
        filtered_df = filtered_df[score_mask]
    
    # Keyword filter
    if keyword_filter:
        keyword_cols = ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']
        available_keyword_cols = [col for col in keyword_cols if col in filtered_df.columns]
        
        if available_keyword_cols:
            # Create a mask for rows containing the keyword in any keyword column
            keyword_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            for col in available_keyword_cols:
                keyword_mask |= filtered_df[col].str.contains(keyword_filter, case=False, na=False)
            filtered_df = filtered_df[keyword_mask]
    
    # Display results
    st.markdown(f"**📊 Showing {len(filtered_df):,} of {len(df):,} profiles**")
    
    if len(filtered_df) == 0:
        st.warning("No profiles match your filter criteria. Try adjusting the filters.")
        return
    
    # Column selection
    st.markdown("**🔧 Choose Columns to Display:**")
    
    # Categorize columns
    basic_cols = ['account', 'full_name', 'followers', 'following', 'avg_engagement', 'posts_count', 'biography']
    profile_cols = ['is_private', 'is_verified', 'is_business_account', 'category_name', 'external_url']
    llm_cols = ['individual_vs_org_score', 'generational_appeal_score', 'professionalization_score', 
               'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'llm_processed']
    
    available_basic = [col for col in basic_cols if col in filtered_df.columns]
    available_profile = [col for col in profile_cols if col in filtered_df.columns]
    available_llm = [col for col in llm_cols if col in filtered_df.columns]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        basic_selected = st.multiselect(
            "Basic Info:",
            available_basic,
            default=available_basic[:4] if len(available_basic) >= 4 else available_basic
        )
    
    with col2:
        profile_selected = st.multiselect(
            "Profile Details:",
            available_profile,
            default=available_profile[:3] if len(available_profile) >= 3 else available_profile
        )
    
    with col3:
        llm_selected = st.multiselect(
            "LLM Analysis:",
            available_llm,
            default=available_llm if len(available_llm) <= 5 else available_llm[:5]
        )
    
    # Combine selected columns
    display_columns = basic_selected + profile_selected + llm_selected
    
    if not display_columns:
        st.warning("Please select at least one column to display.")
        return
    
    # Display the filtered data
    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        height=600
    )
    
    # Quick stats about filtered data
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if 'followers' in filtered_df.columns:
            avg_followers = pd.to_numeric(filtered_df['followers'], errors='coerce').mean()
            st.metric("Avg Followers", f"{avg_followers:,.0f}" if not np.isnan(avg_followers) else "N/A")
    
    with col2:
        if 'avg_engagement' in filtered_df.columns:
            avg_engagement = pd.to_numeric(filtered_df['avg_engagement'], errors='coerce').mean()
            st.metric("Avg Engagement", f"{avg_engagement:.2f}%" if not np.isnan(avg_engagement) else "N/A")
    
    with col3:
        if 'llm_processed' in filtered_df.columns:
            llm_coverage = filtered_df['llm_processed'].sum() / len(filtered_df) * 100
            st.metric("LLM Coverage", f"{llm_coverage:.1f}%")
    
    with col4:
        if 'is_business_account' in filtered_df.columns:
            business_pct = filtered_df['is_business_account'].sum() / len(filtered_df) * 100
            st.metric("Business %", f"{business_pct:.1f}%")
    
    with col5:
        if 'is_verified' in filtered_df.columns:
            verified_pct = filtered_df['is_verified'].sum() / len(filtered_df) * 100
            st.metric("Verified %", f"{verified_pct:.1f}%")
    
    # Export filtered data
    if st.button("💾 Export Filtered Data"):
        csv = filtered_df.to_csv(index=False)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download Filtered CSV",
            data=csv,
            file_name=f"filtered_profiles_{timestamp}.csv",
            mime="text/csv"
        )


def show_data_sample(df):
    """Show a simple sample of the data"""
    st.subheader("📊 Data Sample (First 10 Records)")
    
    # Show key columns for quick overview
    key_columns = ['account', 'full_name', 'followers', 'following', 'posts_count']
    llm_columns = ['individual_vs_org_score', 'generational_appeal_score', 'keyword1', 'keyword2']
    
    available_key_cols = [col for col in key_columns if col in df.columns]
    available_llm_cols = [col for col in llm_columns if col in df.columns]
    
    display_cols = available_key_cols + available_llm_cols
    
    if display_cols:
        st.dataframe(df[display_cols].head(10), use_container_width=True)
    else:
        st.dataframe(df.head(10), use_container_width=True)


def main():
    st.set_page_config(
        page_title="Instagram Profile Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Instagram Profile Analytics Dashboard")
    st.markdown("*Combining BrightData scraping with GPT-5 LLM analysis*")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Make sure you've run `python rebuild_database_simple.py` first.")
        return
    
    # Sidebar with stats
    st.sidebar.header("📈 Data Overview")
    st.sidebar.metric("Total Profiles", f"{len(df):,}")
    
    if 'llm_processed' in df.columns:
        llm_count = df['llm_processed'].sum()
        st.sidebar.metric("With LLM Analysis", f"{llm_count:,}")
        st.sidebar.metric("Analysis Coverage", f"{llm_count/len(df)*100:.1f}%")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 LLM Analysis", "👥 Followers", "📈 Engagement", "🏢 Business", "🔍 Data Explorer", "📋 Quick View"
    ])
    
    with tab1:
        st.header("🧠 LLM Analysis Results")
        st.markdown("Analysis from GPT-5 scoring profiles on individual vs organization, generational appeal, and professionalization.")
        
        if 'llm_processed' in df.columns and df['llm_processed'].any():
            create_score_distributions(df)
            
            st.subheader("🏷️ Keyword Analysis")
            create_keyword_analysis(df)
        else:
            st.warning("No LLM analysis data found. Check your data files.")
    
    with tab2:
        st.header("👥 Follower Analysis")
        create_follower_analysis(df)
    
    with tab3:
        st.header("📈 Engagement Analysis")
        create_engagement_scatter(df)
    
    with tab4:
        st.header("🏢 Business Account Analysis")
        create_business_analysis(df)
    
    with tab5:
        st.header("🔍 Data Explorer")
        st.markdown("*Filter and explore your data with advanced controls*")
        create_advanced_data_panel(df)
    
    with tab6:
        show_data_sample(df)
        
        # Download option
        st.subheader("💾 Export Full Dataset")
        if st.button("Download Complete CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download All Data",
                data=csv,
                file_name="instagram_profiles_with_llm_analysis.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit • Data from BrightData + OpenAI GPT-5*")


if __name__ == "__main__":
    main()