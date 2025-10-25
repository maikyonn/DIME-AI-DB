#!/usr/bin/env python3
"""Streamlit dashboard for exploring LanceDB embeddings and pipeline statistics."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import lancedb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="LanceDB Pipeline Explorer", layout="wide", page_icon="🧭")


@st.cache_data(show_spinner="Loading LanceDB tables…", ttl=300)
def list_tables(db_uri: str) -> List[str]:
    db = lancedb.connect(db_uri)
    return db.table_names()


@st.cache_data(show_spinner="Fetching metadata…", ttl=300)
def load_table_metadata(db_uri: str, table_name: str) -> Dict[str, object]:
    db = lancedb.connect(db_uri)
    table = db.open_table(table_name)
    indices = [str(idx) for idx in table.list_indices()]
    return {
        "row_count": table.count_rows(),
        "schema": str(table.schema),
        "indices": indices,
        "stats": table.stats(),
    }


@st.cache_data(show_spinner="Loading sample rows…", ttl=180)
def load_sample_frame(db_uri: str, table_name: str, sample_size: int, seed: int) -> pd.DataFrame:
    db = lancedb.connect(db_uri)
    table = db.open_table(table_name)
    total_rows = table.count_rows()
    if total_rows == 0:
        return pd.DataFrame()
    limit = min(sample_size, total_rows)
    # Pull core columns once then down-sample so filters reflect the whole dataset.
    arrow_table = table.to_arrow()
    df = arrow_table.to_pandas()
    if limit < len(df):
        rng = np.random.default_rng(seed)
        df = df.sample(n=limit, random_state=rng.integers(0, 2**32 - 1)).reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Loading pipeline metadata…", ttl=300)
def load_pipeline_metadata(output_dir: str) -> Dict[str, Optional[dict]]:
    """Load metadata from pipeline stages."""
    output_path = Path(output_dir)
    metadata = {}

    # Extract stage metadata
    extract_meta = output_path / "facets.meta.json"
    if extract_meta.exists():
        metadata["extract"] = json.loads(extract_meta.read_text())

    # TF-IDF stage marker
    tfidf_marker = output_path / ".tfidf.done"
    if tfidf_marker.exists():
        try:
            metadata["tfidf"] = json.loads(tfidf_marker.read_text())
        except:
            metadata["tfidf"] = {"completed": True}

    # Embedding stage marker
    embed_marker = output_path / ".embed.done"
    if embed_marker.exists():
        try:
            metadata["embed"] = json.loads(embed_marker.read_text())
        except:
            metadata["embed"] = {"completed": True}

    # Combine stage marker
    combine_marker = output_path / ".combine.done"
    if combine_marker.exists():
        try:
            metadata["combine"] = json.loads(combine_marker.read_text())
        except:
            metadata["combine"] = {"completed": True}

    # Load stage marker
    load_marker = output_path / ".load.done"
    if load_marker.exists():
        try:
            metadata["load"] = json.loads(load_marker.read_text())
        except:
            metadata["load"] = {"completed": True}

    return metadata


def check_embedding_columns(df: pd.DataFrame) -> Dict[str, any]:
    """Check for embedding/vector columns and analyze them."""
    results = {
        "has_embeddings": False,
        "embedding_columns": [],
        "embedding_dimensions": {},
        "null_counts": {},
        "sample_embeddings": {},
    }

    # Look for vector/embedding columns
    vector_cols = [col for col in df.columns if 'vector' in col.lower() or 'embedding' in col.lower()]

    for col in vector_cols:
        results["embedding_columns"].append(col)
        results["has_embeddings"] = True

        # Check dimensionality
        non_null = df[col].dropna()
        if len(non_null) > 0:
            first_val = non_null.iloc[0]
            if isinstance(first_val, (list, np.ndarray)):
                results["embedding_dimensions"][col] = len(first_val)
            elif hasattr(first_val, 'shape'):
                results["embedding_dimensions"][col] = first_val.shape[0]

        # Count nulls
        results["null_counts"][col] = df[col].isna().sum()

        # Sample embeddings
        if len(non_null) > 0:
            sample = non_null.iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                results["sample_embeddings"][col] = list(sample[:5]) if len(sample) > 5 else list(sample)

    return results


@st.cache_data(show_spinner="Running full-text search…", ttl=120)
def run_text_search(db_uri: str, table_name: str, query: str, top_k: int) -> pd.DataFrame:
    db = lancedb.connect(db_uri)
    table = db.open_table(table_name)
    results = table.search(query).limit(top_k).to_pandas()
    return results


def render_pipeline_status(pipeline_meta: Dict[str, Optional[dict]]) -> None:
    """Render pipeline stage completion status."""
    st.subheader("🔄 Pipeline Status")

    if not pipeline_meta:
        st.info("No pipeline metadata found. Run the pipeline to see statistics.")
        return

    stages = ["extract", "tfidf", "embed", "combine", "load"]
    stage_names = {
        "extract": "1️⃣ Extract Facets",
        "tfidf": "2️⃣ TF-IDF",
        "embed": "3️⃣ Embeddings",
        "combine": "4️⃣ Combine",
        "load": "5️⃣ Load to DB"
    }

    cols = st.columns(len(stages))
    for idx, stage in enumerate(stages):
        with cols[idx]:
            if stage in pipeline_meta:
                st.success(stage_names[stage])
                meta = pipeline_meta[stage]
                if isinstance(meta, dict):
                    if "record_count" in meta:
                        st.caption(f"{meta['record_count']:,} records")
                    elif "total_rows" in meta:
                        st.caption(f"{meta['total_rows']:,} rows")
                    elif "loaded_rows" in meta:
                        st.caption(f"{meta['loaded_rows']:,} loaded")
            else:
                st.warning(stage_names[stage])
                st.caption("Not complete")

    # Detailed pipeline info
    with st.expander("📊 Pipeline Details"):
        if "extract" in pipeline_meta:
            st.markdown("**Extract Stage**")
            st.json(pipeline_meta["extract"])

        if "load" in pipeline_meta:
            st.markdown("**Load Stage**")
            st.json(pipeline_meta["load"])


def render_embedding_verification(df: pd.DataFrame) -> None:
    """Verify embeddings are present and valid."""
    st.subheader("🔍 Embedding Verification")

    embed_info = check_embedding_columns(df)

    if not embed_info["has_embeddings"]:
        st.error("❌ No embedding columns found in the data!")
        st.info("Expected columns with 'vector' or 'embedding' in the name.")
        return

    st.success(f"✅ Found {len(embed_info['embedding_columns'])} embedding column(s)")

    for col in embed_info["embedding_columns"]:
        with st.expander(f"📈 {col}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Dimension", embed_info["embedding_dimensions"].get(col, "Unknown"))

            with col2:
                null_count = embed_info["null_counts"].get(col, 0)
                null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
                st.metric("Null Count", f"{null_count:,} ({null_pct:.1f}%)")

            with col3:
                coverage = ((len(df) - null_count) / len(df) * 100) if len(df) > 0 else 0
                st.metric("Coverage", f"{coverage:.1f}%")

            # Sample embeddings
            if col in embed_info["sample_embeddings"]:
                st.caption("Sample values (first 5 dimensions):")
                st.code(str(embed_info["sample_embeddings"][col]))


def render_data_quality_checks(df: pd.DataFrame) -> None:
    """Run data quality checks on the dataset."""
    st.subheader("✅ Data Quality Checks")

    checks = []

    # Check for required columns
    required_cols = ["vector_id", "text", "content_type", "platform"]
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
            checks.append({
                "Check": f"{col} present",
                "Status": "✅ Pass" if null_count < len(df) * 0.05 else "⚠️ Warning",
                "Details": f"{null_pct:.1f}% null"
            })
        else:
            checks.append({
                "Check": f"{col} present",
                "Status": "❌ Fail",
                "Details": "Column missing"
            })

    # Check for duplicate vector_ids
    if "vector_id" in df.columns:
        dup_count = df["vector_id"].duplicated().sum()
        checks.append({
            "Check": "Unique vector_ids",
            "Status": "✅ Pass" if dup_count == 0 else "⚠️ Warning",
            "Details": f"{dup_count} duplicates"
        })

    # Check text field populated
    if "text" in df.columns:
        empty_text = df["text"].fillna("").str.strip().eq("").sum()
        empty_pct = (empty_text / len(df) * 100) if len(df) > 0 else 0
        checks.append({
            "Check": "Text populated",
            "Status": "✅ Pass" if empty_pct < 5 else "⚠️ Warning",
            "Details": f"{empty_pct:.1f}% empty"
        })

    # Platform distribution
    if "platform" in df.columns:
        platform_counts = df["platform"].value_counts()
        checks.append({
            "Check": "Platform distribution",
            "Status": "✅ Pass",
            "Details": ", ".join([f"{k}: {v}" for k, v in platform_counts.items()])
        })

    checks_df = pd.DataFrame(checks)
    st.dataframe(checks_df, use_container_width=True, hide_index=True)


def render_metrics(metadata: Dict[str, object], df: pd.DataFrame) -> None:
    total_rows = metadata.get("row_count", 0)
    unique_platforms = df["platform"].dropna().nunique() if "platform" in df else 0
    unique_content_types = df["content_type"].dropna().nunique() if "content_type" in df else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Table Rows", f"{int(total_rows):,}")
    col2.metric("Sample Rows", f"{len(df):,}")
    col3.metric("Platforms", str(unique_platforms))
    col4.metric("Content Types", str(unique_content_types))

    if metadata.get("indices"):
        st.caption("🔍 Indexes: " + ", ".join(metadata["indices"]))


def render_distribution_plots(df: pd.DataFrame) -> None:
    if df.empty:
        return

    numeric_cols = ["followers", "following", "likes_total", "posts_count", "engagement_rate"]
    available_numeric = [col for col in numeric_cols if col in df.columns and df[col].notna().any()]
    if available_numeric:
        col = available_numeric[0]
        fig = px.histogram(df.dropna(subset=[col]), x=col, nbins=40, title=f"Distribution of {col.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)

    if "platform" in df.columns:
        platform_counts = df["platform"].value_counts().reset_index()
        platform_counts.columns = ["platform", "count"]
        fig = px.bar(platform_counts, x="platform", y="count", title="Records per Platform (sample)")
        st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame) -> None:
    st.subheader("Sample Rows")
    if df.empty:
        st.info("No rows to display.")
        return
    column_options = df.columns.tolist()
    default_cols = column_options
    selected_cols = st.multiselect(
        "Columns to display",
        options=column_options,
        default=default_cols,
        key="data_table_columns",
    )
    if not selected_cols:
        st.warning("Select at least one column to display.")
        return
    st.dataframe(df[selected_cols], use_container_width=True)


def render_search_section(db_uri: str, table_name: str) -> None:
    st.subheader("Full-Text Search")
    query = st.text_input("Enter search query", placeholder="e.g. skincare influencer in LA")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        top_k = st.slider("Results", min_value=5, max_value=50, value=10, step=5)
    if not query:
        st.caption("Use the search box above to run a BM25 full-text query against the LanceDB table.")
        return

    results = run_text_search(db_uri, table_name, query, top_k)
    if results.empty:
        st.warning("No matches found for that query.")
        return

    display_cols = [
        "_score",
        "vector_id",
        "platform",
        "content_type",
        "username",
        "display_name",
        "text",
    ]
    available_cols = [col for col in display_cols if col in results.columns]
    st.dataframe(results[available_cols], use_container_width=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    with st.sidebar:
        st.markdown("### Filters")
        content_types = sorted(df["content_type"].dropna().unique().tolist()) if "content_type" in df else []
        selected_content = st.multiselect(
            "Content Types",
            options=content_types,
            default=content_types,
            key="filter_content_types",
        ) if content_types else []

        platforms = sorted(df["platform"].dropna().unique().tolist()) if "platform" in df else []
        selected_platforms = st.multiselect(
            "Platforms",
            options=platforms,
            default=platforms,
            key="filter_platforms",
        ) if platforms else []

        follower_min, follower_max = 0, 0
        if "followers" in df and df["followers"].notna().any():
            follower_min = int(math.floor(df["followers"].min()))
            follower_max = int(math.ceil(df["followers"].max()))
            follower_range = st.slider(
                "Follower Range",
                min_value=follower_min,
                max_value=follower_max,
                value=(follower_min, follower_max),
                key="filter_followers",
            )
        else:
            follower_range = None

        verified_only = st.checkbox("Only verified profiles", value=False, key="filter_verified") if "is_verified" in df else False

    filtered = df.copy()
    if content_types:
        filtered = filtered[filtered["content_type"].isin(selected_content)]
    if platforms:
        filtered = filtered[filtered["platform"].isin(selected_platforms)]
    if follower_range is not None:
        low, high = follower_range
        filtered = filtered[filtered["followers"].fillna(0).between(low, high)]
    if verified_only:
        filtered = filtered[filtered["is_verified"] == True]  # noqa: E712

    return filtered


def main() -> None:
    st.title("🧭 LanceDB Pipeline Explorer")
    st.write("Monitor pipeline execution, verify embeddings, and explore your influencer dataset.")

    # Sidebar configuration
    refresh = False
    with st.sidebar:
        st.markdown("### 🔌 Connection")
        db_uri = st.text_input("Database URI", value="data/lancedb")

        try:
            table_options = list_tables(db_uri)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to list tables: {exc}")
            return
        if not table_options:
            st.warning("No tables found at that URI.")
            return

        table_name = st.selectbox("Table", options=table_options)
        sample_size = st.slider("Sample Size", min_value=100, max_value=5000, value=1000, step=100)

        st.markdown("---")
        st.markdown("### 📁 Pipeline Output")
        pipeline_dir = st.text_input("Pipeline output directory", value="pipeline/output")

        st.markdown("---")
        refresh = st.button("🔄 Refresh Data", help="Reload data from database and pipeline")

    if "sample_seed" not in st.session_state:
        st.session_state.sample_seed = 0
    if refresh:
        st.session_state.sample_seed += 1
        st.cache_data.clear()

    # Load data
    metadata = load_table_metadata(db_uri, table_name)
    sample_df = load_sample_frame(db_uri, table_name, sample_size, st.session_state.sample_seed)
    pipeline_meta = load_pipeline_metadata(pipeline_dir)

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Embeddings",
        "✅ Quality Checks",
        "🔎 Search",
        "📋 Raw Data"
    ])

    with tab1:
        # Pipeline status
        render_pipeline_status(pipeline_meta)

        st.markdown("---")

        # Key metrics
        st.subheader("📈 Dataset Metrics")
        filtered_df = apply_filters(sample_df)
        render_metrics(metadata, filtered_df)

        st.markdown("---")

        # Distribution plots
        st.subheader("📊 Data Distributions")
        render_distribution_plots(filtered_df)

    with tab2:
        # Embedding verification
        render_embedding_verification(sample_df)

        st.markdown("---")

        # Embedding statistics
        if "vector" in " ".join(sample_df.columns).lower():
            st.subheader("📊 Embedding Statistics")

            # Check for vector similarity if we have embeddings
            embed_cols = [col for col in sample_df.columns if 'vector' in col.lower() or 'embedding' in col.lower()]
            if embed_cols:
                st.info(f"💡 **Tip**: Use the Search tab to test semantic similarity using the {embed_cols[0]} column")

    with tab3:
        # Data quality checks
        render_data_quality_checks(sample_df)

        st.markdown("---")

        # Additional statistics
        st.subheader("📊 Column Statistics")
        if not sample_df.empty:
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_stat_col = st.selectbox("Select column for statistics", numeric_cols)
                if selected_stat_col:
                    col_stats = sample_df[selected_stat_col].describe()
                    st.dataframe(col_stats.to_frame().T, use_container_width=True)

    with tab4:
        # Search functionality
        render_search_section(db_uri, table_name)

    with tab5:
        # Raw data table
        filtered_df = apply_filters(sample_df)
        render_data_table(filtered_df)

        with st.expander("🔧 Table Schema"):
            st.code(metadata.get("schema", ""))

        with st.expander("📊 Table Stats"):
            st.json(metadata.get("stats", {}))


if __name__ == "__main__":
    main()
