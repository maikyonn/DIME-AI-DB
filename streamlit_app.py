#!/usr/bin/env python3
"""Streamlit dashboard for exploring LanceDB influencer_facets."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

import lancedb
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="LanceDB Explorer", layout="wide", page_icon="🧭")


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


@st.cache_data(show_spinner="Running full-text search…", ttl=120)
def run_text_search(db_uri: str, table_name: str, query: str, top_k: int) -> pd.DataFrame:
    db = lancedb.connect(db_uri)
    table = db.open_table(table_name)
    results = table.search(query).limit(top_k).to_pandas()
    return results


def render_metrics(metadata: Dict[str, object], df: pd.DataFrame) -> None:
    total_rows = metadata.get("row_count", 0)
    unique_platforms = df["platform"].dropna().nunique() if "platform" in df else 0
    unique_content_types = df["content_type"].dropna().nunique() if "content_type" in df else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Table Rows", f"{int(total_rows):,}")
    col2.metric("Sample Rows", f"{len(df):,}")
    col3.metric("Platforms (sample)", str(unique_platforms))

    if metadata.get("indices"):
        st.caption("Indexes: " + ", ".join(metadata["indices"]))


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
        ) if content_types else []

        platforms = sorted(df["platform"].dropna().unique().tolist()) if "platform" in df else []
        selected_platforms = st.multiselect(
            "Platforms",
            options=platforms,
            default=platforms,
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
            )
        else:
            follower_range = None

        verified_only = st.checkbox("Only verified profiles", value=False) if "is_verified" in df else False

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
    st.title("LanceDB Profile Explorer")
    st.write("Inspect LanceDB embeddings, metrics, and run quick text searches over your influencer dataset.")

    refresh = False
    with st.sidebar:
        st.markdown("### Connection")
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
        refresh = st.button("Refresh sample", help="Resample rows for the preview panels")

    if "sample_seed" not in st.session_state:
        st.session_state.sample_seed = 0
    if refresh:
        st.session_state.sample_seed += 1

    metadata = load_table_metadata(db_uri, table_name)
    sample_df = load_sample_frame(db_uri, table_name, sample_size, st.session_state.sample_seed)
    filtered_df = apply_filters(sample_df)

    render_metrics(metadata, filtered_df)
    render_distribution_plots(filtered_df)
    render_data_table(filtered_df)
    render_search_section(db_uri, table_name)

    with st.expander("Table schema"):
        st.code(metadata.get("schema", ""))
    with st.expander("Table stats"):
        st.json(metadata.get("stats", {}))


if __name__ == "__main__":
    main()
