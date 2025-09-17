"""Utility helpers to normalize short influencer search queries.

Provides lightweight synonym expansion so abbreviated city names like
"SF" or "LA" get augmented with their canonical forms before we build
vector queries. The goal is to keep the logic deterministic and avoid
runtime API calls while improving recall for location-centric searches.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set

# Common US city/region abbreviations and nicknames
default_city_expansions = {
    "sf": ["san francisco", "bay area", "san fran"],
    "sfo": ["san francisco", "bay area"],
    "bay area": ["san francisco", "sf"],
    "la": ["los angeles", "l.a.", "hollywood"],
    "nyc": ["new york city", "new york", "ny"],
    "ny": ["new york", "nyc", "new york city"],
    "chi": ["chicago"],
    "phx": ["phoenix"],
    "atl": ["atlanta"],
    "mia": ["miami"],
    "dallas": ["dfw", "fort worth"],
    "dfw": ["dallas", "fort worth"],
    "dc": ["washington dc", "washington d.c.", "district of columbia"],
    "seattle": ["sea"],
    "sea": ["seattle"],
    "sd": ["san diego", "san diego county"],
    "san diego": ["sd"],
    "austin": ["atx"],
    "atx": ["austin"],
    "nash": ["nashville"],
    "bos": ["boston"],
    "philly": ["philadelphia"],
    "phl": ["philadelphia", "philly"],
    "hou": ["houston"],
    "houston": ["htx"],
    "htx": ["houston"],
    "den": ["denver"],
    "las vegas": ["vegas", "lv"],
    "vegas": ["las vegas", "lv"],
    "lv": ["las vegas", "vegas"],
}

# Two-letter US states to full names
us_state_expansions = {
    "ca": ["california"],
    "ny": ["new york"],
    "wa": ["washington"],
    "or": ["oregon"],
    "tx": ["texas"],
    "fl": ["florida"],
    "il": ["illinois"],
    "ga": ["georgia"],
    "co": ["colorado"],
    "az": ["arizona"],
    "nv": ["nevada"],
    "pa": ["pennsylvania"],
    "ma": ["massachusetts"],
    "dc": ["district of columbia", "washington dc"],
}


_SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Lowercase and strip punctuation while keeping spaces."""
    lowered = query.lower()
    cleaned = _SPECIAL_CHAR_PATTERN.sub(" ", lowered)
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def expand_query_terms(query: str,
                        extra_city_map: Iterable[tuple[str, Iterable[str]]] | None = None) -> List[str]:
    """Expand abbreviated tokens into richer search terms.

    Args:
        query: Raw user query.
        extra_city_map: Optional iterable of (term, synonyms) pairs to extend
            the default mapping at runtime.

    Returns:
        Unique list of expansion strings (excluding the normalized query itself).
    """
    base_tokens = normalize_query(query).split()
    expansions: Set[str] = set()

    # Build composite mapping with optional overrides
    custom_map = {k.lower(): [v.lower() for v in values]
                  for k, values in (extra_city_map or [])}

    def add_expansions(term: str, candidates: Iterable[str]):
        for candidate in candidates:
            candidate = candidate.strip().lower()
            if candidate and candidate != term:
                expansions.add(candidate)

    for token in base_tokens:
        if token in default_city_expansions:
            add_expansions(token, default_city_expansions[token])
        if token in us_state_expansions:
            add_expansions(token, us_state_expansions[token])
        if token in custom_map:
            add_expansions(token, custom_map[token])

    # If we expanded to multi-word terms, also include capitalized variants
    capitalized_variants = {phrase.title() for phrase in expansions if " " in phrase}
    expansions.update(capitalized_variants)

    return sorted(expansions)


def build_augmented_query(query: str,
                          expansions: Iterable[str]) -> str:
    """Append expansions to the normalized query for embedding search."""
    normalized = normalize_query(query)
    if not expansions:
        return normalized
    expansion_text = " ".join(expansions)
    return f"{normalized} {expansion_text}".strip()
