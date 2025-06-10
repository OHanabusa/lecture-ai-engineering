from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List

import pandas as pd

from src.ingest.twitter_client import TwitterClient
from src.preprocessing.cleaner import preprocess_posts
from .sentiment import analyze

logger = logging.getLogger(__name__)


def aggregate_sentiments(probs: List[dict[str, float]]) -> dict[str, float]:
    if not probs:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    df = pd.DataFrame(probs)
    return df.mean().to_dict()


def calculate_metrics(keyword_posts: Dict[str, List[dict[str, float]]]) -> pd.DataFrame:
    rows = []
    total_posts = sum(len(v) for v in keyword_posts.values())
    for kw, posts in keyword_posts.items():
        agg = aggregate_sentiments(posts)
        count = len(posts)
        share = count / total_posts if total_posts else 0
        row = {
            "keyword": kw,
            "count": count,
            "positive%": agg.get("positive", 0.0) * 100,
            "negative%": agg.get("negative", 0.0) * 100,
            "neutral%": agg.get("neutral", 0.0) * 100,
            "share": share * 100,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.debug("Calculated metrics for %d keywords", len(rows))
    return df


def run_analysis(
    keywords: List[str],
    tw_token: str,
    tw_api_key: str = "",
    tw_api_secret: str = "",
    tw_access_token: str = "",
    tw_access_secret: str = "",
    hf_token: str = "",
) -> pd.DataFrame:
    """Collect posts and compute metrics."""
    tw_client = TwitterClient(
        bearer_token=tw_token,
        api_key=tw_api_key or None,
        api_secret=tw_api_secret or None,
        access_token=tw_access_token or None,
        access_secret=tw_access_secret or None,
    )


    keyword_probs: Dict[str, List[dict[str, float]]] = defaultdict(list)

    for kw in keywords:
        tw_posts = tw_client.search_recent(kw)
        posts = preprocess_posts(tw_posts)
        probs = analyze(posts, token=hf_token or None)
        keyword_probs[kw].extend(probs)
    df = calculate_metrics(keyword_probs)
    df.insert(0, "timestamp", datetime.now())
    return df
