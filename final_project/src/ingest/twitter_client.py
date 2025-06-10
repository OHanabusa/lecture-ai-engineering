from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

import tweepy

logger = logging.getLogger(__name__)


class TwitterClient:
    """Wrapper for Twitter recent search."""

    def __init__(self, bearer_token: str) -> None:
        if not bearer_token:
            raise ValueError("Twitter bearer token required")
        self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    def search_recent(self, keyword: str, limit: int = 100) -> List[str]:
        """Search recent tweets within last 24h."""
        query = f"{keyword} -is:retweet lang:ja"
        start_time = datetime.utcnow() - timedelta(days=1)
        tweets = self.client.search_recent_tweets(
            query=query,
            max_results=min(limit, 100),
            start_time=start_time.isoformat("T") + "Z",
            tweet_fields=["text"],
        )
        texts = [t.text for t in tweets.data] if tweets.data else []
        logger.debug("Twitter returned %d tweets for %s", len(texts), keyword)
        return texts
