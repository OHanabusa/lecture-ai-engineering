from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

import tweepy

AuthClient = tweepy.Client
APIClient = tweepy.API

logger = logging.getLogger(__name__)


class TwitterClient:
    """Wrapper for Twitter search using either v2 or v1.1 APIs."""

    def __init__(
        self,
        bearer_token: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        access_token: str | None = None,
        access_secret: str | None = None,
    ) -> None:
        if bearer_token:
            self.v2 = True
            self.client = tweepy.Client(
                bearer_token=bearer_token, wait_on_rate_limit=True
            )
        elif api_key and api_secret and access_token and access_secret:
            self.v2 = False
            auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
            self.client = tweepy.API(auth, wait_on_rate_limit=True)
        else:
            raise ValueError(
                "Twitter credentials required: either bearer token or API key/secret and access token/secret"
            )

    def search_recent(self, keyword: str, limit: int = 100) -> List[str]:
        """Search recent tweets within last 24h."""
        start_time = datetime.utcnow() - timedelta(days=1)
        if self.v2:
            query = f"{keyword} -is:retweet lang:ja"
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),
                start_time=start_time.isoformat("T") + "Z",
                tweet_fields=["text"],
            )
            texts = [t.text for t in tweets.data] if tweets.data else []
        else:
            query = f"{keyword} -filter:retweets lang:ja"
            tweets = self.client.search_tweets(
                q=query,
                count=min(limit, 100),
                tweet_mode="extended",
            )
            texts = [
                t.full_text
                for t in tweets
                if t.created_at and t.created_at >= start_time
            ]
        logger.debug("Twitter returned %d tweets for %s", len(texts), keyword)
        return texts
