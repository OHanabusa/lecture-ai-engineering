from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

from mastodon import Mastodon

logger = logging.getLogger(__name__)


class MastodonClient:
    """Wrapper for Mastodon hashtag search."""

    def __init__(self, base_url: str, access_token: str) -> None:
        if not base_url or not access_token:
            raise ValueError("Mastodon API credentials required")
        self.client = Mastodon(api_base_url=base_url, access_token=access_token)

    def search_recent(self, keyword: str, limit: int = 100) -> List[str]:
        """Search recent toots for a hashtag."""
        start_time = datetime.utcnow() - timedelta(days=1)
        results = self.client.timeline_hashtag(keyword, limit=limit)
        texts = [s["content"] for s in results if s["created_at"] >= start_time]
        logger.debug("Mastodon returned %d toots for %s", len(texts), keyword)
        return texts
