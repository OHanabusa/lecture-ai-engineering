from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import List

from mastodon import Mastodon

logger = logging.getLogger(__name__)


class MastodonClient:
    """Wrapper for Mastodon hashtag search."""

    def __init__(self) -> None:
        base_url = os.getenv("MASTODON_API_BASE_URL")
        token = os.getenv("MASTODON_ACCESS_TOKEN")
        if not base_url or not token:
            raise ValueError("Mastodon API credentials not set")
        self.client = Mastodon(api_base_url=base_url, access_token=token)

    def search_recent(self, keyword: str, limit: int = 100) -> List[str]:
        """Search recent toots for a hashtag."""
        start_time = datetime.utcnow() - timedelta(days=1)
        results = self.client.timeline_hashtag(keyword, limit=limit)
        texts = [s["content"] for s in results if s["created_at"] >= start_time]
        logger.debug("Mastodon returned %d toots for %s", len(texts), keyword)
        return texts
