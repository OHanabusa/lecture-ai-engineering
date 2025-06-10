from __future__ import annotations

import logging
import re
from typing import Iterable, List

import emoji
from sudachipy import dictionary, tokenizer

logger = logging.getLogger(__name__)
_tokenizer = dictionary.Dictionary().create()
_mode = tokenizer.Tokenizer.SplitMode.B

URL_PATTERN = re.compile(r"https?://\S+")
RT_PATTERN = re.compile(r"RT @\w+|via @\w+")


def clean_text(text: str) -> str:
    """Remove URLs, emojis and RT noise."""
    text = URL_PATTERN.sub("", text)
    text = RT_PATTERN.sub("", text)
    text = emoji.replace_emoji(text, "")
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Return content word tokens."""
    morphs = _tokenizer.tokenize(text, _mode)
    content_pos = {"名詞", "動詞", "形容詞", "副詞"}
    tokens = [m.surface() for m in morphs if m.part_of_speech()[0] in content_pos]
    logger.debug("Tokenized into %d tokens", len(tokens))
    return tokens


def preprocess_posts(posts: Iterable[str]) -> List[str]:
    """Clean and tokenize list of posts."""
    results: List[str] = []
    for p in posts:
        cleaned = clean_text(p)
        tokens = tokenize(cleaned)
        if tokens:
            results.append(" ".join(tokens))
    logger.debug("Preprocessed %d posts", len(results))
    return results
