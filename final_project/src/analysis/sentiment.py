from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_pipeline():
    model_name = "rinna/japanese-roberta-base-sentiment"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tok)


def analyze(texts: List[str]) -> List[dict[str, float]]:
    """Return sentiment probabilities per text."""
    pipe = _get_pipeline()
    outputs = pipe(texts, truncation=True)
    results = []
    for out in outputs:
        scores = out["score"] if isinstance(out["score"], list) else [out["score"]]
        label = out["label"].lower()
        probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        if label in probs:
            probs[label] = float(scores[0])
        else:
            probs["neutral"] = float(scores[0])
        results.append(probs)
    logger.debug("Analyzed %d texts", len(results))
    return results
