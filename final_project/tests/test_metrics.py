import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from src.analysis.metrics import calculate_metrics


def test_calculate_metrics():
    data = {
        'kw1': [{'positive': 0.7, 'negative': 0.2, 'neutral': 0.1}],
        'kw2': [{'positive': 0.2, 'negative': 0.6, 'neutral': 0.2},
                {'positive': 0.1, 'negative': 0.7, 'neutral': 0.2}],
    }
    df = calculate_metrics(data)
    assert 'keyword' in df.columns
    assert len(df) == 2
