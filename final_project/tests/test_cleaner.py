import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.cleaner import clean_text, tokenize


def test_clean_text():
    text = 'RT @user: こんにちは https://example.com 😊'
    cleaned = clean_text(text)
    assert 'http' not in cleaned and 'RT' not in cleaned


def test_tokenize():
    tokens = tokenize('今日は天気です')
    assert isinstance(tokens, list)
    assert tokens
