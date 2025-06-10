# Automatic Social Listening (Japanese)

## Setup
1. Create `.env` with the following:
```
TW_BEARER_TOKEN=xxx
MASTODON_API_BASE_URL=https://mastodon.example.com
MASTODON_ACCESS_TOKEN=yyy
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run Streamlit app:
```
streamlit run frontend/ui_app.py
```

## Architecture
```mermaid
graph TD
    A[Twitter] -->|API| B(Ingest)
    C[Mastodon] -->|API| B
    B --> D(Preprocess)
    D --> E(Sentiment)
    E --> F(Metrics)
    F --> G[Streamlit UI]
    F --> H[PDF Report]
```

## License
MIT
