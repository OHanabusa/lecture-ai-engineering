# Automatic Social Listening (Japanese)

## Setup
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Run Streamlit app:
```
streamlit run frontend/ui_app.py
```
At launch, enter your Twitter bearer token and Mastodon credentials in the UI.


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
