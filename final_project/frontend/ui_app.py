from __future__ import annotations

import logging
from typing import List

import pandas as pd
import streamlit as st

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis.metrics import run_analysis
from src.report.reporter import generate_pdf

logging.basicConfig(level=logging.INFO)


st.title("Automatic Social Listening")
keywords_input = st.text_input("Keywords (space separated)")
tw_token = st.text_input("Twitter Bearer Token", type="password")
ma_base_url = st.text_input("Mastodon Base URL")
ma_token = st.text_input("Mastodon Access Token", type="password")
if st.button("Run"):
    keywords: List[str] = [k for k in keywords_input.split() if k]
    if keywords:
        if not (tw_token and ma_base_url and ma_token):
            st.error("All API credentials are required")
        else:
            with st.spinner("Collecting data..."):
                df: pd.DataFrame = run_analysis(keywords, tw_token, ma_base_url, ma_token)
            st.dataframe(df)
            chart_data = df.set_index("keyword")[["positive%", "negative%", "neutral%"]]
            st.bar_chart(chart_data)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "metrics.csv")
            if st.button("Download PDF Report"):
                pdf_path = generate_pdf(df, keywords)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f, pdf_path, mime="application/pdf")
