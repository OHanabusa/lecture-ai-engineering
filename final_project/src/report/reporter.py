from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import markdown2
from weasyprint import HTML

logger = logging.getLogger(__name__)


def df_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def generate_pdf(df: pd.DataFrame, keywords: List[str]) -> str:
    md_content = f"# Social Listening Report\n\nKeywords: {' '.join(keywords)}\n\n"
    md_content += df_to_markdown(df)
    html = markdown2.markdown(md_content)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    HTML(string=html).write_pdf(filename)
    logger.info("Saved report to %s", filename)
    return filename
