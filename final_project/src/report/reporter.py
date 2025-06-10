from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import markdown2
try:
    from weasyprint import HTML  # type: ignore
    WEASYPRINT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    HTML = None  # type: ignore
    WEASYPRINT_ERROR = exc

logger = logging.getLogger(__name__)


def df_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def generate_pdf(df: pd.DataFrame, keywords: List[str]) -> str:
    """Create a PDF report from the metrics dataframe.

    Raises
    ------
    RuntimeError
        If WeasyPrint or its system dependencies are not available.
    """
    if HTML is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "WeasyPrint is not available: " f"{WEASYPRINT_ERROR}"
        )
    md_content = f"# Social Listening Report\n\nKeywords: {' '.join(keywords)}\n\n"
    md_content += df_to_markdown(df)
    html = markdown2.markdown(md_content)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    HTML(string=html).write_pdf(filename)
    logger.info("Saved report to %s", filename)
    return filename
