"""
ExoSift: interactive Kepler target analysis.

    streamlit run app/streamlit_app.py

This file is only the entry point (page config, logo, navigation). The analysis lives in
src/exosift_pipeline.py; the pages are in app/app_pages/ and the UI pieces in app/ui/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="ExoSift | Kepler transit analysis",
    page_icon=":material/travel_explore:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import style  # noqa: E402

if style.LOGO.exists():
    st.logo(str(style.LOGO), size="large")

page = st.navigation(
    [
        st.Page("app_pages/analysis.py", title="Analysis", icon=":material/monitoring:", default=True),
        st.Page("app_pages/methodology.py", title="Methodology", icon=":material/menu_book:"),
        st.Page("app_pages/about.py", title="About", icon=":material/info:"),
    ],
    position="top",
)
page.run()
