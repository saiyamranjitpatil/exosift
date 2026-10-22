"""Reusable presentation pieces (hero, outcome banner, key insight, details strip)."""

from __future__ import annotations

from html import escape
from typing import Optional

import streamlit as st

NASA_MAST_URL = "https://mast.stsci.edu"
LIGHTKURVE_URL = "https://lightkurve.github.io/lightkurve/"


def hero(title: str, chips: list[tuple[str, bool]], status: Optional[str] = None, analyzed_at: Optional[str] = None) -> None:
    chips_html = "".join(f'<span class="exo-chip{" strong" if strong else ""}">{escape(text)}</span>' for text, strong in chips)
    side = ""
    if status:
        side += f'<div class="exo-status">{escape(status)}</div>'
    if analyzed_at:
        side += f'<div class="exo-time">Analyzed {escape(analyzed_at)}</div>'
    st.html(
        f'<div class="exo-hero"><div class="exo-hero-row"><div>'
        f'<h1 class="exo-hero-title">{escape(title)}</h1><div class="exo-chips">{chips_html}</div></div>'
        f'<div class="exo-hero-side">{side}</div></div></div>'
    )


def outcome_banner(tone: str, title: str, summary: str) -> None:
    st.html(
        f'<div class="exo-banner {escape(tone)}"><div class="exo-banner-icon"><div class="exo-planet"></div></div>'
        f'<div><p class="exo-banner-title">{escape(title)}</p><p class="exo-banner-text">{escape(summary)}</p></div></div>'
    )


def key_insight(text: str) -> None:
    st.html(f'<div class="exo-insight"><b>KEY INSIGHT</b>{escape(text)}</div>')


def detail_strip(items: list[tuple[str, str]]) -> None:
    body = "".join(f"<div><span>{escape(k)}</span><b>{escape(v)}</b></div>" for k, v in items)
    st.html(f'<div class="exo-detail">{body}</div>')


def source_popover() -> None:
    with st.popover("NASA/MAST data", icon=":material/satellite_alt:"):
        st.markdown(
            "**Where the data come from**\n\n"
            "Kepler long-cadence (about 29.4 minute) light curves are retrieved through the "
            "Mikulski Archive for Space Telescopes (MAST) with the Lightkurve package. "
            "ExoSift is an independent project and is not affiliated with or endorsed by NASA.\n\n"
            "Flux is the pipeline-corrected PDCSAP column when available, otherwise SAP. "
            "Times are BKJD (BJD - 2454833)."
        )
        c1, c2 = st.columns(2)
        c1.link_button("MAST archive", NASA_MAST_URL, icon=":material/open_in_new:", width="stretch")
        c2.link_button("Lightkurve docs", LIGHTKURVE_URL, icon=":material/open_in_new:", width="stretch")
