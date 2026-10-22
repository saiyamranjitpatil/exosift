"""Scoped CSS for ExoSift's own elements.

Only classes prefixed `exo-` are styled (hero, outcome banner, key-insight callout, chips,
details strip). Streamlit's own widgets are themed through .streamlit/config.toml, not CSS.
The optional header photo (imgs/header_space.jpg) is layered in only if the file exists.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

IMGS_DIR = Path(__file__).resolve().parents[2] / "imgs"
HEADER_IMAGE = IMGS_DIR / "header_space.jpg"
LOGO = IMGS_DIR / "exosift_logo.svg"

_BASE_CSS = """
.exo-hero{position:relative;overflow:hidden;border:1px solid #1F2B45;border-radius:16px;padding:18px 26px;
  background-color:#080D1C;
  background-image:radial-gradient(ellipse at 88% -30%,rgba(56,189,248,.22),transparent 58%),
                   radial-gradient(ellipse at 100% 120%,rgba(129,140,248,.10),transparent 50%);}
.exo-hero::before{content:"";position:absolute;inset:0;opacity:.75;pointer-events:none;
  background-image:
    radial-gradient(1px 1px at 8% 24%,#fff,transparent),radial-gradient(1px 1px at 17% 70%,#9fd4ff,transparent),
    radial-gradient(1.4px 1.4px at 29% 38%,#fff,transparent),radial-gradient(1px 1px at 41% 80%,#cfe8ff,transparent),
    radial-gradient(1px 1px at 52% 18%,#fff,transparent),radial-gradient(1.4px 1.4px at 63% 62%,#9fd4ff,transparent),
    radial-gradient(1px 1px at 74% 30%,#fff,transparent),radial-gradient(1px 1px at 83% 76%,#cfe8ff,transparent),
    radial-gradient(1.4px 1.4px at 92% 22%,#fff,transparent),radial-gradient(1px 1px at 96% 58%,#9fd4ff,transparent);}
.exo-hero > *{position:relative}
.exo-hero-row{display:flex;flex-wrap:wrap;gap:14px;justify-content:space-between;align-items:flex-start}
.exo-hero-title{font-size:1.9rem;font-weight:700;line-height:1.15;margin:0 0 10px 0;color:#E6EDF7}
.exo-chips{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.exo-chip{border:1px solid #2A3A5C;border-radius:999px;padding:2px 12px;font-size:.85rem;color:#C9D6EE;background:rgba(56,189,248,.07)}
.exo-chip.strong{font-weight:700;color:#E6EDF7;background:rgba(56,189,248,.16);border-color:#2F6E96}
.exo-hero-side{text-align:right}
.exo-status{display:inline-block;border:1px solid #B07A12;color:#F5A524;border-radius:999px;padding:3px 14px;font-size:.85rem;font-weight:600;background:rgba(245,165,36,.08)}
.exo-time{font-size:.78rem;color:#8FA1C2;margin-top:6px}
.exo-banner{display:flex;gap:16px;align-items:center;border:1px solid var(--c,#2A6F9A);border-radius:14px;padding:14px 20px;
  background:linear-gradient(90deg,var(--bg,rgba(56,189,248,.08)),transparent 70%)}
.exo-banner.amber{--c:#8A6A1F;--bg:rgba(245,165,36,.09)}
.exo-banner.slate{--c:#334463;--bg:rgba(148,163,184,.07)}
.exo-banner-icon{flex:0 0 46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:rgba(56,189,248,.12)}
.exo-banner.amber .exo-banner-icon{background:rgba(245,165,36,.14)}
.exo-banner.slate .exo-banner-icon{background:rgba(148,163,184,.14)}
.exo-banner-title{font-size:1.3rem;font-weight:700;margin:0;color:#E6EDF7}
.exo-banner-text{color:#B6C4DD;font-size:.93rem;margin:3px 0 0 0}
.exo-insight{border-left:3px solid #F5A524;background:rgba(245,165,36,.06);padding:8px 12px;border-radius:0 8px 8px 0;font-size:.88rem;color:#D5DEEE;margin-top:6px}
.exo-insight b{display:block;color:#F5A524;letter-spacing:.08em;font-size:.68rem;margin-bottom:2px}
.exo-detail{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:10px 18px;align-items:start}
@media (max-width:900px){.exo-detail{grid-template-columns:repeat(2,minmax(110px,1fr))}}
.exo-detail span{display:block;font-size:.68rem;letter-spacing:.05em;text-transform:uppercase;color:#7F90B3}
.exo-detail b{font-size:.92rem;color:#E6EDF7;font-weight:600}
.exo-planet{position:relative;width:22px;height:22px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#7DD3FC,#0E4F7A)}
.exo-planet::after{content:"";position:absolute;left:-10px;top:8px;width:42px;height:8px;border:1.5px solid #38BDF8;border-radius:50%;transform:rotate(-18deg);opacity:.85}
.exo-banner.amber .exo-planet{background:radial-gradient(circle at 35% 30%,#FCD34D,#7A4E0E)}
.exo-banner.amber .exo-planet::after{border-color:#F5A524}
.exo-banner.slate .exo-planet{background:radial-gradient(circle at 35% 30%,#CBD5E1,#334155)}
.exo-banner.slate .exo-planet::after{border-color:#94A3B8}
"""

@st.cache_resource(show_spinner=False)
def _header_image_css() -> str:
    if not HEADER_IMAGE.exists():
        return ""
    data = base64.b64encode(HEADER_IMAGE.read_bytes()).decode("ascii")
    return (
        ".exo-hero{background-image:linear-gradient(90deg,rgba(7,11,22,.94) 0%,rgba(7,11,22,.70) 55%,rgba(7,11,22,.90) 100%),"
        f"url(data:image/jpeg;base64,{data});background-size:cover;background-position:center;}}"
    )


def inject() -> None:
    st.html(f"<style>{_BASE_CSS}{_header_image_css()}</style>")


