# imgs/ — optional visual assets

The dashboard works fully without anything in this folder except `exosift_logo.svg`,
which is already here (a small inline SVG logo + wordmark, not a download).

Space photography was evaluated and judged **not necessary**: the header already gets its
"observatory" feel from a CSS starfield + radial gradient (`app/ui/style.py`), and the
charts are the actual content. The one optional upgrade below is a subtle photographic
header background — purely decorative, skippable.

## Optional: header background photo

| | |
|---|---|
| **Filename** | `imgs/header_space.jpg` |
| **Placement** | Behind the hero banner at the top of the Analysis/Methodology/About pages, under a dark gradient overlay (70-94% opacity) so text stays legible — wired up automatically in `app/ui/style.py:_header_image_css()` if the file exists |
| **Dimensions** | 2400 x 360 px (wide, short — it's a banner strip, not a hero photo) |
| **Format / size budget** | JPG, quality ~80, **under 250 KB** |
| **Content** | A dark field-of-stars or nebula image with most of its detail in the right third (the left/center is covered by text) |

**Where to get one (NASA public-domain imagery, free to reuse):**
- NASA Image and Video Library — https://images.nasa.gov — search "Kepler" or "star field"
  or "nebula". Filter results to ones credited to NASA/JPL-Caltech; most are public domain
  but always check the individual item's credit line before using it elsewhere.
- NASA Kepler mission gallery — https://www.nasa.gov/mission/kepler/ (image galleries
  linked from the mission page).

**Do not** use an image that depicts a real spacecraft or mission patch in a way that
could be read as an official NASA product — ExoSift is an independent project and the
About page already says so; keep the image generic (starfield/nebula), not a specific
mission insignia.

## If you skip this

Nothing breaks. `app/ui/style.py` checks whether `imgs/header_space.jpg` exists before
referencing it; the CSS starfield renders either way.
