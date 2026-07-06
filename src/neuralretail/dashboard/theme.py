"""Dashboard theme — single source of truth for the accent color.

The teal accent is enforced through every page in two ways:

1. ``.streamlit/config.toml`` paints every Streamlit chrome element
   (buttons, sliders, page-picker highlight, progress bars) with the
   same hex value, so the *app shell* matches the *charts*.
2. ``theme.ACCENT`` and ``theme.PALETTE`` are imported by every
   page-level module and passed as ``color_discrete_sequence=`` to
   plotly, so the *charts* match the app shell.

The rule is: no hard-coded hex codes anywhere in
``src/neuralretail/dashboard/`` outside this file. A grep-based test
in ``tests/test_dashboard.py`` enforces it.
"""

from __future__ import annotations

import colorsys

# Single accent — change here, see the whole dashboard update.
ACCENT: str = "#0E8388"

# Derived tints, computed once at import. Five steps for five personas.
ACCENT_DARK: str = "#095A5D"
ACCENT_LIGHT: str = "#5FB7B9"
ACCENT_PALE: str = "#B7DEDF"
ACCENT_MID: str = "#329FA2"

# Multi-category palette (kept small — the dashboard only ever needs to
# distinguish 5 personas or 3 ABC classes).
PALETTE: list[str] = [
    ACCENT_DARK,
    ACCENT,
    ACCENT_MID,
    ACCENT_LIGHT,
    ACCENT_PALE,
]

# Neutral greys for chrome and non-emphasis series (so a single page
# can have, e.g. a coloured forecast line + grey actuals, without
# breaking the one-accent rule).
NEUTRAL: str = "#94A3B8"
NEUTRAL_PALE: str = "#CBD5E1"

# Inline-CSS palette for KPI cards and section headers. The page text
# colour and card borders sit on the neutral side; only the accent
# border and accent delta colour break the otherwise-monochrome card.
TEXT_PRIMARY: str = "#0F172A"
TEXT_MUTED: str = "#64748B"
SURFACE: str = "#FFFFFF"
SURFACE_RAISED: str = "#F8FAFC"
DIVIDER: str = "#E2E8F0"


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(round(c * 255)):02X}" for c in rgb)


def tint(hex_color: str, *, lightness_shift: float) -> str:
    """Return ``hex_color`` with HLS lightness shifted by ``lightness_shift`` (-1..1).

    Useful for one-off hover states or shaded backgrounds in CSS snippets.
    """
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + lightness_shift))
    return _rgb_to_hex(colorsys.hls_to_rgb(h, l, s))
