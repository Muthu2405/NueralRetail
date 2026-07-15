"""Smoke tests for the Streamlit dashboard.

Two goals:

1. Every page script can be run via ``streamlit.testing.v1.AppTest``
   without raising. This catches import errors, missing keys, and
   the most common "first render blew up" failure mode.
2. No hex color literal lives outside ``theme.py`` — keeps the
   one-accent-color rule honest as the dashboard grows.

We do **not** assert on the visual output; Charting is
declarative and the AppTest API doesn't render plotly figures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO_ROOT / "src" / "neuralretail" / "dashboard"
PAGES_DIR = DASHBOARD_DIR / "pages"

# Filenames are stable, but we discover them in case the order changes.
PAGE_FILES = sorted(p for p in PAGES_DIR.glob("*.py") if p.is_file())

# Hex pattern: # followed by exactly 6 hex digits, with word boundaries
# so we don't catch things like '#1234567' or 'rgb(1,2,3)'.
HEX_PATTERN = re.compile(r"(?<![\w/])#[0-9A-Fa-f]{6}(?!\w)")


# ---------------------------------------------------------------------------
# Page-level smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=lambda p: p.name)
def test_page_renders_without_exception(page_path: Path) -> None:
    """Each page boots, runs, and produces at least one title with no exception."""
    at = AppTest.from_file(str(page_path)).run(timeout=60)
    assert not at.exception, f"{page_path.name} raised: {at.exception}"
    # Every page should set a title; some pages set both st.title and st.set_page_config title.
    assert len(at.title) >= 1, f"{page_path.name} has no rendered title"


def test_app_entrypoint_renders() -> None:
    """The landing app.py also boots without exception."""
    at = AppTest.from_file(str(DASHBOARD_DIR / "app.py")).run(timeout=60)
    assert not at.exception, f"app.py raised: {at.exception}"
    # Sidebar widget: at least one multiselect (country filter).
    assert at.multiselect, "app.py did not render the country multiselect"


# ---------------------------------------------------------------------------
# Hex-code discipline
# ---------------------------------------------------------------------------


# Files that may legitimately contain hex colors.
ALLOWED_HEX_FILES = {DASHBOARD_DIR / "theme.py"}

# Files that may legitimately contain hex colors for non-theme reasons
# (e.g. inline HTML/CSS for KPI cards uses the theme constant, so should
# import the constant — but the test is on the LITERAL appearing in code).
DISALLOWED_RAW_HEX = (DASHBOARD_DIR,)  # every file under dashboard/


@pytest.mark.parametrize(
    "py_file",
    sorted(p for p in DASHBOARD_DIR.rglob("*.py")),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_no_raw_hex_outside_theme(py_file: Path) -> None:
    """No hard-coded ``#RRGGBB`` literal in dashboard code outside theme.py.

    Pages should import the constant from ``neuralretail.dashboard.theme``
    instead of inlining colors. This guards against drift if someone
    tweaks the accent on one page without updating the others.
    """
    if py_file in ALLOWED_HEX_FILES:
        return  # theme.py is the one allowed home
    text = py_file.read_text(encoding="utf-8")
    # Allow comments mentioning the pattern (with #) and allow f-string
    # interpolations like f"color:{ACCENT}". We catch *literal* hex.
    matches = HEX_PATTERN.findall(text)
    assert not matches, (
        f"{py_file.relative_to(REPO_ROOT)} contains raw hex color(s) {matches}. "
        f"Import from neuralretail.dashboard.theme instead."
    )
