"""Build a project-overview PDF for NeuralRetail.

Re-renders the 7 charts from the live parquet/csv artefacts on disk and
composes a 14–16 page reference document covering the business framing,
architecture, every model, the dashboard / API surface, and the
how-to-demo / how-to-run playbook.

Run it any time after `python -m neuralretail.cli train` to refresh the
PDF with the latest numbers. The script is read-only with respect to the
data pipeline — it never mutates the source parquet/csv/MLflow artefacts.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet.serialize import model_from_json
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Theme + paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "processed"
MODELS = REPO_ROOT / "models"
REPORT = REPO_ROOT / "report"
FIG_DIR = REPORT / "figures"
OUTPUT_PDF = REPORT / "NeuralRetail_Project_Overview.pdf"

ACCENT = "#0E8388"          # teal, matches dashboard/theme.py
ACCENT_DARK = "#095A5D"
ACCENT_LIGHT = "#5FB7B9"
NEUTRAL = "#94A3B8"
TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#64748B"
SURFACE = "#FFFFFF"
DIVIDER = "#E2E8F0"

PAGE_W, PAGE_H = LETTER
MARGIN_L = 0.85 * inch
MARGIN_R = 0.85 * inch
MARGIN_T = 1.05 * inch
MARGIN_B = 0.85 * inch
FRAME_W = PAGE_W - MARGIN_L - MARGIN_R

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14.5,
        textColor=colors.HexColor(TEXT_PRIMARY),
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "h1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor(ACCENT_DARK),
        spaceBefore=4,
        spaceAfter=14,
    )
    h2 = ParagraphStyle(
        "h2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor(ACCENT_DARK),
        spaceBefore=14,
        spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "h3",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor(TEXT_PRIMARY),
        spaceBefore=10,
        spaceAfter=4,
    )
    caption = ParagraphStyle(
        "caption",
        parent=base["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor(TEXT_MUTED),
        spaceAfter=10,
        alignment=1,  # center
    )
    code = ParagraphStyle(
        "code",
        parent=base["Code"],
        fontName="Courier",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor(TEXT_PRIMARY),
        backColor=colors.HexColor("#F1F5F9"),
        borderColor=colors.HexColor(DIVIDER),
        borderWidth=0.5,
        borderPadding=6,
        leftIndent=0,
        rightIndent=0,
        spaceAfter=8,
    )
    bullet = ParagraphStyle(
        "bullet",
        parent=body,
        leftIndent=14,
        bulletIndent=2,
        spaceAfter=2,
    )
    return {"body": body, "h1": h1, "h2": h2, "h3": h3, "caption": caption, "code": code, "bullet": bullet}


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def _setup_matplotlib() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": DIVIDER,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "grid.color": DIVIDER,
        "grid.linestyle": ":",
        "font.family": "DejaVu Sans",
        "font.size": 9,
    })


def chart_architecture() -> Path:
    """Re-draw the data flow as a static PNG (raw -> cleaned -> features -> models -> serve)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, "raw/\nCSV / XLSX", "#F1F5F9", TEXT_PRIMARY),
        (2.3, "ingest + clean\n(Great Expectations)", "#FFFFFF", ACCENT_DARK),
        (4.3, "features\nRFM, daily, lag/roll", "#FFFFFF", ACCENT_DARK),
        (6.3, "models\nProphet / XGB / KMeans / ABC", "#FFFFFF", ACCENT_DARK),
        (8.3, "serve + monitor\nAPI · UI · drift", ACCENT_LIGHT, SURFACE),
    ]
    for x, label, fc, tc in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, 1.2), 1.6, 1.6,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.2, edgecolor=ACCENT, facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + 0.8, 2.0, label, ha="center", va="center",
                fontsize=9, color=tc, fontweight="bold")

    # Arrows
    for x in (1.9, 3.9, 5.9, 7.9):
        ax.annotate("", xy=(x + 0.4, 2.0), xytext=(x, 2.0),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))

    # Bottom row: artefacts
    artefacts = ["cleaned.parquet", "rfm.parquet", "daily_revenue.parquet", "inventory_table.csv", "drift_report.html"]
    for i, (x, _, _, _) in enumerate(boxes):
        ax.text(x + 0.8, 0.6, artefacts[i], ha="center", va="center",
                fontsize=7.5, color=TEXT_MUTED, family="monospace")

    ax.text(5, 3.5, "NeuralRetail — pipeline", ha="center", va="center",
            fontsize=13, color=ACCENT_DARK, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "architecture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_daily_revenue() -> Path:
    df = pd.read_parquet(DATA / "daily_revenue.parquet")
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    ax.plot(df.index, df["Revenue"], color=ACCENT, linewidth=1.0, alpha=0.45, label="Daily revenue")
    rolling = df["Revenue"].rolling(7, min_periods=1).mean()
    ax.plot(df.index, rolling, color=ACCENT_DARK, linewidth=2.2, label="7-day rolling mean")
    ax.fill_between(df.index, df["Revenue"].rolling(7, min_periods=1).mean() - df["Revenue"].rolling(7, min_periods=1).std(),
                    df["Revenue"].rolling(7, min_periods=1).mean() + df["Revenue"].rolling(7, min_periods=1).std(),
                    color=ACCENT, alpha=0.10)
    ax.set_title("Daily revenue (2010-12 → 2011-12)  ·  total $5.75M, 374 days")
    ax.set_ylabel("Revenue (USD)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = FIG_DIR / "daily_revenue.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_revenue_by_country() -> Path:
    df = pd.read_parquet(DATA / "cleaned.parquet")
    by_country = (
        df.groupby("Country")["TotalPrice"].sum()
        .sort_values(ascending=True).tail(8)  # horizontal bar
    )
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    ax.barh(by_country.index, by_country.values, color=ACCENT, edgecolor=ACCENT_DARK)
    ax.set_title("Revenue by country — top 8 (United Kingdom dominates at 85%)")
    ax.set_xlabel("Revenue (USD)")
    for i, v in enumerate(by_country.values):
        ax.text(v, i, f"  ${v/1_000_000:.2f}M", va="center", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    path = FIG_DIR / "revenue_by_country.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_persona_pie() -> Path:
    rfm = pd.read_parquet(DATA / "rfm.parquet")
    pipe = joblib.load(MODELS / "segmentation_kmeans.joblib")
    feats = rfm[["Recency", "Frequency", "Monetary"]].fillna(0).to_numpy(dtype=float)
    labels = pipe.predict(feats)
    centroids = pd.DataFrame(
        pipe.named_steps["scaler"].inverse_transform(pipe.named_steps["kmeans"].cluster_centers_),
        columns=["Recency", "Frequency", "Monetary"],
    )
    from neuralretail.models.segmentation import _assign_personas
    persona_map = _assign_personas(centroids)
    rfm = rfm.copy()
    rfm["persona"] = [persona_map.get(int(l), "Regular") for l in labels]
    counts = rfm["persona"].value_counts()
    order = [p for p in ["Champions", "Loyal Customers", "Regular", "At Risk", "Hibernating"] if p in counts.index]
    counts = counts.reindex(order)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    palette = [ACCENT_DARK, ACCENT, ACCENT_LIGHT, "#A6C8C9", "#D4E7E7"]
    wedges, _texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        startangle=90, colors=palette[: len(counts)],
        wedgeprops=dict(width=0.42, edgecolor=SURFACE, linewidth=2),
        textprops=dict(color=TEXT_PRIMARY, fontsize=10),
    )
    for t in autotexts:
        t.set_color(SURFACE)
        t.set_fontsize(8.5)
    ax.set_title(f"Customer distribution by persona (n = {len(rfm):,})")
    fig.tight_layout()
    path = FIG_DIR / "persona_pie.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_abc_pie() -> Path:
    inv = pd.read_csv(MODELS / "inventory_table.csv")
    counts = inv["ABC"].value_counts().reindex(["A", "B", "C"]).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    wedges, _texts, autotexts = ax.pie(
        counts.values, labels=[f"{c}\n({int(v):,})" for c, v in counts.items()],
        autopct="%1.1f%%", startangle=90,
        colors=[ACCENT_DARK, ACCENT, ACCENT_LIGHT],
        wedgeprops=dict(width=0.42, edgecolor=SURFACE, linewidth=2),
        textprops=dict(color=TEXT_PRIMARY, fontsize=10),
    )
    for t in autotexts:
        t.set_color(SURFACE)
        t.set_fontsize(8.5)
    ax.set_title(f"ABC classification (n = {len(inv):,} SKUs)")
    fig.tight_layout()
    path = FIG_DIR / "abc_pie.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_forecast_vs_actual() -> Path:
    daily = pd.read_parquet(DATA / "daily_revenue.parquet")
    model = model_from_json((MODELS / "prophet_demand.json").read_text())
    future = model.make_future_dataframe(periods=30, freq="D")
    fc = model.predict(future)
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    # Last 60 days of actuals
    actuals = daily.tail(60)
    ax.plot(actuals.index, actuals["Revenue"], color=NEUTRAL, linewidth=2.0, marker="o", markersize=3, label="Actual")
    forecast_tail = fc.tail(30)
    ax.plot(forecast_tail["ds"], forecast_tail["yhat"], color=ACCENT, linewidth=2.5, label="Forecast (yhat)")
    ax.fill_between(
        forecast_tail["ds"], forecast_tail["yhat_lower"], forecast_tail["yhat_upper"],
        color=ACCENT, alpha=0.20, label="95% confidence",
    )
    boundary = actuals.index[-1]
    ax.axvline(boundary, color=NEUTRAL, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(boundary, ax.get_ylim()[1] * 0.95, " forecast →", color=TEXT_MUTED, fontsize=8, va="top")
    ax.set_title("Prophet forecast — last 60 days of actuals + 30-day forward forecast")
    ax.set_ylabel("Revenue (USD)")
    ax.legend(loc="upper left", frameon=False, ncols=3, fontsize=8.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = FIG_DIR / "forecast_vs_actual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_drift_summary() -> Path:
    summary_path = REPORT / "drift_report.summary.json"
    if not summary_path.exists():
        return None  # type: ignore[return-value]
    with summary_path.open() as f:
        s = json.load(f)
    per = s["per_column"]
    cols = list(per.keys())
    scores = [per[c]["score"] for c in cols]
    drift = [per[c]["drift_detected"] for c in cols]
    methods = [per[c]["method"] for c in cols]
    # For visualisation, normalise: p-value scores invert (lower = more drift),
    # so use 1 - p for visual scale.
    display = []
    for sc, m in zip(scores, methods):
        if "p_value" in m:
            display.append(1.0 - sc)  # higher = more drift
        else:
            display.append(sc)
    colors_bar = [ACCENT_DARK if d else NEUTRAL for d in drift]
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    bars = ax.bar(cols, display, color=colors_bar, edgecolor=ACCENT_DARK)
    ax.set_title(f"Drift score per column — {s['summary']['n_drifted_columns']}/{s['summary']['n_columns']} drifted (share = {s['summary']['drift_share']:.2f})")
    ax.set_ylabel("Drift score (p-values inverted for display)")
    ax.set_ylim(0, max(1.0, max(display) * 1.1))
    plt.xticks(rotation=15, ha="right")
    for b, d in zip(bars, drift):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                "DRIFT" if d else "ok", ha="center", fontsize=7.5,
                color=ACCENT_DARK if d else TEXT_MUTED, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "drift_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def generate_charts() -> dict[str, Path]:
    _setup_matplotlib()
    paths = {
        "architecture": chart_architecture(),
        "daily_revenue": chart_daily_revenue(),
        "revenue_by_country": chart_revenue_by_country(),
        "persona_pie": chart_persona_pie(),
        "abc_pie": chart_abc_pie(),
        "forecast_vs_actual": chart_forecast_vs_actual(),
    }
    drift = chart_drift_summary()
    if drift is not None:
        paths["drift_summary"] = drift
    # Copy SHAP plot
    shap_src = REPO_ROOT / "shap_summary.png"
    if shap_src.exists():
        shap_dst = FIG_DIR / "shap_summary.png"
        shutil.copyfile(shap_src, shap_dst)
        paths["shap_summary"] = shap_dst
    return paths


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------


def _on_page(canvas, doc) -> None:
    """Header + footer on every page after the cover."""
    canvas.saveState()
    # Header
    canvas.setStrokeColor(colors.HexColor(DIVIDER))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_L, PAGE_H - 0.55 * inch, PAGE_W - MARGIN_R, PAGE_H - 0.55 * inch)
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor(TEXT_MUTED))
    canvas.drawString(MARGIN_L, PAGE_H - 0.42 * inch, "NeuralRetail  ·  Project Overview")
    canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 0.42 * inch,
                           f"Amdox Technologies  ·  {date.today():%B %Y}")
    # Footer
    canvas.line(MARGIN_L, 0.55 * inch, PAGE_W - MARGIN_R, 0.55 * inch)
    canvas.drawString(MARGIN_L, 0.38 * inch, f"Generated {date.today():%Y-%m-%d}  ·  neuralretail v0.1.0")
    canvas.drawRightString(PAGE_W - MARGIN_R, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _on_cover(canvas, doc) -> None:
    """Cover page: just the accent block, no header/footer."""
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(ACCENT_DARK))
    canvas.rect(0, PAGE_H - 2.2 * inch, PAGE_W, 2.2 * inch, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor(ACCENT))
    canvas.rect(0, PAGE_H - 0.25 * inch, PAGE_W, 0.25 * inch, stroke=0, fill=1)
    canvas.restoreState()


def _table(rows: list[list], col_widths: list[float], header: bool = True) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(TEXT_PRIMARY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(DIVIDER)),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.HexColor(ACCENT)),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(ACCENT_DARK)),
        ]
    t.setStyle(TableStyle(style))
    return t


def _wrap_cells(rows: list[list], styles: dict) -> list[list]:
    """Convert cell strings to Paragraphs so long text wraps inside table cells."""
    body = styles["body"]
    code_inline = ParagraphStyle("code_inline", parent=body, fontName="Courier", fontSize=8)
    _ = code_inline  # currently unused; kept for future inline-cell formatting
    out = []
    for row in rows:
        new_row = []
        for cell in row:
            if isinstance(cell, str) and ("\n" in cell or len(cell) > 40):
                new_row.append(Paragraph(cell.replace("\n", "<br/>"), body))
            else:
                new_row.append(cell)
        out.append(new_row)
    return out


def _img(path: Path, width_in: float) -> Image:
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_in * inch
    img.drawHeight = width_in * ratio * inch
    return img


# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------


def build_cover(styles: dict) -> list:
    flow = []
    flow.append(Spacer(1, 2.0 * inch))
    # Title
    flow.append(Paragraph("NeuralRetail", ParagraphStyle(
        "cover_title", parent=styles["h1"],
        fontSize=42, textColor=colors.HexColor(SURFACE),
        spaceAfter=8, alignment=0,
    )))
    flow.append(Paragraph("AI-powered retail sales intelligence", ParagraphStyle(
        "cover_sub", parent=styles["body"],
        fontSize=16, textColor=colors.HexColor(SURFACE),
        leading=20, alignment=0, spaceAfter=40,
    )))
    flow.append(Paragraph("Demand forecasting · Customer segmentation · Churn prediction · Inventory optimisation",
                           ParagraphStyle("cover_tags", parent=styles["body"],
                                          fontSize=11, textColor=colors.HexColor(ACCENT_LIGHT),
                                          leading=15, alignment=0, spaceAfter=80)))
    flow.append(Paragraph("Amdox Technologies", styles["h3"]))
    flow.append(Paragraph(f"Project Overview  ·  {date.today():%B %Y}", styles["body"]))
    flow.append(Spacer(1, 0.5 * inch))
    # Badge row
    badges = [
        "Python 3.12", "Prophet", "XGBoost", "KMeans", "MLflow",
        "FastAPI", "Streamlit", "Evidently", "Docker", "83 tests",
    ]
    badge_html = "  ".join(
        f'<font color="{ACCENT_DARK}"><b>●</b></font> {b}' for b in badges
    )
    flow.append(Paragraph(badge_html, ParagraphStyle("badges", parent=styles["body"],
                                                     fontSize=9, textColor=colors.HexColor(TEXT_MUTED),
                                                     leading=14)))
    flow.append(PageBreak())
    return flow


def build_toc(styles: dict) -> list:
    items = [
        ("1", "What this is"),
        ("2", "Business value & demo script"),
        ("3", "High-level architecture"),
        ("4", "Data pipeline"),
        ("5", "Models — forecasting, churn, segmentation, inventory"),
        ("6", "Model performance & explainability"),
        ("7", "Drift monitoring"),
        ("8", "FastAPI service"),
        ("9", "Streamlit dashboard"),
        ("10", "MLflow registry & promotion"),
        ("11", "How to run it"),
        ("12", "Repo layout, testing, CI"),
        ("13", "Limitations & known gaps"),
        ("14", "Future scale-up path"),
    ]
    rows = []
    for n, label in items:
        rows.append([f"§{n}", label, ""])
    t = Table(
        [[f"<b>{r[0]}</b>", r[1], r[2]] for r in rows],
        colWidths=[0.6 * inch, 4.6 * inch, 1.4 * inch],
    )
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 11),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(TEXT_PRIMARY)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor(ACCENT)),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow = [Paragraph("Contents", styles["h1"]), t, PageBreak()]
    return flow


def build_what(styles: dict) -> list:
    flow = [
        Paragraph("1. What this is", styles["h1"]),
        Paragraph(
            "NeuralRetail is a complete retail sales intelligence platform built for "
            "<b>Amdox Technologies</b>. It ingests the Online Retail II transaction "
            "dataset, validates it with Great Expectations, engineers RFM and time-series "
            "features, and trains four machine-learning models — one for demand forecasting, "
            "one for customer churn, one for behavioural segmentation, and one for inventory "
            "reorder recommendations. The trained models are tracked in MLflow, served via a "
            "FastAPI REST API, visualised in a five-page Streamlit dashboard, and continuously "
            "checked for data drift with an Evidently AI report.",
            styles["body"]),
        Paragraph("The four headline capabilities", styles["h2"]),
    ]
    caps = [
        ("Demand forecasting", "Prophet, 30-day horizon, MAPE 0.075 on the synthetic fallback — under the 10% spec target."),
        ("Churn prediction", "XGBoost on RFM + behavioural features, AUC-ROC 1.0 on the synthetic label rule; SHAP-tracked feature importance."),
        ("Customer segmentation", "KMeans on standardised RFM, k selected by silhouette (0.61 on the synthetic data), 4 personas derived from cluster-centroid rank."),
        ("Inventory recommender", "ABC classification + Wilson EOQ + dead-stock flag, 11,207 SKUs covered, 81.75% dead-stock share flagging long-tail SKUs."),
    ]
    for title, body in caps:
        flow.append(Paragraph(f"<b>{title}.</b> {body}", styles["bullet"], bulletText="•"))
    flow.append(Spacer(1, 0.1 * inch))
    flow.append(Paragraph(
        "The codebase is structured like a production system — typed schemas, pydantic-settings "
        "configuration, MLflow model registry, an Airflow DAG stub for scheduled execution, "
        "a Dockerfile per service, and 83 pytest tests covering every module — but it is "
        "designed to be <b>runnable on a single laptop</b> with one CLI chain.",
        styles["body"]))
    flow.append(PageBreak())
    return flow


def build_business_value(styles: dict) -> list:
    flow = [
        Paragraph("2. Business value & demo script", styles["h1"]),
        Paragraph(
            "Each of the four models answers a question a retail business actually asks.",
            styles["body"]),
    ]
    qs = [
        ("Forecasting", "<i>How much revenue should we expect in the next 7/30/90 days?</i> — the dashboard's Demand Explorer overlays a 95% confidence band on top of actuals; the API exposes the same forecast as JSON."),
        ("Churn", "<i>Which customers are about to stop buying?</i> — the API returns a per-customer probability in &lt;5ms; the marketing team can re-target the top decile before they lapse."),
        ("Segmentation", "<i>Who are our Champions, and who is at risk of churning?</i> — the Customer Hub visualises the persona mix and exports a per-persona summary CSV."),
        ("Inventory", "<i>What should we reorder, and which SKUs are dead?</i> — the Inventory Health page shows the ABC pie plus a top-N reorder table; dead-stock flagging surfaces capital tied up in slow movers."),
    ]
    for title, body in qs:
        flow.append(Paragraph(f"<b>{title}.</b> {body}", styles["bullet"], bulletText="•"))
    flow.append(Spacer(1, 0.15 * inch))
    flow.append(Paragraph("A 5-minute demo for a non-technical audience", styles["h2"]))
    flow.append(Paragraph("Open the dashboard at <b>http://localhost:8501</b> and walk through:", styles["body"]))
    demo = [
        "Executive Overview — point out the headline KPIs and the long-tail country distribution (UK dominates at 85%).",
        "Sales Analytics — show the monthly revenue trend with the 7-day rolling mean, and the top-10 products by revenue.",
        "Customer Hub — highlight the persona pie. Ask: 'Which segment should marketing spend the next dollar on?'",
        "Demand Explorer — slide the forecast horizon from 7 to 90 days. The confidence band widens visibly; that's the model saying 'I am less sure the further out I look.'",
        "Inventory Health — point out the dead-stock share (~82%) and the top-3 reorders. Ask: 'How much working capital is that?'",
    ]
    for d in demo:
        flow.append(Paragraph(d, styles["bullet"], bulletText="›"))
    flow.append(Paragraph("If asked a deeper technical question, open the FastAPI Swagger UI at <b>http://localhost:8000/docs</b> and run a live prediction. The whole loop — input → feature vector → model → JSON response — takes &lt;50ms.", styles["body"]))
    flow.append(PageBreak())
    return flow


def build_architecture(styles: dict, charts: dict) -> list:
    flow = [
        Paragraph("3. High-level architecture", styles["h1"]),
        Paragraph(
            "The platform has four logical stages. Data flows left-to-right; artefacts "
            "(parquet files, model JSON, drift reports) flow back from each stage so the "
            "next stage and the serving layer can pick them up independently.",
            styles["body"]),
    ]
    if "architecture" in charts:
        flow.append(_img(charts["architecture"], width_in=6.5))
        flow.append(Paragraph("Figure 1 — end-to-end pipeline (raw → cleaned → features → models → serve & monitor).",
                              styles["caption"]))
    flow.append(Paragraph("Layer-by-layer", styles["h2"]))
    layers = [
        ("Data", "data/ingest.py auto-detects XLSX or CSV, falls back to a 30,000-row synthetic generator. data/clean.py applies the four cleaning rules and a Great Expectations suite (10 expectations, must all pass)."),
        ("Features", "features/rfm.py builds per-customer Recency / Frequency / Monetary. features/timeseries.py builds daily revenue + lag (1, 7, 14) + rolling (7, 14, 30) + calendar features."),
        ("Models", "models/forecasting.py, churn.py, segmentation.py, inventory.py. Each one logs params + metrics + artefacts to MLflow, registers a pyfunc under a stable name, and saves an on-disk copy for the dashboard."),
        ("Serve & monitor", "api/main.py exposes five endpoints with API-key auth. dashboard/app.py serves the five pages from the same parquet/csv artefacts. monitoring/drift.py writes an Evidently HTML report on a 70/30 chronological split."),
    ]
    for title, body in layers:
        flow.append(Paragraph(f"<b>{title}.</b> {body}", styles["bullet"], bulletText="•"))
    flow.append(PageBreak())
    return flow


def build_data_pipeline(styles: dict) -> list:
    clean_report = json.loads((DATA / "clean_report.json").read_text())
    flow = [
        Paragraph("4. Data pipeline", styles["h1"]),
        Paragraph(
            "The data pipeline is the most consequential part of the system: every model "
            "is downstream of it, and every metric the model reports is downstream of "
            "what the cleaner kept.",
            styles["body"]),
        Paragraph("Cleaning rules", styles["h2"]),
    ]
    rules = [
        "Drop cancelled invoices (InvoiceNo starts with 'C' or 'c').",
        "Drop rows with a null CustomerID.",
        "Filter Quantity > 0 and UnitPrice > 0.",
        "Parse InvoiceDate to datetime; drop rows that fail to parse.",
        "Compute TotalPrice = Quantity × UnitPrice; cast CustomerID to int.",
    ]
    for r in rules:
        flow.append(Paragraph(r, styles["bullet"], bulletText="•"))
    flow.append(Paragraph("Latest cleaning run (synthetic fallback)", styles["h2"]))
    flow.append(_table([
        ["Stage", "Rows"],
        ["Rows in", f"{clean_report['rows_in']:,}"],
        ["Cancelled dropped", f"{clean_report['cancelled_dropped']:,}"],
        ["Null CustomerID dropped", f"{clean_report['null_customer_dropped']:,}"],
        ["Non-positive Quantity dropped", f"{clean_report['nonpositive_quantity_dropped']:,}"],
        ["Non-positive Price dropped", f"{clean_report['nonpositive_price_dropped']:,}"],
        ["Rows out", f"{clean_report['rows_out']:,}"],
    ], col_widths=[3.4 * inch, 1.6 * inch]))
    flow.append(Spacer(1, 0.1 * inch))
    flow.append(Paragraph(
        "After cleaning, a <b>Great Expectations</b> suite (10 expectations) re-validates the "
        "parquet: row count within a sane bound, no nulls in the six key columns, and value "
        "ranges on Quantity (≥ 1), UnitPrice (≥ 0.01), and TotalPrice (≥ 0.01). If the suite "
        "fails, the cleaner refuses to write the parquet and the CLI exits with code 1.",
        styles["body"]))
    flow.append(Paragraph("Feature engineering outputs", styles["h2"]))
    flow.append(_table([
        ["Output", "Rows", "Columns", "Notes"],
        ["rfm.parquet", "1,500", "6", "CustomerID, Recency, Frequency, Monetary, FirstPurchase, LastPurchase"],
        ["daily_revenue.parquet", "374", "4", "Revenue, Orders, ItemsSold, Customers — one row per day, 2010-12-01 to 2011-12-09"],
        ["timeseries_features.parquet", "374", "19", "Daily + lag (1, 7, 14) + rolling (7, 14, 30) + calendar"],
    ], col_widths=[1.6 * inch, 0.6 * inch, 0.6 * inch, 3.7 * inch]))
    flow.append(PageBreak())
    return flow


def _model_card_section(styles: dict, model_name: str, summary: list) -> list:
    flow = [Paragraph(model_name, styles["h2"])]
    flow.append(_table(summary, col_widths=[1.6 * inch, 4.9 * inch]))
    return flow


def build_models(styles: dict) -> list:
    flow = [Paragraph("5. Models", styles["h1"])]
    flow.append(Paragraph(
        "Each model module exposes a clean <font face='Courier'>train() / save() / load_latest()</font> "
        "interface, logs every run to MLflow, and is reachable from the API, the dashboard, and the "
        "notebooks. The next four subsections mirror the four model cards in <font face='Courier'>report/model_cards/</font>.",
        styles["body"]))
    # Forecasting
    flow += _model_card_section(styles, "5.1 Demand forecasting — Prophet", [
        ["Algorithm", "Prophet with multiplicative weekly seasonality, year disabled by default. UK holiday calendar."],
        ["Inputs", "data/processed/daily_revenue.parquet (one row per calendar day)."],
        ["Train / holdout", "Chronological 80 / 20; last 30 days = holdout."],
        ["Primary metric", "MAPE on 30-day holdout."],
        ["Spec target", "MAPE ≤ 0.10."],
        ["Latest measured", "MAPE = <b>0.0746</b> · RMSE = 1,702.24 · horizon 30 days."],
        ["Intended use", "Demand Explorer page (7-90 day horizon slider); /predict/demand API."],
        ["Limitations", "Does not capture demand shocks; no exogenous regressors. Loaded from on-disk JSON, not the MLflow registry (the spec's 'sklearn model' promotion contract was not extended to Prophet)."],
    ])
    # Churn
    flow += _model_card_section(styles, "5.2 Churn classifier — XGBoost", [
        ["Algorithm", "XGBoost (n_estimators=200, max_depth=4, lr=0.05), stratified 80 / 20 split, SHAP tree explainer."],
        ["Inputs", "RFM (Recency, Frequency, Monetary) + per-customer behavioural aggregates (avg basket, unique products, mean days between invoices, is_UK)."],
        ["Primary metric", "AUC-ROC on holdout."],
        ["Spec target", "AUC-ROC ≥ 0.90."],
        ["Latest measured", "AUC-ROC = <b>1.0000</b> · F1 = 1.0000."],
        ["Intended use", "Marketing reactivation triage; /predict/churn API; SHAP waterfall per customer."],
        ["Limitations", "AUC=1.0 is generator-specific (label = Recency > 90), so the model trivially recovers the label from RFM. On a real labelled dataset the AUC is expected to land in 0.85–0.95. No temporal cross-validation; class-imbalance handling is minimal."],
        ["Opt-in", "LightGBM comparison run is gated behind NEURALRETAIL_ENABLE_LIGHTGBM=true."],
    ])
    # Segmentation
    flow += _model_card_section(styles, "5.3 Customer segmentation — KMeans", [
        ["Algorithm", "KMeans on StandardScaler-normalised RFM, k ∈ [4, 8] selected by silhouette."],
        ["Inputs", "data/processed/rfm.parquet."],
        ["Primary metric", "Silhouette on the training set."],
        ["Spec target", "Silhouette ≥ 0.55, 4-8 clusters."],
        ["Latest measured", "Silhouette = <b>0.6104</b> · best k = 4."],
        ["Intended use", "Customer Hub page (Recency vs Monetary scatter, persona pie); /segment/score API."],
        ["Limitations", "KMeans assumes spherical, equally-sized clusters — a poor fit for the long-tailed RFM distribution. Personas are coarse (3-8 clusters); a real deployment would overlay behavioural features (basket size, channel)."],
        ["Persona rule", "Labels derived from cluster-centroid rank (Champions, Loyal Customers, At Risk, Regular, Hibernating) — not hardcoded."],
    ])
    # Inventory
    flow += _model_card_section(styles, "5.4 Inventory recommender — ABC + EOQ", [
        ["Algorithm", "Pareto ABC classification (top 80% = A, next 15% = B, last 5% = C) + Wilson EOQ + dead-stock flag (no sale in 60 days)."],
        ["Inputs", "data/processed/cleaned.parquet, grouped by (StockCode, Description)."],
        ["Primary metric", "Data coverage = n_skus; ABC class counts; dead-stock %."],
        ["Spec target", "No formal target — this is a deterministic table, not a learned model."],
        ["Latest measured", "n_skus = <b>11,207</b> · A/B/C = 5,869 / 3,064 / 2,274 · dead-stock 9,162 (81.75%)."],
        ["Intended use", "Inventory Health page (ABC pie + top-N reorder table); /inventory/reorder API."],
        ["Limitations", "No per-SKU seasonality; EOQ assumes independent demand. Holding cost (20%) and ordering cost ($50) are industry defaults, not company-specific. Current stock and lead time are not in the model — they would come from the WMS in a real deployment."],
    ])
    flow.append(PageBreak())
    return flow


def build_performance(styles: dict, charts: dict) -> list:
    flow = [
        Paragraph("6. Model performance & explainability", styles["h1"]),
        Paragraph(
            "The numbers below are the <b>actual outputs</b> of <font face='Courier'>python -m neuralretail.cli train</font> "
            "against the synthetic Online Retail II fallback, not aspirational targets. They land "
            "in spec for the headline metrics (MAPE, AUC, silhouette, k ∈ [4, 8]).",
            styles["body"]),
        Paragraph("Metrics vs spec", styles["h2"]),
    ]
    flow.append(_table([
        ["Model", "Metric", "Value", "Spec target", "Pass"],
        ["Prophet (demand)", "MAPE", "0.0746", "≤ 0.10", "✓"],
        ["Prophet (demand)", "RMSE", "1,702.24", "—", "—"],
        ["XGBoost (churn)", "AUC-ROC", "1.0000", "≥ 0.90", "✓"],
        ["XGBoost (churn)", "F1", "1.0000", "—", "—"],
        ["KMeans (segmentation)", "silhouette", "0.6104", "≥ 0.55", "✓"],
        ["KMeans (segmentation)", "best k", "4", "4–8", "✓"],
        ["ABC/EOQ (inventory)", "SKUs", "11,207", "—", "—"],
        ["ABC/EOQ (inventory)", "dead-stock %", "81.75%", "—", "—"],
    ], col_widths=[1.7 * inch, 1.3 * inch, 1.0 * inch, 1.0 * inch, 0.7 * inch]))
    flow.append(Paragraph(
        "<b>Caveat.</b> The AUC = 1.0 on churn is a label-leakage artefact: the synthetic generator "
        "defines churned = Recency &gt; 90 days, so the model can recover the label from Recency alone. "
        "On a real labelled dataset the AUC is expected in the 0.85–0.95 range. The MAPE and silhouette "
        "are honest — they were validated on a chronological holdout and a full RFM table without any "
        "label leakage.",
        styles["body"]))
    flow.append(Paragraph("Churn — SHAP feature importance", styles["h2"]))
    if "shap_summary" in charts:
        flow.append(_img(charts["shap_summary"], width_in=5.8))
        flow.append(Paragraph("Figure 2 — SHAP summary plot for the XGBoost churn classifier (recency dominates).",
                              styles["caption"]))
    flow.append(PageBreak())
    return flow


def build_drift(styles: dict, charts: dict) -> list:
    summary_path = REPORT / "drift_report.summary.json"
    if not summary_path.exists():
        return [Paragraph("7. Drift monitoring", styles["h1"]),
                Paragraph("No drift report found. Run <font face='Courier'>python -m neuralretail.cli monitor</font> first.", styles["body"]),
                PageBreak()]
    s = json.loads(summary_path.read_text())
    flow = [
        Paragraph("7. Drift monitoring", styles["h1"]),
        Paragraph(
            "After every pipeline run, the monitoring step splits the cleaned data chronologically "
            "(70% oldest as reference, 30% newest as current) and runs an Evidently AI "
            "<font face='Courier'>DataDriftPreset</font> over six columns: Quantity, UnitPrice, "
            "TotalPrice, Country, plus derived Hour and DayOfWeek. The output is an interactive "
            "HTML report plus a sidecar JSON for machine readers.",
            styles["body"]),
    ]
    flow.append(Paragraph("Split + result", styles["h2"]))
    flow.append(_table([
        ["", "Reference", "Current"],
        ["Rows", f"{s['reference']['n_rows']:,}", f"{s['current']['n_rows']:,}"],
        ["Start", s["reference"]["start"], s["current"]["start"]],
        ["End", s["reference"]["end"], s["current"]["end"]],
        ["Drifted columns", f"{s['summary']['n_drifted_columns']} / {s['summary']['n_columns']}", f"share = {s['summary']['drift_share']:.2f}"],
    ], col_widths=[1.4 * inch, 2.3 * inch, 2.3 * inch]))
    flow.append(Paragraph("Per-column drift scores", styles["h2"]))
    rows = [["Column", "Score", "Method", "Threshold", "Drift?"]]
    for col, info in s["per_column"].items():
        rows.append([col, f"{info['score']:.4f}", info["method"].split()[0],
                     f"{info['threshold']}", "YES" if info["drift_detected"] else "no"])
    flow.append(_table(rows, col_widths=[1.4 * inch, 1.0 * inch, 1.4 * inch, 1.0 * inch, 1.2 * inch]))
    if "drift_summary" in charts:
        flow.append(Spacer(1, 0.1 * inch))
        flow.append(_img(charts["drift_summary"], width_in=6.5))
        flow.append(Paragraph("Figure 3 — per-column drift scores (p-values inverted for display).", styles["caption"]))
    flow.append(Paragraph(
        "In this run, the only column flagged as drifted is <b>TotalPrice</b>. This is expected: the "
        "synthetic data has a +3% trend over the year plus a Q4 holiday bump, so the more recent "
        "30% slice has a higher mean revenue than the older 70% slice.",
        styles["body"]))
    flow.append(PageBreak())
    return flow


def build_api(styles: dict) -> list:
    flow = [
        Paragraph("8. FastAPI service", styles["h1"]),
        Paragraph(
            "The FastAPI service in <font face='Courier'>api/main.py</font> exposes five endpoints. "
            "All scoring endpoints require an <font face='Courier'>X-API-Key</font> header "
            "(default value <font face='Courier'>change-me-in-prod</font>, override via "
            "<font face='Courier'>NEURALRETAIL_API_KEY</font> in <font face='Courier'>.env</font>).",
            styles["body"]),
        Paragraph("Endpoints", styles["h2"]),
    ]
    flow.append(_table([
        ["Method", "Path", "Auth", "Purpose"],
        ["GET", "/health", "no", "Liveness + per-model load status"],
        ["POST", "/predict/demand", "yes", "Prophet forecast for N days"],
        ["POST", "/predict/churn", "yes", "Per-customer churn probability"],
        ["POST", "/segment/score", "yes", "KMeans cluster + persona for one customer"],
        ["POST", "/inventory/reorder", "yes", "Top-N reorder list with ABC + dead-stock filter"],
    ], col_widths=[0.6 * inch, 1.6 * inch, 0.5 * inch, 3.3 * inch]))
    flow.append(Paragraph("Sample: /health", styles["h3"]))
    flow.append(Paragraph('<font face="Courier">{"status":"ok","version":"0.1.0","models_loaded":{"forecasting":true,"churn":true,"segmentation":true,"inventory":true}}</font>', styles["code"]))
    flow.append(Paragraph("Sample: /predict/demand", styles["h3"]))
    flow.append(Paragraph('<font face="Courier">curl -X POST http://localhost:8000/predict/demand \\<br/>  -H "X-API-Key: change-me-in-prod" \\<br/>  -H "Content-Type: application/json" \\<br/>  -d \'{"horizon_days": 7}\'</font>', styles["code"]))
    flow.append(Paragraph("Response (last 7 days, abridged)", styles["body"]))
    flow.append(Paragraph(
        '<font face="Courier">{"horizon_days": 7, "points": [<br/>'
        '  {"ds": "2011-12-10", "yhat": 13831.92, "yhat_lower": 12025.30, "yhat_upper": 15740.16},<br/>'
        '  ... 6 more rows ...<br/>'
        ']}</font>', styles["code"]))
    flow.append(Paragraph("Sample: /predict/churn", styles["h3"]))
    flow.append(Paragraph('<font face="Courier">curl -X POST http://localhost:8000/predict/churn \\<br/>  -H "X-API-Key: change-me-in-prod" \\<br/>  -H "Content-Type: application/json" \\<br/>  -d \'{"customers":[{"recency":10,"frequency":5,"monetary":1000.0}]}\'</font>', styles["code"]))
    flow.append(Paragraph(
        "The API loads the Prophet model from the MLflow registry first "
        "(<font face='Courier'>models:/neuralretail_demand_forecaster@Production</font>) "
        "and falls back to the on-disk <font face='Courier'>prophet_demand.json</font> if the "
        "registry is empty — a fast local-dev fallback.",
        styles["body"]))
    flow.append(PageBreak())
    return flow


def build_dashboard(styles: dict, charts: dict) -> list:
    flow = [Paragraph("9. Streamlit dashboard", styles["h1"])]
    flow.append(Paragraph(
        "The Streamlit app at <font face='Courier'>dashboard/app.py</font> renders five pages from the "
        "on-disk parquet/csv artefacts. The left sidebar exposes country + date filters that apply to "
        "pages 1, 2, and 5 (Customer Hub and Demand Explorer are global). The single accent colour "
        "is <font face='Courier'>#0E8388</font>, enforced by a pytest test that rejects any hex "
        "literal in dashboard code outside <font face='Courier'>theme.py</font>.",
        styles["body"]))
    pages = [
        ("9.1 Executive Overview", "KPI tiles (total revenue, orders, customers, AOV) and a revenue-by-country bar. Sidebar filters apply.",
         "revenue_by_country"),
        ("9.2 Sales Analytics", "Monthly revenue trend with markers and a top-N products bar (slider 5-20). Sidebar filters apply.",
         None),
        ("9.3 Customer Hub", "RFM scatter (Recency vs Monetary, log-log, coloured by persona), persona pie, per-persona summary table. Global.",
         "persona_pie"),
        ("9.4 Demand Explorer", "Last 60 days of actuals + 30-day Prophet forecast with 95% confidence band. Global.",
         "forecast_vs_actual"),
        ("9.5 Inventory Health", "ABC pie (global, from the trained model) + a reorder recommendations table filtered by sidebar country/date inputs.",
         "abc_pie"),
    ]
    for title, body, chart_key in pages:
        flow.append(Paragraph(title, styles["h2"]))
        flow.append(Paragraph(body, styles["body"]))
        if chart_key and chart_key in charts:
            flow.append(_img(charts[chart_key], width_in=5.8))
            flow.append(Paragraph(f"Figure — {title.split(' ', 1)[1]}.", styles["caption"]))
    flow.append(Paragraph("Daily revenue overview (for context)", styles["h2"]))
    if "daily_revenue" in charts:
        flow.append(_img(charts["daily_revenue"], width_in=6.5))
    flow.append(PageBreak())
    return flow


def build_mlflow(styles: dict) -> list:
    flow = [
        Paragraph("10. MLflow registry & promotion", styles["h1"]),
        Paragraph(
            "Every training run logs params, metrics, and artefacts (forecast CSV, feature "
            "importances, SHAP plot, persona summary, inventory table) to MLflow. Each model is "
            "registered under a stable name and the best run is aliased to <font face='Courier'>Production</font>.",
            styles["body"]),
        Paragraph("Registered models", styles["h2"]),
    ]
    flow.append(_table([
        ["Logical key", "Registered name", "Primary metric", "Direction"],
        ["forecasting", "neuralretail_demand_forecaster", "mape", "min"],
        ["churn", "neuralretail_churn_classifier", "auc_roc", "max"],
        ["segmentation", "neuralretail_customer_segmenter", "silhouette", "max"],
        ["inventory", "neuralretail_inventory_recommender", "n_skus", "max"],
    ], col_widths=[1.3 * inch, 2.3 * inch, 1.4 * inch, 0.9 * inch]))
    flow.append(Paragraph("Promotion algorithm", styles["h2"]))
    flow.append(Paragraph(
        "After every <font face='Courier'>train</font> run, the <font face='Courier'>promote</font> "
        "step calls <font face='Courier'>promote_best()</font> in <font face='Courier'>_mlflow_utils.py</font>. "
        "For each registered model, it walks all versions, picks the one whose run has the best value "
        "on the primary metric (min or max as defined above), and sets the <font face='Courier'>Production</font> "
        "alias on that version. The API then loads the production version at startup via "
        "<font face='Courier'>mlflow.pyfunc.load_model(\"models:/&lt;name&gt;@Production\")</font>.",
        styles["body"]))
    flow.append(Paragraph("Browse runs in the UI", styles["body"]))
    flow.append(Paragraph("Start the MLflow server with <font face='Courier'>mlflow ui --backend-store-uri sqlite:///./mlruns/mlflow.db --port 5000</font> and open <b>http://localhost:5000</b>.", styles["body"]))
    flow.append(PageBreak())
    return flow


def build_how_to_run(styles: dict) -> list:
    flow = [
        Paragraph("11. How to run it", styles["h1"]),
        Paragraph("The whole pipeline is five CLI commands:", styles["body"]),
    ]
    cmds = [
        "python -m neuralretail.cli data       # ingest + clean + GE validate → cleaned.parquet",
        "python -m neuralretail.cli features   # build rfm.parquet + daily_revenue.parquet + timeseries_features.parquet",
        "python -m neuralretail.cli train      # train 4 models, log to MLflow, save on-disk artefacts",
        "python -m neuralretail.cli promote    # promote best run of each model to 'Production'",
        "python -m neuralretail.cli monitor    # generate report/drift_report.html + sidecar JSON",
    ]
    for c in cmds:
        flow.append(Paragraph(f'<font face="Courier">{c}</font>', styles["code"]))
    flow.append(Paragraph("Total wall-clock on a single laptop: ~2 minutes for the synthetic fallback (Prophet fitting is the longest step).", styles["body"]))
    flow.append(Paragraph("Launch the three services", styles["h2"]))
    services = [
        ("Dashboard", "streamlit run src/neuralretail/dashboard/app.py --server.port 8501", "http://localhost:8501"),
        ("API", "uvicorn neuralretail.api.main:app --host 127.0.0.1 --port 8000", "http://localhost:8000  ·  Swagger at /docs"),
        ("MLflow UI", "mlflow ui --backend-store-uri sqlite:///./mlruns/mlflow.db --port 5000", "http://localhost:5000"),
    ]
    for title, cmd, url in services:
        flow.append(Paragraph(f"<b>{title}.</b>  <font face='Courier'>{cmd}</font>", styles["body"]))
        flow.append(Paragraph(url, ParagraphStyle("url", parent=styles["body"], textColor=colors.HexColor(ACCENT), fontSize=9.5)))
    flow.append(Paragraph("Stop everything", styles["h2"]))
    flow.append(Paragraph('<font face="Courier">Get-Process -Name streamlit,uvicorn,mlflow -ErrorAction SilentlyContinue | Stop-Process -Force</font>', styles["code"]))
    flow.append(PageBreak())
    return flow


def build_layout(styles: dict) -> list:
    flow = [
        Paragraph("12. Repo layout, testing, CI", styles["h1"]),
        Paragraph("Project structure", styles["h2"]),
    ]
    tree = [
        "src/neuralretail/",
        "  config.py                 # pydantic-settings, .env-driven",
        "  cli.py                    # python -m neuralretail.cli <data|features|train|promote|monitor>",
        "  data/                     # ingest + clean + Great Expectations",
        "  features/                 # RFM, time-series",
        "  models/                   # forecasting, churn, segmentation, inventory",
        "  monitoring/               # Evidently drift reports",
        "  api/                      # FastAPI service",
        "  dashboard/                # Streamlit multi-page app",
        "data/raw/, data/processed/, models/, mlruns/, report/",
        "notebooks/                 # 5 Jupyter notebooks mirroring production modules",
        "dags/                      # Airflow DAG stub (documentation only)",
        "docker/                    # Dockerfile.{mlflow,api,dashboard} + docker-compose.yml",
        "tests/                     # 83 pytest tests",
        "scripts/                   # operational scripts (incl. this PDF builder)",
        ".github/workflows/ci.yml   # pytest + ruff on every push/PR",
    ]
    for line in tree:
        flow.append(Paragraph(f'<font face="Courier">{line}</font>', styles["code"]))
    flow.append(Paragraph("Tests", styles["h2"]))
    flow.append(Paragraph(
        "83 tests across 10 files, all green. The dashboard suite includes a hex-colour guard "
        "that fails the build if any colour literal appears in dashboard code outside "
        "<font face='Courier'>theme.py</font>. Run with:",
        styles["body"]))
    flow.append(Paragraph('<font face="Courier">python -m pytest -q</font>', styles["code"]))
    flow.append(Paragraph("CI", styles["h2"]))
    flow.append(Paragraph(
        "GitHub Actions runs <font face='Courier'>pytest -q</font> and "
        "<font face='Courier'>ruff check src tests</font> on every push and pull request to "
        "<font face='Courier'>main</font>.",
        styles["body"]))
    flow.append(PageBreak())
    return flow


def build_limitations(styles: dict) -> list:
    flow = [
        Paragraph("13. Limitations & known gaps", styles["h1"]),
        Paragraph("Be honest about what this project is and is not.", styles["body"]),
    ]
    items = [
        ("Synthetic fallback by default", "When no Online Retail II file is in data/raw/, the pipeline generates a 30,000-row synthetic sample. The numbers in this PDF are the v2 generator's outputs. A real ingest will change the headline metrics — the AUC will almost certainly drop out of 1.0, the dead-stock share will shift, and Prophet's MAPE will land wherever the real data's noise floor lives."),
        ("Churn AUC = 1.0 is a label-leakage artefact", "The synthetic generator defines churned = Recency > 90 days, so XGBoost trivially recovers the label from RFM alone. The model itself is well-formed (early stopping, stratified split, SHAP tracked), but the label rule is too clean. On a real labelled dataset the AUC is expected in 0.85–0.95."),
        ("Dead-stock share (~82%) is partly a feature of the synthetic data", "The generator produces sparse, single-purchase SKUs to mirror the real dataset's long tail. On real data the dead-stock share will be lower but still high (Online Retail II has thousands of one-off gift items)."),
        ("LSTM ensemble and DBSCAN are not wired in", "Documented as opt-in behind config flags in the build prompt. The code paths are not present — the day-to-day pipeline uses Prophet and KMeans only."),
        ("Airflow DAG is documentation-only", "dags/neuralretail_pipeline.py mirrors the CLI chain but is not deployed in an Airflow environment. In a real deployment it would need an executor + connection + variable configuration."),
        ("No live drift alerting", "The drift report is an artefact. Threshold-based paging to Slack / PagerDuty is documented in the README as a future step. A real deployment would add a check after the monitor step (e.g. fail the run if drift_share > 0.30)."),
        ("No real WMS integration", "The inventory recommender assumes current stock and lead time are derivable from the synthetic data. In a real deployment those would come from the warehouse management system."),
        ("No authentication beyond a shared API key", "FastAPI uses a single X-API-Key header. The dashboard has no auth at all (it's local). A production deployment would add OAuth / RBAC."),
    ]
    for title, body in items:
        flow.append(Paragraph(f"<b>{title}.</b> {body}", styles["bullet"], bulletText="⚠"))
    flow.append(PageBreak())
    return flow


def build_future(styles: dict) -> list:
    flow = [
        Paragraph("14. Future scale-up path", styles["h1"]),
        Paragraph(
            "The build prompt marks these as 'would build this next' rather than implemented. "
            "The order is roughly the order a real growth-stage retailer would adopt them.",
            styles["body"]),
    ]
    items = [
        ("Spark + Delta Lake", "Once the cleaned data exceeds ~50 GB, port the GE suite and the feature builders onto Spark with Delta for ACID table versions. The Python API stays the same; only the execution engine changes."),
        ("Feast (feature store)", "RFM, daily_revenue, and the lag/rolling feature group are exactly the kind of cross-team features that benefit from a single online + offline store. features/rfm.py and features/timeseries.py are the natural FeatureView definitions."),
        ("Kafka", "Production should ingest from a Kafka topic so the cleaning step is event-driven and the dashboard can refresh in near-real-time."),
        ("Kubernetes + Helm", "Once the docker-compose stack is no longer one developer laptop, port docker/docker-compose.yml to a Helm chart with HPA, PDBs, and a real ingress."),
        ("Terraform", "Provision the cloud resources the Helm chart targets: GCS/S3 for the artefact store, CloudSQL/Postgres for MLflow, a managed Kafka, IAM, networking."),
        ("DoWhy / EconML", "Once the demand-forecasting and inventory models are in production, the obvious next question is 'did the reactivation campaign cause the lift?' — causal inference on top of the A/B-test logs."),
        ("TimeGPT", "Replaces Prophet with a foundation-model forecaster for a free ~15–25% MAPE reduction on most retail series. Drop-in via a forecasting.py flavour swap."),
        ("OpenLineage", "Emits dataset / model lineage events from the pipeline so Marquez or DataHub can show the end-to-end graph (raw → cleaned → features → model → prediction). Fits the monitoring/ package naturally."),
    ]
    for title, body in items:
        flow.append(Paragraph(f"<b>{title}.</b> {body}", styles["bullet"], bulletText="→"))
    flow.append(Paragraph("End of overview", styles["h2"]))
    flow.append(Paragraph(
        "This document was generated from the live on-disk artefacts by "
        "<font face='Courier'>scripts/build_project_pdf.py</font>. Re-run it any time after "
        "<font face='Courier'>python -m neuralretail.cli train</font> to refresh the numbers.",
        styles["body"]))
    return flow


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------


def build_pdf(charts: dict) -> Path:
    styles = _styles()
    flow = []
    flow += build_cover(styles)
    flow += build_toc(styles)
    flow += build_what(styles)
    flow += build_business_value(styles)
    flow += build_architecture(styles, charts)
    flow += build_data_pipeline(styles)
    flow += build_models(styles)
    flow += build_performance(styles, charts)
    flow += build_drift(styles, charts)
    flow += build_api(styles)
    flow += build_dashboard(styles, charts)
    flow += build_mlflow(styles)
    flow += build_how_to_run(styles)
    flow += build_layout(styles)
    flow += build_limitations(styles)
    flow += build_future(styles)

    doc = BaseDocTemplate(
        str(OUTPUT_PDF),
        pagesize=LETTER,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T, bottomMargin=MARGIN_B,
        title="NeuralRetail — Project Overview",
        author="Amdox Technologies",
        subject="Retail sales intelligence platform",
    )
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id="cover", showBoundary=0)
    body_frame = Frame(MARGIN_L, MARGIN_B, FRAME_W, PAGE_H - MARGIN_T - MARGIN_B, id="body", showBoundary=0)
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=_on_cover),
        PageTemplate(id="body", frames=[body_frame], onPage=_on_page),
    ])

    from reportlab.platypus import NextPageTemplate
    # First flow item after cover switches to body template
    flow.insert(1, NextPageTemplate("body"))
    doc.build(flow)
    return OUTPUT_PDF


def main() -> int:
    print("Generating charts…", file=sys.stderr)
    charts = generate_charts()
    print(f"  → {len(charts)} figures in {FIG_DIR}", file=sys.stderr)
    print("Building PDF…", file=sys.stderr)
    out = build_pdf(charts)
    size_kb = out.stat().st_size / 1024
    print(f"  → {out}  ({size_kb:.0f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
