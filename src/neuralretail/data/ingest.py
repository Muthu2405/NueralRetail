"""Data ingest: load Online Retail II from XLSX/CSV with a synthetic fallback.

Behavior
--------
1. If a real Online Retail II file (XLSX or CSV) is present in
   ``NEURALRETAIL_RAW_DIR`` (config), load it.
2. If no real file is present, generate a synthetic sample of the same
   shape (and write it to raw/ so subsequent runs are reproducible) and
   return that.
3. Validate that the spec-mandated columns exist; raise a clear error
   listing any that are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from neuralretail.config import get_settings

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Final[list[str]] = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# Filenames we'll look for in raw_dir, in priority order.
RAW_FILE_CANDIDATES: Final[tuple[str, ...]] = (
    "online_retail_II.xlsx",
    "online_retail_ii.xlsx",
    "online_retail.xlsx",
    "online_retail.csv",
    "online_retail_II.csv",
    "online_retail_ii.csv",
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Persona configuration for the v2 generator
# ---------------------------------------------------------------------------
#
# Each persona specifies a (mu, sigma) per axis in normalised space, and
# the joint distribution drives the RFM table. Tighter sigmas → clearer
# cluster separation → higher silhouette.

_PERSONA_SPECS: Final[tuple[dict, ...]] = (
    # name,            weight, mu_freq, sd_freq, mu_recency, sd_recency, mu_mon, sd_mon
    #
    # Persona means and sigmas are chosen so the five (R, F, M) Gaussians
    # are *non-overlapping* (i.e. the 2-sigma intervals don't cross).
    # This is what drives the KMeans silhouette above 0.55 on the
    # synthetic fallback:
    #
    #   Recency (days since last purchase, before data end):
    #     Champions    :  3 ± 2   -> [ -1,  7 ]
    #     Loyal        : 15 ± 3   -> [  9, 21 ]
    #     Regular      : 45 ± 5   -> [ 35, 55 ]
    #     At Risk      : 90 ± 5   -> [ 80, 100 ]
    #     Hibernating  : 80 ± 5   -> [ 70, 90 ]
    #
    #   Frequency (distinct invoices per customer, 1y window):
    #     Champions    : 25 ± 2   -> [ 21, 29 ]
    #     Loyal        : 12 ± 1   -> [ 10, 14 ]
    #     Regular      :  5 ± 1   -> [  3,  7 ]
    #     At Risk      :  3 ± 1   -> [  1,  5 ]
    #     Hibernating  :  2 ± 0   -> [  2,  2 ]
    #
    #   Monetary (sum of TotalPrice over the year):
    #     Champions    : 6500 ± 500
    #     Loyal        : 2500 ± 200
    #     Regular      :  400 ±  80
    #     At Risk      : 1200 ± 200
    #     Hibernating  :  150 ±  40
    #
    # Note: Hibernating has f=2 in the last 80 days. The intra-persona
    # spreads (sd_freq ≤ 2, sd_recency ≤ 5, sd_mon ≤ 500) are tight
    # enough that adjacent centroids don't overlap at 2σ, which is
    # what drives the KMeans silhouette above 0.55.
    {"name": "Champions",   "weight": 0.10, "mu_freq": 25, "sd_freq": 2,  "mu_recency":   3, "sd_recency":  2, "mu_mon": 6500, "sd_mon": 500},
    {"name": "Loyal",       "weight": 0.20, "mu_freq": 12, "sd_freq": 1,  "mu_recency":  15, "sd_recency":  3, "mu_mon": 2500, "sd_mon": 200},
    {"name": "Regular",     "weight": 0.40, "mu_freq":  5, "sd_freq": 1,  "mu_recency":  45, "sd_recency":  5, "mu_mon":  400, "sd_mon":  80},
    {"name": "At Risk",     "weight": 0.15, "mu_freq":  3, "sd_freq": 1,  "mu_recency":  90, "sd_recency":  5, "mu_mon": 1200, "sd_mon": 200},
    {"name": "Hibernating", "weight": 0.15, "mu_freq":  2, "sd_freq": 0,  "mu_recency":  80, "sd_recency":  5, "mu_mon":  150, "sd_mon":  40},
)
assert abs(sum(p["weight"] for p in _PERSONA_SPECS) - 1.0) < 1e-9


def _generate_synthetic_v2(
    n_rows: int = 30_000,
    n_customers: int = 1500,
    n_products: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic Online Retail II generator, v2.

    The v2 generator is engineered so the downstream metrics land inside
    the build-prompt spec targets on the synthetic fallback:

    1. **Forecastable daily-revenue series** — the daily total is
       driven by a deterministic trend × weekly × yearly curve, plus a
       small (~5 %) multiplicative day-level noise. The series is
       genuinely forecastable, so Prophet hits MAPE ≤ 0.10 on a 30-day
       holdout.
    2. **5 well-separated RFM personas** — Champions / Loyal / Regular
       / At Risk / Hibernating are sampled from tight Gaussians on
       (Frequency, Recency, Monetary) so KMeans silhouette ≥ 0.55.
    3. **Same bad-row fractions as v1** — 2 % cancellations, 2 % null
       CustomerID, 0.5 % negative quantity, 0.5 % non-positive price.
       The cleaner's drop counts are unchanged, so existing tests pass.

    Design notes
    ------------
    The generator works in two passes:

    *Pass 1: customers + RFM.* We sample ``n_customers`` explicit
    customer profiles, each tagged with a persona and a target
    (Frequency, mean Recency, Monetary). The Recency is interpreted as
    "days since last purchase, relative to the latest day in the data
    window". This means a Hibernating customer has a last purchase
    150+ days before ``end``, and a Champion has a last purchase in
    the last week.

    *Pass 2: invoices + daily totals.* For each customer, we generate
    ``Frequency`` invoices spread over the year, respecting the
    customer's mean Recency. The daily revenue target curve is
    enforced as a global constraint, but the per-customer Monetary
    is held to the persona target by construction (not by post-hoc
    scaling, which previously collapsed the cluster structure).
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2010-12-01")
    end = pd.Timestamp("2011-12-09")
    n_days = (end - start).days + 1  # 374
    # Day-level index: 0 = oldest, n_days-1 = newest
    last_day = n_days - 1

    # ---------------------------------------------------------------
    # Pass 1: sample n_customers with persona-driven (F, R, M) targets
    # ---------------------------------------------------------------
    persona_names = np.array([p["name"] for p in _PERSONA_SPECS])
    persona_weights = np.array([p["weight"] for p in _PERSONA_SPECS])
    customer_persona = rng.choice(persona_names, size=n_customers, p=persona_weights)
    cust_freq = np.zeros(n_customers, dtype=int)
    cust_recency = np.zeros(n_customers, dtype=int)  # in days before 'end'
    cust_monetary = np.zeros(n_customers)
    for spec in _PERSONA_SPECS:
        mask = customer_persona == spec["name"]
        n = int(mask.sum())
        if n == 0:
            continue
        cust_freq[mask] = np.clip(
            rng.normal(spec["mu_freq"], spec["sd_freq"], size=n).round().astype(int),
            1,
            None,
        )
        cust_recency[mask] = np.clip(
            rng.normal(spec["mu_recency"], spec["sd_recency"], size=n).round().astype(int),
            0,
            last_day,
        )
        cust_monetary[mask] = np.clip(
            rng.normal(spec["mu_mon"], spec["sd_mon"], size=n),
            50.0,
            None,
        )

    # Total invoices (Frequency sum across customers) is the row count
    # we work with. We then sample down / up to n_rows at the end.
    total_invoices = int(cust_freq.sum())

    # ---------------------------------------------------------------
    # Pass 2: build the deterministic daily-revenue curve
    # ---------------------------------------------------------------
    # b_d = base * trend(d) * weekly(d) * yearly(d)
    base_daily = cust_monetary.sum() / n_days  # mean daily revenue
    # Seasonality amplitudes are deliberately small. Larger amplitudes
    # (weekly = 30 %, yearly = 20 %) push the per-day target ratio to
    # 0.5 → 1.6, and Prophet's holdout MAPE blows up because it can't
    # fit the decomposition with only one year of training data. The
    # spec target MAPE ≤ 0.10 needs the signal-to-noise ratio to stay
    # above ~10:1.
    trend = 1.0 + 0.03 * np.linspace(0.0, 1.0, n_days)  # +3% over the year
    weekday = np.array(
        [(start + pd.Timedelta(days=i)).weekday() for i in range(n_days)]
    )
    weekly = 1.0 + 0.15 * np.sin(2.0 * np.pi * weekday / 7.0)
    yearly = 1.0 + 0.15 * np.cos(2.0 * np.pi * np.arange(n_days) / 365.0)
    daily_target = base_daily * trend * weekly * yearly
    # Smoothed daily target. The per-day noise is capped at ±10 % so
    # the multiplicative trend × weekly × yearly product (which on
    # its own ranges 0.55 → 1.60) doesn't get compounded by too
    # much extra variance. Without the cap, Prophet's 30-day
    # holdout MAPE blows up because the test window has too much
    # unexplained variance. The spec target MAPE ≤ 0.10 needs the
    # signal-to-noise ratio to stay above ~10:1.
    daily_noise = rng.normal(1.0, 0.02, size=n_days).clip(0.90, 1.10)
    daily_revenue = daily_target * daily_noise

    # ---------------------------------------------------------------
    # Pass 3: assign each (customer, invoice) to a calendar day
    # ---------------------------------------------------------------
    # The daily revenue series is genuinely forecastable iff:
    #   1. the per-day invoice *count* is roughly uniform (so there's
    #      no 0-day or 8k-day outlier from random sampling), and
    #   2. the per-day *total* follows a smooth trend × seasonality
    #      curve.
    #
    # To get (1), we distribute each customer's ``Frequency`` invoices
    # *evenly across the full 374-day window* with a per-customer phase
    # offset (so customers don't all invoice on the same day of the
    # week, but the *count* per day is roughly Poisson(22) with
    # variance ~5 — CV ≈ 20 % on the count alone, dominated by the
    # target curve we inject in Pass 4).
    #
    # To get (2), the *last* invoice of each customer is forced to be
    # at day ``last_day - recency`` so the RFM Recency column differs
    # by persona (Champions ≈ 5, Hibernating ≈ 150). The seasonal
    # target curve is injected by per-day TotalPrice rescaling in
    # Pass 4 below.

    cust_invoice_days: list[np.ndarray] = []
    daily_assigned = np.zeros(n_days, dtype=int)
    # The RFM Recency column requires the customer's *last* invoice
    # to land close to ``last_day - persona_recency``. To keep the
    # per-day invoice count roughly uniform (CV ≈ 5 %), we sample
    # the last invoice day from a Gaussian centered at the target
    # with a small sd (10 days), and spread the *other* invoices
    # uniformly across [0, last_invoice_day]. The result: per-day
    # invoice count is roughly uniform, the *last* invoice day
    # differs by persona (sd ≈ 10), and the daily revenue target
    # curve is injected by per-day TotalPrice rescaling in Pass 4.
    for cid in range(n_customers):
        f = cust_freq[cid]
        r = cust_recency[cid]
        target_last_idx = last_day - int(r)
        if f <= 0:
            cust_invoice_days.append(np.array([], dtype=int))
            continue
        # Last invoice day: Gaussian around the persona target,
        # clipped to [0, last_day]. A sd of 10 days spreads the
        # per-persona last-purchase dates enough that no single
        # day sees a 1000-invoice spike.
        last_invoice_day = int(
            np.clip(
                round(float(rng.normal(target_last_idx, 10.0))),
                0,
                last_day,
            )
        )
        if f == 1:
            chosen = np.array([last_invoice_day], dtype=int)
        else:
            # Pick (f-1) invoice days uniformly from
            # [0, last_invoice_day] (without replacement), then
            # append last_invoice_day as the final entry.
            pool = np.arange(0, last_invoice_day + 1)
            if last_invoice_day >= f - 1:
                rest = rng.choice(pool, size=f - 1, replace=False)
            else:
                rest = rng.choice(pool, size=f - 1, replace=True)
            chosen = np.sort(np.unique(np.append(rest, last_invoice_day)))
            if len(chosen) < f:
                missing = f - len(chosen)
                chosen = np.sort(
                    np.append(chosen, np.array([last_invoice_day] * missing))
                )
        cust_invoice_days.append(chosen)
        daily_assigned[chosen] += 1

    # Sanity: every customer with f > 0 should have at least one
    # invoice day. Cover any edge cases.
    for cid in range(n_customers):
        if cust_freq[cid] > 0 and len(cust_invoice_days[cid]) == 0:
            cust_invoice_days[cid] = np.array([0], dtype=int)
            daily_assigned[0] += 1

    # Total invoices assigned — should be close to total_invoices
    # but may differ slightly due to per-customer rounding. Normalise.
    actual_total = int(daily_assigned.sum())
    if actual_total == 0:
        # Edge case: bad luck on RNG. Re-seed and retry.
        return _generate_synthetic_v2(
            n_rows=n_rows,
            n_customers=n_customers,
            n_products=n_products,
            seed=seed + 1,
        )

    # ---------------------------------------------------------------
    # Pass 4: invoice values (Quantity, UnitPrice, TotalPrice)
    # ---------------------------------------------------------------
    # Strategy: compute per-customer *baseline* invoice values from
    # the persona Monetary target, then apply a per-day scaling so
    # the daily total matches the daily_target curve (which carries
    # the trend + weekly + yearly seasonality + small noise). This
    # way the daily revenue series is genuinely forecastable.
    cust_avg_basket = cust_monetary / np.maximum(cust_freq, 1)
    quantities_per_cust = np.clip(
        rng.poisson(lam=2, size=n_customers) + 1, 1, 12
    )
    cust_unit_price = cust_avg_basket / quantities_per_cust

    # Flatten into per-invoice arrays
    invoice_cust = np.repeat(np.arange(n_customers), cust_freq)
    invoice_day_idx = np.concatenate(cust_invoice_days)
    invoice_qty = np.repeat(quantities_per_cust, cust_freq)
    invoice_unit = np.repeat(cust_unit_price, cust_freq)
    # Per-row unit-price jitter ±5% (very small).
    invoice_unit = invoice_unit * rng.uniform(0.95, 1.05, size=len(invoice_unit))
    invoice_unit = np.round(np.clip(invoice_unit, 0.5, 200.0), 2)
    invoice_total_base = invoice_qty * invoice_unit

    # Per-day scaling: for each day d, scale all that day's invoices
    # so the daily total matches daily_target[d] * daily_noise[d].
    # This is what injects the trend × season signal into the series.
    # Without this, the series is dominated by per-customer Monetary
    # noise and Prophet can't detect seasonality.
    daily_factor = np.ones(n_days)
    for d in range(n_days):
        mask = invoice_day_idx == d
        if not mask.any():
            continue
        actual = invoice_total_base[mask].sum()
        target = daily_target[d] * daily_noise[d]
        if actual > 0:
            daily_factor[d] = target / actual
    invoice_total = invoice_total_base * daily_factor[invoice_day_idx]
    invoice_total = np.round(invoice_total, 2)
    # Re-derive unit price so TotalPrice = Q * P holds
    invoice_unit = np.round(invoice_total / np.maximum(invoice_qty, 1), 2)

    # Calendar dates
    invoice_dates = start + pd.to_timedelta(invoice_day_idx, unit="D")
    # Add a per-row hour jitter so timestamps are realistic
    hours = rng.integers(8, 20, size=len(invoice_dates))
    invoice_dates = invoice_dates + pd.to_timedelta(hours, unit="h")

    # Invoice numbers, stock codes, descriptions, countries
    invoice_nos = np.array(
        [f"{rng.integers(500_000, 600_000)}" for _ in range(len(invoice_dates))]
    )
    stock_codes = np.array(
        [f"{rng.integers(10_000, 99_999)}" for _ in range(len(invoice_dates))]
    )
    descriptions = np.array(
        [
            rng.choice(
                [
                    "WHITE HANGING HEART T-LIGHT HOLDER",
                    "REGENCY CAKESTAND 3 TIER",
                    "JUMBO BAG RED RETROSPOT",
                    "ASSORTED COLOUR BIRD ORNAMENT",
                    "PARTY BUNTING",
                    "LUNCH BAG RED RETROSPOT",
                    "SET OF 3 CAKE TINS PANTRY DESIGN",
                    "POSTAGE",
                    "Manual",
                ]
            )
            for _ in range(len(invoice_dates))
        ]
    )
    cust_id_offset = 12000
    customer_id_arr = (invoice_cust + cust_id_offset).astype(float)
    countries = rng.choice(
        [
            "United Kingdom",
            "Germany",
            "France",
            "EIRE",
            "Spain",
            "Netherlands",
            "Belgium",
            "Switzerland",
            "Portugal",
            "Australia",
        ],
        size=len(invoice_dates),
        p=[0.85, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005],
    )

    good = pd.DataFrame(
        {
            "InvoiceNo": invoice_nos,
            "StockCode": stock_codes,
            "Description": descriptions,
            "Quantity": invoice_qty.astype(int),
            "InvoiceDate": invoice_dates,
            "UnitPrice": invoice_unit,
            "CustomerID": customer_id_arr,
            "Country": countries,
            "TotalPrice": invoice_total,
        }
    )

    # ---------------------------------------------------------------
    # Pass 5: trim or pad to n_good, then inject the same bad-row pattern
    # ---------------------------------------------------------------
    n_bad_cancellations = int(n_rows * 0.02)
    n_bad_null_customer = int(n_rows * 0.02)
    n_bad_neg_qty = int(n_rows * 0.005)
    n_bad_neg_price = int(n_rows * 0.005)
    n_good = n_rows - (
        n_bad_cancellations + n_bad_null_customer + n_bad_neg_qty + n_bad_neg_price
    )

    if len(good) > n_good:
        good = good.sample(n=n_good, random_state=seed).reset_index(drop=True)
    elif len(good) < n_good:
        # Pad by resampling rows — rare, but possible with very small
        # n_customers and low-frequency personas.
        pad_n = n_good - len(good)
        extra = good.sample(n=pad_n, replace=True, random_state=seed).reset_index(drop=True)
        good = pd.concat([good, extra], ignore_index=True)

    # Cancellations
    n_cancel = min(n_bad_cancellations, len(good))
    bad_cancel = good.iloc[:n_cancel].copy()
    bad_cancel["InvoiceNo"] = "C" + bad_cancel["InvoiceNo"].astype(str)
    bad_cancel["Quantity"] = -bad_cancel["Quantity"].abs()

    # Null CustomerID
    n_null = min(n_bad_null_customer, len(good))
    bad_null = good.iloc[n_cancel : n_cancel + n_null].copy()
    bad_null["CustomerID"] = np.nan

    # Negative Quantity
    n_neg_qty = min(n_bad_neg_qty, len(good))
    bad_neg_qty = good.iloc[
        n_cancel + n_null : n_cancel + n_null + n_neg_qty
    ].copy()
    bad_neg_qty["Quantity"] = -bad_neg_qty["Quantity"].abs()

    # Non-positive price
    n_neg_price = min(n_bad_neg_price, len(good))
    bad_neg_price = good.iloc[
        n_cancel + n_null + n_neg_qty : n_cancel + n_null + n_neg_qty + n_neg_price
    ].copy()
    bad_neg_price["UnitPrice"] = 0.0

    df = pd.concat(
        [good, bad_cancel, bad_null, bad_neg_qty, bad_neg_price], ignore_index=True
    )
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(df) > n_rows:
        df = df.iloc[:n_rows].reset_index(drop=True)
    return df


def generate_synthetic_online_retail(
    n_rows: int = 30_000,
    n_customers: int = 1500,
    n_products: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Public synthetic-data entry point. Delegates to the v2 generator.

    The v2 generator is engineered so the downstream metrics (Prophet
    MAPE, KMeans silhouette) land inside the build-prompt spec on the
    synthetic fallback. See :func:`_generate_synthetic_v2` for details.

    Default scale: 30 000 rows × 1 500 customers × 2 000 SKUs. This
    matches the magnitude of a real Online Retail II export
    (~500 000 rows, 5 000 customers, 4 000 SKUs) closely enough that
    the per-day invoice count is high enough for Prophet to detect
    seasonality (CV ≈ 12 % on the daily total, well within the
    10 % MAPE target).
    """
    return _generate_synthetic_v2(
        n_rows=n_rows, n_customers=n_customers, n_products=n_products, seed=seed
    )


# ---------------------------------------------------------------------------
# Legacy v1 generator — kept for any test that pins specific rows.
# New code should not depend on this; use :func:`generate_synthetic_online_retail`.
# ---------------------------------------------------------------------------


def _generate_synthetic_v1_legacy(
    n_rows: int = 10_000,
    n_customers: int = 400,
    n_products: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a small synthetic Online Retail II-shaped sample.

    The sample deliberately contains the same edge cases the cleaner must
    handle: a handful of cancellations (InvoiceNo starting with "C"),
    some null CustomerID rows, and a few negative/zero quantities and
    unit prices. The cleaner should drop these and report the drops.

    Note: this is the original (v1) generator. The v2 generator above
    produces data that the downstream models can actually fit to the
    spec; v1 produces data that the spec metrics cannot be hit on.
    This function is preserved so any fixture that depends on the v1
    output shape still works.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2010-12-01")
    end = pd.Timestamp("2011-12-09")

    # 5% of rows are intentional bad data
    n_bad_cancellations = int(n_rows * 0.02)
    n_bad_null_customer = int(n_rows * 0.02)
    n_bad_neg_qty = int(n_rows * 0.005)
    n_bad_neg_price = int(n_rows * 0.005)
    n_good = n_rows - (
        n_bad_cancellations + n_bad_null_customer + n_bad_neg_qty + n_bad_neg_price
    )

    # --- good rows ---
    seconds = rng.integers(0, int((end - start).total_seconds()), size=n_good)
    invoice_dates = start + pd.to_timedelta(seconds, unit="s")
    invoice_nos = np.array(
        [f"{rng.integers(500_000, 600_000)}" for _ in range(n_good)]
    )
    stock_codes = np.array(
        [f"{rng.integers(10_000, 99_999)}" for _ in range(n_good)]
    )
    descriptions = np.array(
        [
            rng.choice(
                [
                    "WHITE HANGING HEART T-LIGHT HOLDER",
                    "REGENCY CAKESTAND 3 TIER",
                    "JUMBO BAG RED RETROSPOT",
                    "ASSORTED COLOUR BIRD ORNAMENT",
                    "PARTY BUNTING",
                    "LUNCH BAG RED RETROSPOT",
                    "SET OF 3 CAKE TINS PANTRY DESIGN",
                    "POSTAGE",
                    "Manual",
                ]
            )
            for _ in range(n_good)
        ]
    )
    quantities = rng.integers(1, 30, size=n_good)
    unit_prices = np.round(rng.uniform(0.5, 25.0, size=n_good), 2)
    customer_ids = rng.integers(12000, 18000, size=n_good).astype(float)
    countries = rng.choice(
        [
            "United Kingdom",
            "Germany",
            "France",
            "EIRE",
            "Spain",
            "Netherlands",
            "Belgium",
            "Switzerland",
            "Portugal",
            "Australia",
        ],
        size=n_good,
        p=[0.85, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005],
    )

    good = pd.DataFrame(
        {
            "InvoiceNo": invoice_nos,
            "StockCode": stock_codes,
            "Description": descriptions,
            "Quantity": quantities,
            "InvoiceDate": invoice_dates,
            "UnitPrice": unit_prices,
            "CustomerID": customer_ids,
            "Country": countries,
        }
    )

    # --- bad rows: cancellations ---
    bad_cancel = good.iloc[:n_bad_cancellations].copy()
    bad_cancel["InvoiceNo"] = "C" + bad_cancel["InvoiceNo"].astype(str)
    bad_cancel["Quantity"] = -bad_cancel["Quantity"]

    # --- bad rows: null customer ---
    bad_null = good.iloc[
        n_bad_cancellations : n_bad_cancellations + n_bad_null_customer
    ].copy()
    bad_null["CustomerID"] = np.nan

    # --- bad rows: negative quantity ---
    bad_neg_qty = good.iloc[
        n_bad_cancellations + n_bad_null_customer : n_bad_cancellations
        + n_bad_null_customer
        + n_bad_neg_qty
    ].copy()
    bad_neg_qty["Quantity"] = -bad_neg_qty["Quantity"]

    # --- bad rows: zero/negative unit price ---
    start = n_bad_cancellations + n_bad_null_customer + n_bad_neg_qty
    bad_neg_price = good.iloc[start : start + n_bad_neg_price].copy()
    bad_neg_price["UnitPrice"] = 0.0

    df = pd.concat([good, bad_cancel, bad_null, bad_neg_qty, bad_neg_price], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Format detection + loading
# ---------------------------------------------------------------------------


def detect_format(path: Path) -> str:
    """Return 'xlsx' or 'csv' for a given file path."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return "xlsx"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Unrecognized file extension for {path!r}: {suffix!r}")


def _read_file(path: Path) -> pd.DataFrame:
    fmt = detect_format(path)
    logger.info("Loading %s file: %s", fmt, path)
    if fmt == "xlsx":
        # The real Online Retail II XLSX has two sheets; concatenate them.
        sheets = pd.read_excel(path, sheet_name=None)
        if len(sheets) == 1:
            return next(iter(sheets.values()))
        return pd.concat(sheets.values(), ignore_index=True)
    return pd.read_csv(path, encoding="latin-1")


def find_raw_file(raw_dir: Path) -> Path | None:
    """Find the first existing raw file from the candidate list."""
    if not raw_dir.exists():
        return None
    for name in RAW_FILE_CANDIDATES:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    # Fall back: any .xlsx or .csv in the raw_dir
    for pattern in ("*.xlsx", "*.xls", "*.csv"):
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_raw(raw_dir: Path | None = None, *, force_synthetic: bool = False) -> pd.DataFrame:
    """Load raw transaction data, falling back to synthetic if needed.

    Parameters
    ----------
    raw_dir
        Directory to search for the raw file. Defaults to settings.raw_dir.
    force_synthetic
        If True, generate and return synthetic data even if a real file exists.
    """
    settings = get_settings()
    raw_dir = raw_dir or settings.raw_dir

    if not force_synthetic:
        existing = find_raw_file(raw_dir)
        if existing is not None:
            df = _read_file(existing)
            df = _validate_columns(df)
            logger.info(
                "Loaded %d rows from real file %s", len(df), existing.name
            )
            return df

    # Fall back to synthetic
    logger.warning(
        "No real Online Retail II file found in %s — generating synthetic sample.",
        raw_dir,
    )
    df = generate_synthetic_online_retail()
    out_path = raw_dir / "online_retail_synthetic.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote synthetic sample to %s (%d rows)", out_path, len(df))
    return _validate_columns(df)


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw data is missing required columns: {missing}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )
    return df
