"""CosinorAge — a dependency-light (numpy/pandas) reimplementation of the published closed-form model.

Method:  Shim, Fleisch & Barata (2024), "CosinorAge: unified 24-hour circadian rhythm-based
         biological age", npj Digital Medicine. DOI 10.1038/s41746-024-01111-x
Reference implementation (Apache-2.0, (C) 2025 ETH Zurich): https://github.com/ADAMMA-CDHI-ETH-Zurich/CosinorAge
         pip `cosinorage` 1.0.8. We do not depend on it (it pulls claid, skdh, CosinorPy, statsmodels, seaborn).

What is matched to the reference, and where:
  * Cosinor fit — cosinorage/features/utils/cosinor_analysis.py::cosinor_multiday, which calls
    CosinorPy 3.1 cosinor1.fit_cosinor -> test_cosinor_single:
        OLS  y ~ 1 + rrr + sss,  rrr = cos(2*pi*t/1440), sss = sin(2*pi*t/1440),  t = 1..N (minutes)
        model  y = MESOR + amp * cos(2*pi*t/1440 + acr)
        amp = sqrt(b_r^2 + b_s^2);  acr = -atan2(b_s, b_r)  (CosinorPy amp_acr(corrected=True), then
        cosinor.project_acr -> (-pi, pi]);  then cosinor_multiday: `if acrophase > 0: acrophase -= 2*np.pi`
        -> acrophase in (-2*pi, 0].  Peak clock time = -acr/(2*pi)*24 h.
    The reference's t starts at 1 at the first 00:00 minute, i.e. t = minute_of_day + 1 for every row of a
    midnight-aligned series. We use exactly that (cos is 1440-periodic, so the day index drops out).
  * Closed form + coefficients — cosinorage/bioages/cosinorage.py (copied verbatim below).
  * Day filtering — cosinorage/datahandlers/utils/generic.py::filter_generic_data: drop the first day if it
    does not start at 00:00 and the last day if it does not end at 23:59; drop days under 50% coverage
    (`required_daily_coverage` default 0.5); keep the longest run of consecutive days.

Deliberate departures (see `PREPROCESSING_DEPARTURES`): NaN minutes are DROPPED, not filled with 0 and not
linearly interpolated as the reference does (CONTRACT law 1: never fabricate). OLS on the remaining minutes
uses each row's true minute-of-day, so gaps do not shift phase.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .schema import Dataset

# --- Model constants -------------------------------------------------------------------------------
# Copied verbatim from cosinorage/bioages/cosinorage.py (cosinorage 1.0.8),
# Copyright (C) 2025 ETH Zurich, licensed under the Apache License, Version 2.0.
model_params_generic = {
    "shape": 0.01462774,
    "rate": -13.36715309,
    "mesor": -0.03204933,
    "amp1": -0.01971357,
    "phi1": -0.01664718,
    "age": 0.10033692,
}

model_params_female = {
    "shape": 0.01294402,
    "rate": -13.28530410,
    "mesor": -0.02569062,
    "amp1": -0.02170987,
    "phi1": -0.13191562,
    "age": 0.08840283,
}

model_params_male = {
    "shape": 0.013878454,
    "rate": -13.016951633,
    "mesor": -0.023988922,
    "amp1": -0.030620390,
    "phi1": 0.008960155,
    "age": 0.101726103,
}

m_n = -1.405276
m_d = 0.01462774
BA_n = -0.01447851
BA_d = 0.112165
BA_i = 133.5989
# --- end of copied constants -----------------------------------------------------------------------

MINUTES_PER_DAY = 1440
MIN_DAYS = 3                    # our assumption: the reference only requires >= 1 day
REQUIRED_DAILY_COVERAGE = 0.5   # reference default `required_daily_coverage`
WINDOW_DAYS = 90                # ledger convention (CONTRACT.md); the reference uses every kept day

ACCEL_SOURCES = ("edge", "galaxy", "nhanes", "synthetic")   # mirrors schema.validate()

CITATION = {
    "short": "Shim et al. 2024, npj Digit Med",
    "doi": "10.1038/s41746-024-01111-x",
    "url": "https://doi.org/10.1038/s41746-024-01111-x",
    "implementation": "numpy reimplementation of cosinorage 1.0.8 (Apache-2.0, ETH Zurich): "
                      "https://github.com/ADAMMA-CDHI-ETH-Zurich/CosinorAge",
}

PREPROCESSING_DEPARTURES = [
    "NaN minutes dropped; reference fills NaN with 0 on read and linearly interpolates gaps after resampling",
    "no resampling: input is assumed already minute-level (ts floored to the minute, one row per minute)",
    "the reference drops the last day of minute-level data even when complete (its 23:59:59 end check); "
    "we keep a last day that reaches 23:59",
    "no wear detection (the reference does none for ENMO input either: wear = -1)",
]

NO_ENMO_REASON = "CosinorAge requires accelerometer ENMO; this export has none"


class CosinorFitError(ValueError):
    """Not enough usable ENMO for the fit."""


# --------------------------------------------------------------------------------------------------
def _local_naive(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)        # keep local wall-clock time, as the reference does
    return ts.dt.floor("min")


def _filter_days(df: pd.DataFrame) -> pd.DataFrame:
    """Reference filter_generic_data, minute-level: partial first/last day, <50% days, longest run."""
    if df.empty:
        return df
    day = df["ts"].dt.normalize()
    first, last = day.min(), day.max()
    if df.loc[day == first, "ts"].min() != first:
        df = df.loc[day != first]
    day = df["ts"].dt.normalize()
    if not df.empty and df.loc[day == last, "ts"].max() != last + pd.Timedelta(minutes=MINUTES_PER_DAY - 1):
        df = df.loc[day != last]
    if df.empty:
        return df

    day = df["ts"].dt.normalize()
    counts = day.value_counts()
    keep = counts[counts >= REQUIRED_DAILY_COVERAGE * MINUTES_PER_DAY].index
    df = df.loc[day.isin(keep)]
    if df.empty:
        return df

    days = np.array(sorted(df["ts"].dt.normalize().unique()))
    gaps = np.diff(days) != np.timedelta64(1, "D")
    run_id = np.concatenate([[0], np.cumsum(gaps)])
    best = np.bincount(run_id).argmax()            # first longest run, like largest_consecutive_sequence
    best_days = days[run_id == best]
    return df.loc[df["ts"].dt.normalize().isin(best_days)]


def fit_cosinor(minute: pd.DataFrame, window_days: Optional[int] = None) -> dict:
    """Multi-day cosinor fit on minute-level ENMO (milli-g), matching cosinorage's `cosinor_multiday`.

    `minute` needs columns ts (local, tz-aware or naive) and enmo_mg. Raises CosinorFitError when fewer
    than MIN_DAYS usable days remain. `window_days`, if given, keeps the most recent N kept days.
    """
    if "ts" not in minute.columns or "enmo_mg" not in minute.columns:
        raise CosinorFitError("minute table needs ts and enmo_mg")
    df = pd.DataFrame({"ts": _local_naive(minute["ts"]),
                       "enmo": pd.to_numeric(minute["enmo_mg"], errors="coerce")})
    df = df.dropna().drop_duplicates("ts").sort_values("ts")
    df = _filter_days(df)
    if window_days is not None and not df.empty:
        days = np.array(sorted(df["ts"].dt.normalize().unique()))[-window_days:]
        df = df.loc[df["ts"].dt.normalize().isin(days)]
    n_days = int(df["ts"].dt.normalize().nunique()) if not df.empty else 0
    if n_days < MIN_DAYS:
        raise CosinorFitError(f"need >= {MIN_DAYS} complete days of minute ENMO after filtering; have {n_days}")

    mod = (df["ts"].dt.hour * 60 + df["ts"].dt.minute).to_numpy(dtype=float)
    t = mod + 1.0                                    # reference: t = arange(1, N+1) from 00:00
    w = 2.0 * np.pi * t / MINUTES_PER_DAY
    y = df["enmo"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(t), np.cos(w), np.sin(w)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    mesor, b_r, b_s = (float(b) for b in beta)

    amplitude = math.hypot(b_r, b_s)
    acr = -math.atan2(b_s, b_r)                      # CosinorPy amp_acr(corrected=True), mod 2*pi
    acr %= 2.0 * math.pi                             # CosinorPy cosinor.project_acr ...
    if acr > math.pi:
        acr -= 2.0 * math.pi                         # ... -> (-pi, pi]
    if acr > 0:
        acr -= 2.0 * math.pi                         # cosinor_multiday: shift into (-2*pi, 0]
    acrophase_hours = -acr / (2.0 * math.pi) * 24.0  # == reference acrophase_time / 60

    resid = y - X @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "mesor": mesor,
        "amplitude": amplitude,
        "acrophase_rad": acr,
        "acrophase_hours": acrophase_hours,
        "n_days": n_days,
        "n_minutes": int(len(df)),
        "r2": r2,
    }


def cosinor_age(mesor: float, amplitude: float, acrophase_rad: float, age_years: float,
                sex: Optional[str]) -> dict:
    """Closed-form CosinorAge (cosinorage/bioages/cosinorage.py::__compute_cosinor_ages)."""
    s = (sex or "").strip().lower()
    if s == "female":
        coef, model = model_params_female, "female"
    elif s == "male":
        coef, model = model_params_male, "male"
    else:
        coef, model = model_params_generic, "generic"
    xb = (coef["mesor"] * mesor + coef["amp1"] * amplitude + coef["phi1"] * acrophase_rad
          + coef["age"] * age_years + coef["rate"])
    m_val = 1.0 - np.exp((m_n * np.exp(xb)) / m_d)
    value = float(np.log(BA_n * np.log(1.0 - m_val)) / BA_d + BA_i)
    return {"value": value, "advance_years": float(value - age_years), "model": model}


# --------------------------------------------------------------------------------------------------
def _try_estimate(ds: Dataset) -> tuple[Optional[dict], Optional[str]]:
    m = ds.minute
    if m is None or "enmo_mg" not in m.columns or m["enmo_mg"].notna().sum() == 0:
        return None, NO_ENMO_REASON            # steps / HR are never a stand-in for ENMO
    if ds.source not in ACCEL_SOURCES:
        return None, f"source '{ds.source}' has no accelerometer; enmo_mg refused"
    age = ds.profile.age_years if ds.profile is not None else None
    if age is None:
        return None, "CosinorAge needs chronological age; the export carries none"
    try:
        fit = fit_cosinor(m, window_days=WINDOW_DAYS)
    except CosinorFitError as e:
        return None, f"CosinorAge not fit: {e}"
    ca = cosinor_age(fit["mesor"], fit["amplitude"], fit["acrophase_rad"], float(age), ds.profile.sex)
    if not (np.isfinite(ca["value"]) and np.isfinite(ca["advance_years"])):
        return None, "CosinorAge closed form returned a non-finite value for these features"

    notes = [
        f"enmo from {ds.source} accelerometer, minute means in milli-g",
        "CosinorAge was trained on wrist ENMO (UK Biobank, NHANES); a different device or wear site may shift "
        "mesor/amplitude — for accuracy figures see the paper",
        f"{ca['model']} model coefficients; acrophase in radians in (-2pi, 0], peak at "
        f"{fit['acrophase_hours']:.2f} h local",
        f"fit on {fit['n_days']} complete days ({fit['n_minutes']} minutes), cosinor r2 {fit['r2']:.2f}",
    ]
    if ca["model"] == "generic":
        notes.append("sex unknown: generic (sex-pooled) model used")
    notes.append("departs from reference preprocessing: " + "; ".join(PREPROCESSING_DEPARTURES))

    return {
        "value": round(ca["value"], 2),
        "advance_years": round(ca["advance_years"], 2),
        "model": ca["model"],
        "window_days": fit["n_days"],
        "features": {
            "mesor": fit["mesor"],
            "amp": fit["amplitude"],
            "phi": fit["acrophase_rad"],
            "acrophase_hours": fit["acrophase_hours"],
            "r2": fit["r2"],
            "n_minutes": fit["n_minutes"],
        },
        "citation": dict(CITATION),
        "notes": notes,
    }, None


def estimate(ds: Dataset) -> Optional[dict]:
    """CONTRACT `bio_age.cosinor` dict, or None (caller lists cosinor_age under `withheld`)."""
    return _try_estimate(ds)[0]


def withheld_reason(ds: Dataset) -> Optional[str]:
    """Plain-language reason estimate() returned None; None when it did not."""
    return _try_estimate(ds)[1]
