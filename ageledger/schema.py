"""Canonical tables every ingester must produce. Pure data, no logic.

DAILY  — one row per calendar day (local). Missing = NaN/None, never 0.
MINUTE — one row per minute, only when the source has it (Apple Health, edge SQLite). Optional.
PROFILE — age/sex/height/weight if the export carries them; else None.

Column names are the contract between ingest/ and levers/. Do not rename without updating both.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

DAILY_COLUMNS: dict[str, str] = {
    "date": "YYYY-MM-DD, local calendar day the night ENDED on (sleep metrics) / the day happened on (activity)",
    "steps": "total steps that day (int)",
    "rhr": "resting heart rate, bpm (WHOOP: nightly RHR; Apple: HKQuantityTypeIdentifierRestingHeartRate)",
    "hrv_rmssd": "nightly rMSSD, ms (WHOOP 'Heart rate variability (ms)' is rMSSD). Apple gives SDNN → put in hrv_sdnn, NOT here",
    "hrv_sdnn": "SDNN, ms (Apple Health). Different metric from rMSSD; levers treat separately",
    "sleep_min": "total asleep minutes (not in-bed)",
    "sleep_onset": "ISO datetime local, start of main sleep",
    "sleep_offset": "ISO datetime local, end of main sleep",
    "sleep_efficiency": "0–100, asleep / in-bed",
    "sleep_deep_min": "minutes SWS",
    "sleep_rem_min": "minutes REM",
    "resp_rate": "nightly respiratory rate, breaths/min",
    "spo2": "nightly SpO2 %, if the device reports an absolute value (WHOOP 4.0+ does; edge does not)",
    "skin_temp_c": "nightly skin temp °C if absolute, else NaN",
    "vo2max": "ml/kg/min estimate as reported by the device",
    "strain": "device-specific daily load (WHOOP strain 0–21). Not comparable across devices",
    "recovery_pct": "WHOOP 'Recovery score %', 0–100, scored on waking (WHOOP bands: red <34, yellow 34–66, green ≥67). "
                    "Device-specific; Oura readiness is a different score and is NOT mapped here",
    "sleep_need_min": "WHOOP 'Sleep need (min)': the device's own nightly sleep-need estimate, minutes",
    "sleep_debt_min": "WHOOP 'Sleep debt (min)': the device's own accumulated sleep-debt estimate, minutes",
    "sleep_consistency_pct": "WHOOP 'Sleep consistency %', 0–100 (device's own regularity score; not the SRI)",
    "active_kcal": "active energy, kcal",
    "walking_speed_ms": "Apple HKQuantityTypeIdentifierWalkingSpeed daily mean, m/s (older-adult levers)",
    "walking_double_support_pct": "Apple WalkingDoubleSupportPercentage daily mean",
    "walking_step_length_m": "Apple WalkingStepLength daily mean",
    "walking_asymmetry_pct": "Apple WalkingAsymmetryPercentage daily mean",
    "source": "whoop | apple | oura | edge",
}

MINUTE_COLUMNS: dict[str, str] = {
    "ts": "timestamp, local tz-aware, floored to the minute",
    "steps": "steps in that minute (Apple). NaN when unknown",
    "hr": "mean HR in that minute, bpm. NaN when unknown",
    "enmo_mg": "Euclidean-norm-minus-one, milli-g, minute mean. Only from true accelerometer sources (edge 1 Hz ax/ay/az; Galaxy; NHANES). "
               "NEVER derived from steps — CosinorAge was fit on ENMO and refuses anything else",
    "source": "apple | edge",
}

@dataclass
class Profile:
    age_years: Optional[float] = None
    sex: Optional[str] = None            # 'male' | 'female' | None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    notes: list[str] = field(default_factory=list)

@dataclass
class Dataset:
    daily: pd.DataFrame                  # columns ⊆ DAILY_COLUMNS, 'date' is a str index-able column, sorted ascending
    minute: Optional[pd.DataFrame]       # columns ⊆ MINUTE_COLUMNS or None
    profile: Profile
    source: str
    coverage: dict = field(default_factory=dict)   # {"days": int, "first": str, "last": str, "columns_present": [...], "minute_days": int}

def empty_daily() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in DAILY_COLUMNS})

def validate(ds: Dataset) -> list[str]:
    """Return a list of contract violations (empty = ok). Ingesters call this before returning."""
    problems = []
    unknown = set(ds.daily.columns) - set(DAILY_COLUMNS)
    if unknown:
        problems.append(f"daily has unknown columns: {sorted(unknown)}")
    if "date" not in ds.daily.columns:
        problems.append("daily missing 'date'")
    elif ds.daily["date"].duplicated().any():
        problems.append("daily has duplicate dates")
    if ds.minute is not None:
        unknown = set(ds.minute.columns) - set(MINUTE_COLUMNS)
        if unknown:
            problems.append(f"minute has unknown columns: {sorted(unknown)}")
        if "enmo_mg" in ds.minute.columns and ds.source not in ("edge", "galaxy", "nhanes", "synthetic"):
            problems.append(f"source '{ds.source}' must not supply enmo_mg (no accelerometer)")
    return problems
