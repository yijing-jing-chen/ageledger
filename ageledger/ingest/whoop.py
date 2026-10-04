"""WHOOP data export (zip, extracted directory, or a bare physiological_cycles.csv) -> Dataset.

Main table: physiological_cycles.csv (one row per physiological cycle). sleeps.csv only fills sleep
fields the cycles file left empty, with nap rows excluded. workouts.csv / journal_entries.csv are not
needed for any canonical DAILY column and are ignored.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

import numpy as np
import pandas as pd

from ..schema import Dataset, Profile
from ._common import dedup_dates, find_col, finalize, iso, norm, to_num, wallclock

# canonical column -> header patterns (exact first, then substring), excludes
NUMERIC_MAP: list[tuple[str, list[str], list[str]]] = [
    ("rhr", ["resting heart rate (bpm)", "resting heart rate"], []),
    ("hrv_rmssd", ["heart rate variability (ms)", "heart rate variability", "hrv"], []),
    ("skin_temp_c", ["skin temp (celsius)", "skin temp"], []),
    ("spo2", ["blood oxygen %", "blood oxygen", "spo2"], []),
    ("strain", ["day strain"], []),
    ("resp_rate", ["respiratory rate (rpm)", "respiratory rate"], []),
    ("sleep_min", ["asleep duration (min)", "asleep duration"], []),
    ("sleep_deep_min", ["deep (sws) duration (min)", "deep (sws) duration", "sws duration", "deep sleep duration"], []),
    ("sleep_rem_min", ["rem duration (min)", "rem duration", "rem sleep duration"], []),
    ("sleep_efficiency", ["sleep efficiency %", "sleep efficiency"], []),
    ("steps", ["steps", "step count"], []),
    ("vo2max", ["vo2 max", "vo2max", "vo2"], []),
    ("recovery_pct", ["recovery score %", "recovery score"], []),
    ("sleep_need_min", ["sleep need (min)", "sleep need"], []),
    ("sleep_debt_min", ["sleep debt (min)", "sleep debt"], []),
    ("sleep_consistency_pct", ["sleep consistency %", "sleep consistency"], []),
]
SLEEP_COLS = ["sleep_min", "sleep_deep_min", "sleep_rem_min", "sleep_efficiency", "resp_rate",
              "sleep_onset", "sleep_offset", "sleep_need_min", "sleep_debt_min", "sleep_consistency_pct"]
WANTED = {"physiological_cycles.csv", "sleeps.csv"}


def _read_tables(path: str) -> dict[str, pd.DataFrame]:
    p = Path(path)
    out: dict[str, pd.DataFrame] = {}
    if p.is_dir():
        for f in sorted(p.rglob("*.csv")):
            if f.name.lower() in WANTED and f.name.lower() not in out:
                out[f.name.lower()] = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    elif zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as zf:
            for name in zf.namelist():
                base = PurePosixPath(name).name.lower()
                if base in WANTED and base not in out and not name.startswith("__MACOSX"):
                    with zf.open(name) as fh:
                        out[base] = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), dtype=str)
    else:  # a bare csv: treat as the cycles table
        out["physiological_cycles.csv"] = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    if "physiological_cycles.csv" not in out:
        raise ValueError(f"{path}: no physiological_cycles.csv found — not a WHOOP export?")
    return out


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Map a WHOOP table (cycles or sleeps) to canonical columns + 'date'."""
    out = pd.DataFrame(index=df.index)
    for canon, pats, excl in NUMERIC_MAP:
        col = find_col(df.columns, pats, excl)
        if col is None:
            continue
        v = to_num(df[col])
        if canon == "skin_temp_c" and ("fahrenheit" in norm(col) or "(f)" in norm(col)):
            v = (v - 32.0) * 5.0 / 9.0
        if canon == "sleep_efficiency" and v.dropna().le(1.0).all() and v.notna().any():
            v = v * 100.0
        out[canon] = v

    onset_c = find_col(df.columns, ["sleep onset"])
    wake_c = find_col(df.columns, ["wake onset"])
    start_c = find_col(df.columns, ["cycle start time", "cycle start"])
    onset = wallclock(df[onset_c]) if onset_c else pd.Series(pd.NaT, index=df.index)
    wake = wallclock(df[wake_c]) if wake_c else pd.Series(pd.NaT, index=df.index)
    if onset_c:
        out["sleep_onset"] = iso(onset)
    if wake_c:
        out["sleep_offset"] = iso(wake)

    # date = local calendar day the cycle's wake onset falls on.
    # ASSUMPTION: cycles with no recorded sleep (no wake onset) fall back to (cycle start + 12h).date —
    # a cycle starts at sleep onset, typically the previous evening.
    date = wake.dt.normalize()
    if start_c:
        start = wallclock(df[start_c])
        date = date.fillna((start + pd.Timedelta(hours=12)).dt.normalize())
    out["date"] = date.dt.strftime("%Y-%m-%d")
    return out[out["date"].notna()]


def load(path: str) -> Dataset:
    tables = _read_tables(path)
    cyc = _canonical(tables["physiological_cycles.csv"])
    # two cycles on one day (e.g. a split sleep) -> keep the one with the longer sleep, fill gaps from the other
    daily = dedup_dates(cyc, prefer="sleep_min")

    notes = ["WHOOP export carries no age/sex/height/weight — profile must be supplied by the user"]
    sl = tables.get("sleeps.csv")
    if sl is not None and len(sl):
        nap_c = find_col(sl.columns, ["nap"])
        if nap_c is not None:
            is_nap = sl[nap_c].astype("string").str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])
            sl = sl[~is_nap.fillna(False).astype(bool)]
        if len(sl):
            sc = _canonical(sl)
            sc = sc[[c for c in ["date"] + SLEEP_COLS if c in sc.columns]]
            sc = dedup_dates(sc, prefer="sleep_min")
            daily = daily.set_index("date").combine_first(sc.set_index("date")).reset_index()

    return finalize(daily, None, Profile(notes=notes), "whoop")
