"""Shared helpers for ingesters: tolerant column lookup, wall-clock parsing, and the final
Dataset assembly (drop all-NaN columns, sort, coverage, validate)."""
from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..schema import DAILY_COLUMNS, MINUTE_COLUMNS, Dataset, Profile, validate

TEXT_DAILY = {"date", "sleep_onset", "sleep_offset", "source"}


def norm(name) -> str:
    return re.sub(r"\s+", " ", str(name).replace("﻿", "").strip().lower())


def find_col(columns: Iterable, patterns: Iterable[str], exclude: Iterable[str] = ()) -> Optional[str]:
    """Exact (case/space-insensitive) match on any pattern first, then substring match."""
    cols = list(columns)
    normed = [(c, norm(c)) for c in cols]
    pats = [norm(p) for p in patterns]
    excl = [norm(e) for e in exclude]
    for p in pats:
        for c, n in normed:
            if n == p:
                return c
    for p in pats:
        for c, n in normed:
            if p in n and not any(e in n for e in excl):
                return c
    return None


def to_num(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "iufb":
        return s.astype(float)
    t = s.astype("string").str.strip().str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(t, errors="coerce").astype(float)


_TS_RE = r"^\s*(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?"


def wallclock(s: pd.Series) -> pd.Series:
    """Parse timestamps to naive LOCAL wall-clock datetimes, ignoring any UTC offset suffix.
    Handles 'YYYY-MM-DD HH:MM[:SS]', ISO 'T' form, and trailing '+hh:mm' / ' -hhmm' / 'Z'."""
    s = pd.Series(s, copy=False)
    if s.dtype.kind == "M":
        return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
    st = s.astype("string")
    p = st.str.extract(_TS_RE)
    txt = p[0] + " " + p[1].str.zfill(2) + ":" + p[2] + ":" + p[3].fillna("00")
    out = pd.to_datetime(txt, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    miss = out.isna() & st.notna() & (st.str.strip() != "")
    if miss.any():  # odd formats: let pandas guess, keep wall clock
        def _one(x):
            try:
                ts = pd.Timestamp(x)
            except Exception:
                return pd.NaT
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        out = out.astype("datetime64[ns]")
        out[miss] = pd.to_datetime(st[miss].map(_one), errors="coerce")
    return out


def iso(ts: pd.Series) -> pd.Series:
    out = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.where(ts.notna(), None).astype(object)


def date_str(s: pd.Series) -> pd.Series:
    """Any date-ish column -> 'YYYY-MM-DD' strings (NaN when unparseable)."""
    d = wallclock(s)
    miss = d.isna()
    if miss.any():
        d = d.astype("datetime64[ns]")
        d[miss] = pd.to_datetime(pd.Series(s)[miss].astype("string").str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    return d.dt.strftime("%Y-%m-%d").where(d.notna(), None)


def finalize(daily: pd.DataFrame, minute: Optional[pd.DataFrame], profile: Profile, source: str) -> Dataset:
    daily = daily.copy() if daily is not None else pd.DataFrame({"date": []})
    if "date" not in daily.columns:
        daily = daily.reset_index()
    daily["date"] = date_str(daily["date"]) if len(daily) else pd.Series(dtype=object)
    daily = daily[daily["date"].notna()]
    value_cols = []
    for c in DAILY_COLUMNS:
        if c in ("date", "source") or c not in daily.columns:
            continue
        if c not in TEXT_DAILY:
            daily[c] = pd.to_numeric(daily[c], errors="coerce").astype(float)
        if daily[c].notna().any():
            value_cols.append(c)
    daily = daily[["date"] + value_cols]
    if value_cols:
        daily = daily[daily[value_cols].notna().any(axis=1)]
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["source"] = source

    if minute is not None and len(minute):
        mcols = [c for c in MINUTE_COLUMNS if c not in ("ts", "source") and c in minute.columns
                 and minute[c].notna().any()]
        if mcols:
            minute = minute[["ts"] + mcols].copy()
            for c in mcols:
                minute[c] = minute[c].astype(float)
            minute = minute[minute[mcols].notna().any(axis=1)].sort_values("ts").reset_index(drop=True)
            minute["source"] = source
        else:
            minute = None
    else:
        minute = None

    coverage = {
        "days": int(len(daily)),
        "first": daily["date"].iloc[0] if len(daily) else None,
        "last": daily["date"].iloc[-1] if len(daily) else None,
        "columns_present": value_cols,
        "minute_days": int(minute["ts"].dt.date.nunique()) if minute is not None else 0,
    }
    ds = Dataset(daily=daily, minute=minute, profile=profile, source=source, coverage=coverage)
    problems = validate(ds)
    if problems:
        raise ValueError(f"{source} ingest violates the schema contract: " + "; ".join(problems))
    return ds


def dedup_dates(df: pd.DataFrame, prefer: Optional[str] = None) -> pd.DataFrame:
    """One row per date: if `prefer` given, keep the row with the largest value of it; then fill
    remaining gaps from the other rows of the same date (first non-null)."""
    if prefer and prefer in df.columns:
        df = df.sort_values(prefer, ascending=False, na_position="last", kind="stable")
    return df.groupby("date", sort=True, as_index=False).first()


__all__ = ["norm", "find_col", "to_num", "wallclock", "iso", "date_str", "finalize", "dedup_dates", "np"]
