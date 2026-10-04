"""Oura daily CSV export -> Dataset. Column names are matched tolerantly (case/space-insensitive,
exact first then substring)."""
from __future__ import annotations

import pandas as pd

from ..schema import Dataset, Profile
from ._common import date_str, dedup_dates, find_col, finalize, iso, norm, to_num, wallclock

# canonical -> (patterns, excludes, kind); kind: num | dur (seconds unless header says min/hour) | pct
MAP: list[tuple[str, list[str], list[str], str]] = [
    ("rhr", ["average resting heart rate", "resting heart rate"], ["lowest", "score"], "num"),
    ("hrv_rmssd", ["average hrv", "hrv"], ["balance", "score"], "num"),
    ("sleep_min", ["total sleep duration", "total sleep"], [], "dur"),
    ("sleep_deep_min", ["deep sleep duration", "deep sleep"], ["score"], "dur"),
    ("sleep_rem_min", ["rem sleep duration", "rem sleep"], ["score"], "dur"),
    ("sleep_efficiency", ["sleep efficiency"], ["score"], "pct"),
    ("resp_rate", ["respiratory rate", "average breath", "breathing rate"], ["score"], "num"),
    ("steps", ["steps"], ["score"], "num"),
    ("active_kcal", ["activity calories", "active calories"], [], "num"),
    ("spo2", ["average spo2", "spo2 percentage", "spo2"], ["score", "breathing"], "pct"),
    ("vo2max", ["vo2 max", "vo2max", "vo2"], [], "num"),
]


def _read(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        head = fh.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(path, dtype=str, sep=sep, encoding="utf-8-sig")


def load(path: str) -> Dataset:
    raw = _read(path)
    date_c = find_col(raw.columns, ["date", "day", "summary date"])
    if date_c is None:
        raise ValueError(f"{path}: Oura CSV without a 'date' column")
    out = pd.DataFrame({"date": date_str(raw[date_c])})
    for canon, pats, excl, kind in MAP:
        col = find_col(raw.columns, pats, excl)
        if col is None:
            continue
        v = to_num(raw[col])
        n = norm(col)
        if kind == "dur":
            # ASSUMPTION: Oura durations are seconds unless the header says otherwise
            if "hour" in n or "(h)" in n:
                v = v * 60.0
            elif "min" not in n:
                v = v / 60.0
        elif kind == "pct" and v.notna().any() and v.dropna().le(1.0).all():
            v = v * 100.0
        out[canon] = v
    on_c = find_col(raw.columns, ["bedtime start", "bedtime_start", "sleep start"])
    off_c = find_col(raw.columns, ["bedtime end", "bedtime_end", "sleep end"])
    if on_c:
        out["sleep_onset"] = iso(wallclock(raw[on_c]))
    if off_c:
        out["sleep_offset"] = iso(wallclock(raw[off_c]))
    out = out[out["date"].notna()]
    daily = dedup_dates(out, prefer="sleep_min" if "sleep_min" in out.columns else None)
    prof = Profile(notes=["Oura daily CSV carries no age/sex — profile must be supplied by the user"])
    return finalize(daily, None, prof, "oura")
