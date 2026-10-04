"""edge (OpenStrap) SQLite export -> Dataset.

DAILY  from metric_series(date, key, value).
MINUTE from decoded_onehz (1 Hz): hr mean per minute and ENMO (mg). edge is the only accelerometer
source here, so it is the only ingester that emits enmo_mg.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

from ..schema import Dataset, Profile
from ._common import finalize

# metric_series key -> canonical daily column.
# ASSUMPTION: edge 'hrv' is rMSSD (OpenStrap computes RMSSD from beat intervals). 'readiness' has no
# canonical column. SpO2 is never mapped: edge has no absolute SpO2 (schema).
KEY_MAP = {
    "rhr": "rhr", "hrv": "hrv_rmssd", "hrv_rmssd": "hrv_rmssd", "hrv_sdnn": "hrv_sdnn",
    "steps": "steps", "strain": "strain", "resp_rate": "resp_rate",
    "skin_temp_c": "skin_temp_c", "skin_temp": "skin_temp_c", "vo2max": "vo2max", "active_kcal": "active_kcal",
    "sleep_min": "sleep_min",
}
# ASSUMPTION (unverified): sleep_json is a JSON object; these are the key spellings we accept.
SLEEP_KEYS = {
    "sleep_min": ["sleep_min", "asleep_min", "total_sleep_min", "tst_min", "total_min", "duration_min"],
    "sleep_deep_min": ["sleep_deep_min", "deep_min", "deep"],
    "sleep_rem_min": ["sleep_rem_min", "rem_min", "rem"],
    "sleep_efficiency": ["sleep_efficiency", "efficiency"],
    "sleep_onset": ["sleep_onset", "onset", "start", "bedtime_start", "start_ts", "start_ms"],
    "sleep_offset": ["sleep_offset", "offset", "end", "wake", "bedtime_end", "end_ts", "end_ms"],
}
G = 9.80665
CHUNK = 500_000


def _tables(con) -> dict[str, list[str]]:
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")]
    return {n: [r[1] for r in con.execute(f'PRAGMA table_info("{n}")')] for n in names}


def _to_local_iso(x, tz) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().lstrip("-").isdigit()):
        v = float(x)
        unit = "ms" if v > 1e11 else "s"
        return pd.Timestamp(v, unit=unit, tz="UTC").tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        ts = pd.Timestamp(str(x))
    except Exception:
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(tz)
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _daily(con, tz, notes) -> pd.DataFrame:
    ms = pd.read_sql_query("SELECT rowid AS rid, date, key, value FROM metric_series ORDER BY rowid", con)
    if ms.empty:
        return pd.DataFrame({"date": []})
    ms["date"] = ms["date"].astype(str).str.slice(0, 10)
    ms = ms.drop_duplicates(["date", "key"], keep="last")   # re-computed metrics: last write wins
    rows: dict[str, dict] = {}
    for d, k, v in ms[["date", "key", "value"]].itertuples(index=False):
        r = rows.setdefault(d, {"date": d})
        if k in KEY_MAP:
            r[KEY_MAP[k]] = pd.to_numeric(v, errors="coerce")
        elif k == "sleep_json" and v is not None:
            try:
                obj = json.loads(v)
            except (TypeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            low = {str(kk).lower(): vv for kk, vv in obj.items()}
            for canon, cands in SLEEP_KEYS.items():
                hit = next((low[c] for c in cands if c in low and low[c] is not None), None)
                if hit is None or canon in r and pd.notna(r[canon]):
                    continue
                if canon in ("sleep_onset", "sleep_offset"):
                    r[canon] = _to_local_iso(hit, tz)
                else:
                    r[canon] = pd.to_numeric(hit, errors="coerce")
    df = pd.DataFrame(list(rows.values()))
    if "sleep_efficiency" in df.columns:
        e = pd.to_numeric(df["sleep_efficiency"], errors="coerce")
        df["sleep_efficiency"] = e * 100.0 if e.notna().any() and e.dropna().le(1.0).all() else e
    if "skin_temp_c" in df.columns:
        t = pd.to_numeric(df["skin_temp_c"], errors="coerce")
        df["skin_temp_c"] = t.where(t.between(20, 45))   # relative/raw values are not absolute °C
    return df


def _accel_scale(con, where: str, notes) -> Optional[float]:
    """Return the factor that converts ax/ay/az to g, from the median vector norm of a sample."""
    s = pd.read_sql_query(f"SELECT ax, ay, az FROM decoded_onehz WHERE ax IS NOT NULL AND ay IS NOT NULL "
                          f"AND az IS NOT NULL{where} LIMIT 20000", con)
    if s.empty:
        return None
    med = float(np.nanmedian(np.sqrt((s.astype(float) ** 2).sum(axis=1))))
    if 0.5 <= med <= 2.0:
        return 1.0                # g
    if 5.0 <= med <= 15.0:
        return 1.0 / G            # m/s²
    if 500.0 <= med <= 2000.0:
        return 1.0 / 1000.0       # mg
    notes.append(f"accelerometer units unrecognised (median |a| = {med:.3g}; expected ≈1 g, ≈9.81 m/s² or ≈1000 mg) — enmo_mg withheld")
    return None


def _minute(con, cols: list[str], tz, notes) -> Optional[pd.DataFrame]:
    have = set(cols)
    if "ts_ms" not in have:
        notes.append("decoded_onehz has no ts_ms — minute table skipped")
        return None
    # ASSUMPTION: on_wrist NULL = unknown, kept; only an explicit 0 drops the row
    where = " AND (on_wrist IS NULL OR on_wrist != 0)" if "on_wrist" in have else ""
    acc = {"ax", "ay", "az"} <= have
    scale = _accel_scale(con, where, notes) if acc else None
    sel = ["ts_ms"] + (["hr"] if "hr" in have else []) + (["ax", "ay", "az"] if scale is not None else [])
    q = f"SELECT {', '.join(sel)} FROM decoded_onehz WHERE ts_ms IS NOT NULL{where}"
    parts = []
    for ch in pd.read_sql_query(q, con, chunksize=CHUNK):
        m = (ch["ts_ms"].astype("int64") // 60_000).rename("m")
        agg = pd.DataFrame({"m": m})
        if "hr" in ch:
            hr = pd.to_numeric(ch["hr"], errors="coerce").astype(float)
            hr = hr.where((hr > 0) & (hr < 250))   # ASSUMPTION: hr 0 = no lock, not a reading
            agg["hr_s"], agg["hr_n"] = hr.fillna(0.0), hr.notna().astype(float)
        if scale is not None:
            a = ch[["ax", "ay", "az"]].astype(float) * scale
            enmo = np.maximum(0.0, np.sqrt((a ** 2).sum(axis=1, skipna=False)) - 1.0) * 1000.0
            agg["en_s"], agg["en_n"] = enmo.fillna(0.0), enmo.notna().astype(float)
        parts.append(agg.groupby("m").sum())
    if not parts:
        return None
    tot = pd.concat(parts).groupby(level=0).sum()
    out = pd.DataFrame({"ts": pd.to_datetime(tot.index.to_numpy(dtype="int64") * 60_000, unit="ms", utc=True).tz_convert(tz)})
    if "hr_s" in tot:
        out["hr"] = (tot["hr_s"] / tot["hr_n"].replace(0, np.nan)).to_numpy()
    if "en_s" in tot:
        out["enmo_mg"] = (tot["en_s"] / tot["en_n"].replace(0, np.nan)).to_numpy()
    return out


def load(path: str, tz: Optional[str] = None) -> Dataset:
    notes: list[str] = []
    if tz is None:
        tz = "UTC"
        notes.append("edge ts_ms treated as UTC and minute ts left in UTC — pass load(path, tz='America/New_York') "
                     "for local time (cosinor phase depends on it)")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tabs = _tables(con)
        if "metric_series" not in tabs and "decoded_onehz" not in tabs:
            raise ValueError(f"{path}: SQLite file has neither metric_series nor decoded_onehz — not an edge export")
        daily = _daily(con, tz, notes) if "metric_series" in tabs else pd.DataFrame({"date": []})
        minute = _minute(con, tabs["decoded_onehz"], tz, notes) if "decoded_onehz" in tabs else None
    finally:
        con.close()
    notes.append("edge export carries no age/sex — profile must be supplied by the user")
    return finalize(daily, minute, Profile(notes=notes), "edge")
