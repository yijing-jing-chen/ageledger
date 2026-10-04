"""Apple Health export (export.zip, extracted apple_health_export/ dir, or export.xml) -> Dataset.

export.xml can be hundreds of MB: it is streamed with iterparse and every top-level element is cleared
as soon as it has been read. Timestamps carry a UTC offset ("2026-09-01 07:12:33 -0400"); daily
attribution uses the LOCAL wall clock.
"""
from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Optional
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..schema import Dataset, Profile
from ._common import finalize, iso

HK = "HKQuantityTypeIdentifier"
QTYPES = {
    HK + "StepCount": "steps",
    HK + "HeartRate": "hr",
    HK + "RestingHeartRate": "rhr",
    HK + "HeartRateVariabilitySDNN": "hrv_sdnn",
    HK + "RespiratoryRate": "resp_rate",
    HK + "OxygenSaturation": "spo2",
    HK + "VO2Max": "vo2max",
    HK + "ActiveEnergyBurned": "active_kcal",
    HK + "WalkingSpeed": "walking_speed_ms",
    HK + "WalkingDoubleSupportPercentage": "walking_double_support_pct",
    HK + "WalkingStepLength": "walking_step_length_m",
    HK + "WalkingAsymmetryPercentage": "walking_asymmetry_pct",
    HK + "AppleSleepingWristTemperature": "skin_temp_c",
    HK + "Height": "_height",
    HK + "BodyMass": "_weight",
}
SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
SV = "HKCategoryValueSleepAnalysis"
ASLEEP = {SV + "AsleepCore", SV + "AsleepDeep", SV + "AsleepREM", SV + "AsleepUnspecified", SV + "Asleep"}
INBED = SV + "InBed"
AWAKE = SV + "Awake"
SESSION_GAP = pd.Timedelta(hours=2)   # ASSUMPTION: >2h without any sleep record starts a new sleep session


# ---------------------------------------------------------------- streaming parse
def _open_xml(path: str):
    p = Path(path)
    if p.is_dir():
        cands = [f for f in p.rglob("export.xml")]
        if not cands:
            raise ValueError(f"{path}: no export.xml inside")
        return open(cands[0], "rb"), None
    if zipfile.is_zipfile(p):
        zf = zipfile.ZipFile(p)
        names = [n for n in zf.namelist() if PurePosixPath(n).name.lower() == "export.xml"
                 and not n.startswith("__MACOSX")]
        if not names:
            zf.close()
            raise ValueError(f"{path}: zip has no export.xml — not an Apple Health export?")
        return zf.open(names[0]), zf
    return open(p, "rb"), None


def _parse(fh: IO[bytes]):
    qty: dict[str, list] = {k: [] for k in set(QTYPES.values())}
    sleep: list = []
    me: dict = {}
    export_date: Optional[str] = None
    depth = 0
    root = None
    for ev, el in ET.iterparse(fh, events=("start", "end")):
        if ev == "start":
            depth += 1
            if root is None:
                root = el
            continue
        depth -= 1
        if depth != 1:          # only direct children of <HealthData>; nested Correlation records are copies
            continue
        tag = el.tag
        if tag == "Record":
            a = el.attrib
            t = a.get("type", "")
            key = QTYPES.get(t)
            if key is not None:
                qty[key].append((a.get("sourceName", ""), a.get("unit", ""), a.get("value"),
                                 a.get("startDate"), a.get("endDate")))
            elif t == SLEEP_TYPE:
                sleep.append((a.get("sourceName", ""), a.get("value", ""), a.get("startDate"), a.get("endDate")))
        elif tag == "Me":
            me = dict(el.attrib)
        elif tag == "ExportDate":
            export_date = el.attrib.get("value")
        el.clear()
        root.clear()            # drop the processed child from the root so memory stays flat
    return qty, sleep, me, export_date


# ---------------------------------------------------------------- helpers
def _wall(s: pd.Series) -> pd.Series:
    """'YYYY-MM-DD HH:MM:SS ±hhmm' -> naive local wall clock."""
    return pd.to_datetime(s.astype("string").str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")


def _frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["src", "unit", "value", "start_s", "end_s"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    df["start"] = _wall(df["start_s"])
    df["end"] = _wall(df["end_s"])
    df["date"] = df["start"].dt.strftime("%Y-%m-%d")
    return df[df["value"].notna() & df["start"].notna()]


def _pct(v: pd.Series) -> pd.Series:
    """Apple exports '%' quantities as fractions 0–1; scale to 0–100 when the whole series is ≤1."""
    return v * 100.0 if v.notna().any() and v.dropna().abs().le(1.0).all() else v


def _unit_scale(df: pd.DataFrame, table: dict[str, float]) -> pd.Series:
    f = df["unit"].map(lambda u: table.get(str(u).strip().lower(), np.nan))
    return df["value"] * f


def _dedup_sum(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per day keep the single source with the highest daily total (Apple dedups iPhone+Watch;
    summing would double count). Returns (daily total, chosen source per day)."""
    per = df.groupby(["date", "src"])["value"].sum()
    idx = per.groupby(level="date").idxmax()
    chosen = pd.Series([i[1] for i in idx.values], index=idx.index)
    return per.groupby(level="date").max(), chosen


def _union_minutes(starts: np.ndarray, ends: np.ndarray) -> float:
    if len(starts) == 0:
        return 0.0
    order = np.argsort(starts)
    s, e = starts[order], ends[order]
    total = np.timedelta64(0, "s")
    cs, ce = s[0], e[0]
    for a, b in zip(s[1:], e[1:]):
        if a > ce:
            total += ce - cs
            cs, ce = a, b
        elif b > ce:
            ce = b
    total += ce - cs
    return float(total / np.timedelta64(1, "s")) / 60.0


def _spread_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Distribute each interval record's value across the wall-clock minutes it overlaps, pro rata."""
    s = df["start"].to_numpy("datetime64[s]")
    e = df["end"].fillna(df["start"]).to_numpy("datetime64[s]")
    e = np.maximum(e, s)
    v = df["value"].to_numpy(float)
    m0 = s.astype("datetime64[m]")
    m1 = (e - np.timedelta64(1, "s")).astype("datetime64[m]")
    single = (m1 <= m0) | (e == s)
    ts = [m0[single]]
    vals = [v[single]]
    for a, b, val in zip(s[~single], e[~single], v[~single]):
        mins = np.arange(a.astype("datetime64[m]"), (b - np.timedelta64(1, "s")).astype("datetime64[m]")
                         + np.timedelta64(1, "m"), np.timedelta64(1, "m"))
        lo = np.maximum(mins.astype("datetime64[s]"), a)
        hi = np.minimum((mins + np.timedelta64(1, "m")).astype("datetime64[s]"), b)
        w = (hi - lo).astype(float)
        ts.append(mins)
        vals.append(val * w / w.sum())
    out = pd.DataFrame({"ts": np.concatenate(ts).astype("datetime64[ns]"), "steps": np.concatenate(vals)})
    return out.groupby("ts", as_index=False)["steps"].sum()


# ---------------------------------------------------------------- sleep
def _sleep_nights(rows: list) -> pd.DataFrame:
    cols = ["date", "sleep_min", "sleep_deep_min", "sleep_rem_min", "sleep_onset", "sleep_offset",
            "sleep_efficiency", "_on", "_off"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=["src", "value", "start_s", "end_s"])
    df["start"] = _wall(df["start_s"])
    df["end"] = _wall(df["end_s"])
    df["kind"] = np.where(df["value"].isin(ASLEEP), "asleep",
                          np.where(df["value"] == INBED, "inbed", np.where(df["value"] == AWAKE, "awake", "")))
    df = df[(df["kind"] != "") & df["start"].notna() & df["end"].notna() & (df["end"] > df["start"])]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.sort_values("start").reset_index(drop=True)
    # sessionize across all sources: a segment attaches to the running session unless a >2h gap precedes it
    run_end = df["end"].cummax().shift()
    df["session"] = (df["start"] > run_end + SESSION_GAP).cumsum()

    nights = []
    for _, g in df.groupby("session"):
        asl = g[g["kind"] == "asleep"]
        if asl.empty:
            continue
        # prefer the source with the most records that night (among sources that logged sleep); never sum sources
        cand = g[g["src"].isin(set(asl["src"]))].groupby("src").size()
        best = sorted(cand.index, key=lambda s: (-cand[s], s))[0]
        a = asl[asl["src"] == best]
        st, en = a["start"].to_numpy("datetime64[s]"), a["end"].to_numpy("datetime64[s]")
        asleep = _union_minutes(st, en)
        deep = a[a["value"] == SV + "AsleepDeep"]
        rem = a[a["value"] == SV + "AsleepREM"]
        has_stages = a["value"].isin([SV + "AsleepCore", SV + "AsleepDeep", SV + "AsleepREM"]).any()
        inb = g[(g["kind"] == "inbed") & (g["src"] == best)]
        if inb.empty:   # Watch writes stages, iPhone writes InBed: fall back to any source's InBed that night
            inb = g[g["kind"] == "inbed"]
        eff = np.nan
        if not inb.empty:
            ib = _union_minutes(inb["start"].to_numpy("datetime64[s]"), inb["end"].to_numpy("datetime64[s]"))
            if ib > 0 and asleep <= ib:
                eff = asleep / ib * 100.0   # asleep > in-bed means the sources disagree: leave NaN
        on, off = a["start"].min(), a["end"].max()
        nights.append({
            "date": off.strftime("%Y-%m-%d"),          # the night belongs to the day it ENDS on
            "sleep_min": asleep,
            "sleep_deep_min": _union_minutes(deep["start"].to_numpy("datetime64[s]"), deep["end"].to_numpy("datetime64[s]")) if has_stages else np.nan,
            "sleep_rem_min": _union_minutes(rem["start"].to_numpy("datetime64[s]"), rem["end"].to_numpy("datetime64[s]")) if has_stages else np.nan,
            "sleep_onset": on.strftime("%Y-%m-%dT%H:%M:%S"),
            "sleep_offset": off.strftime("%Y-%m-%dT%H:%M:%S"),
            "sleep_efficiency": eff,
            "_on": on, "_off": off,
        })
    if not nights:
        return pd.DataFrame(columns=cols)
    n = pd.DataFrame(nights)
    # several sessions ending the same day (night + nap): the main sleep is the longest
    n = n.sort_values("sleep_min", ascending=False, kind="stable").drop_duplicates("date")
    return n.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- main
def load(path: str) -> Dataset:
    fh, zf = _open_xml(path)
    try:
        qty, sleep_rows, me, export_date = _parse(fh)
    finally:
        fh.close()
        if zf is not None:
            zf.close()

    notes: list[str] = []
    offsets: set[str] = set()
    daily_parts: dict[str, pd.Series] = {}
    minute_parts: list[pd.DataFrame] = []

    nights = _sleep_nights(sleep_rows)
    if len(nights):
        for c in ["sleep_min", "sleep_deep_min", "sleep_rem_min", "sleep_onset", "sleep_offset", "sleep_efficiency"]:
            daily_parts[c] = nights.set_index("date")[c]

    F = {k: _frame(v) for k, v in qty.items() if v}

    if "steps" in F:
        df = F["steps"]
        total, chosen = _dedup_sum(df)
        daily_parts["steps"] = total
        keep = df[df["src"] == df["date"].map(chosen)]
        minute_parts.append(_spread_minutes(keep))
        offsets |= set(keep["start_s"].astype("string").str.slice(20).dropna())
    if "hr" in F:
        df = F["hr"]
        hr = df.assign(ts=df["start"].dt.floor("min")).groupby("ts", as_index=False)["value"].mean()
        minute_parts.append(hr.rename(columns={"value": "hr"}))
        offsets |= set(df["start_s"].astype("string").str.slice(20).dropna())
    for key in ["rhr", "hrv_sdnn"]:
        if key in F:
            daily_parts[key] = F[key].groupby("date")["value"].mean()
    if "spo2" in F:
        df = F["spo2"]
        daily_parts["spo2"] = _pct(df["value"]).groupby(df["date"]).mean()
    if "vo2max" in F:
        df = F["vo2max"].sort_values("start")
        daily_parts["vo2max"] = df.groupby("date")["value"].last()
    if "active_kcal" in F:
        df = F["active_kcal"].copy()
        df["value"] = _unit_scale(df, {"kcal": 1.0, "cal": 1.0, "kj": 1 / 4.184})
        daily_parts["active_kcal"] = _dedup_sum(df.dropna(subset=["value"]))[0]
    if "walking_speed_ms" in F:
        df = F["walking_speed_ms"].copy()
        df["value"] = _unit_scale(df, {"m/s": 1.0, "km/hr": 1 / 3.6, "km/h": 1 / 3.6, "mi/hr": 0.44704, "mph": 0.44704})
        daily_parts["walking_speed_ms"] = df.groupby("date")["value"].mean()
    for key in ["walking_double_support_pct", "walking_asymmetry_pct"]:
        if key in F:
            df = F[key]
            daily_parts[key] = _pct(df["value"]).groupby(df["date"]).mean()
    if "walking_step_length_m" in F:
        df = F["walking_step_length_m"].copy()
        df["value"] = _unit_scale(df, {"m": 1.0, "cm": 0.01, "in": 0.0254, "ft": 0.3048})
        daily_parts["walking_step_length_m"] = df.groupby("date")["value"].mean()
    if "skin_temp_c" in F:
        df = F["skin_temp_c"].copy()
        df["value"] = np.where(df["unit"].str.lower().str.contains("f"), (df["value"] - 32) * 5 / 9, df["value"])
        # attribute to the night: the day the sample ENDS on
        df["date"] = df["end"].fillna(df["start"]).dt.strftime("%Y-%m-%d")
        if df["value"].between(20, 45).mean() > 0.9:
            daily_parts["skin_temp_c"] = df[df["value"].between(20, 45)].groupby("date")["value"].mean()
        else:
            notes.append("AppleSleepingWristTemperature looks relative (deviation), not absolute °C — skin_temp_c withheld")
    if "resp_rate" in F:
        df = F["resp_rate"].sort_values("start")
        rr = df.groupby("date")["value"].mean()          # fallback: plain daily mean
        if len(nights):
            t = df["start"].to_numpy("datetime64[ns]")
            v = df["value"].to_numpy(float)
            sw = {}
            for _, n in nights.iterrows():
                i0 = np.searchsorted(t, np.datetime64(n["_on"], "ns"), "left")
                i1 = np.searchsorted(t, np.datetime64(n["_off"], "ns"), "right")
                if i1 > i0:
                    sw[n["date"]] = float(v[i0:i1].mean())
            rr = pd.Series(sw, dtype=float).combine_first(rr)
        daily_parts["resp_rate"] = rr

    daily = pd.DataFrame(daily_parts)
    daily.index.name = "date"
    daily = daily.reset_index()

    minute = None
    if minute_parts:
        minute = minute_parts[0]
        for m in minute_parts[1:]:
            minute = minute.merge(m, on="ts", how="outer")
        offsets = {o.strip() for o in offsets if o and o.strip()}
        if len(offsets) == 1:
            tz = _parse_offset(next(iter(offsets)))
            if tz is not None:
                minute["ts"] = minute["ts"].dt.tz_localize(tz)
        elif len(offsets) > 1:
            notes.append(f"minute ts kept as naive LOCAL wall clock: export spans {len(offsets)} UTC offsets "
                         f"({', '.join(sorted(offsets))}); a single tz-aware column would shift some days by the DST hour")

    profile = _profile(me, export_date, F, daily, notes)
    return finalize(daily, minute, profile, "apple")


def _parse_offset(o: str) -> Optional[dt.timezone]:
    o = o.strip().replace(":", "")
    if len(o) == 5 and o[0] in "+-" and o[1:].isdigit():
        sign = 1 if o[0] == "+" else -1
        return dt.timezone(sign * dt.timedelta(hours=int(o[1:3]), minutes=int(o[3:5])))
    return None


def _profile(me: dict, export_date: Optional[str], F: dict, daily: pd.DataFrame, notes: list[str]) -> Profile:
    prof = Profile(notes=notes)
    sex = me.get("HKCharacteristicTypeIdentifierBiologicalSex", "")
    prof.sex = {"HKBiologicalSexFemale": "female", "HKBiologicalSexMale": "male"}.get(sex)
    dob_s = me.get("HKCharacteristicTypeIdentifierDateOfBirth", "")
    if dob_s:
        try:
            dob = dt.date.fromisoformat(dob_s[:10])
            if export_date:
                ref = dt.date.fromisoformat(export_date[:10])
            else:
                ref = dt.date.fromisoformat(daily["date"].max()) if len(daily) else dt.date.today()
                notes.append("no <ExportDate>; age computed at the last day with data")
            prof.age_years = float(ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day)))
        except ValueError:
            notes.append(f"unparseable DateOfBirth {dob_s!r}")
    if "_height" in F and len(F["_height"]):
        h = F["_height"].sort_values("start").iloc[-1]
        f = {"cm": 1.0, "m": 100.0, "in": 2.54, "ft": 30.48}.get(str(h["unit"]).lower())
        prof.height_cm = float(h["value"] * f) if f else None
    if "_weight" in F and len(F["_weight"]):
        w = F["_weight"].sort_values("start").iloc[-1]
        f = {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "st": 6.35029318}.get(str(w["unit"]).lower())
        prof.weight_kg = float(w["value"] * f) if f else None
    return prof
