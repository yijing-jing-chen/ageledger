"""Synthetic 90-day person for tests. Not data about anyone; every number here is generated.

`synthetic_person()` -> Dataset with a DAILY frame carrying the columns a WHOOP export would map to, plus
(optionally) a MINUTE frame with hr and enmo_mg (source 'synthetic', which schema.validate allows).
`whoop_like(ds)` strips the minute table and relabels the source: what a WHOOP export actually gives.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ageledger.schema import Dataset, Profile

TZ = "America/New_York"
START = "2026-04-08"          # 180 days end 2026-10-04 (today at build time); inside US daylight time, no DST jump in a 90-day slice

# minute-level ENMO rhythm (milli-g) the cosinor test expects to recover, roughly
ENMO_MESOR = 30.0
ENMO_AMP = 25.0
ENMO_PEAK_H = 14.0


def _ar1(rng: np.random.Generator, n: int, sd: float, phi: float = 0.6) -> np.ndarray:
    """Mild autocorrelated noise with marginal SD `sd`."""
    e = rng.normal(0.0, sd * np.sqrt(1 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def synthetic_person(days: int = 90, age: float = 35, sex: str | None = "female", seed: int = 0,
                     steps_mean: float = 6500, rhr_mean: float = 62, rmssd_mean: float = 45,
                     sleep_mean_min: float = 430, onset_hour: float = 23.5, onset_jitter_h: float = 0.6,
                     vo2max: float = 38.0, with_minute: bool = True) -> Dataset:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=days, freq="D")

    steps = np.clip(steps_mean + _ar1(rng, days, 0.25 * steps_mean), 300, None).round().astype(int)
    rhr = rhr_mean + _ar1(rng, days, 2.0)
    rmssd = np.clip(rmssd_mean + _ar1(rng, days, 0.18 * rmssd_mean), 5, None)
    sleep_min = np.clip(sleep_mean_min + _ar1(rng, days, 35.0, 0.3), 180, 720)
    eff = np.clip(90 + _ar1(rng, days, 3.0, 0.3), 60, 99)
    resp = 14.5 + _ar1(rng, days, 0.4)
    vo2 = vo2max + _ar1(rng, days, 0.4, 0.95)
    strain = np.clip(8 + 4 * (steps - steps_mean) / steps_mean + _ar1(rng, days, 1.5, 0.2), 0, 21)

    # night ending on date D: onset the evening before
    onset_h = onset_hour + rng.normal(0.0, onset_jitter_h, days)
    onset = dates - pd.Timedelta(days=1) + pd.to_timedelta(onset_h, unit="h")
    in_bed_min = sleep_min / (eff / 100.0)
    offset = onset + pd.to_timedelta(in_bed_min, unit="m")
    onset, offset = onset.floor("min"), offset.floor("min")

    # WHOOP-style device scores. Drawn from a SEPARATE generator so adding them leaves every
    # pre-existing column (and the minute table) bit-identical for a given seed.
    rng2 = np.random.default_rng([seed, 1])
    recovery = np.clip(50 + 1.2 * (rmssd - rmssd.mean()) - 2.0 * (rhr - rhr.mean())
                       + rng2.normal(0.0, 8.0, days), 1, 99)
    need = np.empty(days)
    debt = np.empty(days)
    prev = 0.0
    for i in range(days):                     # need rises with yesterday's debt; debt = shortfall vs need
        need[i] = 450 + 0.3 * prev
        debt[i] = max(0.0, need[i] - sleep_min[i])
        prev = debt[i]
    shift_h = np.abs(np.diff(onset_h, prepend=onset_h[0]))
    consistency = np.clip(95 - 30 * shift_h + rng2.normal(0.0, 3.0, days), 0, 100)

    daily = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "steps": steps,
        "rhr": rhr.round(1),
        "hrv_rmssd": rmssd.round(1),
        "sleep_min": sleep_min.round(),
        "sleep_onset": onset.strftime("%Y-%m-%dT%H:%M:%S"),
        "sleep_offset": offset.strftime("%Y-%m-%dT%H:%M:%S"),
        "sleep_efficiency": eff.round(1),
        "resp_rate": resp.round(2),
        "vo2max": vo2.round(1),
        "strain": strain.round(1),
        "recovery_pct": recovery.round(),
        "sleep_need_min": need.round(),
        "sleep_debt_min": debt.round(),
        "sleep_consistency_pct": consistency.round(),
        "source": "synthetic",
    })

    minute = None
    if with_minute:
        # partial first and last day on purpose: the cosinor fit must trim them
        ts = pd.date_range(dates[0] + pd.Timedelta(hours=9, minutes=17),
                           dates[-1] + pd.Timedelta(hours=15), freq="min")
        n = len(ts)
        mod = (ts.hour * 60 + ts.minute).to_numpy(dtype=float)
        rhythm = ENMO_MESOR + ENMO_AMP * np.cos(2 * np.pi * (mod - ENMO_PEAK_H * 60) / 1440)
        enmo = rhythm + rng.normal(0.0, 5.0, n)

        asleep = np.zeros(n, dtype=bool)
        for on, off in zip(onset, offset):
            asleep[(ts >= on) & (ts < off)] = True
        enmo[asleep] = np.abs(rng.normal(1.5, 1.0, asleep.sum()))   # sleep: near zero
        enmo = np.clip(enmo, 0.0, None)

        day_idx = np.clip(((ts - dates[0]) // pd.Timedelta(days=1)).astype(int), 0, days - 1)
        base_rhr = rhr[day_idx]
        hr = np.where(asleep, base_rhr + rng.normal(0, 2, n),
                      base_rhr + 15 + 0.5 * (enmo - ENMO_MESOR) + rng.normal(0, 4, n))

        minute = pd.DataFrame({
            "ts": ts.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward"),
            "hr": hr.round(1),
            "enmo_mg": enmo,
            "source": "synthetic",
        }).dropna(subset=["ts"])   # DST fall-back minutes are ambiguous → dropped, never guessed

    ds = Dataset(daily=daily, minute=minute,
                 profile=Profile(age_years=age, sex=sex),
                 source="synthetic")
    ds.coverage = {
        "days": days, "first": daily["date"].iloc[0], "last": daily["date"].iloc[-1],
        "columns_present": [c for c in daily.columns if c not in ("date", "source")],
        "minute_days": int(minute["ts"].dt.normalize().nunique()) if minute is not None else 0,
    }
    return ds


def whoop_like(ds: Dataset) -> Dataset:
    """Same person as a WHOOP export sees them: daily only, no minute table, no accelerometer."""
    out = copy.deepcopy(ds)
    out.minute = None
    out.source = "whoop"
    out.daily = out.daily.assign(source="whoop")
    out.coverage = {**out.coverage, "minute_days": 0}
    return out
