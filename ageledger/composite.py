"""Composite biological-age estimates from DAILY wearable data (no accelerometer needed).

effective_age      — WHOOP-Age framing: each lever's published HR vs a population-typical value → Gompertz
                     years → sum → chronological age + Σ. Plus a trend re-run on truncated history.
fitness_age        — age at which the population-median VO2max equals yours (norm table in rules.yaml).
hrv_age_equivalent — the rMSSD context bin's midpoint. A norm match, not a clock.

No science lives here beyond the arithmetic; every reference value and assumption comes from rules.yaml.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Optional

import numpy as np
import pandas as pd

from ageledger.schema import Dataset
from ageledger.levers.engine import (
    _assumption_texts, _clean, _dates, _interp_percentile, _numeric, _window, evaluate, load_rules,
)


def _rules(rules):
    return rules or load_rules()


def _scored(lev_result: dict) -> list[dict]:
    return [l for l in (lev_result or {}).get("levers", []) if l.get("years_vs_population") is not None]


def _population_texts(rules: dict) -> dict[str, list[str]]:
    """lever key -> every population-assumption text the rules define for it (lever + age overrides)."""
    out = {}
    for lv in rules.get("levers", []):
        texts = _assumption_texts([lv.get("population_assumption")])
        for ov in lv.get("age_dependent") or []:
            texts += _assumption_texts([ov.get("population_assumption")])
        out[lv["key"]] = texts
    return out


def withheld_reason(ds: Dataset, lev_result: dict, rules=None) -> str:
    rules = _rules(rules)
    need = int(rules["composite"]["min_levers"])
    if getattr(ds.profile, "age_years", None) is None:
        return "needs chronological age — enter it above"
    n = len(_scored(lev_result))
    return f"only {n} levers computed (need {need})"


def _point(ds: Dataset, lev_result: dict, rules: dict, age: float) -> Optional[tuple[float, float, list[dict]]]:
    levers = _scored(lev_result)
    if len(levers) < int(rules["composite"]["min_levers"]):
        return None
    total = round(sum(l["years_vs_population"] for l in levers), 2)
    return round(age + total, 1), total, levers


def _trend(ds: Dataset, rules: dict, age_now: float, current_value: float) -> tuple[list[dict], Optional[float], list[str]]:
    cfg = rules["composite"]
    step, min_hist = int(cfg["trend_step_days"]), int(cfg["trend_min_history_days"])
    min_win = int(cfg.get("trend_min_window_days", 90))
    d = _dates(ds.daily)
    first, last = d.min(), d.max()
    current = [{"as_of": last.strftime("%Y-%m-%d"), "value": current_value}]
    span = (last - first).days + 1
    if span < min_hist:
        return current, None, [f"trend needs ≥{min_hist} days of history; this export spans {span}"]

    points = []
    k = 1
    while True:
        cutoff = last - pd.Timedelta(days=step * k)
        if (cutoff - first).days + 1 < min_win:
            break
        sub = copy.copy(ds)
        sub.daily = ds.daily[d <= cutoff].reset_index(drop=True)
        # chronological age at the cutoff, back-dated from the profile age (taken as age at the export's last date)
        age_then = age_now - (last - cutoff).days / 365.25
        pt = _point(sub, evaluate(sub, rules), rules, age_then)
        if pt is not None:
            points.append({"as_of": cutoff.strftime("%Y-%m-%d"), "value": pt[0]})
        k += 1
    points = list(reversed(points)) + current
    notes = [f"trend: effective age re-computed on the export truncated every {step} days "
             f"(each point uses the {min_win} days ending on its as_of date); chronological age back-dated "
             "from the profile age, so a steady person paces at ≈1.0 years/year"]
    pace = None
    if len(points) >= 3:
        t = np.array([(pd.Timestamp(p["as_of"]) - first).days / 365.25 for p in points])
        v = np.array([p["value"] for p in points])
        pace = round(float(np.polyfit(t, v, 1)[0]), 2)
        notes.append("pace over a short history is noisy: a 0.25-year change in Σ years over 3 months reads as ±1 year/year")
    return points, pace, notes


def effective_age(ds: Dataset, lev_result: dict, rules=None) -> Optional[dict]:
    rules = _rules(rules)
    age = getattr(ds.profile, "age_years", None)
    if age is None:
        return None
    pt = _point(ds, lev_result, rules, float(age))
    if pt is None:
        return None
    value, total, levers = pt
    cfg = rules["composite"]

    contributions = sorted(({
        "key": l["key"], "label": l["label"],
        "years_vs_population": l["years_vs_population"],
        "band": l["band"], "population_band": l["population_band"],
        "confidence": l["confidence"],
    } for l in levers), key=lambda c: -abs(c["years_vs_population"]))

    pop = _population_texts(rules)
    assumptions = _assumption_texts([cfg["independence_assumption"]])
    for l in levers:
        assumptions += [t for t in pop.get(l["key"], []) if t in l.get("assumptions", [])]
    assumptions += _assumption_texts([rules.get("gompertz_mrdt_meta")])

    trend, pace, notes = ([], None, [])
    if ds.daily is not None and not ds.daily.empty:
        trend, pace, notes = _trend(ds, rules, float(age), value)

    return {
        "value": value,
        "advance_years": total,
        "chronological_age": age,
        "n_levers": len(levers),
        "contributions": contributions,
        "trend": trend,
        "pace_years_per_year": pace,
        "rule": ("Effective age = chronological age + Σ over levers of ln(HR vs population-typical) / ln 2 × "
                 f"MRDT {rules['gompertz_mrdt_years']:g} years, each HR taken from the lever's published "
                 "all-cause hazard ratio."),
        "citation": dict(cfg.get("citation") or {}),
        "assumptions": list(dict.fromkeys(assumptions)),
        "confidence": "low",
        "notes": notes,
    }


# --------------------------------------------------------------------------- fitness age

def fitness_age(ds: Dataset, lev_result: dict, rules=None) -> Optional[dict]:
    rules = _rules(rules)
    sex = getattr(ds.profile, "sex", None)
    daily = ds.daily
    if sex not in ("female", "male") or daily is None or daily.empty or "vo2max" not in daily.columns:
        return None
    lever = next(l for l in rules["levers"] if l["key"] == "vo2max")
    rows, _ = _window(daily, lever["window_days"])
    s = _numeric(rows, "vo2max")
    if len(s) < lever["min_days"]:
        return None
    vo2 = round(float(s.median()), 1)

    norms = rules["vo2max_norms"]
    pcts, table = norms["percentiles"], norms[sex]
    pts = []
    for key, vals in table.items():
        lo = int(str(key).split("-")[0])
        pts.append((lo + 5, _interp_percentile(pcts, vals, 50)))
    pts.sort()
    ages = [a for a, _ in pts]
    p50 = [v for _, v in pts]

    # piecewise-linear age(p50); extrapolate from the end segments, then clamp
    if vo2 >= p50[0]:
        i = 0
    elif vo2 <= p50[-1]:
        i = len(pts) - 2
    else:
        i = next(j for j in range(len(pts) - 1) if p50[j] >= vo2 >= p50[j + 1])
    a0, a1, v0, v1 = ages[i], ages[i + 1], p50[i], p50[i + 1]
    raw = a0 + (vo2 - v0) * (a1 - a0) / (v1 - v0) if v1 != v0 else float(a0)
    notes = []
    value = min(80.0, max(20.0, raw))
    if value != raw:
        notes.append(f"raw estimate {raw:.1f} lies outside the 20–80 range the norm table covers — clamped to {value:g}")
    assumptions = [
        "Each decade's 50th-percentile VO2max placed at the decade midpoint (25, 35, …, 75), linear between midpoints",
    ]
    if norms.get("assumption"):
        assumptions.append(norms["assumption_text"])
    if norms.get("verify_before_demo"):
        assumptions.append("vo2max norm table flagged verify_before_demo: " + _clean(norms["citation"]))
    notes.append("Device VO2max is an estimate from HR and pace, not a treadmill test.")
    return {
        "value": round(value, 1),
        "vo2max": vo2,
        "sex": sex,
        "p50_by_age": {str(a): round(v, 2) for a, v in pts},
        "rule": (f"90-day median VO2max ({vo2} ml/kg/min) matched to the age at which the {sex} 50th-percentile "
                 "VO2max equals it (the fitness-age framing of Nes 2013 HUNT / Garmin Fitness Age)."),
        "citation": {"short": "Fitness-age framing: Nes 2013 HUNT / Garmin Fitness Age; norms: Cooper Institute (ACSM)",
                     "ref": _clean(norms["citation"]), "doi": "10.1111/sms.12032"},
        "assumptions": assumptions,
        "confidence": "low",
        "notes": notes,
    }


# --------------------------------------------------------------------------- HRV age equivalent

def hrv_age_equivalent(lev_result: dict, rules=None) -> Optional[dict]:
    ctx = next((c for c in (lev_result or {}).get("context", []) if c.get("key") == "hrv_rmssd_percentile"), None)
    if ctx is None or not ctx.get("closest_bin"):
        return None
    nums = [int(x) for x in re.findall(r"\d+", ctx["closest_bin"])]
    value = (nums[0] + nums[1]) / 2 if len(nums) >= 2 else float(nums[0])
    value = math.floor(value)  # "30–34" -> 32
    notes = [
        "NOT A VALIDATED AGING CLOCK. This is a norm match: the 5-year age bin whose median rMSSD on a "
        "10-second daytime resting ECG (Lifelines) is closest to your overnight wearable rMSSD. Overnight "
        "wearable rMSSD runs higher than daytime 10-s ECG rMSSD, so this reads younger than it should. No years.",
    ]
    if len(nums) < 2:
        notes.append(f"closest bin is open-ended ({ctx['closest_bin']}); value is its lower edge")
    return {
        "value": value,
        "closest_bin": ctx["closest_bin"],
        "your_value": ctx.get("your_value"),
        "bin_median": ctx.get("bin_median"),
        "rule": f"Midpoint of the age bin whose rMSSD median is closest to yours ({ctx['closest_bin']} → {value}).",
        "citation": dict(ctx.get("citation") or {}),
        "assumptions": list(ctx.get("assumptions") or []),
        "confidence": "low",
        "notes": notes + ([ctx["note"]] if ctx.get("note") else []),
    }
