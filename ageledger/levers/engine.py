"""Generic lever engine. The science lives in rules.yaml; this file only knows how to
window, aggregate, look up a band, and convert a hazard ratio to years (Gompertz).

Public API:
    load_rules(path=None) -> dict
    evaluate(ds, rules=None) -> {"levers", "withheld", "context", "constants"}
    years_from_hr(hr, mrdt) -> float
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from ageledger.schema import Dataset
from ageledger.levers.sri import sleep_regularity_index

RULES_PATH = Path(__file__).with_name("rules.yaml")
CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


# --------------------------------------------------------------------------- basics

def load_rules(path: Optional[str | Path] = None) -> dict:
    with open(path or RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def years_from_hr(hr: float, mrdt: float) -> float:
    """Gompertz: years = ln(HR) / ln(2) x MRDT. Positive = older than reference."""
    return math.log(hr) / math.log(2) * mrdt


def _mrdt(rules: dict) -> float:
    v = rules["gompertz_mrdt_years"]
    return float(v["value"] if isinstance(v, dict) else v)


def _assumption_texts(entries) -> list[str]:
    out = []
    for e in entries or []:
        if isinstance(e, dict):
            if e.get("assumption", True) and e.get("text"):
                out.append(e["text"])
        elif isinstance(e, str):
            out.append(e)
    return out


def _clean(s: str) -> str:
    return " ".join(str(s).split())


def _citation(lever: dict) -> dict:
    return {k: v for k, v in (lever.get("citation") or {}).items()}


# --------------------------------------------------------------------------- windowing

def _dates(daily: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(daily["date"].astype(str).str[:10], errors="coerce")


def _window(daily: pd.DataFrame, window_days: int) -> tuple[pd.DataFrame, int]:
    """Rows in the last `window_days` calendar days ending on the export's last date.
    Returns (rows, M) where M = calendar days of that window the export spans."""
    d = _dates(daily)
    if d.isna().all():
        return daily.iloc[0:0], 0
    last, first = d.max(), d.min()
    start = last - pd.Timedelta(days=window_days - 1)
    rows = daily[(d >= start) & (d <= last)]
    m = min(window_days, (last - first).days + 1)
    return rows, m


def _numeric(rows: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(rows[col], errors="coerce").dropna()


# --------------------------------------------------------------------------- lookups

def _in_band(x: float, b: dict) -> bool:
    lo, hi = b.get("lo"), b.get("hi")
    if lo is not None:
        if b.get("lo_exclusive") and not x > lo:
            return False
        if not b.get("lo_exclusive") and not x >= lo:
            return False
    if hi is not None:
        if b.get("hi_inclusive") and not x <= hi:
            return False
        if not b.get("hi_inclusive") and not x < hi:
            return False
    return True


def _find_band(x: float, bands: list[dict]) -> Optional[dict]:
    for b in bands:
        if _in_band(x, b):
            return b
    return None


def _band_by_name(bands: list[dict], name: str) -> dict:
    for b in bands:
        if b["name"] == name:
            return b
    raise KeyError(f"reference band {name!r} not in bands")


def _band_assumptions(*bands: dict) -> list[str]:
    out = []
    for b in bands:
        if b and b.get("assumption") and b.get("assumption_text"):
            out.append(b["assumption_text"])
    return out


def _age_decade_key(age: float) -> Optional[str]:
    if age is None or age < 20 or age >= 80:
        return None
    lo = int(age // 10) * 10
    return f"{lo}-{lo + 9}"


def _interp_percentile(pcts: list[float], vals: list[float], target: float) -> float:
    """Piecewise-linear in percentile space; extrapolates from the nearest segment."""
    if target <= pcts[0]:
        i = 0
    elif target >= pcts[-1]:
        i = len(pcts) - 2
    else:
        i = max(j for j in range(len(pcts) - 1) if pcts[j] <= target)
    p0, p1, v0, v1 = pcts[i], pcts[i + 1], vals[i], vals[i + 1]
    return v0 + (target - p0) * (v1 - v0) / (p1 - p0)


def vo2max_cutpoints(rules: dict, sex: str, age: float, tiers: list[dict]) -> Optional[dict[float, float]]:
    """Percentile -> ml/kg/min cut points for the tier edges, for this sex x age decade."""
    norms = rules["vo2max_norms"]
    key = _age_decade_key(age)
    table = norms.get(sex)
    if key is None or table is None or key not in table:
        return None
    pcts, vals = norms["percentiles"], table[key]
    edges = sorted({t["pct_lo"] for t in tiers if t.get("pct_lo") is not None})
    return {p: round(_interp_percentile(pcts, vals, p), 2) for p in edges}


# --------------------------------------------------------------------------- scoring

def _population_from_bands(cfg: dict, fallback: dict, bands: list[dict], name_key: str = "name"):
    """Population-typical reference for band/tier scoring: (population_hr on the band hr scale, label, assumption).
    `cfg` (an age override) wins over `fallback` (the lever) field by field. (None, None, []) when unset."""
    def get(k):
        return cfg.get(k) if cfg and cfg.get(k) is not None else fallback.get(k)
    texts = _assumption_texts([get("population_assumption")])
    if get("population_hr") is not None:
        return float(get("population_hr")), get("population_label") or f"HR {get('population_hr')}", texts
    if get("population_band") is not None:
        b = next(b for b in bands if b[name_key] == get("population_band"))
        return float(b["hr"]), b[name_key], texts
    return None, None, []


def _result(lever, *, value, n_days, band, ref_band, hr, mrdt, rule, assumptions, notes=None,
            hr_pop=None, pop_band=None, pop_assumptions=()):
    years = years_from_hr(hr, mrdt)
    years_pop = years_from_hr(hr_pop, mrdt) if hr_pop is not None else None
    return {
        "key": lever["key"],
        "label": lever["label"],
        "window_days": lever["window_days"],
        "your_value": value,
        "unit": lever.get("unit"),
        "n_days": int(n_days),
        "band": band,
        "reference_band": ref_band,
        "hr_vs_reference": round(hr, 3),
        "years": round(years, 2),
        "years_recoverable": round(max(0.0, years), 2),
        "population_band": pop_band,
        "hr_vs_population": round(hr_pop, 3) if hr_pop is not None else None,
        "years_vs_population": round(years_pop, 2) if years_pop is not None else None,
        "rule": rule,
        "citation": _citation(lever),
        "assumptions": list(dict.fromkeys(list(assumptions) + list(pop_assumptions))),
        "confidence": lever.get("confidence", "low"),
        "notes": list(notes or []),
    }


def _score_bands(lever, value, n, mrdt, profile):
    bands, ref_name = lever["bands"], lever["reference_band"]
    assumptions = _assumption_texts(lever.get("assumptions"))
    rule = _clean(lever["rule_text"])
    age = getattr(profile, "age_years", None)
    override = None
    if lever.get("age_dependent"):
        if age is None:
            assumptions += _assumption_texts([lever.get("age_unknown_assumption")])
        else:
            for ov in lever["age_dependent"]:
                lo, hi = ov.get("age_min"), ov.get("age_max")
                if (lo is None or age >= lo) and (hi is None or age < hi):
                    bands, ref_name, override = ov["bands"], ov["reference_band"], ov
                    assumptions += _assumption_texts(ov.get("assumptions"))
                    if ov.get("rule_suffix"):
                        rule += f" [{ov['rule_suffix']}]"
                    break
    band = _find_band(value, bands)
    ref = _band_by_name(bands, ref_name)
    hr = band["hr"] / ref["hr"]
    assumptions += _band_assumptions(band, ref)
    pop_hr, pop_label, pop_txt = _population_from_bands(override, lever, bands)
    return _result(lever, value=value, n_days=n, band=band["name"], ref_band=ref["name"], hr=hr, mrdt=mrdt,
                   rule=rule, assumptions=assumptions, notes=lever.get("notes"),
                   hr_pop=band["hr"] / pop_hr if pop_hr else None, pop_band=pop_label, pop_assumptions=pop_txt)


def _score_exponential(lever, value, n, mrdt, profile):
    f = lever["formula"]
    x = max(value, f.get("floor", -math.inf))
    hr = f["base"] ** ((x - f["reference_value"]) / f["per"])
    disp = _find_band(value, lever.get("display_bands", []))
    notes = list(lever.get("notes") or [])
    if value < f.get("floor", -math.inf):
        notes.append(f"value below the {f['floor']} floor of the source curve — scored at {f['floor']}")
    pop_ref = f.get("population_reference_value")
    hr_pop = f["base"] ** ((x - pop_ref) / f["per"]) if pop_ref is not None else None
    return _result(lever, value=value, n_days=n, band=disp["name"] if disp else f"{value}",
                   ref_band=lever["reference_band"], hr=hr, mrdt=mrdt, rule=_clean(lever["rule_text"]),
                   assumptions=_assumption_texts(lever.get("assumptions")), notes=notes,
                   hr_pop=hr_pop, pop_band=f"{pop_ref:g} {lever.get('unit', '')}".strip() if pop_ref is not None else None,
                   pop_assumptions=_assumption_texts([lever.get("population_assumption")]) if pop_ref is not None else [])


def _score_norm_tier(lever, value, n, mrdt, profile, rules):
    tiers = lever["tiers"]
    sex, age = profile.sex, profile.age_years
    cuts = vo2max_cutpoints(rules, sex, age, tiers)
    if cuts is None:
        return None, f"no {lever['key']} norms for sex={sex!r}, age={age} (table covers 20–79, male/female)"
    tier = None
    for t in tiers:
        lo = cuts[t["pct_lo"]] if t.get("pct_lo") is not None else None
        hi = cuts[t["pct_hi"]] if t.get("pct_hi") is not None else None
        if (lo is None or value >= lo) and (hi is None or value < hi):
            tier = t
            break
    ref = next(t for t in tiers if t["name"] == lever["reference_band"])
    hr = tier["hr"] / ref["hr"]
    norms = rules["vo2max_norms"]
    assumptions = _assumption_texts(lever.get("assumptions")) + _band_assumptions(tier, ref)
    if norms.get("assumption"):
        assumptions.append(norms["assumption_text"])
    notes = list(lever.get("notes") or [])
    if norms.get("verify_before_demo"):
        notes.append("vo2max norm table flagged verify_before_demo: " + _clean(norms["citation"]))
    pct_txt = f"{tier.get('pct_lo') or 0}–{tier.get('pct_hi') or 100}th pct"
    band = f"{tier['name']} ({pct_txt}, {sex} {_age_decade_key(age)})"
    rule = _clean(lever["rule_text"]) + " Cut points used (ml/kg/min): " + ", ".join(
        f"p{p:g}={v}" for p, v in sorted(cuts.items()))
    pop_hr, pop_label, pop_txt = _population_from_bands(None, lever, tiers)
    return _result(lever, value=value, n_days=n, band=band, ref_band=ref["name"], hr=hr, mrdt=mrdt,
                   rule=rule, assumptions=assumptions, notes=notes,
                   hr_pop=tier["hr"] / pop_hr if pop_hr else None, pop_band=pop_label,
                   pop_assumptions=pop_txt), None


def _context_nearest_bin(lever, value, n, profile, rules):
    table = rules[lever["reference_table"]]
    sex = profile.sex if profile.sex in ("female", "male") else None
    assumptions = _assumption_texts(lever.get("assumptions"))
    if sex is None:
        assumptions += _assumption_texts([lever.get("sex_unknown_assumption")])
    rows = []
    for b in table["bins"]:
        med = b[sex] if sex else (b["female"] + b["male"]) / 2
        rows.append((abs(value - med), b, med))
    _, best, med = min(rows, key=lambda r: r[0])
    if best.get("interpolated"):
        assumptions.append(table["interpolation_assumption_text"])
    who = {"female": "women", "male": "men", None: "adults (F/M mean)"}[sex]
    youngest = table["bins"][0]
    youngest_med = youngest[sex] if sex else (youngest["female"] + youngest["male"]) / 2
    if value > youngest_med:
        phrase = f"your rMSSD ({value:g} ms) is above the median of {who} aged {youngest['bin']} ({youngest_med:g} ms)"
    else:
        phrase = f"your rMSSD matches the median of {who} aged {best['bin']} ({med:g} ms)"
    note = phrase + " — " + " ".join(lever.get("notes") or [])
    return {
        "key": lever.get("context_key", lever["key"]),
        "your_value": value,
        "unit": lever.get("unit"),
        "n_days": int(n),
        "window_days": lever["window_days"],
        "closest_bin": best["bin"],
        "bin_median": round(med, 2),
        "note": note,
        "rule": _clean(lever["rule_text"]),
        "citation": _citation(lever),
        "assumptions": list(dict.fromkeys(assumptions)),
    }


def _rhr_trend(lever, daily, rows_all) -> Optional[dict]:
    tr = lever.get("trend")
    if not tr:
        return None
    d = _dates(daily)
    if d.isna().all() or (d.max() - d.min()).days + 1 < tr["min_span_days"]:
        return None
    last = d.max()
    vals = pd.to_numeric(daily[lever["input"]], errors="coerce")
    recent_mask = d > last - pd.Timedelta(days=tr["recent_days"])
    prior_mask = (d <= last - pd.Timedelta(days=tr["recent_days"])) & (
        d > last - pd.Timedelta(days=tr["recent_days"] + tr["prior_days"]))
    recent, prior = vals[recent_mask].dropna(), vals[prior_mask].dropna()
    if len(recent) < tr["min_days_each"] or len(prior) < tr["min_days_each"]:
        return None
    delta = float(recent.mean() - prior.mean())
    return {
        "key": f"{lever['key']}_trend",
        "your_value": round(delta, 1),
        "unit": "bpm (last 90d mean − prior 90d mean)",
        "flagged": delta >= tr["flag_delta"],
        "note": _clean(tr["note"]).format(delta=delta) if delta >= tr["flag_delta"]
                else f"RHR change last 90d vs prior 90d: {delta:+.1f} bpm (below the +{tr['flag_delta']} bpm flag). Note only; no years.",
        "citation": dict(tr.get("citation") or {}),
    }


# --------------------------------------------------------------------------- main

def _round_value(lever, v):
    if lever.get("unit") == "steps/day":
        return int(round(v))
    return round(float(v), 1)


def evaluate(ds: Dataset, rules: Optional[dict] = None) -> dict:
    rules = rules or load_rules()
    mrdt = _mrdt(rules)
    daily = ds.daily if ds.daily is not None else pd.DataFrame()
    profile = ds.profile
    levers, withheld, context = [], [], []

    for lever in rules["levers"]:
        key = lever["key"]
        if lever.get("status") == "STUB":
            withheld.append({"key": key, "reason": f"not implemented in v0 (STUB) — {_clean(lever.get('rule_text', ''))}"})
            continue

        inputs = lever["input"] if isinstance(lever["input"], list) else [lever["input"]]
        missing = [c for c in inputs if c not in daily.columns]
        if "date" not in daily.columns or daily.empty:
            withheld.append({"key": key, "reason": "export has no daily rows"})
            continue
        if missing:
            reason = f"no {' / '.join(missing)} column in this export"
            if key == "hrv_rmssd" and "hrv_sdnn" in daily.columns:
                reason += " (it has SDNN, a different metric — not substituted)"
            withheld.append({"key": key, "reason": reason})
            continue

        need = lever.get("requires_profile") or []
        lacking = [f for f in need if getattr(profile, f, None) is None]
        if lacking:
            withheld.append({"key": key, "reason": f"needs profile {' and '.join(lacking)} for age/sex norm tiers; not in this export"})
            continue

        rows, m = _window(daily, lever["window_days"])
        min_days = lever["min_days"]

        if lever["aggregate"] == "sri":
            sri, n = sleep_regularity_index(rows)
            if n < min_days:
                withheld.append({"key": key, "reason": f"only {n} of {m} nights have both sleep_onset and sleep_offset (need {min_days})"})
                continue
            if sri is None:
                withheld.append({"key": key, "reason": "no two consecutive nights with onset/offset — SRI needs t and t+24h"})
                continue
            value = round(sri, 1)
        else:
            col = inputs[0]
            s = _numeric(rows, col)
            n = len(s)
            if n < min_days:
                withheld.append({"key": key, "reason": f"only {n} of {m} days have {col} (need {min_days})"})
                continue
            agg = lever["aggregate"]
            value = float(s.mean() if agg == "mean" else s.median())
            value = _round_value(lever, value)

        scoring = lever["scoring"]
        if scoring == "bands":
            levers.append(_score_bands(lever, value, n, mrdt, profile))
        elif scoring == "exponential":
            res = _score_exponential(lever, value, n, mrdt, profile)
            tr = _rhr_trend(lever, daily, rows)
            if tr:
                context.append(tr)
                if tr["flagged"]:
                    res["notes"].append(tr["note"])
            levers.append(res)
        elif scoring == "norm_tier":
            res, why = _score_norm_tier(lever, value, n, mrdt, profile, rules)
            if res is None:
                withheld.append({"key": key, "reason": why})
            else:
                levers.append(res)
        elif scoring == "context_nearest_bin":
            context.append(_context_nearest_bin(lever, value, n, profile, rules))
            withheld.append({"key": key, "reason": lever["withheld_reason"]})
        else:
            raise ValueError(f"unknown scoring {scoring!r} for lever {key}")

    levers.sort(key=lambda r: (-r["years_recoverable"], CONFIDENCE_ORDER.get(r["confidence"], 9)))
    return {
        "levers": levers,
        "withheld": withheld,
        "context": context,
        "constants": {"gompertz_mrdt_years": mrdt},
    }
