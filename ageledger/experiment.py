"""Experiments: a suggestion with a receipt.

    propose(levers, rules)                       -> up to 3 protocols for the levers worth the most years
    receipt(ds, "steps", "2026-09-01", rules)    -> before/after per metric with a bootstrap CI of the change

The receipt is an N-of-1 before/after observation, never a causal claim; `note` (rules.yaml
experiments.receipt_note) says so in the output. Protocols and windows live in rules.yaml `experiments:`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ageledger.schema import Dataset
from ageledger.levers.sri import sleep_regularity_index

SRI_METRIC = "sleep_regularity_sri"


def _rules(rules: Optional[dict]) -> dict:
    if rules is None:
        from ageledger.levers.engine import load_rules
        rules = load_rules()
    return rules


def _protocols(rules: dict) -> dict[str, dict]:
    return {p["lever"]: p for p in (rules.get("experiments") or {}).get("protocols", [])}


def propose(ledger_levers: list[dict], rules: Optional[dict] = None) -> list[dict]:
    """Top 3 scored levers by years_recoverable that (a) have a protocol and (b) have something to recover
    (years_recoverable > 0 — a lever already in its reference band gets no experiment)."""
    rules = _rules(rules)
    protos = _protocols(rules)
    ranked = sorted((l for l in ledger_levers or [] if l.get("key") in protos
                     and (l.get("years_recoverable") or 0) > 0),
                    key=lambda l: -float(l["years_recoverable"]))
    out = []
    for l in ranked[:3]:
        p = protos[l["key"]]
        out.append({
            "lever": l["key"],
            "name": p["name"],
            "change": p["change"],
            "duration_days": int(p["duration_days"]),
            "metrics": list(p["metrics"]),
            "expected": p["expected"],
            "citation": dict(p.get("citation") or {}),
            "years_recoverable": l["years_recoverable"],
        })
    return out


# --------------------------------------------------------------------------- statistics

def _block_bootstrap_means(x: np.ndarray, rng: np.random.Generator, iters: int, block: int) -> np.ndarray:
    """Circular moving-block bootstrap of the mean: `iters` resampled means of `x` (time-ordered)."""
    n = len(x)
    b = max(1, min(block, n))
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(iters, k))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    idx = idx.reshape(iters, k * b)[:, :n]
    return x[idx].mean(axis=1)


def _sri_pairs(rows: pd.DataFrame) -> tuple[Optional[float], int, np.ndarray]:
    """(window SRI, nights, per-pair SRI values). Each pair of consecutive covered nights contributes one
    1440-minute comparison, so the window SRI is the mean of the per-pair values; the bootstrap resamples them."""
    sri, n = sleep_regularity_index(rows)
    rows = rows.sort_values("date").reset_index(drop=True)
    vals = []
    for i in range(len(rows) - 1):
        v, _ = sleep_regularity_index(rows.iloc[i:i + 2])
        if v is not None:
            vals.append(v)
    return sri, n, np.asarray(vals, dtype=float)


def _empty(metric: str, status: str, **kw) -> dict:
    out = {"metric": metric, "n_before": None, "n_after": None, "mean_before": None, "mean_after": None,
           "delta": None, "ci_low": None, "ci_high": None, "beyond_noise": None, "status": status}
    out.update(kw)
    return out


def _r(x: float, metric: str) -> float:
    return round(float(x), 0 if metric == "steps" else 2)


def receipt(ds: Dataset, lever_key: str, start_date: str, rules: Optional[dict] = None) -> dict:
    rules = _rules(rules)
    ex = rules["experiments"]
    protos = _protocols(rules)
    if lever_key not in protos:
        raise ValueError(f"no experiment protocol for lever {lever_key!r}; have {sorted(protos)}")
    p = protos[lever_key]
    start = pd.Timestamp(str(start_date)[:10])
    nb, na = int(ex["min_days_before"]), int(ex["min_days_after"])
    min_before = int(ex.get("min_values_before", nb // 2))
    iters = int(ex["bootstrap_iterations"])
    block = int(ex.get("bootstrap_block_days", 1))

    daily = ds.daily
    d = pd.to_datetime(daily["date"].astype(str).str[:10], errors="coerce")
    before_rows = daily[(d >= start - pd.Timedelta(days=nb)) & (d <= start - pd.Timedelta(days=1))]
    after_rows = daily[d >= start]
    last = d.max()
    days_elapsed = max(0, int((last - start).days) + 1) if pd.notna(last) else 0

    metrics = []
    for m in p["metrics"]:
        rng = np.random.default_rng(0)
        if m == SRI_METRIC:
            if not {"sleep_onset", "sleep_offset"} <= set(daily.columns):
                metrics.append(_empty(m, "missing", reason="no sleep_onset / sleep_offset columns in this export"))
                continue
            sb, n_b, xb = _sri_pairs(before_rows)
            sa, n_a, xa = _sri_pairs(after_rows)
            mean_b, mean_a = sb, sa
        else:
            if m not in daily.columns:
                metrics.append(_empty(m, "missing", reason=f"no {m} column in this export"))
                continue
            xb = pd.to_numeric(before_rows[m], errors="coerce").dropna().to_numpy(dtype=float)
            xa = pd.to_numeric(after_rows[m], errors="coerce").dropna().to_numpy(dtype=float)
            n_b, n_a = len(xb), len(xa)
            mean_b = float(xb.mean()) if n_b else None
            mean_a = float(xa.mean()) if n_a else None
        if n_a < na:
            metrics.append(_empty(m, "not_yet", n_before=n_b, n_after=n_a, days_after=n_a, need=na))
            continue
        if n_b < min_before or len(xb) < 2 or mean_b is None:
            metrics.append(_empty(m, "no_baseline", n_before=n_b, n_after=n_a, need_before=min_before,
                                  reason=f"only {n_b} values in the {nb} days before {start.date()}"))
            continue
        if len(xa) < 2 or mean_a is None:
            metrics.append(_empty(m, "not_yet", n_before=n_b, n_after=n_a, days_after=n_a, need=na))
            continue
        boot = _block_bootstrap_means(xa, rng, iters, block) - _block_bootstrap_means(xb, rng, iters, block)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        metrics.append({
            "metric": m, "n_before": int(n_b), "n_after": int(n_a),
            "mean_before": _r(mean_b, m), "mean_after": _r(mean_a, m),
            "delta": _r(mean_a - mean_b, m), "ci_low": _r(lo, m), "ci_high": _r(hi, m),
            "beyond_noise": bool(lo > 0 or hi < 0), "status": "ok",
        })

    statuses = {x["status"] for x in metrics}
    status = "ok" if statuses == {"ok"} else ("not_yet" if "ok" not in statuses and "not_yet" in statuses
                                               else ("partial" if "ok" in statuses else sorted(statuses)[0]))
    from ageledger.levers.engine import _assumption_texts, _clean
    return {
        "lever": lever_key,
        "name": p["name"],
        "start_date": str(start.date()),
        "status": status,
        "days_elapsed": days_elapsed,
        "metrics": metrics,
        "note": _clean(ex["receipt_note"]),
        "assumptions": _assumption_texts([ex.get("bootstrap_assumption")]),
        "citation": dict(p.get("citation") or {}),
    }


__all__ = ["propose", "receipt"]
