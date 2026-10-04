"""Experiments: protocol proposals and the before/after receipt."""
from __future__ import annotations

import pytest

from ageledger.experiment import propose, receipt
from ageledger.levers.engine import load_rules
from ageledger.report import build
from tests.synthetic import synthetic_person, whoop_like

RULES = load_rules()


def _person(seed=0, days=180):
    return whoop_like(synthetic_person(days=days, seed=seed, with_minute=False))


def test_proposals_ordered_and_filtered():
    levers = [
        {"key": "rhr", "years_recoverable": 0.5},
        {"key": "steps", "years_recoverable": 2.0},
        {"key": "sleep_regularity", "years_recoverable": 1.0},
        {"key": "vo2max", "years_recoverable": 0.1},
        {"key": "sleep_duration", "years_recoverable": 0.0},      # nothing to recover → no experiment
        {"key": "no_protocol", "years_recoverable": 9.0},
    ]
    props = propose(levers, RULES)
    assert [p["lever"] for p in props] == ["steps", "sleep_regularity", "rhr"]
    assert set(props[0]) == {"lever", "name", "change", "duration_days", "metrics", "expected", "citation",
                             "years_recoverable"}
    assert props[0]["name"] == "Walk to the next quartile" and props[0]["duration_days"] == 14
    assert props[0]["citation"]["doi"] == "10.1016/S2468-2667(21)00302-9"
    assert propose([], RULES) == []


def test_ledger_proposals_follow_lever_ranking():
    L = build(_person())
    props = L["experiments"]["proposals"]
    assert props
    yrs = [p["years_recoverable"] for p in props]
    assert yrs == sorted(yrs, reverse=True)
    assert L["experiments"]["receipt"] is None


def test_receipt_not_yet():
    ds = _person()
    start = ds.daily["date"].iloc[-5]                 # 5 days of data from the start date
    r = receipt(ds, "steps", start, RULES)
    assert r["status"] == "not_yet"
    for m in r["metrics"]:
        assert m["status"] == "not_yet" and m["days_after"] == 5 and m["need"] == 14
        assert m["delta"] is None and m["beyond_noise"] is None


def test_receipt_detects_real_step_jump():
    seed = 9
    flat = _person(seed)
    jump = _person(seed)
    jump.daily.loc[150:, "steps"] += 2500
    start = jump.daily["date"].iloc[150]
    r = receipt(jump, "steps", start, RULES)
    m = next(x for x in r["metrics"] if x["metric"] == "steps")
    assert m["status"] == "ok" and m["n_before"] == 14 and m["n_after"] == 30
    assert m["delta"] == pytest.approx(2500, abs=600)
    assert m["beyond_noise"] is True and m["ci_low"] > 0
    m0 = next(x for x in receipt(flat, "steps", start, RULES)["metrics"] if x["metric"] == "steps")
    assert m["delta"] - m0["delta"] == pytest.approx(2500, abs=1)   # the receipt sees exactly the added steps
    assert r["note"].startswith("Before/after on one person is an N-of-1 observation")
    assert r["citation"]["short"] == "Paluch 2022"


def test_receipt_no_change_ci_contains_zero():
    ds = _person(0)
    start = ds.daily["date"].iloc[150]
    r = receipt(ds, "steps", start, RULES)
    m = next(x for x in r["metrics"] if x["metric"] == "steps")
    assert m["ci_low"] <= 0 <= m["ci_high"] and m["beyond_noise"] is False


def test_receipt_is_deterministic():
    ds = _person(3)
    start = ds.daily["date"].iloc[140]
    assert receipt(ds, "rhr", start, RULES) == receipt(ds, "rhr", start, RULES)


def test_receipt_sri_metric_and_missing_column():
    ds = _person()
    ds.daily = ds.daily.drop(columns=["hrv_rmssd"])
    r = receipt(ds, "sleep_regularity", ds.daily["date"].iloc[150], RULES)
    by = {m["metric"]: m for m in r["metrics"]}
    sri = by["sleep_regularity_sri"]
    assert sri["status"] == "ok" and -100 <= sri["mean_before"] <= 100 and sri["n_before"] == 14
    assert sri["ci_low"] <= sri["delta"] <= sri["ci_high"]
    assert by["hrv_rmssd"]["status"] == "missing" and by["hrv_rmssd"]["delta"] is None
    assert r["status"] == "partial"


def test_receipt_unknown_lever_raises():
    with pytest.raises(ValueError):
        receipt(_person(), "hrv_rmssd", "2026-09-01", RULES)


def test_report_receipt_wiring():
    ds = _person()
    L = build(ds, experiment={"lever": "steps", "start_date": ds.daily["date"].iloc[150]})
    assert L["experiments"]["receipt"]["lever"] == "steps"
    assert L["experiments"]["receipt"]["metrics"][0]["metric"] == "steps"
