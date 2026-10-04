import math

import numpy as np
import pandas as pd
import pytest

from ageledger.schema import Dataset, Profile
from ageledger.levers import evaluate, load_rules, years_from_hr, sleep_regularity_index, vo2max_cutpoints

END = pd.Timestamp("2026-09-30")


def make_daily(days=90, **overrides) -> pd.DataFrame:
    """Synthetic daily frame ending 2026-09-30. Each override is a scalar, list, or callable(i, date)."""
    dates = [END - pd.Timedelta(days=days - 1 - i) for i in range(days)]
    df = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates]})
    for col, v in overrides.items():
        if callable(v):
            df[col] = [v(i, d) for i, d in enumerate(dates)]
        elif isinstance(v, (list, tuple, np.ndarray)):
            df[col] = list(v)
        else:
            df[col] = v
    df["source"] = "synthetic"
    return df


def ds_of(daily, age=None, sex=None):
    return Dataset(daily=daily, minute=None, profile=Profile(age_years=age, sex=sex), source="synthetic")


def regular_sleep(onset_h=23, dur_h=8):
    def on(i, d):
        return (d - pd.Timedelta(days=1) + pd.Timedelta(hours=onset_h)).isoformat()

    def off(i, d):
        return (d - pd.Timedelta(days=1) + pd.Timedelta(hours=onset_h + dur_h)).isoformat()
    return on, off


def lever(out, key):
    return next(l for l in out["levers"] if l["key"] == key)


def withheld(out, key):
    return next(w for w in out["withheld"] if w["key"] == key)


# ------------------------------------------------------------------ core math

def test_years_from_hr():
    assert years_from_hr(1.0, 8.0) == 0
    assert years_from_hr(2.0, 8.0) == pytest.approx(8.0)
    assert years_from_hr(0.5, 8.0) == pytest.approx(-8.0)


def test_rules_load_and_constants():
    rules = load_rules()
    assert rules["gompertz_mrdt_years"] == 8.0
    keys = [l["key"] for l in rules["levers"]]
    assert keys[:6] == ["steps", "rhr", "sleep_duration", "sleep_regularity", "vo2max", "hrv_rmssd"]
    out = evaluate(ds_of(make_daily(30, steps=8000)))
    assert out["constants"] == {"gompertz_mrdt_years": 8.0}


# ------------------------------------------------------------------ steps

def test_steps_q2_years():
    out = evaluate(ds_of(make_daily(90, steps=5800), age=40))
    s = lever(out, "steps")
    assert s["band"].startswith("Q2")
    assert s["hr_vs_reference"] == pytest.approx(0.60 / 0.47, abs=1e-3)
    assert s["years"] == pytest.approx(math.log2(0.60 / 0.47) * 8, abs=0.01)
    assert s["years"] == pytest.approx(2.82, abs=0.01)
    assert s["years_recoverable"] == s["years"]
    assert s["citation"]["doi"] == "10.1016/S2468-2667(21)00302-9"
    assert any("midpoints" in a for a in s["assumptions"])
    assert s["n_days"] == 90


def test_steps_reference_band_is_zero():
    s = lever(evaluate(ds_of(make_daily(90, steps=11000), age=40)), "steps")
    assert s["years"] == 0 and s["years_recoverable"] == 0


def test_steps_age60_plateau_override():
    s = lever(evaluate(ds_of(make_daily(90, steps=7500), age=65)), "steps")
    assert s["years"] == 0
    assert any("≥7,000" in a for a in s["assumptions"])
    young = lever(evaluate(ds_of(make_daily(90, steps=7500), age=40)), "steps")
    assert young["years"] > 0


def test_steps_age_unknown_flags_assumption():
    s = lever(evaluate(ds_of(make_daily(90, steps=7500))), "steps")
    assert any("Age not in profile" in a for a in s["assumptions"])


def test_too_few_days_withheld():
    steps = [np.nan] * 80 + [6000] * 10
    out = evaluate(ds_of(make_daily(90, steps=steps)))
    assert not any(l["key"] == "steps" for l in out["levers"])
    assert withheld(out, "steps")["reason"] == "only 10 of 90 days have steps (need 14)"


def test_missing_column_withheld():
    out = evaluate(ds_of(make_daily(90, steps=8000)))
    assert "no rhr column" in withheld(out, "rhr")["reason"]


# ------------------------------------------------------------------ rhr

def test_rhr_70():
    r = lever(evaluate(ds_of(make_daily(90, rhr=70))), "rhr")
    assert r["hr_vs_reference"] == pytest.approx(1.09, abs=1e-3)
    assert r["years"] == pytest.approx(math.log2(1.09) * 8, abs=0.01)
    assert r["years"] == pytest.approx(0.995, abs=0.01)
    assert r["band"] == "60–80 bpm"


def test_rhr_floor_and_credit():
    r = lever(evaluate(ds_of(make_daily(90, rhr=40))), "rhr")
    assert r["hr_vs_reference"] == pytest.approx(1.09 ** -1.5, abs=1e-3)
    assert r["years"] < 0 and r["years_recoverable"] == 0


def test_rhr_trend_flag():
    rhr = [58] * 90 + [65] * 90
    out = evaluate(ds_of(make_daily(180, rhr=rhr)))
    tr = next(c for c in out["context"] if c["key"] == "rhr_trend")
    assert tr["flagged"] and tr["your_value"] == pytest.approx(7.0)
    assert any("Nauman" in n for n in lever(out, "rhr")["notes"])


def test_rhr_no_trend_under_180_days():
    out = evaluate(ds_of(make_daily(120, rhr=60)))
    assert not any(c["key"] == "rhr_trend" for c in out["context"])


# ------------------------------------------------------------------ sleep duration

@pytest.mark.parametrize("mins,hr", [(400, 1.12), (420, 1.00), (540, 1.00), (560, 1.30)])
def test_sleep_duration_bands(mins, hr):
    s = lever(evaluate(ds_of(make_daily(30, sleep_min=mins))), "sleep_duration")
    assert s["hr_vs_reference"] == pytest.approx(hr)
    assert s["confidence"] == "medium"


# ------------------------------------------------------------------ SRI

def test_sri_perfectly_regular_is_100():
    on, off = regular_sleep()
    sri, n = sleep_regularity_index(make_daily(30, sleep_onset=on, sleep_offset=off))
    assert n == 30
    assert sri == pytest.approx(100.0)


def test_sri_alternating_8h_shift_is_low():
    def on(i, d):
        base = d - pd.Timedelta(days=1)
        return (base + pd.Timedelta(hours=22 if i % 2 == 0 else 30)).isoformat()   # 22:00 prev day / 06:00 same day

    def off(i, d):
        base = d - pd.Timedelta(days=1)
        return (base + pd.Timedelta(hours=30 if i % 2 == 0 else 38)).isoformat()   # 06:00 / 14:00
    sri, n = sleep_regularity_index(make_daily(30, sleep_onset=on, sleep_offset=off))
    assert n == 30
    assert sri == pytest.approx(-100 / 3, abs=1.0)


def test_sri_skips_gaps_and_handles_tz():
    on, off = regular_sleep()
    daily = make_daily(30, sleep_onset=lambda i, d: pd.Timestamp(on(i, d)).tz_localize("America/New_York").isoformat(),
                       sleep_offset=lambda i, d: pd.Timestamp(off(i, d)).tz_localize("America/New_York").isoformat())
    daily.loc[10:12, ["sleep_onset", "sleep_offset"]] = None
    sri, n = sleep_regularity_index(daily)
    assert n == 27 and sri == pytest.approx(100.0)


def test_sri_lever_and_withheld():
    on, off = regular_sleep()
    s = lever(evaluate(ds_of(make_daily(30, sleep_onset=on, sleep_offset=off))), "sleep_regularity")
    assert s["years"] == 0 and s["confidence"] == "low"
    assert any("assumed" in a.lower() for a in s["assumptions"])
    assert "ASSUMED pending" in s["rule"]
    out = evaluate(ds_of(make_daily(10, sleep_onset=on, sleep_offset=off)))
    assert "only 10 of 10 nights" in withheld(out, "sleep_regularity")["reason"]


# ------------------------------------------------------------------ vo2max

def test_vo2max_withheld_without_profile():
    out = evaluate(ds_of(make_daily(30, vo2max=40.0)))
    w = withheld(out, "vo2max")
    assert "age_years" in w["reason"] and "sex" in w["reason"]
    assert not any(l["key"] == "vo2max" for l in out["levers"])


def test_vo2max_tier_with_profile():
    rules = load_rules()
    cuts = vo2max_cutpoints(rules, "female", 35, next(l for l in rules["levers"] if l["key"] == "vo2max")["tiers"])
    assert cuts[25] < cuts[50] < cuts[75] < cuts[97.7]
    v = lever(evaluate(ds_of(make_daily(30, vo2max=(cuts[25] + cuts[50]) / 2), age=35, sex="female")), "vo2max")
    assert v["band"].startswith("below_avg")
    assert v["hr_vs_reference"] == pytest.approx(1.41)
    assert any("verify_before_demo" in n for n in v["notes"])
    assert v["assumptions"]


# ------------------------------------------------------------------ hrv context

def test_hrv_context_present_and_withheld_from_years():
    out = evaluate(ds_of(make_daily(30, hrv_rmssd=42.0), age=33, sex="female"))
    c = next(c for c in out["context"] if c["key"] == "hrv_rmssd_percentile")
    assert c["closest_bin"] == "30–34"
    assert "women aged 30–34" in c["note"] and "directional" in c["note"]
    assert "no published all-cause HR" in withheld(out, "hrv_rmssd")["reason"]
    assert not any(l["key"] == "hrv_rmssd" for l in out["levers"])


def test_hrv_without_sex_uses_mean():
    out = evaluate(ds_of(make_daily(30, hrv_rmssd=30.0)))
    c = next(c for c in out["context"] if c["key"] == "hrv_rmssd_percentile")
    assert any("mean of female and male" in a for a in c["assumptions"])


# ------------------------------------------------------------------ stubs + sorting

def test_stubs_withheld():
    out = evaluate(ds_of(make_daily(30, steps=8000)))
    assert "STUB" in withheld(out, "gait_speed")["reason"]
    assert "STUB" in withheld(out, "activity_fragmentation")["reason"]


def test_sorting_by_years_recoverable():
    on, off = regular_sleep()
    daily = make_daily(90, steps=3000, rhr=70, sleep_min=480, sleep_onset=on, sleep_offset=off)
    out = evaluate(ds_of(daily, age=40))
    yrs = [l["years_recoverable"] for l in out["levers"]]
    assert yrs == sorted(yrs, reverse=True)
    assert [l["key"] for l in out["levers"]][:2] == ["steps", "rhr"]
    # ties at 0 broken by confidence: medium (sleep_duration) before low (sleep_regularity)
    tail = [l["key"] for l in out["levers"] if l["years_recoverable"] == 0]
    assert tail == ["sleep_duration", "sleep_regularity"]
