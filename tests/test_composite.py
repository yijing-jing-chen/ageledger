"""Composite effective age (WHOOP-Age framing), fitness age, HRV age equivalent — from DAILY data only."""
import math

import pytest

from ageledger import composite
from ageledger.levers.engine import (
    _score_bands, _score_exponential, _mrdt, evaluate, load_rules,
)
from ageledger.report import build
from ageledger.schema import Profile
from tests.synthetic import synthetic_person, whoop_like

RULES = load_rules()


def _lever(key):
    return next(l for l in RULES["levers"] if l["key"] == key)


@pytest.fixture(scope="module")
def person180():
    return whoop_like(synthetic_person(days=180, age=35, sex="female"))


@pytest.fixture(scope="module")
def ledger180(person180):
    return build(person180)


def test_composite_arithmetic(ledger180):
    c = ledger180["bio_age"]["composite"]
    assert c is not None
    total = sum(x["years_vs_population"] for x in c["contributions"])
    assert c["advance_years"] == pytest.approx(total, abs=0.011)
    assert c["value"] == pytest.approx(35 + total, abs=0.06)
    assert c["n_levers"] == len(c["contributions"]) >= 2
    mags = [abs(x["years_vs_population"]) for x in c["contributions"]]
    assert mags == sorted(mags, reverse=True)
    assert c["confidence"] == "low"
    assert "independent" in c["assumptions"][0]
    assert any("Q2/Q3 boundary" in a for a in c["assumptions"])
    assert "whoop.com" in c["citation"]["url"]


def test_lever_years_untouched_and_population_fields(ledger180):
    for l in ledger180["levers"]:
        assert l["years_recoverable"] == round(max(0.0, l["years"]), 2)
        assert {"hr_vs_population", "years_vs_population", "population_band"} <= set(l)
        assert l["years_vs_population"] == pytest.approx(
            math.log(l["hr_vs_population"]) / math.log(2) * 8.0, abs=0.02)


def test_trend_points(ledger180):
    c = ledger180["bio_age"]["composite"]
    assert len(c["trend"]) >= 3
    dates = [p["as_of"] for p in c["trend"]]
    assert dates == sorted(dates)
    assert c["trend"][-1]["value"] == c["value"]
    assert c["pace_years_per_year"] is not None


def test_trend_short_history():
    L = build(whoop_like(synthetic_person(days=90, age=35, sex="female")))
    c = L["bio_age"]["composite"]
    assert c is not None
    assert len(c["trend"]) <= 1
    assert c["pace_years_per_year"] is None


def test_fitness_age(ledger180, person180):
    f = ledger180["bio_age"]["fitness_age"]
    assert f is not None and 20 <= f["value"] <= 80
    assert f["sex"] == "female" and f["vo2max"] > 0
    assert any("verify_before_demo" in a for a in f["assumptions"])
    no_sex = whoop_like(synthetic_person(days=90, age=35, sex=None))
    assert composite.fitness_age(no_sex, evaluate(no_sex, RULES), RULES) is None


def test_fitness_age_clamps():
    ds = whoop_like(synthetic_person(days=90, age=35, sex="female", vo2max=60))
    f = composite.fitness_age(ds, evaluate(ds, RULES), RULES)
    assert f["value"] == 20 and any("clamped" in n for n in f["notes"])


def test_hrv_age_equivalent(ledger180):
    h = ledger180["bio_age"]["hrv_age_equivalent"]
    assert h is not None
    lo, hi = (int(x) for x in h["closest_bin"].replace("–", "-").split("-"))
    assert h["value"] == (lo + hi) // 2
    assert "NOT A VALIDATED" in h["notes"][0]
    assert composite.hrv_age_equivalent({"context": []}, RULES) is None


def test_composite_withheld_without_age():
    ds = whoop_like(synthetic_person(days=90, age=35, sex="female"))
    ds.profile = Profile(age_years=None, sex="female")
    L = build(ds)
    assert L["bio_age"]["composite"] is None
    w = next(w for w in L["withheld"] if w["key"] == "composite_age")
    assert "age" in w["reason"]


def test_composite_withheld_too_few_levers():
    lev = {"levers": [{"key": "steps", "years_vs_population": 0.5}]}
    ds = whoop_like(synthetic_person(days=90))
    assert composite.effective_age(ds, lev, RULES) is None
    assert composite.withheld_reason(ds, lev, RULES) == "only 1 levers computed (need 2)"


def test_hand_built_levers():
    mrdt = _mrdt(RULES)
    prof = Profile(age_years=35, sex="female")
    s = _score_bands(_lever("steps"), 5801, 90, mrdt, prof)
    assert s["band"].startswith("Q2")
    assert s["years_vs_population"] == pytest.approx(math.log2(0.60 / 0.574) * 8, abs=0.005)
    assert s["years_vs_population"] == pytest.approx(0.51, abs=0.01)
    r = _score_exponential(_lever("rhr"), 62, 90, mrdt, prof)
    assert r["population_band"] == "70 bpm"
    assert r["years_vs_population"] == pytest.approx(math.log2(1.09 ** ((62 - 70) / 10)) * 8, abs=0.005)
    assert r["years_vs_population"] == pytest.approx(-0.80, abs=0.01)
    # best-band years unchanged by the population reference
    assert s["years"] == pytest.approx(math.log2(0.60 / 0.47) * 8, abs=0.01)

    ds = whoop_like(synthetic_person(days=90, age=35))
    c = composite.effective_age(ds, {"levers": [s, r]}, RULES)
    assert c["value"] == pytest.approx(35 + s["years_vs_population"] + r["years_vs_population"], abs=0.06)
    assert c["contributions"][0]["key"] == "rhr"      # |−0.80| > |+0.51|


def test_steps_age60_override_uses_population():
    mrdt = _mrdt(RULES)
    s = _score_bands(_lever("steps"), 8000, 90, mrdt, Profile(age_years=65, sex="male"))
    assert s["years"] == 0.0
    assert s["years_vs_population"] == pytest.approx(math.log2(0.47 / 0.574) * 8, abs=0.01)
