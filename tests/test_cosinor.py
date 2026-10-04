"""CosinorAge reimplementation vs the reference package (cosinorage 1.0.8) and the synthetic person."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ageledger.cosinor import (NO_ENMO_REASON, CosinorFitError, cosinor_age, estimate, fit_cosinor,
                               withheld_reason)
from ageledger.schema import validate
from tests.synthetic import ENMO_AMP, ENMO_MESOR, ENMO_PEAK_H, synthetic_person, whoop_like


@pytest.fixture(scope="module")
def person():
    return synthetic_person()


def _deterministic(days, mesor, amp, peak_h, start="2026-03-02"):
    """Same series the reference harness fed to cosinorage 1.0.8 (midnight-aligned, full days)."""
    ts = pd.date_range(start, periods=days * 1440, freq="min")
    m = np.arange(days * 1440) % 1440
    k = np.arange(days * 1440)
    y = (mesor + amp * np.cos(2 * np.pi * (m - peak_h * 60) / 1440)
         + 6.0 * np.cos(2 * 2 * np.pi * m / 1440 + 1.0) + 4.0 * np.sin(0.37 * k))
    return pd.DataFrame({"ts": ts, "enmo_mg": y})


# Computed ONCE on 2026-10-04 by running the real cosinorage 1.0.8 (cosinor_multiday -> CosinorPy 3.1
# test_cosinor_single, then bioages.CosinorAge) on _deterministic(...) in a scratch venv.
REFERENCE = [
    # (days, mesor, amp, peak_h, age, sex) -> ref mesor, amplitude, acrophase (rad), cosinorage
    ((7, 30.0, 25.0, 14.0, 35.0, "female"),
     (30.002075497273736, 24.996411201963333, -3.669638243072865, 38.28767346253804)),
    ((5, 22.0, 18.0, 20.0, 52.0, "male"),
     (22.000023281151684, 18.000025469390298, -5.240353249220028, 57.61179335485858)),
    ((4, 15.0, 8.0, 10.0, 70.0, "unknown"),
     (15.000859756865715, 7.998531489906194, -2.6222451715083035, 74.68227415599449)),
]


# (1) fit recovers the rhythm --------------------------------------------------------------------
def test_fit_recovers_pure_rhythm():
    rng = np.random.default_rng(1)
    df = _deterministic(10, 30.0, 25.0, 14.0)
    df["enmo_mg"] = 30 + 25 * np.cos(2 * np.pi * ((df.ts.dt.hour * 60 + df.ts.dt.minute) - 840) / 1440) \
        + rng.normal(0, 5, len(df))
    f = fit_cosinor(df)
    assert f["mesor"] == pytest.approx(30.0, abs=0.2)
    assert f["amplitude"] == pytest.approx(25.0, abs=0.2)
    # reference t starts at 1 at 00:00, so its reported peak sits 1 minute after the true one
    assert f["acrophase_hours"] == pytest.approx(14.0 + 1 / 60, abs=0.03)
    assert -2 * math.pi < f["acrophase_rad"] <= 0
    assert f["n_days"] == 10 and f["n_minutes"] == 14400


def test_fit_recovers_synthetic_person(person):
    f = fit_cosinor(person.minute)
    assert f["n_days"] == 88                      # partial first and last day trimmed
    # the sleep floor (~1.5 mg, 23:30-07:00) flattens the trough, so the cosine fit drifts a little
    assert f["mesor"] == pytest.approx(ENMO_MESOR, abs=4.0)
    assert f["amplitude"] == pytest.approx(ENMO_AMP, abs=5.0)
    assert f["acrophase_hours"] == pytest.approx(ENMO_PEAK_H, abs=0.6)
    assert f["r2"] > 0.8


def test_fit_drops_nan_without_shifting_phase():
    df = _deterministic(6, 30.0, 25.0, 14.0)
    full = fit_cosinor(df)
    holed = df.copy()
    holed.loc[holed.index % 7 == 0, "enmo_mg"] = np.nan      # scattered gaps, all days still >50%
    f = fit_cosinor(holed)
    assert f["n_minutes"] < full["n_minutes"]
    assert f["acrophase_rad"] == pytest.approx(full["acrophase_rad"], abs=0.01)


def test_fit_requires_three_days():
    with pytest.raises(CosinorFitError):
        fit_cosinor(_deterministic(2, 30.0, 25.0, 14.0))


# (2) matches the reference package ---------------------------------------------------------------
@pytest.mark.parametrize("case,ref", REFERENCE)
def test_matches_reference_fit_and_closed_form(case, ref):
    days, mesor, amp, peak_h, age, sex = case
    r_mesor, r_amp, r_phi, r_age = ref
    f = fit_cosinor(_deterministic(days, mesor, amp, peak_h))
    assert f["mesor"] == pytest.approx(r_mesor, abs=1e-9)
    assert f["amplitude"] == pytest.approx(r_amp, abs=1e-9)
    assert f["acrophase_rad"] == pytest.approx(r_phi, abs=1e-9)
    # closed form alone, fed the reference's own features
    ca = cosinor_age(r_mesor, r_amp, r_phi, age, sex)
    assert ca["value"] == pytest.approx(r_age, abs=1e-9)
    assert ca["advance_years"] == pytest.approx(r_age - age, abs=1e-9)
    assert ca["model"] == {"female": "female", "male": "male"}.get(sex, "generic")


# (3) withheld when there is no ENMO ---------------------------------------------------------------
def test_whoop_like_is_withheld(person):
    w = whoop_like(person)
    assert validate(w) == []
    assert estimate(w) is None
    assert withheld_reason(w) == NO_ENMO_REASON


def test_steps_and_hr_never_stand_in_for_enmo():
    ds = synthetic_person(days=10, seed=3)
    ds.minute = ds.minute.drop(columns=["enmo_mg"]).assign(steps=5.0)
    assert estimate(ds) is None
    assert withheld_reason(ds) == NO_ENMO_REASON


def test_missing_age_is_withheld():
    ds = synthetic_person(days=10, seed=4)
    ds.profile.age_years = None
    assert estimate(ds) is None
    assert "age" in withheld_reason(ds)


# (4) full synthetic person gives a plausible value ------------------------------------------------
def test_estimate_full_synthetic(person):
    out = estimate(person)
    assert out is not None and withheld_reason(person) is None
    assert 15 < out["value"] < 70
    assert math.isfinite(out["advance_years"])
    assert out["model"] == "female"
    assert out["window_days"] == 88
    assert set(out["features"]) >= {"mesor", "amp", "phi"}
    assert out["citation"]["doi"] == "10.1038/s41746-024-01111-x"
    assert out["notes"]
