"""App-level tests: page serves, sample ledger is contract-shaped and ranked, bad uploads are refused."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

LEVER_KEYS = {
    "key", "label", "window_days", "your_value", "unit", "n_days", "band", "reference_band",
    "hr_vs_reference", "years", "years_recoverable", "rule", "citation", "assumptions", "confidence",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_index_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "AgeLedger" in r.text


def test_sample_is_ranked_and_contract_shaped(client):
    r = client.get("/api/sample")
    assert r.status_code == 200
    L = r.json()
    for k in ("version", "source", "coverage", "profile", "bio_age", "levers", "withheld", "context"):
        assert k in L, k
    yrs = [lv["years_recoverable"] for lv in L["levers"]]
    assert yrs == sorted(yrs, reverse=True)
    assert len(L["levers"]) == 4
    for lv in L["levers"]:
        assert LEVER_KEYS <= set(lv), LEVER_KEYS - set(lv)
        assert lv["years_recoverable"] == max(0.0, lv["years"])
        assert lv["citation"]["doi"]
        assert lv["confidence"] in ("high", "medium", "low")
    assert L["bio_age"]["cosinor"] is None
    assert any(w["key"] == "vo2max" for w in L["withheld"])


def test_analyze_rejects_unsupported_file(client):
    r = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert 400 <= r.status_code < 500
    assert "detail" in r.json()


def test_analyze_rejects_empty_file(client):
    r = client.post("/api/analyze", files={"file": ("export.csv", b"", "text/csv")})
    assert 400 <= r.status_code < 500


SYNTH_ZIP = ROOT / "data" / "synthetic" / "whoop_synthetic_180d.zip"


def test_sample_carries_composite_and_secondary_estimates(client):
    L = client.get("/api/sample").json()
    assert set(L["profile"]["entered"]) == {"age_years", "sex"}
    cp = L["bio_age"]["composite"]
    assert len(cp["contributions"]) == 4 and len(cp["trend"]) == 3
    assert abs(sum(c["years_vs_population"] for c in cp["contributions"]) - cp["advance_years"]) < 1e-6
    assert abs(cp["value"] - (cp["chronological_age"] + cp["advance_years"])) < 1e-6
    assert L["bio_age"]["fitness_age"]["vo2max"] == 33.1
    assert L["bio_age"]["hrv_age_equivalent"]["value"] == 32


@pytest.mark.skipif(not SYNTH_ZIP.exists(), reason="synthetic WHOOP zip not generated")
def test_analyze_records_user_entered_profile(client):
    with SYNTH_ZIP.open("rb") as fh:
        r = client.post(
            "/api/analyze",
            files={"file": (SYNTH_ZIP.name, fh, "application/zip")},
            data={"age": "35", "sex": "female"},
        )
    assert r.status_code == 200, r.text
    prof = r.json()["profile"]
    assert set(prof["entered"]) == {"age_years", "sex"}
    if not r.json().get("demo"):
        assert prof["age_years"] == 35 and prof["sex"] == "female"


@pytest.mark.parametrize("data", [{"age": "7"}, {"age": "abc"}, {"sex": "other"}])
def test_analyze_rejects_bad_profile_fields(client, data):
    r = client.post("/api/analyze", files={"file": ("export.csv", b"date,steps\n", "text/csv")}, data=data)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)
