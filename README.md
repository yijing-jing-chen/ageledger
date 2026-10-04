# AgeLedger (working name)

Your wearable already knows how fast you're aging. Here's the math.

An open, cited, device-agnostic **pace-of-aging ledger**. Upload a WHOOP / Apple Health / Oura / OpenStrap-edge
export; get back one page: a biological-age estimate when the data supports one, and a **ranked list of levers** —
your value, the published rule, the hazard ratio, the years it's worth, and the paper. Nothing stored.

Why it exists: WHOOP Age, Oura Cardiovascular Age, Garmin Fitness Age all compute this and hide the weights behind a
subscription and their own hardware. The literature behind them is public. This shows the account.

## Live

Browser build (no install, nothing leaves your browser): published as a claude.ai artifact; source in `web/`.
The Python build below is the same engine with a FastAPI wrapper, plus CosinorAge for accelerometer sources.

## Run

```bash
python3 -m venv venv && ./venv/bin/pip install -e ".[dev]"
./venv/bin/uvicorn app.main:app --reload
# open http://127.0.0.1:8000  → upload an export, or "Try the sample"
./venv/bin/python -m pytest -q
```

## How to export your data

| Device | Steps | What you get |
|---|---|---|
| **WHOOP** | app → profile (top-left) → Settings → **Data Export** → zip arrives by email | daily: recovery, RHR, HRV (rMSSD), sleep stages/onset/offset, resp rate, SpO2, skin temp, strain. **No minute-level data** → CosinorAge not available |
| **Apple Health** | Health app → profile (top-right) → **Export All Health Data** → export.zip | daily + minute steps/HR, SDNN, RHR, VO2max, sleep, resp rate, walking speed / steadiness metrics. No accelerometer ENMO |
| **Oura** | account page → export → CSV | daily readiness/sleep/activity |
| **OpenStrap edge** | Settings → export database (SQLite) | 1 Hz HR + accelerometer → **ENMO → CosinorAge works** |

## Browser build

`web/index.html` + `web/engine.js` is a vanilla-JS port of the engine driven by the same `rules.yaml`.
`node tests/parity.mjs` checks it against the Python output field by field. Open it locally with `web/_dev.sh`.

## Pipeline

```
export → ingest/ (canonical DAILY + optional MINUTE) → levers/rules.yaml via engine.py  ┐
                                                      → cosinor.py (CosinorAge, ENMO only) ┴→ report.py → JSON → app/
```

Two laws (inherited from OpenStrap): **never fabricate** — missing input is reported as withheld, never defaulted;
**cite the method** — every "years" number carries a DOI and the rule; anything we assumed is marked `assumption: true`
in `ageledger/levers/rules.yaml` and shown in the UI.

Years = ln(HR) / ln(2) × 8 (mortality-rate doubling time; assumption, literature 7–10; the same framing WHOOP
describes for WHOOP Age). Not medical advice.

## The rules (v0)

| Lever | Source |
|---|---|
| Daily steps | Paluch 2022, Lancet Public Health — DOI 10.1016/S2468-2667(21)00302-9 |
| Resting heart rate | Zhang 2016, CMAJ — DOI 10.1503/cmaj.150535 (+ Nauman 2011 JAMA trend note) |
| Sleep duration | Cappuccio 2010, Sleep 33:585 |
| Sleep regularity (SRI) | Windred 2024, Sleep — DOI 10.1093/sleep/zsad253 (cut points assumed, see yaml) |
| VO2max tier | Mandsager 2018, JAMA Netw Open — DOI 10.1001/jamanetworkopen.2018.3605 (norm table to verify) |
| HRV (context only) | Tegegne 2019, Eur J Prev Cardiol — DOI 10.1177/2047487319872567 |
| Biological age | CosinorAge — Shim et al. 2024, npj Digit Med — DOI 10.1038/s41746-024-01111-x; reference impl Apache-2.0 |

Older-adult levers (gait speed — Studenski 2011; activity fragmentation — Schrack 2019) are stubs in the yaml.

## License
MIT. CosinorAge model constants reproduced from the Apache-2.0 reference implementation with attribution.
