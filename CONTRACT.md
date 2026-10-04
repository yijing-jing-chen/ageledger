# ageledger — build contract (read before touching any module)

**What this is.** Open, cited, device-agnostic pace-of-aging ledger. Input: a wearable export. Output: one JSON
"ledger" — a biological-age estimate when the data supports one, plus a ranked list of levers, each with the
user's value, the rule applied, the published hazard ratio, the years it is worth, and the citation.

**Two laws, inherited from OpenStrap (CONTRIBUTING.md there):**
1. Never fabricate. Missing input ⇒ the lever appears under `withheld` with a plain reason. No defaults, no imputation,
   no deriving one metric from another as a fallback.
2. Cite the method. Any number that turns into "years" names the paper (DOI) and the exact rule. If a threshold is
   our assumption rather than published, it is marked `assumption: true` in the rule data and surfaces in the output.

## Modules and owners
| path | job | depends on |
|---|---|---|
| `ageledger/schema.py` | canonical DAILY / MINUTE tables, `Dataset`, `validate()` | — (done) |
| `ageledger/ingest/{whoop,apple,oura,edge}.py` + `ingest/__init__.py:load(path)` | export file → `Dataset` | schema |
| `ageledger/levers/rules.yaml` + `levers/engine.py` | cited rules → per-lever result; Gompertz years | schema |
| `ageledger/cosinor.py` | ENMO minute series → cosinor fit → CosinorAge (numpy reimplementation of the published 5-coef model) | schema |
| `ageledger/report.py` | assemble ledger JSON from Dataset + levers + cosinor | all above |
| `app/main.py` + `app/static/index.html` | FastAPI `/api/analyze` (multipart upload) → ledger JSON; one static page renders it | report |
| `tests/synthetic.py` + `tests/test_*.py` | 90-day synthetic person generator; unit tests per lever; cosinor cross-check | all |

## Ledger JSON (output of `report.build(ds) -> dict`)
```json
{
  "version": "0.0.1",
  "source": "whoop",
  "coverage": {"days": 180, "first": "2026-04-01", "last": "2026-09-30", "minute_days": 0, "columns_present": ["steps","rhr",...]},
  "profile": {"age_years": 35, "sex": "female", "assumed": ["age_years"]},
  "bio_age": {
    "cosinor": null | {"value": 38.2, "advance_years": 3.2, "window_days": 90, "features": {"mesor":..,"amp":..,"phi":..}, "citation": {...}, "notes": [...]}
  },
  "levers": [
    {
      "key": "steps", "label": "Daily steps", "window_days": 90,
      "your_value": 6210, "unit": "steps/day", "n_days": 88,
      "band": "Q2 (5,801 median)", "reference_band": "Q4 (10,901)",
      "hr_vs_reference": 1.17, "years": +1.8,
      "years_recoverable": 1.8,
      "rule": "90-day mean steps placed in Paluch 2022 quartiles; HR relative to Q4; age<60 plateau 8–10k",
      "citation": {"short": "Paluch 2022, Lancet Public Health", "doi": "10.1016/S2468-2667(21)00302-9"},
      "assumptions": [],
      "confidence": "high|medium|low"
    }
  ],
  "withheld": [
    {"key": "vo2max", "reason": "no vo2max column in this export"},
    {"key": "hrv_rmssd", "reason": "percentile shown under `context`; no published all-cause HR per rMSSD percentile — years not computed"}
  ],
  "context": [
    {"key": "hrv_rmssd_percentile", "your_value": 41.2, "note": "≈ median for women 30–34 (Tegegne 2019, 10-s daytime ECG; wearable overnight rMSSD runs higher — percentile only)", "citation": {...}}
  ],
  "ranking_note": "levers sorted by years_recoverable desc; ties by confidence"
}
```
`years` convention: positive = older than reference (cost), negative = younger than reference (credit).
`years_recoverable` = max(0, years) — what moving to the reference band would give back.

## Gompertz conversion (same framing WHOOP describes publicly)
years = ln(HR) / ln(2) × MRDT, with **MRDT = 8.0 years** (mortality-rate doubling time; literature range 7–10).
MRDT is a single constant in `rules.yaml` (`gompertz_mrdt_years: 8.0`) marked `assumption: true`.

## Rules data (`levers/rules.yaml`) — the lookup table Jing owns
Each lever: `key, label, input (daily column), window_days, aggregate (mean|median|sri|...), min_days,
bands: [{name, lo, hi, hr}], reference_band, citation{short, doi, url}, rule_text, assumptions[], confidence,
age_dependent: optional overrides by age range`. The engine is generic; the science lives in the YAML.
Levers for v0: steps, rhr, sleep_duration, sleep_regularity (SRI), vo2max (tier via norm table), hrv (context only).
Older-adult levers (gait speed bands, activity fragmentation) are v1 — leave stubs with `status: STUB`.

## Non-goals for v0
No persistence, no accounts, no LLM. Everything computed in memory per request. No DB.
