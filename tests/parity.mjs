// Parity: web/engine.js (browser port) vs the Python package, on the same inputs.
// Run from the repo root:  node tests/parity.mjs
// Exits non-zero on any mismatch. Numbers compare within 0.01; strings compare exactly.
import { execSync } from "node:child_process";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
const AL = require(path.join(ROOT, "web", "engine.js"));
const TOL = 0.01;

const PY = String.raw`
import json, math, sys, zipfile, importlib.util, datetime as dt
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from synthetic import synthetic_person, whoop_like
from ageledger.report import build
from ageledger.levers.engine import load_rules
from ageledger.ingest import load

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if o is pd.NaT: return None
    return o

HAVE_CAL = importlib.util.find_spec("ageledger.calendar") is not None and importlib.util.find_spec("ageledger.experiment") is not None
ICS_SAMPLE = open("data/synthetic/calendar_synthetic.ics", encoding="utf-8").read() if HAVE_CAL else None

def build_full(ds, rules, exp=None):
    if not HAVE_CAL:
        return build(ds, rules)
    return build(ds, rules, ics_text=ICS_SAMPLE, experiment=exp)

def pack(ds, L):
    return {"rows": ds.daily.to_dict(orient="records"), "columns": list(ds.daily.columns), "source": ds.source,
            "coverage": ds.coverage, "profile": {"age_years": ds.profile.age_years, "sex": ds.profile.sex,
            "height_cm": ds.profile.height_cm, "weight_kg": ds.profile.weight_kg, "notes": list(ds.profile.notes)}, "ledger": L}

rules = load_rules()
cases = {}
for name, (kw, exp) in {
        "female35_seed7": (dict(days=180, age=35, sex="female", seed=7), {"lever": "steps", "start_date": "2026-09-10"}),
        "male64_seed3": (dict(days=180, age=64, sex="male", seed=3, steps_mean=7400, rhr_mean=71, vo2max=31), {"lever": "sleep_regularity", "start_date": "2026-09-01"}),
        "nosex_noage_100d": (dict(days=100, age=None, sex=None, seed=11), {"lever": "rhr", "start_date": "2026-09-28"})}.items():
    ds = whoop_like(synthetic_person(with_minute=False, **kw))
    c = pack(ds, build_full(ds, rules, exp)); c["experiment"] = exp
    cases[name] = c

zp = "data/synthetic/whoop_synthetic_180d.zip"
ds = load(zp); ds.profile.age_years = 35.0; ds.profile.sex = "female"
with zipfile.ZipFile(zp) as zf:
    names = {n.split("/")[-1].lower(): n for n in zf.namelist()}
    cyc = zf.read(names["physiological_cycles.csv"]).decode("utf-8-sig")
    sl = zf.read(names["sleeps.csv"]).decode("utf-8-sig") if "sleeps.csv" in names else None
zexp = {"lever": "steps", "start_date": "2026-09-15"}
zipcase = pack(ds, build_full(ds, rules, zexp)); zipcase["cycles_csv"] = cyc; zipcase["sleeps_csv"] = sl; zipcase["experiment"] = zexp

ics_cases = {}
if HAVE_CAL:
    from ageledger.calendar import parse_ics
    from tests.test_calendar import ICS as ICS_TEST
    for nm, txt in {"test_calendar.ICS": ICS_TEST, "calendar_synthetic.ics": ICS_SAMPLE}.items():
        ev = parse_ics(txt, horizon_start=dt.date(2026, 10, 4), horizon_end=dt.date(2026, 10, 25))
        ics_cases[nm] = {"text": txt, "events": [{"start": e["start"].isoformat(), "end": e["end"].isoformat(),
                                                  "summary": e["summary"], "all_day": e["all_day"]} for e in ev]}

import tempfile, pathlib
from tests.test_ingest import _apple_xml, test_oura_csv
tmp = pathlib.Path(tempfile.mkdtemp())
(tmp / "export.xml").write_text(_apple_xml())
ads = load(str(tmp / "export.xml"))
test_oura_csv(tmp)
otext = (tmp / "oura_2026-09-03_trends.csv").read_text()
ods = load(str(tmp / "oura_2026-09-03_trends.csv"))
ingest_cases = {"apple_xml": {"text": _apple_xml(), **pack(ads, None)}, "oura_csv": {"text": otext, **pack(ods, None)}}

extra = {"calendar": importlib.util.find_spec("ageledger.calendar") is not None,
         "experiment": importlib.util.find_spec("ageledger.experiment") is not None}
print(json.dumps(clean({"rules": rules, "cases": cases, "zip": zipcase, "extra": extra, "ics": ics_cases,
                        "ics_sample": ICS_SAMPLE, "ingest": ingest_cases})))
`;

const raw = execSync(`${path.join(ROOT, "venv", "bin", "python")} -`, { input: PY, cwd: ROOT, maxBuffer: 256 << 20 }).toString();
const P = JSON.parse(raw);
const rules = P.rules;

// ---------------------------------------------------------------- deep diff
const INFO_ONLY = /\.(ci_low|ci_high|beyond_noise)$/;   // bootstrap CI: numpy PCG64 vs a JS PRNG, so close but not equal
const info = [];
function diff(a, b, at, out) {
  if (INFO_ONLY.test(at)) { if (a !== b) info.push([at, a, b]); return; }
  if (typeof a === "number" && typeof b === "number") { if (Math.abs(a - b) > TOL) out.push([at, a, b]); return; }
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") {
    if (!(a === b || (a == null && b == null))) out.push([at, a, b]);
    return;
  }
  if (Array.isArray(a) !== Array.isArray(b)) { out.push([at, a, b]); return; }
  if (Array.isArray(a)) {
    if (a.length !== b.length) out.push([at + ".length", a.length, b.length]);
    for (let i = 0; i < Math.min(a.length, b.length); i++) diff(a[i], b[i], `${at}[${i}]`, out);
    return;
  }
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) diff(a[k], b[k], at ? `${at}.${k}` : k, out);
}

const fmt = (v) => v === null || v === undefined ? "—" : typeof v === "number" ? String(+v.toFixed(3)) : String(v);
let failures = 0;
const table = [];
function row(caseName, field, py, js) {
  const ok = (typeof py === "number" && typeof js === "number") ? Math.abs(py - js) <= TOL : (py ?? null) === (js ?? null);
  if (!ok) failures++;
  table.push([caseName, field, fmt(py), fmt(js), ok ? "ok" : "MISMATCH"]);
}
function keyFields(name, py, js) {
  const byKey = (L) => Object.fromEntries((L.levers || []).map((l) => [l.key, l]));
  const pl = byKey(py), jl = byKey(js);
  for (const k of new Set([...Object.keys(pl), ...Object.keys(jl)])) {
    for (const f of ["your_value", "years", "years_vs_population"]) row(name, `${k}.${f}`, pl[k]?.[f], jl[k]?.[f]);
  }
  const pc = py.bio_age?.composite, jc = js.bio_age?.composite;
  row(name, "composite.value", pc?.value ?? null, jc?.value ?? null);
  row(name, "composite.pace", pc?.pace_years_per_year ?? null, jc?.pace_years_per_year ?? null);
  const pt = (pc?.trend || []).map((t) => `${t.as_of}=${t.value}`).join(" "), jt = (jc?.trend || []).map((t) => `${t.as_of}=${t.value}`).join(" ");
  row(name, "composite.trend", pt || null, jt || null);
  row(name, "fitness_age.value", py.bio_age?.fitness_age?.value ?? null, js.bio_age?.fitness_age?.value ?? null);
  const ph = (py.context || []).find((c) => c.key === "hrv_rmssd_percentile"), jh = (js.context || []).find((c) => c.key === "hrv_rmssd_percentile");
  row(name, "hrv.closest_bin", ph?.closest_bin ?? null, jh?.closest_bin ?? null);
  row(name, "withheld keys", (py.withheld || []).map((w) => w.key).join(","), (js.withheld || []).map((w) => w.key).join(","));
}

const HAVE_CAL = P.extra.calendar && P.extra.experiment;
function jsLedger(c) {
  const ds = { rows: c.rows, columns: c.columns, source: c.source, coverage: c.coverage,
    profile: Object.assign({}, c.profile, { notes: c.profile.notes.slice() }) };
  return HAVE_CAL ? AL.buildLedger(ds, rules, P.ics_sample, c.experiment, { fallbackTz: null }) : AL.buildLedger(ds, rules);
}
function calFields(name, py, js) {
  if (!HAVE_CAL) return;
  const pw = py.week_plan, jw = js.week_plan;
  (pw?.days || []).forEach((d, i) => row(name, `week ${d.date} rules`, d.matched_rules.join(",") || "(none)", (jw?.days?.[i]?.matched_rules || []).join(",") || "(none)"));
  row(name, "week_meeting_hours", pw?.week_meeting_hours ?? null, jw?.week_meeting_hours ?? null);
  const pr = py.experiments?.receipt, jr = js.experiments?.receipt;
  row(name, "receipt.status", pr?.status ?? null, jr?.status ?? null);
  (pr?.metrics || []).forEach((m, i) => {
    const jm = jr?.metrics?.[i] || {};
    row(name, `receipt.${m.metric}.delta`, m.delta, jm.delta ?? null);
    table.push([name, `receipt.${m.metric}.ci95 (info)`, m.ci_low == null ? "—" : `[${m.ci_low}, ${m.ci_high}]`, jm.ci_low == null ? "—" : `[${jm.ci_low}, ${jm.ci_high}]`, "info"]);
  });
  row(name, "proposals", (py.experiments?.proposals || []).map((p) => p.lever).join(",") || "(none)", (js.experiments?.proposals || []).map((p) => p.lever).join(",") || "(none)");
}

// ---------------------------------------------------------------- 1. engine on Python's canonical rows
const deep = [];
for (const [name, c] of Object.entries(P.cases)) {
  const js = JSON.parse(JSON.stringify(jsLedger(c)));
  keyFields(name, c.ledger, js);
  calFields(name, c.ledger, js);
  const d = []; diff(c.ledger, js, "", d);
  deep.push([name, d]);
}

// ---------------------------------------------------------------- 2. ingest: sample WHOOP zip, JS parser vs Python parser
{
  const z = P.zip;
  const jds = AL.loadWhoop(z.cycles_csv, z.sleeps_csv);
  const d = [];
  const pyCols = z.columns.filter((c) => c !== "source"), jsCols = jds.columns.filter((c) => c !== "source");
  row("zip_ingest", "columns", pyCols.join(","), jsCols.join(","));
  row("zip_ingest", "rows", z.rows.length, jds.rows.length);
  z.rows.forEach((r, i) => pyCols.forEach((c) => diff(r[c], jds.rows[i]?.[c], `rows[${i}].${c}`, d)));
  jds.profile.age_years = 35; jds.profile.sex = "female";
  const js = JSON.parse(JSON.stringify(HAVE_CAL ? AL.buildLedger(jds, rules, P.ics_sample, z.experiment, { fallbackTz: null }) : AL.buildLedger(jds, rules)));
  keyFields("zip_full_js", z.ledger, js);
  calFields("zip_full_js", z.ledger, js);
  diff(z.ledger.week_plan, js.week_plan, "week_plan", d);
  diff(z.ledger.experiments, js.experiments, "experiments", d);
  // notes differ by design only if the JS WHOOP map adds columns the Python one lacks; compare the ledger proper
  diff(z.ledger.levers, js.levers, "levers", d);
  diff(z.ledger.bio_age, js.bio_age, "bio_age", d);
  diff(z.ledger.withheld, js.withheld, "withheld", d);
  diff(z.ledger.context, js.context, "context", d);
  deep.push(["zip_full_js", d]);
}

// ---------------------------------------------------------------- 2b. ingest: Apple Health XML (streamed in odd-sized chunks) and Oura CSV
for (const [nm, c] of Object.entries(P.ingest)) {
  let jds;
  if (nm === "apple_xml") {
    const parser = new AL.AppleParser();
    for (let i = 0, k = 1; i < c.text.length; k = (k * 7 + 3) % 97 + 1) { parser.push(c.text.slice(i, i + k)); i += k; }
    jds = AL.appleAssemble(parser.end());
  } else jds = AL.loadOura(c.text);
  const d = [];
  diff(c.rows.map((r) => { const o = Object.assign({}, r); delete o.source; return o; }),
       jds.rows.map((r) => { const o = Object.assign({}, r); delete o.source; return o; }), "rows", d);
  diff(c.coverage, jds.coverage, "coverage", d);
  diff({ age_years: c.profile.age_years, sex: c.profile.sex, height_cm: c.profile.height_cm, weight_kg: c.profile.weight_kg, notes: c.profile.notes },
       { age_years: jds.profile.age_years, sex: jds.profile.sex, height_cm: jds.profile.height_cm, weight_kg: jds.profile.weight_kg, notes: jds.profile.notes }, "profile", d);
  row(`${nm}`, "rows / minute_days", `${c.rows.length} / ${c.coverage.minute_days}`, `${jds.rows.length} / ${jds.coverage.minute_days}`);
  deep.push([`ingest ${nm}`, d]);
}

// ---------------------------------------------------------------- 3. calendar / experiments (only when the Python side has them)
if (!HAVE_CAL) {
  console.log(`\n(skip) calendar/experiment parity: Python modules present? calendar=${P.extra.calendar} experiment=${P.extra.experiment}`);
} else {
  for (const [nm, c] of Object.entries(P.ics)) {
    const ev = AL.parseICS(c.text, { horizonStart: Date.UTC(2026, 9, 4) / 864e5, horizonEnd: Date.UTC(2026, 9, 25) / 864e5 }).events
      .map((e) => ({ start: new Date(e.start).toISOString().slice(0, 19), end: new Date(e.end).toISOString().slice(0, 19), summary: e.summary, all_day: e.all_day }));
    const pyEv = c.events.map((e) => Object.assign({}, e, { start: e.start.slice(0, 19), end: e.end.slice(0, 19) }));
    row("ics", `${nm} events`, pyEv.length, ev.length);
    const d = []; diff(pyEv, ev, "events", d); deep.push([`ics ${nm}`, d]);
  }
}

// ---------------------------------------------------------------- report
const w = [12, 34, 26, 26, 8];
const line = (cells) => cells.map((c, i) => String(c).length > w[i] ? String(c).slice(0, w[i] - 1) + "…" : String(c).padEnd(w[i])).join(" ");
console.log("\n" + line(["case", "field", "python", "js", ""]));
console.log("-".repeat(110));
table.forEach((r) => console.log(line(r)));
console.log("\nDeep diff of the full ledger (every field; numbers within " + TOL + ", strings exact):");
for (const [name, d] of deep) {
  console.log(`  ${name}: ${d.length ? d.length + " difference(s)" : "identical"}`);
  d.slice(0, 15).forEach(([at, a, b]) => console.log(`    ${at}\n      py: ${JSON.stringify(a)?.slice(0, 200)}\n      js: ${JSON.stringify(b)?.slice(0, 200)}`));
  failures += d.length;
}
if (info.length) {
  console.log(`\nInfo only (bootstrap CI uses numpy's PCG64 in Python and a seeded mulberry32 in JS; same algorithm, different draws): ${info.length} field(s) differ`);
  info.slice(0, 12).forEach(([at, a, b]) => console.log(`    ${at}: py ${JSON.stringify(a)} js ${JSON.stringify(b)}`));
}
console.log(failures ? `\nFAIL: ${failures} mismatch(es)` : "\nPASS: JS engine matches Python on every compared field");
process.exit(failures ? 1 : 0);
