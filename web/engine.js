/* AgeLedger engine: a client-side port of the Python package in ageledger/.
 *
 * Everything runs in the page. Nothing is uploaded. The science lives in rules.yaml (fetched next to this
 * file); this file only ingests exports, windows, aggregates, looks up bands and converts hazard ratios to
 * years, exactly as ageledger/levers/engine.py and ageledger/composite.py do.
 *
 * Browser: <script src="./engine.js"> exposes window.AgeLedger. Needs JSZip (zip exports) and js-yaml
 * (rules.yaml) as globals. Node: require()/vm-load it; module.exports is the same object, and the pure
 * parts (buildLedger, evaluate, sri, parseICS, weekPlan, receipt, ...) need neither library.
 */
(function (root, factory) {
  "use strict";
  var api = factory(root);
  if (typeof module === "object" && module && module.exports) module.exports = api;
  if (root) root.AgeLedger = api;
})(typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : this), function (root) {
  "use strict";

  var VERSION = "0.0.1";
  var DAY = 864e5, MINUTE = 6e4, HOUR = 36e5;

  // ======================================================================= numbers, formatting (Python parity)

  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function nn(v) { return v !== null && v !== undefined && !(typeof v === "number" && isNaN(v)); }
  function uniq(a) { return Array.from(new Set(a)); }
  function clean(s) { return String(s == null ? "" : s).split(/\s+/).filter(Boolean).join(" "); }

  /** Python's round(x, n): exact decimal expansion of the double, ties to even. */
  function pyRound(x, n) {
    n = n || 0;
    if (!isFinite(x)) return x;
    var neg = x < 0, ax = Math.abs(x);
    if (ax >= 1e21) return x;
    var s = ax.toFixed(100), dot = s.indexOf(".");
    var ip = s.slice(0, dot), fp = s.slice(dot + 1);
    var keep = fp.slice(0, n), rest = fp.slice(n);
    var first = rest.charCodeAt(0) - 48, up;
    if (first > 5) up = true;
    else if (first < 5) up = false;
    else if (/[1-9]/.test(rest.slice(1))) up = true;
    else {
      var last = n > 0 ? keep.charCodeAt(n - 1) - 48 : ip.charCodeAt(ip.length - 1) - 48;
      up = last % 2 === 1;
    }
    var big = BigInt(ip + keep);
    if (up) big += 1n;
    var str = big.toString();
    while (str.length < n + 1) str = "0" + str;
    var res = Number(n ? str.slice(0, str.length - n) + "." + str.slice(str.length - n) : str);
    return neg ? -res : res;
  }
  /** Python f"{x:.nf}" */
  function fmtFixed(x, n) { return pyRound(x, n).toFixed(n); }
  /** Python f"{x:g}" for ordinary magnitudes. */
  function fmtG(x) { if (!isNum(x)) return String(x); return String(Number(x.toPrecision(6))); }
  /** Python repr() of a float. */
  function pyFloat(x) { if (!isNum(x)) return String(x); return Number.isInteger(x) && Math.abs(x) < 1e16 ? x.toFixed(1) : String(x); }

  /** numpy's pairwise summation (what pandas .mean() uses), so means match to the last bit. */
  function npSum(a, lo, n) {
    if (lo === undefined) { lo = 0; n = a.length; }
    var i, res;
    if (n < 8) { res = 0; for (i = 0; i < n; i++) res += a[lo + i]; return res; }
    if (n <= 128) {
      var r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
      for (i = 8; i < n - (n % 8); i += 8) for (var j = 0; j < 8; j++) r[j] += a[lo + i + j];
      res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      for (; i < n; i++) res += a[lo + i];
      return res;
    }
    var n2 = Math.floor(n / 2); n2 -= n2 % 8;
    return npSum(a, lo, n2) + npSum(a, lo + n2, n - n2);
  }
  function mean(a) { return a.length ? npSum(a) / a.length : NaN; }
  function median(a) {
    if (!a.length) return NaN;
    var s = a.slice().sort(function (x, y) { return x - y; }), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }
  function stableSort(arr, cmp) {
    return arr.map(function (v, i) { return [v, i]; })
      .sort(function (a, b) { return cmp(a[0], b[0]) || a[1] - b[1]; })
      .map(function (p) { return p[0]; });
  }

  // ======================================================================= dates (wall clock as UTC ms, no DST)

  function dayNum(s) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(s == null ? "" : s));
    return m ? Date.UTC(+m[1], +m[2] - 1, +m[3]) / DAY : NaN;
  }
  function dayStr(n) { return new Date(n * DAY).toISOString().slice(0, 10); }
  var WALL_RE = /^\s*(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?/;
  /** Timestamp string -> naive local wall clock (ms, as if UTC); any UTC offset suffix is ignored. */
  function wallMs(s) {
    if (s == null) return NaN;
    var m = WALL_RE.exec(String(s));
    if (!m) return NaN;
    var mo = +m[2], d = +m[3], h = m[4] ? +m[4] : 0, mi = m[5] ? +m[5] : 0, se = m[6] ? +m[6] : 0;
    if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || se > 59) return NaN;
    return Date.UTC(+m[1], mo - 1, d, h, mi, se);
  }
  var APPLE_TS = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})/;
  function appleMs(s) {
    var m = APPLE_TS.exec(s || "");
    return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]) : NaN;
  }
  function isoWall(ms) { return isNum(ms) ? new Date(ms).toISOString().slice(0, 19) : null; }
  function msDay(ms) { return Math.floor(ms / DAY); }
  function hhmm(ms) { var d = new Date(ms); return pad2(d.getUTCHours()) + ":" + pad2(d.getUTCMinutes()); }
  function pad2(n) { return (n < 10 ? "0" : "") + n; }

  // ======================================================================= schema

  var DAILY_COLUMNS = ["date", "steps", "rhr", "hrv_rmssd", "hrv_sdnn", "sleep_min", "sleep_onset", "sleep_offset",
    "sleep_efficiency", "sleep_deep_min", "sleep_rem_min", "resp_rate", "spo2", "skin_temp_c", "vo2max", "strain",
    "recovery_pct", "sleep_need_min", "sleep_debt_min", "sleep_consistency_pct",
    "active_kcal", "walking_speed_ms", "walking_double_support_pct", "walking_step_length_m", "walking_asymmetry_pct",
    "source"];
  var TEXT_DAILY = { date: 1, sleep_onset: 1, sleep_offset: 1, source: 1 };

  /** Same job as ingest/_common.finalize: keep value columns, drop empty rows, sort, coverage. */
  function finalize(rows, profile, source, minuteDays) {
    rows = rows.filter(function (r) { return typeof r.date === "string" && isNum(dayNum(r.date)); });
    var present = DAILY_COLUMNS.filter(function (c) {
      if (c === "date" || c === "source") return false;
      return rows.some(function (r) { return nn(r[c]) && (TEXT_DAILY[c] || isNum(Number(r[c]))); });
    });
    var out = rows.map(function (r) {
      var o = { date: r.date };
      present.forEach(function (c) {
        var v = r[c];
        if (TEXT_DAILY[c]) o[c] = nn(v) ? v : null;
        else { v = v == null ? NaN : Number(v); o[c] = isNum(v) ? v : null; }
      });
      return o;
    }).filter(function (o) { return present.some(function (c) { return o[c] !== null; }); });
    out.sort(function (a, b) { return a.date < b.date ? -1 : a.date > b.date ? 1 : 0; });
    out.forEach(function (o) { o.source = source; });
    var coverage = {
      days: out.length,
      first: out.length ? out[0].date : null,
      last: out.length ? out[out.length - 1].date : null,
      columns_present: present,
      minute_days: minuteDays || 0
    };
    return { rows: out, columns: ["date"].concat(present, ["source"]), profile: profile, source: source, coverage: coverage };
  }

  // ======================================================================= CSV + tolerant columns

  function parseCSV(text, sep) {
    sep = sep || ",";
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
    var rows = [], row = [], field = "", q = false, i = 0, n = text.length, start;
    while (i < n) {
      var c = text[i];
      if (q) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
          q = false; i++; continue;
        }
        start = i; while (i < n && text[i] !== '"') i++;
        field += text.slice(start, i); continue;
      }
      if (c === '"' && field === "") { q = true; i++; continue; }
      if (c === sep) { row.push(field); field = ""; i++; continue; }
      if (c === "\n" || c === "\r") {
        row.push(field); rows.push(row); row = []; field = "";
        if (c === "\r" && text[i + 1] === "\n") i++;
        i++; continue;
      }
      start = i;
      while (i < n && text[i] !== sep && text[i] !== "\n" && text[i] !== "\r" && text[i] !== '"') i++;
      if (start === i) { field += c; i++; } else field += text.slice(start, i);
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    return rows.filter(function (r) { return !(r.length === 1 && r[0].trim() === ""); });
  }
  function normName(s) { return String(s).replace(/﻿/g, "").trim().toLowerCase().replace(/\s+/g, " "); }
  /** Exact (case/space-insensitive) match on any pattern first, then substring. Returns column index or -1. */
  function findCol(header, patterns, exclude) {
    var normed = header.map(normName), pats = patterns.map(normName), excl = (exclude || []).map(normName), i, p;
    for (p = 0; p < pats.length; p++) for (i = 0; i < normed.length; i++) if (normed[i] === pats[p]) return i;
    for (p = 0; p < pats.length; p++) for (i = 0; i < normed.length; i++) {
      if (normed[i].indexOf(pats[p]) >= 0 && !excl.some(function (e) { return normed[i].indexOf(e) >= 0; })) return i;
    }
    return -1;
  }
  function toNum(v) {
    if (v == null) return NaN;
    if (typeof v === "number") return v;
    var t = String(v).trim().replace(/%/g, "").replace(/,/g, "");
    if (t === "") return NaN;
    var x = Number(t);
    return isNaN(x) ? NaN : x;
  }
  function cell(r, i) { var v = i >= 0 ? r[i] : undefined; return v === undefined || v === "" ? null : v; }
  function csvTable(text, sep) {
    var rows = parseCSV(text, sep);
    if (!rows.length) return { header: [], data: [] };
    var header = rows[0];
    var data = rows.slice(1).map(function (r) { return header.map(function (_, i) { return cell(r, i); }); });
    return { header: header, data: data };
  }

  /** One row per date: keep the row with the largest `prefer`, then fill gaps from same-date rows (pandas groupby.first). */
  function dedupDates(rows, cols, prefer) {
    var arr = rows;
    if (prefer && cols.indexOf(prefer) >= 0) {
      arr = stableSort(rows, function (a, b) {
        var x = a[prefer], y = b[prefer], xn = isNum(x), yn = isNum(y);
        if (!xn && !yn) return 0; if (!xn) return 1; if (!yn) return -1;
        return y - x;
      });
    }
    var by = new Map();
    arr.forEach(function (r) {
      var g = by.get(r.date);
      if (!g) { g = { date: r.date }; by.set(r.date, g); }
      cols.forEach(function (c) { if (!nn(g[c]) && nn(r[c])) g[c] = r[c]; });
    });
    return Array.from(by.keys()).sort().map(function (d) {
      var g = by.get(d); cols.forEach(function (c) { if (!nn(g[c])) g[c] = null; }); return g;
    });
  }
  function combineFirst(a, b, cols) {
    var by = new Map();
    a.forEach(function (r) { by.set(r.date, Object.assign({}, r)); });
    b.forEach(function (r) {
      var g = by.get(r.date);
      if (!g) { by.set(r.date, Object.assign({}, r)); return; }
      cols.forEach(function (c) { if (!nn(g[c]) && nn(r[c])) g[c] = r[c]; });
    });
    return Array.from(by.keys()).sort().map(function (d) { return by.get(d); });
  }

  // ======================================================================= WHOOP (ingest/whoop.py)

  var WHOOP_MAP = [
    ["rhr", ["resting heart rate (bpm)", "resting heart rate"], []],
    ["hrv_rmssd", ["heart rate variability (ms)", "heart rate variability", "hrv"], []],
    ["skin_temp_c", ["skin temp (celsius)", "skin temp"], []],
    ["spo2", ["blood oxygen %", "blood oxygen", "spo2"], []],
    ["strain", ["day strain"], []],
    ["resp_rate", ["respiratory rate (rpm)", "respiratory rate"], []],
    ["sleep_min", ["asleep duration (min)", "asleep duration"], []],
    ["sleep_deep_min", ["deep (sws) duration (min)", "deep (sws) duration", "sws duration", "deep sleep duration"], []],
    ["sleep_rem_min", ["rem duration (min)", "rem duration", "rem sleep duration"], []],
    ["sleep_efficiency", ["sleep efficiency %", "sleep efficiency"], []],
    ["steps", ["steps", "step count"], []],
    ["vo2max", ["vo2 max", "vo2max", "vo2"], []],
    ["recovery_pct", ["recovery score %", "recovery score"], []],
    ["sleep_need_min", ["sleep need (min)", "sleep need"], []],
    ["sleep_debt_min", ["sleep debt (min)", "sleep debt"], []],
    ["sleep_consistency_pct", ["sleep consistency %", "sleep consistency"], []]
  ];
  var WHOOP_SLEEP_COLS = ["sleep_min", "sleep_deep_min", "sleep_rem_min", "sleep_efficiency", "resp_rate",
    "sleep_onset", "sleep_offset", "sleep_need_min", "sleep_debt_min", "sleep_consistency_pct"];

  function whoopCanonical(t) {
    var header = t.header, data = t.data, cols = [];
    var out = data.map(function () { return {}; });
    WHOOP_MAP.forEach(function (m) {
      var canon = m[0], ci = findCol(header, m[1], m[2]);
      if (ci < 0) return;
      var v = data.map(function (r) { return toNum(r[ci]); });
      var nc = normName(header[ci]);
      if (canon === "skin_temp_c" && (nc.indexOf("fahrenheit") >= 0 || nc.indexOf("(f)") >= 0)) v = v.map(function (x) { return (x - 32) * 5 / 9; });
      if (canon === "sleep_efficiency") {
        var ok = v.filter(function (x) { return !isNaN(x); });
        if (ok.length && ok.every(function (x) { return x <= 1; })) v = v.map(function (x) { return x * 100; });
      }
      cols.push(canon);
      v.forEach(function (x, i) { out[i][canon] = isNaN(x) ? null : x; });
    });
    var onC = findCol(header, ["sleep onset"]), wakeC = findCol(header, ["wake onset"]), startC = findCol(header, ["cycle start time", "cycle start"]);
    if (onC >= 0) cols.push("sleep_onset");
    if (wakeC >= 0) cols.push("sleep_offset");
    var rows = [];
    data.forEach(function (r, i) {
      var on = onC >= 0 ? wallMs(r[onC]) : NaN, wake = wakeC >= 0 ? wallMs(r[wakeC]) : NaN;
      if (onC >= 0) out[i].sleep_onset = isoWall(on);
      if (wakeC >= 0) out[i].sleep_offset = isoWall(wake);
      // date = local day the wake onset falls on; cycles without sleep fall back to (cycle start + 12h).date
      var d = isNum(wake) ? msDay(wake) : NaN;
      if (!isNum(d) && startC >= 0) { var st = wallMs(r[startC]); if (isNum(st)) d = msDay(st + 12 * HOUR); }
      if (isNum(d)) { out[i].date = dayStr(d); rows.push(out[i]); }
    });
    return { rows: rows, cols: cols };
  }

  function loadWhoop(cyclesText, sleepsText) {
    var cyc = whoopCanonical(csvTable(cyclesText));
    var daily = dedupDates(cyc.rows, cyc.cols, "sleep_min"), cols = cyc.cols.slice();
    var notes = ["WHOOP export carries no age/sex/height/weight — profile must be supplied by the user"];
    if (sleepsText) {
      var sl = csvTable(sleepsText);
      if (sl.data.length) {
        var napC = findCol(sl.header, ["nap"]);
        if (napC >= 0) sl.data = sl.data.filter(function (r) {
          return ["true", "1", "yes", "y", "t"].indexOf(String(r[napC] == null ? "" : r[napC]).trim().toLowerCase()) < 0;
        });
        if (sl.data.length) {
          var sc = whoopCanonical(sl);
          var keep = WHOOP_SLEEP_COLS.filter(function (c) { return sc.cols.indexOf(c) >= 0; });
          var scRows = dedupDates(sc.rows.map(function (r) {
            var o = { date: r.date }; keep.forEach(function (c) { o[c] = r[c]; }); return o;
          }), keep, "sleep_min");
          cols = uniq(cols.concat(keep));
          daily = combineFirst(daily, scRows, cols);
        }
      }
    }
    return finalize(daily, { age_years: null, sex: null, height_cm: null, weight_kg: null, notes: notes }, "whoop", 0);
  }

  // ======================================================================= Oura (ingest/oura.py)

  var OURA_MAP = [
    ["rhr", ["average resting heart rate", "resting heart rate"], ["lowest", "score"], "num"],
    ["hrv_rmssd", ["average hrv", "hrv"], ["balance", "score"], "num"],
    ["sleep_min", ["total sleep duration", "total sleep"], [], "dur"],
    ["sleep_deep_min", ["deep sleep duration", "deep sleep"], ["score"], "dur"],
    ["sleep_rem_min", ["rem sleep duration", "rem sleep"], ["score"], "dur"],
    ["sleep_efficiency", ["sleep efficiency"], ["score"], "pct"],
    ["resp_rate", ["respiratory rate", "average breath", "breathing rate"], ["score"], "num"],
    ["steps", ["steps"], ["score"], "num"],
    ["active_kcal", ["activity calories", "active calories"], [], "num"],
    ["spo2", ["average spo2", "spo2 percentage", "spo2"], ["score", "breathing"], "pct"],
    ["vo2max", ["vo2 max", "vo2max", "vo2"], [], "num"]
  ];
  function loadOura(text) {
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
    var head = text.split(/\r?\n/, 1)[0] || "";
    var sep = (head.split(";").length > head.split(",").length) ? ";" : ",";
    var t = csvTable(text, sep);
    var dateC = findCol(t.header, ["date", "day", "summary date"]);
    if (dateC < 0) throw new Error("Oura CSV without a 'date' column");
    var cols = [];
    var out = t.data.map(function (r) { var ms = wallMs(r[dateC]); return { date: isNum(ms) ? dayStr(msDay(ms)) : null }; });
    OURA_MAP.forEach(function (m) {
      var ci = findCol(t.header, m[1], m[2]);
      if (ci < 0) return;
      var v = t.data.map(function (r) { return toNum(r[ci]); }), n = normName(t.header[ci]);
      if (m[3] === "dur") {
        if (n.indexOf("hour") >= 0 || n.indexOf("(h)") >= 0) v = v.map(function (x) { return x * 60; });
        else if (n.indexOf("min") < 0) v = v.map(function (x) { return x / 60; });   // ASSUMPTION (as in oura.py): seconds
      } else if (m[3] === "pct") {
        var ok = v.filter(function (x) { return !isNaN(x); });
        if (ok.length && ok.every(function (x) { return x <= 1; })) v = v.map(function (x) { return x * 100; });
      }
      cols.push(m[0]);
      v.forEach(function (x, i) { out[i][m[0]] = isNaN(x) ? null : x; });
    });
    var onC = findCol(t.header, ["bedtime start", "bedtime_start", "sleep start"]);
    var offC = findCol(t.header, ["bedtime end", "bedtime_end", "sleep end"]);
    if (onC >= 0) { cols.push("sleep_onset"); t.data.forEach(function (r, i) { out[i].sleep_onset = isoWall(wallMs(r[onC])); }); }
    if (offC >= 0) { cols.push("sleep_offset"); t.data.forEach(function (r, i) { out[i].sleep_offset = isoWall(wallMs(r[offC])); }); }
    out = out.filter(function (r) { return r.date; });
    var daily = dedupDates(out, cols, cols.indexOf("sleep_min") >= 0 ? "sleep_min" : null);
    return finalize(daily, { age_years: null, sex: null, height_cm: null, weight_kg: null,
      notes: ["Oura daily CSV carries no age/sex — profile must be supplied by the user"] }, "oura", 0);
  }

  // ======================================================================= Apple Health (ingest/apple.py)

  var HK = "HKQuantityTypeIdentifier";
  var QTYPES = {};
  [["StepCount", "steps"], ["HeartRate", "hr"], ["RestingHeartRate", "rhr"], ["HeartRateVariabilitySDNN", "hrv_sdnn"],
    ["RespiratoryRate", "resp_rate"], ["OxygenSaturation", "spo2"], ["VO2Max", "vo2max"], ["ActiveEnergyBurned", "active_kcal"],
    ["WalkingSpeed", "walking_speed_ms"], ["WalkingDoubleSupportPercentage", "walking_double_support_pct"],
    ["WalkingStepLength", "walking_step_length_m"], ["WalkingAsymmetryPercentage", "walking_asymmetry_pct"],
    ["AppleSleepingWristTemperature", "skin_temp_c"], ["Height", "_height"], ["BodyMass", "_weight"]
  ].forEach(function (p) { QTYPES[HK + p[0]] = p[1]; });
  var SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis";
  var SV = "HKCategoryValueSleepAnalysis";
  var ASLEEP = new Set([SV + "AsleepCore", SV + "AsleepDeep", SV + "AsleepREM", SV + "AsleepUnspecified", SV + "Asleep"]);
  var STAGES = new Set([SV + "AsleepCore", SV + "AsleepDeep", SV + "AsleepREM"]);
  var INBED = SV + "InBed", AWAKE = SV + "Awake";
  var SESSION_GAP = 2 * HOUR;   // ASSUMPTION (as in apple.py): >2h without any sleep record starts a new session

  var TAG_RE = /<(\/?)(Record|Correlation|Me|ExportDate)\b([^>]*)>/g;
  var A = {
    type: /\btype="([^"]*)"/, sourceName: /\bsourceName="([^"]*)"/, unit: /\bunit="([^"]*)"/,
    value: /\bvalue="([^"]*)"/, startDate: /\bstartDate="([^"]*)"/, endDate: /\bendDate="([^"]*)"/
  };
  function xmlUnescape(s) {
    if (s.indexOf("&") < 0) return s;
    return s.replace(/&(#x[0-9a-fA-F]+|#\d+|amp|lt|gt|quot|apos);/g, function (_, e) {
      if (e[0] === "#") return String.fromCodePoint(e[1] === "x" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10));
      return { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" }[e];
    });
  }
  function attr(attrs, name) { var m = A[name].exec(attrs); return m ? xmlUnescape(m[1]) : null; }
  function allAttrs(attrs) {
    var o = {}, re = /([\w:.-]+)="([^"]*)"/g, m;
    while ((m = re.exec(attrs))) o[m[1]] = xmlUnescape(m[2]);
    return o;
  }

  /** Streaming scanner over export.xml text: only direct children of <HealthData> (Correlation copies skipped). */
  function AppleParser() {
    this.buf = "";
    this.corr = 0;
    this.qty = {};
    this.steps = [];
    this.hrDates = new Set();
    this.hrOffsets = new Set();
    this.sleep = [];
    this.me = {};
    this.exportDate = null;
  }
  AppleParser.prototype.push = function (text) {
    this.buf += text;
    var cut = this.buf.lastIndexOf(">");
    if (cut < 0) return;
    var s = this.buf.slice(0, cut + 1);
    this.buf = this.buf.slice(cut + 1);
    TAG_RE.lastIndex = 0;
    var m;
    while ((m = TAG_RE.exec(s))) this.tag(m[1] === "/", m[2], m[3]);
  };
  AppleParser.prototype.tag = function (closing, name, attrs) {
    if (name === "Correlation") {
      if (closing) this.corr = Math.max(0, this.corr - 1);
      else if (!/\/\s*$/.test(attrs)) this.corr++;
      return;
    }
    if (closing || this.corr > 0) return;
    if (name === "Record") {
      var tm = A.type.exec(attrs);
      if (!tm) return;
      var t = tm[1], key = QTYPES[t];
      if (key !== undefined) {
        var vs = attr(attrs, "value"), v = vs == null || vs.trim() === "" ? NaN : Number(vs);
        var ss = attr(attrs, "startDate") || "", st = appleMs(ss);
        if (isNaN(v) || !isNum(st)) return;
        var date = ss.slice(0, 10);
        if (key === "hr") { this.hrDates.add(date); this.hrOffsets.add(ss.slice(20).trim()); return; }
        var es = attr(attrs, "endDate") || "", en = appleMs(es);
        var src = attr(attrs, "sourceName") || "";
        if (key === "steps") { this.steps.push({ src: src, v: v, s: st, e: en, date: date, off: ss.slice(20).trim() }); return; }
        (this.qty[key] || (this.qty[key] = [])).push({ src: src, unit: attr(attrs, "unit") || "", v: v, s: st, e: en, date: date });
      } else if (t === SLEEP_TYPE) {
        this.sleep.push({ src: attr(attrs, "sourceName") || "", val: attr(attrs, "value") || "",
          s: appleMs(attr(attrs, "startDate")), e: appleMs(attr(attrs, "endDate")) });
      }
    } else if (name === "Me") {
      this.me = allAttrs(attrs);
    } else if (name === "ExportDate") {
      this.exportDate = allAttrs(attrs).value || null;
    }
  };
  AppleParser.prototype.end = function () { this.push(""); return this; };

  function unionMinutes(iv) {
    if (!iv.length) return 0;
    var a = iv.slice().sort(function (x, y) { return x[0] - y[0]; });
    var total = 0, cs = a[0][0], ce = a[0][1];
    for (var i = 1; i < a.length; i++) {
      if (a[i][0] > ce) { total += ce - cs; cs = a[i][0]; ce = a[i][1]; }
      else if (a[i][1] > ce) ce = a[i][1];
    }
    total += ce - cs;
    return total / 1000 / 60;
  }

  function sleepNights(rows) {
    var df = rows.map(function (r) {
      var kind = ASLEEP.has(r.val) ? "asleep" : r.val === INBED ? "inbed" : r.val === AWAKE ? "awake" : "";
      return { src: r.src, val: r.val, s: r.s, e: r.e, kind: kind };
    }).filter(function (r) { return r.kind && isNum(r.s) && isNum(r.e) && r.e > r.s; });
    if (!df.length) return [];
    df = stableSort(df, function (a, b) { return a.s - b.s; });
    var sessions = [], runEnd = -Infinity;
    df.forEach(function (r, i) {
      if (i === 0 || r.s > runEnd + SESSION_GAP) sessions.push([]);
      sessions[sessions.length - 1].push(r);
      runEnd = Math.max(runEnd, r.e);
    });
    var nights = [];
    sessions.forEach(function (g) {
      var asl = g.filter(function (r) { return r.kind === "asleep"; });
      if (!asl.length) return;
      var srcs = new Set(asl.map(function (r) { return r.src; })), cnt = new Map();
      g.forEach(function (r) { if (srcs.has(r.src)) cnt.set(r.src, (cnt.get(r.src) || 0) + 1); });
      var best = Array.from(cnt.keys()).sort(function (x, y) { return (cnt.get(y) - cnt.get(x)) || (x < y ? -1 : x > y ? 1 : 0); })[0];
      var a = asl.filter(function (r) { return r.src === best; });
      var iv = function (rs) { return rs.map(function (r) { return [r.s, r.e]; }); };
      var asleep = unionMinutes(iv(a));
      var hasStages = a.some(function (r) { return STAGES.has(r.val); });
      var inb = g.filter(function (r) { return r.kind === "inbed" && r.src === best; });
      if (!inb.length) inb = g.filter(function (r) { return r.kind === "inbed"; });
      var eff = null;
      if (inb.length) {
        var ib = unionMinutes(iv(inb));
        if (ib > 0 && asleep <= ib) eff = asleep / ib * 100;
      }
      var on = Math.min.apply(null, a.map(function (r) { return r.s; }));
      var off = Math.max.apply(null, a.map(function (r) { return r.e; }));
      nights.push({
        date: dayStr(msDay(off)),
        sleep_min: asleep,
        sleep_deep_min: hasStages ? unionMinutes(iv(a.filter(function (r) { return r.val === SV + "AsleepDeep"; }))) : null,
        sleep_rem_min: hasStages ? unionMinutes(iv(a.filter(function (r) { return r.val === SV + "AsleepREM"; }))) : null,
        sleep_onset: isoWall(on), sleep_offset: isoWall(off), sleep_efficiency: eff, _on: on, _off: off
      });
    });
    var seen = new Set();
    return stableSort(nights, function (a, b) { return b.sleep_min - a.sleep_min; })
      .filter(function (n) { if (seen.has(n.date)) return false; seen.add(n.date); return true; })
      .sort(function (a, b) { return a.date < b.date ? -1 : 1; });
  }

  function groupMean(rows, valFn, dateFn) {
    var m = new Map();
    rows.forEach(function (r) {
      var v = valFn(r), d = dateFn ? dateFn(r) : r.date;
      if (!m.has(d)) m.set(d, []);
      if (!isNaN(v)) m.get(d).push(v);
    });
    var out = new Map();
    m.forEach(function (vals, d) { out.set(d, vals.length ? mean(vals) : null); });
    return out;
  }
  function pctScale(rows) {
    var ok = rows.filter(function (r) { return !isNaN(r.v); });
    var scale = ok.length && ok.every(function (r) { return Math.abs(r.v) <= 1; });
    return function (r) { return scale ? r.v * 100 : r.v; };
  }
  function unitScale(table) {
    return function (r) { var f = table[String(r.unit).trim().toLowerCase()]; return f === undefined ? NaN : r.v * f; };
  }
  /** Per day keep the single source with the highest daily total (never sum iPhone + Watch). */
  function dedupSum(rows, valFn) {
    var per = new Map();
    rows.forEach(function (r) {
      var v = valFn ? valFn(r) : r.v;
      if (isNaN(v)) return;
      if (!per.has(r.date)) per.set(r.date, new Map());
      var m = per.get(r.date); m.set(r.src, (m.get(r.src) || 0) + v);
    });
    var total = new Map(), chosen = new Map();
    per.forEach(function (m, d) {
      var srcs = Array.from(m.keys()).sort(), best = srcs[0];
      srcs.forEach(function (s) { if (m.get(s) > m.get(best)) best = s; });
      total.set(d, m.get(best)); chosen.set(d, best);
    });
    return { total: total, chosen: chosen };
  }

  function appleAssemble(P) {
    var notes = [], daily = new Map();
    function put(col, map) { map.forEach(function (v, d) { if (!daily.has(d)) daily.set(d, { date: d }); daily.get(d)[col] = v; }); }
    var nights = sleepNights(P.sleep);
    ["sleep_min", "sleep_deep_min", "sleep_rem_min", "sleep_onset", "sleep_offset", "sleep_efficiency"].forEach(function (c) {
      var m = new Map(); nights.forEach(function (n) { m.set(n.date, n[c]); }); put(c, m);
    });
    var F = P.qty, minuteDates = new Set(), offsets = new Set();
    if (P.steps.length) {
      var ds = dedupSum(P.steps);
      put("steps", ds.total);
      P.steps.forEach(function (r) {
        if (ds.chosen.get(r.date) !== r.src) return;
        offsets.add(r.off);
        var e = isNum(r.e) ? Math.max(r.e, r.s) : r.s;
        var m0 = Math.floor(r.s / MINUTE) * MINUTE, m1 = Math.floor((e - 1000) / MINUTE) * MINUTE;
        if (m1 <= m0 || e === r.s) { minuteDates.add(dayStr(msDay(m0))); return; }
        for (var d = msDay(m0); d <= msDay(m1); d++) minuteDates.add(dayStr(d));
      });
    }
    P.hrDates.forEach(function (d) { minuteDates.add(d); });
    P.hrOffsets.forEach(function (o) { offsets.add(o); });
    ["rhr", "hrv_sdnn"].forEach(function (k) { if (F[k]) put(k, groupMean(F[k], function (r) { return r.v; })); });
    if (F.spo2) put("spo2", groupMean(F.spo2, pctScale(F.spo2)));
    if (F.vo2max) {
      var vm = new Map();
      stableSort(F.vo2max, function (a, b) { return a.s - b.s; }).forEach(function (r) { vm.set(r.date, r.v); });
      put("vo2max", vm);
    }
    if (F.active_kcal) put("active_kcal", dedupSum(F.active_kcal, unitScale({ kcal: 1, cal: 1, kj: 1 / 4.184 })).total);
    if (F.walking_speed_ms) put("walking_speed_ms", groupMean(F.walking_speed_ms, unitScale({ "m/s": 1, "km/hr": 1 / 3.6, "km/h": 1 / 3.6, "mi/hr": 0.44704, "mph": 0.44704 })));
    ["walking_double_support_pct", "walking_asymmetry_pct"].forEach(function (k) { if (F[k]) put(k, groupMean(F[k], pctScale(F[k]))); });
    if (F.walking_step_length_m) put("walking_step_length_m", groupMean(F.walking_step_length_m, unitScale({ m: 1, cm: 0.01, "in": 0.0254, ft: 0.3048 })));
    if (F.skin_temp_c) {
      var conv = F.skin_temp_c.map(function (r) {
        var v = String(r.unit).toLowerCase().indexOf("f") >= 0 ? (r.v - 32) * 5 / 9 : r.v;
        return { v: v, date: dayStr(msDay(isNum(r.e) ? r.e : r.s)) };
      });
      var inRange = conv.filter(function (r) { return r.v >= 20 && r.v <= 45; });
      if (inRange.length / conv.length > 0.9) put("skin_temp_c", groupMean(inRange, function (r) { return r.v; }));
      else notes.push("AppleSleepingWristTemperature looks relative (deviation), not absolute °C — skin_temp_c withheld");
    }
    if (F.resp_rate) {
      var rr = stableSort(F.resp_rate, function (a, b) { return a.s - b.s; });
      var fallback = groupMean(rr, function (r) { return r.v; });
      nights.forEach(function (n) {
        var vals = rr.filter(function (r) { return r.s >= n._on && r.s <= n._off; }).map(function (r) { return r.v; });
        if (vals.length) fallback.set(n.date, mean(vals));
      });
      put("resp_rate", fallback);
    }
    offsets = Array.from(offsets).filter(Boolean);
    if (offsets.length > 1) notes.push("minute ts kept as naive LOCAL wall clock: export spans " + offsets.length + " UTC offsets (" +
      offsets.sort().join(", ") + "); a single tz-aware column would shift some days by the DST hour");
    var profile = appleProfile(P, F, daily, notes);
    return finalize(Array.from(daily.values()), profile, "apple", minuteDates.size);
  }

  function appleProfile(P, F, daily, notes) {
    var prof = { age_years: null, sex: null, height_cm: null, weight_kg: null, notes: notes };
    prof.sex = { HKBiologicalSexFemale: "female", HKBiologicalSexMale: "male" }[P.me.HKCharacteristicTypeIdentifierBiologicalSex || ""] || null;
    var dob = P.me.HKCharacteristicTypeIdentifierDateOfBirth || "";
    if (dob) {
      var b = /^(\d{4})-(\d{2})-(\d{2})/.exec(dob), ref = null;
      if (P.exportDate) ref = /^(\d{4})-(\d{2})-(\d{2})/.exec(P.exportDate);
      else {
        var ds = Array.from(daily.keys()).sort();
        ref = /^(\d{4})-(\d{2})-(\d{2})/.exec(ds.length ? ds[ds.length - 1] : new Date().toISOString());
        notes.push("no <ExportDate>; age computed at the last day with data");
      }
      if (b && ref) prof.age_years = (+ref[1]) - (+b[1]) - ((+ref[2] < +b[2] || (+ref[2] === +b[2] && +ref[3] < +b[3])) ? 1 : 0);
      else notes.push("unparseable DateOfBirth '" + dob + "'");
    }
    function latest(rows) { return stableSort(rows, function (a, b) { return a.s - b.s; })[rows.length - 1]; }
    if (F._height && F._height.length) {
      var h = latest(F._height), fh = { cm: 1, m: 100, "in": 2.54, ft: 30.48 }[String(h.unit).toLowerCase()];
      prof.height_cm = fh ? h.v * fh : null;
    }
    if (F._weight && F._weight.length) {
      var w = latest(F._weight), fw = { kg: 1, g: 0.001, lb: 0.45359237, st: 6.35029318 }[String(w.unit).toLowerCase()];
      prof.weight_kg = fw ? w.v * fw : null;
    }
    return prof;
  }

  // ======================================================================= file plumbing (browser)

  function getJSZip() {
    if (root && root.JSZip) return root.JSZip;
    if (typeof require === "function") { try { return require("jszip"); } catch (e) { /* not installed */ } }
    throw new Error("JSZip is not loaded");
  }
  function getYaml() {
    if (root && root.jsyaml) return root.jsyaml;
    if (typeof require === "function") { try { return require("js-yaml"); } catch (e) { /* not installed */ } }
    throw new Error("js-yaml is not loaded");
  }
  function baseName(p) { return String(p).split("/").pop().toLowerCase(); }
  async function toBytes(f) {
    if (f instanceof Uint8Array) return f;
    if (f instanceof ArrayBuffer) return new Uint8Array(f);
    if (f && f.data) return toBytes(f.data);
    if (f && typeof f.arrayBuffer === "function") return new Uint8Array(await f.arrayBuffer());
    throw new Error("unsupported file object");
  }
  async function headBytes(f, n) {
    if (f && typeof f.slice === "function" && typeof f.arrayBuffer === "function" && !(f instanceof Uint8Array)) return new Uint8Array(await f.slice(0, n).arrayBuffer());
    return (await toBytes(f)).subarray(0, n);
  }
  function isZipHead(h) { return h.length >= 4 && h[0] === 0x50 && h[1] === 0x4b && (h[2] === 3 || h[2] === 5 || h[2] === 7); }
  function decode(bytes) { return new TextDecoder("utf-8").decode(bytes); }

  function streamZipEntry(entry, onText, onPct) {
    return new Promise(function (resolve, reject) {
      var dec = new TextDecoder("utf-8"), failed = false;
      var h = entry.internalStream("uint8array");
      h.on("data", function (chunk, meta) {
        if (failed) return;
        try { onText(dec.decode(chunk, { stream: true })); if (onPct && meta) onPct(meta.percent); }
        catch (e) { failed = true; h.pause(); reject(e); }
      }).on("error", function (e) { failed = true; reject(e); })
        .on("end", function () { if (failed) return; try { onText(dec.decode()); resolve(); } catch (e) { reject(e); } })
        .resume();
    });
  }
  async function streamBlob(file, onText, onPct) {
    if (file && typeof file.stream === "function") {
      var reader = file.stream().getReader(), dec = new TextDecoder("utf-8"), done = 0, total = file.size || 1;
      for (;;) {
        var r = await reader.read();
        if (r.done) break;
        done += r.value.length;
        onText(dec.decode(r.value, { stream: true }));
        if (onPct) onPct(100 * done / total);
      }
      onText(dec.decode());
      return;
    }
    onText(decode(await toBytes(file)));
  }

  /** Export file (File/Blob/bytes) -> dataset {rows, columns, profile, source, coverage}. */
  async function ingest(file, opts) {
    opts = opts || {};
    var progress = opts.onProgress || function () {};
    var name = String((file && file.name) || "export");
    var head = await headBytes(file, 65536);
    if (isZipHead(head)) {
      var JSZip = getJSZip();
      var zip = await JSZip.loadAsync(file instanceof Uint8Array || file instanceof ArrayBuffer ? file : (file.data || file));
      var entries = Object.keys(zip.files).filter(function (n) { return !zip.files[n].dir && n.indexOf("__MACOSX") !== 0; });
      var find = function (b) { return entries.find(function (n) { return baseName(n) === b; }); };
      var cyc = find("physiological_cycles.csv");
      if (cyc) {
        progress("Reading WHOOP cycles…");
        var sl = find("sleeps.csv");
        return loadWhoop(await zip.file(cyc).async("string"), sl ? await zip.file(sl).async("string") : null);
      }
      var xml = find("export.xml");
      if (xml) {
        var P = new AppleParser();
        await streamZipEntry(zip.file(xml), function (t) { P.push(t); }, function (pct) {
          progress("Reading Apple Health export.xml… " + Math.round(pct) + "%");
        });
        progress("Building daily rows…");
        return appleAssemble(P.end());
      }
      var csvs = entries.filter(function (n) { return /\.csv$/i.test(n); }).map(baseName).sort().slice(0, 10);
      throw new Error(name + ": archive is neither a WHOOP export (physiological_cycles.csv) nor an Apple Health export (export.xml); csv files seen: " + (csvs.join(", ") || "none"));
    }
    var asc = decode(head.subarray(0, 16));
    if (asc.indexOf("SQLite format 3") === 0) throw new Error("OpenStrap edge SQLite files need the Python version of AgeLedger; this browser build reads WHOOP, Apple Health and Oura exports.");
    var text = decode(head), low = text.toLowerCase();
    if (low.indexOf("healthdata") >= 0 && /^\s*</.test(low)) {
      var P2 = new AppleParser();
      await streamBlob(file, function (t) { P2.push(t); }, function (pct) { progress("Reading export.xml… " + Math.round(pct) + "%"); });
      return appleAssemble(P2.end());
    }
    var first = (text.replace(/^﻿/, "").split(/\r?\n/, 1)[0] || "").toLowerCase();
    if (first.indexOf("cycle start time") >= 0) return loadWhoop(decode(await toBytes(file)), null);
    var cols = first.replace(/;/g, ",").split(",").map(function (c) { return c.trim().replace(/^"|"$/g, ""); });
    if (cols.some(function (c) { return c === "date" || c === "day" || c === "summary date"; })) return loadOura(decode(await toBytes(file)));
    throw new Error(name + ": unrecognised export format (expected a WHOOP zip, Apple Health export.zip or export.xml, or an Oura CSV)");
  }

  // ======================================================================= lever engine (levers/engine.py)

  var CONFIDENCE_ORDER = { high: 0, medium: 1, low: 2 };

  function yearsFromHr(hr, mrdt) { return Math.log(hr) / Math.log(2) * mrdt; }
  function mrdtOf(rules) { var v = rules.gompertz_mrdt_years; return Number(v && typeof v === "object" ? v.value : v); }
  function assumptionTexts(entries) {
    var out = [];
    (entries || []).forEach(function (e) {
      if (e && typeof e === "object") { if ((e.assumption === undefined || e.assumption) && e.text) out.push(e.text); }
      else if (typeof e === "string") out.push(e);
    });
    return out;
  }
  function citationOf(obj) { return Object.assign({}, (obj && obj.citation) || {}); }
  function has(ds, col) { return ds.columns.indexOf(col) >= 0; }

  function spanDays(rows) {
    var first = Infinity, last = -Infinity;
    rows.forEach(function (r) { var d = dayNum(r.date); if (isNum(d)) { if (d < first) first = d; if (d > last) last = d; } });
    return { first: first, last: last, ok: isFinite(first) };
  }
  function windowRows(rows, windowDays) {
    var sp = spanDays(rows);
    if (!sp.ok) return { rows: [], m: 0 };
    var start = sp.last - (windowDays - 1);
    return {
      rows: rows.filter(function (r) { var d = dayNum(r.date); return d >= start && d <= sp.last; }),
      m: Math.min(windowDays, sp.last - sp.first + 1)
    };
  }
  function numeric(rows, col) {
    var out = [];
    rows.forEach(function (r) { var v = r[col]; if (v !== null && v !== undefined && v !== "") { v = Number(v); if (isNum(v)) out.push(v); } });
    return out;
  }

  function inBand(x, b) {
    var lo = b.lo, hi = b.hi;
    if (lo !== null && lo !== undefined) { if (b.lo_exclusive ? !(x > lo) : !(x >= lo)) return false; }
    if (hi !== null && hi !== undefined) { if (b.hi_inclusive ? !(x <= hi) : !(x < hi)) return false; }
    return true;
  }
  function findBand(x, bands) { for (var i = 0; i < (bands || []).length; i++) if (inBand(x, bands[i])) return bands[i]; return null; }
  function bandByName(bands, name) {
    var b = bands.find(function (b) { return b.name === name; });
    if (!b) throw new Error("reference band '" + name + "' not in bands");
    return b;
  }
  function bandAssumptions() {
    var out = [];
    for (var i = 0; i < arguments.length; i++) { var b = arguments[i]; if (b && b.assumption && b.assumption_text) out.push(b.assumption_text); }
    return out;
  }
  function ageDecadeKey(age) {
    if (age === null || age === undefined || age < 20 || age >= 80) return null;
    var lo = Math.floor(age / 10) * 10;
    return lo + "-" + (lo + 9);
  }
  function interpPercentile(pcts, vals, target) {
    var i;
    if (target <= pcts[0]) i = 0;
    else if (target >= pcts[pcts.length - 1]) i = pcts.length - 2;
    else { i = 0; for (var j = 0; j < pcts.length - 1; j++) if (pcts[j] <= target) i = j; }
    var p0 = pcts[i], p1 = pcts[i + 1], v0 = vals[i], v1 = vals[i + 1];
    return v0 + (target - p0) * (v1 - v0) / (p1 - p0);
  }
  function vo2maxCutpoints(rules, sex, age, tiers) {
    var norms = rules.vo2max_norms, key = ageDecadeKey(age), table = norms[sex];
    if (key === null || !table || !table[key]) return null;
    var edges = uniq(tiers.filter(function (t) { return t.pct_lo !== null && t.pct_lo !== undefined; }).map(function (t) { return t.pct_lo; }))
      .sort(function (a, b) { return a - b; });
    var out = new Map();
    edges.forEach(function (p) { out.set(p, pyRound(interpPercentile(norms.percentiles, table[key], p), 2)); });
    return out;
  }

  function populationFromBands(cfg, fallback, bands) {
    function get(k) { return cfg && cfg[k] !== null && cfg[k] !== undefined ? cfg[k] : fallback[k]; }
    var texts = assumptionTexts([get("population_assumption")]);
    if (get("population_hr") !== null && get("population_hr") !== undefined) {
      return [Number(get("population_hr")), get("population_label") || ("HR " + get("population_hr")), texts];
    }
    if (get("population_band") !== null && get("population_band") !== undefined) {
      var b = bands.find(function (b) { return b.name === get("population_band"); });
      return [Number(b.hr), b.name, texts];
    }
    return [null, null, []];
  }

  function result(lever, o) {
    var years = yearsFromHr(o.hr, o.mrdt);
    var yearsPop = o.hr_pop !== null && o.hr_pop !== undefined ? yearsFromHr(o.hr_pop, o.mrdt) : null;
    return {
      key: lever.key, label: lever.label, window_days: lever.window_days,
      your_value: o.value, unit: lever.unit === undefined ? null : lever.unit, n_days: o.n_days,
      band: o.band, reference_band: o.ref_band,
      hr_vs_reference: pyRound(o.hr, 3),
      years: pyRound(years, 2),
      years_recoverable: pyRound(Math.max(0, years), 2),
      population_band: o.pop_band === undefined ? null : o.pop_band,
      hr_vs_population: o.hr_pop !== null && o.hr_pop !== undefined ? pyRound(o.hr_pop, 3) : null,
      years_vs_population: yearsPop !== null ? pyRound(yearsPop, 2) : null,
      rule: o.rule, citation: citationOf(lever),
      assumptions: uniq((o.assumptions || []).concat(o.pop_assumptions || [])),
      confidence: lever.confidence || "low",
      notes: (o.notes || []).slice()
    };
  }

  function scoreBands(lever, value, n, mrdt, profile) {
    var bands = lever.bands, refName = lever.reference_band;
    var assumptions = assumptionTexts(lever.assumptions), rule = clean(lever.rule_text), age = profile.age_years, override = null;
    if (lever.age_dependent) {
      if (age === null || age === undefined) assumptions = assumptions.concat(assumptionTexts([lever.age_unknown_assumption]));
      else {
        for (var i = 0; i < lever.age_dependent.length; i++) {
          var ov = lever.age_dependent[i], lo = ov.age_min, hi = ov.age_max;
          if ((lo === null || lo === undefined || age >= lo) && (hi === null || hi === undefined || age < hi)) {
            bands = ov.bands; refName = ov.reference_band; override = ov;
            assumptions = assumptions.concat(assumptionTexts(ov.assumptions));
            if (ov.rule_suffix) rule += " [" + ov.rule_suffix + "]";
            break;
          }
        }
      }
    }
    var band = findBand(value, bands), ref = bandByName(bands, refName);
    var hr = band.hr / ref.hr;
    assumptions = assumptions.concat(bandAssumptions(band, ref));
    var pop = populationFromBands(override, lever, bands);
    return result(lever, { value: value, n_days: n, band: band.name, ref_band: ref.name, hr: hr, mrdt: mrdt, rule: rule,
      assumptions: assumptions, notes: lever.notes, hr_pop: pop[0] ? band.hr / pop[0] : null, pop_band: pop[1], pop_assumptions: pop[2] });
  }

  function scoreExponential(lever, value, n, mrdt) {
    var f = lever.formula, floor = f.floor === undefined || f.floor === null ? -Infinity : f.floor;
    var x = Math.max(value, floor);
    var hr = Math.pow(f.base, (x - f.reference_value) / f.per);
    var disp = findBand(value, lever.display_bands || []);
    var notes = (lever.notes || []).slice();
    if (value < floor) notes.push("value below the " + f.floor + " floor of the source curve — scored at " + f.floor);
    var popRef = f.population_reference_value;
    var hasPop = popRef !== null && popRef !== undefined;
    return result(lever, { value: value, n_days: n, band: disp ? disp.name : pyFloat(value), ref_band: lever.reference_band, hr: hr, mrdt: mrdt,
      rule: clean(lever.rule_text), assumptions: assumptionTexts(lever.assumptions), notes: notes,
      hr_pop: hasPop ? Math.pow(f.base, (x - popRef) / f.per) : null,
      pop_band: hasPop ? (fmtG(popRef) + " " + (lever.unit || "")).trim() : null,
      pop_assumptions: hasPop ? assumptionTexts([lever.population_assumption]) : [] });
  }

  function scoreNormTier(lever, value, n, mrdt, profile, rules) {
    var tiers = lever.tiers, sex = profile.sex, age = profile.age_years;
    var cuts = vo2maxCutpoints(rules, sex, age, tiers);
    if (cuts === null) return [null, "no " + lever.key + " norms for sex=" + (sex === null || sex === undefined ? "None" : "'" + sex + "'") +
      ", age=" + (age === null || age === undefined ? "None" : pyFloat(Number(age))) + " (table covers 20–79, male/female)"];
    var tier = null;
    for (var i = 0; i < tiers.length; i++) {
      var t = tiers[i];
      var lo = t.pct_lo !== null && t.pct_lo !== undefined ? cuts.get(t.pct_lo) : null;
      var hi = t.pct_hi !== null && t.pct_hi !== undefined ? cuts.get(t.pct_hi) : null;
      if ((lo === null || value >= lo) && (hi === null || value < hi)) { tier = t; break; }
    }
    var ref = tiers.find(function (t) { return t.name === lever.reference_band; });
    var hr = tier.hr / ref.hr, norms = rules.vo2max_norms;
    var assumptions = assumptionTexts(lever.assumptions).concat(bandAssumptions(tier, ref));
    if (norms.assumption) assumptions.push(norms.assumption_text);
    var notes = (lever.notes || []).slice();
    if (norms.verify_before_demo) notes.push("vo2max norm table flagged verify_before_demo: " + clean(norms.citation));
    var pctTxt = (tier.pct_lo || 0) + "–" + (tier.pct_hi || 100) + "th pct";
    var band = tier.name + " (" + pctTxt + ", " + sex + " " + ageDecadeKey(age) + ")";
    var rule = clean(lever.rule_text) + " Cut points used (ml/kg/min): " + Array.from(cuts.keys()).sort(function (a, b) { return a - b; })
      .map(function (p) { return "p" + fmtG(p) + "=" + pyFloat(cuts.get(p)); }).join(", ");
    var pop = populationFromBands(null, lever, tiers);
    return [result(lever, { value: value, n_days: n, band: band, ref_band: ref.name, hr: hr, mrdt: mrdt, rule: rule,
      assumptions: assumptions, notes: notes, hr_pop: pop[0] ? tier.hr / pop[0] : null, pop_band: pop[1], pop_assumptions: pop[2] }), null];
  }

  function contextNearestBin(lever, value, n, profile, rules) {
    var table = rules[lever.reference_table];
    var sex = profile.sex === "female" || profile.sex === "male" ? profile.sex : null;
    var assumptions = assumptionTexts(lever.assumptions);
    if (sex === null) assumptions = assumptions.concat(assumptionTexts([lever.sex_unknown_assumption]));
    var medOf = function (b) { return sex ? b[sex] : (b.female + b.male) / 2; };
    var best = null, bestDiff = Infinity;
    table.bins.forEach(function (b) { var d = Math.abs(value - medOf(b)); if (d < bestDiff) { bestDiff = d; best = b; } });
    var med = medOf(best);
    if (best.interpolated) assumptions.push(table.interpolation_assumption_text);
    var who = sex === "female" ? "women" : sex === "male" ? "men" : "adults (F/M mean)";
    var youngest = table.bins[0], ymed = medOf(youngest), phrase;
    if (value > ymed) phrase = "your rMSSD (" + fmtG(value) + " ms) is above the median of " + who + " aged " + youngest.bin + " (" + fmtG(ymed) + " ms)";
    else phrase = "your rMSSD matches the median of " + who + " aged " + best.bin + " (" + fmtG(med) + " ms)";
    return {
      key: lever.context_key || lever.key, your_value: value, unit: lever.unit === undefined ? null : lever.unit,
      n_days: n, window_days: lever.window_days, closest_bin: best.bin, bin_median: pyRound(med, 2),
      note: phrase + " — " + (lever.notes || []).join(" "),
      rule: clean(lever.rule_text), citation: citationOf(lever), assumptions: uniq(assumptions)
    };
  }

  function formatDelta(tpl, delta) {
    return tpl.replace(/\{delta(?::([^}]*))?\}/g, function (_, spec) {
      var m = /^(\+)?\.(\d+)f$/.exec(spec || "");
      if (!m) return String(delta);
      var s = fmtFixed(delta, +m[2]);
      return m[1] && s[0] !== "-" ? "+" + s : s;
    });
  }

  function rhrTrend(lever, ds) {
    var tr = lever.trend;
    if (!tr) return null;
    var sp = spanDays(ds.rows);
    if (!sp.ok || sp.last - sp.first + 1 < tr.min_span_days) return null;
    var recent = [], prior = [];
    ds.rows.forEach(function (r) {
      var d = dayNum(r.date), v = r[lever.input];
      if (v === null || v === undefined || !isNum(Number(v))) return;
      if (d > sp.last - tr.recent_days) recent.push(Number(v));
      else if (d > sp.last - tr.recent_days - tr.prior_days) prior.push(Number(v));
    });
    if (recent.length < tr.min_days_each || prior.length < tr.min_days_each) return null;
    var delta = mean(recent) - mean(prior), flagged = delta >= tr.flag_delta;
    return {
      key: lever.key + "_trend", your_value: pyRound(delta, 1), unit: "bpm (last 90d mean − prior 90d mean)", flagged: flagged,
      note: flagged ? formatDelta(clean(tr.note), delta)
        : "RHR change last 90d vs prior 90d: " + formatDelta("{delta:+.1f}", delta) + " bpm (below the +" + tr.flag_delta + " bpm flag). Note only; no years.",
      citation: Object.assign({}, tr.citation || {})
    };
  }

  function roundValue(lever, v) { return lever.unit === "steps/day" ? pyRound(v, 0) : pyRound(v, 1); }

  /** Port of engine.evaluate(ds, rules). ds = {rows, columns, profile}. */
  function evaluate(ds, rules) {
    var mrdt = mrdtOf(rules), profile = ds.profile || {}, levers = [], withheld = [], context = [];
    (rules.levers || []).forEach(function (lever) {
      var key = lever.key;
      if (lever.status === "STUB") { withheld.push({ key: key, reason: "not implemented in v0 (STUB) — " + clean(lever.rule_text || "") }); return; }
      var inputs = Array.isArray(lever.input) ? lever.input : [lever.input];
      var missing = inputs.filter(function (c) { return !has(ds, c); });
      if (!has(ds, "date") || !ds.rows.length) { withheld.push({ key: key, reason: "export has no daily rows" }); return; }
      if (missing.length) {
        var reason = "no " + missing.join(" / ") + " column in this export";
        if (key === "hrv_rmssd" && has(ds, "hrv_sdnn")) reason += " (it has SDNN, a different metric — not substituted)";
        withheld.push({ key: key, reason: reason }); return;
      }
      var lacking = (lever.requires_profile || []).filter(function (f) { return profile[f] === null || profile[f] === undefined; });
      if (lacking.length) { withheld.push({ key: key, reason: "needs profile " + lacking.join(" and ") + " for age/sex norm tiers; not in this export" }); return; }
      var w = windowRows(ds.rows, lever.window_days), minDays = lever.min_days, value, n;
      if (lever.aggregate === "sri") {
        var sr = sleepRegularityIndex(w.rows);
        n = sr.n;
        if (n < minDays) { withheld.push({ key: key, reason: "only " + n + " of " + w.m + " nights have both sleep_onset and sleep_offset (need " + minDays + ")" }); return; }
        if (sr.sri === null) { withheld.push({ key: key, reason: "no two consecutive nights with onset/offset — SRI needs t and t+24h" }); return; }
        value = pyRound(sr.sri, 1);
      } else {
        var col = inputs[0], s = numeric(w.rows, col);
        n = s.length;
        if (n < minDays) { withheld.push({ key: key, reason: "only " + n + " of " + w.m + " days have " + col + " (need " + minDays + ")" }); return; }
        value = roundValue(lever, lever.aggregate === "mean" ? mean(s) : median(s));
      }
      var scoring = lever.scoring;
      if (scoring === "bands") levers.push(scoreBands(lever, value, n, mrdt, profile));
      else if (scoring === "exponential") {
        var res = scoreExponential(lever, value, n, mrdt), tr = rhrTrend(lever, ds);
        if (tr) { context.push(tr); if (tr.flagged) res.notes.push(tr.note); }
        levers.push(res);
      } else if (scoring === "norm_tier") {
        var nt = scoreNormTier(lever, value, n, mrdt, profile, rules);
        if (nt[0] === null) withheld.push({ key: key, reason: nt[1] }); else levers.push(nt[0]);
      } else if (scoring === "context_nearest_bin") {
        context.push(contextNearestBin(lever, value, n, profile, rules));
        withheld.push({ key: key, reason: lever.withheld_reason });
      } else throw new Error("unknown scoring '" + scoring + "' for lever " + key);
    });
    levers = stableSort(levers, function (a, b) {
      return (b.years_recoverable - a.years_recoverable) ||
        ((CONFIDENCE_ORDER[a.confidence] === undefined ? 9 : CONFIDENCE_ORDER[a.confidence]) - (CONFIDENCE_ORDER[b.confidence] === undefined ? 9 : CONFIDENCE_ORDER[b.confidence]));
    });
    return { levers: levers, withheld: withheld, context: context, constants: { gompertz_mrdt_years: mrdt } };
  }

  // ======================================================================= SRI (levers/sri.py)

  /** Builds the minute grid. Each night (row date = day the night ENDED) owns [date-1 12:00, date 12:00). */
  function sriGrid(rows) {
    var nights = [];
    (rows || []).forEach(function (r) {
      var on = wallMs(r.sleep_onset), off = wallMs(r.sleep_offset), day = dayNum(r.date);
      if (!isNum(on) || !isNum(off) || !isNum(day)) return;
      if (!(off - on > 0 && off - on < DAY)) return;
      nights.push([day, on, off]);
    });
    if (!nights.length) return { n: 0 };
    var days = uniq(nights.map(function (x) { return x[0]; })).sort(function (a, b) { return a - b; });
    var origin = days[0] * DAY - 12 * HOUR, nBlocks = days[days.length - 1] - days[0] + 2;
    var asleep = new Uint8Array(nBlocks * 1440), covered = new Uint8Array(nBlocks);
    nights.forEach(function (x) {
      covered[x[0] - days[0]] = 1;
      var a = Math.max(Math.trunc((x[1] - origin) / MINUTE), 0), b = Math.min(Math.trunc((x[2] - origin) / MINUTE), asleep.length);
      if (b > a) asleep.fill(1, a, b);
    });
    var pairs = [];
    for (var k = 0; k < nBlocks - 1; k++) {
      if (!(covered[k] && covered[k + 1])) continue;
      var s = k * 1440, same = 0;
      for (var i = 0; i < 1440; i++) if (asleep[s + i] === asleep[s + 1440 + i]) same++;
      pairs.push({ block: k, same: same });
    }
    return { n: nights.length, pairs: pairs };
  }
  function sleepRegularityIndex(rows) {
    var g = sriGrid(rows);
    if (!g.n) return { sri: null, n: 0 };
    if (!g.pairs.length) return { sri: null, n: g.n };
    var same = 0; g.pairs.forEach(function (p) { same += p.same; });
    return { sri: 200 * same / (g.pairs.length * 1440) - 100, n: g.n };
  }
  /** Per day-pair SRI values; their mean is exactly the window's SRI (each pair weighs 1440 minutes). */
  function sriPairValues(rows) {
    var g = sriGrid(rows);
    return g.n ? g.pairs.map(function (p) { return 200 * p.same / 1440 - 100; }) : [];
  }

  // ======================================================================= composite (composite.py)

  function scored(lev) { return ((lev && lev.levers) || []).filter(function (l) { return l.years_vs_population !== null && l.years_vs_population !== undefined; }); }
  function populationTexts(rules) {
    var out = {};
    (rules.levers || []).forEach(function (lv) {
      var t = assumptionTexts([lv.population_assumption]);
      (lv.age_dependent || []).forEach(function (ov) { t = t.concat(assumptionTexts([ov.population_assumption])); });
      out[lv.key] = t;
    });
    return out;
  }
  function compositeWithheldReason(ds, lev, rules) {
    if (ds.profile.age_years === null || ds.profile.age_years === undefined) return "needs chronological age — enter it above";
    return "only " + scored(lev).length + " levers computed (need " + Number(rules.composite.min_levers) + ")";
  }
  function point(lev, rules, age) {
    var levers = scored(lev);
    if (levers.length < Number(rules.composite.min_levers)) return null;
    var total = pyRound(levers.reduce(function (t, l) { return t + l.years_vs_population; }, 0), 2);
    return [pyRound(age + total, 1), total, levers];
  }
  function trend(ds, rules, ageNow, currentValue) {
    var cfg = rules.composite, step = Number(cfg.trend_step_days), minHist = Number(cfg.trend_min_history_days);
    var minWin = Number(cfg.trend_min_window_days === undefined ? 90 : cfg.trend_min_window_days);
    var sp = spanDays(ds.rows), current = [{ as_of: dayStr(sp.last), value: currentValue }], span = sp.last - sp.first + 1;
    if (span < minHist) return [current, null, ["trend needs ≥" + minHist + " days of history; this export spans " + span]];
    var points = [];
    for (var k = 1; ; k++) {
      var cutoff = sp.last - step * k;
      if (cutoff - sp.first + 1 < minWin) break;
      var sub = Object.assign({}, ds, { rows: ds.rows.filter(function (r) { return dayNum(r.date) <= cutoff; }) });
      var ageThen = ageNow - (sp.last - cutoff) / 365.25;
      var pt = point(evaluate(sub, rules), rules, ageThen);
      if (pt !== null) points.push({ as_of: dayStr(cutoff), value: pt[0] });
    }
    points = points.reverse().concat(current);
    var notes = ["trend: effective age re-computed on the export truncated every " + step + " days (each point uses the " + minWin +
      " days ending on its as_of date); chronological age back-dated from the profile age, so a steady person paces at ≈1.0 years/year"];
    var pace = null;
    if (points.length >= 3) {
      var t = points.map(function (p) { return (dayNum(p.as_of) - sp.first) / 365.25; }), v = points.map(function (p) { return p.value; });
      var tm = t.reduce(function (a, b) { return a + b; }, 0) / t.length, vm = v.reduce(function (a, b) { return a + b; }, 0) / v.length;
      var sxy = 0, sxx = 0;
      t.forEach(function (ti, i) { sxy += (ti - tm) * (v[i] - vm); sxx += (ti - tm) * (ti - tm); });
      pace = pyRound(sxy / sxx, 2);
      notes.push("pace over a short history is noisy: a 0.25-year change in Σ years over 3 months reads as ±1 year/year");
    }
    return [points, pace, notes];
  }
  function effectiveAge(ds, lev, rules) {
    var age = ds.profile.age_years;
    if (age === null || age === undefined) return null;
    var pt = point(lev, rules, Number(age));
    if (pt === null) return null;
    var cfg = rules.composite;
    var contributions = stableSort(pt[2].map(function (l) {
      return { key: l.key, label: l.label, years_vs_population: l.years_vs_population, band: l.band, population_band: l.population_band, confidence: l.confidence };
    }), function (a, b) { return Math.abs(b.years_vs_population) - Math.abs(a.years_vs_population); });
    var pop = populationTexts(rules), assumptions = assumptionTexts([cfg.independence_assumption]);
    pt[2].forEach(function (l) { (pop[l.key] || []).forEach(function (t) { if ((l.assumptions || []).indexOf(t) >= 0) assumptions.push(t); }); });
    assumptions = assumptions.concat(assumptionTexts([rules.gompertz_mrdt_meta]));
    var tr = ds.rows.length ? trend(ds, rules, Number(age), pt[0]) : [[], null, []];
    return {
      value: pt[0], advance_years: pt[1], chronological_age: age, n_levers: pt[2].length, contributions: contributions,
      trend: tr[0], pace_years_per_year: tr[1],
      rule: "Effective age = chronological age + Σ over levers of ln(HR vs population-typical) / ln 2 × MRDT " + fmtG(mrdtOf(rules)) +
        " years, each HR taken from the lever's published all-cause hazard ratio.",
      citation: Object.assign({}, cfg.citation || {}), assumptions: uniq(assumptions), confidence: "low", notes: tr[2]
    };
  }
  function fitnessAge(ds, lev, rules) {
    var sex = ds.profile.sex;
    if ((sex !== "female" && sex !== "male") || !ds.rows.length || !has(ds, "vo2max")) return null;
    var lever = rules.levers.find(function (l) { return l.key === "vo2max"; });
    var s = numeric(windowRows(ds.rows, lever.window_days).rows, "vo2max");
    if (s.length < lever.min_days) return null;
    var vo2 = pyRound(median(s), 1), norms = rules.vo2max_norms, table = norms[sex];
    var pts = Object.keys(table).map(function (key) {
      return [parseInt(String(key).split("-")[0], 10) + 5, interpPercentile(norms.percentiles, table[key], 50)];
    }).sort(function (a, b) { return a[0] - b[0]; });
    var ages = pts.map(function (p) { return p[0]; }), p50 = pts.map(function (p) { return p[1]; }), i;
    if (vo2 >= p50[0]) i = 0;
    else if (vo2 <= p50[p50.length - 1]) i = pts.length - 2;
    else { for (i = 0; i < pts.length - 1; i++) if (p50[i] >= vo2 && vo2 >= p50[i + 1]) break; }
    var a0 = ages[i], a1 = ages[i + 1], v0 = p50[i], v1 = p50[i + 1];
    var raw = v1 !== v0 ? a0 + (vo2 - v0) * (a1 - a0) / (v1 - v0) : a0;
    var notes = [], value = Math.min(80, Math.max(20, raw));
    if (value !== raw) notes.push("raw estimate " + fmtFixed(raw, 1) + " lies outside the 20–80 range the norm table covers — clamped to " + fmtG(value));
    var assumptions = ["Each decade's 50th-percentile VO2max placed at the decade midpoint (25, 35, …, 75), linear between midpoints"];
    if (norms.assumption) assumptions.push(norms.assumption_text);
    if (norms.verify_before_demo) assumptions.push("vo2max norm table flagged verify_before_demo: " + clean(norms.citation));
    notes.push("Device VO2max is an estimate from HR and pace, not a treadmill test.");
    var byAge = {}; pts.forEach(function (p) { byAge[String(p[0])] = pyRound(p[1], 2); });
    return {
      value: pyRound(value, 1), vo2max: vo2, sex: sex, p50_by_age: byAge,
      rule: "90-day median VO2max (" + pyFloat(vo2) + " ml/kg/min) matched to the age at which the " + sex +
        " 50th-percentile VO2max equals it (the fitness-age framing of Nes 2013 HUNT / Garmin Fitness Age).",
      citation: { short: "Fitness-age framing: Nes 2013 HUNT / Garmin Fitness Age; norms: Cooper Institute (ACSM)", ref: clean(norms.citation), doi: "10.1111/sms.12032" },
      assumptions: assumptions, confidence: "low", notes: notes
    };
  }
  function hrvAgeEquivalent(lev) {
    var ctx = ((lev && lev.context) || []).find(function (c) { return c.key === "hrv_rmssd_percentile"; });
    if (!ctx || !ctx.closest_bin) return null;
    var nums = (String(ctx.closest_bin).match(/\d+/g) || []).map(Number);
    var value = Math.floor(nums.length >= 2 ? (nums[0] + nums[1]) / 2 : nums[0]);
    var notes = ["NOT A VALIDATED AGING CLOCK. This is a norm match: the 5-year age bin whose median rMSSD on a 10-second daytime resting ECG (Lifelines) is closest to your overnight wearable rMSSD. Overnight wearable rMSSD runs higher than daytime 10-s ECG rMSSD, so this reads younger than it should. No years."];
    if (nums.length < 2) notes.push("closest bin is open-ended (" + ctx.closest_bin + "); value is its lower edge");
    return {
      value: value, closest_bin: ctx.closest_bin, your_value: ctx.your_value, bin_median: ctx.bin_median,
      rule: "Midpoint of the age bin whose rMSSD median is closest to yours (" + ctx.closest_bin + " → " + value + ").",
      citation: Object.assign({}, ctx.citation || {}), assumptions: (ctx.assumptions || []).slice(), confidence: "low",
      notes: notes.concat(ctx.note ? [ctx.note] : [])
    };
  }

  var NO_ENMO_REASON = "CosinorAge requires accelerometer ENMO; this export has none";

  /** Port of report.build(ds, rules, ics_text, experiment): the ledger JSON in the CONTRACT shape, plus week_plan and
   *  experiments. CosinorAge needs accelerometer ENMO, which no browser-readable export carries. */
  function buildLedger(ds, rules, icsText, experiment, opts) {
    var lev = evaluate(ds, rules), withheld = lev.withheld.slice();
    withheld.push({ key: "cosinor_age", reason: NO_ENMO_REASON });
    var comp = effectiveAge(ds, lev, rules);
    if (comp === null) withheld.push({ key: "composite_age", reason: compositeWithheldReason(ds, lev, rules) });
    var p = ds.profile || {};
    var prof = { age_years: nn(p.age_years) ? p.age_years : null, sex: p.sex || null,
      height_cm: nn(p.height_cm) ? p.height_cm : null, weight_kg: nn(p.weight_kg) ? p.weight_kg : null };
    prof.assumed = ["age_years", "sex"].filter(function (k) { return prof[k] === null; });
    return {
      version: VERSION, source: ds.source, coverage: ds.coverage, profile: prof,
      bio_age: { cosinor: null, composite: comp, fitness_age: fitnessAge(ds, lev, rules), hrv_age_equivalent: hrvAgeEquivalent(lev) },
      levers: lev.levers, withheld: withheld, context: lev.context, constants: lev.constants,
      ranking_note: "levers sorted by years_recoverable desc; ties by confidence",
      notes: (p.notes || []).slice(),
      week_plan: icsText ? weekPlan(ds, icsText, rules, Object.assign({ levers: lev.levers }, opts || {})) : null,
      experiments: {
        proposals: proposals(lev.levers, rules),
        receipt: experiment ? receipt(ds, rules, experiment.lever, experiment.start_date) : null
      }
    };
  }

  // ======================================================================= calendar: ICS parser (calendar.py)

  var tzFmtCache = {};
  function validTz(tz) {
    if (!tz) return false;
    try { new Intl.DateTimeFormat("en-US", { timeZone: tz }); return true; } catch (e) { return false; }
  }
  function zoneOf(name) { name = name ? String(name).trim() : ""; return name && validTz(name) ? name : null; }
  /** Instant (UTC ms) -> wall clock in tz (ms, as if UTC). */
  function utcToWall(utc, tz) {
    var f = tzFmtCache[tz] || (tzFmtCache[tz] = new Intl.DateTimeFormat("en-US", { timeZone: tz, hourCycle: "h23",
      year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    var p = {}; f.formatToParts(new Date(utc)).forEach(function (x) { p[x.type] = x.value; });
    return Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour % 24, +p.minute, +p.second);
  }
  function wallToUtc(wall, tz) {
    var off = utcToWall(wall, tz) - wall, utc = wall - off, off2 = utcToWall(utc, tz) - utc;
    return off2 === off ? utc : wall - off2;
  }
  function toLocal(ms, zone, isUtc, local) {
    if (!local) return ms;
    if (isUtc) return utcToWall(ms, local);
    if (zone) return zone === local ? ms : utcToWall(wallToUtc(ms, zone), local);
    return ms;
  }
  function unfold(text) {
    var lines = [];
    String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n").forEach(function (raw) {
      if ((raw[0] === " " || raw[0] === "\t") && lines.length) lines[lines.length - 1] += raw.slice(1);
      else if (raw.trim()) lines.push(raw);
    });
    return lines;
  }
  function splitProp(line) {
    var q = false, colon = -1;
    for (var i = 0; i < line.length; i++) {
      if (line[i] === '"') q = !q;
      else if (line[i] === ":" && !q) { colon = i; break; }
    }
    if (colon < 0) return { name: line.toUpperCase(), params: {}, value: "" };
    var parts = line.slice(0, colon).split(";"), params = {};
    parts.slice(1).forEach(function (p) { var k = p.indexOf("="); if (k >= 0) params[p.slice(0, k).toUpperCase()] = p.slice(k + 1).replace(/^"+|"+$/g, ""); });
    return { name: parts[0].toUpperCase(), params: params, value: line.slice(colon + 1) };
  }
  function icsUnescape(v) { return String(v).replace(/\\([\;,nN])/g, function (_, c) { return c === "n" || c === "N" ? "\n" : c; }); }
  /** -> {ms (naive wall clock in its own zone), allDay, zone, utc}; ms null when unparseable. */
  function parseDt(value, params) {
    var v = String(value).trim();
    if ((params.VALUE || "").toUpperCase() === "DATE" || /^\d{8}$/.test(v)) {
      var dm = /^(\d{4})(\d{2})(\d{2})/.exec(v);
      return { ms: dm ? Date.UTC(+dm[1], +dm[2] - 1, +dm[3]) : null, allDay: true, zone: null, utc: false };
    }
    var m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})?(Z?)$/.exec(v);
    if (!m) return { ms: null, allDay: false, zone: null, utc: false };
    var ms = Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +(m[6] || 0));
    if (m[7] === "Z") return { ms: ms, allDay: false, zone: null, utc: true };
    return { ms: ms, allDay: false, zone: zoneOf(params.TZID), utc: false };
  }
  function parseDuration(s) {
    var m = /^([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$/.exec(String(s || "").trim());
    if (!m) return null;
    var ms = ((+m[2] || 0) * 7 + (+m[3] || 0)) * DAY + (+m[4] || 0) * HOUR + (+m[5] || 0) * MINUTE + (+m[6] || 0) * 1000;
    return m[1] === "-" ? -ms : ms;
  }
  var ICS_DAYS = { MO: 0, TU: 1, WE: 2, TH: 3, FR: 4, SA: 5, SU: 6 };
  var MAX_OCCURRENCES = 1000;
  function weekdayMon(ms) { return (new Date(ms).getUTCDay() + 6) % 7; }

  function occurrences(start, rule, horizonEnd, until) {
    var freq = String(rule.FREQ || "").toUpperCase();
    if (freq !== "DAILY" && freq !== "WEEKLY") return [start];
    var interval = Math.max(1, parseInt(rule.INTERVAL || "1", 10) || 1);
    var count = /^\d+$/.test(rule.COUNT || "") ? parseInt(rule.COUNT, 10) : null;
    var limit = horizonEnd !== null ? horizonEnd : start + 366 * DAY;
    if (until !== null) limit = Math.min(limit, until);
    var out = [];
    if (freq === "DAILY") {
      for (var cur = start; cur <= limit && (count === null || out.length < count) && out.length < MAX_OCCURRENCES; cur += interval * DAY) out.push(cur);
      return out;
    }
    var days = uniq(String(rule.BYDAY || "").split(",").map(function (d) { return d.trim(); })
      .filter(function (d) { return /^[A-Z]{2}$/.test(d) && ICS_DAYS[d] !== undefined; }).map(function (d) { return ICS_DAYS[d]; }))
      .sort(function (a, b) { return a - b; });
    if (!days.length) days = [weekdayMon(start)];
    var week0 = start - weekdayMon(start) * DAY;
    for (var k = 0; out.length < MAX_OCCURRENCES; k++) {
      var wk = week0 + k * interval * 7 * DAY;
      if (wk > limit) break;
      for (var i = 0; i < days.length; i++) {
        var c = wk + days[i] * DAY;
        if (c < start || c > limit) continue;
        if (count !== null && out.length >= count) return out;
        out.push(c);
      }
    }
    return out;
  }

  function calendarZone(lines, tz) {
    if (tz) return zoneOf(tz);
    var i, p;
    for (i = 0; i < lines.length; i++) { p = splitProp(lines[i]); if (p.name === "X-WR-TIMEZONE") { var z = zoneOf(p.value); if (z) return z; } }
    var counts = new Map();
    lines.forEach(function (ln) { var q = splitProp(ln); if ((q.name === "DTSTART" || q.name === "DTEND") && q.params.TZID) counts.set(q.params.TZID, (counts.get(q.params.TZID) || 0) + 1); });
    var names = stableSort(Array.from(counts.keys()), function (a, b) { return counts.get(b) - counts.get(a); });
    for (i = 0; i < names.length; i++) { if (zoneOf(names[i])) return zoneOf(names[i]); }
    return null;
  }

  /**
   * ICS text -> {events: [{start, end, summary, all_day}], tz}. start/end are wall-clock ms in the calendar's zone.
   * Zone: opts.tz, else X-WR-TIMEZONE, else the most common IANA TZID, else opts.fallbackTz (the browser's), else
   * wall clock as written. CANCELLED skipped; all-day TRANSPARENT holds skipped; DAILY/WEEKLY RRULE expanded within
   * [horizonStart, horizonEnd] days; EXDATE and RECURRENCE-ID dates removed. Port of calendar.parse_ics.
   */
  function parseICS(text, opts) {
    opts = opts || {};
    var lines = unfold(text);
    var local = calendarZone(lines, opts.tz) || zoneOf(opts.fallbackTz) || null;
    var hEnd = isNum(opts.horizonEnd) ? (opts.horizonEnd + 1) * DAY - 1 : null;
    var hStart = isNum(opts.horizonStart) ? opts.horizonStart * DAY : null;
    var raw = [], cur = null, depth = 0;
    lines.forEach(function (ln) {
      var p = splitProp(ln), name = p.name, value = p.value;
      if (name === "BEGIN") {
        if (value.trim().toUpperCase() === "VEVENT" && cur === null) { cur = { exdates: [] }; depth = 0; }
        else if (cur !== null) depth++;
        return;
      }
      if (name === "END") {
        if (cur !== null) { if (value.trim().toUpperCase() === "VEVENT" && depth === 0) { raw.push(cur); cur = null; } else depth--; }
        return;
      }
      if (cur === null || depth > 0) return;
      if (name === "DTSTART" || name === "DTEND" || name === "RECURRENCE-ID") cur[name] = parseDt(value, p.params);
      else if (name === "EXDATE") value.split(",").forEach(function (v) { cur.exdates.push(parseDt(v, p.params)); });
      else if (name === "SUMMARY") cur.summary = icsUnescape(value).trim();
      else if (["STATUS", "TRANSP", "UID", "RRULE", "DURATION"].indexOf(name) >= 0) cur[name] = value.trim();
    });
    var overrides = new Map();
    raw.forEach(function (ev) {
      var r = ev["RECURRENCE-ID"];
      if (r && ev.UID && r.ms !== null) {
        if (!overrides.has(ev.UID)) overrides.set(ev.UID, new Set());
        overrides.get(ev.UID).add(msDay(toLocal(r.ms, r.zone, r.utc, local)));
      }
    });
    var out = [];
    raw.forEach(function (ev) {
      var st = ev.DTSTART;
      if (String(ev.STATUS || "").toUpperCase() === "CANCELLED" || !st || st.ms === null) return;
      if (st.allDay && String(ev.TRANSP || "").toUpperCase() === "TRANSPARENT") return;
      var length, de = ev.DTEND, dur = ev.DURATION ? parseDuration(ev.DURATION) : null;
      if (de && de.ms !== null) length = st.allDay ? de.ms - st.ms : toLocal(de.ms, de.zone, de.utc, local) - toLocal(st.ms, st.zone, st.utc, local);
      else if (dur !== null) length = dur;
      else length = st.allDay ? DAY : 0;
      if (length < 0) length = 0;
      var starts = [st.ms];
      if (ev.RRULE && !ev["RECURRENCE-ID"]) {
        var rule = {};
        ev.RRULE.split(";").forEach(function (part) { var k = part.indexOf("="); if (k >= 0) rule[part.slice(0, k).toUpperCase()] = part.slice(k + 1); });
        var until = null;
        if (rule.UNTIL) {
          var u = parseDt(rule.UNTIL, {});
          if (u.ms !== null) {
            var um = u.ms;
            if (u.utc && st.zone) um = utcToWall(um, st.zone);
            until = rule.UNTIL.length === 8 ? msDay(um) * DAY + DAY - 1 : um;
          }
        }
        var hOwn = hEnd !== null && st.zone && local && !st.allDay ? hEnd + DAY : hEnd;
        starts = occurrences(st.ms, rule, hOwn, until);
      }
      var excluded = new Set();
      ev.exdates.forEach(function (x) { if (x.ms !== null) excluded.add(msDay(x.allDay ? x.ms : toLocal(x.ms, x.zone, x.utc, local))); });
      if (ev.RRULE && !ev["RECURRENCE-ID"] && overrides.has(ev.UID)) overrides.get(ev.UID).forEach(function (d) { excluded.add(d); });
      starts.forEach(function (s) {
        var sl = st.allDay ? s : toLocal(s, st.zone, st.utc, local);
        if (excluded.has(msDay(sl))) return;
        var el = sl + length;
        if (hEnd !== null && sl > hEnd) return;
        if (hStart !== null && el < hStart) return;
        out.push({ start: sl, end: el, summary: ev.summary || "", all_day: !!st.allDay });
      });
    });
    out.sort(function (a, b) {
      return (a.start - b.start) || ((a.all_day ? 0 : 1) - (b.all_day ? 0 : 1)) || (a.summary < b.summary ? -1 : a.summary > b.summary ? 1 : 0);
    });
    return { events: out, tz: local };
  }

  // ======================================================================= calendar: day load, body state, week plan

  function isHard(summary, keywords) {
    var s = summary.toLowerCase();
    return (keywords || []).some(function (k) { return s.indexOf(String(k).toLowerCase()) >= 0; });
  }
  /** Load for one local day (day number). Port of calendar.day_load. */
  function dayLoad(events, day, rules) {
    var cal = rules.calendar, dl = cal.day_load, d0 = day * DAY, d1 = d0 + DAY;
    var timed = events.filter(function (e) {
      return (!e.all_day && e.start < d1 && e.end > d0) || (!e.all_day && e.start === e.end && d0 <= e.start && e.start < d1);
    });
    var allDay = events.filter(function (e) { return e.all_day && e.start < d1 && e.end > d0; });
    timed = stableSort(timed, function (a, b) { return (a.start - b.start) || (a.end - b.end); });
    var ms = 0, cs = null, ce = null;
    timed.forEach(function (e) {
      var s = Math.max(e.start, d0), en = Math.min(e.end, d1);
      if (ce === null || s > ce) { if (ce !== null) ms += ce - cs; cs = s; ce = en; }
      else ce = Math.max(ce, en);
    });
    if (ce !== null) ms += ce - cs;
    var gap = Number(dl.back_to_back_gap_min) * MINUTE, b2b = 0, latest = null;
    timed.forEach(function (e) {
      if (latest !== null && e.start - latest <= gap) b2b++;
      latest = latest === null ? e.end : Math.max(latest, e.end);
    });
    var startsToday = timed.filter(function (e) { return e.start >= d0; }).map(function (e) { return e.start; });
    var first = startsToday.length ? hhmm(Math.min.apply(null, startsToday)) : null;
    var hard = uniq(allDay.concat(timed).filter(function (e) { return e.summary && isHard(e.summary, cal.hard_event_keywords); })
      .map(function (e) { return e.summary; }));
    var cutoff = d0 + Number(dl.evening_cutoff_hour) * HOUR;
    return {
      meeting_hours: pyRound(ms / HOUR, 2), n_events: timed.length, back_to_back: b2b,
      first_event_time: first, hard_events: hard, evening_event: timed.some(function (e) { return e.end > cutoff; })
    };
  }

  function bandName(value, bands) {
    if (value === null) return null;
    for (var i = 0; i < bands.length; i++) {
      var lo = bands[i].lo, hi = bands[i].hi;
      if ((lo === null || lo === undefined || value >= lo) && (hi === null || hi === undefined || value < hi)) return bands[i].name;
    }
    return null;
  }
  /** Body signals as of the export's last day; every field null when its input is absent. Port of calendar.body_state. */
  function bodyState(ds, rules, levers) {
    var bs = rules.calendar.body_state;
    var out = { as_of: null, recovery: null, recovery_pct: null, sleep_debt_min: null, sleep_debt_high: null,
      rhr_rising: null, rhr_rising_days: null, sri: null };
    if (levers) {
      var s = levers.find(function (l) { return l.key === "sleep_regularity"; });
      if (s && s.your_value !== null && s.your_value !== undefined) out.sri = s.your_value;
    }
    if (!has(ds, "date") || !ds.rows.length) return out;
    var asOf = ds.rows.map(function (r) { return String(r.date).slice(0, 10); }).sort().pop();
    out.as_of = asOf;
    function lastValue(col) {
      if (!has(ds, col)) return null;
      var rows = ds.rows.filter(function (r) { return String(r.date).slice(0, 10) === asOf; });
      if (!rows.length) return null;
      var v = rows[rows.length - 1][col];
      v = v === null || v === undefined || v === "" ? NaN : Number(v);
      return isNum(v) ? v : null;
    }
    out.recovery_pct = lastValue("recovery_pct");
    out.recovery = bandName(out.recovery_pct, bs.recovery_bands);
    out.sleep_debt_min = lastValue("sleep_debt_min");
    out.sleep_debt_high = out.sleep_debt_min === null ? null : out.sleep_debt_min >= Number(bs.sleep_debt_high_min);
    if (has(ds, "rhr")) {
      var last = dayNum(asOf), base = [], byDay = new Map();
      var winDays = Number(bs.rhr_baseline_days === undefined ? 28 : bs.rhr_baseline_days);
      ds.rows.forEach(function (r) {
        var d = dayNum(r.date), v = r.rhr === null || r.rhr === undefined || r.rhr === "" ? NaN : Number(r.rhr);
        if (!isNum(v) || !isNum(d)) return;
        byDay.set(d, v);
        if (d > last - winDays && d <= last) base.push(v);
      });
      if (base.length >= Number(bs.rhr_baseline_min_days === undefined ? 14 : bs.rhr_baseline_min_days)) {
        var thresh = median(base) + Number(bs.rhr_rising_margin_bpm), n = 0;
        for (var c = last; byDay.has(c) && byDay.get(c) > thresh; c--) n++;
        out.rhr_rising = n >= Number(bs.rhr_rising_days);
        out.rhr_rising_days = n;
      }
    }
    return out;
  }

  var BODY_KEYS = { recovery: "recovery", sleep_debt_high: "sleep_debt_high", rhr_rising: "rhr_rising" };
  var BODY_SOURCE = { recovery: "recovery_pct", sleep_debt_high: "sleep_debt_min", rhr_rising: "rhr" };
  var WEEKDAYS_MON = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  function fmtNum(x) { return fmtG(pyRound(Number(x), 1)); }
  function fill(tpl, vars) {
    return String(tpl).replace(/\{(\w+)\}/g, function (m, k) { return Object.prototype.hasOwnProperty.call(vars, k) ? String(vars[k]) : m; });
  }

  /** Body state (export's last day, carried forward) x the calendar for the 7 days from that day. Port of calendar.week_plan. */
  function weekPlan(ds, icsText, rules, opts) {
    opts = opts || {};
    var cal = rules.calendar, dl = cal.day_load, horizon = opts.horizonDays || 7;
    var today = opts.today || (ds.coverage && ds.coverage.last) || (ds.rows.length ? ds.rows.map(function (r) { return String(r.date); }).sort().pop() : null);
    if (!today) throw new Error("week_plan needs `today` or a dataset with dates");
    var t0 = dayNum(today), days = [];
    for (var i = 0; i < horizon; i++) days.push(t0 + i);
    var parsed = parseICS(icsText, { tz: opts.tz, fallbackTz: opts.fallbackTz, horizonStart: t0, horizonEnd: days[days.length - 1] + 1 });
    var events = parsed.events;
    var body = bodyState(ds, rules, opts.levers);
    var notes = [];
    if (body.as_of && body.as_of !== dayStr(t0)) notes.push("Body state is from " + body.as_of + "; the plan starts " + dayStr(t0) + ".");
    var loads = new Map();
    days.concat([days[days.length - 1] + 1]).forEach(function (d) { loads.set(d, dayLoad(events, d, rules)); });
    var weekHours = pyRound(days.reduce(function (t, d) { return t + loads.get(d).meeting_hours; }, 0), 2);
    if (!days.some(function (d) { return loads.get(d).n_events || loads.get(d).hard_events.length; }))
      notes.push("The calendar has no events between " + dayStr(days[0]) + " and " + dayStr(days[days.length - 1]) + ".");
    var mult = dl.week_heavy_multiplier === undefined ? 3 : Number(dl.week_heavy_multiplier);
    var heavyWeek = weekHours >= Number(dl.meeting_hours_heavy) * mult;
    var lightCut = dl.day_light_hours === undefined ? 2 : Number(dl.day_light_hours);
    var withheld = new Map();
    (cal.rules || []).forEach(function (r) {
      Object.keys(r.when || {}).forEach(function (k) {
        if (BODY_KEYS[k] && body[BODY_KEYS[k]] === null && !withheld.has(r.key)) withheld.set(r.key, k);
      });
    });
    withheld.forEach(function (k, key) {
      notes.push("Rule " + key + " withheld: needs " + k + ", and this export has no usable " + BODY_SOURCE[k] + " " +
        (k === "rhr_rising" ? "(or too few days for the 28-day baseline)" : "on its last day") + ".");
    });
    var planDays = days.map(function (d, i) {
      var ld = loads.get(d), nxt = loads.get(d + 1), matched = [], suggestions = [];
      (cal.rules || []).forEach(function (r) {
        if (withheld.has(r.key)) return;
        if (r.scope === "week" && i > 0) return;
        var firstLoad = r.first_event_day === "next" ? nxt : ld;
        var ok = Object.keys(r.when || {}).every(function (k) {
          var want = r.when[k];
          switch (k) {
            case "recovery": return body.recovery === want;
            case "day_has_hard_event": return (ld.hard_events.length > 0) === !!want;
            case "sleep_debt_high": return body.sleep_debt_high === !!want;
            case "first_event_before_hour": return firstLoad.first_event_time !== null && parseInt(firstLoad.first_event_time.slice(0, 2), 10) < parseInt(want, 10);
            case "rhr_rising": return body.rhr_rising === !!want;
            case "week_meeting_hours_heavy": return heavyWeek === !!want;
            case "back_to_back_heavy": return (ld.back_to_back >= parseInt(dl.back_to_back_heavy, 10)) === !!want;
            case "evening_event": return ld.evening_event === !!want;
            case "day_meeting_hours_light": return (ld.meeting_hours < lightCut) === !!want;
            default: throw new Error("calendar rule " + r.key + ": unknown `when` key '" + k + "'");
          }
        });
        if (!ok) return;
        matched.push(r.key);
        if (suggestions.length < 2) {
          suggestions.push({ key: r.key, why: r.why, text: fill(r.suggest, {
            hard_event: ld.hard_events.length ? ld.hard_events[0] : "high-stakes event",
            first_event_time: firstLoad.first_event_time || "",
            back_to_back: ld.back_to_back,
            week_meeting_hours: fmtNum(weekHours),
            rhr_rising_days: body.rhr_rising_days !== null ? body.rhr_rising_days : ""
          }) });
        }
      });
      return { date: dayStr(d), weekday: WEEKDAYS_MON[weekdayMon(d * DAY)], load: ld, suggestions: suggestions, matched_rules: matched };
    });
    var plan = {
      start: dayStr(t0), horizon_days: horizon, days: planDays, body: body, week_meeting_hours: weekHours,
      assumptions: assumptionTexts(cal.assumptions), notes: notes
    };
    Object.defineProperty(plan, "tz", { value: parsed.tz, enumerable: false });
    return plan;
  }

  // ======================================================================= experiments (experiment.py)

  var SRI_METRIC = "sleep_regularity_sri";
  function protocolsOf(rules) {
    var m = new Map(); (((rules.experiments || {}).protocols) || []).forEach(function (p) { m.set(p.lever, p); }); return m;
  }
  /** Top 3 scored levers by years_recoverable that have a protocol and something to recover. Port of experiment.propose. */
  function proposals(levers, rules) {
    var protos = protocolsOf(rules);
    var ranked = stableSort((levers || []).filter(function (l) { return protos.has(l.key) && (l.years_recoverable || 0) > 0; }),
      function (a, b) { return b.years_recoverable - a.years_recoverable; });
    return ranked.slice(0, 3).map(function (l) {
      var p = protos.get(l.key);
      return { lever: l.key, name: p.name, change: p.change, duration_days: parseInt(p.duration_days, 10), metrics: (p.metrics || []).slice(),
        expected: p.expected, citation: Object.assign({}, p.citation || {}), years_recoverable: l.years_recoverable };
    });
  }
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function percentile(sorted, q) {
    var pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  /** Circular moving-block bootstrap of the mean (same algorithm as experiment._block_bootstrap_means; JS PRNG). */
  function blockBootstrapMeans(x, rnd, iters, block) {
    var n = x.length, b = Math.max(1, Math.min(block, n)), k = Math.ceil(n / b), out = new Float64Array(iters);
    for (var it = 0; it < iters; it++) {
      var s = 0, taken = 0;
      for (var j = 0; j < k && taken < n; j++) {
        var st = Math.floor(rnd() * n);
        for (var o = 0; o < b && taken < n; o++, taken++) s += x[(st + o) % n];
      }
      out[it] = s / n;
    }
    return out;
  }
  function sriPairs(rows) {
    var w = sleepRegularityIndex(rows), sorted = rows.slice().sort(function (a, b) { return a.date < b.date ? -1 : a.date > b.date ? 1 : 0; }), vals = [];
    for (var i = 0; i < sorted.length - 1; i++) { var v = sleepRegularityIndex([sorted[i], sorted[i + 1]]).sri; if (v !== null) vals.push(v); }
    return { sri: w.sri, n: w.n, vals: vals };
  }
  function emptyMetric(metric, status, kw) {
    return Object.assign({ metric: metric, n_before: null, n_after: null, mean_before: null, mean_after: null, delta: null,
      ci_low: null, ci_high: null, beyond_noise: null, status: status }, kw || {});
  }
  var BOOTSTRAP_SEED = 0;
  /** Before/after receipt for one lever's protocol started on startDate. Port of experiment.receipt (CI from a JS PRNG). */
  function receipt(ds, rules, leverKey, startDate) {
    var ex = rules.experiments, protos = protocolsOf(rules);
    if (!protos.has(leverKey)) throw new Error("no experiment protocol for lever '" + leverKey + "'; have " + Array.from(protos.keys()).sort().join(", "));
    var p = protos.get(leverKey), start = dayNum(String(startDate).slice(0, 10));
    if (!isNum(start)) throw new Error("start date must be YYYY-MM-DD");
    var nb = parseInt(ex.min_days_before, 10), na = parseInt(ex.min_days_after, 10);
    var minBefore = ex.min_values_before !== undefined ? parseInt(ex.min_values_before, 10) : Math.floor(nb / 2);
    var iters = parseInt(ex.bootstrap_iterations, 10), block = parseInt(ex.bootstrap_block_days === undefined ? 1 : ex.bootstrap_block_days, 10);
    var before = ds.rows.filter(function (r) { var d = dayNum(r.date); return d >= start - nb && d <= start - 1; });
    var after = ds.rows.filter(function (r) { return dayNum(r.date) >= start; });
    var sp = spanDays(ds.rows), daysElapsed = sp.ok ? Math.max(0, sp.last - start + 1) : 0;
    var metrics = (p.metrics || []).map(function (m, idx) {
      var rnd = mulberry32(BOOTSTRAP_SEED + idx * 7919), xb, xa, nB, nA, mb, ma;
      if (m === SRI_METRIC) {
        if (!has(ds, "sleep_onset") || !has(ds, "sleep_offset")) return emptyMetric(m, "missing", { reason: "no sleep_onset / sleep_offset columns in this export" });
        var B = sriPairs(before), Aa = sriPairs(after);
        xb = B.vals; xa = Aa.vals; nB = B.n; nA = Aa.n; mb = B.sri; ma = Aa.sri;
      } else {
        if (!has(ds, m)) return emptyMetric(m, "missing", { reason: "no " + m + " column in this export" });
        xb = numeric(before, m); xa = numeric(after, m); nB = xb.length; nA = xa.length;
        mb = nB ? mean(xb) : null; ma = nA ? mean(xa) : null;
      }
      if (nA < na) return emptyMetric(m, "not_yet", { n_before: nB, n_after: nA, days_after: nA, need: na });
      if (nB < minBefore || xb.length < 2 || mb === null)
        return emptyMetric(m, "no_baseline", { n_before: nB, n_after: nA, need_before: minBefore, reason: "only " + nB + " values in the " + nb + " days before " + dayStr(start) });
      if (xa.length < 2 || ma === null) return emptyMetric(m, "not_yet", { n_before: nB, n_after: nA, days_after: nA, need: na });
      var ba = blockBootstrapMeans(xa, rnd, iters, block), bb = blockBootstrapMeans(xb, rnd, iters, block), boot = [];
      for (var i = 0; i < iters; i++) boot.push(ba[i] - bb[i]);
      boot.sort(function (a, b) { return a - b; });
      var lo = percentile(boot, 0.025), hi = percentile(boot, 0.975), dg = m === "steps" ? 0 : 2;
      return { metric: m, n_before: nB, n_after: nA, mean_before: pyRound(mb, dg), mean_after: pyRound(ma, dg), delta: pyRound(ma - mb, dg),
        ci_low: pyRound(lo, dg), ci_high: pyRound(hi, dg), beyond_noise: lo > 0 || hi < 0, status: "ok" };
    });
    var st = new Set(metrics.map(function (x) { return x.status; }));
    var status = (st.size === 1 && st.has("ok")) ? "ok" : (!st.has("ok") && st.has("not_yet")) ? "not_yet" : st.has("ok") ? "partial" : Array.from(st).sort()[0];
    return {
      lever: leverKey, name: p.name, start_date: dayStr(start), status: status, days_elapsed: daysElapsed, metrics: metrics,
      note: clean(ex.receipt_note), assumptions: assumptionTexts([ex.bootstrap_assumption]), citation: Object.assign({}, p.citation || {})
    };
  }

  // ======================================================================= top level

  async function loadRules(url) {
    var res = await fetch(url || "./rules.yaml");
    if (!res.ok) throw new Error("could not load rules.yaml (" + res.status + ")");
    return getYaml().load(await res.text());
  }
  /** .ics file, or a .zip of .ics files (Google Calendar's export), -> ICS text. */
  async function readCalendar(file) {
    var head = await headBytes(file, 4);
    if (isZipHead(head)) {
      var zip = await getJSZip().loadAsync(file.data || file), texts = [];
      var names = Object.keys(zip.files).filter(function (n) { return !zip.files[n].dir && /\.ics$/i.test(n) && n.indexOf("__MACOSX") !== 0; });
      if (!names.length) throw new Error("That zip holds no .ics files.");
      for (var i = 0; i < names.length; i++) texts.push(await zip.file(names[i]).async("string"));
      return texts.join("\n");
    }
    var text = decode(await toBytes(file));
    if (text.indexOf("BEGIN:VCALENDAR") < 0 && text.indexOf("BEGIN:VEVENT") < 0) throw new Error("That file is not an .ics calendar export.");
    return text;
  }
  function browserTz() { try { return Intl.DateTimeFormat().resolvedOptions().timeZone || null; } catch (e) { return null; } }
  function applyProfile(ds, age, sex) {
    var entered = [];
    if (age !== null && age !== undefined && String(age).trim() !== "") {
      var a = Number(age);
      if (!isNum(a) || a < 10 || a > 110) throw new Error("Age must be between 10 and 110, or left blank.");
      ds.profile.age_years = a; ds.profile.notes.push("age_years entered by user"); entered.push("age_years");
    }
    if (sex === "female" || sex === "male") { ds.profile.sex = sex; ds.profile.notes.push("sex entered by user"); entered.push("sex"); }
    return entered;
  }

  /**
   * analyze({exportFile | ds, icsFile | icsText, age, sex, experiment: {lever, start_date} | null, rules, onProgress})
   * -> ledger (CONTRACT shape) + week_plan + experiments + profile.entered. Also accepts analyze(exportFile, opts).
   */
  async function analyze(a, b) {
    var opts = (a && (a.exportFile || a.ds || a.icsText || a.icsFile)) ? a : Object.assign({}, b || {}, { exportFile: a });
    var progress = opts.onProgress || function () {};
    var rules = opts.rules || await loadRules(opts.rulesUrl);
    var ds = opts.ds;
    if (!ds) {
      if (!opts.exportFile) throw new Error("no export file");
      progress("Reading the export…");
      ds = await ingest(opts.exportFile, { onProgress: progress });
    }
    if (!ds.rows.length) throw new Error("This export has no daily rows we can read.");
    var entered = applyProfile(ds, opts.age, opts.sex);
    var icsText = opts.icsText || (opts.icsFile ? await readCalendar(opts.icsFile) : null);
    progress("Scoring levers…");
    var L = buildLedger(ds, rules, icsText, opts.experiment || null, { fallbackTz: opts.fallbackTz === undefined ? browserTz() : opts.fallbackTz });
    L.profile.entered = entered;
    L.profile.assumed = L.profile.assumed.filter(function (k) { return entered.indexOf(k) < 0; });
    Object.defineProperty(L, "_ds", { value: ds, enumerable: false });
    Object.defineProperty(L, "_rules", { value: rules, enumerable: false });
    Object.defineProperty(L, "_ics", { value: icsText, enumerable: false });
    return L;
  }

  return {
    VERSION: VERSION,
    analyze: analyze, ingest: ingest, loadRules: loadRules, readCalendar: readCalendar,
    buildLedger: buildLedger, evaluate: evaluate, effectiveAge: effectiveAge, fitnessAge: fitnessAge, hrvAgeEquivalent: hrvAgeEquivalent,
    sleepRegularityIndex: sleepRegularityIndex, sriPairValues: sriPairValues,
    parseICS: parseICS, dayLoad: dayLoad, bodyState: bodyState, weekPlan: weekPlan,
    proposals: proposals, receipt: receipt,
    loadWhoop: loadWhoop, loadOura: loadOura, AppleParser: AppleParser, appleAssemble: appleAssemble,
    parseCSV: parseCSV, pyRound: pyRound, DAILY_COLUMNS: DAILY_COLUMNS
  };
});
