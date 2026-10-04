"""Calendar join: body state (from the export) x the coming week (from an .ics export).

    events = parse_ics(text)                       # stdlib-only RFC 5545 subset
    day_load(events, "2026-10-06")                 # meeting hours, back-to-back, first event, hard events
    body_state(ds, rules)                          # recovery band, sleep debt, RHR streak — None when absent
    week_plan(ds, text, rules)                     # per-day suggestions from rules.yaml `calendar.rules`

Every threshold lives in rules.yaml `calendar:` and is an assumption (no literature maps recovery to
meeting load). A rule whose body input is missing never fires; it is listed once in `notes`.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from ageledger.schema import Dataset

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_ICS_DAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
MAX_OCCURRENCES = 1000


# =========================================================================== ICS parsing

def _unfold(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        elif raw.strip():
            lines.append(raw)
    return lines


def _split_prop(line: str) -> tuple[str, dict, str]:
    """'DTSTART;TZID=America/New_York:20261005T090000' -> ('DTSTART', {'TZID': ...}, '2026...')."""
    in_q, colon = False, None
    for i, ch in enumerate(line):
        if ch == '"':
            in_q = not in_q
        elif ch == ":" and not in_q:
            colon = i
            break
    if colon is None:
        return line.upper(), {}, ""
    head, value = line[:colon], line[colon + 1:]
    parts = head.split(";")
    params = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.upper()] = v.strip('"')
    return parts[0].upper(), params, value


def _unescape(v: str) -> str:
    return re.sub(r"\\([\\;,nN])", lambda m: "\n" if m.group(1) in "nN" else m.group(1), v)


def _zone(name: Optional[str]) -> Optional[ZoneInfo]:
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def _parse_dt(value: str, params: dict) -> tuple[Optional[datetime], bool, Optional[ZoneInfo], bool]:
    """-> (naive wall-clock datetime in its own zone, all_day, zone, is_utc)."""
    v = value.strip()
    if params.get("VALUE", "").upper() == "DATE" or re.fullmatch(r"\d{8}", v):
        try:
            return datetime.strptime(v[:8], "%Y%m%d"), True, None, False
        except ValueError:
            return None, True, None, False
    m = re.fullmatch(r"(\d{8})T(\d{4})(\d{2})?(Z?)", v)
    if not m:
        return None, False, None, False
    dt = datetime.strptime(m.group(1) + m.group(2) + (m.group(3) or "00"), "%Y%m%d%H%M%S")
    if m.group(4) == "Z":
        return dt, False, None, True
    return dt, False, _zone(params.get("TZID")), False


def _to_local(dt: datetime, zone: Optional[ZoneInfo], is_utc: bool, local: Optional[ZoneInfo]) -> datetime:
    """Convert a naive wall-clock time from its own zone to the calendar's local zone (naive).
    Floating times, unknown (non-IANA) TZIDs, or no known local zone: wall clock kept as written."""
    if local is None:
        return dt
    if is_utc:
        return dt.replace(tzinfo=timezone.utc).astimezone(local).replace(tzinfo=None)
    if zone is not None:
        return dt.replace(tzinfo=zone).astimezone(local).replace(tzinfo=None)
    return dt


def _duration(v: str) -> Optional[timedelta]:
    m = re.fullmatch(r"([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", v.strip())
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    w, d, h, mi, s = (int(x) if x else 0 for x in m.groups()[1:])
    return sign * timedelta(weeks=w, days=d, hours=h, minutes=mi, seconds=s)


def _parse_rrule(v: str) -> dict:
    out = {}
    for part in v.split(";"):
        if "=" in part:
            k, val = part.split("=", 1)
            out[k.upper()] = val
    return out


def _occurrences(start: datetime, rule: dict, horizon_end: Optional[datetime],
                 until: Optional[datetime]) -> list[datetime]:
    """Starts (naive, in the event's own zone) for FREQ=DAILY|WEEKLY. INTERVAL, COUNT, UNTIL honoured.
    WEEKLY with BYDAY: plain weekday codes only (no ordinals); otherwise DTSTART's weekday."""
    freq = rule.get("FREQ", "").upper()
    if freq not in ("DAILY", "WEEKLY"):
        return [start]
    interval = max(1, int(rule.get("INTERVAL", "1") or 1))
    count = int(rule["COUNT"]) if rule.get("COUNT", "").isdigit() else None
    limit = horizon_end if horizon_end is not None else start + timedelta(days=366)
    if until is not None:
        limit = min(limit, until)
    out: list[datetime] = []
    if freq == "DAILY":
        cur, step = start, timedelta(days=interval)
        while cur <= limit and (count is None or len(out) < count) and len(out) < MAX_OCCURRENCES:
            out.append(cur)
            cur += step
        return out
    days = sorted({_ICS_DAYS[d.strip()[-2:]] for d in rule.get("BYDAY", "").split(",")
                   if d.strip()[-2:] in _ICS_DAYS and re.fullmatch(r"[A-Z]{2}", d.strip())})
    if not days:
        days = [start.weekday()]
    week0 = start - timedelta(days=start.weekday())
    k = 0
    while len(out) < MAX_OCCURRENCES:
        wk = week0 + timedelta(weeks=k * interval)
        if wk > limit:
            break
        for wd in days:
            cur = wk + timedelta(days=wd)
            if cur < start or cur > limit:
                continue
            if count is not None and len(out) >= count:
                return out
            out.append(cur)
        k += 1
    return out


def _calendar_zone(lines: list[str], tz: Optional[str]) -> Optional[ZoneInfo]:
    if tz:
        return _zone(tz)
    for ln in lines:
        name, _, value = _split_prop(ln)
        if name == "X-WR-TIMEZONE":
            z = _zone(value.strip())
            if z:
                return z
    tzids: dict[str, int] = {}
    for ln in lines:
        name, params, _ = _split_prop(ln)
        if name in ("DTSTART", "DTEND") and params.get("TZID"):
            tzids[params["TZID"]] = tzids.get(params["TZID"], 0) + 1
    for name in sorted(tzids, key=lambda k: -tzids[k]):
        z = _zone(name)
        if z:
            return z
    return None


def parse_ics(text: str, tz: Optional[str] = None, horizon_start: Optional[date] = None,
              horizon_end: Optional[date] = None) -> list[dict]:
    """ICS text -> [{start, end, summary, all_day}] sorted by start; start/end are naive LOCAL datetimes.

    Local zone: `tz` if given, else X-WR-TIMEZONE, else the most common IANA TZID; TZID= and UTC (Z)
    times are converted to it. Floating times and non-IANA TZIDs keep their wall clock.
    STATUS:CANCELLED is skipped; an all-day TRANSP:TRANSPARENT event (a "free" hold) is skipped.
    RRULE FREQ=DAILY|WEEKLY expands within [horizon_start, horizon_end] (default: up to a year past
    DTSTART); EXDATE dates and RECURRENCE-ID overrides remove the matching occurrence. Other FREQs:
    only the first occurrence.
    """
    lines = _unfold(text or "")
    local = _calendar_zone(lines, tz)
    h_end = datetime.combine(horizon_end, time.max) if horizon_end else None
    h_start = datetime.combine(horizon_start, time.min) if horizon_start else None

    raw_events: list[dict] = []
    cur: Optional[dict] = None
    depth = 0                                   # nested components inside a VEVENT (VALARM)
    for ln in lines:
        name, params, value = _split_prop(ln)
        if name == "BEGIN":
            if value.strip().upper() == "VEVENT" and cur is None:
                cur, depth = {"exdates": []}, 0
            elif cur is not None:
                depth += 1
            continue
        if name == "END":
            if cur is not None:
                if value.strip().upper() == "VEVENT" and depth == 0:
                    raw_events.append(cur)
                    cur = None
                else:
                    depth -= 1
            continue
        if cur is None or depth > 0:
            continue
        if name in ("DTSTART", "DTEND", "RECURRENCE-ID"):
            cur[name] = _parse_dt(value, params)
        elif name == "EXDATE":
            for v in value.split(","):
                cur["exdates"].append(_parse_dt(v, params))
        elif name == "SUMMARY":
            cur["summary"] = _unescape(value).strip()
        elif name in ("STATUS", "TRANSP", "UID", "RRULE", "DURATION"):
            cur[name] = value.strip()

    overrides: dict[str, set] = {}
    for ev in raw_events:
        if ev.get("RECURRENCE-ID") and ev.get("UID") and ev["RECURRENCE-ID"][0] is not None:
            rid, all_day, z, utc = ev["RECURRENCE-ID"]
            overrides.setdefault(ev["UID"], set()).add(_to_local(rid, z, utc, local).date())

    out: list[dict] = []
    for ev in raw_events:
        if ev.get("STATUS", "").upper() == "CANCELLED" or not ev.get("DTSTART") or ev["DTSTART"][0] is None:
            continue
        start, all_day, zone, utc = ev["DTSTART"]
        if all_day and ev.get("TRANSP", "").upper() == "TRANSPARENT":
            continue
        if ev.get("DTEND") and ev["DTEND"][0] is not None:
            end_raw, _, ezone, eutc = ev["DTEND"]
            length = (_to_local(end_raw, ezone, eutc, local) - _to_local(start, zone, utc, local)) if not all_day \
                else end_raw - start
        elif ev.get("DURATION") and _duration(ev["DURATION"]) is not None:
            length = _duration(ev["DURATION"])
        else:
            length = timedelta(days=1) if all_day else timedelta(0)
        if length < timedelta(0):
            length = timedelta(0)

        starts = [start]
        if ev.get("RRULE") and not ev.get("RECURRENCE-ID"):
            rule = _parse_rrule(ev["RRULE"])
            until = None
            if rule.get("UNTIL"):
                u, _, uz, uutc = _parse_dt(rule["UNTIL"], {})
                if u is not None:
                    # UNTIL in UTC -> express in the event's own zone for comparison with its wall clock
                    if uutc and zone is not None:
                        u = u.replace(tzinfo=timezone.utc).astimezone(zone).replace(tzinfo=None)
                    until = u if not (len(rule["UNTIL"]) == 8) else datetime.combine(u.date(), time.max)
            h_end_own = h_end
            if h_end is not None and zone is not None and local is not None and not all_day:
                h_end_own = h_end + timedelta(days=1)          # slack for zone offsets; trimmed below
            starts = _occurrences(start, rule, h_end_own, until)

        excluded = {(_to_local(d, z, u, local) if not a else d).date()
                    for d, a, z, u in ev["exdates"] if d is not None}
        excluded |= overrides.get(ev.get("UID"), set()) if ev.get("RRULE") and not ev.get("RECURRENCE-ID") else set()

        for s in starts:
            s_local = s if all_day else _to_local(s, zone, utc, local)
            if s_local.date() in excluded:
                continue
            e_local = s_local + length
            if h_end is not None and s_local > h_end:
                continue
            if h_start is not None and e_local < h_start:
                continue
            out.append({"start": s_local, "end": e_local, "summary": ev.get("summary", ""), "all_day": bool(all_day)})
    out.sort(key=lambda e: (e["start"], not e["all_day"], e["summary"]))
    return out


# =========================================================================== day load

def _as_date(day) -> date:
    if isinstance(day, datetime):
        return day.date()
    if isinstance(day, date):
        return day
    return datetime.strptime(str(day)[:10], "%Y-%m-%d").date()


def _cal_rules(rules: Optional[dict]) -> dict:
    if rules is None:
        from ageledger.levers.engine import load_rules
        rules = load_rules()
    return rules["calendar"]


def _is_hard(summary: str, keywords: Iterable[str]) -> bool:
    s = summary.lower()
    return any(str(k).lower() in s for k in keywords)


def day_load(events: list[dict], day, rules: Optional[dict] = None) -> dict:
    """Load for one local calendar day.

    meeting_hours: union of timed events clipped to the day (double-booking is not double time).
    n_events: timed events overlapping the day (all-day events are not meetings, but their summaries
    still count toward hard_events). back_to_back: consecutive timed events (by start) whose gap after
    the latest end so far is ≤ back_to_back_gap_min (overlaps count). evening_event: a timed event
    ends after evening_cutoff_hour.
    """
    cal = _cal_rules(rules)
    dl = cal["day_load"]
    d = _as_date(day)
    d0 = datetime.combine(d, time.min)
    d1 = d0 + timedelta(days=1)
    timed = [e for e in events if not e["all_day"] and e["start"] < d1 and e["end"] > d0
             or (not e["all_day"] and e["start"] == e["end"] and d0 <= e["start"] < d1)]
    all_day = [e for e in events if e["all_day"] and e["start"] < d1 and e["end"] > d0]
    timed.sort(key=lambda e: (e["start"], e["end"]))

    hours, cur_s, cur_e = 0.0, None, None
    for e in timed:
        s, en = max(e["start"], d0), min(e["end"], d1)
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                hours += (cur_e - cur_s).total_seconds() / 3600
            cur_s, cur_e = s, en
        else:
            cur_e = max(cur_e, en)
    if cur_e is not None:
        hours += (cur_e - cur_s).total_seconds() / 3600

    gap = timedelta(minutes=float(dl["back_to_back_gap_min"]))
    b2b, latest_end = 0, None
    for e in timed:
        if latest_end is not None and e["start"] - latest_end <= gap:
            b2b += 1
        latest_end = e["end"] if latest_end is None else max(latest_end, e["end"])

    starts_today = [e["start"] for e in timed if e["start"] >= d0]
    first = min(starts_today).strftime("%H:%M") if starts_today else None
    hard = list(dict.fromkeys(e["summary"] for e in all_day + timed
                              if e["summary"] and _is_hard(e["summary"], cal.get("hard_event_keywords", []))))
    cutoff = d0 + timedelta(hours=float(dl["evening_cutoff_hour"]))
    evening = any(e["end"] > cutoff for e in timed)
    return {
        "meeting_hours": round(hours, 2),
        "n_events": len(timed),
        "back_to_back": b2b,
        "first_event_time": first,
        "hard_events": hard,
        "evening_event": bool(evening),
    }


# =========================================================================== body state

def _last_value(daily: pd.DataFrame, col: str, as_of: str) -> Optional[float]:
    if col not in daily.columns:
        return None
    row = daily[daily["date"].astype(str).str[:10] == as_of]
    if row.empty:
        return None
    v = pd.to_numeric(row[col], errors="coerce").iloc[-1]
    return None if pd.isna(v) else float(v)


def _band(value: Optional[float], bands: list[dict]) -> Optional[str]:
    if value is None:
        return None
    for b in bands:
        lo, hi = b.get("lo"), b.get("hi")
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return b["name"]
    return None


def _rhr_streak(daily: pd.DataFrame, bs: dict, as_of: str) -> tuple[Optional[bool], Optional[int]]:
    if "rhr" not in daily.columns:
        return None, None
    d = pd.to_datetime(daily["date"].astype(str).str[:10], errors="coerce")
    v = pd.to_numeric(daily["rhr"], errors="coerce")
    last = pd.Timestamp(as_of)
    win = (d > last - pd.Timedelta(days=int(bs.get("rhr_baseline_days", 28)))) & (d <= last)
    base = v[win].dropna()
    if len(base) < int(bs.get("rhr_baseline_min_days", 14)):
        return None, None
    thresh = float(base.median()) + float(bs["rhr_rising_margin_bpm"])
    by_day = dict(zip(d[v.notna()], v[v.notna()]))
    n, cur = 0, last
    while cur in by_day and by_day[cur] > thresh:
        n += 1
        cur -= pd.Timedelta(days=1)
    return n >= int(bs["rhr_rising_days"]), n


def body_state(ds: Dataset, rules: Optional[dict] = None, levers: Optional[list[dict]] = None) -> dict:
    """Body signals as of the export's last day. Every field is None when its input is absent — never defaulted.
    `sri` is read from an engine result's sleep_regularity lever when `levers` is given."""
    cal = _cal_rules(rules)
    bs = cal["body_state"]
    daily = ds.daily if ds.daily is not None else pd.DataFrame()
    as_of = None
    if "date" in daily.columns and len(daily):
        as_of = str(sorted(daily["date"].astype(str).str[:10])[-1])
    out = {"as_of": as_of, "recovery": None, "recovery_pct": None, "sleep_debt_min": None,
           "sleep_debt_high": None, "rhr_rising": None, "rhr_rising_days": None, "sri": None}
    if levers:
        sri = next((l for l in levers if l.get("key") == "sleep_regularity"), None)
        if sri is not None and sri.get("your_value") is not None:
            out["sri"] = sri["your_value"]
    if as_of is None:
        return out
    rec = _last_value(daily, "recovery_pct", as_of)
    out["recovery_pct"] = rec
    out["recovery"] = _band(rec, bs["recovery_bands"])
    debt = _last_value(daily, "sleep_debt_min", as_of)
    out["sleep_debt_min"] = debt
    out["sleep_debt_high"] = None if debt is None else bool(debt >= float(bs["sleep_debt_high_min"]))
    out["rhr_rising"], out["rhr_rising_days"] = _rhr_streak(daily, bs, as_of)
    return out


# =========================================================================== week plan

BODY_KEYS = {"recovery": "recovery", "sleep_debt_high": "sleep_debt_high", "rhr_rising": "rhr_rising"}
BODY_SOURCE = {"recovery": "recovery_pct", "sleep_debt_high": "sleep_debt_min", "rhr_rising": "rhr"}


class _SafeDict(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def _fmt_num(x) -> str:
    return f"{round(float(x), 1):g}"


def week_plan(ds: Dataset, ics_text: str, rules: Optional[dict] = None, horizon_days: int = 7,
              today=None, levers: Optional[list[dict]] = None) -> dict:
    if rules is None:
        from ageledger.levers.engine import load_rules
        rules = load_rules()
    cal = rules["calendar"]
    dl = cal["day_load"]
    if today is None:
        today = (ds.coverage or {}).get("last")
        if today is None and ds.daily is not None and len(ds.daily):
            today = str(sorted(ds.daily["date"].astype(str))[-1])
    if today is None:
        raise ValueError("week_plan needs `today` or a dataset with dates")
    t0 = _as_date(today)
    days = [t0 + timedelta(days=i) for i in range(horizon_days)]
    events = parse_ics(ics_text, horizon_start=t0, horizon_end=days[-1] + timedelta(days=1))

    body = body_state(ds, rules, levers)
    notes: list[str] = []
    if body["as_of"] and body["as_of"] != t0.isoformat():
        notes.append(f"Body state is from {body['as_of']}; the plan starts {t0.isoformat()}.")
    loads = {d: day_load(events, d, rules) for d in days + [days[-1] + timedelta(days=1)]}
    week_hours = round(sum(loads[d]["meeting_hours"] for d in days), 2)
    if not any(loads[d]["n_events"] or loads[d]["hard_events"] for d in days):
        notes.append(f"The calendar has no events between {days[0].isoformat()} and {days[-1].isoformat()}.")

    heavy_week = week_hours >= float(dl["meeting_hours_heavy"]) * float(dl.get("week_heavy_multiplier", 3))
    light_cut = float(dl.get("day_light_hours", 2.0))

    withheld: dict[str, str] = {}
    for r in cal["rules"]:
        for k in r["when"]:
            if k in BODY_KEYS and body[BODY_KEYS[k]] is None and r["key"] not in withheld:
                withheld[r["key"]] = k
    for key, k in withheld.items():
        src = BODY_SOURCE[k]
        notes.append(f"Rule {key} withheld: needs {k}, and this export has no usable {src} "
                     f"{'(or too few days for the 28-day baseline)' if k == 'rhr_rising' else 'on its last day'}.")

    plan_days = []
    for i, d in enumerate(days):
        ld = loads[d]
        nxt = loads[d + timedelta(days=1)]
        matched, suggestions = [], []
        for r in cal["rules"]:
            if r["key"] in withheld:
                continue
            if r.get("scope") == "week" and i > 0:
                continue
            first_load = nxt if r.get("first_event_day") == "next" else ld
            ok = True
            for k, want in r["when"].items():
                if k == "recovery":
                    have = body["recovery"] == want
                elif k == "day_has_hard_event":
                    have = bool(ld["hard_events"]) == bool(want)
                elif k == "sleep_debt_high":
                    have = body["sleep_debt_high"] == bool(want)
                elif k == "first_event_before_hour":
                    ft = first_load["first_event_time"]
                    have = ft is not None and int(ft[:2]) < int(want)
                elif k == "rhr_rising":
                    have = body["rhr_rising"] == bool(want)
                elif k == "week_meeting_hours_heavy":
                    have = heavy_week == bool(want)
                elif k == "back_to_back_heavy":
                    have = (ld["back_to_back"] >= int(dl["back_to_back_heavy"])) == bool(want)
                elif k == "evening_event":
                    have = ld["evening_event"] == bool(want)
                elif k == "day_meeting_hours_light":
                    have = (ld["meeting_hours"] < light_cut) == bool(want)
                else:
                    raise ValueError(f"calendar rule {r['key']}: unknown `when` key {k!r}")
                if not have:
                    ok = False
                    break
            if not ok:
                continue
            matched.append(r["key"])
            if len(suggestions) < 2:
                fill = _SafeDict(
                    hard_event=ld["hard_events"][0] if ld["hard_events"] else "high-stakes event",
                    first_event_time=first_load["first_event_time"] or "",
                    back_to_back=ld["back_to_back"],
                    week_meeting_hours=_fmt_num(week_hours),
                    rhr_rising_days=body["rhr_rising_days"] if body["rhr_rising_days"] is not None else "",
                )
                suggestions.append({"key": r["key"], "text": r["suggest"].format_map(fill), "why": r["why"]})
        plan_days.append({
            "date": d.isoformat(),
            "weekday": WEEKDAYS[d.weekday()],
            "load": ld,
            "suggestions": suggestions,
            "matched_rules": matched,
        })

    from ageledger.levers.engine import _assumption_texts
    return {
        "start": t0.isoformat(),
        "horizon_days": horizon_days,
        "days": plan_days,
        "body": body,
        "week_meeting_hours": week_hours,
        "assumptions": _assumption_texts(cal.get("assumptions")),
        "notes": notes,
    }


__all__ = ["parse_ics", "day_load", "body_state", "week_plan"]
