"""Calendar join: ICS parsing, day load, body state (None-handling), and the 7-day plan."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ageledger.calendar import body_state, day_load, parse_ics, week_plan
from ageledger.levers.engine import load_rules
from ageledger.report import build
from tests.synthetic import synthetic_person, whoop_like

ROOT = Path(__file__).resolve().parent.parent
ICS_PATH = ROOT / "data" / "synthetic" / "calendar_synthetic.ics"

RULES = load_rules()

ICS = "\r\n".join([
    "BEGIN:VCALENDAR",
    "VERSION:2.0",
    "X-WR-TIMEZONE:America/New_York",
    "BEGIN:VTIMEZONE",
    "TZID:America/New_York",
    "BEGIN:DAYLIGHT",
    "DTSTART:19700308T020000",             # must not be read as an event
    "TZOFFSETTO:-0400",
    "END:DAYLIGHT",
    "END:VTIMEZONE",
    # TZID event with a folded SUMMARY (continuation line starts with one space)
    "BEGIN:VEVENT",
    "DTSTART;TZID=America/New_York:20261006T100000",
    "DTEND;TZID=America/New_York:20261006T113000",
    "SUMMARY:Vendor Negotiation: Acme renewal\\, multi-year term and a cap on uplif",
    " t at anniversary",
    "BEGIN:VALARM",
    "DESCRIPTION:This is an event reminder",
    "TRIGGER:-PT10M",
    "END:VALARM",
    "END:VEVENT",
    # UTC event: 18:00Z = 14:00 EDT
    "BEGIN:VEVENT",
    "DTSTART:20261006T180000Z",
    "DTEND:20261006T190000Z",
    "SUMMARY:Finance check-in",
    "END:VEVENT",
    # Europe/London 15:00 BST = 10:00 EDT
    "BEGIN:VEVENT",
    "DTSTART;TZID=Europe/London:20261007T150000",
    "DTEND;TZID=Europe/London:20261007T160000",
    "SUMMARY:London sync",
    "END:VEVENT",
    # non-IANA TZID (Outlook style): wall clock kept
    "BEGIN:VEVENT",
    "DTSTART;TZID=Eastern Standard Time:20261007T130000",
    "DTEND;TZID=Eastern Standard Time:20261007T133000",
    "SUMMARY:Outlook invite",
    "END:VEVENT",
    # all-day
    "BEGIN:VEVENT",
    "DTSTART;VALUE=DATE:20261008",
    "DTEND;VALUE=DATE:20261009",
    "SUMMARY:Board day",
    "END:VEVENT",
    # all-day + transparent: a free hold, skipped
    "BEGIN:VEVENT",
    "DTSTART;VALUE=DATE:20261009",
    "DTEND;VALUE=DATE:20261010",
    "SUMMARY:Tentative hold",
    "TRANSP:TRANSPARENT",
    "END:VEVENT",
    # cancelled: skipped
    "BEGIN:VEVENT",
    "DTSTART;TZID=America/New_York:20261006T160000",
    "DTEND;TZID=America/New_York:20261006T170000",
    "SUMMARY:Cancelled thing",
    "STATUS:CANCELLED",
    "END:VEVENT",
    # weekly recurrence with one EXDATE
    "BEGIN:VEVENT",
    "UID:weekly-1",
    "DTSTART;TZID=America/New_York:20261005T090000",
    "DTEND;TZID=America/New_York:20261005T093000",
    "RRULE:FREQ=WEEKLY;COUNT=4",
    "EXDATE;TZID=America/New_York:20261012T090000",
    "SUMMARY:1:1 with Priya",
    "END:VEVENT",
    "END:VCALENDAR",
]) + "\r\n"


def _by(events, summary_prefix):
    return [e for e in events if e["summary"].startswith(summary_prefix)]


def test_parse_ics_tz_utc_allday_unfold():
    ev = parse_ics(ICS)
    neg = _by(ev, "Vendor Negotiation")
    assert len(neg) == 1
    assert neg[0]["summary"] == "Vendor Negotiation: Acme renewal, multi-year term and a cap on uplift at anniversary"
    assert neg[0]["start"] == datetime(2026, 10, 6, 10, 0) and neg[0]["end"] == datetime(2026, 10, 6, 11, 30)
    assert neg[0]["all_day"] is False

    fin = _by(ev, "Finance")[0]
    assert fin["start"] == datetime(2026, 10, 6, 14, 0) and fin["end"] == datetime(2026, 10, 6, 15, 0)

    lon = _by(ev, "London")[0]
    assert lon["start"] == datetime(2026, 10, 7, 10, 0)

    assert _by(ev, "Outlook")[0]["start"] == datetime(2026, 10, 7, 13, 0)

    board = _by(ev, "Board day")[0]
    assert board["all_day"] is True
    assert board["start"] == datetime(2026, 10, 8) and board["end"] == datetime(2026, 10, 9)

    assert not _by(ev, "Tentative hold") and not _by(ev, "Cancelled")
    assert not any("reminder" in e["summary"] for e in ev)          # VALARM text never leaks
    assert len(ev) == 5 + 3                                          # 5 singles + 3 of 4 weekly (one EXDATE)
    assert [e["start"].date().isoformat() for e in _by(ev, "1:1")] == ["2026-10-05", "2026-10-19", "2026-10-26"]
    assert ev == sorted(ev, key=lambda e: (e["start"], not e["all_day"], e["summary"]))


def test_rrule_expands_only_within_horizon():
    from datetime import date
    ev = parse_ics(ICS, horizon_start=date(2026, 10, 4), horizon_end=date(2026, 10, 11))
    assert [e["start"].date().isoformat() for e in _by(ev, "1:1")] == ["2026-10-05"]


def _ev(day, a, b, summary="x"):
    return {"start": datetime.fromisoformat(f"{day}T{a}"), "end": datetime.fromisoformat(f"{day}T{b}"),
            "summary": summary, "all_day": False}


def test_day_load_back_to_back_and_fields():
    d = "2026-10-05"
    events = [
        _ev(d, "08:30", "09:00", "Staff sync"),
        _ev(d, "09:00", "09:30", "1:1 with Priya"),         # gap 0  -> b2b 1
        _ev(d, "09:40", "10:30", "Budget"),                 # gap 10 -> b2b 2 (≤ 10 counts)
        _ev(d, "10:41", "11:00", "Vendor demo"),            # gap 11 -> not
        _ev(d, "10:50", "11:30", "Overlap"),                # overlaps -> b2b 3
        _ev(d, "14:00", "15:00", "QBR"),                    # far
        _ev(d, "18:30", "19:30", "Dinner"),                 # ends after 19:00 -> evening
        {"start": datetime(2026, 10, 5), "end": datetime(2026, 10, 6), "summary": "Board offsite", "all_day": True},
        _ev("2026-10-06", "09:00", "10:00", "Negotiation tomorrow"),
    ]
    ld = day_load(events, d, RULES)
    assert ld["back_to_back"] == 3
    assert ld["n_events"] == 7
    assert ld["first_event_time"] == "08:30"
    assert ld["evening_event"] is True
    # union: 08:30–10:30 minus 09:30–09:40 = 1h50; 10:41–11:30 = 49m; 1h; 1h  → 4.65 h
    assert ld["meeting_hours"] == pytest.approx(110 / 60 + 49 / 60 + 1 + 1, abs=0.01)
    assert set(ld["hard_events"]) == {"Board offsite", "1:1 with Priya", "Vendor demo", "QBR"}
    assert "Negotiation tomorrow" not in ld["hard_events"]

    empty = day_load(events, "2026-10-07", RULES)
    assert empty == {"meeting_hours": 0.0, "n_events": 0, "back_to_back": 0, "first_event_time": None,
                     "hard_events": [], "evening_event": False}


def test_evening_cutoff_is_strict():
    ld = day_load([_ev("2026-10-05", "18:00", "19:00")], "2026-10-05", RULES)
    assert ld["evening_event"] is False


def _person(**kw):
    return whoop_like(synthetic_person(days=180, with_minute=False, **kw))


def test_body_state_none_when_columns_absent():
    ds = _person()
    ds.daily = ds.daily.drop(columns=["recovery_pct", "sleep_debt_min", "rhr"])
    b = body_state(ds, RULES)
    assert b["as_of"] == "2026-10-04"
    for k in ("recovery", "recovery_pct", "sleep_debt_min", "sleep_debt_high", "rhr_rising", "rhr_rising_days", "sri"):
        assert b[k] is None, k


def test_body_state_none_when_last_day_missing_value():
    ds = _person()
    ds.daily.loc[ds.daily.index[-1], "recovery_pct"] = np.nan
    b = body_state(ds, RULES)
    assert b["recovery"] is None and b["recovery_pct"] is None      # not carried from an older day


def test_body_state_values_and_rhr_streak():
    ds = _person()
    d = ds.daily
    d.loc[d.index[-1], "recovery_pct"] = 20
    d.loc[d.index[-1], "sleep_debt_min"] = 60
    base = float(d["rhr"].iloc[-28:].median())
    d.loc[d.index[-4], "rhr"] = base - 1
    d.loc[d.index[-3:], "rhr"] = base + 6
    b = body_state(ds, RULES, levers=[{"key": "sleep_regularity", "your_value": 71.5}])
    assert b["recovery"] == "red" and b["recovery_pct"] == 20
    assert b["sleep_debt_min"] == 60 and b["sleep_debt_high"] is True
    assert b["rhr_rising"] is True and b["rhr_rising_days"] == 3
    assert b["sri"] == 71.5


@pytest.mark.parametrize("pct,band", [(33, "red"), (34, "yellow"), (66, "yellow"), (67, "green")])
def test_recovery_band_edges(pct, band):
    ds = _person()
    ds.daily.loc[ds.daily.index[-1], "recovery_pct"] = pct
    assert body_state(ds, RULES)["recovery"] == band


def test_week_plan_from_synthetic_fixtures():
    text = ICS_PATH.read_text(encoding="utf-8")
    ds = _person()
    plan = week_plan(ds, text, RULES)
    assert plan["start"] == "2026-10-04" and len(plan["days"]) == 7
    assert [d["date"] for d in plan["days"]][-1] == "2026-10-10"
    assert sum(len(d["suggestions"]) for d in plan["days"]) >= 1
    assert all(len(d["suggestions"]) <= 2 for d in plan["days"])
    for d in plan["days"]:
        assert set(d) == {"date", "weekday", "load", "suggestions", "matched_rules"}
        assert [s["key"] for s in d["suggestions"]] == d["matched_rules"][:2]
        for s in d["suggestions"]:
            assert "{" not in s["text"]
    assert plan["week_meeting_hours"] == pytest.approx(sum(d["load"]["meeting_hours"] for d in plan["days"]), abs=0.02)
    assert plan["assumptions"] and all(isinstance(a, str) for a in plan["assumptions"])
    mon = plan["days"][1]
    assert mon["weekday"] == "Mon" and mon["load"]["back_to_back"] >= 4 and "fragmented_day" in mon["matched_rules"]


def test_red_hard_event_fires_on_negotiation_day():
    text = ICS_PATH.read_text(encoding="utf-8")
    ds = _person()
    ds.daily.loc[ds.daily.index[-1], "recovery_pct"] = 15
    plan = week_plan(ds, text, RULES)
    assert plan["body"]["recovery"] == "red"
    tue = next(d for d in plan["days"] if d["date"] == "2026-10-06")
    assert any("Negotiation" in h for h in tue["load"]["hard_events"])
    first = tue["suggestions"][0]
    assert first["key"] == "red_hard_event"
    assert first["text"].startswith("Move the Vendor Negotiation: Acme renewal if you can.")
    sat = next(d for d in plan["days"] if d["date"] == "2026-10-10")
    assert "red_hard_event" not in sat["matched_rules"]


def test_rules_withheld_when_body_input_missing():
    text = ICS_PATH.read_text(encoding="utf-8")
    ds = _person()
    ds.daily = ds.daily.drop(columns=["recovery_pct"])
    plan = week_plan(ds, text, RULES)
    fired = {k for d in plan["days"] for k in d["matched_rules"]}
    assert not fired & {"red_hard_event", "green_empty_day", "yellow_default"}
    withheld_notes = [n for n in plan["notes"] if n.startswith("Rule ")]
    assert sum("red_hard_event" in n for n in withheld_notes) == 1
    assert any("yellow_default" in n for n in withheld_notes)


def test_debt_early_meeting_uses_next_days_first_event():
    text = ICS_PATH.read_text(encoding="utf-8")
    ds = _person()
    ds.daily.loc[ds.daily.index[-1], "sleep_debt_min"] = 90
    plan = week_plan(ds, text, RULES)
    sun = plan["days"][0]                       # Monday's first meeting is 08:30
    s = next(s for s in sun["suggestions"] if s["key"] == "debt_early_meeting")
    assert s["text"].endswith("Your first meeting is at 08:30.")


def test_week_scope_rule_shows_once():
    text = ICS_PATH.read_text(encoding="utf-8")
    rules = load_rules()
    rules["calendar"]["day_load"]["week_heavy_multiplier"] = 1     # 5 h/week → heavy
    ds = _person()
    d = ds.daily
    d.loc[d.index[-3:], "rhr"] = float(d["rhr"].iloc[-28:].median()) + 6
    plan = week_plan(ds, text, rules)
    hits = [d["date"] for d in plan["days"] if "rhr_rising_heavy_week" in d["matched_rules"]]
    assert hits == ["2026-10-04"]


def test_report_week_plan_only_with_ics():
    ds = _person()
    assert build(ds)["week_plan"] is None
    L = build(ds, ics_text=ICS_PATH.read_text(encoding="utf-8"))
    assert L["week_plan"]["body"]["sri"] is not None
    assert len(L["week_plan"]["days"]) == 7
