"""Write a realistic synthetic WHOOP export zip (180 days, with a Steps column) to data/synthetic/.
Run: ./venv/bin/python tests/make_fixtures.py   → data/synthetic/whoop_synthetic_180d.zip
Demo-safe: no real person's data."""
from __future__ import annotations
import csv, io, sys, zipfile
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from synthetic import synthetic_person

CYCLE_HDR = ["Cycle start time", "Cycle end time", "Cycle timezone", "Recovery score %", "Resting heart rate (bpm)",
             "Heart rate variability (ms)", "Skin temp (celsius)", "Blood oxygen %", "Day Strain", "Energy burned (cal)",
             "Max HR (bpm)", "Average HR (bpm)", "Sleep onset", "Wake onset", "Sleep performance %",
             "Respiratory rate (rpm)", "Asleep duration (min)", "In bed duration (min)", "Light sleep duration (min)",
             "Deep (SWS) duration (min)", "REM duration (min)", "Awake duration (min)", "Sleep need (min)",
             "Sleep debt (min)", "Sleep efficiency %", "Sleep consistency %", "Steps"]

def _csv(header, rows):
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(header); w.writerows(rows); return buf.getvalue()

def _fmt(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def write(path: Path, days=180, seed=7, **kw) -> Path:
    ds = synthetic_person(days=days, with_minute=False, seed=seed, **kw)
    d = ds.daily
    rows = []
    for _, r in d.iterrows():
        onset, offset = pd.Timestamp(r["sleep_onset"]), pd.Timestamp(r["sleep_offset"])
        asleep = float(r["sleep_min"]); inbed = asleep / max(float(r.get("sleep_efficiency", 90)) / 100, 0.5)
        deep = asleep * 0.20; rem = asleep * 0.22; light = asleep - deep - rem
        rows.append([_fmt(onset), _fmt(offset + pd.Timedelta(hours=15)), "UTC-04:00",
                     f"{r['recovery_pct']:.0f}", f"{r['rhr']:.0f}", f"{r['hrv_rmssd']:.0f}", "", "", f"{r.get('strain', 10):.1f}", "",
                     "", "", _fmt(onset), _fmt(offset), "", f"{r.get('resp_rate', 15):.1f}",
                     f"{asleep:.0f}", f"{inbed:.0f}", f"{light:.0f}", f"{deep:.0f}", f"{rem:.0f}", f"{inbed-asleep:.0f}",
                     f"{r['sleep_need_min']:.0f}", f"{r['sleep_debt_min']:.0f}",
                     f"{r.get('sleep_efficiency', 90):.0f}", f"{r['sleep_consistency_pct']:.0f}", f"{r['steps']:.0f}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("physiological_cycles.csv", _csv(CYCLE_HDR, rows))
        zf.writestr("sleeps.csv", "Cycle start time,Cycle end time,Cycle timezone,Sleep onset,Wake onset,Nap\n")
        zf.writestr("workouts.csv", "Cycle start time,Workout start time\n")
        zf.writestr("journal_entries.csv", "Cycle start time,Question text,Answered yes\n")
    return path

# ------------------------------------------------------------------ calendar (.ics)
# A Google-Calendar-style export for the 3 weeks after the synthetic export's last day (2026-10-04 = "today").
# Times are America/New_York wall clock (EDT, UTC-4, the whole span — DST ends 2026-11-01).
# kind: "tz" = DTSTART;TZID=..., "utc" = written as ...Z (converted from local), "date" = all-day VALUE=DATE.
ICS_EVENTS = [
    # date,        start,   end,     summary,                                         kind,  extra
    ("2026-10-04", None,    None,    "Marathon long run (personal)",                  "date", {"TRANSP": "TRANSPARENT"}),
    ("2026-10-05", "08:30", "09:00", "Staff sync",                                    "tz",   {}),
    ("2026-10-05", "09:00", "09:30", "1:1 with Priya",                                "tz",   {"RRULE": "FREQ=WEEKLY;COUNT=3"}),
    ("2026-10-05", "09:35", "10:30", "Spend review: software & cloud",                "tz",   {}),
    ("2026-10-05", "10:30", "11:00", "Budget prep",                                   "tz",   {}),
    ("2026-10-05", "11:05", "12:00", "Vendor demo: observability platform",           "tz",   {}),
    ("2026-10-05", "14:00", "15:00", "Finance partner check-in",                      "utc",  {}),
    ("2026-10-06", "10:00", "11:30", "Vendor Negotiation: Acme renewal",              "tz",   {"DESCRIPTION": "Walk-away: flat renewal at current unit price. Levers: multi-year term in exchange for a cap on uplift; true-down at anniversary; remove auto-renew clause."}),
    ("2026-10-06", "13:00", "13:30", "Prep: QBR deck",                                "tz",   {}),
    ("2026-10-06", "18:30", "20:00", "Team dinner",                                   "tz",   {}),
    ("2026-10-07", None,    None,    "Offsite planning (tentative hold)",             "date", {"TRANSP": "TRANSPARENT"}),
    ("2026-10-07", "09:00", "10:00", "Architecture review",                           "tz",   {}),
    ("2026-10-07", "15:00", "15:30", "Coffee chat",                                   "tz",   {}),
    ("2026-10-08", "08:00", "08:30", "Early sync with APAC",                          "utc",  {}),
    ("2026-10-08", "11:00", "12:00", "QBR: cloud vendor",                             "tz",   {}),
    ("2026-10-08", "12:00", "12:30", "Lunch with Dana",                               "tz",   {}),
    ("2026-10-08", "16:00", "16:30", "Design sync",                                   "tz",   {"STATUS": "CANCELLED"}),
    ("2026-10-09", "10:00", "10:30", "Weekly metrics",                                "tz",   {}),
    ("2026-10-12", None,    None,    "Indigenous Peoples' Day (office open)",         "date", {}),
    ("2026-10-12", "10:00", "11:00", "Contract redlines",                             "tz",   {}),
    ("2026-10-13", "14:00", "15:00", "Presentation: Q4 savings plan to leadership",   "tz",   {}),
    ("2026-10-13", "15:00", "15:30", "Debrief",                                       "tz",   {}),
    ("2026-10-14", "09:00", "09:30", "Standup",                                       "tz",   {}),
    ("2026-10-14", "09:30", "10:00", "Procurement intake",                            "tz",   {}),
    ("2026-10-14", "10:00", "11:00", "Security review: new AI tool",                  "tz",   {}),
    ("2026-10-14", "11:00", "11:30", "Legal sync",                                    "tz",   {}),
    ("2026-10-14", "11:30", "12:00", "Vendor call",                                   "utc",  {}),
    ("2026-10-15", "19:00", "21:00", "Industry meetup",                               "tz",   {}),
    ("2026-10-16", "13:00", "14:30", "Performance calibration",                       "tz",   {}),
    ("2026-10-19", "08:00", "09:00", "Board prep",                                    "tz",   {}),
    ("2026-10-20", "10:00", "11:00", "Negotiation follow-up: Acme",                   "tz",   {}),
    ("2026-10-21", "13:00", "14:00", "Interview: Senior Analyst",                     "tz",   {}),
    ("2026-10-22", None,    None,    "Travel day",                                    "date", {}),
    ("2026-10-23", "15:00", "16:00", "Exec readout",                                  "tz",   {}),
    ("2026-10-25", None,    None,    "Family brunch",                                 "date", {"TRANSP": "TRANSPARENT"}),
]

ICS_VTIMEZONE = """BEGIN:VTIMEZONE
TZID:America/New_York
X-LIC-LOCATION:America/New_York
BEGIN:DAYLIGHT
TZOFFSETFROM:-0500
TZOFFSETTO:-0400
TZNAME:EDT
DTSTART:19700308T020000
RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=2SU
END:DAYLIGHT
BEGIN:STANDARD
TZOFFSETFROM:-0400
TZOFFSETTO:-0500
TZNAME:EST
DTSTART:19701101T020000
RRULE:FREQ=YEARLY;BYMONTH=11;BYDAY=1SU
END:STANDARD
END:VTIMEZONE"""


def _fold(line: str) -> list[str]:
    """RFC 5545 line folding at 75 octets (continuation lines start with one space)."""
    out, cur = [], line
    while len(cur.encode("utf-8")) > 75:
        cut = 75 if not out else 74
        while len(cur[:cut].encode("utf-8")) > (75 if not out else 74):
            cut -= 1
        out.append(cur[:cut] if not out else " " + cur[:cut])
        cur = cur[cut:]
    out.append(cur if not out else " " + cur)
    return out


def _ics_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def ics(events=ICS_EVENTS, tz: str = "America/New_York") -> str:
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo
    z = ZoneInfo(tz)
    lines = ["BEGIN:VCALENDAR", "PRODID:-//Google Inc//Google Calendar 70.9054//EN", "VERSION:2.0",
             "CALSCALE:GREGORIAN", "METHOD:PUBLISH", "X-WR-CALNAME:Work (synthetic)", f"X-WR-TIMEZONE:{tz}"]
    lines += ICS_VTIMEZONE.splitlines()
    for i, (day, st, en, summary, kind, extra) in enumerate(events):
        ev = ["BEGIN:VEVENT"]
        d = day.replace("-", "")
        if kind == "date":
            nxt = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d")
            ev += [f"DTSTART;VALUE=DATE:{d}", f"DTEND;VALUE=DATE:{nxt}"]
        else:
            a = datetime.strptime(f"{day} {st}", "%Y-%m-%d %H:%M")
            b = datetime.strptime(f"{day} {en}", "%Y-%m-%d %H:%M")
            if kind == "utc":
                fa = a.replace(tzinfo=z).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                fb = b.replace(tzinfo=z).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                ev += [f"DTSTART:{fa}", f"DTEND:{fb}"]
            else:
                ev += [f"DTSTART;TZID={tz}:{a:%Y%m%dT%H%M%S}", f"DTEND;TZID={tz}:{b:%Y%m%dT%H%M%S}"]
        if "RRULE" in extra:
            ev.append(f"RRULE:{extra['RRULE']}")
        ev += ["DTSTAMP:20261004T120000Z", f"UID:synthetic-{i:03d}@ageledger.test",
               "CREATED:20260901T000000Z", "LAST-MODIFIED:20261001T000000Z", "SEQUENCE:0",
               f"STATUS:{extra.get('STATUS', 'CONFIRMED')}", f"SUMMARY:{_ics_text(summary)}",
               f"TRANSP:{extra.get('TRANSP', 'OPAQUE')}"]
        if "DESCRIPTION" in extra:
            ev.append(f"DESCRIPTION:{_ics_text(extra['DESCRIPTION'])}")
        if kind != "date":
            ev += ["BEGIN:VALARM", "ACTION:DISPLAY", "DESCRIPTION:This is an event reminder",
                   "TRIGGER:-P0DT0H10M0S", "END:VALARM"]
        ev.append("END:VEVENT")
        lines += ev
    lines.append("END:VCALENDAR")
    folded = [f for ln in lines for f in _fold(ln)]
    return "\r\n".join(folded) + "\r\n"


def write_ics(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ics().encode("utf-8"))
    return path


if __name__ == "__main__":
    root = Path(__file__).parent.parent / "data" / "synthetic"
    print(write(root / "whoop_synthetic_180d.zip"))
    print(write_ics(root / "calendar_synthetic.ics"))
