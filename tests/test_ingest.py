"""Ingest tests: tiny hand-built fixtures for each export format, asserting canonical values."""
from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
import zipfile

import numpy as np
import pandas as pd
import pytest

from ageledger.ingest import load, sniff
from ageledger.schema import DAILY_COLUMNS, MINUTE_COLUMNS, validate


def _common_checks(ds, source):
    assert ds.source == source
    assert validate(ds) == []
    d = ds.daily
    assert list(d["date"]) == sorted(d["date"])
    assert set(d.columns) <= set(DAILY_COLUMNS)
    for c in d.columns:
        if c not in ("date", "source"):
            assert d[c].notna().any(), f"all-NaN column {c} should not be emitted"
    assert ds.coverage["days"] == len(d)
    assert ds.coverage["first"] == d["date"].iloc[0] and ds.coverage["last"] == d["date"].iloc[-1]
    assert set(ds.coverage["columns_present"]) == set(d.columns) - {"date", "source"}


def _row(ds, date):
    r = ds.daily[ds.daily["date"] == date]
    assert len(r) == 1, date
    return r.iloc[0]


# ------------------------------------------------------------------ WHOOP
CYCLE_HDR = ["Cycle start time", "Cycle end time", "Cycle timezone", "Recovery score %", "Resting heart rate (bpm)",
             "Heart rate variability (ms)", "Skin temp (celsius)", "Blood oxygen %", "Day Strain", "Energy burned (cal)",
             "Max HR (bpm)", "Average HR (bpm)", "Sleep onset", "Wake onset", "Sleep performance %",
             "Respiratory rate (rpm)", "Asleep duration (min)", "In bed duration (min)", "Light sleep duration (min)",
             "Deep (SWS) duration (min)", "REM duration (min)", "Awake duration (min)", "Sleep need (min)",
             "Sleep debt (min)", "Sleep efficiency %", "Sleep consistency %"]
SLEEP_HDR = ["Cycle start time", "Cycle end time", "Cycle timezone", "Sleep onset", "Wake onset", "Sleep performance %",
             "Respiratory rate (rpm)", "Asleep duration (min)", "In bed duration (min)", "Light sleep duration (min)",
             "Deep (SWS) duration (min)", "REM duration (min)", "Awake duration (min)", "Sleep need (min)",
             "Sleep debt (min)", "Sleep efficiency %", "Sleep consistency %", "Nap"]


def _csv(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _whoop_rows():
    cyc, slp = [], []
    for i in range(5):
        d0, d1 = f"2026-09-{i + 1:02d}", f"2026-09-{i + 2:02d}"
        onset, wake = f"{d0} 23:10:00", f"{d1} 07:05:00"
        asleep = 420 + i
        if i == 3:   # day 5's cycle lost its sleep fields; sleeps.csv must fill them (and a longer nap must not)
            cyc.append([f"{d0} 23:10:00", f"{d1} 23:00:00", "UTC-04:00", "60", "52", "70", "33.5", "96.5", "11.0",
                        "2400", "160", "70", onset, wake, "", "", "", "", "", "", "", "", "", "", "", ""])
        else:
            cyc.append([f"{d0} 23:10:00", f"{d1} 23:00:00", "UTC-04:00", "60", str(50 + i), str(60 + i), "33.5",
                        "96.5", f"{10 + i}.5", "2400", "160", "70", onset, wake, "85", "14.5", str(asleep), "475",
                        "200", "90", "100", "55", "480", "20", "88", "80"])
        slp.append([f"{d0} 23:10:00", f"{d1} 23:00:00", "UTC-04:00", onset, wake, "85", "15.0", str(asleep), "475",
                    "200", "90", "100", "55", "480", "20", "88", "80", "false"])
    # a 600-minute "nap" ending on 2026-09-05 — excluded by Nap=true
    slp.append(["2026-09-05 12:00:00", "", "UTC-04:00", "2026-09-05 12:00:00", "2026-09-05 22:00:00", "", "13", "600",
                "600", "0", "0", "0", "0", "", "", "99", "", "true"])
    return cyc, slp


def _whoop_zip(tmp_path):
    cyc, slp = _whoop_rows()
    z = tmp_path / "my_whoop_data_2026_09_06.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("physiological_cycles.csv", _csv(CYCLE_HDR, cyc))
        zf.writestr("sleeps.csv", _csv(SLEEP_HDR, slp))
        zf.writestr("workouts.csv", "Cycle start time,Workout start time\n")
        zf.writestr("journal_entries.csv", "Cycle start time,Question text,Answered yes\n")
    return z


def test_whoop_zip(tmp_path):
    z = _whoop_zip(tmp_path)
    assert sniff(str(z)) == "whoop"
    ds = load(str(z))
    _common_checks(ds, "whoop")
    d = ds.daily
    assert list(d["date"]) == ["2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"]
    r = _row(ds, "2026-09-02")
    assert r["rhr"] == 50 and r["hrv_rmssd"] == 60 and r["strain"] == 10.5
    assert r["sleep_min"] == 420 and r["sleep_deep_min"] == 90 and r["sleep_rem_min"] == 100
    assert r["sleep_efficiency"] == 88 and r["resp_rate"] == 14.5 and r["spo2"] == 96.5 and r["skin_temp_c"] == 33.5
    assert r["sleep_onset"] == "2026-09-01T23:10:00" and r["sleep_offset"] == "2026-09-02T07:05:00"
    # day whose cycle had no sleep fields: filled from sleeps.csv main sleep, nap ignored
    r5 = _row(ds, "2026-09-05")
    assert r5["sleep_min"] == 423 and r5["resp_rate"] == 15.0 and r5["rhr"] == 52
    # columns the export did not carry are absent, not NaN
    assert "steps" not in d.columns and "vo2max" not in d.columns and "active_kcal" not in d.columns
    assert ds.minute is None
    assert ds.profile.age_years is None and ds.profile.sex is None and ds.profile.notes
    assert d["sleep_min"].dtype == float


def test_whoop_extracted_dir_and_optional_columns(tmp_path):
    cyc, _ = _whoop_rows()
    hdr = CYCLE_HDR + ["Steps", "VO2 Max (ml/kg/min)"]
    cyc = [r + [str(8000 + i), "42.5"] for i, r in enumerate(cyc)]
    d = tmp_path / "whoop_export"
    d.mkdir()
    (d / "physiological_cycles.csv").write_text(_csv(hdr, cyc))
    ds = load(str(d))
    _common_checks(ds, "whoop")
    assert _row(ds, "2026-09-03")["steps"] == 8001
    assert _row(ds, "2026-09-03")["vo2max"] == 42.5


# ------------------------------------------------------------------ Apple
def _rec(t, src, unit, value, start, end=None):
    end = end or start
    return (f'<Record type="HKQuantityTypeIdentifier{t}" sourceName="{src}" sourceVersion="10" unit="{unit}" '
            f'creationDate="{end} -0400" startDate="{start} -0400" endDate="{end} -0400" value="{value}"/>')


def _sleep(src, stage, start, end):
    return (f'<Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="{src}" '
            f'startDate="{start} -0400" endDate="{end} -0400" value="HKCategoryValueSleepAnalysis{stage}"/>')


APPLE_RECORDS = [
    # steps day 1: iPhone 3000 (2 records), Watch 5000 (3 records) -> keep 5000, never 8000
    _rec("StepCount", "Jing's iPhone", "count", 1000, "2026-09-01 08:00:00", "2026-09-01 08:10:00"),
    _rec("StepCount", "Jing's iPhone", "count", 2000, "2026-09-01 12:00:00", "2026-09-01 12:20:00"),
    _rec("StepCount", "Jing's Apple Watch", "count", 120, "2026-09-01 08:00:00", "2026-09-01 08:02:00"),
    _rec("StepCount", "Jing's Apple Watch", "count", 2380, "2026-09-01 12:00:00", "2026-09-01 12:30:00"),
    _rec("StepCount", "Jing's Apple Watch", "count", 2500, "2026-09-01 17:00:30", "2026-09-01 17:00:50"),
    # steps day 2: only iPhone
    _rec("StepCount", "Jing's iPhone", "count", 4000, "2026-09-02 09:00:00", "2026-09-02 09:30:00"),
    _rec("HeartRate", "Jing's Apple Watch", "count/min", 60, "2026-09-01 08:00:10"),
    _rec("HeartRate", "Jing's Apple Watch", "count/min", 70, "2026-09-01 08:00:40"),
    _rec("HeartRate", "Jing's Apple Watch", "count/min", 90, "2026-09-01 08:01:05"),
    _rec("RestingHeartRate", "Jing's Apple Watch", "count/min", 54, "2026-09-01 00:00:00", "2026-09-01 23:59:00"),
    _rec("RestingHeartRate", "Jing's Apple Watch", "count/min", 56, "2026-09-02 00:00:00", "2026-09-02 23:59:00"),
    _rec("HeartRateVariabilitySDNN", "Jing's Apple Watch", "ms", 40, "2026-09-02 02:00:00", "2026-09-02 02:01:00"),
    _rec("HeartRateVariabilitySDNN", "Jing's Apple Watch", "ms", 50, "2026-09-02 03:00:00", "2026-09-02 03:01:00"),
    _rec("OxygenSaturation", "Jing's Apple Watch", "%", 0.97, "2026-09-02 03:00:00"),
    _rec("OxygenSaturation", "Jing's Apple Watch", "%", 0.95, "2026-09-02 04:00:00"),
    _rec("VO2Max", "Jing's Apple Watch", "mL/min·kg", 40, "2026-09-01 09:00:00"),
    _rec("VO2Max", "Jing's Apple Watch", "mL/min·kg", 41, "2026-09-01 18:00:00"),
    _rec("ActiveEnergyBurned", "Jing's Apple Watch", "kcal", 300, "2026-09-01 10:00:00", "2026-09-01 11:00:00"),
    _rec("ActiveEnergyBurned", "Jing's iPhone", "kcal", 100, "2026-09-01 10:00:00", "2026-09-01 11:00:00"),
    _rec("WalkingSpeed", "Jing's iPhone", "km/hr", 4.5, "2026-09-01 12:00:00"),
    _rec("WalkingDoubleSupportPercentage", "Jing's iPhone", "%", 0.28, "2026-09-01 12:00:00"),
    _rec("WalkingStepLength", "Jing's iPhone", "cm", 70, "2026-09-01 12:00:00"),
    # respiratory rate: one in the night window (02:00) and one daytime on day 2; day 3 daytime only
    _rec("RespiratoryRate", "Jing's Apple Watch", "count/min", 14, "2026-09-02 02:00:00"),
    _rec("RespiratoryRate", "Jing's Apple Watch", "count/min", 20, "2026-09-02 15:00:00"),
    _rec("RespiratoryRate", "Jing's Apple Watch", "count/min", 16, "2026-09-03 15:00:00"),
    # night ending 2026-09-02: Watch stages (6 records), AutoSleep one long block (1 record), iPhone InBed
    _sleep("Jing's Apple Watch", "AsleepCore", "2026-09-01 23:00:00", "2026-09-01 23:50:00"),
    _sleep("Jing's Apple Watch", "Awake", "2026-09-01 23:50:00", "2026-09-02 00:00:00"),
    _sleep("Jing's Apple Watch", "AsleepCore", "2026-09-02 00:00:00", "2026-09-02 01:00:00"),
    _sleep("Jing's Apple Watch", "AsleepDeep", "2026-09-02 01:00:00", "2026-09-02 02:00:00"),
    _sleep("Jing's Apple Watch", "AsleepREM", "2026-09-02 02:00:00", "2026-09-02 03:00:00"),
    _sleep("Jing's Apple Watch", "AsleepCore", "2026-09-02 03:00:00", "2026-09-02 06:30:00"),
    _sleep("AutoSleep", "AsleepUnspecified", "2026-09-01 22:30:00", "2026-09-02 06:45:00"),
    _sleep("Jing's iPhone", "InBed", "2026-09-01 22:45:00", "2026-09-02 06:45:00"),
    # afternoon nap on 2026-09-02 — separate session, must not become the main sleep or be added to it
    _sleep("Jing's Apple Watch", "AsleepCore", "2026-09-02 13:00:00", "2026-09-02 13:30:00"),
    # night ending 2026-09-03: AutoSleep only
    _sleep("AutoSleep", "AsleepUnspecified", "2026-09-03 00:30:00", "2026-09-03 07:00:00"),
]


def _apple_xml():
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE HealthData [\n<!ELEMENT HealthData (ExportDate,Me,(Record|Correlation|Workout|ActivitySummary)*)>\n]>\n'
            '<HealthData locale="en_US">\n'
            ' <ExportDate value="2026-09-03 20:00:00 -0400"/>\n'
            ' <Me HKCharacteristicTypeIdentifierDateOfBirth="1990-09-04" HKCharacteristicTypeIdentifierBiologicalSex='
            '"HKBiologicalSexFemale" HKCharacteristicTypeIdentifierBloodType="HKBloodTypeNotSet"/>\n'
            + "\n".join(" " + r for r in APPLE_RECORDS) +
            '\n <Correlation type="HKCorrelationTypeIdentifierBloodPressure" startDate="2026-09-01 08:00:00 -0400" '
            'endDate="2026-09-01 08:00:00 -0400">\n  '
            + _rec("StepCount", "Nested", "count", 99999, "2026-09-01 08:00:00") +
            '\n </Correlation>\n</HealthData>\n')


def test_apple_zip(tmp_path):
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("apple_health_export/export.xml", _apple_xml())
        zf.writestr("apple_health_export/export_cda.xml", "<ClinicalDocument/>")
    assert sniff(str(z)) == "apple"
    ds = load(str(z))
    _common_checks(ds, "apple")
    d1, d2, d3 = _row(ds, "2026-09-01"), _row(ds, "2026-09-02"), _row(ds, "2026-09-03")
    # steps: max single source per day, not the sum; nested Correlation record ignored
    assert d1["steps"] == 5000 and d2["steps"] == 4000
    assert d1["rhr"] == 54 and d2["rhr"] == 56
    assert d2["hrv_sdnn"] == 45 and "hrv_rmssd" not in ds.daily.columns
    assert d2["spo2"] == pytest.approx(96.0)
    assert d1["vo2max"] == 41
    assert d1["active_kcal"] == 300
    assert d1["walking_speed_ms"] == pytest.approx(1.25)
    assert d1["walking_double_support_pct"] == pytest.approx(28.0)
    assert d1["walking_step_length_m"] == pytest.approx(0.70)
    # sleep: Watch chosen (most records), segment ending before midnight still counted, nap excluded, no cross-source sum
    assert d2["sleep_min"] == pytest.approx(440)
    assert d2["sleep_deep_min"] == pytest.approx(60) and d2["sleep_rem_min"] == pytest.approx(60)
    assert d2["sleep_onset"] == "2026-09-01T23:00:00" and d2["sleep_offset"] == "2026-09-02T06:30:00"
    assert d2["sleep_efficiency"] == pytest.approx(440 / 480 * 100)
    assert pd.isna(d1["sleep_min"])
    assert d3["sleep_min"] == pytest.approx(390) and pd.isna(d3["sleep_deep_min"])
    # resp rate: sleep-window mean on day 2 (14, not 17); plain daily mean on day 3
    assert d2["resp_rate"] == pytest.approx(14) and d3["resp_rate"] == pytest.approx(16)
    # profile
    assert ds.profile.sex == "female" and ds.profile.age_years == 35.0   # birthday is the day after export
    # minute table
    m = ds.minute
    assert m is not None and set(m.columns) <= set(MINUTE_COLUMNS) and "enmo_mg" not in m.columns
    assert m["ts"].iloc[0].utcoffset() == pd.Timedelta(hours=-4)          # single offset -> tz-aware
    loc = m.set_index(m["ts"].dt.tz_localize(None))
    assert loc.loc[pd.Timestamp("2026-09-01 08:00"), "hr"] == 65
    assert loc.loc[pd.Timestamp("2026-09-01 08:01"), "hr"] == 90
    assert loc.loc[pd.Timestamp("2026-09-01 08:00"), "steps"] == pytest.approx(60)   # Watch only, spread 2 min
    assert loc.loc[pd.Timestamp("2026-09-01 08:01"), "steps"] == pytest.approx(60)
    assert loc.loc[pd.Timestamp("2026-09-01 17:00"), "steps"] == pytest.approx(2500)
    day1_minutes = loc[loc.index.normalize() == pd.Timestamp("2026-09-01")]["steps"].sum()
    assert day1_minutes == pytest.approx(5000)
    assert ds.coverage["minute_days"] == 2


def test_apple_bare_xml(tmp_path):
    x = tmp_path / "export.xml"
    x.write_text(_apple_xml())
    assert sniff(str(x)) == "apple"
    assert _row(load(str(x)), "2026-09-01")["steps"] == 5000


# ------------------------------------------------------------------ Oura
def test_oura_csv(tmp_path):
    hdr = ["date", "Sleep Score", "Total Sleep Duration", "Bedtime Start", "Bedtime End", "Sleep Efficiency",
           "Deep Sleep Duration", "REM Sleep Duration", "Average Resting Heart Rate", "Lowest Resting Heart Rate",
           "Average HRV", "HRV Balance Score", "Respiratory Rate", "Steps", "Activity Calories", "Temperature Deviation (°C)"]
    rows = [
        ["2026-09-01", "80", "25200", "2026-08-31T23:15:00-04:00", "2026-09-01T07:00:00-04:00", "90", "4800", "5400",
         "55.2", "50", "48", "70", "14.2", "8000", "450", "0.1"],
        ["2026-09-02", "75", "24000", "2026-09-01T23:30:00-04:00", "2026-09-02T06:55:00-04:00", "88", "4200", "6000",
         "56.0", "51", "45", "71", "14.0", "", "", "-0.2"],
        ["2026-09-03", "", "", "", "", "", "", "", "", "", "", "", "", "11000", "600", ""],
    ]
    f = tmp_path / "oura_2026-09-03_trends.csv"
    f.write_text(_csv(hdr, rows))
    assert sniff(str(f)) == "oura"
    ds = load(str(f))
    _common_checks(ds, "oura")
    r = _row(ds, "2026-09-01")
    assert r["sleep_min"] == 420 and r["sleep_deep_min"] == 80 and r["sleep_rem_min"] == 90
    assert r["rhr"] == 55.2 and r["hrv_rmssd"] == 48 and r["resp_rate"] == 14.2
    assert r["sleep_efficiency"] == 90 and r["steps"] == 8000 and r["active_kcal"] == 450
    assert r["sleep_onset"] == "2026-08-31T23:15:00" and r["sleep_offset"] == "2026-09-01T07:00:00"
    assert pd.isna(_row(ds, "2026-09-02")["steps"]) and _row(ds, "2026-09-03")["steps"] == 11000
    assert "skin_temp_c" not in ds.daily.columns and "spo2" not in ds.daily.columns   # deviation is not absolute
    assert ds.minute is None


# ------------------------------------------------------------------ edge
EDGE_T0 = pd.Timestamp("2026-09-01 12:00:00", tz="UTC")


def _edge_db(tmp_path, scale=1.0, name="edge.sqlite"):
    p = tmp_path / name
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE decoded_onehz (device_id TEXT, ts_ms INTEGER, rec_ts INTEGER, counter INTEGER, hr REAL, "
                "ax REAL, ay REAL, az REAL, spo2_red_raw INTEGER, spo2_ir_raw INTEGER, skin_temp_raw INTEGER, "
                "step_count INTEGER, step_cadence REAL, activity_class INTEGER, skin_temp_c REAL, on_wrist INTEGER)")
    con.execute("CREATE TABLE metric_series (date TEXT, key TEXT, value)")
    rows = []
    for day in range(2):
        base = int((EDGE_T0 + pd.Timedelta(days=day)).timestamp() * 1000)
        for s in range(2 * 3600):                       # 2 h of 1 Hz data
            minute = s // 60
            az = 1.0
            if minute == 5:
                az = 1.1                                # ENMO 100 mg
            elif minute == 6:
                az = 0.9                                # below 1 g -> clipped to 0
            hr = 60.0 + (s % 2) * 10                    # alternating 60/70 -> 65
            if minute == 7:
                hr = 0.0                                # no lock -> NaN
            on = 0 if minute == 8 else 1                # off-wrist minute dropped entirely
            rows.append(("dev1", base + s * 1000, base, s, hr, 0.0, 0.0, az * scale, 0, 0, 0, 0, 0.0, 0, 33.0, on))
    con.executemany("INSERT INTO decoded_onehz VALUES (" + ",".join("?" * 16) + ")", rows)
    ms = [("2026-09-01", "rhr", 52), ("2026-09-01", "hrv", 61.5), ("2026-09-01", "steps", 9000),
          ("2026-09-01", "readiness", 77), ("2026-09-01", "strain", 9.1), ("2026-09-01", "resp_rate", 15.1),
          ("2026-09-01", "sleep_json", json.dumps({"sleep_min": 432, "deep_min": 85, "rem_min": 95,
                                                    "efficiency": 0.9, "onset": "2026-08-31T23:30:00",
                                                    "offset": "2026-09-01T07:12:00"})),
          ("2026-09-02", "rhr", 53), ("2026-09-02", "rhr", 54),   # rewritten: last wins
          ("2026-09-02", "steps", 7000)]
    con.executemany("INSERT INTO metric_series VALUES (?,?,?)", ms)
    con.commit()
    con.close()
    return p


def test_edge_sqlite(tmp_path):
    p = _edge_db(tmp_path)
    assert sniff(str(p)) == "edge"
    ds = load(str(p))
    _common_checks(ds, "edge")
    r1, r2 = _row(ds, "2026-09-01"), _row(ds, "2026-09-02")
    assert r1["rhr"] == 52 and r1["hrv_rmssd"] == 61.5 and r1["steps"] == 9000 and r1["strain"] == 9.1
    assert r1["resp_rate"] == 15.1 and r1["sleep_min"] == 432 and r1["sleep_deep_min"] == 85
    assert r1["sleep_rem_min"] == 95 and r1["sleep_efficiency"] == pytest.approx(90)
    assert r1["sleep_onset"] == "2026-08-31T23:30:00" and r1["sleep_offset"] == "2026-09-01T07:12:00"
    assert r2["rhr"] == 54 and r2["steps"] == 7000 and pd.isna(r2["sleep_min"])
    assert "spo2" not in ds.daily.columns
    m = ds.minute
    assert set(m.columns) == {"ts", "hr", "enmo_mg", "source"}
    assert len(m) == 2 * (120 - 1)                      # off-wrist minute 8 dropped each day
    assert str(m["ts"].dt.tz) == "UTC"
    by = m.set_index("ts")
    t = lambda mi, day=0: EDGE_T0 + pd.Timedelta(days=day, minutes=mi)
    assert by.loc[t(0), "enmo_mg"] == pytest.approx(0.0) and by.loc[t(0), "hr"] == pytest.approx(65)
    assert by.loc[t(5), "enmo_mg"] == pytest.approx(100.0)
    assert by.loc[t(6), "enmo_mg"] == pytest.approx(0.0)
    assert pd.isna(by.loc[t(7), "hr"]) and by.loc[t(7), "enmo_mg"] == pytest.approx(0.0)
    assert t(8) not in by.index
    assert by.loc[t(5, 1), "enmo_mg"] == pytest.approx(100.0)
    assert ds.coverage["minute_days"] == 2


@pytest.mark.parametrize("scale", [9.80665, 1000.0])
def test_edge_accel_units(tmp_path, scale):
    ds = load(str(_edge_db(tmp_path, scale=scale)))
    by = ds.minute.set_index("ts")
    assert by.loc[EDGE_T0 + pd.Timedelta(minutes=5), "enmo_mg"] == pytest.approx(100.0)


def test_edge_local_tz(tmp_path):
    ds = load(str(_edge_db(tmp_path)), tz="America/New_York")
    first = ds.minute["ts"].iloc[0]
    assert first.tz is not None and (first.hour, first.minute) == (8, 0)   # 12:00 UTC = 08:00 EDT


def test_unrecognised(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello,world\n1,2\n")
    with pytest.raises(ValueError):
        load(str(f))
