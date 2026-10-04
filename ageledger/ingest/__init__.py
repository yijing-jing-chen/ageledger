"""Export file -> canonical `Dataset`.

    from ageledger.ingest import load, sniff
    ds = load("my_whoop_export.zip")         # whoop | apple | oura | edge, detected from content
    ds = load("edge.sqlite", tz="Europe/Berlin")   # tz only matters for edge (ts_ms is epoch UTC)

Every ingester returns `daily` sorted by date with only the columns the source actually provided,
`coverage` filled, and passes `schema.validate` (else ValueError).
"""
from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

from ..schema import Dataset

SOURCES = ("whoop", "apple", "oura", "edge")


def sniff(path: str) -> str:
    """Return 'whoop' | 'apple' | 'oura' | 'edge' from the file's content (not just its extension)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.is_dir():
        names = {f.name.lower() for f in p.rglob("*") if f.is_file()}
        return _from_names(names, path)
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as zf:
            names = {PurePosixPath(n).name.lower() for n in zf.namelist() if not n.startswith("__MACOSX")}
        return _from_names(names, path)
    with open(p, "rb") as fh:
        head = fh.read(65536)
    if head.startswith(b"SQLite format 3\x00"):
        return "edge"
    low = head.lower()
    if b"healthdata" in low and (low.lstrip().startswith(b"<?xml") or low.lstrip().startswith(b"<")):
        return "apple"
    text = head.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0].lower() if text.strip() else ""
    if "cycle start time" in first:
        return "whoop"            # a bare physiological_cycles.csv
    cols = [c.strip().strip('"') for c in first.replace(";", ",").split(",")]
    if any(c in ("date", "day", "summary date") for c in cols):
        return "oura"
    raise ValueError(f"{path}: unrecognised export format (expected WHOOP zip, Apple export.zip/xml, Oura CSV, edge SQLite)")


def _from_names(names: set[str], path: str) -> str:
    if "physiological_cycles.csv" in names:
        return "whoop"
    if "export.xml" in names:
        return "apple"
    csvs = [n for n in names if n.endswith(".csv")]
    raise ValueError(f"{path}: archive is neither a WHOOP export (physiological_cycles.csv) nor an Apple Health "
                     f"export (export.xml); csv files seen: {sorted(csvs)[:10]}")


def load(path: str, source: Optional[str] = None, tz: Optional[str] = None) -> Dataset:
    """Sniff `path` (or trust `source`) and dispatch to the matching ingester."""
    path = str(path)
    src = source or sniff(path)
    if src == "whoop":
        from . import whoop
        return whoop.load(path)
    if src == "apple":
        from . import apple
        return apple.load(path)
    if src == "oura":
        from . import oura
        return oura.load(path)
    if src == "edge":
        from . import edge
        return edge.load(path, tz=tz)
    raise ValueError(f"unknown source {src!r}; expected one of {SOURCES}")


__all__ = ["load", "sniff", "SOURCES"]
