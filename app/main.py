"""AgeLedger web app: one upload endpoint, one static page.

Pipeline per request: upload -> tempfile -> ageledger.ingest.load -> ageledger.report.build -> JSON.
Nothing is persisted; the tempfile is deleted in `finally`.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
SAMPLE_PATH = APP_DIR / "sample_ledger.json"

# Make `ageledger` importable when uvicorn is started from elsewhere.
_ROOT = str(APP_DIR.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MAX_BYTES = 512 * 1024 * 1024
CHUNK = 1024 * 1024
ALLOWED_SUFFIXES = (".zip", ".csv", ".xml", ".sqlite", ".db")
AGE_MIN, AGE_MAX = 10.0, 110.0
SEXES = ("female", "male")


def _load_pipeline():
    """Import lazily so the app runs (and picks the modules up) while other builders are still writing them."""
    try:
        from ageledger.ingest import load  # type: ignore
        from ageledger.report import build  # type: ignore
        return load, build, None
    except Exception as exc:  # ImportError, or a half-written module raising at import time
        return None, None, f"{type(exc).__name__}: {exc}"


def _jsonable(obj):
    """Coerce numpy/pandas scalars and NaN into plain JSON. report.build should already do this; belt and braces."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    item = getattr(obj, "item", None)  # numpy scalar
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    iso = getattr(obj, "isoformat", None)  # date / Timestamp
    if callable(iso):
        return iso()
    return str(obj)


def _parse_age(raw: Optional[str]) -> Optional[float]:
    """Optional `age` form field. Blank means not given; anything else must be a number in [10, 110]."""
    if raw is None or not str(raw).strip():
        return None
    try:
        age = float(str(raw).strip())
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Age must be a number of years between 10 and 110 (got '{raw}').")
    if not math.isfinite(age) or not (AGE_MIN <= age <= AGE_MAX):
        raise HTTPException(status_code=422, detail=f"Age must be between 10 and 110 years (got {raw}).")
    return age


def _parse_sex(raw: Optional[str]) -> Optional[str]:
    """Optional `sex` form field: 'female', 'male', or blank (prefer not to say)."""
    if raw is None or not str(raw).strip():
        return None
    sex = str(raw).strip().lower()
    if sex not in SEXES:
        raise HTTPException(status_code=422, detail=f"Sex must be 'female', 'male', or left blank (got '{raw}').")
    return sex


def _sample() -> dict:
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def _suffix(filename: str) -> str:
    name = (filename or "").lower()
    for s in ALLOWED_SUFFIXES:
        if name.endswith(s):
            return s
    return ""


def _banner_url() -> str:
    host, port = "127.0.0.1", "8000"
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            port = argv[i + 1]
        elif a.startswith("--port="):
            port = a.split("=", 1)[1]
        elif a == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
        elif a.startswith("--host="):
            host = a.split("=", 1)[1]
    host = os.environ.get("AGELEDGER_HOST", host)
    port = os.environ.get("AGELEDGER_PORT", port)
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{port}/"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load, _build, err = _load_pipeline()
    engine = "ready" if load else f"not importable yet ({err}); /api/analyze will serve the sample ledger"
    print(f"\n  AgeLedger  ->  {_banner_url()}\n  analysis engine: {engine}\n", flush=True)
    yield


app = FastAPI(title="AgeLedger", version="0.0.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/api/sample")
def sample() -> JSONResponse:
    return JSONResponse(_sample())


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    age: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
) -> JSONResponse:
    # Validate the profile fields before touching the upload, so a typo fails fast with a plain message.
    age_years = _parse_age(age)
    sex_val = _parse_sex(sex)
    entered = [k for k, v in (("age_years", age_years), ("sex", sex_val)) if v is not None]

    suffix = _suffix(file.filename or "")
    if not suffix:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file '{file.filename}'. Upload a .zip, .csv, .xml, .sqlite or .db export.",
        )

    fd, path = tempfile.mkstemp(prefix="ageledger-", suffix=suffix)
    try:
        written = 0
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="File is larger than 512 MB.")
                out.write(chunk)
        if written == 0:
            raise HTTPException(status_code=422, detail="The uploaded file is empty.")

        load, build, err = _load_pipeline()
        if load is None:
            # Engine not built yet: say so loudly rather than pretend the sample is the user's data.
            ledger = _sample()
            ledger["demo"] = True
            ledger["notice"] = "The analysis engine isn't installed in this build, so this is the sample ledger, not your data."
            return JSONResponse(ledger)

        try:
            ds = load(path)
            # User-entered profile wins over anything the export carried (WHOOP/Oura carry neither).
            if age_years is not None:
                ds.profile.age_years = age_years
                ds.profile.notes.append("age_years entered by user")
            if sex_val is not None:
                ds.profile.sex = sex_val
                ds.profile.notes.append("sex entered by user")
            ledger = build(ds)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc) or "Could not read this export.")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed ({type(exc).__name__}): {exc}")
        ledger = _jsonable(ledger)
        prof = ledger.get("profile")
        if not isinstance(prof, dict):
            prof = ledger["profile"] = {}
        prof["entered"] = entered
        if entered and isinstance(prof.get("assumed"), list):
            prof["assumed"] = [k for k in prof["assumed"] if k not in entered]
        return JSONResponse(ledger)
    finally:
        await file.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AGELEDGER_PORT", "8000"))
    os.environ.setdefault("AGELEDGER_PORT", str(port))
    uvicorn.run(app, host=os.environ.get("AGELEDGER_HOST", "127.0.0.1"), port=port)
