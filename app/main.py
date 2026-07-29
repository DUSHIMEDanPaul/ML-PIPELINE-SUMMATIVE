"""
FastAPI serving layer for the HAM10000 skin-lesion classifier.

Endpoints
---------
GET  /                 -> redirect to /docs
GET  /health           -> status, model loaded, uptime, model file mtime
POST /predict          -> multipart image, returns label + probability + latency
POST /upload           -> bulk images (multipart list OR a .zip) -> batch_id + counts
POST /retrain          -> accepts a batch_id, returns a job_id IMMEDIATELY
GET  /status/{job_id}  -> queued | running | done | failed (+ progress, metrics)
GET  /metrics          -> model test metrics + request count / mean / p95 latency

Design guarantees
-----------------
* /retrain never blocks: it validates, creates a job file + lock, schedules a
  BackgroundTask (run in Starlette's threadpool, so the heavy sync retrain does
  NOT stall the event loop) and returns a job_id in milliseconds.
* A lock file (jobs/retrain.lock) prevents concurrent retrains.
* Every request's latency is pushed to a rolling in-memory buffer that /metrics
  summarises.
* Every handler returns useful JSON on error, never a raw stack trace.
"""

from __future__ import annotations

import json
import os
import shutil
import asyncio
import time
import uuid
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from src import model as model_module
from src import preprocessing
from src.prediction import predict as predict_one

# ---------------------------------------------------------------------------
# Paths (relative to repo root, env-overridable)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", REPO_ROOT / "uploads"))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", REPO_ROOT / "jobs"))
RETRAIN_LOCK = JOBS_DIR / "retrain.lock"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
_ZIP_EXTS = {".zip"}

# Bound concurrent TF inferences per process so a burst of requests queues
# cheaply instead of oversubscribing the CPU (dozens of inference threads
# thrashing on a few cores collapses throughput). Sized to the container's CPU
# budget; excess requests wait their turn rather than degrading everyone's.
MAX_CONCURRENT_INFERENCE = int(os.environ.get("MAX_CONCURRENT_INFERENCE", "3"))
_INFER_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)

# ---------------------------------------------------------------------------
# In-memory request-latency telemetry
# ---------------------------------------------------------------------------
BOOT_TIME = time.time()
_LATENCIES: deque[float] = deque(maxlen=2000)  # milliseconds, rolling
_REQUEST_COUNT = 0
_TELEMETRY_LOCK = Lock()


def _record_latency(ms: float) -> None:
    global _REQUEST_COUNT
    with _TELEMETRY_LOCK:
        _REQUEST_COUNT += 1
        _LATENCIES.append(ms)


def _latency_summary() -> dict:
    with _TELEMETRY_LOCK:
        count = _REQUEST_COUNT
        samples = list(_LATENCIES)
    if not samples:
        return {"request_count": count, "mean_latency_ms": None, "p95_latency_ms": None}
    ordered = sorted(samples)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "request_count": count,
        "mean_latency_ms": round(sum(ordered) / len(ordered), 2),
        "p95_latency_ms": round(ordered[p95_idx], 2),
        "samples_in_window": len(ordered),
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HAM10000 Skin-Lesion Classifier API",
    description="Binary benign/malignant classifier serving layer.",
    version="1.0.0",
)


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    # Skip telemetry/doc noise so the numbers reflect real API usage.
    if request.url.path not in ("/metrics", "/health", "/favicon.ico") and not request.url.path.startswith("/docs"):
        _record_latency(elapsed_ms)
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.on_event("startup")
def _warm_start() -> None:
    """Optionally pre-load the model so the first /predict isn't slow."""
    if os.environ.get("WARM_START", "1") == "1":
        try:
            model_module.load()
        except Exception:  # noqa: BLE001  (health will still report not-loaded)
            pass


# ---------------------------------------------------------------------------
# Uniform error handling: JSON, never a stack trace
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc), "path": request.url.path},
    )


def _bad_request(detail: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": "bad_request", "detail": detail})


# ---------------------------------------------------------------------------
# Root & health
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    mtime = model_module.model_file_mtime()
    return {
        "status": "ok",
        "model_loaded": model_module.is_loaded(),
        "model_file_present": mtime is not None,
        "model_file_mtime": mtime,
        "model_file_mtime_iso": (
            datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat() if mtime else None
        ),
        "uptime_seconds": round(time.time() - BOOT_TIME, 1),
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        raw = await file.read()
    except Exception as exc:  # noqa: BLE001
        return _bad_request(f"Could not read upload: {exc}")

    if not raw:
        return _bad_request("Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        return _bad_request(
            f"File too large ({len(raw)} bytes > {MAX_UPLOAD_BYTES}).", status=413
        )

    try:
        # Offload the blocking TF inference to a worker thread so it never
        # stalls the event loop, and bound concurrency with a semaphore so a
        # burst queues cheaply instead of thrashing the CPU.
        async with _INFER_SEMAPHORE:
            result = await run_in_threadpool(predict_one, raw)
    except preprocessing.PreprocessError as exc:
        return _bad_request(f"Invalid image: {exc}")
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"error": "model_unavailable", "detail": str(exc)})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": "prediction_failed", "detail": str(exc)})

    result["filename"] = file.filename
    return result


# ---------------------------------------------------------------------------
# Bulk upload
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_endpoint(files: List[UploadFile] = File(...)):
    if not files:
        return _bad_request("No files supplied.")

    batch_id = uuid.uuid4().hex[:12]
    batch_dir = UPLOADS_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    extracted = 0
    rejected: list[dict] = []

    for upload in files:
        name = os.path.basename(upload.filename or "")
        if not name:
            rejected.append({"filename": upload.filename, "reason": "no filename"})
            continue
        try:
            data = await upload.read()
        except Exception as exc:  # noqa: BLE001
            rejected.append({"filename": name, "reason": f"read error: {exc}"})
            continue

        if len(data) > MAX_UPLOAD_BYTES:
            rejected.append({"filename": name, "reason": "over size limit"})
            continue

        ext = Path(name).suffix.lower()
        if ext in _ZIP_EXTS:
            extracted += _extract_zip(data, batch_dir, rejected)
        else:
            (batch_dir / name).write_bytes(data)
            saved += 1

    # Count per-class from whatever labelled images landed in the batch dir.
    counts = _count_batch(batch_dir)

    if counts["total_images"] == 0:
        shutil.rmtree(batch_dir, ignore_errors=True)
        return _bad_request(
            "No usable images found in upload. Provide images (jpg/png) named "
            "0_*/1_* or inside benign/ and malignant/ folders (a .zip is fine)."
        )

    return {
        "batch_id": batch_id,
        "saved_files": saved,
        "extracted_from_zip": extracted,
        "rejected": rejected,
        "counts": counts,
    }


def _extract_zip(data: bytes, batch_dir: Path, rejected: list) -> int:
    """Safely extract image entries from a zip into batch_dir. Returns count."""
    import io

    count = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member = info.filename
                # Guard against path traversal / absolute paths in the archive.
                dest = (batch_dir / member).resolve()
                if not str(dest).startswith(str(batch_dir.resolve())):
                    rejected.append({"filename": member, "reason": "unsafe zip path"})
                    continue
                if info.file_size > MAX_UPLOAD_BYTES:
                    rejected.append({"filename": member, "reason": "zip member too large"})
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                count += 1
    except zipfile.BadZipFile:
        rejected.append({"filename": "<zip>", "reason": "corrupt zip"})
    return count


def _count_batch(batch_dir: Path) -> dict:
    """Per-class counts of labelled images in a batch directory."""
    benign = malignant = unlabeled = 0
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in batch_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in allowed:
            continue
        label = preprocessing.parse_label(p)
        if label == 1:
            malignant += 1
        elif label == 0:
            benign += 1
        else:
            unlabeled += 1
    return {
        "benign": benign,
        "malignant": malignant,
        "unlabeled": unlabeled,
        "total_images": benign + malignant + unlabeled,
    }


# ---------------------------------------------------------------------------
# Retrain (async) + job status
# ---------------------------------------------------------------------------
class RetrainRequest(BaseModel):
    batch_id: str
    epochs: Optional[int] = None
    subsample: Optional[int] = None


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _write_job(job_id: str, **updates) -> dict:
    path = _job_path(job_id)
    state = {}
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    state.update(updates)
    state["job_id"] = job_id  # always record the id (never pass it via updates)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return state


def _read_job(job_id: str) -> Optional[dict]:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"job_id": job_id, "status": "failed", "error": "corrupt job file"}


# Coarse progress weighting per retrain stage, for a UI progress bar.
_STAGE_PROGRESS = {
    "prepared": 5,
    "loading_base": 15,
    "train": 20,      # scaled across epochs below
    "evaluating": 85,
    "promoting": 95,
    "done": 100,
}


def _run_retrain(job_id: str, batch_id: str, epochs: Optional[int], subsample: Optional[int]) -> None:
    """Background worker (runs in Starlette threadpool). Owns the lock file."""
    try:
        _write_job(job_id, status="running", progress=1, stage="starting")

        batch_dir = UPLOADS_DIR / batch_id
        if not batch_dir.is_dir():
            _write_job(job_id, status="failed", progress=0,
                       error=f"Unknown batch_id '{batch_id}'.")
            return

        try:
            X, y = preprocessing.build_array_from_dir(batch_dir)
        except preprocessing.PreprocessError as exc:
            _write_job(job_id, status="failed", progress=0, error=str(exc))
            return

        _write_job(job_id, status="running", progress=3, stage="loaded_data",
                   uploaded_counts=preprocessing.count_by_class(y))

        n_epochs = epochs or model_module.RETRAIN_EPOCHS

        def progress_cb(event: dict) -> None:
            stage = event.get("stage", "")
            pct = _STAGE_PROGRESS.get(stage, None)
            if stage == "train":
                ep = event.get("epoch", 0)
                # spread the training band (20 -> 80) across epochs
                pct = 20 + int(60 * (ep / max(1, n_epochs)))
            update = {"stage": stage, "last_event": event}
            if pct is not None:
                update["progress"] = pct
            _write_job(job_id, status="running", **update)

        report = model_module.retrain(
            X, y,
            epochs=n_epochs,
            subsample=subsample or model_module.RETRAIN_SUBSAMPLE,
            progress_cb=progress_cb,
        )

        _write_job(job_id, status="done", progress=100, stage="done", report=report)
    except Exception as exc:  # noqa: BLE001
        _write_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Always release the concurrency lock.
        try:
            RETRAIN_LOCK.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/retrain")
def retrain_endpoint(req: RetrainRequest, background_tasks: BackgroundTasks):
    batch_dir = UPLOADS_DIR / req.batch_id
    if not batch_dir.is_dir():
        return _bad_request(f"Unknown batch_id '{req.batch_id}'. Upload a batch first.")

    # Guard against concurrent retrains with an atomic lock-file create.
    try:
        fd = os.open(RETRAIN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, datetime.now(timezone.utc).isoformat().encode())
        os.close(fd)
    except FileExistsError:
        return JSONResponse(
            status_code=409,
            content={"error": "retrain_in_progress",
                     "detail": "Another retrain is already running. Try again once it finishes."},
        )

    # From here the lock is held; if setup fails before the background task can
    # take ownership (which releases it in its finally), release it here so the
    # lock never leaks and blocks future retrains.
    try:
        job_id = uuid.uuid4().hex[:12]
        _write_job(
            job_id,
            batch_id=req.batch_id,
            status="queued",
            progress=0,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        background_tasks.add_task(_run_retrain, job_id, req.batch_id, req.epochs, req.subsample)
    except Exception as exc:  # noqa: BLE001
        RETRAIN_LOCK.unlink(missing_ok=True)
        return JSONResponse(
            status_code=500,
            content={"error": "retrain_setup_failed", "detail": str(exc)},
        )

    return {"job_id": job_id, "status": "queued", "batch_id": req.batch_id,
            "poll": f"/status/{job_id}"}


@app.get("/status/{job_id}")
def status_endpoint(job_id: str):
    job = _read_job(job_id)
    if job is None:
        return _bad_request(f"Unknown job_id '{job_id}'.", status=404)
    return job


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.get("/metrics")
def metrics_endpoint():
    meta = model_module.get_meta()
    return {
        "model": {
            "test_metrics": meta.get("test_metrics", {}),
            "operating_threshold": meta.get("operating_threshold"),
            "trained_at": meta.get("trained_at"),
            "retrained_at": meta.get("retrained_at"),
            "model_file_mtime": model_module.model_file_mtime(),
        },
        "requests": _latency_summary(),
        "uptime_seconds": round(time.time() - BOOT_TIME, 1),
    }
