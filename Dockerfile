# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# HAM10000 classifier — single image used by BOTH the API and the UI service
# (they differ only by the runtime command in docker-compose.yml).
#
# Multi-stage: the builder compiles/downloads all wheels into a venv; the
# runtime stage copies only that venv, so build tooling never ships in the
# final image. Base is python:3.11-slim with tensorflow-cpu.
# ---------------------------------------------------------------------------

# ---------- Stage 1: build the virtualenv --------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is only needed to compile any sdist-only deps; it stays in
# this stage and is discarded.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ---------- Stage 2: runtime ---------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Quiet TensorFlow and disable oneDNN so results match training numerics.
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# Runtime OS libs: curl for HEALTHCHECK, libgomp1 for TensorFlow's OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bring in the pre-built virtualenv.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Non-root user (fixed uid so bind-mounted dirs have predictable ownership).
RUN useradd --create-home --uid 1000 appuser

# Copy the project (models/ is included so the image works without bind mounts).
COPY --chown=appuser:appuser . /app

# Writable runtime dirs (bind mounts overlay these at runtime).
RUN mkdir -p /app/uploads /app/jobs /app/data/cache \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

# Default HEALTHCHECK targets the API; the UI service overrides it in compose.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default command runs the API; the UI service overrides `command:` in compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
