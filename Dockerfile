# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — Fake Account Detection API
# ──────────────────────────────────────────────────────────────────────────────
# Stage: single-stage Python 3.11 slim image
# Exposes: port 5000
# Dataset: mounted at /data via docker-compose volume
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
# gcc needed for some compiled wheel builds (e.g. numpy C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer-caching optimisation) ────────────
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app/ /app/

# ── Directory for PyVis HTML output ──────────────────────────────────────────
RUN mkdir -p /tmp/viz

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser /app /tmp/viz
USER appuser

# ── Environment defaults (overridable via docker-compose / -e flags) ──────────
ENV DATASET_PATH=/data/twitter_combined.txt \
    DATASET_SEP=" " \
    MAX_ROWS=500000 \
    TOP_N=5000 \
    CHUNKSIZE=100000 \
    PORT=5000 \
    FLASK_DEBUG=false \
    PRELOAD_DATA=true

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

EXPOSE 5000

# ── Start with gunicorn for production (4 workers) ────────────────────────────
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--preload", \
     "--timeout", "300", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "main:app"]