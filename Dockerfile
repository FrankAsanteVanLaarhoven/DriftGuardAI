# syntax=docker/dockerfile:1
# Multi-stage build: hermetic uv-locked deps, non-root, read-only-rootfs friendly.

FROM python:3.12-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv
WORKDIR /app
# Layer 1: dependency graph only (cached unless the lock changes).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DRIFTGUARD_LOG_LEVEL=INFO
WORKDIR /app

# Unprivileged runtime user.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
# Committed fallback + evaluative artifacts. The primary (if trained before build)
# is copied too; otherwise the service starts degraded on the baseline.
COPY models ./models
COPY artifacts ./artifacts

# MLflow (shares this image) stores its sqlite backend under /mlflow. Pre-create it owned by the
# unprivileged uid so a fresh named volume mounted there inherits writable ownership — otherwise
# the non-root server cannot open the database file.
RUN mkdir -p /mlflow && chown 10001:10001 /mlflow

USER 10001
EXPOSE 8000

# Liveness: the process is up. /ready governs traffic admission in Kubernetes.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').getcode()==200 else 1)"

CMD ["uvicorn", "driftguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
