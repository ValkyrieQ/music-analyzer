# syntax=docker/dockerfile:1.7
#
# Backend image: FastAPI + ffmpeg + yt-dlp + Demucs/torch + librosa.
#
# Two stages so the ~2.5 GB of build tooling torch drags in never reaches the runtime
# image. The wheel cache is mounted rather than copied, so rebuilds after a code change
# are fast and the layer stays small.

# --------------------------------------------------------------------------------------
# Stage 1: build the virtualenv
# --------------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0 \
    PYTHONDONTWRITEBYTECODE=1

# soundfile/librosa need libsndfile at build time; the rest is for compiling any wheel
# that has no prebuilt arm64 artefact.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch is installed first and on its own so the (large, slow) layer caches independently
# of our own requirements.
#
# Deliberately from PyPI rather than download.pytorch.org/whl/cpu: on linux/arm64 the
# PyPI wheels are already CPU-only — there are no aarch64 CUDA builds — so the CPU index
# buys nothing here, and it is unreachable behind TLS-inspecting proxies. On an amd64
# host, add `--index-url https://download.pytorch.org/whl/cpu` to avoid pulling ~2 GB of
# CUDA libraries that a CPU container never uses.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install torch==2.5.1 torchaudio==2.5.1

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# --------------------------------------------------------------------------------------
# Stage 2: runtime
# --------------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# ffmpeg carries the rubberband filter we use for pitch shifting; librubberband2 is its
# shared dependency. libsndfile1 is for soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        librubberband2 \
        libsndfile1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional corporate TLS-inspection root CA(s).
#
# On a network that intercepts TLS (Zscaler and friends), the container cannot verify
# upstream certificates and Demucs' first-run weight download fails with
# CERTIFICATE_VERIFY_FAILED. Dropping the proxy's root CA into backend/certs/*.crt makes it
# trusted here. The glob has a `.keep` fallback so the build still works when the directory
# is empty — on a normal network you need none of this.
COPY certs/ /usr/local/share/ca-certificates/extra/
RUN find /usr/local/share/ca-certificates/extra -name '*.crt' -type f | grep -q . \
        && update-ca-certificates \
        || echo "no extra CA certificates supplied; using the default trust store"

COPY --from=builder /opt/venv /opt/venv

# Point Python's TLS at the system trust store, which now includes any extra CAs above.
# requests/urllib3 (and therefore torch.hub, which fetches the Demucs weights) default to
# certifi's bundle and would otherwise ignore what we just installed.
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    LOG_DIR=/app/log \
    # Cache Demucs weights on a volume so they survive container replacement.
    TORCH_HOME=/data/models/torch \
    XDG_CACHE_HOME=/data/models/cache \
    NUMBA_CACHE_DIR=/tmp/numba

# NOTE: OMP_NUM_THREADS is deliberately *not* set here. It used to be pinned to 1 with the
# comment "torch manages its own pool", which is false — OpenMP latches the value at library
# init (on `import torch`), so Demucs ran single-threaded and separation took ~4x longer than
# it needed to. entrypoint.sh computes the real budget from the container's CPU allowance and
# exports it before Python starts. Set ANALYZER_THREADS to override.

# Run as a non-root user; /data is a mounted volume so its ownership is set at runtime.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data /app/log \
    && chown -R appuser:appuser /data /app

WORKDIR /app
COPY --chown=appuser:appuser entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
COPY --chown=appuser:appuser app ./app
# Verification/smoke-test helpers. Small, and worth having available in a running
# container to check a deployment: `docker compose exec api python tools/verify.py`.
COPY --chown=appuser:appuser tools ./tools

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "75"]
