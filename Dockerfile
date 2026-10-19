# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, on their own layer, so code changes rebuild quickly.
# requirements.txt is pinned with hashes (see the header of that file to refresh it).
COPY requirements.txt .
RUN pip install --require-hashes -r requirements.txt

COPY alma ./alma
COPY frontend ./frontend
# Placeholder signal map; in production mount a volume over this file (see docker-compose.override.yml).
COPY signals.json ./signals.json
# Default telegram format; in production mount a volume over this file.
COPY udp_format.json ./udp_format.json

# No root in the container.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin alma
USER alma

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

# TLS and the public URL are handled by the reverse proxy; see README. Behind a proxy uvicorn can show the real
# client IPs in the logs if FORWARDED_ALLOW_IPS is set to the proxy network.
CMD ["uvicorn", "alma.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
