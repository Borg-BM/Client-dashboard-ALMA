# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Eerst alleen de afhankelijkheden: aparte laag, dus code-wijzigingen herbouwen snel.
# requirements.txt is vastgepind met hashes (zie de kop van dat bestand om hem te vernieuwen).
COPY requirements.txt .
RUN pip install --require-hashes -r requirements.txt

COPY alma ./alma
COPY frontend ./frontend
# Placeholder-signaalmap; in productie met een volume over deze file heen mounten (zie docker-compose.yml).
COPY signals.json ./signals.json
# Standaard telegramformaat; in productie met een volume over deze file heen mounten.
COPY udp_format.json ./udp_format.json

# Geen root in de container.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin alma
USER alma

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

# TLS en de publieke URL regelt de reverse proxy; zie README. Achter een proxy kan uvicorn de echte
# client-IP's in de logs tonen als FORWARDED_ALLOW_IPS op het proxy-netwerk wordt gezet.
CMD ["uvicorn", "alma.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
