# syntax=docker/dockerfile:1
# Built and run on linux/amd64 (as required by the brief).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright WATCHTOWER_DATA_DIR=/data
WORKDIR /app

# Playwright's bundled Node driver needs libatomic before it can install Chromium's own dependencies.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libatomic1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first (cached layer), then Chromium only (used for attach/repair discovery, never for regular reads).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m playwright install --with-deps chromium \
 && rm -rf /var/lib/apt/lists/*

COPY watchtower ./watchtower
# The demo portal for the heal page (/_demo/portal/, /_ui/heal); WATCHTOWER_DEMO=0 leaves it unmounted.
COPY mock_portal ./mock_portal
RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8080
CMD ["uvicorn", "watchtower.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
