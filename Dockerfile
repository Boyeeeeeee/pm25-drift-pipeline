FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so rebuilds after a code change reuse this cached layer
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Only what the API needs: code, SQL views, monitoring config, the trained model and the logs
COPY src ./src
COPY sql ./sql
COPY config ./config
COPY models/latest.joblib ./models/latest.joblib
COPY logs ./logs

# Run as a normal user, not root
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Build a fresh DuckDB from the committed logs, then start the API
CMD ["sh", "-c", "python src/logstore.py import && exec uvicorn serve:app --app-dir src --host 0.0.0.0 --port 8000"]
