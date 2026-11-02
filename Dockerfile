FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements change.
COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

# Create the non-root user BEFORE copying, then copy files owned by it.
RUN useradd --create-home appuser

COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser models/fish_classifier.joblib models/fish_classifier.joblib

USER appuser

EXPOSE 8000

# Healthy only when the model is actually loaded, not just when HTTP answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import json,sys,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)); sys.exit(0 if r['model_loaded'] else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
