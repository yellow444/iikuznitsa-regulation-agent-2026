FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CORPUS_DIR=/data/corpus E5_MODEL_DIR=/data/model QUALITY_DB=/data/state/quality.sqlite3

WORKDIR /app
COPY requirements.txt requirements-semantic.txt ./
RUN pip install --no-cache-dir -r requirements-semantic.txt \
    && groupadd --system agent \
    && useradd --system --gid agent agent \
    && install -d -o agent -g agent /data/state
COPY app.py retrieval.py semantic.py index.html ./

USER agent
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
