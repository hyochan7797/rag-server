# FastAPI RAG service

This folder is ready to become a separate FastAPI repository.

## Files to keep

Keep all files in this folder except local secrets and caches:

```text
Dockerfile
requirements.txt
main.py
rag_pipeline.py
fss_crawler.py
filter_extraction.py
query_expansion.py
document_aliases.py
eval_retrieval.py
eval_golden_set.json
test_integration.py
app/data/
.dockerignore
.env.example
.gitignore
docker-compose.yml
README.md
```

Do not commit:

```text
.env
__pycache__/
.venv/
.cache/
```

## Local setup

```bash
cp .env.example .env
docker compose up -d qdrant
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Public tunnel for AWS Spring

Run FastAPI locally, then expose it:

```bash
ngrok http 8000
```

Use the generated HTTPS URL in the Spring repository's EC2 `.env`:

```text
FASTAPI_URL=https://your-ngrok-url/chat
FASTAPI_ADMIN_URL=https://your-ngrok-url/admin/refresh
```

The `ADMIN_API_KEY` value must match between this FastAPI `.env` and the Spring EC2 `.env`.

## Data refresh

After Qdrant is running and API keys are set:

```bash
curl -X POST http://localhost:8000/admin/refresh \
  -H "X-Admin-Key: <ADMIN_API_KEY>"
```

The service no longer crawls or embeds data on startup. It only reuses existing Qdrant data and waits for manual refresh when empty.
