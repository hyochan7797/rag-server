# FastAPI RAG Server

This repository runs the AI/RAG server locally. The Spring server can run on AWS EC2 and call this server through a public tunnel such as ngrok.

## Local Setup

Create `.env` from the example and fill in your real keys:

```bash
cp .env.example .env
```

Required values:

```text
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
FSS_API_KEY=...
ADMIN_API_KEY=...
```

`ADMIN_API_KEY` must be the same value as the Spring EC2 `.env`.

## Run Qdrant

```bash
docker compose up -d qdrant
```

## Run FastAPI

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Load Vector Data

Run this after Qdrant is up and `.env` has the API keys:

```bash
curl -X POST http://localhost:8000/admin/refresh \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY"
```

Chat test:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"question":"전세대출 추천해줘"}'
```

## Connect Spring on AWS

Expose local FastAPI:

```bash
ngrok http 8000
```

Set these values in the Spring EC2 `.env`:

```text
FASTAPI_URL=https://YOUR_NGROK_DOMAIN/chat
FASTAPI_ADMIN_URL=https://YOUR_NGROK_DOMAIN/admin/refresh
ADMIN_API_KEY=same-value-as-fastapi
LOAN_REFRESH_CRON=-
```

Then restart Spring on EC2:

```bash
cd ~/rag
docker compose -f docker-compose.spring-aws.yml pull
docker compose -f docker-compose.spring-aws.yml up -d
```

The service does not crawl or embed data during startup. Use `/admin/refresh` manually when you need to rebuild the vector store.
