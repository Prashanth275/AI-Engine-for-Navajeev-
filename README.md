# Motherhood Companion AI — Backend

Cloud-powered AI backend for the Navajeev Motherhood Companion Application.

```
Flutter / Client
       |
       v
   FastAPI API
      /   \
     /     \
    v       v
Pinecone   Ollama Cloud
Cloud      gpt-oss:20b
Embeddings
llama-text-embed-v2
```

## Architecture

- **LLM Text Generation**: Ollama Cloud (`gpt-oss:20b` via `https://ollama.com` with Bearer authentication)
- **Vector Embeddings**: Pinecone Hosted Inference (`llama-text-embed-v2`, 768 dimensions)
- **Vector Database**: Pinecone Serverless Index (`langchainvector-cloud`, 301 knowledge-base chunks indexed)
- **API Framework**: FastAPI
- **Zero Local GPU / Daemon Requirements**: Fully cloud-backed inference and retrieval; no local Ollama service or GPU required at runtime.

---

## Setup & Configuration

### 1. Configure `.env`

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=langchainvector-cloud
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your-ollama-api-key
OLLAMA_MODEL=gpt-oss:20b
PORT=8000
```


### 2. Install Dependencies

```bash
pip install -r requirements_backend.txt
```

### 3. Knowledge Base Ingestion (Offline Only)

The 301 knowledge chunks are already generated and hosted in the Pinecone cloud index (`langchainvector-cloud`).
The reference PDF is used solely for offline batch ingestion (`bot.py`) and is **not required at runtime or in production deployments**.

---

## Running the Application

### Local Development
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment
```bash
uvicorn backend:app --host 0.0.0.0 --port $PORT
```

---

## Endpoints

- `GET /` — Health check & Ollama Cloud connectivity status
- `POST /ask` — Knowledge RAG endpoint (Pinecone Hosted query embedding & retrieval + Ollama Cloud generation)
- `POST /ai/insight` — Module tracker insights (`sleep`, `wellbeing`, `feeding`, `growth`, etc. via Ollama Cloud)
- `POST /ai/recommend` — Personalized weekly guidance and action plan via Ollama Cloud
