# Motherhood Companion AI — Local Ollama Backend

100% Local & Free AI backend for the Navajeev Motherhood Companion Flutter Application.

## Architecture
- **LLM Text Generation**: Ollama `llama3.2:3b` (local CPU/GPU)
- **Vector Embeddings**: Ollama `nomic-embed-text` (768 dimensions)
- **Vector Database**: Pinecone Serverless Index (`langchainvector-ollama`)
- **API Framework**: FastAPI
- **OpenAI Dependencies**: 0 (100% Free / Local inference)

## Setup & Running

1. **Install and run Ollama**:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

2. **Configure `.env`**:
   ```env
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=langchainvector-ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:3b
   OLLAMA_EMBED_MODEL=nomic-embed-text
   ```

3. **Ingest Documents (one-time setup)**:
   ```bash
   python bot.py
   ```

4. **Start the FastAPI Server**:
   ```bash
   uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
   ```

## Endpoints
- `GET /` — Health check & Ollama status
- `POST /ask` — Knowledge RAG endpoint
- `POST /ai/insight` — Module tracker insights (sleep, wellbeing, feeding, growth, etc.)
- `POST /ai/recommend` — Personalized weekly guidance
