import os
from dotenv import load_dotenv

# Explicitly load .env from current directory first
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.pinecone_service import get_vectorstore
from services.ollama_service import check_ollama_health, get_ollama_config
from ai_engine.rag import run_rag
from ai_engine.insight_engine import run_insight
from ai_engine.recommendation_engine import run_recommendation
from schemas.schemas import (
    QuestionRequest, QuestionResponse,
    InsightRequest, InsightResponse,
    RecommendRequest, RecommendResponse,
    HealthResponse
)

app = FastAPI(
    title="Motherhood Companion AI Backend (Ollama Local)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None


@app.on_event("startup")
async def startup_event():
    global vectorstore
    print("--------------------------------------------------")
    print("[STARTUP] Initializing Motherhood Companion AI...")
    base_url, model, *_ = get_ollama_config()
    print(f"[CONFIG] Ollama URL: {base_url} | Model: {model}")
    
    health = check_ollama_health()
    if health.get("available"):
        if health.get("model_installed"):
            print(f"[STATUS] Ollama is connected and model '{model}' is ready.")
        else:
            print(f"[WARNING] Ollama is running, but model '{model}' is not installed.")
            print(f"[ACTION REQUIRED] Run 'ollama pull {model}' in your command prompt.")
    else:
        print(f"[NOTICE] Ollama server is currently offline at {base_url}.")
        print("[ACTION REQUIRED] Start Ollama by running 'ollama serve' or opening the Ollama desktop app.")

    try:
        print("[VECTORSTORE] Initializing Pinecone vectorstore...")
        vectorstore = get_vectorstore()
        print("[VECTORSTORE] Pinecone vectorstore successfully connected.")
    except Exception as e:
        print(f"[VECTORSTORE NOTICE] {e}")
    print("--------------------------------------------------")


@app.get("/", response_model=HealthResponse)
async def root():
    base_url, model, *_ = get_ollama_config()
    health = check_ollama_health()
    
    ollama_status = "connected" if health.get("available") and health.get("model_installed") else "offline_or_model_missing"
    msg = f"Backend running | Ollama: {ollama_status} ({model} @ {base_url})"
    return HealthResponse(
        status="healthy",
        message=msg
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        result = run_rag(vectorstore, request.question)

        return QuestionResponse(
            answer=result["answer"],
            context=result["context"] if request.include_context else None,
            success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        return QuestionResponse(
            answer=f"Error generating answer: {str(e)}",
            success=False
        )


@app.post("/ai/insight", response_model=InsightResponse)
async def get_insight(request: InsightRequest):
    """
    Analyzes user tracker data and returns a structured insight using Ollama.

    Supported modules:
    - sleep, wellbeing, feeding, growth, trimester, appointments, notifications, dashboard
    """
    try:
        data = request.data.copy()

        if request.baby_age_weeks is not None:
            data["baby_age_weeks"] = request.baby_age_weeks

        if request.subject is not None and "subject" not in data:
            data["subject"] = request.subject

        result = run_insight(request.module, data)

        return InsightResponse(
            module=request.module,
            success=True,
            result=result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/recommend", response_model=RecommendResponse)
async def get_recommendation(request: RecommendRequest):
    """
    Generates a personalized weekly guidance plan based on user profile using Ollama.
    """
    try:
        data = request.model_dump(exclude_none=True)

        result = run_recommendation(data)

        return RecommendResponse(
            success=True,
            result=result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

