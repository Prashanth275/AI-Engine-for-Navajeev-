import os
import json
import hashlib
import urllib.request
import urllib.error
from typing import Optional
from utils.text_utils import normalize_query
from dotenv import load_dotenv
from fastapi import HTTPException

# Explicitly load .env from project root directory
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache.json")

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = {}
else:
    cache = {}


def save_cache():
    temp_file = CACHE_FILE + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(temp_file, CACHE_FILE)
    except Exception as e:
        print(f"[CACHE NOTICE] Failed to save cache: {e}")


def get_ollama_config():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    return base_url, model


def check_ollama_health() -> dict:
    """
    Checks if Ollama is running and whether the configured model is installed.
    """
    base_url, model = get_ollama_config()
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                model_installed = any(
                    m == model or m.startswith(f"{model}:") or model.startswith(f"{m}:")
                    for m in models
                )
                return {
                    "available": True,
                    "model_installed": model_installed,
                    "configured_model": model,
                    "installed_models": models,
                    "base_url": base_url,
                }
    except urllib.error.URLError as e:
        return {
            "available": False,
            "model_installed": False,
            "configured_model": model,
            "error": f"Connection refused at {base_url}. Is Ollama running?",
            "base_url": base_url,
        }
    except Exception as e:
        return {
            "available": False,
            "model_installed": False,
            "configured_model": model,
            "error": str(e),
            "base_url": base_url,
        }


def generate_with_ollama(prompt: str, question: Optional[str] = None) -> str:
    """
    Send a prompt to local Ollama and return the text response.
    Enforces clean output and caches response in cache.json.
    """
    base_url, model = get_ollama_config()

    cache_input = question if question else prompt
    clean_q = normalize_query(cache_input)
    cache_key = hashlib.md5(f"{model}:{clean_q}".encode("utf-8")).hexdigest()

    if cache_key in cache:
        print(f"[CACHE HIT] Returning cached response for key: {cache_key[:8]}...")
        return cache[cache_key]

    print(f"[OLLAMA INFERENCE] Calling Model: {model} at {base_url}...")

    formatted_prompt = f"""IMPORTANT FORMATTING RULES:
- Do NOT use markdown asterisks or stars (no * or **)
- Use clean, compassionate plain text or strict JSON when requested
- Maintain helpful maternal and infant health guidance

{prompt}
"""

    payload = {
        "model": model,
        "prompt": formatted_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status == 200:
                result_data = json.loads(resp.read().decode("utf-8"))
                answer = result_data.get("response", "").strip()

                if not answer:
                    raise HTTPException(
                        status_code=500,
                        detail="Ollama returned an empty response.",
                    )

                cache[cache_key] = answer
                if len(cache) > 1000:
                    first_key = next(iter(cache))
                    del cache[first_key]
                save_cache()

                return answer
            else:
                raise HTTPException(
                    status_code=resp.status,
                    detail=f"Ollama server returned HTTP {resp.status}",
                )

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            pass

        if e.code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Ollama model '{model}' not found on server {base_url}. Please run 'ollama pull {model}'. Details: {error_body}",
            )
        raise HTTPException(
            status_code=e.code,
            detail=f"Ollama HTTP error ({e.code}): {error_body or str(e)}",
        )

    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is unavailable at {base_url}. Please ensure Ollama is running ('ollama serve'). Error: {e.reason}",
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama generation failed: {str(e)}",
        )
