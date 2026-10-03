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
    api_key = os.getenv("OLLAMA_API_KEY", "")
    return base_url, model, api_key


def check_ollama_health() -> dict:
    """
    Checks if Ollama (Cloud or Local) is running and accessible.
    """
    base_url, model, api_key = get_ollama_config()
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(f"{base_url}/api/tags", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                model_installed = any(
                    m == model or m.startswith(f"{model}:") or model.startswith(f"{m}:")
                    for m in models
                ) if models else True
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
            "error": f"Connection refused at {base_url}. Is Ollama accessible?",
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


def normalize_response_text(text: str) -> str:
    replacements = {
        "\u202f": " ",
        "\u00a0": " ",
        "\u200b": "",
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": " - ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }

    for search, replace in replacements.items():
        text = text.replace(search, replace)

    return text


def is_refusal_response(text: str) -> bool:
    """
    Checks if LLM response is a document-not-found refusal.
    """
    if not text or not isinstance(text, str):
        return False
    lowered = text.lower()
    refusal_patterns = [
        "couldn't find this information in the document",
        "could not find this information in the document",
        "couldn't find the answer in this document",
        "could not find the answer in this document",
        "couldn't find any information",
        "could not find any information",
        "not present in the context",
        "not found in the document",
        "information is not present in the document",
        "i couldn't find this information",
        "i could not find this information",
        "i couldn't find the answer",
        "i could not find the answer",
    ]
    return any(p in lowered for p in refusal_patterns)


def generate_with_ollama(prompt: str, question: Optional[str] = None, user_context: Optional[str] = None) -> str:
    """
    Send a prompt to Ollama (Cloud or Local) and return the text response.
    Enforces clean output and caches response in cache.json.
    Cache key depends on model, question/input identifier (including user_context if present), and the full prompt.
    """
    base_url, model, api_key = get_ollama_config()

    if user_context and user_context.strip() and question:
        cache_input = f"{user_context.strip()}:{question}"
    else:
        cache_input = question if question else prompt
    clean_q = normalize_query(cache_input)
    # Cache key depends on model, normalized query with context, and prompt (retrieved context)
    cache_key = hashlib.md5(f"{model}:{clean_q}:{prompt}".encode("utf-8")).hexdigest()

    if cache_key in cache:
        print(f"[CACHE HIT] Returning cached response for key: {cache_key[:8]}...")
        return cache[cache_key]

    is_cloud = bool(api_key or "ollama.com" in base_url)
    target_type = "Cloud" if is_cloud else "Local"
    print(f"[OLLAMA INFERENCE] Calling {target_type} Model: {model} at {base_url}...")

    formatted_prompt = f"""IMPORTANT FORMATTING RULES:
- Do NOT use markdown asterisks or stars (no * or **)
- Use clean, compassionate plain text or strict JSON when requested
- Maintain helpful maternal and infant health guidance

{prompt}
"""

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if is_cloud:
        endpoint = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": formatted_prompt}
            ],
            "stream": False,
        }
    else:
        endpoint = f"{base_url}/api/generate"
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
        endpoint,
        data=req_data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status == 200:
                result_data = json.loads(resp.read().decode("utf-8"))

                if is_cloud:
                    message = result_data.get("message", {})
                    answer = message.get("content", "").strip()
                else:
                    answer = result_data.get("response", "").strip()

                if not answer:
                    raise HTTPException(
                        status_code=500,
                        detail="Ollama returned an empty response.",
                    )

                # Normalize special Unicode spaces and typographic characters
                answer = normalize_response_text(answer)

                # Only cache valid answers, never cache refusal responses
                if not is_refusal_response(answer):
                    cache[cache_key] = answer
                    if len(cache) > 1000:
                        first_key = next(iter(cache))
                        del cache[first_key]
                    save_cache()
                else:
                    print(f"[CACHE NOTICE] Refusal response detected ('{answer[:40]}...'). Skipping cache save.")

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
                detail=f"Ollama model '{model}' not found on server {base_url}. Details: {error_body}",
            )
        raise HTTPException(
            status_code=e.code,
            detail=f"Ollama HTTP error ({e.code}): {error_body or str(e)}",
        )

    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is unavailable at {base_url}. Error: {e.reason}",
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama generation failed: {str(e)}",
        )
