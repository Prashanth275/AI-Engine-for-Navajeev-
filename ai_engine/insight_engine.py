"""
Insight Engine — Tracker AI
Handles: /ai/insight endpoint using Ollama Local LLM
Flow: user tracker data → prompt template → Ollama → structured JSON insight
"""

import json
import re
from fastapi import HTTPException
from prompts.module_prompts import get_prompt
from services.ollama_service import generate_with_ollama


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    # Strip markdown code blocks ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def run_insight(module: str, data: dict) -> dict:
    try:
        prompt = get_prompt(module, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cache_identifier = f"insight_{module}_{json.dumps(data, sort_keys=True)}"
    raw_response = generate_with_ollama(prompt, cache_identifier)

    # Dashboard returns plain text, everything else returns JSON
    if module == "dashboard":
        return {"insight": raw_response, "module": module}

    try:
        cleaned = _clean_json_text(raw_response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            result["module"] = module
            return result
        return {
            "module": module,
            "result": result,
            "insight": raw_response
        }
    except Exception:
        # Fallback to plain text wrapping if model didn't output strict JSON
        return {
            "module": module,
            "insight": raw_response,
            "trend": "stable",
            "action": "Review your recent logs and consult your healthcare provider if needed."
        }
