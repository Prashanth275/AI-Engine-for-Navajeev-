"""
Recommendation Engine — Smart Guidance AI
Handles: /ai/recommend endpoint using Ollama Local LLM
"""

import json
import re
from prompts.module_prompts import get_prompt
from services.ollama_service import generate_with_ollama


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def run_recommendation(data: dict) -> dict:
    prompt = get_prompt("recommendation", data)
    cache_identifier = f"recommend_{json.dumps(data, sort_keys=True)}"
    raw_response = generate_with_ollama(prompt, cache_identifier)

    try:
        cleaned = _clean_json_text(raw_response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        return {
            "weekly_focus": raw_response,
            "daily_routine": ["Rest and stay hydrated"],
            "nutrition_tips": ["Eat balanced, nutrient-rich meals"],
            "self_care": ["Take a short restful break daily"]
        }
    except Exception:
        return {
            "weekly_focus": raw_response,
            "daily_routine": ["Rest and stay hydrated"],
            "nutrition_tips": ["Eat balanced, nutrient-rich meals"],
            "self_care": ["Take a short restful break daily"]
        }
