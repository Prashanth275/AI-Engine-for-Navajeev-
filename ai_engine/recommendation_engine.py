"""
Recommendation Engine — Smart Guidance AI
Handles: /ai/recommend endpoint using Ollama Local LLM
"""

import json
import re
from prompts.module_prompts import get_prompt
from services.ollama_service import generate_with_ollama


def _parse_recommendation_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        res = json.loads(cleaned)
        if isinstance(res, dict):
            return res
    except Exception:
        pass

    match_braces = re.search(r"\{[\s\S]*\}", cleaned)
    if match_braces:
        try:
            res = json.loads(match_braces.group(0))
            if isinstance(res, dict):
                return res
        except Exception:
            pass

    start_brace = cleaned.find("{")
    if start_brace != -1:
        snippet = cleaned[start_brace:].strip()
        repairs = [snippet + "}", snippet + '"}', snippet + '"\n}']
        for r in repairs:
            try:
                res = json.loads(r)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass

    return {}


def run_recommendation(data: dict) -> dict:
    prompt = get_prompt("recommendation", data)
    cache_identifier = f"recommend_{json.dumps(data, sort_keys=True)}"
    raw_response = generate_with_ollama(prompt, cache_identifier)

    parsed = _parse_recommendation_json(raw_response)
    if parsed and isinstance(parsed, dict):
        return parsed

    return {
        "weekly_focus": raw_response,
        "daily_routine": ["Rest and stay hydrated"],
        "nutrition_tips": ["Eat balanced, nutrient-rich meals"],
        "self_care": ["Take a short restful break daily"]
    }
