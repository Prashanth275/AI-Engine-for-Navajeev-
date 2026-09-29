import json
import re
from prompts.module_prompts import get_prompt
from services.ollama_service import generate_with_ollama


def _parse_recommendation_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    # 1. Direct JSON parse
    try:
        res = json.loads(cleaned)
        if isinstance(res, dict) and "daily_routine" in res:
            return res
    except Exception:
        pass

    # 2. Outermost braces parse
    match_braces = re.search(r"\{[\s\S]*\}", cleaned)
    if match_braces:
        try:
            res = json.loads(match_braces.group(0))
            if isinstance(res, dict) and "daily_routine" in res:
                return res
        except Exception:
            pass

    # 3. Quick repair for trailing quote/comma anomalies (e.g. `\",]`)
    repaired = re.sub(r'\\?",\s*\]', '"]', cleaned)
    try:
        res = json.loads(repaired)
        if isinstance(res, dict) and "daily_routine" in res:
            return res
    except Exception:
        pass

    # 4. Regex extraction fallback for structured recommendation fields
    extracted = {}

    # Extract string fields
    for field in ["weekly_focus", "dev_activity"]:
        m = re.search(rf'"{field}"\s*:\s*"(.*?)(?<!\\)"', cleaned, re.DOTALL)
        if m:
            extracted[field] = m.group(1).replace(r'\"', '"').replace(r'\n', '\n').strip()

    # Extract list fields
    for field in ["daily_routine", "nutrition_tips", "self_care"]:
        m = re.search(rf'"{field}"\s*:\s*\[([\s\S]*?)\]', cleaned)
        if m:
            list_content = m.group(1)
            items = []
            for item_match in re.finditer(r'"(.*?)(?<!\\)"', list_content, re.DOTALL):
                val = item_match.group(1).replace(r'\"', '"').replace(r'\n', '\n').strip()
                if val and val != ",":
                    items.append(val)
            if items:
                extracted[field] = items

    if "daily_routine" in extracted or "weekly_focus" in extracted:
        extracted.setdefault("daily_routine", ["Rest and maintain a flexible daily rhythm."])
        extracted.setdefault("nutrition_tips", ["Stay well hydrated and eat balanced meals."])
        extracted.setdefault("self_care", ["Take a restorative break when possible."])
        extracted.setdefault("dev_activity", "Engage in gentle bonding and relaxation.")
        extracted.setdefault("weekly_focus", "Focus on gentle, stage-appropriate care and wellbeing this week.")
        return extracted

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
