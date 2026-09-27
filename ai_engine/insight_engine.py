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


def _parse_llm_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    
    # 1. Strip markdown code blocks ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    # 2. Try direct json.loads
    try:
        res = json.loads(cleaned)
        if isinstance(res, dict):
            return res
    except Exception:
        pass

    # 3. Try finding outermost { ... }
    match_braces = re.search(r"\{[\s\S]*\}", cleaned)
    if match_braces:
        try:
            res = json.loads(match_braces.group(0))
            if isinstance(res, dict):
                return res
        except Exception:
            pass

    # 4. Try repairing unclosed JSON (e.g. truncated closing brace/quote)
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

    # 5. Regex field extraction fallback
    extracted = {}
    for k, v in re.findall(r'"([a-zA-Z0-9_]+)"\s*:\s*"(.*?)(?<!\\)"', cleaned, re.DOTALL):
        extracted[k] = v.replace(r'\"', '"').replace(r'\n', '\n').replace(r'\r', '')
    for k, v in re.findall(r'"([a-zA-Z0-9_]+)"\s*:\s*(true|false)', cleaned, re.IGNORECASE):
        extracted[k] = (v.lower() == 'true')

    if "insight" in extracted:
        return extracted

    return {}


def _contains_prompt_leak(text: str) -> bool:
    if not text or not isinstance(text, str):
        return True
    lowered = text.lower()
    leak_tokens = [
        "must clearly",
        "critical rule",
        "dataset contain",
        "instruction",
        "prompt",
        "strictly",
        "exactly",
        "do not",
        "only refer to",
        "placeholder",
        "return only json",
        "keys:",
        "rules:",
    ]
    return any(token in lowered for token in leak_tokens)


def calculate_sleep_metrics(data: dict) -> dict:
    logs_list = data.get("sleep_logs", [])
    subject = data.get("subject", "mother")
    baby_age_weeks = data.get("baby_age_weeks")

    if not logs_list:
        return {
            "num_days": 0,
            "avg_hours": 0.0,
            "min_hours": 0.0,
            "max_hours": 0.0,
            "avg_wakings": 0.0,
            "trend": "insufficient_data",
            "who_comparison": "No sleep logs available for comparison.",
            "severity": "watch",
            "subject": subject,
            "target_reference": "7-9 hours per night" if subject == "mother" else "age-appropriate rest",
        }

    # Sort chronologically by date
    sorted_logs = sorted(logs_list, key=lambda x: str(x.get("date", "")))
    num_days = len(sorted_logs)

    hours_list = [float(log.get("total_hours", 0.0)) for log in sorted_logs]
    wakings_list = [float(log.get("night_wakings", 0)) for log in sorted_logs]

    avg_hours = round(sum(hours_list) / max(num_days, 1), 1)
    min_hours = round(min(hours_list), 1)
    max_hours = round(max(hours_list), 1)
    avg_wakings = round(sum(wakings_list) / max(num_days, 1), 1)

    # Deterministic mathematical trend calculation
    if num_days < 3:
        trend = "insufficient_data"
    else:
        mid = num_days // 2
        earlier_avg = sum(hours_list[:mid]) / mid
        later_avg = sum(hours_list[mid:]) / (num_days - mid)
        diff = later_avg - earlier_avg
        if diff >= 0.5:
            trend = "improving"
        elif diff <= -0.5:
            trend = "declining"
        else:
            trend = "stable"

    # Deterministic reference comparison and severity
    if subject == "mother":
        target_reference = "recommended adult sleep guidelines (7-9 hrs/night)"
        if avg_hours >= 7.0:
            who_comparison = f"Averaging {avg_hours} hrs/night, meeting standard adult rest guidelines (7-9 hrs)."
            severity = "normal"
        elif avg_hours >= 5.0:
            who_comparison = f"Averaging {avg_hours} hrs/night, below the recommended 7-9 hrs adult rest target."
            severity = "watch"
        else:
            who_comparison = f"Averaging {avg_hours} hrs/night, indicating significant postpartum sleep deficit."
            severity = "consult_doctor"
    else:
        if baby_age_weeks:
            target_reference = f"recommended sleep guidelines for a {baby_age_weeks}-week-old baby"
            who_comparison = f"Averaging {avg_hours} hrs/day across {num_days} logged days for a {baby_age_weeks}-week-old infant."
        else:
            target_reference = "recommended infant sleep guidelines"
            who_comparison = f"Averaging {avg_hours} hrs/day across {num_days} logged days."

        if avg_hours >= 11.0:
            severity = "normal"
        elif avg_hours >= 8.5:
            severity = "watch"
        else:
            severity = "consult_doctor"

    return {
        "num_days": num_days,
        "avg_hours": avg_hours,
        "min_hours": min_hours,
        "max_hours": max_hours,
        "avg_wakings": avg_wakings,
        "trend": trend,
        "who_comparison": who_comparison,
        "severity": severity,
        "subject": subject,
        "target_reference": target_reference,
    }


def run_insight(module: str, data: dict) -> dict:
    if module == "sleep":
        metrics = calculate_sleep_metrics(data)
        data["metrics"] = metrics

        try:
            prompt = get_prompt(module, data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        cache_identifier = f"insight_sleep_v2_{json.dumps(data, sort_keys=True)}"
        raw_response = generate_with_ollama(prompt, cache_identifier)

        parsed = _parse_llm_json(raw_response)
        raw_insight = parsed.get("insight") if isinstance(parsed, dict) else None
        raw_action = parsed.get("action") if isinstance(parsed, dict) else None

        # Validate insight and action for prompt leaks
        if raw_insight and isinstance(raw_insight, str) and not _contains_prompt_leak(raw_insight):
            insight = raw_insight.strip()
        else:
            insight = f"Over the logged {metrics['num_days']} days, sleep averaged {metrics['avg_hours']} hours per day with a {metrics['trend']} pattern."

        if raw_action and isinstance(raw_action, str) and not _contains_prompt_leak(raw_action):
            action = raw_action.strip()
        else:
            action = "Prioritize rest whenever possible and maintain a consistent, relaxing bedtime routine."

        return {
            "module": "sleep",
            "insight": insight,
            "trend": metrics["trend"],
            "who_comparison": metrics["who_comparison"],
            "action": action,
            "severity": metrics["severity"],
        }

    try:
        prompt = get_prompt(module, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cache_identifier = f"insight_{module}_{json.dumps(data, sort_keys=True)}"
    raw_response = generate_with_ollama(prompt, cache_identifier)

    # Dashboard returns plain text, everything else returns JSON
    if module == "dashboard":
        return {"insight": raw_response, "module": module}

    parsed = _parse_llm_json(raw_response)
    if parsed:
        parsed["module"] = module
        return parsed

    # Fallback to plain text wrapping if model output had no parseable JSON
    return {
        "module": module,
        "insight": raw_response,
        "trend": "stable",
        "action": "Review your recent logs and consult your healthcare provider if needed."
    }
