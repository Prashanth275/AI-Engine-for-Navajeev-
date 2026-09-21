import sys
import json
sys.path.insert(0, r"D:\apps\Preg-chatbot-ollama")

import ai_engine.insight_engine as insight_mod
import ai_engine.recommendation_engine as rec_mod

# Mock Ollama output for sleep module
mock_sleep_json = """```json
{
  "insight": "Baby slept 14 hours across 4 days with consistent patterns.",
  "trend": "stable",
  "who_comparison": "Meets WHO guidelines for 8-week-old infants.",
  "action": "Maintain regular bedtime routine.",
  "severity": "normal"
}
```"""

insight_mod.generate_with_ollama = lambda prompt, q=None: mock_sleep_json

sleep_res = insight_mod.run_insight("sleep", {"baby_age_weeks": 8, "sleep_logs": [{"date": "2026-09-01", "total_hours": 14}]})
print("Sleep Insight Result:", json.dumps(sleep_res, indent=2))
assert sleep_res["module"] == "sleep"
assert sleep_res["severity"] == "normal"
assert sleep_res["trend"] == "stable"

# Mock Ollama output for wellbeing module
mock_wellbeing_json = """```json
{
  "insight": "Mother reported steady mood with moderate energy levels.",
  "trend": "improving",
  "sleep_mood_correlation": "Higher energy observed following restful nights.",
  "coping_suggestion": "Practice 5 minutes of deep breathing before bed.",
  "severity": "normal",
  "show_helpline": false,
  "action": "Stay hydrated and take small rests."
}
```"""

insight_mod.generate_with_ollama = lambda prompt, q=None: mock_wellbeing_json

wellbeing_res = insight_mod.run_insight("wellbeing", {"baby_age_weeks": 8, "mood_logs": [{"date": "2026-09-01", "mood_score": 8}]})
print("Wellbeing Insight Result:", json.dumps(wellbeing_res, indent=2))
assert wellbeing_res["module"] == "wellbeing"
assert wellbeing_res["show_helpline"] is False

# Mock Ollama output for recommendation engine
mock_rec_json = """```json
{
  "daily_routine": ["Morning tummy time", "Midday nap", "Evening wind-down"],
  "nutrition_tips": ["Stay hydrated", "High-protein snack", "Iron-rich dinner"],
  "self_care": ["15-minute gentle walk", "Relaxing bath"],
  "dev_activity": "High-contrast visual card play",
  "weekly_focus": "Strengthening neck muscles during supervised tummy time"
}
```"""

rec_mod.generate_with_ollama = lambda prompt, q=None: mock_rec_json

rec_res = rec_mod.run_recommendation({"baby_age_weeks": 8, "mood_trend": "stable"})
print("Recommendation Result:", json.dumps(rec_res, indent=2))
assert "daily_routine" in rec_res
assert "weekly_focus" in rec_res

print("\n-------------------------------------------------------------")
print("SUCCESS: All module schemas and parsing verified successfully!")
print("-------------------------------------------------------------")
