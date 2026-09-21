"""
Prompt Templates — one per module.
Each returns a plain-text prompt string filled with the user's data.
"""
from datetime import date

def dashboard_prompt(data: dict) -> str:
    return f"""You are a compassionate maternal health AI assistant.

Based on the summary below, write a warm personalized daily brief.
Include: a 2-sentence summary of how baby and mother are doing,
the top 2 action items for today, and one encouraging note.

Baby age: {data.get('baby_age_weeks')} weeks
Last night sleep: {data.get('sleep_hours')} hours ({data.get('sleep_wakings')} wakings)
Feeds in last 24hrs: {data.get('feed_count')}
Mother mood score: {data.get('mood_score')}/10
Symptoms logged: {data.get('symptoms', 'None')}

Write in warm, friendly, conversational tone. 5 to 7 sentences. No bullet points."""



def sleep_prompt(data: dict) -> str:
    logs_list = data.get('sleep_logs', [])
    
    logs = "\n".join([
        f"- {log['date']}: {log['total_hours']} hrs, {log.get('night_wakings', 0)} wakings"
        for log in logs_list
    ])

    subject = data.get('subject', 'baby')
    context_note = data.get('context', '')
    num_days = len(logs_list)
    baby_age = data.get('baby_age_weeks')

    if subject == 'mother':
        analysis_target = "a postpartum or pregnant mother"
        norms_reference = "healthy adult sleep norms (7-9 hours per night)"
        who_label = "adult sleep norms"
    else:
        age_str = f"{baby_age}-week-old baby" if baby_age else "infant"
        analysis_target = f"a {age_str}"
        norms_reference = f"WHO recommended sleep norms for a {age_str}"
        who_label = "WHO norms for this age"

    return f"""
You are a STRICT maternal health AI assistant specializing in sleep analysis.

CRITICAL RULES (DO NOT BREAK):
- The dataset contains EXACTLY {num_days} days
- You MUST ONLY refer to these {num_days} days
- NEVER say 7 days, 10 days, weekly pattern, or anything else
- If {num_days} = 4 → you MUST say "4 days"
- Do NOT assume missing days
- Do NOT generalize beyond given logs
- Do NOT generate, assume, or modify any dates
- Do NOT mention today's date

{context_note}

Analyze sleep for {analysis_target}.

Sleep logs (ONLY {num_days} days):
{logs}

Compare against {norms_reference}.

Do NOT assume or infer data for days not listed.
Only analyse what is actually logged. Do not penalise for missing days.


Return a JSON object with these exact keys:
{{
  "insight": "must clearly refer to EXACTLY {num_days} days only",
  "trend": "improving or declining or stable or insufficient_data",
  "who_comparison": "comparison to {who_label}",
  "action": "one specific recommendation",
  "severity": "normal or watch or consult_doctor"
}}

STRICT:
- If {num_days} <= 2 → trend = "insufficient_data"
- DO NOT mention any number other than {num_days}
- DO NOT invent extra days

Return ONLY JSON.
"""


def feeding_prompt(data: dict) -> str:
    logs = "\n".join([
        f"- {log['date']}: {log['feed_count']} feeds, avg gap {log.get('avg_gap_hours', '?')} hrs, type: {log.get('type', 'breast')}"
        for log in data.get('feeding_logs', [])
    ])
    return f"""You are a maternal health AI assistant specializing in infant nutrition.

Analyze the feeding data below for a {data.get('baby_age_weeks')}-week-old baby.

Feeding logs (last 7 days):
{logs}
Feeding type: {data.get('feeding_type', 'breastfeeding')}
Mother concern: {data.get('concern', 'None')}

Return a JSON object with these exact keys:
{{
  "insight": "feeding frequency analysis in 2 sentences",
  "frequency_status": "adequate or low or high",
  "action": "one specific recommendation",
  "tip": "one practical tip based on feeding type",
  "severity": "normal or watch or consult_doctor"
}}

Return only the JSON. No extra text."""


def growth_prompt(data: dict) -> str:
    measurements = "\n".join([
        f"- {m['date']}: weight {m.get('weight_kg', '?')}kg, height {m.get('height_cm', '?')}cm, head {m.get('head_cm', '?')}cm"
        for m in data.get('measurements', [])
    ])
    return f"""You are a maternal health AI assistant specializing in infant growth.

Analyze the growth measurements below for a {data.get('baby_age_weeks')}-week-old
{data.get('gender', 'baby')} against WHO Child Growth Standards.

Measurements:
{measurements}

Return a JSON object with these exact keys:
{{
  "insight": "interpretation of measurements in 2 sentences",
  "weight_status": "brief weight assessment",
  "height_status": "brief height assessment",
  "trend": "on_track or monitor or consult_doctor",
  "milestone_prediction": "developmental milestone prediction",
  "action": "one specific recommendation",
  "severity": "normal or watch or consult_doctor"
}}

Return only the JSON. No extra text."""


def trimester_prompt(data: dict) -> str:
    symptoms = ", ".join(data.get('symptoms', [])) or "None"
    return f"""You are a compassionate maternal health AI assistant for pregnancy guidance.

Mother is at week {data.get('pregnancy_week')} of pregnancy.
Symptoms logged this week: {symptoms}
Concern: {data.get('concern', 'None')}

Return a JSON object with these exact keys:
{{
  "week_summary": "what is happening with baby and mother body at week {data.get('pregnancy_week')} in 2-3 sentences",
  "symptom_assessment": [
    {{"symptom": "symptom name", "status": "normal or watch or urgent", "note": "brief note"}}
  ],
  "action_items": ["item 1", "item 2", "item 3"],
  "action": "most important single action",
  "severity": "normal or watch or consult_doctor"
}}

Return only the JSON. No extra text."""


def wellbeing_prompt(data: dict) -> str:
    mood_logs = "\n".join([
        f"- {log['date']}: mood {log['mood_score']}/5, sleep {log.get('sleep_hours', '?')} hrs, note: {log.get('note', 'none')}"
        for log in data.get('mood_logs', [])
    ])
    return f"""You are a compassionate maternal mental health AI assistant.

Analyze the mood and wellbeing data below for a postpartum mother
whose baby is {data.get('baby_age_weeks')} weeks old.

Mood logs (last 14 days):
{mood_logs}
Days postpartum: {data.get('days_postpartum', 'unknown')}

Return a JSON object with these exact keys:
{{
  "insight": "mood trend analysis in 2 sentences",
  "trend": "improving or declining or stable or fluctuating",
  "sleep_mood_correlation": "brief correlation observation",
  "coping_suggestion": "one compassionate practical suggestion",
  "severity": "normal or watch or seek_support",
  "show_helpline": true or false,
  "action": "one recommended action"
}}

Set show_helpline to true only if mood score is consistently below 2/5.
Return only the JSON. No extra text."""


def appointments_prompt(data: dict) -> str:
    recent = data.get('recent_summary', {})
    return f"""You are a maternal health AI assistant helping prepare for a medical visit.

Appointment: {data.get('appointment_type', 'pediatrician visit')} on {data.get('appointment_date')}
Baby age: {data.get('baby_age_weeks')} weeks

Recent tracker summary:
- Sleep: {recent.get('sleep_summary', 'not provided')}
- Feeding: {recent.get('feeding_summary', 'not provided')}
- Growth: {recent.get('growth_summary', 'not provided')}
- Mood: {recent.get('mood_summary', 'not provided')}
- Symptoms: {recent.get('symptoms', 'none')}

Return a JSON object with these exact keys:
{{
  "insight": "one sentence summary of what needs attention",
  "questions": ["question 1", "question 2", "question 3", "question 4", "question 5"],
  "bring": ["item 1", "item 2"],
  "urgent_items": ["urgent item if any"]
}}

Return only the JSON. No extra text."""


def notifications_prompt(data: dict) -> str:
    return f"""You are a maternal health AI assistant generating smart alerts.

Current data:
- Baby age: {data.get('baby_age_weeks')} weeks
- Hours since last feed: {data.get('hours_since_last_feed')}
- Last sleep duration: {data.get('last_sleep_hours')} hours
- Mood score today: {data.get('mood_score')}/10
- Missed vaccine: {data.get('missed_vaccine', 'none')}
- Next appointment in days: {data.get('next_appointment_days')}
- Weight trend: {data.get('weight_trend', 'stable')}
- Symptoms today: {data.get('symptoms_today', 'none')}

Generate only genuinely relevant alerts. Skip if data is normal.

Return a JSON object with this exact key:
{{
  "alerts": [
    {{
      "title": "short alert title",
      "message": "helpful alert message",
      "severity": "critical or moderate or info",
      "module": "sleep or feeding or growth or wellbeing or appointments"
    }}
  ]
}}

Return only the JSON. No extra text."""


def recommendation_prompt(data: dict) -> str:
    return f"""You are a maternal health AI assistant generating a personalized weekly plan.

User profile:
- Baby age: {data.get('baby_age_weeks')} weeks
- Sleep pattern: {data.get('sleep_pattern', 'not provided')}
- Feeding pattern: {data.get('feeding_pattern', 'not provided')}
- Mother mood trend: {data.get('mood_trend', 'stable')}
- Pregnancy week (if applicable): {data.get('pregnancy_week', 'N/A')}
- Top concern: {data.get('top_concern', 'general guidance')}

Return a JSON object with these exact keys:
{{
  "daily_routine": ["routine step 1", "routine step 2", "routine step 3"],
  "nutrition_tips": ["tip 1", "tip 2", "tip 3"],
  "self_care": ["suggestion 1", "suggestion 2"],
  "dev_activity": "one developmental activity to try this week",
  "weekly_focus": "the main focus for this week in one sentence"
}}

Return only the JSON. No extra text."""


# -------------------------------------------------------
# ROUTER — maps module name to its prompt function
# -------------------------------------------------------

PROMPT_REGISTRY = {
    "dashboard": dashboard_prompt,
    "sleep": sleep_prompt,
    "feeding": feeding_prompt,
    "growth": growth_prompt,
    "trimester": trimester_prompt,
    "wellbeing": wellbeing_prompt,
    "appointments": appointments_prompt,
    "notifications": notifications_prompt,
    "recommendation": recommendation_prompt,
}


def get_prompt(module: str, data: dict) -> str:
    fn = PROMPT_REGISTRY.get(module)
    if not fn:
        raise ValueError(
            f"Unknown module: '{module}'. "
            f"Valid options: {list(PROMPT_REGISTRY.keys())}"
        )
    return fn(data)
