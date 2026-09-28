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
    metrics = data.get("metrics")
    if not metrics:
        logs_list = data.get("sleep_logs", [])
        num_days = len(logs_list)
        hours = [float(l.get("total_hours", 0.0)) for l in logs_list] if logs_list else [0.0]
        avg_h = round(sum(hours) / max(num_days, 1), 1)
        subject = data.get("subject", "mother")
        metrics = {
            "num_days": num_days,
            "avg_hours": avg_h,
            "min_hours": min(hours),
            "max_hours": max(hours),
            "avg_wakings": 0.0,
            "trend": "stable" if num_days >= 3 else "insufficient_data",
            "target_reference": "7-9 hours per night" if subject == "mother" else "age-appropriate rest",
            "subject": subject,
        }

    subject_str = "postpartum/new mother" if metrics.get("subject") == "mother" else "infant"
    wakings_note = f", with an average of {metrics['avg_wakings']} night wakings" if metrics.get("avg_wakings", 0) > 0 else ""

    return f"""You are a compassionate maternal and infant health AI assistant.

Write a personalized, encouraging sleep summary for a {subject_str}.

Calculated Facts:
- Logged period: {metrics['num_days']} days
- Average sleep: {metrics['avg_hours']} hours per day (range: {metrics['min_hours']}h to {metrics['max_hours']}h){wakings_note}
- Overall pattern: {metrics['trend']}
- Target reference: {metrics['target_reference']}

Guidelines:
- Write in a supportive, empathetic tone.
- Do NOT repeat instructions, rules, or internal placeholders.
- Do NOT invent future durations or claim arbitrary numbers of days.
- "insight": 2 natural, user-facing sentences reflecting on the {metrics['num_days']}-day average of {metrics['avg_hours']} hours and {metrics['trend']} pattern.
- "action": 1 practical, gentle recommendation to support rest.

Return ONLY a valid JSON object matching this structure:
{{
  "insight": "Empathetic 2-sentence summary of sleep observations",
  "action": "Practical suggestion to support healthy rest"
}}"""


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
    baby_age = data.get("baby_age_weeks")
    pregnancy_week = data.get("pregnancy_week")
    sleep_pattern = data.get("sleep_pattern", "not provided")
    feeding_pattern = data.get("feeding_pattern", "not provided")
    mood_trend = data.get("mood_trend", "stable")
    top_concern = data.get("top_concern", "general guidance")

    stage_desc = []
    if baby_age is not None:
        stage_desc.append(f"Baby age: {baby_age} weeks")
    if pregnancy_week is not None:
        stage_desc.append(f"Pregnancy week: {pregnancy_week}")
    stage_str = ", ".join(stage_desc) if stage_desc else "Maternal / infant care stage"

    return f"""You are a compassionate maternal and infant health AI assistant.
Generate personalized, supportive weekly guidance based strictly on the provided profile.

User Profile:
- Stage: {stage_str}
- Sleep pattern: {sleep_pattern}
- Feeding pattern: {feeding_pattern}
- Mother mood trend: {mood_trend}
- Top concern: {top_concern}

Strict Grounding & Quality Rules:
- Use ONLY information provided in the user profile above.
- Do NOT invent facts, symptoms, diagnoses, or clinical conditions about the mother or baby.
- Do NOT invent exact clock times (e.g., do NOT mention times like "7:00 AM", "5:00 PM").
- Do NOT create a rigid feeding or sleep timetable; keep suggestions flexible, cue-based, and responsive.
- Do NOT invent arbitrary durations, quantities, or strict numerical measurements unless specified in user data.
- Do NOT present generated recommendations as if they are past logged user data.
- Ensure "dev_activity" is strictly appropriate for the baby's stated age in weeks (or pregnancy stage).
- Ensure "top_concern" ({top_concern}) directly and meaningfully influences "daily_routine", "dev_activity", and "weekly_focus".
- If a profile field is not provided, do not pretend that information is known.
- For "nutrition_tips", provide general, supportive dietary suggestions only; do not invent medical diets or clinical requirements.
- Avoid all medical diagnosis or treatment claims.

Return ONLY a valid JSON object matching these exact keys and structure:
{{
  "daily_routine": [
    "Practical, flexible routine habit 1 addressing {top_concern}",
    "Practical, flexible routine habit 2",
    "Practical, flexible routine habit 3"
  ],
  "nutrition_tips": [
    "Supportive nutrition tip 1",
    "Supportive nutrition tip 2",
    "Supportive nutrition tip 3"
  ],
  "self_care": [
    "Realistic, brief maternal self-care suggestion 1",
    "Realistic, brief maternal self-care suggestion 2"
  ],
  "dev_activity": "One age-appropriate bonding or developmental activity tailored to baby age and {top_concern}",
  "weekly_focus": "One clear, encouraging overarching focus sentence for this week"
}}

Return ONLY valid JSON. No markdown. No explanation outside the JSON."""

# ROUTER — maps module name to its prompt function

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
