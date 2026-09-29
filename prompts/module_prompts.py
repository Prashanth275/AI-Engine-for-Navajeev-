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
    sleep_pattern = data.get("sleep_pattern")
    feeding_pattern = data.get("feeding_pattern")
    mood_trend = data.get("mood_trend")
    top_concern = data.get("top_concern")

    is_pregnancy = pregnancy_week is not None and baby_age is None
    is_postpartum = baby_age is not None

    profile_items = []
    if is_pregnancy:
        profile_items.append(f"- Stage: Pregnancy (Week {pregnancy_week})")
    elif is_postpartum:
        profile_items.append(f"- Stage: Postpartum (Baby is {baby_age} weeks old)")
    else:
        profile_items.append("- Stage: General maternal / infant care")

    if sleep_pattern and str(sleep_pattern).strip().lower() not in ["none", "null", "not provided"]:
        profile_items.append(f"- Sleep pattern: {sleep_pattern}")
    else:
        profile_items.append("- Sleep pattern: Not provided (do not assume or invent sleep metrics or physical symptoms)")

    if is_postpartum:
        if feeding_pattern and str(feeding_pattern).strip().lower() not in ["none", "null", "not provided"]:
            profile_items.append(f"- Infant feeding pattern: {feeding_pattern}")
        else:
            profile_items.append("- Infant feeding pattern: Not provided (do not assume breastfeeding or formula)")
    elif not is_pregnancy and feeding_pattern:
        profile_items.append(f"- Feeding pattern: {feeding_pattern}")

    if mood_trend and str(mood_trend).strip().lower() not in ["none", "null", "not provided"]:
        profile_items.append(f"- Mother mood trend: {mood_trend}")
    else:
        profile_items.append("- Mother mood trend: Not provided")

    if top_concern and str(top_concern).strip().lower() not in ["none", "null", "not provided", "general guidance"]:
        profile_items.append(f"- Top concern: {top_concern}")
    else:
        profile_items.append("- Top concern: None specified")

    profile_text = "\n".join(profile_items)

    if is_pregnancy:
        stage_rules = f"""STAGE DIRECTIVE (PREGNANCY — WEEK {pregnancy_week}):
- The mother is currently PREGNANT at Week {pregnancy_week}. The baby is not yet born.
- STRICTLY FORBIDDEN: Do NOT mention any infant or newborn care activities (NO tummy time, NO diapering, NO bottle/breastfeeding routines, NO stroller walks with baby, NO infant play, NO newborn sleep training).
- "dev_activity" MUST be ONE gentle prenatal bonding or maternal relaxation activity for the pregnant mother (e.g. resting hands on belly while practicing calm, slow breathing, or listening to soothing music). Do NOT assume or describe observable fetal movements or reactions.
- Avoid generic filler phrases (e.g. 'support your growing bump'). Focus directly on the practical wellness action (e.g. 'Start your day with a quiet moment of breathing or reflection').
- Conservative Movement & Rest: Do NOT prescribe specific exercises, unusual movement instructions, awkward body positions (e.g. AVOID 'flick your feet', 'seated reclining with pillow under belly', specific yoga poses), or imply positions are curative. Use simple conservative phrasing (e.g. 'Find a comfortable, supported resting position', 'Try gentle movement if it feels comfortable for you', 'Use pillows for comfortable support')."""
    else:
        stage_rules = f"""STAGE DIRECTIVE (POSTPARTUM — BABY AGE: {baby_age} WEEKS):
- The user is POSTPARTUM with an infant who is {baby_age} weeks old.
- Balance infant care with maternal recovery and rest.
- "dev_activity" MUST be exactly ONE age-appropriate, supervised awake bonding or developmental activity specifically suitable for a {baby_age}-week-old infant (e.g. supervised awake tummy time on a firm, flat mat, high-contrast visual tracking, or gentle responsive talking). Always specify tummy time as supervised and awake.
- Do NOT turn daily_routine into infant developmental exercises. daily_routine is for maternal wellness, daily rhythm, and responsive caregiving habits.
- Decouple maternal routines from feeding events: Do NOT anchor maternal self-care, journaling, or snacks to infant feeding events (e.g. AVOID 'As you wait for a bottle...', 'before or after feeds'). Frame maternal actions independently (e.g. 'When you have a quiet moment...').
- If infant feeding pattern is not specified, do NOT assume breastfeeding vs. formula."""

    concern_instruction = f"""TOP CONCERN GUIDANCE:
- Address the user's top concern ({top_concern}) thoughtfully and faithfully as stated.
- Keep user concerns distinct from feeding data: If the top concern is general (e.g. 'establishing a consistent daily rhythm' or 'sleep quality'), do NOT automatically conflate or rewrite it into an infant feeding concern (such as 'baby feeding rhythm') simply because feeding data is present.""" if (top_concern and str(top_concern).strip().lower() not in ["none", "null", "not provided", "general guidance"]) else """TOP CONCERN GUIDANCE:
- No specific concern provided. Provide balanced, encouraging stage-appropriate guidance without fabricating problems."""

    return f"""You are a compassionate maternal and infant health AI assistant.
Generate grounded, supportive weekly guidance based strictly on the provided profile.

User Profile:
{profile_text}

{stage_rules}

{concern_instruction}

STRICT QUALITY & SAFETY CONSTRAINTS (MANDATORY):
1. FLEXIBLE DAILY ROUTINE (NO FIXED DAYPART SEQUENCING):
   - In daily_routine, DO NOT use rigid daypart prefixes or fixed schedule labels (e.g. AVOID 'Morning: ...', 'Afternoon: ...', 'Evening: ...', 'At breakfast...', 'At lunch...', 'At bedtime...').
   - Use flexible, adaptable language instead (e.g. 'When you have a quiet moment...', 'During a calm part of the day...', 'When you have an opportunity...', 'As you wind down...', 'Whenever it feels comfortable...').
2. NO HERBAL DRINKS OR REMEDIES:
   - NEVER spontaneously recommend herbal tea, herbal water, herbal remedies, or unverified plant-based infusions.
   - For hydration, recommend plain water, fluids according to thirst, or ordinary balanced food-based hydration. Do not invent medical benefits from beverages.
3. NO INFERRED PHYSICAL SYMPTOMS OR EMOTIONAL CONDITIONS:
   - When sleep data or mood is logged, do NOT invent physical symptoms (such as fatigue, pain, dizziness, weakness, exhaustion) unless explicitly logged. (Use 'When you need a moment to reset...' instead of 'When you notice faint fatigue...').
   - Do NOT label unsupplied emotional states (such as isolation, loneliness, anxiety, depression) unless explicitly logged. (Use 'Invite a trusted friend or family member to talk or help with a small task' instead of 'easing feelings of isolation').
   - Do NOT diagnose any medical or mental health conditions.
4. NO INVENTED QUANTITIES OR DURATIONS:
   - DO NOT invent numerical durations (e.g. "5 minutes", "20 minutes", "15-minute walks").
   - DO NOT invent numerical quantities, frequencies, or measurements (e.g. "8 cups of water", "2 liters", "3 times a day", "30g protein").
   - Use qualitative, self-paced phrasing (e.g. "a brief pause", "a comfortable period", "hydrate regularly according to thirst", "include balanced nourishment across your meals", "rest as needed").
5. NO UNSUPPORTED PHYSIOLOGICAL OR BENEFIT CLAIMS:
   - DO NOT make unsupported physiological or benefit claims (e.g. claiming foods or routines 'support joint comfort', 'reduce inflammation', 'support circulation', 'maintain milk supply', 'boost breastmilk production', 'support fetal development', or 'prevent fatigue') unless directly justified by supplied context.
   - Use neutral, practical wording (e.g. 'Choose a varied combination of foods that provides steady nourishment and energy', 'Stay well hydrated with water whenever you feel thirsty').
6. NO ASSUMED FETAL RESPONSES & NO GENERIC PREGNANCY FILLER:
   - In pregnancy, DO NOT state that the baby kicks or moves in response to maternal actions. Keep bonding grounded in the mother's own mindful connection and calming rest.
   - Avoid generic filler phrases like 'support your growing bump'. Focus directly on the practical wellness action.
7. SPARSE DATA DISCIPLINE:
   - If data is missing or not provided, do NOT fabricate symptoms, routines, or metrics. Provide simple, supportive, stage-appropriate baseline guidance.
8. SECTION PURPOSES & VARIETY:
   - weekly_focus: Exactly 1 clear, encouraging overarching focus sentence for this week.
   - daily_routine: Exactly 3 practical, flexible habits for maternal wellness, daily rhythm, and responsive care. Avoid rigid timetables, daypart labels, arbitrary durations, or multiple infant developmental exercises.
   - nutrition_tips: Exactly 3 realistic, balanced food and hydration suggestions (water/regular foods, qualitative, neutral benefits, NO herbal teas or remedies).
   - self_care: Exactly 2 restorative maternal suggestions.
   - dev_activity: Exactly 1 stage-appropriate activity (prenatal bonding/relaxation for pregnancy OR supervised awake activity for infant).
   - Do NOT repeat identical advice across sections.
9. FORMATTING:
   - Return strict, valid JSON only. Use single quotes or simple words inside text values (no unescaped double quotes).

Return ONLY a valid JSON object matching these exact keys and structure:
{{
  "weekly_focus": "One clear, encouraging overarching focus sentence for this week",
  "daily_routine": [
    "Practical, flexible daily habit 1",
    "Practical, flexible daily habit 2",
    "Practical, flexible daily habit 3"
  ],
  "nutrition_tips": [
    "Practical nutrition or hydration tip 1",
    "Practical nutrition or hydration tip 2",
    "Practical nutrition or hydration tip 3"
  ],
  "self_care": [
    "Restorative maternal self-care suggestion 1",
    "Restorative maternal self-care suggestion 2"
  ],
  "dev_activity": "One focused stage-appropriate activity"
}}

Return ONLY valid JSON. No markdown fences. No extra commentary."""

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
