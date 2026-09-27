from pydantic import BaseModel, Field
from typing import Optional, Any


class QuestionRequest(BaseModel):
    question: str
    include_context: bool = True


class QuestionResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    success: bool = True


class InsightRequest(BaseModel):
    module: str = Field(..., description="sleep | feeding | growth | trimester | wellbeing | appointments | notifications | dashboard")
    subject: Optional[str] = None
    baby_age_weeks: Optional[int] = None
    data: dict = Field(..., description="Module-specific tracker data")


class InsightResponse(BaseModel):
    module: str
    success: bool = True
    result: Any


class RecommendRequest(BaseModel):
    baby_age_weeks: Optional[int] = None
    sleep_pattern: Optional[str] = None
    feeding_pattern: Optional[str] = None
    mood_trend: Optional[str] = None
    pregnancy_week: Optional[int] = None
    top_concern: Optional[str] = None


class RecommendResponse(BaseModel):
    success: bool = True
    result: Any


class HealthResponse(BaseModel):
    status: str
    message: str
