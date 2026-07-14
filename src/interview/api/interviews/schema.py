from datetime import datetime

from pydantic import BaseModel, Field

from interview.schemas.report import FinalReport


class InterviewResultSummary(BaseModel):
    result_id: int
    session_id: int
    overall_score: float
    created_at: datetime


class InterviewResultResponse(BaseModel):
    result_id: int
    session_id: int
    overall_score: float
    topic_scores: dict[str, float]
    report: FinalReport
    created_at: datetime

class InterviewSessionResponse(BaseModel):
    session_id: int
    mode: str
    status: str
    started_at: datetime
    
class WeakTopicsResponse(BaseModel):
    topics: list[str] = Field(default_factory=list)