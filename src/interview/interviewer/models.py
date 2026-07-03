# interviewer/models.py (신규)
from pydantic import BaseModel
from schemas.events import InterviewerEvent  # 실제 import 경로는 프로젝트 기준으로

class DeliveryMetrics(BaseModel):
    speech_rate_wpm: float | None = None
    filler_count: int | None = None
    duration_seconds: float | None = None

class AdaptedInput(BaseModel):
    # 무슨 일이 일어났나
    event: InterviewerEvent
    # 어떻게 말했나
    delivery_metrics: DeliveryMetrics | None = None