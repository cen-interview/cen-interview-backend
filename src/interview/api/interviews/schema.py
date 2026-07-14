"""면접 세션과 최종 결과 API에서 사용하는 Pydantic 응답 스키마.

이 모듈은 면접 결과 목록, 상세 리포트, 면접 세션 상태와 약점 주제를
클라이언트에 전달하기 위한 API 응답 형식을 정의한다.

주요 스키마:
    InterviewResultSummary:
        전체 면접 이력 목록에서 사용하는 요약 정보.

    InterviewResultResponse:
        특정 면접의 FinalReport를 포함하는 상세 결과.

    InterviewSessionResponse:
        면접 세션의 진행 방식과 현재 상태 정보.

    WeakTopicsResponse:
        다음 면접 개인화에 사용할 약점 주제 목록.

사용 API:
    GET /api/interview-results:
        list[InterviewResultSummary] 반환.

    GET /api/interview-results/latest:
        InterviewResultResponse 반환.

    GET /api/interview-results/{result_id}:
        InterviewResultResponse 반환.

    GET /api/interview-results/weak-topics:
        WeakTopicsResponse 반환.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from interview.schemas.report import FinalReport

# 전체 면접 결과 목록에 사용하는 요약 응답.
class InterviewResultSummary(BaseModel):
    result_id: int
    session_id: int
    overall_score: float
    created_at: datetime

# 특정 면접의 최종 리포트를 포함하는 상세 응답.
class InterviewResultResponse(BaseModel):
    result_id: int
    session_id: int
    overall_score: float
    topic_scores: dict[str, float]
    report: FinalReport
    created_at: datetime

# 면접 세션의 기본 정보와 진행 상태 응답.
class InterviewSessionResponse(BaseModel):
    session_id: int
    mode: str
    status: str
    started_at: datetime

# 최근 면접에서 점수가 낮았던 주제 목록 응답.   
class WeakTopicsResponse(BaseModel):
    topics: list[str] = Field(default_factory=list)