"""
AnswerEvaluation / CompetencyModel / FinalReport

Assessment는 메인 질문과 후속 질문을 하나의 질문 세트로 묶어 평가한다.
면접 중 AnswerEvaluation을 누적하고, 종료 시 FinalReport를 생성한다.


QualityTrace
- 어디에 둠: AnswerEvaluation 안
- 범위: 질문 세트 1개
- 목적: 오개념, 가산점, 확인 질문, 정정 흐름 기록
- 예: main에서 misconception → challenge에서 sufficient

CompetencyModel
- 어디에 둠: AssessmentAgent.competency
- 범위: 면접 전체
- 목적: topic별 점수, 평균, 전체 강점/약점/추천 누적
- 예: JPA 평균 72점, Docker 90점


"""


from __future__ import annotations


from pydantic import BaseModel, Field
from interview.schemas.question import QuestionCategory

class QualityTrace(BaseModel):
    question_kind: str
    quality: str
    target: str | None = None
    rationale: list[str] = Field(default_factory=list)
    evaluation_source: str = "llm"
    rubric_version: str | None = None
    rubric_question_similarity: float | None = None
    
class CodeAnalysis(BaseModel):
    """topic
  분석 대상 기술
  예: SQLAlchemy, FastAPI Middleware

source_file
  실제 코드가 있는 파일
  예: src/interview/api/users/service.py

current_code
  Evidence에서 가져온 현재 프로젝트 코드

code_assessment
  현재 코드가 어떤 상태인지 설명
  예: 현재 버전에서 정상 동작하지만 최신 스타일로 변경 가능

answer_status
  메인 답변과 꼬리질문을 합친 답변 상태
  예: answered, partially_answered, unknown, misconception

expected_answer
  사용자가 질문에 답했어야 하는 핵심 내용

compatibility_status
  현재 코드와 최신 방식의 관계
  예: current_valid, upgrade_option, deprecated, incorrect

modern_code
  Context7에서 확인한 최신 방식의 코드
  최신 대안이 없으면 None

improvement_reason
  최신 방식으로 변경할 이유

references
  Context7 또는 공식 문서 링크
    """
    topic: str
    source_file: str | None = None
    current_code: str
    code_assessment: str
    answer_status: str
    expected_answer: str
    compatibility_status: str
    modern_code: str | None = None
    improvement_reason: str
    references: list[str] = Field(default_factory=list)
    
class AnswerEvaluation(BaseModel):
    # 질문 정보
    question_id: str
    topic: str
    question: str
    # 메인 답변 + 파생 질문 답변을 합친 전체 답변 요약
    answer_summary: str
    score: float = Field(ge=0.0, le=100.0)
    # 평가 코멘트
    comment: str
    delivery_note: str | None = None
    
    quality_trace: list[QualityTrace] = Field(default_factory=list)
    
    question_category: QuestionCategory
    question_evidence_ids: list[str] = Field(default_factory=list)
    assessment_evidence_ids: list[str] = Field(default_factory=list)
    code_analysis: list[CodeAnalysis] = Field(default_factory=list)



class CompetencyModel(BaseModel):
    """면접 내내 누적되는 역량 상태."""

    topic_scores: dict[str, float] = Field(default_factory=dict)

    strengths: list[str] = Field(default_factory=list)
    improvement_points: list[str] = Field(default_factory=list)
    learning_recommendations: list[str] = Field(default_factory=list)
    average_score: float = 0


class FinalReport(BaseModel):

    # 면접 전체 요약
    summary: str

    # 종합 점수
    overall_score: float = Field(ge=0.0, le=100.0)

    # 전체 강점
    strengths: list[str]

    # 전체 보완 포인트
    improvement_points: list[str]

    # 추천 학습 방향
    learning_recommendations: list[str]

    # 문항별 평가 10개
    evaluations: list[AnswerEvaluation]
