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
    
class CodeAnalysis(BaseModel):
    """PROJECT 질문의 Evidence를 기반으로 생성한 코드 분석."""

    # 분석 대상 기술 또는 코드 주제
    topic: str

    # Evidence가 추출된 실제 파일 경로
    source_file: str | None = None

    # Evidence에 포함된 현재 프로젝트 코드
    current_code: str

    # 현재 코드의 역할과 상태에 대한 설명
    code_assessment: str

    # 사용자의 답변 상태
    # 예: answered, partially_answered, unknown, misconception
    answer_status: str

    # 질문에 답할 때 포함했어야 하는 핵심 내용
    expected_answer: str

    # 외부 최신 문서를 사용하지 않으므로 기본값 고정
    compatibility_status: str = "not_evaluated"


    # 현재 코드 또는 답변에서 개선할 부분
    improvement_reason: str

    # 외부 레퍼런스를 사용하지 않으므로 빈 목록
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
    # 프론트에 전달할 코드 분석
    question_category: QuestionCategory | None = None
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
