from pydantic import BaseModel, Field


class RubricCriterion(BaseModel):
    """기술 질문에서 충족해야 하는 개별 정답 요소."""

    criterion_id: str
    description: str
    required: bool = True
    weight: float = Field(default=1.0, ge=0.0)


class RubricCandidate(BaseModel):
    """사용자 확인 전까지 세션에 보관하는 rubric 후보."""

    question_id: str
    topic: str
    question: str
    criteria: list[RubricCriterion] = Field(default_factory=list)
    rubric_version: str = "v1"


class RubricMatchResult(BaseModel):
    """현재 답변이 공개 rubric의 필수 기준을 충족했는지 나타낸다."""

    question_id: str
    rubric_version: str
    criterion_similarities: dict[str, float] = Field(default_factory=dict)
    required_criteria_count: int = 0
    matched_required_count: int = 0
    threshold: float = 0.8
    matched_rubric_question_id: str | None = None
    question_similarity: float | None = None

    @property
    def all_required_matched(self) -> bool:
        return (
            self.required_criteria_count > 0
            and self.matched_required_count == self.required_criteria_count
        )

    @property
    def required_coverage(self) -> float:
        if self.required_criteria_count == 0:
            return 0.0
        return self.matched_required_count / self.required_criteria_count

    @property
    def is_sufficient(self) -> bool:
        """필수 기준 중 최대 3개를 충족하면 rubric 평가가 가능하다."""
        return (
            self.required_criteria_count > 0
            and self.matched_required_count
            >= min(3, self.required_criteria_count)
        )
