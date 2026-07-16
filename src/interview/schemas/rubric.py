from pydantic import BaseModel, Field, model_validator


class RubricCriterion(BaseModel):
    """기술 질문에서 충족해야 하는 개별 정답 요소."""

    criterion_id: str
    description: str
    required: bool = False
    weight: float = Field(default=1.0, ge=0.0)


class RubricCandidate(BaseModel):
    """사용자 확인 전까지 세션에 보관하는 rubric 후보."""

    question_id: str
    topic: str
    question: str
    criteria: list[RubricCriterion] = Field(default_factory=list)
    rubric_version: str = "v1"

    @model_validator(mode="after")
    def normalize_required_criteria(self) -> "RubricCandidate":
        """필수 기준을 1~2개로 제한해 부가 설명의 필수화를 막는다."""
        if not self.criteria:
            return self

        required_indices = [
            index
            for index, criterion in enumerate(self.criteria)
            if criterion.required
        ]
        if not required_indices:
            required_indices = [0]
        else:
            required_indices = required_indices[:2]

        required_index_set = set(required_indices)
        self.criteria = [
            criterion.model_copy(
                update={"required": index in required_index_set}
            )
            for index, criterion in enumerate(self.criteria)
        ]
        return self


class RubricSource(BaseModel):
    """LLM 호출 전 공유 대상 여부를 판별하는 질문 세트 원문."""

    question_id: str
    topic: str
    question: str
    answer: str


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
