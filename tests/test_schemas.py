import pytest
from pydantic import ValidationError

from interview.schemas.evidence import EvidenceChunk
from interview.schemas.events import AnswerSubmitted
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.schemas.signals import AnswerQuality, AnswerQualitySignal


def test_question_schema_valid():
    question = Question(
        question_id="q-1",
        text="FastAPI에서 Depends를 사용하는 이유는?",
        topic="FastAPI",
        difficulty=Difficulty.EASY,
        kind=QuestionKind.MAIN,
    )

    assert question.difficulty == Difficulty.EASY
    assert question.kind == QuestionKind.MAIN


def test_question_schema_invalid_kind():
    with pytest.raises(ValidationError):
        Question(
            question_id="q-1",
            text="질문",
            topic="FastAPI",
            difficulty=Difficulty.EASY,
            kind="hint",
        )


def test_answer_quality_signal_valid():
    signal = AnswerQualitySignal(
        question_id="q-1",
        quality=AnswerQuality.SHALLOW,
        missing_keywords=["Depends"],
        covered_keywords=[],
        misconception_note=None,
        rationale="임시 평가",
    )

    assert signal.quality == AnswerQuality.SHALLOW


def test_answer_quality_signal_invalid_quality():
    with pytest.raises(ValidationError):
        AnswerQualitySignal(
            question_id="q-1",
            quality="conflict",
        )


def test_evidence_chunk_valid():
    chunk = EvidenceChunk(
        chunk_id="c-1",
        text="FastAPI Depends는 의존성 주입에 사용된다.",
        source_type="notion",
        source_url="https://example.com",
        topic="FastAPI",
        confidence=0.8,
    )

    assert chunk.source_type == "notion"


def test_answer_submitted_valid():
    event = AnswerSubmitted(
        session_id="s-1",
        question_id="q-1",
        text="Depends는 의존성 주입에 사용됩니다.",
    )

    assert event.type == "answer_submitted"