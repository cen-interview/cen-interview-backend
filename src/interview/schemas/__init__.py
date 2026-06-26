"""
공용 계약 (schemas) 패키지.

모든 에이전트가 여기서 타입을 import 한다.
  예: from interview.schemas import Question, AnswerQualitySignal

⚠️ 이 패키지 안의 모델을 수정할 때는 반드시 팀 합의 후 함께 바꾼다.
"""
from .events import (
    AnswerSubmitted,
    BaseEvent,
    EndRequested,
    InterviewerEvent,
    NoResponseTimeout,
    ReplayRequested,
    SilenceDetected,
)
from .evidence import EvidenceChunk, RetrievalResult, SourceType
from .question import Difficulty, Question, QuestionKind
from .signals import AnswerQuality, AnswerQualitySignal
from .report import AnswerEvaluation, CompetencyModel, FinalReport

__all__ = [
    # events
    "BaseEvent",
    "AnswerSubmitted",
    "EndRequested",
    "SilenceDetected",
    "ReplayRequested",
    "NoResponseTimeout",
    "InterviewerEvent",
    # evidence
    "EvidenceChunk",
    "RetrievalResult",
    "SourceType",
    # question
    "Question",
    "Difficulty",
    "QuestionKind",
    # signals
    "AnswerQualitySignal",
    "AnswerQuality",
    # report
    "AnswerEvaluation",
    "CompetencyModel",
    "FinalReport",
]
