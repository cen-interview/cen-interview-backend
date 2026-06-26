"""공용 계약(schemas).

⭐ 이 패키지는 4명 모두가 의존하는 "계약"이다. 여기 모델을 바꾸면 다른
에이전트가 깨질 수 있으니, 변경 시 반드시 팀과 합의하고 같이 수정한다.

다른 모듈은 보통 여기서 바로 import 한다:
    from interview.schemas import AnswerQualitySignal, Question, ...
"""

from interview.schemas.events import (
    AnswerSubmitted,
    EndRequested,
    InterviewEvent,
    Mode,
    NoResponseTimeout,
    ReplayRequested,
    SilenceDetected,
)
from interview.schemas.evidence import CoverageMap, EvidenceChunk, SourceType
from interview.schemas.question import Difficulty, Question, QuestionKind
from interview.schemas.report import (
    AnswerEvaluation,
    CompetencyModel,
    FinalReport,
)
from interview.schemas.signals import AnswerQualitySignal, QualityLevel

__all__ = [
    # events
    "Mode", "InterviewEvent", "AnswerSubmitted", "EndRequested",
    "ReplayRequested", "SilenceDetected", "NoResponseTimeout",
    # evidence
    "EvidenceChunk", "SourceType", "CoverageMap",
    # question
    "Question", "Difficulty", "QuestionKind",
    # signals
    "AnswerQualitySignal", "QualityLevel",
    # report
    "AnswerEvaluation", "CompetencyModel", "FinalReport",
]
