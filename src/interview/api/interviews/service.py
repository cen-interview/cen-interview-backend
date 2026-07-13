from datetime import datetime, timezone

from sqlalchemy.orm import Session

from interview.api.interviews.schema import InterviewResultResponse
from interview.schemas.report import FinalReport
from interview.api.interviews.model import (
    InterviewResult,
    InterviewSession,
    InterviewSessionStatus,
)
from interview.schemas.events import Mode

def create_interview_session_record(
    db: Session,
    *,
    user_id: int,
    mode: Mode,
) -> InterviewSession:
    """면접 시작 시 DB 세션을 생성하고 발급된 정수 ID를 반환한다."""

    session = InterviewSession(
        user_id=user_id,
        mode=mode,
        status=InterviewSessionStatus.IN_PROGRESS,
    )

    try:
        db.add(session)
        db.commit()
        db.refresh(session)
    except Exception:
        db.rollback()
        raise

    return session


def save_interview_result(
    db: Session,
    *,
    user_id: int,
    session_id: int,
    report: FinalReport,
    topic_scores: dict[str, float],
) -> InterviewResult:
    """사용자의 면접 세션을 종료하고 최종 결과를 한 번만 저장한다."""

    interview_session = (
        db.query(InterviewSession)
        .filter(
            InterviewSession.session_id == session_id,
            InterviewSession.user_id == user_id,
        )
        .first()
    )

    if interview_session is None:
        raise ValueError("면접 세션을 찾을 수 없습니다.")

    existing = (
        db.query(InterviewResult)
        .filter(InterviewResult.session_id == session_id)
        .first()
    )

    if existing is not None:
        return existing

    result = InterviewResult(
        session_id=session_id,
        overall_score=report.overall_score,
        final_report_json=report.model_dump(mode="json"),
        topic_scores_json=topic_scores,
    )

    try:
        interview_session.status = InterviewSessionStatus.COMPLETED
        interview_session.ended_at = datetime.now(timezone.utc)
        db.add(result)
        db.commit()
        db.refresh(result)
    except Exception:
        db.rollback()
        raise

    return result


def get_interview_result(
    db: Session,
    *,
    user_id: int,
    result_id: int,
) -> InterviewResult | None:
    return (
        db.query(InterviewResult)
        .join(
            InterviewSession,
            InterviewResult.session_id == InterviewSession.session_id,
        )
        .filter(
            InterviewResult.id == result_id,
            InterviewSession.user_id == user_id,
        )
        .first()
    )


def get_latest_interview_result(
    db: Session,
    *,
    user_id: int,
) -> InterviewResult | None:
    return (
        db.query(InterviewResult)
        .join(
            InterviewSession,
            InterviewResult.session_id == InterviewSession.session_id,
        )
        .filter(InterviewSession.user_id == user_id)
        .order_by(InterviewResult.created_at.desc())
        .first()
    )


def to_result_response(
    result: InterviewResult,
) -> InterviewResultResponse:
    return InterviewResultResponse(
        result_id=result.id,
        session_id=result.session_id,
        overall_score=result.overall_score,
        topic_scores=result.topic_scores_json,
        report=FinalReport.model_validate(result.final_report_json),
        created_at=result.created_at,
    )


def get_weak_topics(
    db: Session,
    *,
    user_id: int,
    limit: int = 3,
) -> list[str]:
    latest = get_latest_interview_result(
        db,
        user_id=user_id,
    )

    if latest is None:
        return []

    sorted_topics = sorted(
        latest.topic_scores_json.items(),
        key=lambda item: item[1],
    )

    return [
        topic
        for topic, _score in sorted_topics[:limit]
    ]
