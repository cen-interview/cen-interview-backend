"""면접 세션과 최종 결과의 DB 작업을 담당하는 서비스 모듈.

이 모듈은 면접 시작 시 DB 세션을 생성하고, 면접 종료 시 FinalReport와
주제별 점수를 저장한다. 또한 로그인 사용자의 결과 목록, 최근 결과,
특정 결과와 약점 주제를 조회하는 기능을 제공한다.

주요 기능:
    get_interview_session_by_runtime_id:
        LangGraph에서 사용하는 runtime_session_id와 사용자 ID로
        DB 면접 세션을 조회한다.

    create_interview_session_record:
        면접 시작 시 interview_sessions에 in_progress 상태의 행을 생성한다.

    save_interview_result:
        면접 종료 시 interview_results에 최종 리포트를 저장하고
        면접 세션 상태를 completed로 변경한다.
        동일한 세션의 결과는 중복 저장하지 않는다.

    get_interview_result:
        결과 ID와 사용자 ID로 특정 면접 결과를 조회한다.

    get_latest_interview_result:
        현재 사용자의 가장 최근 면접 결과를 조회한다.

    get_interview_results:
        현재 사용자의 전체 면접 결과를 최신순으로 조회한다.

    get_interview_history:
        마이페이지용 완료 면접 통계와 페이지 단위 기록을 조회한다.

    to_result_response:
        InterviewResult ORM 객체를 API 응답 모델로 변환한다.

    get_weak_topics:
        최근 면접의 주제별 점수를 낮은 순으로 정렬하여
        다음 면접에서 참고할 약점 주제를 반환한다.

트랜잭션:
    데이터 생성과 상태 변경 중 오류가 발생하면 rollback을 수행하고
    예외를 다시 전달한다. 정상 처리되면 commit 후 ORM 객체를 refresh한다.
"""


from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from interview.api.interviews.schema import (
    InterviewHistoryItem,
    InterviewHistoryResponse,
    InterviewHistorySummary,
    InterviewResultResponse,
)
from interview.schemas.report import FinalReport
from interview.api.interviews.model import (
    InterviewResult,
    InterviewSession,
    InterviewSessionStatus,
)
from interview.schemas.events import Mode
from sqlalchemy.dialects.postgresql import insert

# 런타임 세션 ID와 사용자 ID로 DB 면접 세션을 조회한다.
def get_interview_session_by_runtime_id(
    db: Session,
    *,
    runtime_session_id: str,
    user_id: int,
) -> InterviewSession | None:
    return (
        db.query(InterviewSession)
        .filter(
            InterviewSession.runtime_session_id
            == runtime_session_id,
            InterviewSession.user_id == user_id,
        )
        .first()
    )

# 면접 시작 정보를 DB에 저장한다.
def create_interview_session_record(
    db: Session,
    *,
    runtime_session_id: str,
    user_id: int,
    mode: Mode,
) -> InterviewSession:
    now = datetime.now(timezone.utc)

    # 기존 진행 세션 종료
    db.query(InterviewSession).filter(
        InterviewSession.user_id == user_id,
        InterviewSession.status
        == InterviewSessionStatus.IN_PROGRESS,
    ).update(
        {
            InterviewSession.status:
                InterviewSessionStatus.CANCELLED,
            InterviewSession.ended_at: now,
        },
        synchronize_session=False,
    )

    # 새 세션 생성
    session = InterviewSession(
        runtime_session_id=runtime_session_id,
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

# 종료된 면접의 최종 결과를 한 번만 저장한다.
def save_interview_result(
    db: Session,
    *,
    user_id: int,
    runtime_session_id: str,
    report: FinalReport,
    topic_scores: dict[str, float],
) -> InterviewResult:
    """사용자의 면접 세션을 종료하고 최종 결과를 한 번만 저장한다."""

    interview_session = (
        db.query(InterviewSession)
        .filter(
            InterviewSession.runtime_session_id
            == runtime_session_id,
            InterviewSession.user_id == user_id,
        )
        .first()
    )

    if interview_session is None:
        raise ValueError("면접 세션을 찾을 수 없습니다.")

    existing = (
        db.query(InterviewResult)
        .filter(
            InterviewResult.session_id
            == interview_session.session_id
        )
        .first()
    )

    if existing is not None:
        return existing

    statement = (
        insert(InterviewResult)
        .values(
            session_id=interview_session.session_id,
            overall_score=report.overall_score,
            final_report_json=report.model_dump(mode="json"),
            topic_scores_json=topic_scores,
        )
        .on_conflict_do_nothing(
            index_elements=[InterviewResult.session_id],
        )
    )

    try:
        interview_session.status = InterviewSessionStatus.COMPLETED
        interview_session.ended_at = datetime.now(timezone.utc)

        db.execute(statement)
        db.commit()

    except Exception:
        db.rollback()
        raise

    result = (
        db.query(InterviewResult)
        .filter(
            InterviewResult.session_id
            == interview_session.session_id
        )
        .one()
    )

    return result

# 결과 ID로 현재 사용자의 특정 면접 결과를 조회한다.
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

# 현재 사용자의 가장 최근 면접 결과를 조회한다.
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

# 현재 사용자의 전체 면접 결과를 최신순으로 조회한다.
def get_interview_results(
    db: Session,
    *,
    user_id: int,
) -> list[InterviewResult]:
    """현재 사용자의 전체 면접 결과를 최신순으로 반환한다."""

    return (
        db.query(InterviewResult)
        .join(
            InterviewSession,
            InterviewResult.session_id
            == InterviewSession.session_id,
        )
        .filter(
            InterviewSession.user_id == user_id
        )
        .order_by(
            InterviewResult.created_at.desc()
        )
        .all()
    )


def get_interview_history(
    db: Session,
    *,
    user_id: int,
    page: int,
    size: int,
) -> InterviewHistoryResponse:
    """마이페이지용 완료 면접 기록과 요약 통계를 조회한다.

    로그인 사용자의 완료된 면접만 대상으로 총 연습 횟수와 종합 점수
    평균을 계산하고, 실제 종료 시각의 내림차순으로 한 페이지를 반환한다.
    평균 점수는 화면에서 바로 사용할 수 있도록 정수로 반올림하며 기록이
    없으면 ``None``을 반환한다.

    Args:
        db:
            면접 세션과 결과를 조회할 SQLAlchemy DB 세션.

        user_id:
            기록을 조회할 로그인 사용자의 ID.

        page:
            1부터 시작하는 페이지 번호.

        size:
            한 페이지에 포함할 최대 기록 수.

    Returns:
        요약 통계, 완료 면접 목록과 페이지 정보를 담은 응답 모델.
    """
    completed_filter = (
        InterviewSession.user_id == user_id,
        InterviewSession.status == InterviewSessionStatus.COMPLETED,
    )

    total, average_score = (
        db.query(
            func.count(InterviewResult.id),
            func.avg(InterviewResult.overall_score),
        )
        .select_from(InterviewResult)
        .join(
            InterviewSession,
            InterviewResult.session_id == InterviewSession.session_id,
        )
        .filter(*completed_filter)
        .one()
    )

    rows = (
        db.query(InterviewResult, InterviewSession)
        .join(
            InterviewSession,
            InterviewResult.session_id == InterviewSession.session_id,
        )
        .filter(*completed_filter)
        .order_by(
            InterviewSession.ended_at.desc(),
            InterviewResult.id.desc(),
        )
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )

    items = [
        InterviewHistoryItem(
            result_id=result.id,
            session_id=interview_session.runtime_session_id,
            completed_at=interview_session.ended_at or result.created_at,
            mode=(
                interview_session.mode.value
                if isinstance(interview_session.mode, Mode)
                else str(interview_session.mode)
            ),
            overall_score=result.overall_score,
        )
        for result, interview_session in rows
    ]

    return InterviewHistoryResponse(
        summary=InterviewHistorySummary(
            total_practice_count=total,
            average_score=(
                int(round(float(average_score)))
                if average_score is not None
                else None
            ),
        ),
        items=items,
        page=page,
        size=size,
        total=total,
    )


# InterviewResult ORM 객체를 API 상세 응답으로 변환한다.
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

# 최근 면접 결과에서 점수가 낮은 주제를 반환한다.
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
