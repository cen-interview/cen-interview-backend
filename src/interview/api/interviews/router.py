"""면접 최종 결과 조회 API를 제공하는 FastAPI 라우터.

이 모듈은 로그인한 사용자가 자신의 면접 결과와 약점 주제를 조회할 수 있는
HTTP API를 정의한다. 실제 DB 조회와 응답 변환은 interviews.service에
위임하고, 이 라우터는 인증, 요청값 전달, HTTP 오류 처리를 담당한다.

제공 API:
    GET /api/interview-results/latest:
        현재 사용자의 가장 최근 면접 결과를 반환한다.

    GET /api/interview-results
        → 전체 면접 결과 목록

    GET /api/interview-results/history:
        마이페이지용 완료 면접 통계와 페이지 단위 기록을 반환한다.

    GET /api/interview-results/weak-topics:
        가장 최근 면접의 주제별 점수를 기준으로 점수가 낮은 주제를 반환한다.
        반환된 주제는 다음 면접의 질문 개인화에 사용할 수 있다.

    GET /api/interview-results/{result_id}:
        결과 ID로 현재 사용자의 특정 면접 결과를 조회한다.

인증:
    모든 API는 get_current_user 의존성을 사용한다.
    요청의 Access Token으로 로그인 사용자를 확인하고, 다른 사용자의
    면접 결과는 조회할 수 없도록 user_id를 조회 조건에 포함한다.

역할 분리:
    router:
        인증된 사용자 확인, HTTP 요청·응답 및 예외 처리.

    service:
        DB 조회, 사용자 소유권 확인, ORM 객체의 응답 모델 변환.

    schema:
        API 응답 데이터의 형식과 타입 정의.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from interview.api.auth.dependency import get_current_user
from interview.api.database import get_db
from interview.api.interviews.schema import (
    InterviewHistoryResponse,
    InterviewResultResponse,
    InterviewResultSummary,
    WeakTopicsResponse,
)
from interview.api.interviews.service import (
    get_interview_history,
    get_interview_result,
    get_interview_results,
    get_latest_interview_result,
    get_weak_topics,
    to_result_response,
)
from interview.api.users.model import User


router = APIRouter(
    prefix="/interview-results",
    tags=["Interview Results"],
)


# 현재 로그인한 사용자의 가장 최근 면접 결과를 반환한다.
@router.get("/latest", response_model=InterviewResultResponse)
def get_latest_result(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = get_latest_interview_result(
        db,
        user_id=current_user.id,
    )

    if result is None:
        raise HTTPException(404, "저장된 면접 결과가 없습니다.")

    return to_result_response(result)

# 현재 사용자의 전체 면접 결과를 최신순으로 반환한다.
@router.get("",response_model=list[InterviewResultSummary],)
def read_results(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """현재 사용자의 전체 면접 결과를 최신순으로 반환한다."""

    results = get_interview_results(
        db,
        user_id=current_user.id,
    )

    return [
        InterviewResultSummary(
            result_id=result.id,
            session_id=result.session_id,
            overall_score=result.overall_score,
            created_at=result.created_at,
        )
        for result in results
    ]


@router.get("/history", response_model=InterviewHistoryResponse)
def read_history(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """마이페이지용 완료 면접 기록과 요약 통계를 반환한다.

    Args:
        page:
            1부터 시작하는 페이지 번호.

        size:
            한 페이지에서 조회할 기록 수. 최대 100건까지 허용한다.

        db:
            면접 기록을 조회할 SQLAlchemy DB 세션.

        current_user:
            Access Token으로 인증된 현재 사용자.

    Returns:
        완료 면접 통계와 최근 완료순 기록 목록.
    """
    return get_interview_history(
        db,
        user_id=current_user.id,
        page=page,
        size=size,
    )


# 현재 사용자의 최근 면접에서 점수가 낮은 주제를 반환한다.
@router.get("/weak-topics", response_model=WeakTopicsResponse)
def read_weak_topics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return WeakTopicsResponse(
        topics=get_weak_topics(
            db,
            user_id=current_user.id,
        )
    )


# 결과 ID로 현재 사용자의 특정 면접 결과를 조회한다.
@router.get("/{result_id}", response_model=InterviewResultResponse)
def read_result(
    result_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = get_interview_result(
        db,
        user_id=current_user.id,
        result_id=result_id,
    )

    if result is None:
        raise HTTPException(404, "면접 결과를 찾을 수 없습니다.")

    return to_result_response(result)
