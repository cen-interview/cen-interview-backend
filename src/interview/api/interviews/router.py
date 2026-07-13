from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from interview.api.auth.dependency import get_current_user
from interview.api.database import get_db
from interview.api.interviews.schema import (
    InterviewResultResponse,
    WeakTopicsResponse,
)
from interview.api.interviews.service import (
    get_interview_result,
    get_latest_interview_result,
    get_weak_topics,
    to_result_response,
)
from interview.api.users.model import User


router = APIRouter(
    prefix="/interview-results",
    tags=["Interview Results"],
)


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