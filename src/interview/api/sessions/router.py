"""면접 세션 생성과 이벤트 처리를 제공하는 FastAPI 라우터."""

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException

from interview.api.auth.dependency import get_current_user
from interview.api.sessions.schema import EventRequest, StartRequest
from interview.api.users.model import User
from interview.evidence.store import get_store
from interview.interviewer.adapters import from_chat, from_voice
from interview.interviewer.facade import (
    InterviewSession,
    create_session as create_interview_session,
    get_session,
)
from interview.interviewer.session import SessionState
from interview.schemas.events import Mode
from interview.schemas.question import Question
from sqlalchemy.orm import Session

from interview.api.database import get_db
from interview.api.interviews.service import (
    create_interview_session_record,
    get_interview_session_by_runtime_id,
    get_weak_topics,
    save_interview_result,
)

router = APIRouter(prefix="/sessions", tags=["Interview Sessions"])

SessionFactory = Callable[..., tuple[InterviewSession, Question]]


def get_interview_session_factory() -> SessionFactory:
    """운영 환경에서 사용할 면접 세션 생성 함수를 반환한다.

    FastAPI dependency로 분리했기 때문에 테스트에서는 이 함수만 override하여
    실제 Strategy와 FakeAssessment가 담긴 세션 factory를 주입할 수 있다.

    Returns:
        실제 StrategyAgent와 AssessmentAgent를 사용하는 세션 생성 함수.
    """
    return create_interview_session


@router.post("")
def start_session(
    req: StartRequest,
    current_user: User = Depends(get_current_user),
    session_factory: SessionFactory = Depends(get_interview_session_factory),
    db: Session = Depends(get_db),
):
    """면접 세션을 생성하고 compiled graph가 만든 첫 질문을 반환한다.

    Args:
        req:
            면접 시작 요청. mode는 ``chat`` 또는 ``voice``여야 한다.

        session_factory:
            세션을 생성할 함수. 운영에서는 실제 의존성을 사용하고 테스트에서는
            FastAPI dependency override로 FakeAssessment를 주입할 수 있다.

    Returns:
        생성된 세션 ID와 첫 질문, 면접관 발화 큐, 종료 여부.

    Raises:
        HTTPException:
            지원하지 않는 mode가 전달되면 400 응답을 반환한다.
    """
    try:
        mode = Mode(req.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"unknown mode: {req.mode}",
        )

    user_id = str(current_user.id)

    coverage = get_store().build_coverage_map(user_id=user_id)

    weak_history_topics = get_weak_topics(
        db,
        user_id=current_user.id,
    )

    session, _ = session_factory(
        mode,
        coverage=coverage,
        user_id=user_id,
        weak_history_topics=weak_history_topics,
    )
    state = session.get_state()

    create_interview_session_record(
        db=db,
        runtime_session_id=state.session_id,
        user_id=current_user.id,
        mode=mode,
    )

    return _session_response(state)


@router.post("/{session_id}/events")
def post_event(
    session_id: str,                          # 진행할 면접
    req: EventRequest,                        # 제출한 답변
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """raw 입력을 세션 mode에 맞게 변환하고 중단된 그래프를 재개한다.

    세션 생성 시 저장한 mode와 현재 질문 ID를 사용해 AdaptedInput을 만든다.
    종료된 세션이면 그래프를 다시 실행하지 않고 기존 상태와 리포트를 반환한다.

    Args:
        session_id:
            이벤트를 전달할 면접 세션 ID.

        req:
            action과 채널별 입력 데이터가 담긴 이벤트 요청.

    Returns:
        이벤트 처리 후 최신 질문, 발화 큐, 종료 여부와 선택적 리포트.

    Raises:
        HTTPException:
            세션이 없으면 404, payload를 변환할 수 없으면 400을 반환한다.
    """
    try:
        session = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    
    db_session = get_interview_session_by_runtime_id(
        db,
        runtime_session_id=session_id,
        user_id=current_user.id,
    )

    if db_session is None:
        raise HTTPException(
            status_code=404,
            detail="면접 세션을 찾을 수 없습니다.",
        )

    state = session.get_state()

    if state.finished:
        result = _save_finished_result(
            db=db,
            current_user=current_user,
            runtime_session_id=session_id,
            session=session,
        )

        return _session_response(
            state,
            result_id=result.id,
        )

    question_id = (
        state.current_question.question_id
        if state.current_question is not None
        else ""
    )

    try:
        payload = req.to_adapter_payload()
        adapted_input = (
            from_voice(session_id, question_id, payload)
            if state.mode == Mode.VOICE.value
            else from_chat(session_id, question_id, payload)
        )
        state = session.submit_event(
            adapted_input,
            client_event_id=req.client_event_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    result_id = None

    # 이번 답변 또는 종료 요청으로 면접이 종료된 경우
    if state.finished:
        result = _save_finished_result(
            db=db,
            current_user=current_user,
            runtime_session_id=session_id,
            session=session,
        )
        result_id = result.id

    return _session_response(
        state,
        result_id=result_id,
    )

def _save_finished_result(
    *,
    db: Session,
    current_user: User,
    runtime_session_id: str,
    session: InterviewSession,
):
    report = session.finalize()

    topic_scores = {
        evaluation.topic: evaluation.score
        for evaluation in report.evaluations
    }

    return save_interview_result(
        db=db,
        user_id=current_user.id,
        runtime_session_id=runtime_session_id,
        report=report,
        topic_scores=topic_scores,
    )


def _session_response(
    state: SessionState,
    result_id: int | None = None,
) -> dict:
    """SessionState를 세션 생성과 이벤트 API의 공통 응답으로 변환한다.

    Args:
        state:
            compiled graph의 최신 체크포인트에서 복원한 세션 상태.

    Returns:
        세션 ID, 현재 질문, TTS가 안내 문장과 질문을 순서대로 재생할 수 있는
        면접관 발화 큐, 오류, 종료 여부와 최종 리포트를 JSON 직렬화 가능한
        값으로 정리한 dict.
    """
    return {
        "session_id": state.session_id,
        "finished": state.finished,
        "result_id": result_id,
        "question": (
            state.current_question.model_dump(mode="json")
            if state.current_question is not None and not state.finished
            else None
        ),
        "utterance_queue": state.utterance_queue,
        "last_utterance": state.last_utterance,
        "transcript": [turn.model_dump(mode="json") for turn in state.transcript],
        "turn_type": state.turn_type,
        "error": state.error,
        "report": state.report if state.finished else None,
    }
