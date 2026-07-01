"""FastAPI fake 면접 흐름 테스트."""

from fastapi.testclient import TestClient

from interview.api.main import SESSIONS, app


def test_session_answer_end_flow():
    SESSIONS.clear()
    client = TestClient(app)

    started = client.post("/sessions", json={"mode": "chat", "max_questions": 3})

    assert started.status_code == 200
    body = started.json()
    session_id = body["session_id"]
    question = body["question"]

    answered = client.post(
        f"/sessions/{session_id}/answer",
        json={
            "question_id": question["question_id"],
            "text": "기술 면접 질문에 대해 핵심 개념과 프로젝트 경험을 함께 설명하겠습니다.",
        },
    )

    assert answered.status_code == 200
    assert answered.json()["question"] is not None

    ended = client.post(f"/sessions/{session_id}/end")

    assert ended.status_code == 200
    ended_body = ended.json()
    assert ended_body["finished"] is True
    assert ended_body["report"]["evaluations"]
