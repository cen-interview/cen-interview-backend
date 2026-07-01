from fastapi.testclient import TestClient

from interview.api.main import app


client = TestClient(app)


def test_session_flow_until_10_main_questions():
    print("\n[1] 세션 생성")
    response = client.post("/api/sessions")
    assert response.status_code == 200

    data = response.json()
    session_id = data["session_id"]
    question = data["question"]

    main_count = 1

    print(f"[MAIN {main_count}] {question['text']}")

    turn = 0

    while True:
        turn += 1

        response = client.post(
            f"/api/sessions/{session_id}/answer",
            json={"text": f"{turn}번째 임시 답변입니다."},
        )
        assert response.status_code == 200

        data = response.json()
        next_question = data["next_question"]
        finished = data["finished"]

        print(f"\n[TURN {turn}]")
        print(data)

        if finished:
            print("\n[종료] max_questions 도달")
            break

        assert next_question is not None

        if next_question["kind"] == "main":
            main_count += 1
            print(f"[MAIN {main_count}] {next_question['text']}")
        else:
            print(f"[{next_question['kind'].upper()}] {next_question['text']}")

    assert main_count == 10
    assert finished is True