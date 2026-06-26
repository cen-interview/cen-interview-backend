"""Interviewer 라우팅 예시 테스트.

Strategy/Assessment 를 가짜로 끼워 라우팅만 검증하는 패턴을 보여준다.
(담당 C 가 agent.py 를 채우면 이 테스트가 의미를 가진다.)
"""

import pytest

# TODO(담당 C): InterviewerAgent 가 채워지면 아래 스켈레톤을 활성화.
#
# from interview.interviewer import InterviewerAgent, SessionState
# from interview.schemas.events import AnswerSubmitted, Mode
#
# class FakeAssessment:
#     def __init__(self, signal): self._signal = signal
#     def evaluate(self, **kw): return self._signal
#
# class FakeStrategy:
#     def next_follow_up(self, topic, kw): ...   # 가짜 Question 반환
#
# def test_shallow_answer_routes_to_follow_up(sample_question, shallow_signal):
#     ...

@pytest.mark.skip(reason="agent.py 구현 후 활성화")
def test_placeholder():
    pass
