"""개인화 관련 로직.

get_weak_topics()는 Assessment 담당이 실제 구현을 제공할 예정이다.
그전까지는 이 stub이 항상 빈 리스트를 반환해 "이전 이력 없음"과 동일하게
동작하게 한다. D가 실제 함수를 dev에 push하면, 아래 함수 내용만 실제
호출로 교체하면 된다 (pick_topic 노드 쪽 코드는 안 건드려도 됨).
"""


def get_weak_topics(user_id: str | None) -> list[str]:
    """이전 면접 이력에서 약점으로 판단된 주제 목록을 반환한다.

    [STUB] 실제 구현은 Assessment 담당 예정. 현재는 항상 빈 리스트를
    반환하여, 이전 이력이 없는 것과 동일하게 동작한다

    Args:
        user_id: 사용자 식별자. None이면(비회원/미배선 상태) 무조건
            빈 리스트를 반환한다.

    Returns:
        약점 주제 이름 목록. 지금은 항상 빈 리스트.
    """
    if user_id is None:
        return []
    return []