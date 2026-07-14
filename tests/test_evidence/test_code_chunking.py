"""코드 구조를 보존하는 Evidence 청킹 검증."""

from interview.evidence.code_chunking import split_code_units


def test_python_chunk_keeps_comment_with_function() -> None:
    """함수 직전의 설명 주석은 함수 본문과 같은 청크에 남아야 한다."""
    text = '''# 토큰을 발급한다.
def issue_token(user_id: int) -> str:
    return f"token:{user_id}"

def revoke_token(token: str) -> None:
    print(token)
'''

    units = split_code_units(text, "python", max_chars=200)

    assert len(units) == 2
    assert units[0].startswith("# 토큰을 발급한다.")
    assert "def issue_token" in units[0]
    assert "def revoke_token" in units[1]


def test_java_chunk_keeps_class_block_together_when_it_fits() -> None:
    """중괄호 기반 Java 클래스는 내부 메서드와 함께 하나의 단위가 된다."""
    text = '''// 인증 서비스
public class AuthService {
    public String issueToken() {
        return "token";
    }
}
'''

    units = split_code_units(text, "java", max_chars=500)

    assert units == [text.strip()]
