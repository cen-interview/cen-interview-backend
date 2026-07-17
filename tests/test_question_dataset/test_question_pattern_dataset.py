"""질문 패턴 데이터 파일 확인 로직을 검증한다."""

from pathlib import Path

from interview.evidence.question_pattern_dataset import find_question_pattern_csvs


def test_find_question_pattern_csvs_returns_only_csv_files(tmp_path: Path) -> None:
    """설정 디렉터리의 CSV 파일만 이름순으로 반환한다."""
    (tmp_path / "interview_question_patterns_b.csv").write_text("b", encoding="utf-8")
    (tmp_path / "interview_question_patterns_a.csv").write_text("a", encoding="utf-8")
    (tmp_path / "unrelated.csv").write_text("ignored", encoding="utf-8")
    (tmp_path / "README.txt").write_text("ignored", encoding="utf-8")

    result = find_question_pattern_csvs(tmp_path)

    assert [path.name for path in result] == [
        "interview_question_patterns_a.csv",
        "interview_question_patterns_b.csv",
    ]


def test_find_question_pattern_csvs_returns_empty_for_missing_directory(
    tmp_path: Path,
) -> None:
    """설정 디렉터리가 없어도 예외 없이 빈 목록을 반환한다."""
    result = find_question_pattern_csvs(tmp_path / "missing")

    assert result == []
