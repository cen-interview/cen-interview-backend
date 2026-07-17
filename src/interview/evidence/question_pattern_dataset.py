"""질문 패턴 정적 데이터 파일의 배치 경로를 확인한다."""

from pathlib import Path


def find_question_pattern_csvs(data_path: str | Path) -> list[Path]:
    """설정된 디렉터리에 존재하는 질문 패턴 CSV 파일을 반환한다."""
    directory = Path(data_path).expanduser().resolve()
    if not directory.is_dir():
        return []

    return sorted(
        path for path in directory.glob("interview_question_patterns_*.csv") if path.is_file()
    )
