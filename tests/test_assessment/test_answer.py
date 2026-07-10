from interview.assessment.agent import AssessmentAgent
from interview.schemas.question import Question, QuestionCategory, QuestionKind, Difficulty

assessment = AssessmentAgent()

question = Question(
    question_id="q-nplus1-1",
    text="JPA에서 N+1 문제가 왜 발생하는지 설명해주세요.",
    topic="JPA",
    difficulty=Difficulty.MEDIUM,
    kind=QuestionKind.MAIN,
    category=QuestionCategory.TECHNICAL,
)

signal = assessment.evaluate(
    question=question,
    answer_text="어... 쿼리를 날릴 때 발생합니다.",
)

print(signal.model_dump())