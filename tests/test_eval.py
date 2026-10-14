from eval import CASES, Case, grade_answer


def test_there_are_ten_cases_with_refusals():
    assert len(CASES) == 10
    assert any(c.is_refusal for c in CASES)


def test_grading_requires_keyword_and_source():
    case = Case("q", ["14,950"], "eastmere_university.md")
    assert grade_answer(case, "It costs £14,950.\n\nSources: eastmere_university.md")
    assert grade_answer(case, "It costs £14950. Sources: eastmere_university.md")
    assert not grade_answer(case, "It costs £14,950.")  # no citation
    assert not grade_answer(case, "It costs £13,800. Sources: eastmere_university.md")


def test_grading_refusals():
    case = Case("Oxford fees?")
    assert grade_answer(case, "I don't have that information in my knowledge base.")
    assert not grade_answer(case, "Oxford costs £30,000.")
