"""Evaluate UniPath AI on 10 questions with known answers.

A case passes when the answer contains every expected keyword AND cites the
expected source file. "Refusal" cases pass when the agent says it does not
have the information instead of guessing.

Usage:
    python eval.py                   # full agent eval (needs an LLM API key)
    python eval.py --verbose         # also print every answer
    python eval.py --retrieval-only  # no API key: check the right file is retrieved
"""

import argparse
from dataclasses import dataclass, field

from unipath.tools import compare_fees, search_knowledge_base


@dataclass
class Case:
    question: str
    # Each item is a keyword, or a tuple of acceptable alternatives.
    expected: list[str | tuple[str, ...]] = field(default_factory=list)
    source: str | None = None  # None = the agent should refuse
    tool: str = "search_knowledge_base"  # tool used for --retrieval-only

    @property
    def is_refusal(self) -> bool:
        return self.source is None


CASES = [
    Case("What IELTS score do I need for the International Foundation Year at Kingsford?",
         ["5.5"], "kingsford_university.md"),
    Case("How much is the tuition fee for the foundation year at the University of Eastmere?",
         ["14,950"], "eastmere_university.md"),
    Case("When is the application deadline for Highmoor University's September intake?",
         ["30 June"], "highmoor_university.md"),
    Case("Which university has the cheapest International Foundation Year?",
         ["Riverton", "13,500"], "riverton_university.md", tool="compare_fees"),
    Case("What Baccalauréat average do I need for the Northbridge foundation year?",
         [("12/20", "12 out of 20", "12 / 20")], "northbridge_university.md"),
    Case("Is there a scholarship for Algerian students at Riverton? How much is it?",
         ["2,000"], "riverton_university.md"),
    Case("How much does the UK Student visa application cost?",
         ["520"], "student_visa.md"),
    Case("What are the typical monthly living costs in Edinburgh?",
         ["1,150"], "scholarships_and_living_costs.md"),
    Case("What is the tuition fee for the University of Oxford foundation year?"),
    Case("What GPA do I need to get into a community college in the United States?"),
]

REFUSAL_MARKERS = ("don't have that information", "do not have that information",
                   "not in my knowledge base", "no information")


def _norm(text: str) -> str:
    return text.lower().replace(",", "").replace("’", "'")


def _has(answer: str, keyword: str | tuple[str, ...]) -> bool:
    options = keyword if isinstance(keyword, tuple) else (keyword,)
    return any(_norm(option) in _norm(answer) for option in options)


def grade_answer(case: Case, answer: str) -> bool:
    if case.is_refusal:
        return any(marker in _norm(answer) for marker in REFUSAL_MARKERS)
    return all(_has(answer, k) for k in case.expected) and case.source in answer


def run_agent_eval(verbose: bool) -> float:
    from unipath.agent import ask, build_agent

    agent = build_agent()
    passed = 0
    for i, case in enumerate(CASES, 1):
        answer = ask(agent, case.question)["answer"]
        ok = grade_answer(case, answer)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  Q{i:>2}: {case.question}")
        if verbose or not ok:
            expected = "a refusal" if case.is_refusal else f"{case.expected} + {case.source}"
            print(f"        expected: {expected}")
            print("        answer:   " + answer.replace("\n", "\n                  "))
    return passed / len(CASES)


def run_retrieval_eval() -> float:
    answerable = [c for c in CASES if not c.is_refusal]
    passed = 0
    for i, case in enumerate(answerable, 1):
        if case.tool == "compare_fees":
            result = compare_fees.invoke({"universities": []})
        else:
            result = search_knowledge_base.invoke({"query": case.question})
        ok = f"[source: {case.source}]" in result and all(_has(result, k) for k in case.expected)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  Q{i:>2}: {case.question}  ->  {case.source}")
    return passed / len(answerable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true",
                        help="only test retrieval (no LLM / API key needed)")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every answer")
    args = parser.parse_args()

    if args.retrieval_only:
        accuracy = run_retrieval_eval()
        print(f"\nRetrieval accuracy: {accuracy:.0%}")
    else:
        accuracy = run_agent_eval(args.verbose)
        print(f"\nAccuracy: {accuracy:.0%} ({round(accuracy * len(CASES))}/{len(CASES)})")


if __name__ == "__main__":
    main()
