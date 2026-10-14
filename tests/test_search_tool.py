import re

import pytest

from unipath.tools import NOT_FOUND, search_knowledge_base


def search(query: str) -> str:
    return search_knowledge_base.invoke({"query": query})


@pytest.mark.usefixtures("search_store")
class TestSearchKnowledgeBase:
    def test_results_are_tagged_with_source_files(self):
        result = search("What IELTS score do I need for the Kingsford foundation year?")
        sources = re.findall(r"\[source: ([\w.]+)\]", result)
        assert sources and all(s.endswith(".md") for s in sources)
        assert sources[0] == "kingsford_university.md"

    @pytest.mark.parametrize(
        "query, expected_source",
        [
            ("Riverton scholarship for Algerian students", "riverton_university.md"),
            ("How much does the student visa application cost?", "student_visa.md"),
            ("Highmoor application deadline", "highmoor_university.md"),
            ("Do I need a foundation year with the Algerian Baccalaureate?", "algerian_baccalaureate.md"),
        ],
    )
    def test_retrieves_the_right_file(self, query, expected_source):
        assert f"[source: {expected_source}]" in search(query)

    @pytest.mark.parametrize("query", ["best pizza recipe", "What is the capital of Australia?"])
    def test_off_topic_questions_return_not_found(self, query):
        assert search(query) == NOT_FOUND
