"""Agent wiring tests using a scripted fake LLM (no API key needed)."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from unipath.agent import ask, build_agent


class ScriptedLLM(BaseChatModel):
    """Returns pre-written AIMessages in order, ignoring the input."""

    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


@pytest.mark.usefixtures("search_store")
def test_agent_calls_tools_and_collects_sources():
    llm = ScriptedLLM(
        responses=[
            tool_call("search_knowledge_base", {"query": "Kingsford IELTS foundation"}),
            tool_call("compare_fees", {"universities": ["Kingsford", "Riverton"]}),
            AIMessage(content="Kingsford needs IELTS 5.5.\n\nSources: kingsford_university.md"),
        ]
    )
    result = ask(build_agent(llm), "IELTS for Kingsford, and compare fees with Riverton?")

    assert llm.calls == 3  # agent -> tools -> agent -> tools -> agent
    assert result["answer"].endswith("Sources: kingsford_university.md")
    assert "kingsford_university.md" in result["retrieved_sources"]
    assert "riverton_university.md" in result["retrieved_sources"]


def test_agent_keeps_chat_history():
    llm = ScriptedLLM(responses=[AIMessage(content="first"), AIMessage(content="second")])
    agent = build_agent(llm)
    turn1 = ask(agent, "hello")
    turn2 = ask(agent, "again", history=turn1["messages"])
    assert turn2["answer"] == "second"
    assert len(turn2["messages"]) == 4
    assert turn2["retrieved_sources"] == []
