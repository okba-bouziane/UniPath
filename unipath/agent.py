"""LangGraph agent: an LLM node that can call the two knowledge-base tools.

    START -> agent --(tool calls?)--> tools -> agent ... -> END
"""

import re

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from unipath import config
from unipath.tools import TOOLS

NO_ANSWER = "I don't have that information in my knowledge base."

SYSTEM_PROMPT = f"""You are UniPath AI, an assistant that helps Algerian students understand \
UK university pathway programmes (International Foundation Year, International Year One, \
Pre-Master's, Graduate Diploma), entry requirements, IELTS scores, fees, deadlines, \
scholarships, visas and living costs.

Rules:
1. Answer ONLY with information returned by your tools. Never use outside knowledge, \
even if you think you know the answer.
2. For every factual question, call a tool first. Use `compare_fees` when the user wants to \
compare or list tuition fees or deposits across universities; use `search_knowledge_base` \
for everything else. You may call tools several times.
3. Cite your sources. Every tool result is tagged like `[source: file.md]`. Finish every \
answer with a line in exactly this format:
   Sources: file1.md, file2.md
4. If the tools return NO_RELEVANT_RESULTS, or the results do not contain the answer, \
reply with: "{NO_ANSWER}" and briefly suggest checking the university's official website. \
Do not guess. If only part of the question is covered, answer that part and say which part \
is missing.
5. Be concise and friendly. Use GBP (£) amounts exactly as written in the data. Use a short \
table when comparing several universities.
"""


def get_llm() -> BaseChatModel:
    provider = config.LLM_PROVIDER
    if provider not in config.DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER must be one of {list(config.DEFAULT_MODELS)}, got '{provider}'")
    return init_chat_model(config.llm_model(), model_provider=provider, temperature=0)


def build_agent(llm: BaseChatModel | None = None):
    """Compile the LangGraph agent. Pass an llm to override the .env provider."""
    model = (llm or get_llm()).bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def message_text(message: BaseMessage) -> str:
    """Plain text of a message (content can be a string or a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def ask(agent, question: str, history: list[BaseMessage] | None = None) -> dict:
    """Run one turn. Returns the answer, the files the tools retrieved, and the new history."""
    history = list(history or [])
    result = agent.invoke({"messages": [*history, HumanMessage(question)]})
    new_messages = result["messages"][len(history):]

    retrieved: list[str] = []
    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            for source in re.findall(r"\[source: ([^\]]+)\]", message_text(msg)):
                if source not in retrieved:
                    retrieved.append(source)

    final = next(m for m in reversed(new_messages) if isinstance(m, AIMessage))
    return {
        "answer": message_text(final),
        "retrieved_sources": retrieved,
        "messages": result["messages"],
    }
