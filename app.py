"""UniPath AI — Streamlit chat UI.

Run with:  streamlit run app.py
"""

import os
from html import escape
from pathlib import Path

import streamlit as st

# On Streamlit Community Cloud, keys come from st.secrets. Copy them into the
# environment *before* importing unipath, which reads its config from env vars.
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:  # no secrets file locally; .env is used instead
    pass

from unipath import config  # noqa: E402
from unipath.agent import ask, build_agent  # noqa: E402
from unipath.ingestion import list_source_files  # noqa: E402
from unipath.tools import get_store  # noqa: E402

STYLE_FILE = Path(__file__).parent / "unipath" / "style.css"

EXAMPLE_QUESTIONS = [
    "What IELTS score do I need for the foundation year at Kingsford?",
    "Compare the foundation year fees at Northbridge, Eastmere and Riverton.",
    "I have a Licence in computer science. Which pathway should I choose?",
    "When is the January deadline at Northbridge University?",
    "Are there scholarships for Algerian students?",
    "How much money do I need to show for a Student visa outside London?",
]

st.set_page_config(page_title="UniPath AI", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading the knowledge base…")
def load_agent():
    get_store()  # builds the Chroma index on first run (e.g. on Streamlit Cloud)
    return build_agent()


def init_state() -> None:
    st.session_state.setdefault("chat", [])     # [{"role", "content", "sources"}] for display
    st.session_state.setdefault("history", [])  # LangChain messages for the agent
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("asked", 0)


def questions_left() -> int:
    return max(config.MAX_QUESTIONS_PER_SESSION - st.session_state.asked, 0)


def clear_chat() -> None:
    st.session_state.chat, st.session_state.history = [], []


def queue_question(question: str) -> None:
    st.session_state.pending = question


def render_sidebar() -> None:
    with st.sidebar:
        st.title("🎓 UniPath AI")
        st.caption("UK pathway programmes for Algerian students")

        st.button("🧹 Clear chat", on_click=clear_chat, use_container_width=True,
                  disabled=not st.session_state.chat)
        st.caption(f"Questions left in this session: **{questions_left()}**")

        st.subheader("Source files")
        for path in list_source_files():
            with st.expander(path.name):
                st.markdown(path.read_text(encoding="utf-8"))

        st.divider()
        st.caption(f"LLM: `{config.LLM_PROVIDER}` · `{config.llm_model()}`")
        st.caption(f"Embeddings: `{config.EMBEDDING_MODEL.split('/')[-1]}` (local)")


def render_sources(sources: list[str]) -> None:
    chips = "".join(f'<span class="up-chip">{escape(s)}</span>' for s in sources)
    st.markdown(
        f'<div class="up-sources"><span class="up-sources-label">Sources retrieved:</span>{chips}</div>',
        unsafe_allow_html=True,
    )


def render_message(message: dict) -> None:
    is_assistant = message["role"] == "assistant"
    with st.chat_message(message["role"], avatar="🎓" if is_assistant else None):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])


def render_empty_state() -> None:
    st.markdown(
        '<div class="up-empty"><strong>Not sure where to start?</strong><br>'
        "Pick a question below, or type your own.</div>",
        unsafe_allow_html=True,
    )
    for question in EXAMPLE_QUESTIONS:
        st.button(question, key=f"ex_{question}", on_click=queue_question, args=(question,),
                  use_container_width=True)


def render_notices() -> None:
    st.info(
        "All universities and figures in this demo are **sample data**. Always check the "
        "official university website before applying.",
        icon="ℹ️",
    )
    if missing := config.missing_api_key():
        st.warning(
            f"`{missing}` is not set. Add it to `.env` (local) or to the app's Secrets "
            "(Streamlit Cloud) to start chatting.",
            icon="🔑",
        )
    if questions_left() == 0:
        st.warning(
            f"This public demo allows {config.MAX_QUESTIONS_PER_SESSION} questions per session "
            "to keep the free LLM quota available for everyone. Run it locally for unlimited use.",
            icon="⏳",
        )


def answer_question(question: str) -> None:
    st.session_state.asked += 1
    user_message = {"role": "user", "content": question}
    st.session_state.chat.append(user_message)
    render_message(user_message)

    with st.spinner("Searching the knowledge base…"):
        try:
            result = ask(load_agent(), question, st.session_state.history)
        except Exception as exc:
            st.session_state.asked -= 1  # a failed call should not use up the budget
            st.session_state.chat.pop()
            st.error(
                "Could not get an answer. Check that `LLM_PROVIDER` and the matching API key "
                "are set in `.env` (or in Streamlit secrets), then try again.",
                icon="⚠️",
            )
            with st.expander("Error details"):
                st.code(str(exc))
            return

    st.session_state.history = result["messages"]
    st.session_state.chat.append(
        {"role": "assistant", "content": result["answer"], "sources": result["retrieved_sources"]}
    )
    st.rerun()  # redraw so the sidebar counter and empty state reflect the new turn


def main() -> None:
    st.markdown(f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    init_state()
    render_sidebar()

    st.markdown(
        '<div class="up-hero"><h1>Ask about UK university pathways</h1>'
        "<p>Foundation years, entry requirements, IELTS, fees, deadlines, scholarships and visas.</p></div>",
        unsafe_allow_html=True,
    )
    render_notices()

    can_ask = not config.missing_api_key() and questions_left() > 0
    typed = st.chat_input("Ask about IELTS, fees, deadlines…", disabled=not can_ask)
    question, st.session_state.pending = typed or st.session_state.pending, None
    question = question if can_ask else None

    for message in st.session_state.chat:
        render_message(message)
    if question:
        answer_question(question)
    elif not st.session_state.chat:
        render_empty_state()


main()
