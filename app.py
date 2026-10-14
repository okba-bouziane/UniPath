"""UniPath AI — Streamlit chat UI.

Run with:  streamlit run app.py
"""

import os

import streamlit as st

# On Streamlit Community Cloud, keys come from st.secrets. Copy them into the
# environment *before* importing unipath, which reads its config from env vars.
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:  # no secrets file locally — .env is used instead
    pass

from unipath import config  # noqa: E402
from unipath.agent import ask, build_agent  # noqa: E402
from unipath.ingestion import list_source_files  # noqa: E402
from unipath.tools import get_store  # noqa: E402

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


def sidebar() -> None:
    with st.sidebar:
        st.title("🎓 UniPath AI")
        st.caption("UK pathway programmes for Algerian students")

        with st.popover("💡 Example questions", use_container_width=True):
            for q in EXAMPLE_QUESTIONS:
                if st.button(q, key=f"ex_{q}", use_container_width=True):
                    st.session_state.pending = q

        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.chat, st.session_state.history = [], []

        st.subheader("Source files")
        for path in list_source_files():
            with st.expander(path.name):
                st.markdown(path.read_text(encoding="utf-8"))

        st.divider()
        st.caption(f"LLM: `{config.LLM_PROVIDER}` · `{config.llm_model()}`")
        st.caption(f"Embeddings: `{config.EMBEDDING_MODEL.split('/')[-1]}` (local)")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"], avatar="🎓" if message["role"] == "assistant" else None):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Retrieved from: " + ", ".join(f"`{s}`" for s in message["sources"]))


def main() -> None:
    init_state()
    sidebar()

    st.header("Ask about UK university pathways")
    st.info(
        "All universities and figures in this demo are **sample data** — always check the "
        "official university website before applying.",
        icon="ℹ️",
    )

    if missing := config.missing_api_key():
        st.warning(
            f"`{missing}` is not set. Add it to `.env` (local) or to the app's Secrets "
            "(Streamlit Cloud) to start chatting.",
            icon="🔑",
        )

    for message in st.session_state.chat:
        render_message(message)

    question = st.chat_input("e.g. What IELTS score do I need for a foundation year?")
    question = question or st.session_state.pending
    st.session_state.pending = None
    if not question:
        return

    user_msg = {"role": "user", "content": question}
    st.session_state.chat.append(user_msg)
    render_message(user_msg)

    with st.spinner("Searching the knowledge base…"):
        try:
            result = ask(load_agent(), question, st.session_state.history)
        except Exception as exc:
            st.error(
                f"Could not get an answer: {exc}\n\n"
                "Check that `LLM_PROVIDER` and the matching API key are set in `.env` "
                "(or in Streamlit secrets)."
            )
            st.session_state.chat.pop()
            return

    st.session_state.history = result["messages"]
    bot_msg = {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["retrieved_sources"],
    }
    st.session_state.chat.append(bot_msg)
    render_message(bot_msg)


main()
