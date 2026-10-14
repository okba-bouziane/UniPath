"""The two tools the agent can call."""

from langchain_chroma import Chroma
from langchain_core.tools import tool

from unipath import config
from unipath.fees import load_fee_table, match_universities
from unipath.ingestion import build_vectorstore, get_vectorstore, is_empty

NOT_FOUND = "NO_RELEVANT_RESULTS: the knowledge base has no information about this."

_store: Chroma | None = None


def get_store() -> Chroma:
    """Open the persisted Chroma store, building it on first run if it is empty."""
    global _store
    if _store is None:
        _store = get_vectorstore()
        if is_empty(_store):
            _store, _ = build_vectorstore()
    return _store


def set_store(store: Chroma | None) -> None:
    """Swap the store (used by tests)."""
    global _store
    _store = store


@tool
def search_knowledge_base(query: str) -> str:
    """Search the UniPath knowledge base about UK university pathway programmes
    (foundation year, international year one, pre-master's), entry requirements,
    IELTS scores, fees, deadlines, scholarships, visas and living costs.

    Returns the most relevant text chunks, each tagged with its source file.
    """
    results = get_store().similarity_search_with_score(query, k=config.TOP_K)
    # Chroma returns cosine distance; relevance = 1 - distance.
    relevant = [doc for doc, distance in results if 1 - distance >= config.MIN_RELEVANCE]
    if not relevant:
        return NOT_FOUND
    return "\n\n---\n\n".join(
        f"[source: {doc.metadata['source']}]\n{doc.page_content}" for doc in relevant
    )


@tool
def compare_fees(universities: list[str]) -> str:
    """Compare tuition fees and deposits (in GBP) for pathway programmes across
    universities. Pass university names, e.g. ["Kingsford", "Riverton"].
    Pass an empty list to compare all universities in the knowledge base.
    """
    rows = load_fee_table()
    known = sorted({r.university for r in rows})
    if universities:
        selected, unknown = match_universities(universities, known)
    else:
        selected, unknown = known, []

    lines = []
    if selected:
        lines.append("| University | Programme | Tuition fee (GBP) | Deposit (GBP) | Source |")
        lines.append("|---|---|---|---|---|")
        chosen = sorted(
            (r for r in rows if r.university in selected),
            key=lambda r: (r.programme, r.tuition_gbp),
        )
        for r in chosen:
            lines.append(
                f"| {r.university} | {r.programme} | £{r.tuition_gbp:,} | £{r.deposit_gbp:,} "
                f"| [source: {r.source}] |"
            )
        for programme in sorted({r.programme for r in chosen}):
            options = [r for r in chosen if r.programme == programme]
            if len(options) > 1:
                cheapest = min(options, key=lambda r: r.tuition_gbp)
                lines.append(
                    f"\nCheapest {programme}: {cheapest.university} (£{cheapest.tuition_gbp:,})."
                )
    if unknown:
        lines.append(
            f"\nNO_RELEVANT_RESULTS for: {', '.join(unknown)}. "
            f"Known universities: {', '.join(known)}."
        )
    return "\n".join(lines) if lines else NOT_FOUND


TOOLS = [search_knowledge_base, compare_fees]
