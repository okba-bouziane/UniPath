from unipath import config
from unipath.ingestion import (
    build_vectorstore,
    list_source_files,
    load_documents,
    split_documents,
)


def test_loads_every_markdown_file():
    docs = load_documents()
    assert len(docs) == len(list_source_files()) >= 10
    assert all(doc.metadata["source"].endswith(".md") for doc in docs)


def test_every_data_file_is_marked_as_sample_data():
    for doc in load_documents():
        assert "SAMPLE DATA" in doc.page_content, doc.metadata["source"]


def test_chunks_keep_source_and_heading_context():
    chunks = split_documents(load_documents())
    assert len(chunks) > len(list_source_files())
    assert all(len(c.page_content) <= config.CHUNK_SIZE + 200 for c in chunks)

    fee_chunks = [
        c for c in chunks
        if c.metadata["source"] == "kingsford_university.md"
        and c.metadata.get("section", "").startswith("Fees")
    ]
    assert fee_chunks, "expected a fees chunk for Kingsford"
    # The heading is prepended so the chunk is self-describing.
    assert fee_chunks[0].page_content.startswith("Kingsford University London")
    assert "21,500" in fee_chunks[0].page_content


def test_chunk_ids_are_unique():
    ids = [c.metadata["chunk_id"] for c in split_documents(load_documents())]
    assert len(ids) == len(set(ids))


def test_build_vectorstore_persists_all_chunks(tmp_path):
    store, n_chunks = build_vectorstore(config.DATA_DIR, tmp_path)
    assert store._collection.count() == n_chunks
    # Rebuilding resets instead of duplicating.
    store, n_again = build_vectorstore(config.DATA_DIR, tmp_path)
    assert store._collection.count() == n_again == n_chunks


def test_ingest_custom_folder(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "demo.md").write_text("# Demo Uni\n\n## Fees\n\nThe foundation year costs 1,000 GBP.")
    store, n_chunks = build_vectorstore(data, tmp_path / "db")
    assert n_chunks == 1
    hit = store.similarity_search("foundation fees", k=1)[0]
    assert hit.metadata["source"] == "demo.md"
