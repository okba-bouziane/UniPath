"""Load markdown files, split them into chunks and store them in Chroma."""

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from unipath import config

HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]


def list_source_files(data_dir: Path = config.DATA_DIR) -> list[Path]:
    return sorted(Path(data_dir).glob("*.md"))


def load_documents(data_dir: Path = config.DATA_DIR) -> list[Document]:
    """Read every markdown file in data_dir as one Document."""
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in list_source_files(data_dir)
    ]


def split_documents(docs: list[Document]) -> list[Document]:
    """Split by markdown headers first, then by size.

    Each chunk is prefixed with its title/section so it still makes sense
    on its own (e.g. a fees table keeps the university name).
    """
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks: list[Document] = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            heading = " > ".join(
                section.metadata[key] for _, key in HEADERS
                if key in section.metadata
            )
            for piece in size_splitter.split_text(section.page_content):
                metadata = {"source": doc.metadata["source"], **section.metadata}
                text = f"{heading}\n\n{piece}" if heading else piece
                chunks.append(Document(page_content=text, metadata=metadata))

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(persist_dir: Path = config.CHROMA_DIR) -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vectorstore(
    data_dir: Path = config.DATA_DIR, persist_dir: Path = config.CHROMA_DIR
) -> tuple[Chroma, int]:
    """(Re)build the Chroma collection from scratch. Returns (store, n_chunks)."""
    store = get_vectorstore(persist_dir)
    store.reset_collection()
    chunks = split_documents(load_documents(data_dir))
    store.add_documents(chunks, ids=[str(c.metadata["chunk_id"]) for c in chunks])
    return store, len(chunks)


def is_empty(store: Chroma) -> bool:
    return store._collection.count() == 0
