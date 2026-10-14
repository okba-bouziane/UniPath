"""Build the Chroma vector store from the markdown files in data/.

Usage:
    python ingest.py
"""

from unipath import config
from unipath.ingestion import build_vectorstore, list_source_files


def main() -> None:
    files = list_source_files()
    print(f"Found {len(files)} markdown files in {config.DATA_DIR}")
    _, n_chunks = build_vectorstore()
    print(f"Stored {n_chunks} chunks in {config.CHROMA_DIR} (collection '{config.COLLECTION_NAME}')")


if __name__ == "__main__":
    main()
