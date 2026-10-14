import pytest

from unipath import config, tools
from unipath.ingestion import build_vectorstore


@pytest.fixture(scope="session")
def vectorstore(tmp_path_factory):
    """A Chroma store built from the real data/ folder in a temp directory."""
    store, _ = build_vectorstore(config.DATA_DIR, tmp_path_factory.mktemp("chroma"))
    return store


@pytest.fixture
def search_store(vectorstore):
    """Point the search tool at the temp store for the duration of a test."""
    tools.set_store(vectorstore)
    yield vectorstore
    tools.set_store(None)
