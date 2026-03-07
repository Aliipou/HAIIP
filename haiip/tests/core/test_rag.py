"""Tests for core/rag.py — RAGEngine."""

import numpy as np
import pytest

from haiip.core.rag import Document, RAGEngine


@pytest.fixture
def engine() -> RAGEngine:
    e = RAGEngine(top_k=3)
    e.initialize()
    return e


@pytest.fixture
def engine_with_docs(engine: RAGEngine) -> RAGEngine:
    docs = [
        Document(
            content="HDF failure occurs when cooling is insufficient.",
            title="HDF Guide",
            source="manual",
        ),
        Document(
            content="Tool wear failure TWF happens at high tool wear values.",
            title="TWF Guide",
            source="manual",
        ),
        Document(
            content="Power failure PWF is caused by torque * speed product exceeding limit.",
            title="PWF Guide",
            source="manual",
        ),
        Document(
            content="Overstrain failure OSF is caused by excessive torque.",
            title="OSF Guide",
            source="manual",
        ),
        Document(
            content="Normal operation: air temp ~300K, process temp ~310K.",
            title="Normal Ops",
            source="sop",
        ),
    ]
    engine.add_documents(docs)
    return engine


# ── Document ──────────────────────────────────────────────────────────────────


def test_document_id_is_stable():
    d = Document(content="hello world", title="Test")
    assert d.doc_id == d.doc_id


def test_document_id_differs_for_different_content():
    d1 = Document(content="hello world", title="Test")
    d2 = Document(content="different content", title="Test")
    assert d1.doc_id != d2.doc_id


# ── RAGEngine initialization ──────────────────────────────────────────────────


def test_engine_initializes():
    e = RAGEngine()
    e.initialize()
    assert e.is_initialized


def test_engine_empty_document_count():
    e = RAGEngine()
    e.initialize()
    assert e.document_count == 0


# ── Add documents ─────────────────────────────────────────────────────────────


def test_add_documents(engine):
    docs = [Document(content="Test doc", title="Doc 1")]
    engine.add_documents(docs)
    assert engine.document_count == 1


def test_add_documents_deduplication(engine):
    doc = Document(content="Unique content here", title="Unique Doc")
    engine.add_documents([doc])
    engine.add_documents([doc])  # duplicate
    assert engine.document_count == 1


def test_add_text_chunks(engine):
    long_text = "word " * 2000
    n = engine.add_text(long_text, title="Long Doc", chunk_size=100)
    assert n > 1
    assert engine.document_count == n


# ── Query ─────────────────────────────────────────────────────────────────────


def test_query_empty_kb(engine):
    result = engine.query("What causes HDF failure?")
    assert "No documents" in result.answer
    assert result.confidence == 0.0
    assert result.sources == []


def test_query_returns_answer(engine_with_docs):
    result = engine_with_docs.query("What causes HDF failure?")
    assert len(result.answer) > 0
    assert result.confidence >= 0.0
    assert isinstance(result.sources, list)


def test_query_sources_have_required_fields(engine_with_docs):
    result = engine_with_docs.query("tool wear failure")
    for src in result.sources:
        assert "title" in src
        assert "score" in src
        assert "excerpt" in src


def test_query_with_machine_filter(engine):
    engine.add_document(
        Document(
            content="CNC-001 specific maintenance procedure.",
            title="CNC-001 Manual",
            machine_id="CNC-001",
        )
    )
    engine.add_document(
        Document(
            content="CNC-002 specific maintenance procedure.",
            title="CNC-002 Manual",
            machine_id="CNC-002",
        )
    )
    result = engine.query("maintenance procedure", machine_id="CNC-001")
    machine_sources = [s for s in result.sources if "CNC-001" in s.get("title", "")]
    assert len(machine_sources) >= 1


def test_query_retrieval_time_recorded(engine_with_docs):
    result = engine_with_docs.query("torque failure")
    assert result.retrieval_time_ms >= 0.0


def test_query_llm_not_used_without_key(engine_with_docs):
    assert engine_with_docs._llm is None
    result = engine_with_docs.query("What is OSF?")
    assert result.llm_used is False


# ── Persistence ───────────────────────────────────────────────────────────────


def test_save_and_load_documents(tmp_path):
    e1 = RAGEngine(persist_dir=tmp_path)
    e1.initialize()
    e1.add_documents(
        [
            Document(content="Test content for persistence", title="Test Doc"),
        ]
    )
    e1._save_index()

    e2 = RAGEngine(persist_dir=tmp_path)
    e2.initialize()
    assert e2.document_count == 1


# ── TF-IDF fallback ───────────────────────────────────────────────────────────


def test_tfidf_embed_shape():
    texts = ["hello world", "foo bar baz"]
    emb = RAGEngine._tfidf_embed(texts, dim=384)
    assert emb.shape == (2, 384)


def test_tfidf_embed_normalized():
    texts = ["hello world test"]
    emb = RAGEngine._tfidf_embed(texts, dim=384)
    norm = np.linalg.norm(emb[0])
    assert abs(norm - 1.0) < 1e-5


# ── Chunking ──────────────────────────────────────────────────────────────────


def test_chunk_text():
    text = "word " * 1000
    chunks = RAGEngine._chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 100
