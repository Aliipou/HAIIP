"""RAG (Retrieval-Augmented Generation) engine for industrial knowledge queries.

Architecture:
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, no API cost)
- Vector store: FAISS (in-memory, persisted to disk)
- LLM: OpenAI GPT-4o-mini (configurable; falls back to template answers)
- Documents: maintenance manuals, fault logs, ISO standards, SME procedures

Query flow:
    1. Embed user question
    2. Retrieve top-k similar documents from FAISS
    3. Build context prompt with retrieved chunks
    4. Call LLM with context + question
    5. Return answer + source citations + confidence

Usage:
    engine = RAGEngine()
    engine.add_documents(docs)
    result = engine.query("What causes HDF failure in CNC machines?")
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Dimension of all-MiniLM-L6-v2 embeddings
EMBEDDING_DIM = 384

# Max tokens sent to LLM as context
MAX_CONTEXT_CHARS = 3000


@dataclass
class Document:
    """A single document chunk in the knowledge base."""
    content: str
    title: str
    source: str = "manual"
    machine_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return hashlib.sha256(f"{self.title}:{self.content[:100]}".encode()).hexdigest()[:16]


@dataclass
class QueryResult:
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    retrieval_time_ms: float
    llm_used: bool
    machine_context: dict[str, Any] | None = None


class RAGEngine:
    """FAISS-backed RAG engine with optional LLM integration.

    Degrades gracefully:
    - No sentence-transformers → TF-IDF fallback embeddings
    - No OpenAI key → template-based answers from retrieved context
    - No FAISS → linear search fallback
    """

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        openai_api_key: str = "",
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.model_name = model_name
        self.top_k = top_k
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model

        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None
        self._index: Any = None  # FAISS index
        self._embed_model: Any = None
        self._llm: Any = None
        self._initialized = False

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self) -> "RAGEngine":
        """Load embedding model and optionally LLM. Call once at startup."""
        self._embed_model = self._load_embedding_model()

        if self.openai_api_key:
            self._llm = self._load_llm()

        if self.persist_dir and (self.persist_dir / "faiss.index").exists():
            self._load_index()

        self._initialized = True
        logger.info(
            "RAGEngine initialized: model=%s, llm=%s, docs=%d",
            self.model_name,
            "openai" if self._llm else "none",
            len(self._documents),
        )
        return self

    def _load_embedding_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded: %s", self.model_name)
            return model
        except ImportError:
            logger.warning("sentence-transformers not installed — using TF-IDF fallback")
            return None

    def _load_llm(self) -> Any:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI LLM loaded: %s", self.openai_model)
            return client
        except ImportError:
            logger.warning("openai not installed — LLM disabled")
            return None

    # ── Document management ───────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the knowledge base and rebuild the index."""
        new_ids = {doc.doc_id for doc in documents}
        existing_ids = {doc.doc_id for doc in self._documents}
        to_add = [d for d in documents if d.doc_id not in existing_ids]

        if not to_add:
            logger.info("No new documents to add (all duplicates)")
            return

        self._documents.extend(to_add)
        self._rebuild_index()

        if self.persist_dir:
            self._save_index()

        logger.info("Added %d documents (total=%d)", len(to_add), len(self._documents))

    def add_document(self, doc: Document) -> None:
        self.add_documents([doc])

    def add_text(
        self,
        content: str,
        title: str,
        source: str = "manual",
        machine_id: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> int:
        """Chunk a long text and add all chunks. Returns number of chunks added."""
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        docs = [
            Document(
                content=chunk,
                title=f"{title} (chunk {i+1}/{len(chunks)})",
                source=source,
                machine_id=machine_id,
            )
            for i, chunk in enumerate(chunks)
        ]
        self.add_documents(docs)
        return len(docs)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        machine_id: str | None = None,
        top_k: int | None = None,
    ) -> QueryResult:
        """Answer a natural-language question using retrieved context."""
        if not self._documents:
            return QueryResult(
                answer="No documents in the knowledge base. Please ingest documents first.",
                sources=[],
                confidence=0.0,
                retrieval_time_ms=0.0,
                llm_used=False,
            )

        k = top_k or self.top_k
        t0 = time.monotonic()

        # Retrieve relevant documents
        retrieved = self._retrieve(question, k=k, machine_id=machine_id)
        retrieval_ms = (time.monotonic() - t0) * 1000

        # Build context
        context = self._build_context(retrieved)

        # Generate answer
        if self._llm:
            answer, llm_used = self._llm_answer(question, context), True
        else:
            answer, llm_used = self._template_answer(question, retrieved), False

        # Confidence: based on retrieval similarity scores
        avg_score = float(np.mean([r["score"] for r in retrieved])) if retrieved else 0.0
        confidence = round(min(avg_score, 1.0), 4)

        sources = [
            {
                "title": r["doc"].title,
                "source": r["doc"].source,
                "score": round(r["score"], 4),
                "excerpt": r["doc"].content[:200] + "...",
            }
            for r in retrieved
        ]

        return QueryResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieval_time_ms=round(retrieval_ms, 2),
            llm_used=llm_used,
            machine_context={"machine_id": machine_id} if machine_id else None,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(
        self,
        query: str,
        k: int = 5,
        machine_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k most relevant documents."""
        if not self._documents:
            return []

        query_emb = self._embed([query])[0]

        if self._index is not None:
            scores, indices = self._faiss_search(query_emb, k * 2)
        else:
            scores, indices = self._linear_search(query_emb, k * 2)

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            if machine_id and doc.machine_id and doc.machine_id != machine_id:
                continue
            results.append({"doc": doc, "score": float(score)})

        return results[:k]

    def _faiss_search(self, query_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = query_emb.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, k)
        return scores[0], indices[0]

    def _linear_search(self, query_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._embeddings is None:
            return np.array([]), np.array([])
        sims = np.dot(self._embeddings, query_emb) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        top_k = min(k, len(sims))
        indices = np.argsort(sims)[::-1][:top_k]
        return sims[indices], indices

    # ── Answer generation ─────────────────────────────────────────────────────

    def _llm_answer(self, question: str, context: str) -> str:
        system_prompt = (
            "You are an industrial AI assistant for SME manufacturers. "
            "Answer questions about machine maintenance, anomaly detection, and predictive maintenance. "
            "Base your answer strictly on the provided context. "
            "If the context doesn't contain enough information, say so clearly."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        try:
            response = self._llm.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content or "No answer generated."
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return self._template_answer(question, [])

    @staticmethod
    def _template_answer(question: str, retrieved: list[dict[str, Any]]) -> str:
        if not retrieved:
            return f"No relevant documents found for: '{question}'"
        top = retrieved[0]["doc"]
        return (
            f"Based on the knowledge base:\n\n"
            f"**{top.title}**\n{top.content[:500]}\n\n"
            f"({len(retrieved)} related documents found. Configure OpenAI API key for AI-generated answers.)"
        )

    @staticmethod
    def _build_context(retrieved: list[dict[str, Any]]) -> str:
        parts = []
        total = 0
        for r in retrieved:
            chunk = f"[{r['doc'].title}]\n{r['doc'].content}"
            if total + len(chunk) > MAX_CONTEXT_CHARS:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n---\n\n".join(parts)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_model is not None:
            return np.array(self._embed_model.encode(texts, normalize_embeddings=True))
        # TF-IDF fallback: simple bag-of-words hash embedding
        return self._tfidf_embed(texts)

    @staticmethod
    def _tfidf_embed(texts: list[str], dim: int = EMBEDDING_DIM) -> np.ndarray:
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim  # noqa: S324
                result[i, idx] += 1.0
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result

    def _rebuild_index(self) -> None:
        texts = [d.content for d in self._documents]
        self._embeddings = self._embed(texts).astype(np.float32)

        try:
            import faiss
            dim = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized
            self._index.add(self._embeddings)
            logger.info("FAISS index built: %d vectors, dim=%d", len(self._documents), dim)
        except ImportError:
            logger.warning("faiss-cpu not installed — using linear search fallback")
            self._index = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_index(self) -> None:
        import pickle
        path = self.persist_dir
        path.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(path / "faiss.index"))

        if self._embeddings is not None:
            np.save(str(path / "embeddings.npy"), self._embeddings)

        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

        logger.info("RAGEngine persisted to %s", path)

    def _load_index(self) -> None:
        import pickle
        path = self.persist_dir

        try:
            import faiss
            self._index = faiss.read_index(str(path / "faiss.index"))
        except (ImportError, Exception):
            self._index = None

        emb_path = path / "embeddings.npy"
        if emb_path.exists():
            self._embeddings = np.load(str(emb_path))

        doc_path = path / "documents.pkl"
        if doc_path.exists():
            with open(doc_path, "rb") as f:
                self._documents = pickle.load(f)  # noqa: S301 — internal use only

        logger.info("RAGEngine loaded from %s (%d docs)", path, len(self._documents))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def is_initialized(self) -> bool:
        return self._initialized
