"""
Retrieval layer. Two implementations share one interface:

- ChromaRetriever: queries the persistent Chroma collection built by
  src/ingest.py. This is what the real agent and the live eval run use.
- InMemoryRetriever: brute-force cosine similarity over an in-memory list of
  chunks. No Chroma, no disk index, no ingestion step required — used by
  unit tests so `pytest` never depends on a prior `python -m src.ingest`
  run or on chromadb's on-disk state.

Both return the same RetrievalResult shape, so src/agent_graph.py never
needs to know which backend it's talking to.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence

from . import config
from .embedder import Embedder


@dataclass
class RetrievalResult:
    text: str
    metadata: dict
    score: float

    def citation(self) -> str:
        m = self.metadata
        return f"{m.get('title', 'unknown source')} — \u201c{m.get('section', '')}\u201d ({m.get('organization', '')}, {m.get('year', '')})"


class InMemoryRetriever:
    def __init__(self, chunks: Sequence[dict], embedder: Embedder):
        # chunks: [{"text": ..., "metadata": {...}}, ...]
        self.embedder = embedder
        self.chunks = list(chunks)
        self._vectors = embedder.embed([c["text"] for c in self.chunks]) if self.chunks else []

    def retrieve(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> List[RetrievalResult]:
        if not self.chunks:
            return []
        q_vec = self.embedder.embed([query])[0]
        scored = [
            RetrievalResult(text=c["text"], metadata=c["metadata"], score=Embedder.cosine_similarity(q_vec, v))
            for c, v in zip(self.chunks, self._vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


class ChromaRetriever:
    def __init__(self, persist_dir: str = config.CHROMA_PERSIST_DIR, mock_embeddings: bool = False):
        import chromadb  # local import: optional dependency

        self.embedder = Embedder(mock=mock_embeddings)
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(config.COLLECTION_NAME)

    def retrieve(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> List[RetrievalResult]:
        q_vec = self.embedder.embed([query])[0]
        result = self.collection.query(query_embeddings=[q_vec], n_results=top_k)
        out = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma returns a distance; convert to a similarity-like score in [0, 1].
            score = max(0.0, 1.0 - dist)
            out.append(RetrievalResult(text=doc, metadata=meta, score=score))
        return out


def is_sufficient(results: List[RetrievalResult]) -> bool:
    if not results:
        return False
    return results[0].score >= config.SUFFICIENCY_SCORE_THRESHOLD


def build_mock_retriever_from_guidelines() -> InMemoryRetriever:
    """
    Convenience constructor: loads data/guidelines/*.md straight into an
    InMemoryRetriever with a mock embedder. Used by tests, the mock eval
    run, and `app.py --mock` so all three exercise the exact same guideline
    text without requiring a prior `python -m src.ingest` run.
    """
    from .ingest import load_all_guidelines  # local import: avoids a module-load cycle

    chunks = load_all_guidelines()
    embedder = Embedder(mock=True)
    return InMemoryRetriever(
        chunks=[
            {
                "text": c.text,
                "metadata": {
                    "doc_id": c.doc_id, "title": c.title, "organization": c.organization,
                    "year": c.year, "source_url": c.source_url, "section": c.section,
                },
            }
            for c in chunks
        ],
        embedder=embedder,
    )
