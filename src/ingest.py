"""
Ingest data/guidelines/*.md into a persistent Chroma collection.

Each guideline file is a small YAML-frontmatter + Markdown document (see
data/guidelines/*.md). Chunking is heading-based: one chunk per `# Section`,
which keeps each chunk topically coherent and gives the citation step a
clean, human-readable section name to point to — "WHO Guideline for the
Pharmacological Treatment of Hypertension in Adults, section 'Treatment
target'" is a far more useful citation than "chunk 14".

Run: python -m src.ingest
"""
from __future__ import annotations
import glob
import os
import re
from dataclasses import dataclass, field
from typing import List

from . import config
from .embedder import Embedder

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_HEADING_RE = re.compile(r"^#\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    doc_id: str
    title: str
    organization: str
    year: str
    source_url: str
    section: str
    chunk_id: str = field(init=False)

    def __post_init__(self):
        self.chunk_id = f"{self.doc_id}::{self.section}".replace(" ", "_").lower()


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    fm_text, body = match.groups()
    meta = {}
    for line in fm_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body


def load_guideline_file(path: str) -> List[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    meta, body = _parse_frontmatter(raw)

    headings = list(_HEADING_RE.finditer(body))
    chunks: List[Chunk] = []
    for i, heading_match in enumerate(headings):
        section = heading_match.group(1).strip()
        start = heading_match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        section_text = body[start:end].strip()
        if not section_text:
            continue

        # A section written as several distinct lines (e.g. one line per BP
        # grade) is really several atomic facts, not one. Chunking the whole
        # section as a single blob would dilute retrieval: a query about one
        # grade would only ever match a chunk that also contains four other
        # grades' worth of unrelated vocabulary. Split line-structured
        # sections into one chunk per line; leave genuine single-paragraph
        # sections as one chunk.
        lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
        if len(lines) > 1:
            for line in lines:
                sub_label = line.split(":", 1)[0].strip() if ":" in line else line[:40]
                chunks.append(
                    Chunk(
                        text=line,
                        doc_id=meta.get("doc_id", os.path.basename(path)),
                        title=meta.get("title", os.path.basename(path)),
                        organization=meta.get("organization", "unknown"),
                        year=meta.get("year", "unknown"),
                        source_url=meta.get("source_url", ""),
                        section=f"{section} — {sub_label}",
                    )
                )
        else:
            chunks.append(
                Chunk(
                    text=section_text,
                    doc_id=meta.get("doc_id", os.path.basename(path)),
                    title=meta.get("title", os.path.basename(path)),
                    organization=meta.get("organization", "unknown"),
                    year=meta.get("year", "unknown"),
                    source_url=meta.get("source_url", ""),
                    section=section,
                )
            )
    return chunks


def load_all_guidelines(guidelines_dir: str = config.GUIDELINES_DIR) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(glob.glob(os.path.join(guidelines_dir, "*.md"))):
        chunks.extend(load_guideline_file(path))
    return chunks


def build_index(mock_embeddings: bool = False, persist_dir: str = config.CHROMA_PERSIST_DIR):
    import chromadb  # local import: optional dependency

    chunks = load_all_guidelines()
    if not chunks:
        raise RuntimeError(f"No guideline chunks found under {config.GUIDELINES_DIR}")

    embedder = Embedder(mock=mock_embeddings)
    vectors = embedder.embed([c.text for c in chunks])

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(config.COLLECTION_NAME)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "organization": c.organization,
                "year": c.year,
                "source_url": c.source_url,
                "section": c.section,
            }
            for c in chunks
        ],
    )
    print(f"Ingested {len(chunks)} chunks from {config.GUIDELINES_DIR} into '{config.COLLECTION_NAME}' at {persist_dir}")
    return collection


if __name__ == "__main__":
    mock = os.getenv("CLINICAL_AGENT_MOCK", "").lower() in {"1", "true", "yes"}
    build_index(mock_embeddings=mock)
