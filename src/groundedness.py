"""
Groundedness scoring: this is the eval-gate logic, promoted from "a script
I run afterwards" to a function the agent graph calls before it's allowed
to return an answer.

Method (cheap, free, no extra LLM call): split the draft answer into
sentences, embed each one, and compare it against the embeddings of the
retrieved evidence chunks. A sentence that isn't semantically close to
*any* retrieved chunk is treated as ungrounded. The overall score is the
average of each sentence's best match.

This is a deliberately simple, fast, deterministic heuristic — good enough
to catch an answer that drifted away from its cited evidence, cheap enough
to run on every single generation without burning API quota. A stricter
(and more expensive) alternative — a second LLM call acting as a judge — is
left as a documented extension point rather than the default, precisely
because it costs an extra request against the free-tier rate limit for
every single answer.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

from .embedder import Embedder
from .retrieval import RetrievalResult

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class GroundednessReport:
    score: float
    sentence_scores: List[float]
    ungrounded_sentences: List[str]


def _split_sentences(text: str) -> List[str]:
    # Drop the mock/debug tag if present so it doesn't skew scoring.
    cleaned = re.sub(r"\[MOCK RESPONSE.*?\]", "", text)
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    return parts


def score_groundedness(
    answer_text: str,
    evidence: List[RetrievalResult],
    embedder: Embedder,
) -> GroundednessReport:
    sentences = _split_sentences(answer_text)
    if not sentences:
        return GroundednessReport(score=0.0, sentence_scores=[], ungrounded_sentences=[])
    if not evidence:
        return GroundednessReport(score=0.0, sentence_scores=[0.0] * len(sentences), ungrounded_sentences=sentences)

    evidence_vecs = embedder.embed([e.text for e in evidence])
    sentence_vecs = embedder.embed(sentences)

    sentence_scores: List[float] = []
    ungrounded: List[str] = []
    for sentence, s_vec in zip(sentences, sentence_vecs):
        best = max(Embedder.cosine_similarity(s_vec, e_vec) for e_vec in evidence_vecs)
        sentence_scores.append(best)
        if best < 0.30:  # a much looser per-sentence floor than the overall gate
            ungrounded.append(sentence)

    overall = sum(sentence_scores) / len(sentence_scores)
    return GroundednessReport(score=overall, sentence_scores=sentence_scores, ungrounded_sentences=ungrounded)
