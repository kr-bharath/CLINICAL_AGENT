"""
Embeddings, also pluggable. Real mode uses a local sentence-transformers
model (free, no API, no rate limit — this is the piece that should never
depend on a quota). Mock mode uses a cheap deterministic bag-of-words hash
vector so retrieval logic can be unit-tested without downloading any model
weights (useful in network-restricted CI/sandbox environments).
"""
from __future__ import annotations
import hashlib
import os
import re
from typing import List

import numpy as np

from . import config

_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    def __init__(self, mock: bool = False, dim: int = 256):
        self.mock = mock or os.getenv("CLINICAL_AGENT_MOCK", "").lower() in {"1", "true", "yes"}
        self.dim = dim
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import: optional dependency
            self._model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        return self._model

    def _mock_embed_one(self, text: str) -> np.ndarray:
        """
        Deterministic hashed bag-of-words vector. Not semantically meaningful
        the way a real embedding model is, but two texts that share more
        words land closer together — enough to exercise sufficiency /
        groundedness thresholds in tests without a network call.
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD_RE.findall(text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.mock:
            return [self._mock_embed_one(t).tolist() for t in texts]
        model = self._get_model()
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)
