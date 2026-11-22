"""Okapi BM25 lexical retrieval (small, dependency-free implementation).

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))      (non-negative variant)

Scores are unbounded, non-negative, and only comparable between documents for the SAME
query. They are not probabilities and not comparable with cosine similarity.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np

# Small, fixed English stop-word list. Domain words such as "recalled" are kept: their
# low IDF already down-weights them.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to "
    "was were will with".split()
)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return [t for t in _TOKEN.findall(text) if len(t) > 1 and t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over pre-tokenised documents.

    k1 controls term-frequency saturation (the 5th "salmonella" adds little), b controls
    length normalisation (long notices are not favoured just for containing more words).
    IDF uses the +1 variant so very common terms never get a negative weight."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25 needs at least one document")
        self.k1 = k1
        self.b = b
        tokenized = [tokenize(doc) for doc in documents]
        self._term_freqs = [Counter(tokens) for tokens in tokenized]
        self._doc_lengths = np.array([len(tokens) for tokens in tokenized], dtype=np.float64)
        self._avgdl = float(self._doc_lengths.mean()) or 1.0

        doc_freq: Counter[str] = Counter()
        for freqs in self._term_freqs:
            doc_freq.update(freqs.keys())
        n = len(documents)
        self._idf = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._term_freqs)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query` (0 where no query term occurs)."""
        result = np.zeros(len(self._term_freqs), dtype=np.float64)
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        if not terms:
            return result
        norm = self.k1 * (1.0 - self.b + self.b * self._doc_lengths / self._avgdl)
        for i, freqs in enumerate(self._term_freqs):
            total = 0.0
            for term in terms:
                tf = freqs.get(term)
                if tf:
                    total += self._idf[term] * tf * (self.k1 + 1.0) / (tf + norm[i])
            result[i] = total
        return result
