"""Retrieval for the Arm filing: dense, sparse, hybrid, and reranked.

Dense search alone misses exact table lookups such as "4,007", because the
embedding blurs the digits. BM25 alone misses paraphrased questions. Hybrid
fuses both rankings, then a cross-encoder reranks the shortlist.
"""

import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60

# The cross-encoder truncates at 512 tokens, but chunks reach ~2,800 tokens of
# dense financial text, so scoring a whole chunk judges only its first fifth.
# Chunks are scored as overlapping windows instead, each comfortably inside the
# limit, and a chunk takes the score of its best window.
RERANK_WINDOW_WORDS = 250
RERANK_WINDOW_OVERLAP = 50

CHUNK_PATHS = {
    "fixed": "data/fixed_chunks.json",
    "semantic": "data/semantic_chunks.json",
}

# Keeps "4,007" and "87.1%" as single tokens so table lookups stay searchable.
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9.,%$-]*")


def tokenize(text):
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [token.strip(".,-") for token in tokens if token.strip(".,-")]


def _windows(text, size=RERANK_WINDOW_WORDS, overlap=RERANK_WINDOW_OVERLAP):
    """Split a chunk into overlapping word windows that fit the reranker."""
    words = text.split()

    if len(words) <= size:
        return [text]

    step = size - overlap

    return [
        " ".join(words[start:start + size])
        for start in range(0, len(words), step)
        if words[start:start + size]
    ]


def _reciprocal_rank_fusion(rankings):
    """Merge several ranked ID lists into one score per chunk."""
    fused = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    return fused


class Retriever:
    """Loads one chunking strategy's index and serves the retrieval modes."""

    _reranker = None

    # Query embeddings are independent of chunking strategy, so cache them
    # once per process. The evaluation sweep reuses the same queries across
    # every configuration.
    _query_cache = {}

    def __init__(self, strategy="semantic", load_reranker=True):
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY was not found.")

        self.strategy = strategy
        self.chunks = json.loads(
            Path(CHUNK_PATHS[strategy]).read_text(encoding="utf-8")
        )

        index_path = Path("data/index") / f"{strategy}.faiss"

        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} is missing. Run build_vector_store.py first."
            )

        self.index = faiss.read_index(str(index_path))

        # A rebuilt chunk file with a stale index would silently return the
        # wrong passages, so refuse to run rather than report bad results.
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"{index_path} holds {self.index.ntotal} vectors but "
                f"{CHUNK_PATHS[strategy]} holds {len(self.chunks)} chunks. "
                "Re-run build_vector_store.py."
            )
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key, model=EMBEDDING_MODEL
        )

        self.bm25 = BM25Okapi(
            [tokenize(chunk["text"]) for chunk in self.chunks]
        )

        if load_reranker and Retriever._reranker is None:
            Retriever._reranker = CrossEncoder(RERANK_MODEL)

    def _query_vector(self, query):
        if query not in Retriever._query_cache:
            Retriever._query_cache[query] = self.embeddings.embed_query(query)

        return Retriever._query_cache[query]

    def dense(self, query, k):
        vector = np.array([self._query_vector(query)], dtype="float32")
        faiss.normalize_L2(vector)

        _, ids = self.index.search(vector, min(k, len(self.chunks)))

        return [int(chunk_id) for chunk_id in ids[0] if chunk_id != -1]

    def sparse(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:k]

        return [int(chunk_id) for chunk_id in ranked if scores[chunk_id] > 0]

    def hybrid(self, query, k):
        dense_ids = self.dense(query, k)
        sparse_ids = self.sparse(query, k)

        fused = _reciprocal_rank_fusion([dense_ids, sparse_ids])

        return sorted(fused, key=fused.get, reverse=True)[:k]

    def rerank(self, query, chunk_ids, top_k, windowed=True):
        """Score candidates with the cross-encoder and keep the best top_k.

        With windowed=False the whole chunk is passed to the model and silently
        truncated at 512 tokens; that is the naive variant the evaluation
        reports as a comparison.
        """
        if not chunk_ids:
            return []

        pairs = []
        owners = []

        for chunk_id in chunk_ids:
            text = self.chunks[chunk_id]["text"]
            segments = _windows(text) if windowed else [text]

            for segment in segments:
                pairs.append((query, segment))
                owners.append(chunk_id)

        scores = Retriever._reranker.predict(pairs)

        # A chunk is worth its best-matching window.
        best = {}

        for chunk_id, score in zip(owners, scores):
            if chunk_id not in best or score > best[chunk_id]:
                best[chunk_id] = float(score)

        ordered = sorted(best.items(), key=lambda pair: pair[1], reverse=True)

        return [(int(chunk_id), score) for chunk_id, score in ordered[:top_k]]

    def search(self, query, mode="hybrid", candidates=20, top_k=5,
               use_reranker=True, windowed_rerank=True):
        """Return [(chunk_id, score)] for the chosen retrieval configuration."""
        retrieve = {
            "dense": self.dense,
            "sparse": self.sparse,
            "hybrid": self.hybrid,
        }[mode]

        shortlist = retrieve(query, candidates)

        if use_reranker:
            return self.rerank(
                query, shortlist, top_k, windowed=windowed_rerank
            )

        # Without a reranker the retrieval order stands; score is the rank.
        return [
            (chunk_id, 1.0 / (rank + 1))
            for rank, chunk_id in enumerate(shortlist[:top_k])
        ]

    def cite(self, chunk_id):
        """Human-readable page citation for a chunk."""
        pages = self.chunks[chunk_id].get("pages") or []

        if not pages:
            return "page unknown"

        if len(pages) == 1:
            return f"page {pages[0]}"

        return f"pages {pages[0]}-{pages[-1]}"
