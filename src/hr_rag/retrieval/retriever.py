from dataclasses import replace
from typing import Protocol

from hr_rag.types import RetrievedChunk


class CandidateSource(Protocol):
    def hybrid_search(self, query: str, limit: int) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]: ...


class Retriever:
    """Hybrid recall (dense + BM25) followed by cross-encoder reranking and a relevance floor."""

    def __init__(
        self,
        store: CandidateSource,
        reranker: Reranker,
        prefetch_k: int,
        top_n: int,
        min_score: float,
    ):
        self.store = store
        self.reranker = reranker
        self.prefetch_k = prefetch_k
        self.top_n = top_n
        self.min_score = min_score

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        candidates = self.store.hybrid_search(query, limit=self.prefetch_k)
        if not candidates:
            return []
        ranked = self.reranker.rerank(query, [c.text for c in candidates], self.top_n)
        return [
            replace(candidates[index], score=score)
            for index, score in ranked
            if score >= self.min_score
        ]
